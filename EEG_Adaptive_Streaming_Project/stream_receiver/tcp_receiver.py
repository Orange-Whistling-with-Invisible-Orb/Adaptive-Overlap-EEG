import socket
import numpy as np
import time
import threading
import queue
import struct
from pathlib import Path
from scipy.signal import butter, filtfilt

HOST = "127.0.0.1"
PORT = 50007
N_CHANNELS = 19
PACKET_SAMPLES = 60


class BasicPreprocessor:
    """基本脑电预处理：带通 + 按通道标准化"""

    def __init__(self, sample_rate=200):
        self.sample_rate = sample_rate
        self.filter_coefficients = self._design_filters()

    def _design_filters(self):
        b, a = butter(4, [0.5, 40], btype="bandpass", fs=self.sample_rate)
        return b, a

    def preprocess(self, data):
        # 注意：这里按包（60个点）进行filtfilt滤波可能会有边缘效应，
        # 实际严谨的做法是将滤波也放到长窗口中。这里暂时保留原逻辑。
        processed_data = np.copy(data)
        b, a = self.filter_coefficients

        for i in range(processed_data.shape[0]):
            processed_data[i, :] = filtfilt(b, a, processed_data[i, :])

        mean = np.mean(processed_data, axis=1, keepdims=True)
        std = np.std(processed_data, axis=1, keepdims=True) + 1e-8
        processed_data = (processed_data - mean) / std
        return processed_data


class EEGReceiver(threading.Thread):
    """网络流接收端（保持原样，负责收包并存入队列）"""

    def __init__(self, host=HOST, port=PORT):
        super().__init__()
        self.daemon = True  # 设置为守护线程，程序退出时自动终止
        self.host = host
        self.port = port
        self._running = True
        self.socket = None
        self.conn = None
        self.ready_event = threading.Event()
        self.start_error = None
        self.data_queue = queue.Queue()
        self.delay_log = []
        self.received_packet_count = 0
        self.listening_event = threading.Event()
        self.connected_event = threading.Event()
        # 日志路径（你可以根据实际环境修改）
        self.log_file = Path("delay_log.txt")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _recv_exact(conn, size, running_flag=None):
        chunks = bytearray()
        while len(chunks) < size:
            if running_flag is not None and not running_flag():
                return None
            try:
                part = conn.recv(size - len(chunks))
            except socket.timeout:
                # 超时并不视为致命错误，继续等待后续数据
                continue
            except OSError:
                # 连接被本地关闭或其他网络错误，按断连处理
                return None
            if not part:
                return None
            chunks.extend(part)
        return bytes(chunks)

    def wait_until_listening(self, timeout=None):
        return self.listening_event.wait(timeout=timeout)

    def wait_until_connected(self, timeout=None):
        return self.connected_event.wait(timeout=timeout)

    def stop(self):
        self._running = False
        self.connected_event.clear()
        self.listening_event.clear()
        if self.conn:
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.conn.close()
            except Exception:
                pass
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.socket.close()
            except Exception:
                pass

    def run(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Windows 上优先独占端口，避免同端口多进程“复用”导致连接落到旧进程
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(1.0)
            print(f"[Receiver] Listening on {self.host}:{self.port}")
            self.listening_event.set()
            self.ready_event.set()

            while self._running:
                try:
                    conn, addr = self.socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    # stop() 关闭 socket 后会触发该异常，属于正常退出路径
                    break

                self.conn = conn
                print(f"[Receiver] Connected by {addr}")
                self.connected_event.set()
                conn.settimeout(1.0)

                try:
                    while self._running:
                        timestamp_bytes = self._recv_exact(
                            conn, 8, running_flag=lambda: self._running
                        )
                        if not timestamp_bytes:
                            break

                        data_bytes = self._recv_exact(
                            conn,
                            N_CHANNELS * PACKET_SAMPLES * 4,
                            running_flag=lambda: self._running,
                        )
                        if not data_bytes:
                            break

                        timestamp = struct.unpack("<Q", timestamp_bytes)[0]
                        eeg_data = np.frombuffer(data_bytes, dtype=np.float32).reshape(
                            N_CHANNELS, PACKET_SAMPLES, order="F"
                        )

                        current_time = int(time.time() * 1000)
                        delay = current_time - timestamp
                        self.delay_log.append(delay)
                        self.received_packet_count += 1
                        recv_perf_ms = time.perf_counter() * 1000.0

                        # 为不影响性能，可以考虑批量写入日志，这里保持原样
                        with open(self.log_file, "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {delay} ms\n")

                        self.data_queue.put(
                            {
                                "timestamp": timestamp,
                                "eeg_data": eeg_data,
                                "delay": delay,
                                "recv_perf_ms": recv_perf_ms,
                                "packet_id": self.received_packet_count,
                            }
                        )
                finally:
                    self.connected_event.clear()
                    try:
                        conn.close()
                    except Exception:
                        pass
                    self.conn = None

        except Exception as e:
            self.start_error = e
            self.ready_event.set()
            print(f"[Receiver Error] {e}")
        finally:
            if not self.ready_event.is_set():
                self.ready_event.set()
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
            if self.socket:
                try:
                    self.socket.close()
                except Exception:
                    pass
            print("[Receiver] Thread finished")


class StreamDispatcher(threading.Thread):
    """
    流式分发器（替代原来的DataPreprocessor）
    任务：从队列拿数据 -> 基础预处理 -> 喂给自适应窗口管理器
    """

    def __init__(self, receiver, window_manager, sample_rate=200, on_packet_received=None):
        super().__init__()
        self.daemon = True  # 设置为守护线程，程序退出时自动终止
        self.receiver = receiver
        self.running = True
        self.preprocessor = BasicPreprocessor(sample_rate=sample_rate)
        # 核心：持有窗口管理器的引用
        self.window_manager = window_manager
        self.packet_count = 0
        self.on_packet_received = on_packet_received

    def run(self):
        while self.running and (
            self.receiver.is_alive() or not self.receiver.data_queue.empty()
        ):
            try:
                packet = self.receiver.data_queue.get(timeout=0.1)
                eeg_data = packet["eeg_data"]
                recv_perf_ms = packet.get("recv_perf_ms")
                packet_id = packet.get("packet_id")

                if self.on_packet_received is not None:
                    try:
                        self.on_packet_received(eeg_data, packet_id, recv_perf_ms)
                    except TypeError:
                        # 兼容旧版签名: callback(eeg_data)
                        self.on_packet_received(eeg_data)

                # 1. 基础预处理 (滤波+标准化)
                processed_packet = self.preprocessor.preprocess(eeg_data)
                self.packet_count += 1

                # 2. 将处理后的小包推送给窗口管理器进行自适应切分与重构
                self.window_manager.receive_packet(
                    processed_packet, packet_receive_time_ms=recv_perf_ms
                )

            except queue.Empty:
                continue

    def stop(self):
        self.running = False
