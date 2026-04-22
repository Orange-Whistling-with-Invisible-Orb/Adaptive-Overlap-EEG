import socket
import numpy as np
import time
import random
import argparse
from pathlib import Path

HOST = "127.0.0.1"
PORT = 50007


class EEGStreamSender:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.data_files = []
        self.current_file_idx = 0

    def load_data_files(self, data_dir):
        """加载所有污染数据文件"""
        data_dir = Path(data_dir)
        self.data_files = list(data_dir.glob("Contaminated_*.npy"))
        if not self.data_files:
            raise ValueError(f"No Contaminated_*.npy files found in {data_dir}")
        print(f"Loaded {len(self.data_files)} contaminated data files")

    def connect(self):
        """连接到接收端"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        retries = 10
        for i in range(retries):
            try:
                self.socket.connect((self.host, self.port))
                print(f"Connected to {self.host}:{self.port}")
                return True
            except Exception as e:
                print(f"Connection failed (attempt {i+1}/{retries}): {e}")
                time.sleep(1)
        return False

    def send_data(
        self,
        duration=60,
        sample_rate_hz=200.0,
        sample_rate_jitter_hz=20.0,
        samples_per_packet=60,
        emulate_fragmentation=False,  # 默认关闭碎片模拟，避免接收端阻塞
    ):
        """发送数据。

        说明：
        - sample_rate_hz 表示时间采样率（目标约 200Hz）。
        - 每个包包含 samples_per_packet 个时间点。
        - 包发送节拍由 samples_per_packet / sample_rate_hz 决定。
        """
        if not self.socket:
            print("Not connected. Call connect() first.")
            return

        start_time = time.perf_counter()
        end_time = start_time + duration
        packet_count = 0
        next_send_time = start_time

        while time.perf_counter() < end_time:
            # 随机选择一个数据文件
            file_path = random.choice(self.data_files)
            print(f"Sending data from: {file_path.name}")

            # 加载数据
            try:
                data = np.load(file_path)
                if data.ndim == 3:
                    # 假设形状为 (subject, channel, time)
                    subject_idx = random.randint(0, data.shape[0] - 1)
                    eeg_data = data[subject_idx]
                elif data.ndim == 2:
                    # 假设形状为 (channel, time)
                    eeg_data = data
                else:
                    print(f"Unexpected data shape: {data.shape}")
                    continue
            except Exception as e:
                print(f"Error loading data: {e}")
                continue

            # 发送数据
            try:
                # 计算数据包大小
                n_channels, n_samples = eeg_data.shape
                if n_channels != 19:
                    print(f"Skip file due to unexpected channel count: {n_channels}")
                    continue

                # 发送数据片段
                for i in range(0, n_samples, samples_per_packet):
                    end_idx = min(i + samples_per_packet, n_samples)
                    data_chunk = eeg_data[:, i:end_idx]

                    # 确保数据块大小为60个样本
                    if data_chunk.shape[1] != samples_per_packet:
                        # 如果不足60个样本，跳过
                        continue

                    # 添加时间戳
                    timestamp = int(time.time() * 1000)
                    timestamp_bytes = timestamp.to_bytes(8, byteorder="little")

                    # 转换数据为bytes
                    data_bytes = data_chunk.astype(np.float32).tobytes(
                        order="F"
                    )  # Fortran order

                    # 发送数据包（以随机小片段发送，模拟网络传输）
                    packet = timestamp_bytes + data_bytes
                    total_size = len(packet)
                    sent = 0

                    if emulate_fragmentation:
                        while sent < total_size:
                            # 随机决定本次发送的字节数（10-200字节）
                            chunk_size = random.randint(10, 200)
                            chunk = packet[sent : sent + chunk_size]
                            self.socket.sendall(chunk)
                            sent += len(chunk)
                            # 保持微小碎片抖动，避免破坏整体 200Hz 节拍
                            time.sleep(random.uniform(0, 0.001))
                    else:
                        self.socket.sendall(packet)

                    packet_count += 1
                    print(f"Sent packet {packet_count}, size: {total_size} bytes")

                    # 以采样率时间轴做节拍控制，在 200Hz 附近波动
                    low_hz = max(1.0, sample_rate_hz - sample_rate_jitter_hz)
                    high_hz = max(low_hz, sample_rate_hz + sample_rate_jitter_hz)
                    current_rate_hz = random.uniform(low_hz, high_hz)
                    current_packet_interval = samples_per_packet / current_rate_hz
                    next_send_time += current_packet_interval
                    now = time.perf_counter()
                    sleep_time = next_send_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    print(f"Current send sample rate: {current_rate_hz:.1f} Hz")

                    # 检查是否超时
                    if time.perf_counter() >= end_time:
                        break
            except Exception as e:
                print(f"Error sending data: {e}")
                break

    def close(self):
        """关闭连接"""
        if self.socket:
            try:
                self.socket.close()
                print("Connection closed")
            except Exception as e:
                print(f"Error closing connection: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG 流式发送端")
    parser.add_argument("--host", type=str, default=HOST, help="接收端 IP")
    parser.add_argument("--port", type=int, default=PORT, help="接收端端口")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="d:\\Desktop\\BCI_project\\code_基于自适应重叠率滑动窗口的实时脑电信号预处理方法研究\\EEG_Adaptive_Streaming_Project\\data\\contaminated",
        help="污染数据目录，需包含 Contaminated_*.npy",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="发送时长（秒）")
    args = parser.parse_args()

    sender = EEGStreamSender(host=args.host, port=args.port)
    sender.load_data_files(args.data_dir)

    if sender.connect():
        try:
            sender.send_data(
                duration=args.duration,
                sample_rate_hz=200.0,
                sample_rate_jitter_hz=20.0,
                samples_per_packet=60,
            )
        finally:
            sender.close()
