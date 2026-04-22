import time
from pathlib import Path
import sys

import numpy as np


class STFNetWindowDenoiser:
    """
    窗口级 STFNet 去噪器。
    输入/输出均为 (C, T) 的 numpy.float32。
    """

    def __init__(self, checkpoint_path, device="cpu", model_input_len=500):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_input_len = int(model_input_len)
        if self.model_input_len < 1:
            self.model_input_len = 500

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"STFNet checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
        except Exception as e:
            raise ImportError(f"Failed to import torch for STFNet inference: {e}") from e

        self.torch = torch
        self.device = self._resolve_device(device)
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def _resolve_device(self, device):
        device = str(device)
        if device.startswith("cuda") and not self.torch.cuda.is_available():
            print("[STFNet] CUDA unavailable, fallback to CPU.")
            return self.torch.device("cpu")
        return self.torch.device(device)

    def _load_model(self):
        # 训练脚本保存的是完整 model 对象；兼容不同 torch 版本。
        # 确保能反序列化 stfnet_model.STFNet
        project_root = Path(__file__).resolve().parents[1]
        stfnet_module_dir = project_root / "stfnet_module"
        if str(stfnet_module_dir) not in sys.path:
            sys.path.append(str(stfnet_module_dir))
        try:
            model = self.torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            model = self.torch.load(str(self.checkpoint_path), map_location=self.device)
        return model

    @staticmethod
    def _resample_time(sig, target_len):
        """
        线性插值重采样时间轴: (C, T) -> (C, target_len)
        """
        c, t = sig.shape
        if t == target_len:
            return sig
        if t <= 1 or target_len <= 1:
            return np.repeat(sig[:, :1], target_len, axis=1)
        src_x = np.linspace(0.0, 1.0, t, dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        out = np.empty((c, target_len), dtype=np.float32)
        for ch in range(c):
            out[ch] = np.interp(dst_x, src_x, sig[ch]).astype(np.float32)
        return out

    def denoise(self, window):
        """
        window: np.ndarray, shape=(C, T), float32
        """
        x = np.asarray(window, dtype=np.float32)
        orig_len = x.shape[1]
        if orig_len != self.model_input_len:
            x = self._resample_time(x, self.model_input_len)

        inp = self.torch.from_numpy(x).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            out = self.model(inp).squeeze(0).detach().cpu().numpy().astype(np.float32)

        if orig_len != self.model_input_len:
            out = self._resample_time(out, orig_len)
        return out


class AdaptiveWindowManager:
    """
    自适应多重叠率滑动窗口管理器 (V3)
    默认设定为 N=1 (无重叠)，实现了 Overlap-Add 机制的兼容性处理。
    """
    def __init__(
        self,
        window_size=500,
        N=1,
        n_channels=19,
        stfnet_checkpoint=None,
        stfnet_device="cpu",
        stfnet_input_len=500,
    ):
        """
        参数:
            N (重叠度参数): 默认为 1，即重叠率 rho = (1-1)/1 = 0%
        """
        self.L = window_size
        self.requested_N = N
        self.N = N
        self.n_channels = n_channels

        # 参数兜底：保证任意输入都可运行
        if self.L < 1:
            print(f"[Warning] window_size={self.L} 非法，已自动修正为 1")
            self.L = 1

        if self.N < 1:
            print(f"[Warning] N={self.N} 非法，已自动修正为 1 (无重叠模式)")
            self.N = 1

        # 非整除时采用 floor 步长，允许局部覆盖数超过 N；尾部不足窗口的数据自动丢弃
        self.S = max(1, self.L // self.N)
        self.max_overlap_count = int(np.ceil(self.L / self.S))
        self.stfnet_denoiser = None
        self.stfnet_enabled = stfnet_checkpoint is not None
        self.last_stfnet_infer_ms = None
        self.stfnet_infer_ms = []
        if self.stfnet_enabled:
            self.stfnet_denoiser = STFNetWindowDenoiser(
                checkpoint_path=stfnet_checkpoint,
                device=stfnet_device,
                model_input_len=stfnet_input_len,
            )
            print(
                f"[STFNet] Enabled. ckpt={stfnet_checkpoint}, device={self.stfnet_denoiser.device}"
            )
        
        # 缓冲区初始化
        self.input_buffer = np.empty((n_channels, 0), dtype=np.float32)
        self.input_time_buffer = np.empty((0,), dtype=np.float64)
        self.ola_buffer = np.zeros((n_channels, self.L), dtype=np.float32)
        self.count_buffer = np.zeros(self.L, dtype=np.float32)
        
        self.reconstructed_count = 0 
        self.on_reconstructed_chunk = None
        # 记录每个输出窗口（chunk）的端到端时延（从收到首包到窗口输出）
        self.output_latency_ms = []
        self.last_output_latency_ms = None

    def receive_packet(self, packet, packet_receive_time_ms=None):
        """接收小包并执行加权融合"""
        if packet_receive_time_ms is None:
            packet_receive_time_ms = time.perf_counter() * 1000.0

        packet_len = packet.shape[1]
        self.input_buffer = np.concatenate([self.input_buffer, packet], axis=1)
        packet_times = np.full((packet_len,), packet_receive_time_ms, dtype=np.float64)
        self.input_time_buffer = np.concatenate([self.input_time_buffer, packet_times], axis=0)

        while self.input_buffer.shape[1] >= self.L:
            window = self.input_buffer[:, :self.L]

            if self.stfnet_denoiser is not None:
                infer_t0 = time.perf_counter() * 1000.0
                processed_window = self.stfnet_denoiser.denoise(window)
                self.last_stfnet_infer_ms = (time.perf_counter() * 1000.0) - infer_t0
                self.stfnet_infer_ms.append(float(self.last_stfnet_infer_ms))
            else:
                processed_window = window

            # 叠加至重构缓冲区
            self.ola_buffer += processed_window
            self.count_buffer += 1.0

            # 提取前 S 个点并归一化输出
            counts = self.count_buffer[:self.S]
            reconstructed_chunk = self.ola_buffer[:, :self.S] / (counts + 1e-8)

            # 使用输出块最早样本的接收时间作为起点，衡量窗口完整输出时延
            chunk_start_receive_time_ms = (
                self.input_time_buffer[0]
                if self.input_time_buffer.size > 0
                else packet_receive_time_ms
            )
            latency_ms = (time.perf_counter() * 1000.0) - chunk_start_receive_time_ms
            self.last_output_latency_ms = float(latency_ms)
            self.output_latency_ms.append(self.last_output_latency_ms)
            
            self.reconstructed_count += 1
            if self.on_reconstructed_chunk:
                try:
                    self.on_reconstructed_chunk(
                        reconstructed_chunk, self.reconstructed_count, self.last_output_latency_ms
                    )
                except TypeError:
                    # 兼容旧版回调签名: callback(chunk, chunk_id)
                    self.on_reconstructed_chunk(reconstructed_chunk, self.reconstructed_count)

            # 缓冲区步进滑动
            self.input_buffer = self.input_buffer[:, self.S:]
            self.input_time_buffer = self.input_time_buffer[self.S:]
            
            # 信号叠加区左移 S
            self.ola_buffer = np.concatenate([
                self.ola_buffer[:, self.S:], 
                np.zeros((self.n_channels, self.S), dtype=np.float32)
            ], axis=1)
            
            # 计数区同步左移 S
            self.count_buffer = np.concatenate([
                self.count_buffer[self.S:], 
                np.zeros(self.S, dtype=np.float32)
            ])