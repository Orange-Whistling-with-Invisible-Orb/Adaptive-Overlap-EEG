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


class DualScaleWeightInferencer:
    """
    双尺度位置评估头推理器。
    输入 K 个窗口 [K, C, L]，可输出：
    - 位置相关 softmax 权重 [K, S]（normalize=True）
    - 位置相关未约束 logits [K, S]（normalize=False）
    兼容旧调用：未指定 chunk_len 时，返回窗口级平均权重 [K]。

    说明：
    - 训练端使用“logits + 初始位置偏置 -> softmax(logits / temperature)”得到权重；
    - 推理端这里对齐同一规则，避免训练/推理机制不一致带来的数值漂移。
    """

    def __init__(
        self,
        checkpoint_path,
        n_channels: int,
        window_len: int,
        device="cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.n_channels = int(n_channels)
        self.window_len = int(window_len)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Fusion weight checkpoint not found: {self.checkpoint_path}"
            )
        try:
            import torch
        except Exception as e:
            raise ImportError(f"Failed to import torch for fusion scorer: {e}") from e

        self.torch = torch
        self.device = self._resolve_device(device)
        self.softmax_temperature = 1.0
        self.init_logit_bias_strength = 0.0
        self.init_window_weights = None
        self.model = self._load_model()
        self.model.eval().to(self.device)

    def _resolve_device(self, device):
        device = str(device)
        if device.startswith("cuda") and not self.torch.cuda.is_available():
            print("[FusionWeight] CUDA unavailable, fallback to CPU.")
            return self.torch.device("cpu")
        return self.torch.device(device)

    def _load_model(self):
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))

        from core_algorithm.dual_scale_scorer import DualScalePositionScorer

        try:
            ckpt = self.torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            ckpt = self.torch.load(str(self.checkpoint_path), map_location=self.device)

        # 1) 若直接存的是模型对象
        if hasattr(ckpt, "forward") and hasattr(ckpt, "state_dict"):
            return ckpt

        # 2) 常规 state_dict 格式
        if not isinstance(ckpt, dict):
            raise TypeError(
                f"Unsupported fusion checkpoint format: {type(ckpt)} at {self.checkpoint_path}"
            )

        self.softmax_temperature = float(max(1e-6, ckpt.get("softmax_temperature", 1.0)))
        self.init_logit_bias_strength = float(
            max(0.0, ckpt.get("init_logit_bias_strength", 0.0))
        )
        self.init_window_weights = self._normalize_init_window_weights(
            ckpt.get("init_window_weights", None)
        )

        model = DualScalePositionScorer(
            n_channels=int(ckpt.get("n_channels", self.n_channels)),
            window_len=int(ckpt.get("window_len", self.window_len)),
            local_kernel=int(ckpt.get("local_kernel", 15)),
        )
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "")
            if nk.startswith("scorer."):
                nk = nk[len("scorer.") :]
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
        return model

    @staticmethod
    def _resample_stack_time(windows, target_len):
        # windows: [K, C, T] -> [K, C, target_len]
        if windows.shape[-1] == target_len:
            return windows
        out = []
        for i in range(windows.shape[0]):
            out.append(STFNetWindowDenoiser._resample_time(windows[i], target_len))
        return np.stack(out, axis=0).astype(np.float32)

    @staticmethod
    def _softmax_over_windows(logits: np.ndarray) -> np.ndarray:
        """对每个时刻沿 K 维做 softmax，得到正数且列和为 1 的权重。"""
        shifted = logits - np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(shifted)
        denom = np.sum(exp_logits, axis=0, keepdims=True)
        return exp_logits / np.clip(denom, 1e-12, None)

    @staticmethod
    def _normalize_init_window_weights(init_window_weights):
        if init_window_weights is None:
            return None
        arr = np.asarray(init_window_weights, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        if not np.isfinite(arr).all():
            return None
        arr = np.clip(arr, 0.0, None)
        s = float(arr.sum())
        if s <= 0:
            return None
        return arr / s

    def _apply_initial_position_bias(self, logits: np.ndarray) -> np.ndarray:
        """
        logits: [K, S]
        与训练端 _apply_initial_position_bias 保持一致。
        """
        if logits.ndim != 2:
            return logits
        k, _ = logits.shape
        if k <= 1:
            return logits

        if self.init_window_weights is not None:
            base = np.asarray(self.init_window_weights, dtype=np.float32).reshape(-1)
            if base.size == k:
                prior_w = base
            else:
                # 线性插值到当前活跃窗口数 K，并归一化。
                src_x = np.linspace(0.0, 1.0, num=base.size, dtype=np.float32)
                dst_x = np.linspace(0.0, 1.0, num=k, dtype=np.float32)
                prior_w = np.interp(dst_x, src_x, base).astype(np.float32)
                prior_w = np.clip(prior_w, 1e-6, None)
                prior_w = prior_w / np.clip(float(prior_w.sum()), 1e-12, None)
            prior_logit = np.log(prior_w + 1e-12)
            prior_logit = prior_logit - np.mean(prior_logit)
            return logits + prior_logit[:, None]

        if self.init_logit_bias_strength <= 0:
            return logits
        pos = np.linspace(-1.0, 1.0, num=k, dtype=np.float32)
        prior = (pos**2) - np.mean(pos**2)
        return logits + float(self.init_logit_bias_strength) * prior[:, None]

    def predict(
        self,
        windows_kcl: np.ndarray,
        chunk_len: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        windows_kcl: [K, C, L]
        chunk_len:
            - 指定时返回 [K, chunk_len] 的位置相关权重
            - 不指定时返回 [K]（对全窗口位置权重取均值后的兼容输出）
        normalize:
            - True: 返回沿 K 维归一化后的 softmax 权重（非负，列和为 1）
            - False: 返回未归一化 logits（可正可负，列和不受约束）
        """
        windows = np.asarray(windows_kcl, dtype=np.float32)
        if windows.ndim != 3:
            raise ValueError(
                f"DualScaleWeightInferencer expects [K,C,L], got {windows.shape}"
            )
        k, c, l = windows.shape
        if c != self.n_channels:
            raise ValueError(f"Channel mismatch: expected {self.n_channels}, got {c}")
        if l != self.window_len:
            windows = self._resample_stack_time(windows, self.window_len)

        return_vector = chunk_len is None
        out_len = self.window_len if chunk_len is None else int(chunk_len)
        if out_len < 1 or out_len > self.window_len:
            raise ValueError(
                f"chunk_len must be in [1, {self.window_len}], got {out_len}"
            )

        x = self.torch.from_numpy(windows).unsqueeze(0).to(self.device)  # [1,K,C,L]
        with self.torch.no_grad():
            logits, _ = self.model(x, output_len=out_len)  # [1,K,out_len]
        logits_np = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [K,out_len]
        logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=0.0, neginf=0.0)
        biased_logits = self._apply_initial_position_bias(logits_np)

        if normalize:
            scaled_logits = biased_logits / float(self.softmax_temperature)
            w = self._softmax_over_windows(scaled_logits).astype(np.float32)
        else:
            w = biased_logits.astype(np.float32)

        if return_vector:
            w_vec = np.mean(w, axis=1).astype(np.float32)
            if not normalize:
                return w_vec
            s = float(w_vec.sum())
            if s <= 0:
                return np.ones((k,), dtype=np.float32) / max(1, k)
            return w_vec / s
        return w


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
        fusion_weight_checkpoint=None,
        fusion_weight_device="cpu",
        fusion_use_unconstrained_logits=False,
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

        self.direct_passthrough = self.N == 1
        self.fusion_use_unconstrained_logits = bool(fusion_use_unconstrained_logits)
        self.weight_inferencer = None
        if self.direct_passthrough and fusion_weight_checkpoint is not None:
            print("[FusionWeight] N=1 mode: bypass fusion scorer and use direct STFNet output.")
        self.weight_enabled = (
            fusion_weight_checkpoint is not None and not self.direct_passthrough
        )
        self.last_weight_infer_ms = None
        self.weight_infer_ms = []
        self.last_weight_vector = None
        self.last_weight_map = None
        if self.weight_enabled:
            self.weight_inferencer = DualScaleWeightInferencer(
                checkpoint_path=fusion_weight_checkpoint,
                n_channels=self.n_channels,
                window_len=self.L,
                device=fusion_weight_device,
            )
            fusion_mode = (
                "unconstrained_logits"
                if self.fusion_use_unconstrained_logits
                else "softmax_normalized"
            )
            print(
                f"[FusionWeight] Enabled. ckpt={fusion_weight_checkpoint}, "
                f"device={self.weight_inferencer.device}, mode={fusion_mode}"
            )
        
        # 缓冲区初始化
        self.input_buffer = np.empty((n_channels, 0), dtype=np.float32)
        self.input_time_buffer = np.empty((0,), dtype=np.float64)
        self.ola_buffer = np.zeros((n_channels, self.L), dtype=np.float32)
        self.count_buffer = np.zeros(self.L, dtype=np.float32)
        self.active_full_windows = []
        self.active_shifted_windows = []
        self.active_remaining_steps = []
        
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

            if self.direct_passthrough:
                # N=1 严格退化：直接输出 STFNet 窗口，不经过融合打分网络。
                reconstructed_chunk = processed_window[:, : self.S]
            elif self.weight_inferencer is not None:
                # 进入“可学习权重”融合分支
                self.active_full_windows.append(processed_window)
                self.active_shifted_windows.append(processed_window.copy())
                self.active_remaining_steps.append(self.max_overlap_count)

                # 与训练保持一致：使用 active_shifted 作为打分输入
                score_stack = np.stack(self.active_shifted_windows, axis=0)  # [K,C,L]
                weight_t0 = time.perf_counter() * 1000.0
                # 默认使用与训练一致的 softmax 归一化权重，避免数值爆炸。
                weights = self.weight_inferencer.predict(
                    score_stack,
                    chunk_len=self.S,
                    normalize=(not self.fusion_use_unconstrained_logits),
                )  # [K,S]
                self.last_weight_infer_ms = (time.perf_counter() * 1000.0) - weight_t0
                self.weight_infer_ms.append(float(self.last_weight_infer_ms))
                self.last_weight_map = weights.copy()
                self.last_weight_vector = np.mean(weights, axis=1).astype(np.float32)

                seg_stack = np.stack(
                    [w[:, : self.S] for w in self.active_shifted_windows], axis=0
                )  # [K,C,S]
                reconstructed_chunk = np.sum(
                    seg_stack * weights[:, None, :], axis=0, dtype=np.float32
                )
            else:
                # 旧版“均值融合”分支
                self.ola_buffer += processed_window
                self.count_buffer += 1.0
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

            if self.direct_passthrough:
                # N=1 直通模式无活动窗口状态。
                pass
            elif self.weight_inferencer is not None:
                # 学习权重分支下，推进每个活动窗口状态
                new_full = []
                new_shifted = []
                new_remaining = []
                for full_w, shifted_w, rem in zip(
                    self.active_full_windows,
                    self.active_shifted_windows,
                    self.active_remaining_steps,
                ):
                    shifted_next = np.concatenate(
                        [
                            shifted_w[:, self.S :],
                            np.zeros((self.n_channels, self.S), dtype=np.float32),
                        ],
                        axis=1,
                    )
                    rem = rem - 1
                    if rem > 0:
                        new_full.append(full_w)
                        new_shifted.append(shifted_next)
                        new_remaining.append(rem)
                self.active_full_windows = new_full
                self.active_shifted_windows = new_shifted
                self.active_remaining_steps = new_remaining
            else:
                # 均值融合分支保持原逻辑
                self.ola_buffer = np.concatenate(
                    [
                        self.ola_buffer[:, self.S :],
                        np.zeros((self.n_channels, self.S), dtype=np.float32),
                    ],
                    axis=1,
                )
                self.count_buffer = np.concatenate(
                    [self.count_buffer[self.S :], np.zeros(self.S, dtype=np.float32)]
                )