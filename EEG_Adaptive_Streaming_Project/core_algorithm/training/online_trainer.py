from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core_algorithm.dual_scale_scorer import DualScalePositionScorer
from core_algorithm.training.loss_functions import combined_reconstruction_loss
from stream_receiver.tcp_receiver import BasicPreprocessor


class StreamingFusionTrainer:
    """
    训练“窗口权重网络”的流式仿真器。

    关键点：
    - STFNet 冻结，仅作为每个窗口的预处理器。
    - DualScalePositionScorer 可训练，通过流式融合误差反向更新。
    - 融合逻辑遵循 window_len / overlap_n / packet_samples 的在线推进。
    """

    def __init__(
        self,
        stfnet_checkpoint: str,
        n_channels: int,
        window_len: int,
        overlap_n: int,
        packet_samples: int,
        device: str = "cpu",
        sample_rate: float = 200.0,
        preprocess_mode: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        local_kernel: int = 15,
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        grad_clip_norm: float = 5.0,
        entropy_reg_weight: float = 0.01,
        softmax_temperature: float = 1.0,
        init_logit_bias_strength: float = 0.35,
        init_window_weights: Optional[list[float]] = None,
    ):
        self.n_channels = int(n_channels)
        self.L = int(window_len)
        self.N = max(1, int(overlap_n))
        self.direct_passthrough = self.N == 1
        self.S = max(1, self.L // self.N)
        self.max_overlap_count = int((self.L + self.S - 1) // self.S)
        self.packet_samples = max(1, int(packet_samples))
        self.sample_rate = float(sample_rate)
        self.preprocess_mode = preprocess_mode
        self.mse_weight = float(mse_weight)
        self.l1_weight = float(l1_weight)
        self.grad_clip_norm = float(grad_clip_norm)
        self.entropy_reg_weight = float(max(0.0, entropy_reg_weight))
        self.softmax_temperature = float(max(1e-6, softmax_temperature))
        self.init_logit_bias_strength = float(max(0.0, init_logit_bias_strength))
        self.init_window_weights = self._normalize_init_window_weights(init_window_weights)

        self.device = self._resolve_device(device)
        self.stfnet = self._load_frozen_stfnet(stfnet_checkpoint).to(self.device).eval()

        self.scorer = DualScalePositionScorer(
            n_channels=self.n_channels, window_len=self.L, local_kernel=local_kernel
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.scorer.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        self.preprocessor = BasicPreprocessor(
            sample_rate=self.sample_rate, preprocess_mode=self.preprocess_mode
        )

    @staticmethod
    def _normalize_init_window_weights(
        init_window_weights: Optional[list[float]],
    ) -> Optional[np.ndarray]:
        if init_window_weights is None:
            return None
        arr = np.asarray(init_window_weights, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        if not np.isfinite(arr).all():
            raise ValueError("init_window_weights contains non-finite values.")
        if np.any(arr < 0):
            raise ValueError("init_window_weights must be >= 0.")
        s = float(arr.sum())
        if s <= 0:
            raise ValueError("init_window_weights sum must be > 0.")
        return arr / s

    @staticmethod
    def _to_tensor(seq, device: torch.device) -> torch.Tensor:
        t = torch.as_tensor(seq, dtype=torch.float32, device=device)
        if t.ndim != 2:
            raise ValueError(f"Expected sequence shape [C,T], got {tuple(t.shape)}")
        return t

    def _resolve_device(self, device: str) -> torch.device:
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("[Trainer] CUDA unavailable, fallback to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    def _load_frozen_stfnet(self, checkpoint_path: str) -> nn.Module:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"STFNet checkpoint not found: {ckpt}")

        project_root = Path(__file__).resolve().parents[2]
        stfnet_module_dir = project_root / "stfnet_module"
        if str(stfnet_module_dir) not in sys.path:
            sys.path.append(str(stfnet_module_dir))

        try:
            model = torch.load(
                str(ckpt), map_location=self.device, weights_only=False
            )
        except TypeError:
            model = torch.load(str(ckpt), map_location=self.device)

        if not hasattr(model, "forward"):
            raise TypeError(
                f"Checkpoint {ckpt} is not a full model object and cannot be used directly."
            )

        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @staticmethod
    def _resample_time_torch(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        x: [C, T] -> [C, target_len]
        """
        if x.shape[1] == target_len:
            return x
        if x.shape[1] <= 1 or target_len <= 1:
            return x[:, :1].repeat(1, target_len)
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    def _stfnet_denoise(self, window_cont: torch.Tensor) -> torch.Tensor:
        # STFNet 按 500 点训练，若 L!=500，做时间轴重采样后再还原长度。
        orig_len = int(window_cont.shape[1])
        x = window_cont
        if orig_len != 500:
            x = self._resample_time_torch(x, 500)
        with torch.no_grad():
            out = self.stfnet(x.unsqueeze(0)).squeeze(0)
        if orig_len != 500:
            out = self._resample_time_torch(out, orig_len)
        return out.detach()

    @staticmethod
    def _chunk_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(torch.mean((pred - target) ** 2).detach().cpu().item())

    @staticmethod
    def _chunk_snr_db(pred: torch.Tensor, target: torch.Tensor) -> float:
        signal_power = torch.mean(target**2)
        noise_power = torch.mean((target - pred) ** 2)
        snr = 10.0 * torch.log10((signal_power + 1e-12) / (noise_power + 1e-12))
        return float(snr.detach().cpu().item())

    def _apply_initial_position_bias(self, logits: torch.Tensor) -> torch.Tensor:
        """
        给窗口 logits 注入固定位置先验，避免初始接近 1/K 均分。
        该偏置是确定性的，不依赖随机噪声。
        """
        if logits.ndim not in (2, 3):
            raise ValueError(
                f"logits must be [B,K] or [B,K,T], got shape={tuple(logits.shape)}"
            )
        k = logits.shape[1]
        if k <= 1:
            return logits

        if self.init_window_weights is not None:
            base = torch.as_tensor(
                self.init_window_weights, dtype=logits.dtype, device=logits.device
            )
            if int(base.numel()) == k:
                prior_w = base
            else:
                # 用户给定长度与当前活跃窗口数不一致时，线性插值到 K 再归一化。
                prior_w = F.interpolate(
                    base.view(1, 1, -1), size=k, mode="linear", align_corners=True
                ).view(-1)
                prior_w = torch.clamp(prior_w, min=1e-6)
                prior_w = prior_w / torch.sum(prior_w)
            prior_logit = torch.log(prior_w + 1e-12)
            prior_logit = prior_logit - torch.mean(prior_logit)
            if logits.ndim == 2:
                return logits + prior_logit.unsqueeze(0)
            return logits + prior_logit.view(1, k, 1)

        if self.init_logit_bias_strength <= 0:
            return logits
        pos = torch.linspace(-1.0, 1.0, steps=k, device=logits.device, dtype=logits.dtype)
        # 两端高、中间低的轻微偏置，打破均分对称性
        prior = (pos**2) - torch.mean(pos**2)
        if logits.ndim == 2:
            return logits + self.init_logit_bias_strength * prior.unsqueeze(0)
        return logits + self.init_logit_bias_strength * prior.view(1, k, 1)

    def _weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.softmax_temperature, dim=1)

    @staticmethod
    def _weight_entropy(weights: torch.Tensor) -> torch.Tensor:
        """
        支持:
        - [K]: 单时刻窗口权重熵
        - [K,S]: 位置相关权重的平均熵（对 S 维求均值）
        - [B,K,S]: batch 版本平均熵
        """
        if weights.ndim == 1:
            return -(weights * torch.log(weights + 1e-12)).sum()
        if weights.ndim == 2:
            ent_t = -(weights * torch.log(weights + 1e-12)).sum(dim=0)  # [S]
            return ent_t.mean()
        if weights.ndim == 3:
            ent_bt = -(weights * torch.log(weights + 1e-12)).sum(dim=1)  # [B,S]
            return ent_bt.mean()
        raise ValueError(f"Unsupported weight shape for entropy: {tuple(weights.shape)}")

    def _preprocess_packet_pair(
        self, cont_pkt: np.ndarray, pure_pkt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        对 contaminated/pure 同步预处理，保证监督目标与输入处于同一尺度域。

        设计原则：
        - preprocess_mode=2: 不预处理，原样返回。
        - preprocess_mode=1: 带通后做“按 contaminated 标准差”的成对标准化：
            cont <- (cont - mean(cont)) / std(cont)
            pure <- (pure - mean(pure)) / std(cont)
        这与 STFNet 训练使用的 Standardization 规则保持一致，可避免目标尺度漂移。
        """
        x = np.asarray(cont_pkt, dtype=np.float32)
        y = np.asarray(pure_pkt, dtype=np.float32)
        if self.preprocess_mode == 2:
            return np.copy(x), np.copy(y)

        # 与在线基础预处理一致：先带通（短包时 _safe_bandpass 会自动回退）
        x_f = self.preprocessor._safe_bandpass(x)
        y_f = self.preprocessor._safe_bandpass(y)

        x_mean = np.mean(x_f, axis=1, keepdims=True)
        y_mean = np.mean(y_f, axis=1, keepdims=True)
        x_std = np.std(x_f, axis=1, keepdims=True) + 1e-8

        x_n = (x_f - x_mean) / x_std
        y_n = (y_f - y_mean) / x_std
        x_n = np.nan_to_num(x_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_n = np.nan_to_num(y_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return x_n, y_n

    def _run_subject(
        self,
        cont_seq: torch.Tensor,
        pure_seq: torch.Tensor,
        train: bool,
        collect_output: bool = False,
        collect_weights: bool = False,
        subject_idx: int = 0,
        epoch_idx: int = 0,
        mode: str = "train",
    ):
        if cont_seq.shape != pure_seq.shape:
            raise ValueError(
                f"cont/pure shape mismatch: {tuple(cont_seq.shape)} vs {tuple(pure_seq.shape)}"
            )
        if cont_seq.shape[1] < self.L:
            return {
                "losses": [],
                "recon_losses": [],
                "mses": [],
                "snrs_db": [],
                "effective_ks": [],
                "chunks": 0,
                "pred_chunks": [],
                "target_chunks": [],
                "weight_records": [],
            }

        buffer_cont = torch.zeros((self.n_channels, 0), device=self.device)
        buffer_pure = torch.zeros((self.n_channels, 0), device=self.device)

        active_full = []
        active_shifted = []
        active_remaining = []

        losses = []
        recon_losses = []
        mses = []
        snrs_db = []
        effective_ks = []
        pred_chunks = []
        tgt_chunks = []
        weight_records = []

        total_t = int(cont_seq.shape[1])
        self.scorer.train(mode=train)
        chunk_idx = 0

        for pkt_start in range(0, total_t, self.packet_samples):
            pkt_end = min(total_t, pkt_start + self.packet_samples)
            pkt_cont = cont_seq[:, pkt_start:pkt_end]
            pkt_pure = pure_seq[:, pkt_start:pkt_end]

            # 与 supervised 目标对齐：contaminated / pure 成对同步预处理
            pkt_cont_np = pkt_cont.detach().cpu().numpy()
            pkt_pure_np = pkt_pure.detach().cpu().numpy()
            pkt_cont_np, pkt_pure_np = self._preprocess_packet_pair(
                pkt_cont_np, pkt_pure_np
            )
            pkt_cont = torch.from_numpy(pkt_cont_np).to(self.device)
            pkt_pure = torch.from_numpy(pkt_pure_np).to(self.device)

            buffer_cont = torch.cat([buffer_cont, pkt_cont], dim=1)
            buffer_pure = torch.cat([buffer_pure, pkt_pure], dim=1)

            while buffer_cont.shape[1] >= self.L and buffer_pure.shape[1] >= self.L:
                # 1) STFNet 窗口预处理
                window_cont = buffer_cont[:, : self.L]
                denoised_window = self._stfnet_denoise(window_cont)
                target_chunk = buffer_pure[:, : self.S]
                if self.direct_passthrough:
                    # N=1 严格退化：直接用 STFNet 输出，不引入融合权重网络。
                    fused_chunk = denoised_window[:, : self.S]
                    recon_loss = combined_reconstruction_loss(
                        fused_chunk,
                        target_chunk,
                        mse_weight=self.mse_weight,
                        l1_weight=self.l1_weight,
                    )
                    entropy_term = torch.zeros(
                        (), dtype=fused_chunk.dtype, device=fused_chunk.device
                    )
                    loss = recon_loss
                    w_profile = torch.ones((1,), dtype=fused_chunk.dtype, device=self.device)
                    max_weight_value = 1.0
                else:
                    active_full.append(denoised_window)
                    active_shifted.append(denoised_window.clone())
                    active_remaining.append(self.max_overlap_count)

                    # 2) 双尺度评估头计算位置相关权重（与推理保持一致，用 shifted 对齐）
                    score_stack = torch.stack(active_shifted, dim=0).unsqueeze(0)  # [1,K,C,L]
                    if train:
                        logits, _ = self.scorer(score_stack, output_len=self.S)  # [1,K,S]
                        biased_logits = self._apply_initial_position_bias(logits)
                        weights = self._weights_from_logits(biased_logits)  # [1,K,S]
                    else:
                        with torch.no_grad():
                            logits, _ = self.scorer(score_stack, output_len=self.S)
                            biased_logits = self._apply_initial_position_bias(logits)
                            weights = self._weights_from_logits(biased_logits)
                    w_map = weights.squeeze(0)  # [K,S]

                    # 3) 对当前输出 chunk(前 S 点)做位置相关加权融合
                    seg_stack = torch.stack(
                        [s[:, : self.S] for s in active_shifted], dim=0
                    )  # [K,C,S]
                    fused_chunk = (w_map[:, None, :] * seg_stack).sum(dim=0)  # [C,S]

                    recon_loss = combined_reconstruction_loss(
                        fused_chunk,
                        target_chunk,
                        mse_weight=self.mse_weight,
                        l1_weight=self.l1_weight,
                    )
                    entropy_term = self._weight_entropy(w_map)
                    loss = recon_loss - self.entropy_reg_weight * entropy_term

                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if self.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.scorer.parameters(), self.grad_clip_norm
                            )
                        self.optimizer.step()
                    w_profile = torch.mean(w_map, dim=1)  # [K]
                    max_weight_value = float(torch.max(w_map).detach().cpu().item())

                losses.append(float(loss.detach().cpu().item()))
                recon_losses.append(float(recon_loss.detach().cpu().item()))
                chunk_mse = self._chunk_mse(fused_chunk, target_chunk)
                chunk_snr_db = self._chunk_snr_db(fused_chunk, target_chunk)
                mses.append(chunk_mse)
                snrs_db.append(chunk_snr_db)
                entropy_value = float(entropy_term.detach().cpu().item())
                chunk_effective_k = float(np.exp(entropy_value))
                effective_ks.append(chunk_effective_k)
                if collect_output:
                    pred_chunks.append(fused_chunk.detach().cpu())
                    tgt_chunks.append(target_chunk.detach().cpu())
                if collect_weights:
                    w_np = w_profile.detach().cpu().numpy().astype(float)
                    weight_records.append(
                        {
                            "epoch": int(epoch_idx),
                            "mode": str(mode),
                            "subject_idx": int(subject_idx),
                            "chunk_idx": int(chunk_idx),
                            "k_active": int(w_np.shape[0]),
                            "weights": w_np.tolist(),
                            "entropy": entropy_value,
                            "effective_k": chunk_effective_k,
                            "max_weight": max_weight_value,
                            "recon_loss": float(recon_loss.detach().cpu().item()),
                            "total_loss": float(loss.detach().cpu().item()),
                            "mse": chunk_mse,
                            "snr_db": chunk_snr_db,
                        }
                    )
                chunk_idx += 1

                # 4) 步进：输入缓冲左移 S
                buffer_cont = buffer_cont[:, self.S :]
                buffer_pure = buffer_pure[:, self.S :]

                # 5) 活跃窗口状态左移 S，并移除无效窗口
                if not self.direct_passthrough:
                    new_full = []
                    new_shifted = []
                    new_remaining = []
                    for full_w, shift_w, rem in zip(
                        active_full, active_shifted, active_remaining
                    ):
                        shifted = torch.cat(
                            [
                                shift_w[:, self.S :],
                                torch.zeros(
                                    (self.n_channels, self.S),
                                    dtype=shift_w.dtype,
                                    device=self.device,
                                ),
                            ],
                            dim=1,
                        )
                        rem = rem - 1
                        if rem > 0:
                            new_full.append(full_w)
                            new_shifted.append(shifted.detach())
                            new_remaining.append(rem)
                    active_full, active_shifted, active_remaining = (
                        new_full,
                        new_shifted,
                        new_remaining,
                    )

        return {
            "losses": losses,
            "recon_losses": recon_losses,
            "mses": mses,
            "snrs_db": snrs_db,
            "effective_ks": effective_ks,
            "chunks": len(losses),
            "pred_chunks": pred_chunks,
            "target_chunks": tgt_chunks,
            "weight_records": weight_records,
        }

    def run_epoch(self, samples, train: bool, epoch_idx: int = 0, verbose: bool = True):
        """
        samples: list[(cont_seq[C,T], pure_seq[C,T])]
        """
        epoch_losses = []
        epoch_recon_losses = []
        epoch_mses = []
        epoch_snrs_db = []
        epoch_effective_ks = []
        total_chunks = 0
        epoch_weight_records = []
        t0 = time.time()
        # 无随机扰动计数器；训练过程完全确定性（给定同样seed与数据）。

        for i, (cont_np, pure_np) in enumerate(samples, start=1):
            cont_seq = self._to_tensor(cont_np, self.device)
            pure_seq = self._to_tensor(pure_np, self.device)
            mode = "train" if train else "val"
            out = self._run_subject(
                cont_seq,
                pure_seq,
                train=train,
                collect_output=False,
                collect_weights=True,
                subject_idx=i - 1,
                epoch_idx=epoch_idx,
                mode=mode,
            )
            epoch_losses.extend(out["losses"])
            epoch_recon_losses.extend(out["recon_losses"])
            epoch_mses.extend(out["mses"])
            epoch_snrs_db.extend(out["snrs_db"])
            epoch_effective_ks.extend(out["effective_ks"])
            total_chunks += out["chunks"]
            epoch_weight_records.extend(out["weight_records"])

            if verbose:
                mode = "train" if train else "val"
                cur_loss = (
                    sum(out["losses"]) / len(out["losses"]) if out["losses"] else float("nan")
                )
                print(
                    f"[{mode}] epoch={epoch_idx:03d} subject={i}/{len(samples)} "
                    f"chunks={out['chunks']} mean_loss={cur_loss:.6f}"
                )

        elapsed = time.time() - t0
        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        mean_recon_loss = (
            sum(epoch_recon_losses) / len(epoch_recon_losses)
            if epoch_recon_losses
            else float("inf")
        )
        mean_mse = sum(epoch_mses) / len(epoch_mses) if epoch_mses else float("inf")
        mean_snr_db = (
            sum(epoch_snrs_db) / len(epoch_snrs_db) if epoch_snrs_db else float("-inf")
        )
        mean_effective_k = (
            sum(epoch_effective_ks) / len(epoch_effective_ks)
            if epoch_effective_ks
            else 0.0
        )
        max_k = self.max_overlap_count
        weight_means = [0.0] * max_k
        if len(epoch_weight_records) > 0:
            padded = np.zeros((len(epoch_weight_records), max_k), dtype=np.float32)
            ent = []
            maxw = []
            for ridx, rec in enumerate(epoch_weight_records):
                w = np.asarray(rec["weights"], dtype=np.float32)
                k = min(max_k, w.shape[0])
                padded[ridx, :k] = w[:k]
                ent.append(float(rec["entropy"]))
                maxw.append(float(rec["max_weight"]))
            weight_means = [float(v) for v in padded.mean(axis=0).tolist()]
            entropy_mean = float(np.mean(ent))
            max_weight_mean = float(np.mean(maxw))
        else:
            entropy_mean = 0.0
            max_weight_mean = 0.0

        return {
            "mean_loss": float(mean_loss),
            "mean_recon_loss": float(mean_recon_loss),
            "mean_mse": float(mean_mse),
            "mean_snr_db": float(mean_snr_db),
            "mean_effective_k": float(mean_effective_k),
            "chunks": int(total_chunks),
            "elapsed_sec": float(elapsed),
            "weight_records": epoch_weight_records,
            "weight_means": weight_means,
            "weight_entropy_mean": entropy_mean,
            "weight_max_mean": max_weight_mean,
        }

    def reconstruct_sample(self, cont_np, pure_np):
        """
        返回单条序列的融合输出与目标（用于保存可视化/调试结果）。
        """
        cont_seq = self._to_tensor(cont_np, self.device)
        pure_seq = self._to_tensor(pure_np, self.device)
        out = self._run_subject(cont_seq, pure_seq, train=False, collect_output=True)
        if len(out["pred_chunks"]) == 0:
            return None, None
        pred = torch.cat(out["pred_chunks"], dim=1).numpy()
        tgt = torch.cat(out["target_chunks"], dim=1).numpy()
        return pred, tgt
