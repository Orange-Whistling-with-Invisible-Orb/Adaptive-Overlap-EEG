from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from core_algorithm.dual_scale_scorer import DualScalePositionScorer
from core_algorithm.training.loss_functions import combined_reconstruction_loss


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
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        local_kernel: int = 15,
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        grad_clip_norm: float = 5.0,
    ):
        self.n_channels = int(n_channels)
        self.L = int(window_len)
        self.N = max(1, int(overlap_n))
        self.S = max(1, self.L // self.N)
        self.max_overlap_count = int((self.L + self.S - 1) // self.S)
        self.packet_samples = max(1, int(packet_samples))
        self.mse_weight = float(mse_weight)
        self.l1_weight = float(l1_weight)
        self.grad_clip_norm = float(grad_clip_norm)

        self.device = self._resolve_device(device)
        self.stfnet = self._load_frozen_stfnet(stfnet_checkpoint).to(self.device).eval()

        self.scorer = DualScalePositionScorer(
            n_channels=self.n_channels, window_len=self.L, local_kernel=local_kernel
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.scorer.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )

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

    def _run_subject(
        self,
        cont_seq: torch.Tensor,
        pure_seq: torch.Tensor,
        train: bool,
        collect_output: bool = False,
    ):
        if cont_seq.shape != pure_seq.shape:
            raise ValueError(
                f"cont/pure shape mismatch: {tuple(cont_seq.shape)} vs {tuple(pure_seq.shape)}"
            )
        if cont_seq.shape[1] < self.L:
            return {
                "losses": [],
                "chunks": 0,
                "pred_chunks": [],
                "target_chunks": [],
            }

        buffer_cont = torch.zeros((self.n_channels, 0), device=self.device)
        buffer_pure = torch.zeros((self.n_channels, 0), device=self.device)

        active_full = []
        active_shifted = []
        active_remaining = []

        losses = []
        pred_chunks = []
        tgt_chunks = []

        total_t = int(cont_seq.shape[1])
        self.scorer.train(mode=train)

        for pkt_start in range(0, total_t, self.packet_samples):
            pkt_end = min(total_t, pkt_start + self.packet_samples)
            pkt_cont = cont_seq[:, pkt_start:pkt_end]
            pkt_pure = pure_seq[:, pkt_start:pkt_end]
            buffer_cont = torch.cat([buffer_cont, pkt_cont], dim=1)
            buffer_pure = torch.cat([buffer_pure, pkt_pure], dim=1)

            while buffer_cont.shape[1] >= self.L and buffer_pure.shape[1] >= self.L:
                # 1) STFNet 窗口预处理
                window_cont = buffer_cont[:, : self.L]
                denoised_window = self._stfnet_denoise(window_cont)
                active_full.append(denoised_window)
                active_shifted.append(denoised_window.clone())
                active_remaining.append(self.max_overlap_count)

                # 2) 双尺度评估头计算权重
                full_stack = torch.stack(active_full, dim=0).unsqueeze(0)  # [1,K,C,L]
                if train:
                    _, weights = self.scorer(full_stack)  # [1,K]
                else:
                    with torch.no_grad():
                        _, weights = self.scorer(full_stack)
                w = weights.squeeze(0)  # [K]

                # 3) 对当前输出 chunk(前 S 点)做加权融合
                seg_stack = torch.stack(
                    [s[:, : self.S] for s in active_shifted], dim=0
                )  # [K,C,S]
                fused_chunk = (w[:, None, None] * seg_stack).sum(dim=0)  # [C,S]
                target_chunk = buffer_pure[:, : self.S]

                loss = combined_reconstruction_loss(
                    fused_chunk,
                    target_chunk,
                    mse_weight=self.mse_weight,
                    l1_weight=self.l1_weight,
                )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.scorer.parameters(), self.grad_clip_norm
                        )
                    self.optimizer.step()

                losses.append(float(loss.detach().cpu().item()))
                if collect_output:
                    pred_chunks.append(fused_chunk.detach().cpu())
                    tgt_chunks.append(target_chunk.detach().cpu())

                # 4) 步进：输入缓冲左移 S
                buffer_cont = buffer_cont[:, self.S :]
                buffer_pure = buffer_pure[:, self.S :]

                # 5) 活跃窗口状态左移 S，并移除无效窗口
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
            "chunks": len(losses),
            "pred_chunks": pred_chunks,
            "target_chunks": tgt_chunks,
        }

    def run_epoch(self, samples, train: bool, epoch_idx: int = 0, verbose: bool = True):
        """
        samples: list[(cont_seq[C,T], pure_seq[C,T])]
        """
        epoch_losses = []
        total_chunks = 0
        t0 = time.time()

        for i, (cont_np, pure_np) in enumerate(samples, start=1):
            cont_seq = self._to_tensor(cont_np, self.device)
            pure_seq = self._to_tensor(pure_np, self.device)
            out = self._run_subject(cont_seq, pure_seq, train=train, collect_output=False)
            epoch_losses.extend(out["losses"])
            total_chunks += out["chunks"]

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
        return {
            "mean_loss": float(mean_loss),
            "chunks": int(total_chunks),
            "elapsed_sec": float(elapsed),
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
