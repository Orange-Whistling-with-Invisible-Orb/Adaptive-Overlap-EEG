from __future__ import annotations

import torch
from torch import nn


class DualScalePositionScorer(nn.Module):
    """
    双尺度位置评估头（Dual-Scale Positional Evaluation Head）。

    输入:
        windows: [B, K, C, L]
          - B: batch size
          - K: 当前融合时刻参与竞争的窗口数
          - C: 通道数
          - L: 窗口长度

    输出:
        logits:  [B, K, T]
        weights: [B, K, T]  (对每个时刻 t 沿 K 维 softmax 归一化)
        其中 T = output_len（未指定时 T=L）。
    """

    def __init__(self, n_channels: int, window_len: int, local_kernel: int = 15):
        super().__init__()
        if local_kernel < 1:
            local_kernel = 1
        if local_kernel % 2 == 0:
            local_kernel += 1

        self.n_channels = int(n_channels)
        self.window_len = int(window_len)
        self.local_kernel = int(local_kernel)

        # 局部支路：DW-Conv1d(K=15, P=7) + GMP
        self.local_dwconv = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            kernel_size=self.local_kernel,
            padding=self.local_kernel // 2,
            groups=self.n_channels,
            bias=True,
        )

        # 全局支路：Conv1d(K=L, P=0)，输出 [B, C, 1]
        self.global_conv = nn.Conv1d(
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            kernel_size=self.window_len,
            padding=0,
            groups=self.n_channels,
            bias=True,
        )

        # 跨通道表决：1x1 Conv1d, [B,C,1] -> [B,1,1]
        self.vote_conv = nn.Conv1d(
            in_channels=self.n_channels, out_channels=1, kernel_size=1, bias=True
        )

    def forward(
        self, windows: torch.Tensor, output_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if windows.ndim != 4:
            raise ValueError(
                f"DualScalePositionScorer expects 4D input [B,K,C,L], got {windows.shape}"
            )
        bsz, n_win, n_ch, win_len = windows.shape
        if n_ch != self.n_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.n_channels}, got {n_ch}"
            )
        if win_len != self.window_len:
            raise ValueError(
                f"Window length mismatch: expected {self.window_len}, got {win_len}"
            )
        if output_len is None:
            output_len = win_len
        output_len = int(output_len)
        if output_len < 1 or output_len > win_len:
            raise ValueError(
                f"output_len must be in [1, {win_len}], got {output_len}"
            )

        # [B,K,C,L] -> [B*K,C,L]
        x = windows.reshape(bsz * n_win, n_ch, win_len)

        # 局部分支保留时序分辨率，支持位置相关权重 w_k[t]
        local_feat = self.local_dwconv(x)  # [B*K,C,L]
        global_feat = self.global_conv(x)  # [B*K,C,1]
        merged_feat = local_feat + global_feat  # 全局特征按时间广播
        logit_map = self.vote_conv(merged_feat).squeeze(1)  # [B*K,L]
        logits = logit_map.view(bsz, n_win, win_len)[:, :, :output_len]  # [B,K,T]
        weights = torch.softmax(logits, dim=1)
        return logits, weights


def fuse_windows_with_weights(
    windows: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    将窗口按 softmax 权重融合。

    windows: [B, K, C, T]
    weights: [B, K, T] 或 [B, K]
    return:  [B, C, T]
    """
    if windows.ndim != 4:
        raise ValueError(
            f"Invalid shapes: windows={windows.shape}, weights={weights.shape}"
        )
    if weights.ndim == 2:
        weights = weights[:, :, None]
    if weights.ndim != 3:
        raise ValueError(
            f"Invalid shapes: windows={windows.shape}, weights={weights.shape}"
        )
    if (
        windows.shape[0] != weights.shape[0]
        or windows.shape[1] != weights.shape[1]
        or windows.shape[3] != weights.shape[2]
    ):
        raise ValueError(
            f"Shape mismatch: windows={windows.shape}, weights={weights.shape}"
        )
    return (weights[:, :, None, :] * windows).sum(dim=1)
