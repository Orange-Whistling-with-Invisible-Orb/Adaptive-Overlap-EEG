from __future__ import annotations

import torch


def reconstruction_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """基础重构损失：MSE。"""
    return torch.mean((pred - target) ** 2)


def reconstruction_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """基础重构损失：L1。"""
    return torch.mean(torch.abs(pred - target))


def combined_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
) -> torch.Tensor:
    """
    混合重构损失。
    默认只使用 MSE（l1_weight=0），可按需加入 L1 稳定训练。
    """
    loss = mse_weight * reconstruction_mse(pred, target)
    if l1_weight > 0:
        loss = loss + l1_weight * reconstruction_l1(pred, target)
    return loss
