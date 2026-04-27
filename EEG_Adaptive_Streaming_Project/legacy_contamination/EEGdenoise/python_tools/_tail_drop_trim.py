from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TailTrimInfo:
    trimmed: bool
    cut_index: int
    removed_samples: int
    reason: str


def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(int(win), dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def trim_tail_drop_anomaly(
    data_ct: np.ndarray,
    fs: float,
    *,
    enabled: bool = True,
    search_tail_sec: float = 5.0,
    min_persist_sec: float = 0.8,
    rms_ratio_thresh: float = 0.25,
    mean_z_thresh: float = 6.0,
    min_keep_sec: float = 5.0,
) -> tuple[np.ndarray, TailTrimInfo]:
    """
    检测并截掉尾部“突降异常”。

    输入:
      data_ct: (C, T)
      fs: 采样率

    规则:
      - 只在最后 search_tail_sec 秒内搜索，避免误伤中段信号；
      - 同时参考能量突降(rms)与均值突降(mean)；
      - 需在 min_persist_sec 内持续满足低值条件才截断。
    """
    x = np.asarray(data_ct, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (C, T), got {x.shape}")
    c, t = x.shape
    if c <= 0 or t <= 0:
        return x, TailTrimInfo(False, t, 0, "empty")
    if not enabled:
        return x, TailTrimInfo(False, t, 0, "disabled")

    fs = float(max(1.0, fs))
    min_keep = int(round(max(1.0, min_keep_sec) * fs))
    if t <= (min_keep + int(round(1.0 * fs))):
        return x, TailTrimInfo(False, t, 0, "too_short")

    tail_len = int(round(max(1.0, search_tail_sec) * fs))
    search_start = max(min_keep, t - tail_len)
    if search_start <= 0:
        return x, TailTrimInfo(False, t, 0, "invalid_search_range")

    # 统计量使用“跨通道均值/能量”
    rms_t = np.sqrt(np.mean(np.square(x), axis=0) + 1e-12)
    mean_t = np.mean(x, axis=0)

    smooth_win = max(3, int(round(0.1 * fs)))
    rms_s = _smooth_1d(rms_t, smooth_win)
    mean_s = _smooth_1d(mean_t, smooth_win)

    base_rms = rms_s[:search_start]
    base_mean = mean_s[:search_start]
    if base_rms.size < 8 or base_mean.size < 8:
        return x, TailTrimInfo(False, t, 0, "baseline_too_short")

    base_rms_med = float(np.median(base_rms))
    base_mean_med = float(np.median(base_mean))
    base_mean_mad = float(np.median(np.abs(base_mean - base_mean_med))) + 1e-12
    base_mean_sigma = 1.4826 * base_mean_mad

    low_rms = rms_s < (base_rms_med * float(max(1e-6, rms_ratio_thresh)))
    low_mean = mean_s < (base_mean_med - float(mean_z_thresh) * base_mean_sigma)
    cond = low_rms | low_mean

    persist = max(1, int(round(max(0.1, min_persist_sec) * fs)))
    need_true = max(1, int(round(0.8 * persist)))

    onset = None
    for i in range(search_start, t - persist + 1):
        if int(np.sum(cond[i : i + persist])) >= need_true:
            onset = i
            break

    if onset is None:
        return x, TailTrimInfo(False, t, 0, "no_tail_drop")

    removed = int(t - onset)
    # 去掉太短的误检
    if removed < int(round(0.3 * fs)):
        return x, TailTrimInfo(False, t, 0, "drop_too_short")

    if onset <= min_keep:
        return x, TailTrimInfo(False, t, 0, "keep_too_short")

    return x[:, :onset], TailTrimInfo(True, int(onset), removed, "tail_drop_detected")

