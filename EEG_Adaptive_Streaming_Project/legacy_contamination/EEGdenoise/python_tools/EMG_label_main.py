"""
EMG_label_main.py

对应 MATLAB 脚本: EMG_label_main.m
用途: 从原始 EMG 数据中切分伪迹片段，做基础质量筛选，并导出 .mat/.npy。

说明:
- MATLAB 里依赖 EMG_cut / visual_check，本脚本内置了等价的 Python 版本。
- 默认按 1 秒窗口切段 (window_sec=1.0)。
- 也支持直接从 EDF 生成 EMG proxy（高频带通后的 EEG 片段），用于纯 Python 流程。
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt, resample_poly

try:
    from ._tail_drop_trim import trim_tail_drop_anomaly
except Exception:
    from _tail_drop_trim import trim_tail_drop_anomaly


def to_microvolt(signal: np.ndarray, unit: str) -> np.ndarray:
    """把原始单位统一转换到 uV，便于后续阈值筛选。"""
    u = unit.lower()
    if u == "uv":
        return signal.astype(np.float32)
    if u == "mv":
        return (signal * 1000.0).astype(np.float32)
    if u == "v":
        return (signal * 1_000_000.0).astype(np.float32)
    raise ValueError(f"Unsupported unit: {unit}")


def emg_cut(
    signal_1d: np.ndarray, raw_fs: int, target_fs: int, raw_unit: str, window_sec: float
) -> np.ndarray:
    """
    等价 MATLAB 的 EMG_cut: 重采样 + 定长切段。
    输出 shape: (N, T)
    """
    sig = to_microvolt(np.asarray(signal_1d).reshape(-1), raw_unit)
    sig_rs = resample_poly(sig, up=target_fs, down=raw_fs).astype(np.float32)
    win = int(round(window_sec * target_fs))
    if sig_rs.size < win:
        return np.empty((0, win), dtype=np.float32)

    out = []
    for s in range(0, sig_rs.size - win + 1, win):
        out.append(sig_rs[s : s + win])
    return np.asarray(out, dtype=np.float32)


def visual_check(epochs: np.ndarray, z_th: float = 6.0) -> np.ndarray:
    """
    对应 MATLAB 的 visual_check（自动版）：
    用 MAD 去掉振幅异常段，避免极端伪迹污染分布。
    """
    if epochs.size == 0:
        return epochs
    amp = np.max(np.abs(epochs), axis=1)
    med = np.median(amp)
    mad = np.median(np.abs(amp - med)) + 1e-8
    z = np.abs(amp - med) / (1.4826 * mad)
    return epochs[z <= z_th]


def extract_d_eraw(mat_path: str) -> np.ndarray:
    """
    从 dataXXA.mat/dataXXB.mat 中读取 D.Eraw。
    若数据组织不同，请按你本地 .mat 结构改这里。
    """
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "D" not in d:
        raise KeyError(f"{mat_path} does not contain key 'D'")

    obj = d["D"]
    if hasattr(obj, "Eraw"):
        eraw = obj.Eraw
    elif isinstance(obj, np.ndarray) and obj.size > 0 and hasattr(obj.item(), "Eraw"):
        eraw = obj.item().Eraw
    else:
        raise KeyError(f"Cannot parse D.Eraw from {mat_path}")

    arr = np.asarray(eraw, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D Eraw, got shape {arr.shape}")
    return arr


def build_emg_epochs(
    data_path: str,
    subjects: List[int],
    conditions: List[str],
    raw_fs: int,
    target_fs: int,
    raw_unit: str,
    window_sec: float,
) -> np.ndarray:
    """遍历被试和条件，逐通道切段并合并。"""
    epoch_all = []
    for sid in subjects:
        for cond in conditions:
            fid = os.path.join(data_path, f"data{sid:02d}{cond}.mat")
            emgs = extract_d_eraw(fid)  # MATLAB: D.Eraw
            # MATLAB 里 EMGs 形状是 (sample_num, channel_num)
            for ch in range(emgs.shape[1]):
                tmp = emgs[:, ch]
                epochs = emg_cut(
                    tmp,
                    raw_fs=raw_fs,
                    target_fs=target_fs,
                    raw_unit=raw_unit,
                    window_sec=window_sec,
                )
                if epochs.size > 0:
                    epoch_all.append(epochs)

    if not epoch_all:
        raise RuntimeError("No EMG epochs extracted. Check input path and file format.")

    emg_epochs = np.concatenate(epoch_all, axis=0).astype(np.float32)
    emg_epochs = visual_check(emg_epochs)
    return emg_epochs


def build_emg_proxy_from_edf(
    edf_path: str,
    target_fs: int,
    window_sec: float,
    edf_channels: int,
    band_low: float,
    band_high: float,
    tail_drop_trim: bool = True,
    tail_drop_search_sec: float = 5.0,
    tail_drop_min_persist_sec: float = 0.8,
    tail_drop_rms_ratio: float = 0.25,
    tail_drop_mean_z: float = 6.0,
) -> np.ndarray:
    """从 EDF 提取高频成分作为 EMG proxy，输出 shape=(N, T)。"""
    try:
        import mne
    except ImportError as e:
        raise ImportError("mne is required for --edf-path mode") from e

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    if target_fs > 0:
        raw.resample(target_fs)

    use_ch = min(edf_channels, len(raw.ch_names))
    raw.pick(raw.ch_names[:use_ch])

    # mne 返回单位为 V，这里统一转成 uV
    data_uv = raw.get_data().astype(np.float32) * 1_000_000.0

    fs = float(raw.info["sfreq"])
    data_uv, trim_info = trim_tail_drop_anomaly(
        data_uv,
        fs=fs,
        enabled=bool(tail_drop_trim),
        search_tail_sec=float(tail_drop_search_sec),
        min_persist_sec=float(tail_drop_min_persist_sec),
        rms_ratio_thresh=float(tail_drop_rms_ratio),
        mean_z_thresh=float(tail_drop_mean_z),
    )
    if trim_info.trimmed:
        print(
            f"[tail-trim] {os.path.basename(edf_path)}: "
            f"cut@{trim_info.cut_index}, removed={trim_info.removed_samples} samples"
        )

    nyq = fs * 0.5
    hi = min(float(band_high), nyq * 0.98)
    lo = float(band_low)
    if not (0.0 < lo < hi < nyq):
        raise ValueError(
            f"Invalid EMG band [{band_low}, {band_high}] for fs={fs}. "
            f"Expect 0 < low < high < {nyq:.2f}."
        )

    b, a = butter(4, [lo / nyq, hi / nyq], btype="bandpass")

    epoch_all = []
    for ch in range(data_uv.shape[0]):
        sig = filtfilt(b, a, data_uv[ch]).astype(np.float32)
        epochs = emg_cut(
            sig,
            raw_fs=int(round(fs)),
            target_fs=target_fs,
            raw_unit="uV",
            window_sec=window_sec,
        )
        if epochs.size > 0:
            epoch_all.append(epochs)

    if not epoch_all:
        raise RuntimeError("No EMG proxy epochs extracted from EDF.")

    emg_epochs = np.concatenate(epoch_all, axis=0).astype(np.float32)
    emg_epochs = visual_check(emg_epochs)
    return emg_epochs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate EMG epochs (.mat/.npy) from MAT or EDF inputs."
    )
    p.add_argument(
        "--data-path",
        default="",
        help="Folder containing data01A.mat/data01B.mat ...",
    )
    p.add_argument(
        "--edf-path",
        default="",
        help="Input EDF path. If set, EMG proxy will be extracted from EDF in Python.",
    )
    p.add_argument("--subject-start", type=int, default=1)
    p.add_argument("--subject-end", type=int, default=15)
    p.add_argument("--conditions", nargs="+", default=["A", "B"])
    p.add_argument("--raw-fs", type=int, default=2048)
    p.add_argument("--target-fs", type=int, default=512)
    p.add_argument(
        "--raw-unit", default="mV", choices=["uV", "mV", "V", "uv", "mv", "v"]
    )
    p.add_argument("--edf-channels", type=int, default=19)
    p.add_argument("--edf-emg-band-low", type=float, default=20.0)
    p.add_argument("--edf-emg-band-high", type=float, default=95.0)
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument(
        "--disable-tail-drop-trim",
        action="store_true",
        help="Disable auto tail-drop anomaly trimming for EDF input.",
    )
    p.add_argument(
        "--tail-drop-search-sec",
        type=float,
        default=5.0,
        help="Only search drop in last N seconds.",
    )
    p.add_argument(
        "--tail-drop-min-persist-sec",
        type=float,
        default=0.8,
        help="Drop condition must persist at least this duration.",
    )
    p.add_argument(
        "--tail-drop-rms-ratio",
        type=float,
        default=0.25,
        help="RMS drop ratio threshold vs baseline median.",
    )
    p.add_argument(
        "--tail-drop-mean-z",
        type=float,
        default=6.0,
        help="Mean drop threshold in robust z-score.",
    )
    p.add_argument("--output-dir", default=".")
    p.add_argument("--output-prefix", default="EMG_epochs_new")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.edf_path:
        print("[mode] EDF -> EMG proxy")
        emg_epochs = build_emg_proxy_from_edf(
            edf_path=args.edf_path,
            target_fs=args.target_fs,
            window_sec=args.window_sec,
            edf_channels=args.edf_channels,
            band_low=args.edf_emg_band_low,
            band_high=args.edf_emg_band_high,
            tail_drop_trim=not args.disable_tail_drop_trim,
            tail_drop_search_sec=args.tail_drop_search_sec,
            tail_drop_min_persist_sec=args.tail_drop_min_persist_sec,
            tail_drop_rms_ratio=args.tail_drop_rms_ratio,
            tail_drop_mean_z=args.tail_drop_mean_z,
        )
    else:
        if not args.data_path:
            raise ValueError("Either --edf-path or --data-path must be provided")
        print("[mode] MAT(D.Eraw) -> EMG epochs")
        subjects = list(range(args.subject_start, args.subject_end + 1))
        emg_epochs = build_emg_epochs(
            data_path=args.data_path,
            subjects=subjects,
            conditions=args.conditions,
            raw_fs=args.raw_fs,
            target_fs=args.target_fs,
            raw_unit=args.raw_unit,
            window_sec=args.window_sec,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_mat = os.path.join(args.output_dir, f"{args.output_prefix}.mat")
    out_npy = os.path.join(args.output_dir, f"{args.output_prefix}.npy")

    savemat(out_mat, {"EMG_epochs": emg_epochs, "fs": args.target_fs})
    np.save(out_npy, emg_epochs.astype(np.float32))

    print("Saved:")
    print(out_mat)
    print(out_npy)
    print("Shape:", emg_epochs.shape)


if __name__ == "__main__":
    main()
