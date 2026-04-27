"""
EOG_label_main.py

对应 MATLAB 脚本: EOG_label_main.m
用途: 从 EEG 中构造 EOG（垂直/水平），切段、筛选并导出 .mat/.npy。

说明:
- MATLAB 中常见构造:
  vEOG = ch24 - (ch23 + ch25) * 0.5
  hEOG = ch23 - ch25
- 这里默认按同样公式，通道索引按 MATLAB 1-based 转成 Python 0-based。
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly

try:
    from ._tail_drop_trim import trim_tail_drop_anomaly
except Exception:
    from _tail_drop_trim import trim_tail_drop_anomaly


def to_microvolt(signal: np.ndarray, unit: str) -> np.ndarray:
    """统一单位到 uV。"""
    u = unit.lower()
    if u == "uv":
        return signal.astype(np.float32)
    if u == "mv":
        return (signal * 1000.0).astype(np.float32)
    if u == "v":
        return (signal * 1_000_000.0).astype(np.float32)
    raise ValueError(f"Unsupported unit: {unit}")


def cut_epochs(
    signal_1d: np.ndarray, raw_fs: int, target_fs: int, raw_unit: str, window_sec: float
) -> np.ndarray:
    """重采样 + 定长切段，输出 (N, T)。"""
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
    """自动去掉振幅极端异常段，模拟 MATLAB visual_check 作用。"""
    if epochs.size == 0:
        return epochs
    amp = np.max(np.abs(epochs), axis=1)
    med = np.median(amp)
    mad = np.median(np.abs(amp - med)) + 1e-8
    z = np.abs(amp - med) / (1.4826 * mad)
    return epochs[z <= z_th]


def _extract_eeg_data(mat_path: str, eeg_key: str) -> np.ndarray:
    """
    从 .mat 中读取 EEG data，目标形状 (channels, time)。
    默认 key='EEG' 时尝试读取结构体中的 data 字段。
    """
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if eeg_key in d:
        obj = d[eeg_key]
        if hasattr(obj, "data"):
            arr = np.asarray(obj.data, dtype=np.float32)
        else:
            arr = np.asarray(obj, dtype=np.float32)
    elif "data" in d:
        arr = np.asarray(d["data"], dtype=np.float32)
    else:
        # 尝试寻找一个像 EEG 的二维数组。
        cands = []
        for k, v in d.items():
            if k.startswith("__"):
                continue
            vv = np.asarray(v)
            if vv.ndim == 2 and min(vv.shape) >= 16:
                cands.append((k, vv))
        if not cands:
            raise KeyError(f"Cannot find EEG array in {mat_path}")
        arr = np.asarray(cands[0][1], dtype=np.float32)

    # 若是 (time, channels) 则转置
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def _extract_eeg_from_edf(
    edf_path: str,
    *,
    tail_drop_trim: bool = True,
    tail_drop_search_sec: float = 5.0,
    tail_drop_min_persist_sec: float = 0.8,
    tail_drop_rms_ratio: float = 0.25,
    tail_drop_mean_z: float = 6.0,
) -> tuple[np.ndarray, int]:
    """从 EDF 读取 EEG，返回 (channels, time) 和原始采样率。"""
    try:
        import mne
    except ImportError as e:
        raise ImportError("mne is required for --edf-path. Please install mne.") from e

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    arr = raw.get_data().astype(np.float32)
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    raw_fs = int(round(float(raw.info["sfreq"])))
    arr, trim_info = trim_tail_drop_anomaly(
        arr,
        fs=float(raw_fs),
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
    return arr, raw_fs


def build_eog_epochs(
    mat_path: str,
    eeg_key: str,
    edf_path: str,
    raw_fs: int,
    target_fs: int,
    raw_unit: str,
    window_sec: float,
    mode: str,
    eog_ch_idx_1based: list[int],
    tail_drop_trim: bool,
    tail_drop_search_sec: float,
    tail_drop_min_persist_sec: float,
    tail_drop_rms_ratio: float,
    tail_drop_mean_z: float,
) -> np.ndarray:
    """
    根据 EEG 通道构造 EOG 信号并切段。
    mode:
    - vertical: 仅 vEOG
    - horizontal: 仅 hEOG
    - both: 两者合并
    """
    if edf_path:
        eeg, fs_from_edf = _extract_eeg_from_edf(
            edf_path,
            tail_drop_trim=tail_drop_trim,
            tail_drop_search_sec=tail_drop_search_sec,
            tail_drop_min_persist_sec=tail_drop_min_persist_sec,
            tail_drop_rms_ratio=tail_drop_rms_ratio,
            tail_drop_mean_z=tail_drop_mean_z,
        )
        if raw_fs <= 0:
            raw_fs = fs_from_edf
    else:
        eeg = _extract_eeg_data(mat_path, eeg_key=eeg_key)

    n_ch = eeg.shape[0]
    if n_ch < 3:
        raise ValueError(f"EEG channels not enough, need >=3, got {n_ch}")

    # 默认按 MATLAB 通道 23/24/25（1-based）构造。
    # 若通道数不足，则自动回退为最后 3 个通道，避免直接报错。
    ch23, ch24, ch25 = [int(i) - 1 for i in eog_ch_idx_1based]
    if min(ch23, ch24, ch25) < 0:
        raise ValueError(f"Invalid --eog-ch-idx: {eog_ch_idx_1based}")

    if max(ch23, ch24, ch25) >= n_ch:
        ch23, ch24, ch25 = n_ch - 3, n_ch - 2, n_ch - 1
        print(
            "[warn] Requested EOG channels out of range. "
            f"Fallback to last three channels (1-based): {ch23 + 1}, {ch24 + 1}, {ch25 + 1}"
        )

    v_eog = eeg[ch24, :] - 0.5 * (eeg[ch23, :] + eeg[ch25, :])
    h_eog = eeg[ch23, :] - eeg[ch25, :]

    epochs = []
    if mode in ("vertical", "both"):
        epochs.append(cut_epochs(v_eog, raw_fs, target_fs, raw_unit, window_sec))
    if mode in ("horizontal", "both"):
        epochs.append(cut_epochs(h_eog, raw_fs, target_fs, raw_unit, window_sec))

    epochs = [e for e in epochs if e.size > 0]
    if not epochs:
        raise RuntimeError("No EOG epochs extracted.")

    eog_epochs = np.concatenate(epochs, axis=0).astype(np.float32)
    eog_epochs = visual_check(eog_epochs)
    return eog_epochs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate EOG epochs from EEG (.edf/.mat), export .npy (and optional .mat)"
    )
    p.add_argument("--mat-path", default="", help="Input .mat containing EEG data")
    p.add_argument("--edf-path", default="", help="Input .edf file (recommended)")
    p.add_argument(
        "--eeg-key", default="EEG", help="Top-level key for EEG struct/array"
    )
    p.add_argument(
        "--raw-fs",
        type=int,
        default=0,
        help="Raw sampling rate. If --edf-path is used and this is 0, auto-read from EDF.",
    )
    p.add_argument("--target-fs", type=int, default=500)
    p.add_argument(
        "--raw-unit", default="uV", choices=["uV", "mV", "V", "uv", "mv", "v"]
    )
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
    p.add_argument(
        "--mode", default="vertical", choices=["vertical", "horizontal", "both"]
    )
    p.add_argument(
        "--eog-ch-idx",
        type=int,
        nargs=3,
        default=[23, 24, 25],
        help="Three 1-based EEG channel indices used as ch23/ch24/ch25 in EOG formula",
    )
    p.add_argument("--output-dir", default=".")
    p.add_argument("--output-prefix", default="EOG_epochs_new")
    p.add_argument("--save-mat", action="store_true", help="Also save .mat output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (not args.mat_path) and (not args.edf_path):
        raise ValueError("Please provide one of --edf-path or --mat-path")

    eog_epochs = build_eog_epochs(
        mat_path=args.mat_path,
        eeg_key=args.eeg_key,
        edf_path=args.edf_path,
        raw_fs=args.raw_fs,
        target_fs=args.target_fs,
        raw_unit=args.raw_unit,
        window_sec=args.window_sec,
        mode=args.mode,
        eog_ch_idx_1based=args.eog_ch_idx,
        tail_drop_trim=not args.disable_tail_drop_trim,
        tail_drop_search_sec=args.tail_drop_search_sec,
        tail_drop_min_persist_sec=args.tail_drop_min_persist_sec,
        tail_drop_rms_ratio=args.tail_drop_rms_ratio,
        tail_drop_mean_z=args.tail_drop_mean_z,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_npy = os.path.join(args.output_dir, f"{args.output_prefix}.npy")
    np.save(out_npy, eog_epochs.astype(np.float32))

    print("Saved:")
    print(out_npy)
    if args.save_mat:
        out_mat = os.path.join(args.output_dir, f"{args.output_prefix}.mat")
        savemat(out_mat, {"EOG_epochs": eog_epochs, "fs": args.target_fs})
        print(out_mat)

    print("Shape:", eog_epochs.shape)


if __name__ == "__main__":
    main()
