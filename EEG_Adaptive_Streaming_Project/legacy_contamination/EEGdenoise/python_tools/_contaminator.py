import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

try:
    # Preferred when running as module:
    # python -m legacy_contamination.EEGdenoise.python_tools._contaminator
    from .EEGdenoiseNet_data_prepare import prepare_data
except Exception:
    # Fallback when running as script inside python_tools/
    from EEGdenoiseNet_data_prepare import prepare_data

try:
    from ._tail_drop_trim import trim_tail_drop_anomaly
except Exception:
    from _tail_drop_trim import trim_tail_drop_anomaly

META_KEYS = {"__header__", "__version__", "__globals__"}


def _ensure_2d(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (N, T), got shape={arr.shape}")
    return arr.astype(np.float32)


def _fit_noise_length(noise_vec: np.ndarray, target_len: int) -> np.ndarray:
    if noise_vec.shape[0] == target_len:
        return noise_vec
    if noise_vec.shape[0] > target_len:
        return noise_vec[:target_len]
    repeat = (target_len + noise_vec.shape[0] - 1) // noise_vec.shape[0]
    return np.tile(noise_vec, repeat)[:target_len]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _scale_noise_to_snr(
    signal: np.ndarray, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    snr_linear = 10.0 ** (snr_db / 20.0)
    return noise * (_rms(signal) / (_rms(noise) * snr_linear + 1e-12))


def _load_mat_array(mat_path: str, key: Optional[str]) -> np.ndarray:
    mat = sio.loadmat(mat_path)
    if key:
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {mat_path}")
        return _ensure_2d(key, mat[key])

    candidates = [
        (k, v)
        for k, v in mat.items()
        if k not in META_KEYS and isinstance(v, np.ndarray)
    ]
    for k, v in candidates:
        if v.ndim in (1, 2):
            return _ensure_2d(k, v)
    raise ValueError(f"Cannot infer 2D array from {mat_path}. Please pass --clean-key")


def _load_edf_array(
    edf_path: str,
    channels: int = 19,
    target_fs: int = 200,
    tail_drop_trim: bool = True,
    tail_drop_search_sec: float = 5.0,
    tail_drop_min_persist_sec: float = 0.8,
    tail_drop_rms_ratio: float = 0.25,
    tail_drop_mean_z: float = 6.0,
) -> np.ndarray:
    """从 EDF 读取并重采样，返回 (C, T)。"""
    try:
        import mne
    except ImportError as e:
        raise ImportError("mne is required for --clean-edf") from e

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    if target_fs > 0:
        raw.resample(target_fs)
    use_ch = min(channels, len(raw.ch_names))
    raw.pick(raw.ch_names[:use_ch])
    arr = raw.get_data().astype(np.float32)
    arr, trim_info = trim_tail_drop_anomaly(
        arr,
        fs=float(raw.info["sfreq"]),
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
    return arr


def _to_epochs_2d(clean_ct: np.ndarray, epoch_len: int = 500) -> np.ndarray:
    """(C, T) -> (N, epoch_len)，N 为按通道展开后的 epoch 数。"""
    if clean_ct.ndim != 2:
        raise ValueError(f"Expected (C, T), got {clean_ct.shape}")
    t = (clean_ct.shape[1] // epoch_len) * epoch_len
    if t < epoch_len:
        raise ValueError("Signal too short for one epoch")
    x = clean_ct[:, :t]
    return (
        x.reshape(x.shape[0], -1, epoch_len).transpose(1, 0, 2).reshape(-1, epoch_len)
    )


def _split_indices(n: int, train_per: float) -> Tuple[slice, slice, slice]:
    n_train = int(round(n * train_per))
    n_val = int(round((n - n_train) / 2.0))
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def _save_epoch_outputs(
    prefix: str, out_dir: str, data: Tuple[np.ndarray, ...]
) -> None:
    (
        train_noisy,
        train_clean,
        val_noisy,
        val_clean,
        test_noisy,
        test_clean,
        test_std,
    ) = data

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"train_{prefix}_input.npy"), train_noisy)
    np.save(os.path.join(out_dir, f"train_{prefix}_label.npy"), train_clean)
    np.save(os.path.join(out_dir, f"val_{prefix}_input.npy"), val_noisy)
    np.save(os.path.join(out_dir, f"val_{prefix}_label.npy"), val_clean)
    np.save(os.path.join(out_dir, f"test_{prefix}_input.npy"), test_noisy)
    np.save(os.path.join(out_dir, f"test_{prefix}_label.npy"), test_clean)
    np.save(os.path.join(out_dir, f"test_{prefix}_std.npy"), test_std)


def _prepare_hybrid_epoch_data(
    eeg_all: np.ndarray,
    noise_emg: np.ndarray,
    noise_eog: np.ndarray,
    combin_num: int,
    train_per: float,
    snr_db_min: float,
    snr_db_max: float,
    emg_weight: float,
    seed: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    eeg_all = _ensure_2d("EEG_all", eeg_all)
    noise_emg = _ensure_2d("noise_emg", noise_emg)
    noise_eog = _ensure_2d("noise_eog", noise_eog)

    n = min(eeg_all.shape[0], noise_emg.shape[0], noise_eog.shape[0])
    eeg = eeg_all[:n]
    emg = noise_emg[:n]
    eog = noise_eog[:n]
    t = eeg.shape[1]

    idx = rng.permutation(n)
    eeg = eeg[idx]
    emg = emg[idx]
    eog = eog[idx]

    tr, va, te = _split_indices(n, train_per)
    eeg_tr, eeg_va, eeg_te = eeg[tr], eeg[va], eeg[te]
    emg_tr, emg_va, emg_te = emg[tr], emg[va], emg[te]
    eog_tr, eog_va, eog_te = eog[tr], eog[va], eog[te]

    def build_train() -> Tuple[np.ndarray, np.ndarray]:
        out_x, out_y = [], []
        for _ in range(combin_num):
            perm_emg = rng.permutation(eeg_tr.shape[0])
            perm_eog = rng.permutation(eeg_tr.shape[0])
            for i in range(eeg_tr.shape[0]):
                clean = eeg_tr[i]
                n_emg = emg_tr[perm_emg[i]]
                n_eog = eog_tr[perm_eog[i]]
                mixed_noise = emg_weight * n_emg + (1.0 - emg_weight) * n_eog
                snr_db = float(rng.uniform(snr_db_min, snr_db_max))
                mixed_noise = _scale_noise_to_snr(clean, mixed_noise, snr_db)
                noisy = clean + mixed_noise
                std = np.std(noisy) + 1e-12
                out_x.append((noisy / std).astype(np.float32))
                out_y.append((clean / std).astype(np.float32))
        return np.asarray(out_x), np.asarray(out_y)

    def build_eval(
        clean_arr: np.ndarray, emg_arr: np.ndarray, eog_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        snr_grid = np.linspace(snr_db_min, snr_db_max, num=10)
        out_x, out_y, out_std = [], [], []
        for snr_db in snr_grid:
            for i in range(clean_arr.shape[0]):
                clean = clean_arr[i]
                mixed_noise = emg_weight * emg_arr[i] + (1.0 - emg_weight) * eog_arr[i]
                mixed_noise = _scale_noise_to_snr(clean, mixed_noise, float(snr_db))
                noisy = clean + mixed_noise
                std = np.std(noisy) + 1e-12
                out_x.append((noisy / std).astype(np.float32))
                out_y.append((clean / std).astype(np.float32))
                out_std.append(std)
        return (
            np.asarray(out_x),
            np.asarray(out_y),
            np.asarray(out_std, dtype=np.float32),
        )

    x_tr, y_tr = build_train()
    x_va, y_va, _ = build_eval(eeg_va, emg_va, eog_va)
    x_te, y_te, std_te = build_eval(eeg_te, emg_te, eog_te)
    if x_tr.shape[1] != t:
        raise RuntimeError("Unexpected hybrid output shape")
    return x_tr, y_tr, x_va, y_va, x_te, y_te, std_te


def _load_clean_sim_dict(clean_mat_path: str) -> List[Tuple[int, np.ndarray]]:
    clean_mat = sio.loadmat(clean_mat_path)
    sim_entries: List[Tuple[int, np.ndarray]] = []
    for key, value in clean_mat.items():
        if key.startswith("sim") and key.endswith("_resampled"):
            idx = int(key[3:].split("_")[0])
            sim_entries.append((idx, np.asarray(value, dtype=np.float32)))
    if not sim_entries:
        raise ValueError("No keys like sim1_resampled found in clean mat")
    sim_entries.sort(key=lambda x: x[0])
    return sim_entries


def _load_clean_sim_from_edf(
    edf_path: str,
    channels: int,
    target_fs: int,
    tail_drop_trim: bool,
    tail_drop_search_sec: float,
    tail_drop_min_persist_sec: float,
    tail_drop_rms_ratio: float,
    tail_drop_mean_z: float,
) -> List[Tuple[int, np.ndarray]]:
    clean = _load_edf_array(
        edf_path,
        channels=channels,
        target_fs=target_fs,
        tail_drop_trim=tail_drop_trim,
        tail_drop_search_sec=tail_drop_search_sec,
        tail_drop_min_persist_sec=tail_drop_min_persist_sec,
        tail_drop_rms_ratio=tail_drop_rms_ratio,
        tail_drop_mean_z=tail_drop_mean_z,
    )
    return [(1, clean)]


def _make_contaminated_multichannel(
    clean_sig: np.ndarray,
    noise_emg: np.ndarray,
    noise_eog: np.ndarray,
    mode: str,
    snr_db_min: float,
    snr_db_max: float,
    emg_prob: float,
    eog_prob: float,
    hybrid_emg_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if clean_sig.ndim != 2:
        raise ValueError(f"clean signal must be (C, T), got {clean_sig.shape}")

    out = np.zeros_like(clean_sig, dtype=np.float32)
    t = clean_sig.shape[1]
    mode = mode.lower()

    for c in range(clean_sig.shape[0]):
        clean = clean_sig[c].astype(np.float32)
        n_emg = _fit_noise_length(noise_emg[rng.integers(0, noise_emg.shape[0])], t)
        n_eog = _fit_noise_length(noise_eog[rng.integers(0, noise_eog.shape[0])], t)

        if mode == "emg":
            noise = n_emg
        elif mode == "eog":
            noise = n_eog
        elif mode == "hybrid":
            noise = hybrid_emg_weight * n_emg + (1.0 - hybrid_emg_weight) * n_eog
        elif mode == "mixed":
            p = rng.random()
            if p < emg_prob:
                noise = n_emg
            elif p < emg_prob + eog_prob:
                noise = n_eog
            else:
                noise = hybrid_emg_weight * n_emg + (1.0 - hybrid_emg_weight) * n_eog
        else:
            raise ValueError(f"Unsupported contamination mode: {mode}")

        snr_db = float(rng.uniform(snr_db_min, snr_db_max))
        noise = _scale_noise_to_snr(clean, noise, snr_db)
        out[c] = clean + noise

    return out


def run_epoch_mode(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    if args.clean_npy:
        eeg_all = _ensure_2d("clean_npy", np.load(args.clean_npy))
    elif args.clean_edf:
        clean_ct = _load_edf_array(
            args.clean_edf,
            channels=args.edf_channels,
            target_fs=args.edf_target_fs,
            tail_drop_trim=not args.disable_tail_drop_trim,
            tail_drop_search_sec=args.tail_drop_search_sec,
            tail_drop_min_persist_sec=args.tail_drop_min_persist_sec,
            tail_drop_rms_ratio=args.tail_drop_rms_ratio,
            tail_drop_mean_z=args.tail_drop_mean_z,
        )
        eeg_all = _ensure_2d(
            "clean_edf_epochs", _to_epochs_2d(clean_ct, epoch_len=args.epoch_len)
        )
    elif args.clean_mat:
        eeg_all = _load_mat_array(args.clean_mat, args.clean_key)
    else:
        raise ValueError("Either --clean-npy or --clean-mat must be provided")

    noise_eog = _ensure_2d("eog_npy", np.load(args.eog_npy))
    if args.emg_npy and os.path.exists(args.emg_npy):
        noise_emg = _ensure_2d("emg_npy", np.load(args.emg_npy))
    elif args.allow_eog_as_emg:
        print("[warn] emg_npy not found, fallback to eog_npy as EMG source")
        noise_emg = noise_eog
    else:
        raise FileNotFoundError(f"EMG file not found: {args.emg_npy}")
    os.makedirs(args.out_dir, exist_ok=True)

    targets: List[str]
    if args.combo == "all":
        targets = ["eog", "emg", "hybrid"]
    else:
        targets = [args.combo]

    for target in targets:
        if target == "eog":
            out = prepare_data(
                eeg_all, noise_eog, args.combin_num, args.train_per, "EOG"
            )
        elif target == "emg":
            out = prepare_data(
                eeg_all, noise_emg, args.combin_num, args.train_per, "EMG"
            )
        elif target == "hybrid":
            out = _prepare_hybrid_epoch_data(
                eeg_all=eeg_all,
                noise_emg=noise_emg,
                noise_eog=noise_eog,
                combin_num=args.combin_num,
                train_per=args.train_per,
                snr_db_min=args.snr_min,
                snr_db_max=args.snr_max,
                emg_weight=args.hybrid_emg_weight,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unsupported combo={target}")

        _save_epoch_outputs(target, args.out_dir, out)
        print(f"Saved epoch pair files for: {target} -> {args.out_dir}")


def run_stfnet_mode(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    noise_eog = _ensure_2d("eog_npy", np.load(args.eog_npy))
    if args.emg_npy and os.path.exists(args.emg_npy):
        noise_emg = _ensure_2d("emg_npy", np.load(args.emg_npy))
    elif args.allow_eog_as_emg:
        print("[warn] emg_npy not found, fallback to eog_npy as EMG source")
        noise_emg = noise_eog
    else:
        raise FileNotFoundError(f"EMG file not found: {args.emg_npy}")

    if args.clean_edf:
        sim_entries = _load_clean_sim_from_edf(
            args.clean_edf,
            channels=args.edf_channels,
            target_fs=args.edf_target_fs,
            tail_drop_trim=not args.disable_tail_drop_trim,
            tail_drop_search_sec=args.tail_drop_search_sec,
            tail_drop_min_persist_sec=args.tail_drop_min_persist_sec,
            tail_drop_rms_ratio=args.tail_drop_rms_ratio,
            tail_drop_mean_z=args.tail_drop_mean_z,
        )
    else:
        sim_entries = _load_clean_sim_dict(args.clean_mat)

    out_contaminated: Dict[str, np.ndarray] = {}
    clean_stack, noisy_stack = [], []

    for idx, clean_sig in sim_entries:
        con = _make_contaminated_multichannel(
            clean_sig=clean_sig,
            noise_emg=noise_emg,
            noise_eog=noise_eog,
            mode=args.combo,
            snr_db_min=args.snr_min,
            snr_db_max=args.snr_max,
            emg_prob=args.mixed_emg_prob,
            eog_prob=args.mixed_eog_prob,
            hybrid_emg_weight=args.hybrid_emg_weight,
            rng=rng,
        )
        out_contaminated[f"sim{idx}_con"] = con
        clean_stack.append(clean_sig.astype(np.float32))
        noisy_stack.append(con.astype(np.float32))

    clean_arr = np.asarray(clean_stack, dtype=np.float32)
    noisy_arr = np.asarray(noisy_stack, dtype=np.float32)

    if args.out_contaminated_mat:
        os.makedirs(os.path.dirname(args.out_contaminated_mat) or ".", exist_ok=True)
        sio.savemat(args.out_contaminated_mat, out_contaminated)
        print(f"Saved contaminated mat: {args.out_contaminated_mat}")

    if args.out_pure_mat:
        out_pure = {f"sim{idx}_resampled": sig for idx, sig in sim_entries}
        sio.savemat(args.out_pure_mat, out_pure)
        print(f"Saved pure mat copy: {args.out_pure_mat}")

    if args.out_contaminated_npy:
        os.makedirs(os.path.dirname(args.out_contaminated_npy) or ".", exist_ok=True)
        np.save(args.out_contaminated_npy, noisy_arr)
        print(
            f"Saved contaminated npy: {args.out_contaminated_npy}, shape={noisy_arr.shape}"
        )

    if args.out_clean_npy:
        os.makedirs(os.path.dirname(args.out_clean_npy) or ".", exist_ok=True)
        np.save(args.out_clean_npy, clean_arr)
        print(f"Saved clean npy: {args.out_clean_npy}, shape={clean_arr.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate contaminated EEG pairs for STFNet"
    )

    parser.add_argument("--mode", choices=["epoch", "stfnet"], default="epoch")
    parser.add_argument(
        "--combo", choices=["eog", "emg", "hybrid", "mixed", "all"], default="all"
    )

    parser.add_argument("--clean-npy", type=str, default="")
    parser.add_argument("--clean-edf", type=str, default="")
    parser.add_argument("--clean-mat", type=str, default="data/Pure_Data.mat")
    parser.add_argument("--clean-key", type=str, default="")
    parser.add_argument(
        "--emg-npy", type=str, default="data/数据处理_手动污染/emg_all.npy"
    )
    parser.add_argument(
        "--eog-npy", type=str, default="data/数据处理_手动污染/eog_all.npy"
    )
    parser.add_argument("--allow-eog-as-emg", action="store_true")
    parser.add_argument("--out-dir", type=str, default="data/generated_pairs")

    parser.add_argument("--combin-num", type=int, default=3)
    parser.add_argument("--train-per", type=float, default=0.6)
    parser.add_argument("--snr-min", type=float, default=-7.0)
    parser.add_argument("--snr-max", type=float, default=2.0)
    parser.add_argument("--hybrid-emg-weight", type=float, default=0.5)
    parser.add_argument("--mixed-emg-prob", type=float, default=0.4)
    parser.add_argument("--mixed-eog-prob", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edf-channels", type=int, default=19)
    parser.add_argument("--edf-target-fs", type=int, default=200)
    parser.add_argument(
        "--disable-tail-drop-trim",
        action="store_true",
        help="Disable auto tail-drop anomaly trimming for EDF input.",
    )
    parser.add_argument(
        "--tail-drop-search-sec",
        type=float,
        default=5.0,
        help="Only search drop in last N seconds.",
    )
    parser.add_argument(
        "--tail-drop-min-persist-sec",
        type=float,
        default=0.8,
        help="Drop condition must persist at least this duration.",
    )
    parser.add_argument(
        "--tail-drop-rms-ratio",
        type=float,
        default=0.25,
        help="RMS drop ratio threshold vs baseline median.",
    )
    parser.add_argument(
        "--tail-drop-mean-z",
        type=float,
        default=6.0,
        help="Mean drop threshold in robust z-score.",
    )
    parser.add_argument("--epoch-len", type=int, default=500)

    parser.add_argument("--out-contaminated-mat", type=str, default="")
    parser.add_argument("--out-pure-mat", type=str, default="")
    parser.add_argument(
        "--out-contaminated-npy", type=str, default="data/Contaminated_Data_Custom.npy"
    )
    parser.add_argument(
        "--out-clean-npy", type=str, default="data/Pure_Data_Custom.npy"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "epoch":
        if args.combo == "mixed":
            raise ValueError("--combo mixed is only supported in --mode stfnet")
        run_epoch_mode(args)
    else:
        if args.combo == "all":
            raise ValueError("--combo all is only supported in --mode epoch")
        run_stfnet_mode(args)


if __name__ == "__main__":
    main()
