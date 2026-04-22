from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    # Keep training args aligned with train_stfnet.py.
    parser = argparse.ArgumentParser(
        description="Randomly merge 10 datasets then train STFNet"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="stfnet_module/checkpoints")
    parser.add_argument(
        "--log_dir", type=str, default="stfnet_module/checkpoints/json_file"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=10)

    # Merge-specific args.
    parser.add_argument("--contaminated_dir", type=str, default="data/contaminated")
    parser.add_argument(
        "--combo", type=str, default="hybrid", choices=["eog", "emg", "hybrid", "mixed"]
    )
    parser.add_argument("--merge_count", type=int, default=10)
    parser.add_argument(
        "--min_len",
        type=int,
        default=5000,
        help="Minimum time length to keep a pair before random sampling",
    )
    return parser.parse_args()


def _to_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")


def _collect_pairs(contaminated_dir: Path, combo: str) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    prefix = f"Contaminated_{combo}_"
    for nos_path in sorted(contaminated_dir.glob(f"{prefix}*.npy")):
        sid = nos_path.stem.replace(prefix, "", 1)
        pure_path = contaminated_dir / f"Pure_{sid}.npy"
        if pure_path.exists():
            pairs.append((nos_path, pure_path, sid))
    return pairs


def _filter_pairs_by_min_len(
    pairs: list[tuple[Path, Path, str]], min_len: int
) -> list[tuple[Path, Path, str]]:
    kept: list[tuple[Path, Path, str]] = []
    dropped: list[tuple[str, int]] = []
    for nos_path, pure_path, sid in pairs:
        x = _to_3d(np.load(nos_path, mmap_mode="r"))
        y = _to_3d(np.load(pure_path, mmap_mode="r"))
        t = min(x.shape[-1], y.shape[-1])
        if t >= min_len:
            kept.append((nos_path, pure_path, sid))
        else:
            dropped.append((sid, t))

    print(f"Pairs kept (len>={min_len}): {len(kept)}")
    if dropped:
        print(f"Pairs dropped (len<{min_len}): {len(dropped)}")
    return kept


def _merge_selected(
    selected: list[tuple[Path, Path, str]], merged_dir: Path, combo: str, timestamp: str
) -> tuple[Path, Path]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    min_t = None
    min_c = None
    for nos_path, pure_path, _ in selected:
        x = _to_3d(np.load(nos_path))
        y = _to_3d(np.load(pure_path))
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Subject count mismatch: {nos_path.name} vs {pure_path.name}"
            )
        t = min(x.shape[-1], y.shape[-1])
        c = min(x.shape[1], y.shape[1])
        min_t = t if min_t is None else min(min_t, t)
        min_c = c if min_c is None else min(min_c, c)
        xs.append(x)
        ys.append(y)

    x_aligned = [x[:, :min_c, :min_t] for x in xs]
    y_aligned = [y[:, :min_c, :min_t] for y in ys]

    x_merged = np.concatenate(x_aligned, axis=0).astype(np.float32)
    y_merged = np.concatenate(y_aligned, axis=0).astype(np.float32)

    merged_dir.mkdir(parents=True, exist_ok=True)
    nos_merged = merged_dir / f"Contaminated_{combo}_merged_{timestamp}.npy"
    pure_merged = merged_dir / f"Pure_merged_{timestamp}.npy"
    np.save(nos_merged, x_merged)
    np.save(pure_merged, y_merged)

    print(f"Merged NOS: {nos_merged} shape={x_merged.shape}")
    print(f"Merged EEG: {pure_merged} shape={y_merged.shape}")
    return pure_merged, nos_merged


def _run_training(
    args: argparse.Namespace, eeg_path: Path, nos_path: Path, timestamp: str
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "stfnet_module" / "train_stfnet.py"

    save_dir_ts = Path(args.save_dir) / f"run_{timestamp}"
    log_dir_ts = Path(args.log_dir) / f"run_{timestamp}"

    cmd = [
        sys.executable,
        str(train_script),
        "--device",
        args.device,
        "--EEG_path",
        str(eeg_path),
        "--NOS_path",
        str(nos_path),
        "--save_dir",
        str(save_dir_ts),
        "--log_dir",
        str(log_dir_ts),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--folds",
        str(args.folds),
        "--seed",
        str(args.seed),
        "--depth",
        str(args.depth),
    ]

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    contaminated_dir = (project_root / args.contaminated_dir).resolve()

    pairs = _collect_pairs(contaminated_dir, args.combo)
    pairs = _filter_pairs_by_min_len(pairs, args.min_len)
    if len(pairs) < args.merge_count:
        raise ValueError(
            f"Not enough valid pairs for combo={args.combo}. found={len(pairs)}, required={args.merge_count}, min_len={args.min_len}"
        )

    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.choice(len(pairs), size=args.merge_count, replace=False)
    selected = [pairs[int(i)] for i in chosen_idx]

    print(f"Selected {len(selected)} pairs:")
    for _, _, sid in selected:
        print("-", sid)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_dir = contaminated_dir / "merged"
    eeg_merged, nos_merged = _merge_selected(
        selected, merged_dir, args.combo, timestamp
    )

    _run_training(args, eeg_merged, nos_merged, timestamp)


if __name__ == "__main__":
    main()
