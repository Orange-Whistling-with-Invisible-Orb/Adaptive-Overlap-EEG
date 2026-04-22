from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    # Keep training args aligned with train_stfnet.py.
    parser = argparse.ArgumentParser(
        description="Pick one dataset from top-length 15% pool and train STFNet"
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

    # Selection-specific args.
    parser.add_argument("--contaminated_dir", type=str, default="data/contaminated")
    parser.add_argument(
        "--combo",
        type=str,
        default="hybrid",
        choices=["eog", "emg", "hybrid", "mixed"],
    )
    parser.add_argument(
        "--top_ratio",
        type=float,
        default=0.15,
        help="Select random one from top ratio by length",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=5000,
        help="Minimum length to keep candidate pair",
    )
    return parser.parse_args()


def _to_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")


def _collect_pair_lengths(
    contaminated_dir: Path, combo: str
) -> list[tuple[int, Path, Path, str]]:
    out: list[tuple[int, Path, Path, str]] = []
    prefix = f"Contaminated_{combo}_"
    for nos_path in sorted(contaminated_dir.glob(f"{prefix}*.npy")):
        sid = nos_path.stem.replace(prefix, "", 1)
        pure_path = contaminated_dir / f"Pure_{sid}.npy"
        if not pure_path.exists():
            continue

        x = _to_3d(np.load(nos_path, mmap_mode="r"))
        y = _to_3d(np.load(pure_path, mmap_mode="r"))
        if x.shape[0] != y.shape[0]:
            continue

        t = min(x.shape[-1], y.shape[-1])
        out.append((int(t), nos_path, pure_path, sid))
    return out


def _pick_one_top15(
    pairs: list[tuple[int, Path, Path, str]], top_ratio: float, min_len: int, seed: int
) -> tuple[Path, Path, str, int]:
    valid = [p for p in pairs if p[0] >= min_len]
    if not valid:
        raise ValueError(
            f"No valid pairs with length >= {min_len}. Please lower --min_len or regenerate data."
        )

    valid.sort(key=lambda z: z[0], reverse=True)
    top_n = max(1, math.ceil(len(valid) * top_ratio))
    top_pool = valid[:top_n]

    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(top_pool)))
    t, nos_path, pure_path, sid = top_pool[idx]

    print(f"Valid pairs: {len(valid)}")
    print(f"Top pool size ({top_ratio:.2%}): {len(top_pool)}")
    print(f"Selected sid: {sid}, length: {t}")
    return nos_path, pure_path, sid, t


def _run_training(
    args: argparse.Namespace, eeg_path: Path, nos_path: Path, sid: str
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "stfnet_module" / "train_stfnet.py"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{sid}"
    save_dir_ts = Path(args.save_dir) / run_name
    log_dir_ts = Path(args.log_dir) / run_name

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

    best_overall = save_dir_ts / "best_overall.pth"
    if not best_overall.exists():
        raise FileNotFoundError(
            f"Training finished but no best_overall.pth produced at: {best_overall}. "
            "Please check training logs for NaN/invalid val_mse."
        )
    print(f"[run] best checkpoint: {best_overall}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    contaminated_dir = (project_root / args.contaminated_dir).resolve()

    pairs = _collect_pair_lengths(contaminated_dir, args.combo)
    if not pairs:
        raise ValueError(
            f"No matched pairs for combo={args.combo} in {contaminated_dir}"
        )

    nos_path, pure_path, sid, _ = _pick_one_top15(
        pairs=pairs,
        top_ratio=args.top_ratio,
        min_len=args.min_len,
        seed=args.seed,
    )

    _run_training(args, pure_path, nos_path, sid)


if __name__ == "__main__":
    main()
