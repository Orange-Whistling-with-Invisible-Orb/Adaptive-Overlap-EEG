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
        description="Randomly pick multiple datasets from top-length pool and train STFNet"
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
    parser.add_argument("--folds", type=int, default=1)

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
        help="Candidate pool ratio by length (descending)",
    )
    parser.add_argument(
        "--pick_count",
        type=int,
        default=5,
        help="Randomly pick this many datasets from top pool",
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


def _pick_topk_from_pool(
    pairs: list[tuple[int, Path, Path, str]],
    top_ratio: float,
    min_len: int,
    seed: int,
    pick_count: int,
) -> list[tuple[Path, Path, str, int]]:
    valid = [p for p in pairs if p[0] >= min_len]
    if not valid:
        raise ValueError(
            f"No valid pairs with length >= {min_len}. Please lower --min_len or regenerate data."
        )

    valid.sort(key=lambda z: z[0], reverse=True)
    top_n = max(1, math.ceil(len(valid) * top_ratio))
    top_pool = valid[:top_n]
    if len(top_pool) < 2:
        raise ValueError(
            f"Top pool too small for random multi-dataset training. top_pool={len(top_pool)}, "
            f"valid={len(valid)}, top_ratio={top_ratio:.3f}"
        )
    if pick_count <= 0:
        raise ValueError("--pick_count must be > 0")
    if pick_count > len(top_pool):
        raise ValueError(
            f"pick_count={pick_count} exceeds top pool size={len(top_pool)}. "
            "Lower --pick_count or increase --top_ratio."
        )

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(top_pool), size=pick_count, replace=False)
    selected = [top_pool[int(i)] for i in chosen_idx]

    print(f"Valid pairs: {len(valid)}")
    print(f"Top pool size ({top_ratio:.2%}): {len(top_pool)}")
    print(f"Selected datasets: {len(selected)}")
    for t, _, _, sid in selected:
        print(f"- sid={sid}, length={t}")
    return [(nos_path, pure_path, sid, t) for t, nos_path, pure_path, sid in selected]


def _run_training(
    args: argparse.Namespace, eeg_path: Path, nos_path: Path, run_suffix: str
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "stfnet_module" / "train_stfnet.py"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{run_suffix}"
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


def _merge_selected(
    selected: list[tuple[Path, Path, str, int]], merged_dir: Path, timestamp: str
) -> tuple[Path, Path, str]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sids: list[str] = []
    min_t = None
    min_c = None

    for nos_path, pure_path, sid, _ in selected:
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
        sids.append(sid)

    x_aligned = [x[:, :min_c, :min_t] for x in xs]
    y_aligned = [y[:, :min_c, :min_t] for y in ys]
    x_merged = np.concatenate(x_aligned, axis=0).astype(np.float32)
    y_merged = np.concatenate(y_aligned, axis=0).astype(np.float32)

    merged_dir.mkdir(parents=True, exist_ok=True)
    nos_merged = merged_dir / f"Contaminated_top_pool_merged_{timestamp}.npy"
    pure_merged = merged_dir / f"Pure_top_pool_merged_{timestamp}.npy"
    np.save(nos_merged, x_merged)
    np.save(pure_merged, y_merged)

    print(f"Merged NOS: {nos_merged} shape={x_merged.shape}")
    print(f"Merged EEG: {pure_merged} shape={y_merged.shape}")
    run_suffix = f"top{len(sids)}_{'_'.join(sids)}"
    if len(run_suffix) > 120:
        run_suffix = f"top{len(sids)}"
    return pure_merged, nos_merged, run_suffix


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    contaminated_dir = (project_root / args.contaminated_dir).resolve()

    pairs = _collect_pair_lengths(contaminated_dir, args.combo)
    if not pairs:
        raise ValueError(
            f"No matched pairs for combo={args.combo} in {contaminated_dir}"
        )

    selected = _pick_topk_from_pool(
        pairs=pairs,
        top_ratio=args.top_ratio,
        min_len=args.min_len,
        seed=args.seed,
        pick_count=args.pick_count,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_dir = contaminated_dir / "merged"
    eeg_merged, nos_merged, run_suffix = _merge_selected(selected, merged_dir, timestamp)

    _run_training(args, eeg_merged, nos_merged, run_suffix)


if __name__ == "__main__":
    main()
