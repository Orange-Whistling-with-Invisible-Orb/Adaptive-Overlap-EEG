from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairMeta:
    nos_path: Path
    pure_path: Path
    combo: str
    sid: str
    min_t: int
    min_c: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge contaminated pairs from a target folder and run STFNet training"
    )
    # Keep training args aligned with train_stfnet.py.
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
    parser.add_argument(
        "--keep_fold_models",
        action="store_true",
        help="Pass through to train_stfnet.py",
    )

    # Pair collection args.
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Target folder to scan recursively for .npy pairs. "
        "If not set, fallback to --contaminated_dir under project root.",
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Multiple target folders to scan recursively for .npy pairs. "
            "Example: --data_dirs data/contaminated_a data/contaminated_b"
        ),
    )
    parser.add_argument("--contaminated_dir", type=str, default="data/contaminated")
    parser.add_argument(
        "--combo",
        type=str,
        default="all",
        choices=["all", "eog", "emg", "hybrid", "mixed"],
        help="all means use every direct Contaminated_* pair",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=5000,
        help="Drop pair if min(clean_T, noisy_T) is smaller than this threshold",
    )
    return parser.parse_args()


def _to_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")


def _parse_combo_sid(nos_stem: str) -> tuple[str, str] | None:
    if not nos_stem.startswith("Contaminated_"):
        return None
    tail = nos_stem[len("Contaminated_") :]
    if "_" not in tail:
        return None
    combo, sid = tail.split("_", 1)
    if len(combo) == 0 or len(sid) == 0:
        return None
    return combo, sid


def _collect_direct_pairs(
    data_dir: Path, combo_filter: str, min_len: int
) -> list[PairMeta]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pairs: list[PairMeta] = []
    npy_files = sorted(data_dir.rglob("*.npy"), key=lambda p: str(p).lower())
    pure_candidates: dict[str, list[Path]] = {}
    for p in npy_files:
        if p.name.startswith("Pure_"):
            pure_candidates.setdefault(p.name, []).append(p)

    for nos_path in npy_files:
        if not nos_path.is_file():
            continue

        parsed = _parse_combo_sid(nos_path.stem)
        if parsed is None:
            continue
        combo, sid = parsed
        if combo_filter != "all" and combo != combo_filter:
            continue

        pure_name = f"Pure_{sid}.npy"
        pure_list = pure_candidates.get(pure_name, [])
        if len(pure_list) == 0:
            print(f"[skip] no matching pure file for {nos_path.name}")
            continue
        pure_list = sorted(
            pure_list,
            key=lambda p: (
                0 if p.parent == nos_path.parent else 1,
                len(p.parts),
                str(p).lower(),
            ),
        )
        pure_path = pure_list[0]
        if len(pure_list) > 1:
            print(
                f"[warn] multiple pure candidates for sid={sid}, "
                f"use: {pure_path}"
            )

        try:
            x = _to_3d(np.load(nos_path, mmap_mode="r"))
            y = _to_3d(np.load(pure_path, mmap_mode="r"))
        except Exception as exc:
            print(f"[skip] failed loading pair ({nos_path.name}): {exc}")
            continue

        if x.shape[0] != y.shape[0]:
            print(
                f"[skip] subject count mismatch: {nos_path.name} vs {pure_path.name}, "
                f"{x.shape[0]} != {y.shape[0]}"
            )
            continue

        pair_min_t = int(min(x.shape[-1], y.shape[-1]))
        pair_min_c = int(min(x.shape[1], y.shape[1]))
        if pair_min_t < min_len:
            print(f"[skip] {sid} too short: min_t={pair_min_t} < min_len={min_len}")
            continue

        pairs.append(
            PairMeta(
                nos_path=nos_path,
                pure_path=pure_path,
                combo=combo,
                sid=sid,
                min_t=pair_min_t,
                min_c=pair_min_c,
            )
        )

    return pairs


def _resolve_data_dirs(args: argparse.Namespace, project_root: Path) -> list[Path]:
    if args.data_dirs:
        out = []
        for raw in args.data_dirs:
            p0 = Path(raw)
            p = p0.resolve() if p0.is_absolute() else (project_root / p0).resolve()
            out.append(p)
        return out

    if args.data_dir:
        p0 = Path(args.data_dir)
        p = p0.resolve() if p0.is_absolute() else (project_root / p0).resolve()
        return [p]

    return [(project_root / args.contaminated_dir).resolve()]


def _merge_pairs(
    pairs: list[PairMeta], merged_dir: Path, combo_tag: str, timestamp: str
) -> tuple[Path, Path]:
    min_t = min(p.min_t for p in pairs)
    min_c = min(p.min_c for p in pairs)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for p in pairs:
        x = _to_3d(np.load(p.nos_path))
        y = _to_3d(np.load(p.pure_path))
        xs.append(x[:, :min_c, :min_t])
        ys.append(y[:, :min_c, :min_t])

    x_merged = np.concatenate(xs, axis=0).astype(np.float32)
    y_merged = np.concatenate(ys, axis=0).astype(np.float32)

    merged_dir.mkdir(parents=True, exist_ok=True)
    nos_merged = merged_dir / f"Contaminated_{combo_tag}_direct_merged_{timestamp}.npy"
    pure_merged = merged_dir / f"Pure_direct_merged_{timestamp}.npy"
    np.save(nos_merged, x_merged)
    np.save(pure_merged, y_merged)

    print(f"[merge] pairs={len(pairs)}, min_c={min_c}, min_t={min_t}")
    print(f"[merge] NOS:  {nos_merged} shape={x_merged.shape}")
    print(f"[merge] EEG:  {pure_merged} shape={y_merged.shape}")
    return pure_merged, nos_merged


def _run_training(
    args: argparse.Namespace, eeg_path: Path, nos_path: Path, combo_tag: str, timestamp: str
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "stfnet_module" / "train_stfnet.py"

    run_name = f"direct_{combo_tag}_{timestamp}"
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
    if args.keep_fold_models:
        cmd.append("--keep_fold_models")

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    best_overall = save_dir_ts / "best_overall.pth"
    if not best_overall.exists():
        raise FileNotFoundError(
            f"Training finished but no best_overall.pth found at: {best_overall}"
        )
    print(f"[run] best checkpoint: {best_overall}")
    return best_overall


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    combo_tag = args.combo
    data_dirs = _resolve_data_dirs(args, project_root)
    pairs: list[PairMeta] = []
    for data_dir in data_dirs:
        pairs.extend(
            _collect_direct_pairs(
                data_dir=data_dir, combo_filter=args.combo, min_len=args.min_len
            )
        )
    if not pairs:
        raise ValueError(
            f"No valid direct pairs found in data_dirs={data_dirs} for combo={args.combo} "
            f"with min_len={args.min_len}"
        )

    print("[collect] source dirs:")
    for d in data_dirs:
        print(f"[collect] - {d}")
    print(f"[collect] valid direct pairs: {len(pairs)}")
    for p in pairs:
        print(f"[collect] - combo={p.combo} sid={p.sid} min_t={p.min_t}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(data_dirs) == 1:
        merged_dir = data_dirs[0] / "merged"
    else:
        merged_dir = (project_root / "data" / "merged_multi_direct").resolve()
    eeg_merged, nos_merged = _merge_pairs(
        pairs=pairs, merged_dir=merged_dir, combo_tag=combo_tag, timestamp=timestamp
    )
    _run_training(
        args=args,
        eeg_path=eeg_merged,
        nos_path=nos_merged,
        combo_tag=combo_tag,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
