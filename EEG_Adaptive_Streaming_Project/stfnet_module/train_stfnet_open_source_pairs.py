from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command pipeline for open-source dict npy files: "
            "prepare direct pairs + launch STFNet training"
        )
    )
    parser.add_argument(
        "--pure_dict",
        type=str,
        default="data/open_source_data/Pure_Data.npy",
        help="Path to Pure_Data.npy (dict format).",
    )
    parser.add_argument(
        "--contaminated_dict",
        type=str,
        default="data/open_source_data/Contaminated_Data.npy",
        help="Path to Contaminated_Data.npy (dict format).",
    )
    parser.add_argument(
        "--pair_dir",
        type=str,
        default=None,
        help=(
            "Output directory for generated direct pairs. "
            "If omitted, auto-create with timestamp."
        ),
    )
    parser.add_argument(
        "--existing_pair_dir",
        type=str,
        default="",
        help=(
            "Use an existing direct-pair directory directly "
            "(contains Pure_<sid>.npy and Contaminated_<combo>_<sid>.npy). "
            "When provided, skip dict->pair generation."
        ),
    )
    parser.add_argument(
        "--combo",
        type=str,
        default="hybrid",
        choices=["all", "eog", "emg", "hybrid", "mixed"],
        help="Contaminated file naming combo tag.",
    )
    parser.add_argument(
        "--sid_width",
        type=int,
        default=3,
        help="ID width for generated sid (e.g. 001, 002, ...).",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="open_source_mat",
        help="Tag used in output folder and log names.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Manual timestamp override (default: current time).",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only generate direct pairs, do not start training.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.0,
        help=(
            "Test split ratio in [0,1). "
            "If >0, split pairs into train/test folders and train on train only."
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Random seed for train/test split (default: use --seed).",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="",
        help=(
            "Output root directory for split dataset. "
            "If empty and split enabled, auto-create with timestamp."
        ),
    )

    # Training args (forward to train_stfnet_direct_pairs.py).
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="stfnet_module/checkpoints")
    parser.add_argument(
        "--log_dir", type=str, default="stfnet_module/checkpoints/json_file"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--min_len", type=int, default=5000)
    parser.add_argument("--keep_fold_models", action="store_true")
    parser.add_argument(
        "--train_log",
        type=str,
        default=None,
        help="Training pipeline log path. If omitted, auto-create with timestamp.",
    )
    return parser.parse_args()


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    p0 = Path(raw_path)
    return p0.resolve() if p0.is_absolute() else (project_root / p0).resolve()


def _natural_key(text: str) -> list[object]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def _load_dict_npy(path: Path) -> dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    try:
        data = obj.item()
    except Exception as exc:
        raise ValueError(f"Expected dict npy at {path}, but .item() failed: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    return data


def _prepare_direct_pairs(
    pure_dict: dict[str, np.ndarray],
    contaminated_dict: dict[str, np.ndarray],
    pair_dir: Path,
    combo: str,
    sid_width: int,
) -> int:
    pure_keys = sorted(pure_dict.keys(), key=_natural_key)
    cont_keys = sorted(contaminated_dict.keys(), key=_natural_key)
    if len(pure_keys) != len(cont_keys):
        raise ValueError(
            "Pure/Contaminated trial count mismatch: "
            f"{len(pure_keys)} != {len(cont_keys)}"
        )

    pair_dir.mkdir(parents=True, exist_ok=True)
    for i, (pk, ck) in enumerate(zip(pure_keys, cont_keys), start=1):
        sid = f"{i:0{sid_width}d}"
        pure_arr = np.asarray(pure_dict[pk], dtype=np.float32)
        cont_arr = np.asarray(contaminated_dict[ck], dtype=np.float32)

        if pure_arr.ndim != 2 or cont_arr.ndim != 2:
            raise ValueError(
                "Expected 2D arrays in dict npy. "
                f"Got pure={pure_arr.shape}, contaminated={cont_arr.shape}"
            )

        # Save as (1, C, T) to align with direct pair training loader.
        np.save(pair_dir / f"Pure_{sid}.npy", pure_arr[None, ...])
        np.save(pair_dir / f"Contaminated_{combo}_{sid}.npy", cont_arr[None, ...])

    return len(pure_keys)


def _parse_combo_sid_from_contaminated(stem: str) -> tuple[str, str] | None:
    if not stem.startswith("Contaminated_"):
        return None
    tail = stem[len("Contaminated_") :]
    if "_" not in tail:
        return None
    combo_tag, sid = tail.split("_", 1)
    if len(combo_tag) == 0 or len(sid) == 0:
        return None
    return combo_tag, sid


def _collect_pairs_from_dir(
    pair_dir: Path, combo_filter: str
) -> list[tuple[str, str, Path, Path]]:
    if not pair_dir.exists():
        raise FileNotFoundError(f"pair_dir not found: {pair_dir}")

    pure_map: dict[str, Path] = {}
    contaminated_list: list[tuple[str, str, Path]] = []
    for p in sorted(pair_dir.glob("*.npy"), key=lambda x: x.name.lower()):
        stem = p.stem
        if stem.startswith("Pure_"):
            sid = stem[len("Pure_") :]
            if sid:
                pure_map[sid] = p
            continue

        parsed = _parse_combo_sid_from_contaminated(stem)
        if parsed is None:
            continue
        combo_tag, sid = parsed
        if combo_filter != "all" and combo_tag != combo_filter:
            continue
        contaminated_list.append((combo_tag, sid, p))

    out: list[tuple[str, str, Path, Path]] = []
    for combo_tag, sid, cont_path in contaminated_list:
        pure_path = pure_map.get(sid)
        if pure_path is None:
            continue
        out.append((sid, combo_tag, pure_path, cont_path))
    return out


def _split_pair_dir(
    source_pair_dir: Path,
    split_root: Path,
    combo_filter: str,
    split_ratio: float,
    split_seed: int,
) -> tuple[Path, Path, int, int]:
    if not (0.0 <= split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in [0,1), got {split_ratio}")

    pairs = _collect_pairs_from_dir(source_pair_dir, combo_filter=combo_filter)
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least 2 valid pairs to split, got {len(pairs)} from {source_pair_dir}"
        )

    n_total = len(pairs)
    n_test = int(round(n_total * split_ratio))
    n_test = max(1, n_test)
    n_test = min(n_total - 1, n_test)

    rng = np.random.RandomState(int(split_seed))
    perm = rng.permutation(n_total)
    test_idx = set(int(x) for x in perm[:n_test])

    train_dir = split_root / "train"
    test_dir = split_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_pair_dir": str(source_pair_dir),
        "split_root": str(split_root),
        "combo_filter": combo_filter,
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
        "total_pairs": int(n_total),
        "train_pairs": [],
        "test_pairs": [],
    }

    for i, (sid, combo_tag, pure_path, cont_path) in enumerate(pairs):
        dst_dir = test_dir if i in test_idx else train_dir
        shutil.copy2(pure_path, dst_dir / pure_path.name)
        shutil.copy2(cont_path, dst_dir / cont_path.name)
        rec = {
            "sid": sid,
            "combo": combo_tag,
            "pure_file": pure_path.name,
            "contaminated_file": cont_path.name,
        }
        if i in test_idx:
            manifest["test_pairs"].append(rec)
        else:
            manifest["train_pairs"].append(rec)

    import json

    (split_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return train_dir, test_dir, len(manifest["train_pairs"]), len(manifest["test_pairs"])


def _build_train_cmd(
    project_root: Path,
    args: argparse.Namespace,
    pair_dir: Path,
    save_dir_ts: Path,
    log_dir_ts: Path,
) -> list[str]:
    script = (project_root / "stfnet_module" / "train_stfnet_direct_pairs.py").resolve()
    cmd = [
        sys.executable,
        str(script),
        "--data_dir",
        str(pair_dir),
        "--combo",
        args.combo,
        "--min_len",
        str(args.min_len),
        "--device",
        args.device,
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
        "--save_dir",
        str(save_dir_ts),
        "--log_dir",
        str(log_dir_ts),
    ]
    if args.keep_fold_models:
        cmd.append("--keep_fold_models")
    return cmd


def _run_and_tee(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[command]\n")
        f.write(" ".join(cmd))
        f.write("\n\n[output]\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    split_seed = int(args.seed if args.split_seed is None else args.split_seed)
    split_enabled = float(args.split_ratio) > 0.0

    if args.existing_pair_dir.strip():
        pair_dir = _resolve_path(args.existing_pair_dir.strip(), project_root)
        pair_count = len(_collect_pairs_from_dir(pair_dir, combo_filter=args.combo))
        print(f"[prepare] use existing pair_dir: {pair_dir}")
        print(f"[prepare] matched pairs (combo={args.combo}): {pair_count}")
    else:
        pure_path = _resolve_path(args.pure_dict, project_root)
        contaminated_path = _resolve_path(args.contaminated_dict, project_root)

        if args.pair_dir:
            pair_dir = _resolve_path(args.pair_dir, project_root)
        else:
            pair_dir = (
                project_root
                / "data"
                / "open_source_data"
                / f"direct_pairs_{args.run_tag}_{timestamp}"
            ).resolve()

        pure_dict = _load_dict_npy(pure_path)
        cont_dict = _load_dict_npy(contaminated_path)
        pair_count = _prepare_direct_pairs(
            pure_dict=pure_dict,
            contaminated_dict=cont_dict,
            pair_dir=pair_dir,
            combo=args.combo,
            sid_width=args.sid_width,
        )
        print(f"[prepare] pair_dir: {pair_dir}")
        print(f"[prepare] pairs:    {pair_count}")

    train_data_dir = pair_dir
    if split_enabled:
        if args.split_dir.strip():
            split_root = _resolve_path(args.split_dir.strip(), project_root)
        else:
            split_root = (
                project_root
                / "data"
                / "open_source_data"
                / f"direct_pairs_split_{args.run_tag}_{timestamp}"
            ).resolve()
        train_dir, test_dir, n_train, n_test = _split_pair_dir(
            source_pair_dir=pair_dir,
            split_root=split_root,
            combo_filter=args.combo,
            split_ratio=float(args.split_ratio),
            split_seed=split_seed,
        )
        train_data_dir = train_dir
        print(f"[split] split_root: {split_root}")
        print(f"[split] train_dir:  {train_dir} (pairs={n_train})")
        print(f"[split] test_dir:   {test_dir} (pairs={n_test})")
        print(f"[split] manifest:   {split_root / 'split_manifest.json'}")

    if args.prepare_only:
        print("[prepare] prepare_only=True, stop before training.")
        return

    save_dir_root = _resolve_path(args.save_dir, project_root)
    log_dir_root = _resolve_path(args.log_dir, project_root)
    save_dir_ts = save_dir_root / f"{args.run_tag}_{timestamp}"
    log_dir_ts = log_dir_root / f"{args.run_tag}_{timestamp}"

    if args.train_log:
        train_log = _resolve_path(args.train_log, project_root)
    else:
        train_log = save_dir_root / f"train_{args.run_tag}_{timestamp}.log"

    cmd = _build_train_cmd(
        project_root=project_root,
        args=args,
        pair_dir=train_data_dir,
        save_dir_ts=save_dir_ts,
        log_dir_ts=log_dir_ts,
    )
    print("[run] start training pipeline...")
    print(f"[run] log file: {train_log}")
    _run_and_tee(cmd=cmd, log_path=train_log)
    print("[run] done.")

    best_list = sorted(
        save_dir_ts.rglob("best_overall.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if best_list:
        print(f"[run] best checkpoint: {best_list[0]}")
    else:
        print("[run] warning: no best_overall.pth found under save_dir.")


if __name__ == "__main__":
    main()
