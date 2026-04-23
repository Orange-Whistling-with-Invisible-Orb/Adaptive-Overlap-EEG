from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core_algorithm.training.online_trainer import StreamingFusionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练双尺度位置评估头（流式窗口权重网络）"
    )
    parser.add_argument(
        "--contaminated_path",
        type=str,
        default="",
        help="指定污染数据路径（可选，不指定则按 combo 自动挑选）",
    )
    parser.add_argument(
        "--pure_path",
        type=str,
        default="",
        help="指定纯净数据路径（可选，默认按污染文件名自动匹配）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/contaminated",
        help="数据目录，包含 Contaminated_*.npy 与 Pure_*.npy",
    )
    parser.add_argument(
        "--combo",
        type=str,
        default="hybrid",
        choices=["eog", "emg", "hybrid", "mixed"],
        help="自动选样时的污染类型",
    )
    parser.add_argument(
        "--sid",
        type=str,
        default="",
        help="自动选样时指定 sid（例如 02_f0）",
    )
    parser.add_argument(
        "--pick",
        type=str,
        default="longest",
        choices=["longest", "random"],
        help="自动选样策略：最长样本或随机样本",
    )

    parser.add_argument(
        "--stfnet_ckpt",
        type=str,
        default="stfnet_module/best_overall.pth",
        help="冻结 STFNet 权重路径",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--window_len", type=int, default=500, help="滑窗长度 L")
    parser.add_argument("--overlap_n", type=int, default=3, help="重叠参数 N")
    parser.add_argument("--packet_samples", type=int, default=60, help="每包样本点数")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--local_kernel", type=int, default=15, help="局部分支卷积核")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="MSE loss 权重")
    parser.add_argument("--l1_weight", type=float, default=0.0, help="L1 loss 权重")

    parser.add_argument("--results_dir", type=str, default="results", help="结果根目录")
    parser.add_argument("--run_name", type=str, default="", help="手动指定运行名")
    return parser.parse_args()


def _to_3d(arr: np.ndarray, n_channels: int = 19) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[0] == n_channels:
            return x[None, ...]
        if x.shape[1] == n_channels:
            return np.transpose(x, (1, 0))[None, ...]
        raise ValueError(f"2D array cannot be interpreted as [C,T], got {x.shape}")
    if x.ndim == 3:
        # [S,C,T]
        if x.shape[1] == n_channels:
            return x
        # [S,T,C]
        if x.shape[2] == n_channels:
            return np.transpose(x, (0, 2, 1))
        raise ValueError(f"3D array cannot be interpreted as [S,C,T], got {x.shape}")
    raise ValueError(f"Expected 2D/3D array, got {x.shape}")


def _derive_pure_path_from_contaminated(contaminated_path: Path) -> Path:
    if not contaminated_path.name.startswith("Contaminated_"):
        raise ValueError(
            f"Cannot derive pure path from filename: {contaminated_path.name}"
        )
    stem = contaminated_path.stem.replace("Contaminated_", "", 1)
    if "_" not in stem:
        raise ValueError(f"Unexpected contaminated filename: {contaminated_path.name}")
    sid = stem.split("_", 1)[1]
    return contaminated_path.with_name(f"Pure_{sid}{contaminated_path.suffix}")


def _resolve_pair(args: argparse.Namespace, project_root: Path) -> tuple[Path, Path]:
    if args.contaminated_path.strip():
        cont_path = (project_root / args.contaminated_path).resolve()
        pure_path = (
            (project_root / args.pure_path).resolve()
            if args.pure_path.strip()
            else _derive_pure_path_from_contaminated(cont_path)
        )
        if not cont_path.exists():
            raise FileNotFoundError(f"Contaminated file not found: {cont_path}")
        if not pure_path.exists():
            raise FileNotFoundError(f"Pure file not found: {pure_path}")
        return cont_path, pure_path

    data_dir = (project_root / args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    cands = sorted(data_dir.glob(f"Contaminated_{args.combo}_*.npy"))
    if args.sid.strip():
        cands = [p for p in cands if p.stem.endswith(args.sid)]
    pairs = []
    for c in cands:
        p = _derive_pure_path_from_contaminated(c)
        if p.exists():
            try:
                arr = np.load(c, mmap_mode="r")
                t = arr.shape[-1]
            except Exception:
                continue
            pairs.append((int(t), c, p))
    if not pairs:
        raise FileNotFoundError(
            f"No matched contaminated/pure pairs found in {data_dir} for combo={args.combo} sid={args.sid}"
        )
    if args.pick == "random":
        idx = random.randint(0, len(pairs) - 1)
        _, cont_path, pure_path = pairs[idx]
    else:
        pairs.sort(key=lambda z: z[0], reverse=True)
        _, cont_path, pure_path = pairs[0]
    return cont_path, pure_path


def _load_aligned_pair(cont_path: Path, pure_path: Path, n_channels: int = 19):
    cont = _to_3d(np.load(cont_path), n_channels=n_channels)
    pure = _to_3d(np.load(pure_path), n_channels=n_channels)
    s = min(cont.shape[0], pure.shape[0])
    t = min(cont.shape[2], pure.shape[2])
    cont = cont[:s, :, :t]
    pure = pure[:s, :, :t]
    return cont.astype(np.float32), pure.astype(np.float32)


def _split_samples(
    contaminated: np.ndarray,
    pure: np.ndarray,
    val_ratio: float,
    window_len: int,
    step_size: int,
):
    s = contaminated.shape[0]
    if s >= 2:
        val_count = max(1, int(round(s * val_ratio)))
        train_count = max(1, s - val_count)
        if train_count >= s:
            train_count = s - 1
        train = [(contaminated[i], pure[i]) for i in range(train_count)]
        val = [(contaminated[i], pure[i]) for i in range(train_count, s)]
        if len(val) == 0:
            val = train[-1:]
        return train, val

    # 单 subject：按时间切分 train/val
    c = contaminated[0]
    p = pure[0]
    t = c.shape[1]
    min_len = window_len + step_size
    if t < 2 * min_len:
        # 太短就直接同一条做 train/val（会提示）
        print(
            f"[Warn] sequence too short for strict train/val split (T={t}), reuse same sample for val."
        )
        return [(c, p)], [(c, p)]

    split_t = int(t * (1.0 - val_ratio))
    split_t = max(min_len, min(split_t, t - min_len))
    train = [(c[:, :split_t], p[:, :split_t])]
    # 为避免边界损失，val 向前回看 step_size
    val = [(c[:, split_t - step_size :], p[:, split_t - step_size :])]
    return train, val


def _save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    stfnet_ckpt = (project_root / args.stfnet_ckpt).resolve()
    if not stfnet_ckpt.exists():
        raise FileNotFoundError(f"STFNet checkpoint not found: {stfnet_ckpt}")

    cont_path, pure_path = _resolve_pair(args, project_root)
    print("[Init] selected pair:")
    print(f"  contaminated: {cont_path}")
    print(f"  pure        : {pure_path}")
    print(f"  stfnet_ckpt : {stfnet_ckpt}")

    contaminated, pure = _load_aligned_pair(cont_path, pure_path, n_channels=19)
    print(
        f"[Data] loaded shape contaminated={contaminated.shape}, pure={pure.shape}"
    )

    step_size = max(1, args.window_len // max(1, args.overlap_n))
    train_samples, val_samples = _split_samples(
        contaminated, pure, args.val_ratio, args.window_len, step_size
    )
    print(
        f"[Data] train_samples={len(train_samples)}, val_samples={len(val_samples)}, "
        f"L={args.window_len}, N={args.overlap_n}, S={step_size}, packet={args.packet_samples}"
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = pure_path.stem.replace("Pure_", "")
    run_name = args.run_name or f"N{args.overlap_n}_fusion_weight_{now}_{sid}"
    run_dir = (project_root / args.results_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run] results dir: {run_dir}")

    trainer = StreamingFusionTrainer(
        stfnet_checkpoint=str(stfnet_ckpt),
        n_channels=19,
        window_len=args.window_len,
        overlap_n=args.overlap_n,
        packet_samples=args.packet_samples,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        local_kernel=args.local_kernel,
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
    )

    config = {
        "contaminated_path": str(cont_path),
        "pure_path": str(pure_path),
        "stfnet_ckpt": str(stfnet_ckpt),
        "device": args.device,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "window_len": args.window_len,
        "overlap_n": args.overlap_n,
        "step_size": step_size,
        "packet_samples": args.packet_samples,
        "val_ratio": args.val_ratio,
        "local_kernel": args.local_kernel,
        "mse_weight": args.mse_weight,
        "l1_weight": args.l1_weight,
        "seed": args.seed,
    }
    _save_json(run_dir / "config.json", config)

    metrics_csv = run_dir / "metrics.csv"
    best_ckpt = run_dir / "best_weight_network.pth"
    last_ckpt = run_dir / "last_weight_network.pth"
    best_val = float("inf")
    history = []

    with open(metrics_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_chunks",
                "val_loss",
                "val_chunks",
                "train_sec",
                "val_sec",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            print(f"\n[Epoch {epoch}/{args.epochs}] ------------------------------")
            train_stat = trainer.run_epoch(
                train_samples, train=True, epoch_idx=epoch, verbose=True
            )
            val_stat = trainer.run_epoch(
                val_samples, train=False, epoch_idx=epoch, verbose=True
            )

            print(
                f"[Epoch {epoch:03d}] train_loss={train_stat['mean_loss']:.6f} "
                f"(chunks={train_stat['chunks']}, {train_stat['elapsed_sec']:.2f}s), "
                f"val_loss={val_stat['mean_loss']:.6f} "
                f"(chunks={val_stat['chunks']}, {val_stat['elapsed_sec']:.2f}s)"
            )

            writer.writerow(
                [
                    epoch,
                    train_stat["mean_loss"],
                    train_stat["chunks"],
                    val_stat["mean_loss"],
                    val_stat["chunks"],
                    train_stat["elapsed_sec"],
                    val_stat["elapsed_sec"],
                ]
            )

            ckpt_obj = {
                "state_dict": trainer.scorer.state_dict(),
                "n_channels": 19,
                "window_len": args.window_len,
                "local_kernel": args.local_kernel,
                "overlap_n": args.overlap_n,
                "packet_samples": args.packet_samples,
                "epoch": epoch,
                "train_loss": float(train_stat["mean_loss"]),
                "val_loss": float(val_stat["mean_loss"]),
                "stfnet_checkpoint": str(stfnet_ckpt),
            }
            torch.save(ckpt_obj, last_ckpt)
            if val_stat["mean_loss"] < best_val:
                best_val = val_stat["mean_loss"]
                torch.save(ckpt_obj, best_ckpt)
                print(
                    f"[Best] updated: epoch={epoch}, val_loss={best_val:.6f}, path={best_ckpt}"
                )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_stat["mean_loss"]),
                    "val_loss": float(val_stat["mean_loss"]),
                }
            )

    # 保存一条验证样本重构结果，便于离线分析
    if len(val_samples) > 0:
        pred, tgt = trainer.reconstruct_sample(val_samples[0][0], val_samples[0][1])
        if pred is not None and tgt is not None:
            np.save(run_dir / "val_recon_pred.npy", pred.astype(np.float32))
            np.save(run_dir / "val_recon_target.npy", tgt.astype(np.float32))
            print(
                f"[Save] val reconstruction: pred={pred.shape}, target={tgt.shape}"
            )

    _save_json(run_dir / "history.json", {"best_val_loss": best_val, "history": history})
    print("\n[Done] training finished")
    print(f"  best checkpoint : {best_ckpt}")
    print(f"  last checkpoint : {last_ckpt}")
    print(f"  metrics csv     : {metrics_csv}")


if __name__ == "__main__":
    main()
