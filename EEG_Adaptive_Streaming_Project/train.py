from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from core_algorithm.training.online_trainer import StreamingFusionTrainer


def _parse_init_window_weights(raw: str) -> list[float]:
    txt = str(raw).strip()
    if not txt:
        return []
    parts = [p.strip() for p in txt.split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    if len(vals) == 0:
        raise argparse.ArgumentTypeError(
            "--init_window_weights 需要逗号分隔浮点数，例如: 0.5,0.3,0.2"
        )
    return vals


def _build_init_weight_tag(weights: list[float]) -> str:
    vals = [f"{float(v):.3f}".rstrip("0").rstrip(".") for v in weights]
    safe = [v.replace(".", "p") if v else "0" for v in vals]
    if len(safe) > 6:
        safe = safe[:6] + [f"k{len(vals)}"]
    return "initw_" + "_".join(safe)


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
        required=True,
        help="数据目录（必填），包含 Contaminated_*.npy 与 Pure_*.npy",
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
        help="自动选样策略：最长样本优先或随机抽样",
    )
    parser.add_argument(
        "--pick_count",
        type=int,
        default=5,
        help="自动选样时联合训练的数据对数量（默认 5）",
    )
    parser.add_argument(
        "--min_pair_len",
        type=int,
        default=0,
        help="自动选样时最小长度阈值（单位: 采样点，默认 0 表示不过滤）",
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
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=4,
        help="验证集无改进时触发降学习率的等待 epoch 数",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="触发降学习率时的乘法因子",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="学习率下限",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（不传则每次自动随机）",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="交叉验证折数（当前 train.py 仅支持 1）",
    )

    parser.add_argument("--window_len", type=int, default=500, help="滑窗长度 L")
    parser.add_argument("--overlap_n", type=int, default=3, help="重叠参数 N")
    parser.add_argument("--packet_samples", type=int, default=60, help="每包样本点数")
    parser.add_argument("--sample_rate", type=float, default=200.0, help="采样率(Hz)")
    parser.add_argument(
        "--preprocess_mode",
        type=int,
        default=1,
        choices=[1, 2],
        help="基础预处理模式: 1=正常预处理(默认), 2=不预处理",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--local_kernel", type=int, default=15, help="局部分支卷积核")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="MSE loss 权重")
    parser.add_argument("--l1_weight", type=float, default=0.0, help="L1 loss 权重")
    parser.add_argument(
        "--entropy_reg_weight",
        type=float,
        default=0.01,
        help="权重熵正则系数（>0 可抑制 one-hot 塌缩）",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=1.0,
        help="权重 softmax 温度（>1 更平滑，<1 更尖锐）",
    )
    parser.add_argument(
        "--init_logit_bias_strength",
        type=float,
        default=0.35,
        help="初始logit位置先验强度（避免初始接近1/K均分）",
    )
    parser.add_argument(
        "--init_window_weights",
        type=_parse_init_window_weights,
        default=None,
        help=(
            "手动指定初始窗口权重(逗号分隔, 如 0.6,0.3,0.1)。"
            "传入后将覆盖 --init_logit_bias_strength 的位置先验。"
        ),
    )

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


def _collect_auto_pairs(
    args: argparse.Namespace, project_root: Path
) -> list[tuple[int, Path, Path]]:
    data_dir = (project_root / args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    cands = sorted(data_dir.glob(f"Contaminated_{args.combo}_*.npy"))
    if args.sid.strip():
        cands = [p for p in cands if p.stem.endswith(args.sid)]

    pairs = []
    for c in cands:
        p = _derive_pure_path_from_contaminated(c)
        if not p.exists():
            continue
        try:
            arr_c = np.load(c, mmap_mode="r")
            arr_p = np.load(p, mmap_mode="r")
            t = min(int(arr_c.shape[-1]), int(arr_p.shape[-1]))
        except Exception:
            continue
        if t >= int(args.min_pair_len):
            pairs.append((t, c, p))
    return pairs


def _resolve_pairs(
    args: argparse.Namespace, project_root: Path
) -> list[tuple[int, Path, Path]]:
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
        try:
            arr_c = np.load(cont_path, mmap_mode="r")
            arr_p = np.load(pure_path, mmap_mode="r")
            t = min(int(arr_c.shape[-1]), int(arr_p.shape[-1]))
        except Exception:
            t = 0
        return [(t, cont_path, pure_path)]

    pairs = _collect_auto_pairs(args, project_root)
    if not pairs:
        raise FileNotFoundError(
            "No matched contaminated/pure pairs found for auto selection. "
            f"combo={args.combo}, sid={args.sid}, min_pair_len={args.min_pair_len}"
        )

    pick_count = max(1, int(args.pick_count))
    if args.pick == "random":
        if pick_count >= len(pairs):
            return pairs
        return random.sample(pairs, k=pick_count)

    pairs.sort(key=lambda z: z[0], reverse=True)
    return pairs[:pick_count]


def _load_aligned_pairs(
    pairs: list[tuple[int, Path, Path]], n_channels: int = 19
) -> tuple[np.ndarray, np.ndarray]:
    cont_list = []
    pure_list = []
    global_t = None

    for _, cont_path, pure_path in pairs:
        cont = _to_3d(np.load(cont_path), n_channels=n_channels)
        pure = _to_3d(np.load(pure_path), n_channels=n_channels)
        s = min(cont.shape[0], pure.shape[0])
        t = min(cont.shape[2], pure.shape[2])
        cont = cont[:s, :, :t]
        pure = pure[:s, :, :t]
        global_t = t if global_t is None else min(global_t, t)
        cont_list.append(cont)
        pure_list.append(pure)

    if not cont_list or global_t is None:
        raise RuntimeError("No valid pairs after loading and alignment.")

    cont_out = np.concatenate([x[:, :, :global_t] for x in cont_list], axis=0)
    pure_out = np.concatenate([x[:, :, :global_t] for x in pure_list], axis=0)
    return cont_out.astype(np.float32), pure_out.astype(np.float32)


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
    if int(args.folds) != 1:
        raise ValueError(
            f"train.py 当前仅支持 --folds 1（收到 {args.folds}）。"
            "该脚本为单次 train/val 划分流程，不执行 K-fold。"
        )
    if args.seed is None:
        used_seed = None
        seed_mode = "no_seed"
    else:
        used_seed = int(args.seed)
        seed_mode = "fixed"
        random.seed(used_seed)
        np.random.seed(used_seed)
        torch.manual_seed(used_seed)
    print(f"[Init] seed_mode={seed_mode}, seed={used_seed}")

    project_root = Path(__file__).resolve().parent
    stfnet_ckpt = (project_root / args.stfnet_ckpt).resolve()
    if not stfnet_ckpt.exists():
        raise FileNotFoundError(f"STFNet checkpoint not found: {stfnet_ckpt}")

    selected_pairs = _resolve_pairs(args, project_root)
    print(f"[Init] selected pairs: {len(selected_pairs)}")
    for idx, (t, cont_path, pure_path) in enumerate(selected_pairs, start=1):
        print(f"  [{idx}] contaminated: {cont_path}")
        print(f"      pure        : {pure_path}")
        print(f"      min_t       : {t}")
    print(f"  stfnet_ckpt : {stfnet_ckpt}")

    contaminated, pure = _load_aligned_pairs(selected_pairs, n_channels=19)
    print(
        f"[Data] loaded shape contaminated={contaminated.shape}, pure={pure.shape}"
    )

    step_size = max(1, args.window_len // max(1, args.overlap_n))
    train_samples, val_samples = _split_samples(
        contaminated, pure, args.val_ratio, args.window_len, step_size
    )
    print(
        f"[Data] train_samples={len(train_samples)}, val_samples={len(val_samples)}, "
        f"L={args.window_len}, N={args.overlap_n}, S={step_size}, "
        f"packet={args.packet_samples}, preprocess_mode={args.preprocess_mode}"
    )
    if args.init_window_weights is not None:
        print(
            f"[Init] init_window_weights(raw)={args.init_window_weights} "
            f"(will be normalized in trainer)"
        )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = (
        selected_pairs[0][2].stem.replace("Pure_", "")
        if len(selected_pairs) == 1
        else f"multi{len(selected_pairs)}"
    )
    init_weight_tag = (
        _build_init_weight_tag(args.init_window_weights)
        if args.init_window_weights is not None
        else ""
    )
    if args.run_name:
        run_name = args.run_name
        if init_weight_tag and init_weight_tag not in run_name:
            run_name = f"{run_name}_{init_weight_tag}"
    else:
        if init_weight_tag:
            run_name = f"N{args.overlap_n}_fusion_weight_{init_weight_tag}_{now}_{sid}"
        else:
            run_name = f"N{args.overlap_n}_fusion_weight_{now}_{sid}"
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
        sample_rate=args.sample_rate,
        preprocess_mode=args.preprocess_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        local_kernel=args.local_kernel,
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        entropy_reg_weight=args.entropy_reg_weight,
        softmax_temperature=args.softmax_temperature,
        init_logit_bias_strength=args.init_logit_bias_strength,
        init_window_weights=args.init_window_weights,
    )

    config = {
        "contaminated_path": (
            str(selected_pairs[0][1]) if len(selected_pairs) == 1 else "MULTI_SELECTED"
        ),
        "pure_path": (
            str(selected_pairs[0][2]) if len(selected_pairs) == 1 else "MULTI_SELECTED"
        ),
        "selected_pairs": [
            {"contaminated": str(c), "pure": str(p), "time_len": int(t)}
            for t, c, p in selected_pairs
        ],
        "stfnet_ckpt": str(stfnet_ckpt),
        "pick": args.pick,
        "pick_count": args.pick_count,
        "min_pair_len": args.min_pair_len,
        "device": args.device,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "min_lr": args.min_lr,
        "window_len": args.window_len,
        "overlap_n": args.overlap_n,
        "step_size": step_size,
        "packet_samples": args.packet_samples,
        "sample_rate": args.sample_rate,
        "preprocess_mode": args.preprocess_mode,
        "val_ratio": args.val_ratio,
        "local_kernel": args.local_kernel,
        "mse_weight": args.mse_weight,
        "l1_weight": args.l1_weight,
        "entropy_reg_weight": args.entropy_reg_weight,
        "softmax_temperature": args.softmax_temperature,
        "init_logit_bias_strength": args.init_logit_bias_strength,
        "init_window_weights": args.init_window_weights,
        "seed": args.seed,
        "folds": args.folds,
        "seed_mode": seed_mode,
        "seed_used": used_seed,
    }
    _save_json(run_dir / "config.json", config)

    metrics_csv = run_dir / "metrics.csv"
    weights_csv = run_dir / "weights_trace.csv"
    best_ckpt = run_dir / "best_weight_network.pth"
    last_ckpt = run_dir / "last_weight_network.pth"
    best_val = float("inf")
    history = []
    max_w_cols = max(1, int((args.window_len + step_size - 1) // step_size))

    with open(metrics_csv, "w", newline="", encoding="utf-8") as fcsv, open(
        weights_csv, "w", newline="", encoding="utf-8"
    ) as fw:
        writer = csv.writer(fcsv)
        w_writer = csv.writer(fw)

        metric_header = [
            "epoch",
            "lr",
            "train_chunks",
            "val_chunks",
            "train_loss",
            "val_loss",
            "train_mse",
            "val_mse",
            "train_snr_db",
            "val_snr_db",
            "train_weight_entropy_mean",
            "val_weight_entropy_mean",
            "train_weight_max_mean",
            "val_weight_max_mean",
            "train_effective_k_mean",
            "val_effective_k_mean",
        ]
        metric_header.extend([f"train_w{i}" for i in range(1, max_w_cols + 1)])
        metric_header.extend([f"val_w{i}" for i in range(1, max_w_cols + 1)])
        writer.writerow(metric_header)

        w_header = [
            "epoch",
            "mode",
            "subject_idx",
            "chunk_idx",
            "k_active",
            "effective_k",
            "entropy",
            "max_weight",
            "mse",
            "snr_db",
        ] + [f"w{i}" for i in range(1, max_w_cols + 1)]
        w_writer.writerow(w_header)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode="min",
            factor=float(args.lr_factor),
            patience=int(max(1, args.lr_patience)),
            min_lr=float(args.min_lr),
        )

        def write_weight_rows(records):
            for rec in records:
                w = rec["weights"]
                wr = [
                    rec["epoch"],
                    rec["mode"],
                    rec["subject_idx"],
                    rec["chunk_idx"],
                    rec["k_active"],
                    rec.get("effective_k", 0.0),
                    rec["entropy"],
                    rec["max_weight"],
                    rec.get("mse", 0.0),
                    rec.get("snr_db", 0.0),
                ]
                wr.extend(w + [0.0] * (max_w_cols - len(w)))
                w_writer.writerow(wr)

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
            print(
                f"[Epoch {epoch:03d}] mse(train/val)="
                f"{train_stat['mean_mse']:.6f}/{val_stat['mean_mse']:.6f}, "
                f"snr_db(train/val)="
                f"{train_stat['mean_snr_db']:.3f}/{val_stat['mean_snr_db']:.3f}"
            )
            print(
                f"[Epoch {epoch:03d}] weight(train): entropy={train_stat['weight_entropy_mean']:.4f}, "
                f"max={train_stat['weight_max_mean']:.4f}; "
                f"effective_k={train_stat['mean_effective_k']:.3f}; "
                f"weight(val): entropy={val_stat['weight_entropy_mean']:.4f}, "
                f"effective_k={val_stat['mean_effective_k']:.3f}, "
                f"max={val_stat['weight_max_mean']:.4f}"
            )

            row = [
                epoch,
                float(trainer.optimizer.param_groups[0]["lr"]),
                train_stat["chunks"],
                val_stat["chunks"],
                train_stat["mean_loss"],
                val_stat["mean_loss"],
                train_stat["mean_mse"],
                val_stat["mean_mse"],
                train_stat["mean_snr_db"],
                val_stat["mean_snr_db"],
                train_stat["weight_entropy_mean"],
                val_stat["weight_entropy_mean"],
                train_stat["weight_max_mean"],
                val_stat["weight_max_mean"],
                train_stat["mean_effective_k"],
                val_stat["mean_effective_k"],
            ]
            train_means = train_stat["weight_means"]
            val_means = val_stat["weight_means"]
            row.extend(train_means + [0.0] * (max_w_cols - len(train_means)))
            row.extend(val_means + [0.0] * (max_w_cols - len(val_means)))
            writer.writerow(row)

            scheduler.step(val_stat["mean_loss"])

            write_weight_rows(train_stat["weight_records"])
            write_weight_rows(val_stat["weight_records"])

            ckpt_obj = {
                "state_dict": trainer.scorer.state_dict(),
                "n_channels": 19,
                "window_len": args.window_len,
                "local_kernel": args.local_kernel,
                "overlap_n": args.overlap_n,
                "packet_samples": args.packet_samples,
                "sample_rate": args.sample_rate,
                "preprocess_mode": args.preprocess_mode,
                "entropy_reg_weight": args.entropy_reg_weight,
                "softmax_temperature": args.softmax_temperature,
                "init_logit_bias_strength": args.init_logit_bias_strength,
                "init_window_weights": args.init_window_weights,
                "epoch": epoch,
                "train_loss": float(train_stat["mean_loss"]),
                "val_loss": float(val_stat["mean_loss"]),
                "train_recon_loss": float(train_stat["mean_recon_loss"]),
                "val_recon_loss": float(val_stat["mean_recon_loss"]),
                "train_mse": float(train_stat["mean_mse"]),
                "val_mse": float(val_stat["mean_mse"]),
                "train_snr_db": float(train_stat["mean_snr_db"]),
                "val_snr_db": float(val_stat["mean_snr_db"]),
                "train_effective_k": float(train_stat["mean_effective_k"]),
                "val_effective_k": float(val_stat["mean_effective_k"]),
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
                    "train_mse": float(train_stat["mean_mse"]),
                    "val_mse": float(val_stat["mean_mse"]),
                    "train_snr_db": float(train_stat["mean_snr_db"]),
                    "val_snr_db": float(val_stat["mean_snr_db"]),
                    "train_effective_k": float(train_stat["mean_effective_k"]),
                    "val_effective_k": float(val_stat["mean_effective_k"]),
                    "train_weight_entropy_mean": float(
                        train_stat["weight_entropy_mean"]
                    ),
                    "val_weight_entropy_mean": float(val_stat["weight_entropy_mean"]),
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
    print(f"  weights csv     : {weights_csv}")


if __name__ == "__main__":
    main()
