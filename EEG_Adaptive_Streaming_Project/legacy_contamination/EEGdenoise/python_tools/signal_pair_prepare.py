"""
signal_pair_prepare.py

对应 MATLAB 脚本: signal_pair_prepare.m
用途: 对已有 epochs 做重采样、随机打乱、数量对齐，并保存为 .mat/.npy。

典型用途:
- 把某一类 epochs（如 EEG/EOG/EMG）整理成训练前可直接使用的二维数组 (N, T)。
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import resample_poly


def load_epochs(mat_path: str, key: str) -> np.ndarray:
    """从 .mat 读取指定 key，并确保输出是二维数组 (N, T)。"""
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if key not in d:
        raise KeyError(f"Key '{key}' not found in {mat_path}")
    x = np.asarray(d[key], dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D epochs, got shape {x.shape}")
    return x


def load_epochs_npy(npy_path: str) -> np.ndarray:
    """从 .npy 读取 epochs，并确保是二维数组 (N, T)。"""
    x = np.load(npy_path)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D epochs from npy, got shape {x.shape}")
    return x


def maybe_resample_epochs(
    epochs: np.ndarray, raw_fs: int, target_fs: int
) -> np.ndarray:
    """按行（每个 epoch）重采样：shape 仍保持 (N, T_new)。"""
    if raw_fs == target_fs:
        return epochs.astype(np.float32)
    out = resample_poly(epochs, up=target_fs, down=raw_fs, axis=1)
    return out.astype(np.float32)


def random_scramble(epochs: np.ndarray, seed: int) -> np.ndarray:
    """随机打乱样本顺序，对应 MATLAB 的 randperm。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(epochs.shape[0])
    return epochs[idx]


def align_count(epochs: np.ndarray, target_n: int) -> np.ndarray:
    """
    对齐样本数量:
    - 若样本不足: 复制前若干条补齐
    - 若样本过多: 截断到 target_n
    """
    n = epochs.shape[0]
    if n == target_n:
        return epochs
    if n > target_n:
        return epochs[:target_n]

    need = target_n - n
    rep = epochs[:need]
    return np.concatenate([rep, epochs], axis=0)


def build_pairs(
    input_path: str,
    key: str,
    input_type: str,
    raw_fs: int,
    target_fs: int,
    target_n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    主流程:
    1) 读取 epochs
    2) 重采样
    3) 随机打乱
    4) 数量对齐

    返回:
    - epochs_rs: 重采样后原始顺序
    - epochs_final: 打乱并对齐后的结果
    """
    if input_type == "mat":
        epochs = load_epochs(input_path, key=key)
    elif input_type == "npy":
        epochs = load_epochs_npy(input_path)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
    epochs_rs = maybe_resample_epochs(epochs, raw_fs=raw_fs, target_fs=target_fs)
    epochs_rand = random_scramble(epochs_rs, seed=seed)
    epochs_final = align_count(epochs_rand, target_n=target_n)
    return epochs_rs.astype(np.float32), epochs_final.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resample/scramble/align epochs and export as .npy (optional .mat)"
    )
    p.add_argument("--input-path", required=True, help="Input path (.npy or .mat)")
    p.add_argument("--input-type", choices=["npy", "mat"], default="npy")
    p.add_argument(
        "--key",
        default="",
        help="Variable name inside .mat (only for --input-type mat)",
    )
    p.add_argument("--raw-fs", type=int, default=500)
    p.add_argument("--target-fs", type=int, default=512)
    p.add_argument("--target-n", type=int, default=4514)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--output-prefix", default="epochs_prepared")
    p.add_argument("--save-mat", action="store_true", help="Also save .mat output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_type == "mat" and not args.key:
        raise ValueError("--key is required when --input-type mat")

    epochs_rs, epochs_final = build_pairs(
        input_path=args.input_path,
        key=args.key,
        input_type=args.input_type,
        raw_fs=args.raw_fs,
        target_fs=args.target_fs,
        target_n=args.target_n,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_npy = os.path.join(args.output_dir, f"{args.output_prefix}.npy")
    np.save(out_npy, epochs_final.astype(np.float32))

    print("Saved:")
    print(out_npy)
    if args.save_mat:
        out_mat = os.path.join(args.output_dir, f"{args.output_prefix}.mat")
        savemat(
            out_mat,
            {
                "epochs_resampled": epochs_rs,
                "epochs_prepared": epochs_final,
                "fs": args.target_fs,
            },
        )
        print(out_mat)
    print("resampled shape:", epochs_rs.shape)
    print("prepared shape:", epochs_final.shape)


if __name__ == "__main__":
    main()
