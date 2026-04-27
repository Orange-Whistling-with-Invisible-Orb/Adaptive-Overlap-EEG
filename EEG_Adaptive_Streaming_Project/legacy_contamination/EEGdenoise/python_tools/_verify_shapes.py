"""
_verify_shapes.py

用途: 验证 emg_all、eog_all 和 clean_eeg 的形状和有效性

示例:
python data_contaminated/_verify_shapes.py \
  --emg-npy "data/数据处理_手动污染/emg_all.npy" \
  --eog-npy "data/数据处理_手动污染/eog_all.npy" \
  --clean-npy "data/generated_pairs/clean_01_f0.npy"
"""

from __future__ import annotations

import argparse

import numpy as np


def verify_npy_file(name: str, path: str) -> None:
    """验证单个 .npy 文件的形状和有效性。"""
    try:
        data = np.load(path)
        print(f"\n{name}:")
        print(f"  Path: {path}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Is 2D: {data.ndim == 2}")
        print(f"  All finite: {np.isfinite(data).all()}")
        if np.isnan(data).any():
            print(f"  ⚠️  Contains NaN: {np.isnan(data).sum()} values")
        if np.isinf(data).any():
            print(f"  ⚠️  Contains Inf: {np.isinf(data).sum()} values")
        if data.ndim == 2:
            print(f"  Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
    except FileNotFoundError:
        print(f"\n❌ {name}: File not found at {path}")
    except Exception as e:
        print(f"\n❌ {name}: Error - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify shape and validity of EEG npy files"
    )
    parser.add_argument("--emg-npy", help="Path to emg_all.npy")
    parser.add_argument("--eog-npy", help="Path to eog_all.npy")
    parser.add_argument("--clean-npy", help="Path to clean EEG .npy")
    args = parser.parse_args()

    print("=" * 60)
    print("NPY File Verification")
    print("=" * 60)

    files = {
        "EMG": args.emg_npy,
        "EOG": args.eog_npy,
        "Clean EEG": args.clean_npy,
    }

    for name, path in files.items():
        if path:
            verify_npy_file(name, path)

    # 检查一致性
    print("\n" + "=" * 60)
    if args.clean_npy and (args.emg_npy or args.eog_npy):
        try:
            clean = np.load(args.clean_npy)
            clean_len = clean.shape[1] if clean.ndim == 2 else clean.shape[0]

            consistency_ok = True
            if args.emg_npy:
                emg = np.load(args.emg_npy)
                emg_len = emg.shape[1] if emg.ndim == 2 else emg.shape[0]
                match = emg_len == clean_len
                print(f"EMG length ({emg_len}) == Clean length ({clean_len}): {match}")
                consistency_ok = consistency_ok and match

            if args.eog_npy:
                eog = np.load(args.eog_npy)
                eog_len = eog.shape[1] if eog.ndim == 2 else eog.shape[0]
                match = eog_len == clean_len
                print(f"EOG length ({eog_len}) == Clean length ({clean_len}): {match}")
                consistency_ok = consistency_ok and match

            print(
                f"\n{'✅ All checks passed!' if consistency_ok else '❌ Some checks failed!'}"
            )
        except Exception as e:
            print(f"Consistency check error: {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
