from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pick_noise_dir(contaminated_root: Path, noise_dir_name: str) -> Path:
    if noise_dir_name in ("noise", "nosie"):
        out = contaminated_root / noise_dir_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    # Keep compatibility with the existing typo folder name.
    nosie = contaminated_root / "nosie"
    if nosie.exists():
        nosie.mkdir(parents=True, exist_ok=True)
        return nosie

    noise = contaminated_root / "noise"
    noise.mkdir(parents=True, exist_ok=True)
    return noise


def main() -> None:
    default_root = str(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser(
        description="Batch contaminate all EDF files under data/ground_truth"
    )
    parser.add_argument(
        "--project-root",
        default=default_root,
        help="EEG_Adaptive_Streaming_Project root",
    )
    parser.add_argument("--ground-truth-dir", default="data/ground_truth")
    parser.add_argument("--contaminated-dir", default="data/contaminated")
    parser.add_argument(
        "--combo", choices=["eog", "emg", "hybrid", "mixed"], default="hybrid"
    )
    parser.add_argument("--target-fs", type=int, default=200)
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--epoch-len", type=int, default=500)
    parser.add_argument("--window-sec", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-dir-name", choices=["auto", "noise", "nosie"], default="auto"
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    gt_dir = (root / args.ground_truth_dir).resolve()
    contaminated_root = (root / args.contaminated_dir).resolve()
    clean_dir = contaminated_root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir = pick_noise_dir(contaminated_root, args.noise_dir_name)

    tools_dir = (
        root / "legacy_contamination" / "EEGdenoise" / "python_tools"
    ).resolve()
    slicer = str(tools_dir / "_slicer.py")
    emg_main = str(tools_dir / "EMG_label_main.py")
    eog_main = str(tools_dir / "EOG_label_main.py")
    pair_main = str(tools_dir / "signal_pair_prepare.py")
    verify_main = str(tools_dir / "_verify_shapes.py")
    contam_main = str(tools_dir / "_contaminator.py")

    edf_files = sorted(gt_dir.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found in {gt_dir}")

    py = sys.executable

    for edf in edf_files:
        sid = edf.stem
        print(f"\n===== Processing {sid} =====")

        clean_npy = clean_dir / f"clean_{sid}.npy"
        emg_epochs = noise_dir / f"EMG_epochs_{sid}.npy"
        eog_epochs = noise_dir / f"EOG_epochs_{sid}.npy"
        emg_all = noise_dir / f"emg_all_{sid}.npy"
        eog_all = noise_dir / f"eog_all_{sid}.npy"
        contaminated_out = contaminated_root / f"Contaminated_{args.combo}_{sid}.npy"
        pure_out = contaminated_root / f"Pure_{sid}.npy"

        run_cmd(
            [
                py,
                slicer,
                "--edf-path",
                str(edf),
                "--target-fs",
                str(args.target_fs),
                "--channels",
                str(args.channels),
                "--epoch-len",
                str(args.epoch_len),
                "--out-npy",
                str(clean_npy),
            ]
        )

        run_cmd(
            [
                py,
                emg_main,
                "--edf-path",
                str(edf),
                "--target-fs",
                str(args.target_fs),
                "--edf-channels",
                str(args.channels),
                "--edf-emg-band-low",
                "20",
                "--edf-emg-band-high",
                "95",
                "--window-sec",
                str(args.window_sec),
                "--output-dir",
                str(noise_dir),
                "--output-prefix",
                f"EMG_epochs_{sid}",
            ]
        )

        run_cmd(
            [
                py,
                eog_main,
                "--edf-path",
                str(edf),
                "--raw-fs",
                "0",
                "--target-fs",
                str(args.target_fs),
                "--raw-unit",
                "uV",
                "--window-sec",
                str(args.window_sec),
                "--mode",
                "both",
                "--eog-ch-idx",
                "20",
                "21",
                "22",
                "--output-dir",
                str(noise_dir),
                "--output-prefix",
                f"EOG_epochs_{sid}",
            ]
        )

        target_n = int(np.load(clean_npy).shape[0])

        run_cmd(
            [
                py,
                pair_main,
                "--input-path",
                str(eog_epochs),
                "--input-type",
                "npy",
                "--raw-fs",
                str(args.target_fs),
                "--target-fs",
                str(args.target_fs),
                "--target-n",
                str(target_n),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(noise_dir),
                "--output-prefix",
                f"eog_all_{sid}",
            ]
        )

        run_cmd(
            [
                py,
                pair_main,
                "--input-path",
                str(emg_epochs),
                "--input-type",
                "npy",
                "--raw-fs",
                str(args.target_fs),
                "--target-fs",
                str(args.target_fs),
                "--target-n",
                str(target_n),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(noise_dir),
                "--output-prefix",
                f"emg_all_{sid}",
            ]
        )

        run_cmd(
            [
                py,
                verify_main,
                "--emg-npy",
                str(emg_all),
                "--eog-npy",
                str(eog_all),
                "--clean-npy",
                str(clean_npy),
            ]
        )

        run_cmd(
            [
                py,
                contam_main,
                "--mode",
                "stfnet",
                "--combo",
                args.combo,
                "--clean-edf",
                str(edf),
                "--emg-npy",
                str(emg_all),
                "--eog-npy",
                str(eog_all),
                "--out-contaminated-npy",
                str(contaminated_out),
                "--out-clean-npy",
                str(pure_out),
                "--seed",
                str(args.seed),
            ]
        )

    print("\nAll EDF files processed.")
    print(f"Final contaminated outputs: {contaminated_root}")


if __name__ == "__main__":
    main()
