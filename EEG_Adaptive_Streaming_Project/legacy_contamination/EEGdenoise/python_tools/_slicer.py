"""
_slicer.py

用途:
- 从 EDF 读取 EEG，重采样，选通道，按固定窗口切片
- 导出 clean epochs 到 .npy（形状: N x T）

示例:
python data_contaminated/_slicer.py \
  --edf-path "data/数据处理（已完成部分ASD）- 标签矫正/01_f0.edf" \
  --target-fs 200 \
  --channels 19 \
  --epoch-len 500 \
  --out-npy "data/generated_pairs/clean_01_f0.npy"
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
	from ._tail_drop_trim import trim_tail_drop_anomaly
except Exception:
	from _tail_drop_trim import trim_tail_drop_anomaly


def edf_to_clean_epochs(
	edf_path: str,
	target_fs: int,
	channels: int,
	epoch_len: int,
	tail_drop_trim: bool = True,
	tail_drop_search_sec: float = 5.0,
	tail_drop_min_persist_sec: float = 0.8,
	tail_drop_rms_ratio: float = 0.25,
	tail_drop_mean_z: float = 6.0,
) -> np.ndarray:
	"""读取 EDF 并输出二维 epochs，shape=(N, epoch_len)。"""
	try:
		import mne
	except ImportError as e:
		raise ImportError("mne is required. Please install it first.") from e

	raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
	raw.resample(target_fs)

	use_ch = min(channels, len(raw.ch_names))
	raw.pick(raw.ch_names[:use_ch])

	x = raw.get_data().astype(np.float32)  # (C, T)
	x, trim_info = trim_tail_drop_anomaly(
		x,
		fs=float(raw.info["sfreq"]),
		enabled=bool(tail_drop_trim),
		search_tail_sec=float(tail_drop_search_sec),
		min_persist_sec=float(tail_drop_min_persist_sec),
		rms_ratio_thresh=float(tail_drop_rms_ratio),
		mean_z_thresh=float(tail_drop_mean_z),
	)
	if trim_info.trimmed:
		print(
			f"[tail-trim] {os.path.basename(edf_path)}: "
			f"cut@{trim_info.cut_index}, removed={trim_info.removed_samples} samples"
		)
	t = (x.shape[1] // epoch_len) * epoch_len
	if t < epoch_len:
		raise ValueError(
			f"Signal too short for one epoch: total={x.shape[1]}, epoch_len={epoch_len}"
		)

	x = x[:, :t]
	# (C, T) -> (C, K, L) -> (K, C, L) -> (K*C, L)
	epochs = x.reshape(x.shape[0], -1, epoch_len).transpose(1, 0, 2).reshape(-1, epoch_len)
	return epochs.astype(np.float32)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Slice EDF into clean EEG epochs (N, T) and export .npy"
	)
	parser.add_argument("--edf-path", required=True, help="Input .edf path")
	parser.add_argument("--target-fs", type=int, default=200, help="Resample frequency")
	parser.add_argument("--channels", type=int, default=19, help="Number of channels to keep")
	parser.add_argument("--epoch-len", type=int, default=500, help="Epoch length in samples")
	parser.add_argument(
		"--disable-tail-drop-trim",
		action="store_true",
		help="Disable auto tail-drop anomaly trimming.",
	)
	parser.add_argument(
		"--tail-drop-search-sec",
		type=float,
		default=5.0,
		help="Only search drop in last N seconds.",
	)
	parser.add_argument(
		"--tail-drop-min-persist-sec",
		type=float,
		default=0.8,
		help="Drop condition must persist at least this duration.",
	)
	parser.add_argument(
		"--tail-drop-rms-ratio",
		type=float,
		default=0.25,
		help="RMS drop ratio threshold vs baseline median.",
	)
	parser.add_argument(
		"--tail-drop-mean-z",
		type=float,
		default=6.0,
		help="Mean drop threshold in robust z-score.",
	)
	parser.add_argument("--out-npy", required=True, help="Output .npy path")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	epochs = edf_to_clean_epochs(
		edf_path=args.edf_path,
		target_fs=args.target_fs,
		channels=args.channels,
		epoch_len=args.epoch_len,
		tail_drop_trim=not args.disable_tail_drop_trim,
		tail_drop_search_sec=args.tail_drop_search_sec,
		tail_drop_min_persist_sec=args.tail_drop_min_persist_sec,
		tail_drop_rms_ratio=args.tail_drop_rms_ratio,
		tail_drop_mean_z=args.tail_drop_mean_z,
	)

	os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
	np.save(args.out_npy, epochs)
	print("saved:", args.out_npy)
	print("clean_epochs", epochs.shape)


if __name__ == "__main__":
	main()
