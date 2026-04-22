import math
import numpy as np
import scipy
import torch
from sklearn.model_selection import KFold

from tools import Filter_EEG, Standardization


def DivideDataset(person_num=54, n_splits=10):
    if person_num <= 1:
        ids = np.arange(person_num)
        return [ids], [ids], [ids]

    n_splits = min(n_splits, person_num)
    n_splits = max(2, n_splits)
    x = np.arange(person_num)
    kf = KFold(n_splits=n_splits, shuffle=False)
    train_list, val_list, test_list = [], [], []
    for train_index, test_index in kf.split(x):
        val_count = max(1, int(len(train_index) * 0.1))
        val_idx = train_index[:val_count]
        tr_idx = train_index[val_count:]
        if len(tr_idx) == 0:
            tr_idx = val_idx
        train_list.append(tr_idx)
        val_list.append(val_idx)
        test_list.append(test_index)
    return train_list, val_list, test_list


def _load_numpy_subjects(path):
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected npy shape (S,C,T) or (C,T), got {arr.shape}")
    return arr


def _fix_signal_length(sig, target_len):
    # No padding: keep only available complete content up to target_len.
    return sig[:, : min(sig.shape[1], target_len)]


def _subject_to_eval_windows(sig, win=500, n_win=10):
    # No padding: only keep complete windows.
    n_full = sig.shape[1] // win
    n_use = min(n_full, n_win)
    if n_use <= 0:
        return None
    sig = sig[:, : win * n_use]
    return sig.reshape(sig.shape[0], n_use, win)


def _apply_filter_and_standardize(reference, data_artifact, fs=200):
    for c in range(reference.shape[0]):
        reference[c], data_artifact[c] = Filter_EEG(reference[c], fs=fs), Filter_EEG(
            data_artifact[c], fs=fs
        )
    reference, data_artifact = Standardization(reference, data_artifact)
    return reference, data_artifact


def _load_from_numpy_paths(eeg_path, nos_path, fold):
    return _load_from_numpy_paths_with_splits(eeg_path, nos_path, fold, n_splits=10)


def _split_indices(total_num, fold, n_splits):
    if total_num <= 1:
        ids = np.arange(total_num)
        return ids, ids, ids

    if n_splits < 2:
        # 退化为 80/10/10 固定划分
        ids = np.arange(total_num)
        test_count = max(1, int(total_num * 0.1))
        val_count = max(1, int(total_num * 0.1))
        test_idx = ids[-test_count:]
        val_idx = ids[-(test_count + val_count) : -test_count]
        train_idx = ids[: -(test_count + val_count)] if total_num > (test_count + val_count) else ids
        if len(train_idx) == 0:
            train_idx = val_idx if len(val_idx) > 0 else test_idx
        if len(val_idx) == 0:
            val_idx = train_idx
        return train_idx, val_idx, test_idx

    n_splits = min(max(2, n_splits), total_num)
    kf = KFold(n_splits=n_splits, shuffle=False)
    splits = list(kf.split(np.arange(total_num)))
    train_index, test_index = splits[fold % len(splits)]

    val_count = max(1, int(len(train_index) * 0.1))
    val_idx = train_index[:val_count]
    train_idx = train_index[val_count:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx, test_index


def _filter_standardize_window(reference_w, data_artifact_w):
    reference_w = np.asarray(reference_w, dtype=np.float32).copy()
    data_artifact_w = np.asarray(data_artifact_w, dtype=np.float32).copy()
    for c in range(reference_w.shape[0]):
        reference_w[c], data_artifact_w[c] = Filter_EEG(
            reference_w[c], fs=200
        ), Filter_EEG(data_artifact_w[c], fs=200)
    reference_w, data_artifact_w = Standardization(reference_w, data_artifact_w)
    return reference_w, data_artifact_w


def _load_single_subject_windows(clean_sig, noisy_sig, fold, n_splits):
    min_t = min(clean_sig.shape[1], noisy_sig.shape[1])
    n_windows = min_t // 500
    if n_windows < 3:
        raise ValueError(
            f"Single-subject data only has {n_windows} full windows (need >=3). "
            "Please reduce --folds or use merged multi-subject data."
        )

    clean_win = clean_sig[:, : n_windows * 500].reshape(clean_sig.shape[0], n_windows, 500)
    noisy_win = noisy_sig[:, : n_windows * 500].reshape(noisy_sig.shape[0], n_windows, 500)

    train_idx, val_idx, test_idx = _split_indices(
        total_num=n_windows, fold=fold, n_splits=n_splits
    )

    train_data, train_noisy = [], []
    for w in train_idx:
        ref_w, nos_w = _filter_standardize_window(clean_win[:, w, :], noisy_win[:, w, :])
        train_data.append(ref_w)
        train_noisy.append(nos_w)

    if len(train_data) == 0:
        raise ValueError("No training windows after split in single-subject mode.")

    def build_eval_from_window_ids(ids):
        ref_list, nos_list = [], []
        for w in ids:
            ref_w, nos_w = _filter_standardize_window(
                clean_win[:, w, :], noisy_win[:, w, :]
            )
            ref_list.append(ref_w[:, None, :])  # (C,1,500)
            nos_list.append(nos_w[:, None, :])  # (C,1,500)
        if len(ref_list) == 0:
            raise ValueError("No eval windows after split in single-subject mode.")
        return np.concatenate(ref_list, axis=1), np.concatenate(nos_list, axis=1)

    eeg_val_data, nos_val_data = build_eval_from_window_ids(val_idx)
    eeg_test_data, nos_test_data = build_eval_from_window_ids(test_idx)

    return (
        train_data,
        train_noisy,
        eeg_val_data,
        nos_val_data,
        eeg_test_data,
        nos_test_data,
    )


def _load_from_numpy_paths_with_splits(eeg_path, nos_path, fold, n_splits):
    clean_subjects = _load_numpy_subjects(eeg_path)
    noisy_subjects = _load_numpy_subjects(nos_path)

    person_num = min(clean_subjects.shape[0], noisy_subjects.shape[0])
    clean_subjects = clean_subjects[:person_num]
    noisy_subjects = noisy_subjects[:person_num]

    # 单主体输入时，按时间窗切分进行“窗口级”交叉验证，避免所有 fold 完全一致。
    if person_num == 1:
        return _load_single_subject_windows(
            clean_subjects[0], noisy_subjects[0], fold=fold, n_splits=n_splits
        )

    train_list, val_list, test_list = DivideDataset(
        person_num=person_num, n_splits=n_splits
    )
    use_fold = fold % len(train_list)
    train_id, val_id, test_id = (
        train_list[use_fold],
        val_list[use_fold],
        test_list[use_fold],
    )

    fix_length = 5500
    train_data, train_noisy = [], []
    for n in train_id:
        reference = _fix_signal_length(clean_subjects[n], fix_length)
        data_artifact = _fix_signal_length(noisy_subjects[n], fix_length)
        if reference.shape[1] <= 500 or data_artifact.shape[1] <= 500:
            continue
        reference, data_artifact = _apply_filter_and_standardize(
            reference, data_artifact, fs=200
        )
        train_data.append(reference)
        train_noisy.append(data_artifact)

    if len(train_data) == 0:
        raise ValueError(
            "No valid training subject after no-padding filter. "
            "Need at least one sample with length > 500."
        )

    def build_eval(ids):
        ref_list, nos_list = [], []
        for n in ids:
            reference = _subject_to_eval_windows(clean_subjects[n], win=500, n_win=10)
            data_artifact = _subject_to_eval_windows(
                noisy_subjects[n], win=500, n_win=10
            )
            if reference is None or data_artifact is None:
                continue
            for c in range(reference.shape[0]):
                for w in range(reference.shape[1]):
                    reference[c, w] = Filter_EEG(reference[c, w], fs=200)
                    data_artifact[c, w] = Filter_EEG(data_artifact[c, w], fs=200)
            for w in range(reference.shape[1]):
                ref_w = reference[:, w, :]
                nos_w = data_artifact[:, w, :]
                ref_w, nos_w = Standardization(ref_w, nos_w)
                reference[:, w, :] = ref_w
                data_artifact[:, w, :] = nos_w
            ref_list.append(reference)
            nos_list.append(data_artifact)

        if len(ref_list) == 0:
            raise ValueError(
                "No valid eval/test window after no-padding filter. "
                "Need at least one subject with length >= 500."
            )

        ref_cat = np.concatenate(ref_list, axis=1)
        nos_cat = np.concatenate(nos_list, axis=1)
        return ref_cat, nos_cat

    eeg_val_data, nos_val_data = build_eval(val_id)
    eeg_test_data, nos_test_data = build_eval(test_id)

    return (
        train_data,
        train_noisy,
        eeg_val_data,
        nos_val_data,
        eeg_test_data,
        nos_test_data,
    )


def LoadEEGData(eeg_path, nos_path, fold, n_splits=10):
    if eeg_path.lower().endswith(".npy") and nos_path.lower().endswith(".npy"):
        return _load_from_numpy_paths_with_splits(
            eeg_path, nos_path, fold, n_splits=n_splits
        )

    data_signals = scipy.io.loadmat(eeg_path)
    noisy_artifact = scipy.io.loadmat(nos_path)

    del (
        data_signals["__header__"],
        data_signals["__version__"],
        data_signals["__globals__"],
    )
    del (
        noisy_artifact["__header__"],
        noisy_artifact["__version__"],
        noisy_artifact["__globals__"],
    )

    person_num = len(
        [
            k
            for k in data_signals.keys()
            if k.startswith("sim") and k.endswith("_resampled")
        ]
    )
    if person_num <= 0:
        person_num = 54

    train_list, val_list, test_list = DivideDataset(person_num=person_num, n_splits=10)
    use_fold = fold % len(train_list)
    train_id, val_id, test_id = (
        train_list[use_fold],
        val_list[use_fold],
        test_list[use_fold],
    )

    fix_length = 5500
    train_data, train_noisy = [], []
    for n in train_id:
        n = n + 1
        reference = data_signals[f"sim{n}_resampled"]
        data_artifact = noisy_artifact[f"sim{n}_con"]
        shapes = data_artifact.shape
        if shapes[1] < fix_length:
            data_artifact = np.concatenate(
                [data_artifact, torch.zeros(size=(shapes[0], fix_length - shapes[1]))],
                axis=1,
            )
            reference = np.concatenate(
                [reference, torch.zeros(size=(shapes[0], fix_length - shapes[1]))],
                axis=1,
            )
        else:
            data_artifact = data_artifact[:, :fix_length]
            reference = reference[:, :fix_length]

        for c in range(reference.shape[0]):
            reference[c], data_artifact[c] = Filter_EEG(
                reference[c], fs=200
            ), Filter_EEG(data_artifact[c], fs=200)
        reference, data_artifact = Standardization(reference, data_artifact)

        train_data.append(reference)
        train_noisy.append(data_artifact)

    eeg_train_data, nos_train_data = train_data, train_noisy

    val_data, val_noisy = np.zeros((19, 10 * len(val_id), 500)), np.zeros(
        (19, 10 * len(val_id), 500)
    )
    for i in range(len(val_id)):
        n = val_id[i] + 1
        reference = data_signals[f"sim{n}_resampled"]
        data_artifact = noisy_artifact[f"sim{n}_con"]

        reference, data_artifact = reference[:, 0:5000], data_artifact[:, 0:5000]

        for c in range(reference.shape[0]):
            reference[c], data_artifact[c] = Filter_EEG(
                reference[c], fs=200
            ), Filter_EEG(data_artifact[c], fs=200)
        reference, data_artifact = Standardization(reference, data_artifact)

        reference, data_artifact = reference.reshape(
            19, -1, 500
        ), data_artifact.reshape(19, -1, 500)
        val_data[:, i * 10 : (i + 1) * 10, :] = reference
        val_noisy[:, i * 10 : (i + 1) * 10, :] = data_artifact

    eeg_val_data, nos_val_data = val_data, val_noisy

    test_data, test_noisy = np.zeros((19, 10 * len(test_id), 500)), np.zeros(
        (19, 10 * len(test_id), 500)
    )
    for i in range(len(test_id)):
        n = test_id[i] + 1
        reference = data_signals[f"sim{n}_resampled"]
        data_artifact = noisy_artifact[f"sim{n}_con"]

        reference, data_artifact = reference[:, 0:5000], data_artifact[:, 0:5000]

        for c in range(reference.shape[0]):
            reference[c], data_artifact[c] = Filter_EEG(
                reference[c], fs=200
            ), Filter_EEG(data_artifact[c], fs=200)
        reference, data_artifact = Standardization(reference, data_artifact)

        reference, data_artifact = reference.reshape(
            19, -1, 500
        ), data_artifact.reshape(19, -1, 500)
        test_data[:, i * 10 : (i + 1) * 10, :] = reference
        test_noisy[:, i * 10 : (i + 1) * 10, :] = data_artifact

    eeg_test_data, nos_test_data = test_data, test_noisy
    return (
        eeg_train_data,
        nos_train_data,
        eeg_val_data,
        nos_val_data,
        eeg_test_data,
        nos_test_data,
    )


class GetEEGData:
    def __init__(self, EEG_data, NOS_data, batch_size=128):
        self.EEG_data, self.NOS_data = EEG_data, NOS_data
        self.batch_size = batch_size

    def len(self):
        return math.ceil(self.EEG_data.shape[1] / self.batch_size)

    def get_item(self, item):
        eeg_data = self.EEG_data[:, item, :]
        nos_data = self.NOS_data[:, item, :]
        return nos_data, eeg_data

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min(
            (batch_id + 1) * self.batch_size, self.EEG_data.shape[1]
        )
        eeg_nos_batch, eeg_batch = [], []
        for item in range(start_id, end_id):
            eeg_nos_data, eeg_data = self.get_item(item)
            eeg_nos_batch.append(eeg_nos_data)
            eeg_batch.append(eeg_data)
        eeg_nos_batch, eeg_batch = np.array(eeg_nos_batch), np.array(eeg_batch)
        return eeg_nos_batch, eeg_batch


class GetEEGData_train:
    def __init__(self, EEG_data, NOS_data, batch_size=128, device="cuda:0"):
        self.device = device
        self.EEG_list = torch.Tensor(
            np.concatenate((np.array(EEG_data), np.array(EEG_data)), axis=0)
        ).to(self.device)
        self.NOS_list = torch.Tensor(
            np.concatenate((np.array(NOS_data), np.array(EEG_data)), axis=0)
        ).to(self.device)
        self.batch_size = batch_size
        self.random_shuffle()

    def len(self):
        return math.floor(self.start_point_idxs.shape[0] / self.batch_size)

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min(
            (batch_id + 1) * self.batch_size, self.start_point_idxs.shape[0]
        )
        start_point_batch = self.start_point_idxs[start_id:end_id]
        sample_batch = self.sample_idxs[start_id:end_id]
        eeg_samples = self.EEG_list[sample_batch, ...]
        nos_samples = self.NOS_list[sample_batch, ...]
        gather_idx = (
            torch.Tensor(np.array(list(range(500))))
            .to(self.device)
            .long()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gather_idx = gather_idx.repeat((eeg_samples.shape[0], eeg_samples.shape[1], 1))
        gather_idx = gather_idx + start_point_batch.unsqueeze(-1).unsqueeze(-1)
        eeg_batch = eeg_samples.gather(index=gather_idx, dim=2)
        eeg_nos_batch = nos_samples.gather(index=gather_idx, dim=2)
        return eeg_nos_batch, eeg_batch

    def random_shuffle(self):
        num_per_epoch_sample = 100
        self.start_point_idxs, self.sample_idxs = [], []
        for i in range(self.EEG_list.shape[0]):
            max_start = int(self.EEG_list.shape[2] - 500)
            if max_start <= 0:
                # 当样本恰好 500 点时仍可训练，起点固定为 0
                idx = np.zeros((1,), dtype=np.int64)
            else:
                idx = np.random.permutation(max_start)[:num_per_epoch_sample]
            self.start_point_idxs.append(idx)
            self.sample_idxs.append(np.zeros(shape=(idx.shape[0],), dtype=np.int64) + i)
        self.start_point_idxs = np.concatenate(self.start_point_idxs, axis=0)
        self.sample_idxs = np.concatenate(self.sample_idxs, axis=0)
        shuffle = np.random.permutation(self.start_point_idxs.shape[0])
        self.start_point_idxs = (
            torch.Tensor(self.start_point_idxs[shuffle]).to(self.device).long()
        )
        self.sample_idxs = (
            torch.Tensor(self.sample_idxs[shuffle]).to(self.device).long()
        )
