import numpy as np
import torch
from scipy.signal import butter, filtfilt

EPS = 1e-8


def rrmse_multichannel(predict, truth):
    num = np.sqrt(((predict - truth) ** 2).mean().mean())
    den = np.sqrt((truth**2).mean().mean())
    den = max(den, EPS)
    out = num / den
    return float(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))


def acc_multichannel(predict, truth):
    acc = []
    for i in range(predict.shape[0]):
        p = predict[i]
        t = truth[i]
        # Constant signals make corrcoef undefined; map them to 0 correlation.
        if np.std(p) < EPS or np.std(t) < EPS:
            acc.append(0.0)
            continue
        c = np.corrcoef(p, t)[1, 0]
        acc.append(float(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)))
    return float(np.mean(np.array(acc, dtype=np.float32)))


def cal_SNR_multichannel(predict, truth):
    ps = np.sum(np.sum(np.square(truth), axis=-1), axis=-1)
    pn = np.sum(np.sum(np.square((predict - truth)), axis=-1), axis=-1)
    ratio = (ps + EPS) / (pn + EPS)
    out = 10 * np.log10(ratio)
    return float(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))


def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    ps = np.sum(np.square(truth), axis=-1)
    pn = np.sum(np.square((predict - truth)), axis=-1)
    ratio = (ps + EPS) / (pn + EPS)
    out = 10 * np.log10(ratio)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(out)


def Filter_EEG(eeg, f1=0.5, f2=40, fs=200):
    b1, a1 = butter(5, [f1 / fs * 2, f2 / fs * 2], "bandpass")
    return filtfilt(b1, a1, eeg)


def Standardization(eeg_data, eeg_nos_data):
    for i in range(eeg_data.shape[0]):
        eeg_data[i] = eeg_data[i] - np.mean(eeg_data[i])
        eeg_nos_data[i] = eeg_nos_data[i] - np.mean(eeg_nos_data[i])
        std = np.std(eeg_nos_data[i])
        std = std if std > EPS else 1.0
        eeg_data[i] = eeg_data[i] / std
        eeg_nos_data[i] = eeg_nos_data[i] / std

    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
    eeg_nos_data = np.nan_to_num(eeg_nos_data, nan=0.0, posinf=0.0, neginf=0.0)
    return eeg_data, eeg_nos_data
