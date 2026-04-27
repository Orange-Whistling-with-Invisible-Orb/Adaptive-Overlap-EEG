import sklearn.model_selection as ms
import numpy as np
import scipy.io as sio
import math

# Author: Haoming Zhang
# The code here not only include data importing, but also data standardization and the generation of analog noise signals
# 作者: Haoming Zhang
# 本文件包含数据导入、标准化以及仿真噪声信号的生成，用于训练去噪模型。


def get_rms(records):
    """
    Calculate the root mean square (RMS) of the input sequence.
    用途: 用于按 RMS 比例缩放噪声以达到期望的 SNR。

    计算输入序列的均方根（RMS）。
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def random_signal(signal, combin_num):
    """
    Randomly disturb and augment the signal set.
    - signal: ndarray with shape (N, T)
    - combin_num: number of augmentations (returns combin_num shuffled sets)
    Returns: ndarray of shape (combin_num, N, T)

    随机打乱并扩增信号集合。
    - signal: ndarray，形状 (N, T)
    - combin_num: 扩增次数（返回 combin_num 个打乱后的集合）
    返回: ndarray，shape (combin_num, N, T)
    """
    random_result = []

    for i in range(combin_num):
        # Randomly permute sample indices
        # 对样本行做随机置换
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)

    random_result = np.array(random_result)
    return random_result


def prepare_data(EEG_all, noise_all, combin_num, train_per, noise_type):
    """
    Prepare training, validation and test data by combining EEG and noise segments.

    Steps (English):
      1. Randomize samples
      2. Align EEG and noise counts depending on noise_type (EMG reuse or EOG truncate)
      3. Split into train/validation/test
      4. For multiple SNRs, scale noise by RMS and add to EEG
      5. Normalize each epoch by the std of the noisy signal

    中文说明:
      1) 随机重排样本
      2) 根据噪声类型调整样本数（EMG 重用 EEG，EOG 截断 EEG）
      3) 划分 train/val/test
      4) 在多个 SNR 下将噪声按 RMS 缩放并与 EEG 相加
      5) 对每个 epoch 按带噪信号的 std 做标准化

    返回:
      noiseEEG_train_end_standard, EEG_train_end_standard,
      noiseEEG_val_end_standard, EEG_val_end_standard,
      noiseEEG_test_end_standard, EEG_test_end_standard, std_VALUE
    """

    # 1) Initial randomization of inputs (only once)
    # 1) 将输入打乱（仅一次）并去除冗余维度
    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    noise_all_random = np.squeeze(random_signal(signal=noise_all, combin_num=1))

    # 2) Balance sample counts according to noise type
    # 2) 根据噪声类型平衡样本数量
    if noise_type == 'EMG':
        # For EMG: reuse some EEG segments to match the larger number of EMG noise segments
        # EMG 噪声段可能比 EEG 多，复用部分 EEG 段以匹配数量
        reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
        EEG_reuse = EEG_all_random[0:reuse_num, :]
        EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
        print('EEG segments after reuse: ', EEG_all_random.shape[0])

    elif noise_type == 'EOG':
        # For EOG: truncate EEG to match the fewer EOG noise segments
        # EOG 噪声段可能较少，裁剪 EEG 到噪声段数量
        EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]
        print('EEG segments after drop: ', EEG_all_random.shape[0])

    # 3) Split into train/validation/test sets
    # 3) 划分样本数量
    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    validation_num = round((EEG_all_random.shape[0] - train_num) / 2)

    train_eeg = EEG_all_random[0:train_num, :]
    validation_eeg = EEG_all_random[train_num:train_num + validation_num, :]
    test_eeg = EEG_all_random[train_num + validation_num:EEG_all_random.shape[0], :]

    train_noise = noise_all_random[0:train_num, :]
    validation_noise = noise_all_random[train_num:train_num + validation_num, :]
    test_noise = noise_all_random[train_num + validation_num:noise_all_random.shape[0], :]

    # 4) Data augmentation for training set: generate combin_num shuffled copies and stack them
    # 4) 对训练集做 combin_num 倍的数据增强（每次打乱后堆叠）
    EEG_train = random_signal(signal=train_eeg, combin_num=combin_num).reshape(combin_num * train_eeg.shape[0], timepoint)
    NOISE_train = random_signal(signal=train_noise, combin_num=combin_num).reshape(combin_num * train_noise.shape[0], timepoint)

    #################################  Simulate noise for training set ##############################
    #################################  模拟训练集噪声 ##############################

    # Generate random SNR for each training sample (in dB)
    # 生成训练集每个样本的随机 SNR（单位 dB）
    SNR_train_dB = np.random.uniform(-7, 2, (EEG_train.shape[0]))
    print(SNR_train_dB.shape)
    SNR_train = 10 ** (0.1 * (SNR_train_dB))

    noiseEEG_train = []
    NOISE_train_adjust = []
    for i in range(EEG_train.shape[0]):
        eeg = EEG_train[i].reshape(EEG_train.shape[1])
        noise = NOISE_train[i].reshape(NOISE_train.shape[1])

        # Scale the noise according to the target SNR
        # 根据目标 SNR 缩放噪声
        coe = get_rms(eeg) / (get_rms(noise) * SNR_train[i])
        noise = noise * coe
        neeg = noise + eeg

        NOISE_train_adjust.append(noise)
        noiseEEG_train.append(neeg)

    noiseEEG_train = np.array(noiseEEG_train)
    NOISE_train_adjust = np.array(NOISE_train_adjust)

    # Normalize each epoch by the std of the noisy signal (helps training convergence)
    # 按每个带噪样本的标准差对数据进行标准化（方便模型训练）
    EEG_train_end_standard = []
    noiseEEG_train_end_standard = []
    for i in range(noiseEEG_train.shape[0]):
        eeg_train_all_std = EEG_train[i] / np.std(noiseEEG_train[i])
        EEG_train_end_standard.append(eeg_train_all_std)

        noiseeeg_train_end_standard = noiseEEG_train[i] / np.std(noiseEEG_train[i])
        noiseEEG_train_end_standard.append(noiseeeg_train_end_standard)

    noiseEEG_train_end_standard = np.array(noiseEEG_train_end_standard)
    EEG_train_end_standard = np.array(EEG_train_end_standard)
    print('training data prepared', noiseEEG_train_end_standard.shape, EEG_train_end_standard.shape)

    #################################  Simulate noise for validation set ##############################
    #################################  模拟验证集噪声 ##############################

    SNR_val_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_val = 10 ** (0.1 * (SNR_val_dB))

    eeg_val = np.array(validation_eeg)
    noise_val = np.array(validation_noise)

    # For each SNR value, combine validation samples with scaled noise
    # 对每个 SNR 值重复组合验证集样本
    EEG_val = []
    noise_EEG_val = []
    for i in range(10):
        noise_eeg_val = []
        for j in range(eeg_val.shape[0]):
            eeg = eeg_val[j]
            noise = noise_val[j]

            coe = get_rms(eeg) / (get_rms(noise) * SNR_val[i])
            noise = noise * coe
            neeg = noise + eeg

            noise_eeg_val.append(neeg)

        EEG_val.extend(eeg_val)
        noise_EEG_val.extend(noise_eeg_val)

    noise_EEG_val = np.array(noise_EEG_val)
    EEG_val = np.array(EEG_val)

    # Standardize each noisy validation sample by its std
    # 按每个带噪样本的标准差对数据做标准化
    EEG_val_end_standard = []
    noiseEEG_val_end_standard = []
    for i in range(noise_EEG_val.shape[0]):
        std_value = np.std(noise_EEG_val[i])

        eeg_val_all_std = EEG_val[i] / std_value
        EEG_val_end_standard.append(eeg_val_all_std)

        noiseeeg_val_end_standard = noise_EEG_val[i] / std_value
        noiseEEG_val_end_standard.append(noiseeeg_val_end_standard)

    noiseEEG_val_end_standard = np.array(noiseEEG_val_end_standard)
    EEG_val_end_standard = np.array(EEG_val_end_standard)
    print('validation data prepared, validation data shape: ', noiseEEG_val_end_standard.shape, EEG_val_end_standard.shape)

    #################################  Simulate noise for test set ##############################
    #################################  模拟测试集噪声 ##############################

    SNR_test_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_test = 10 ** (0.1 * (SNR_test_dB))

    eeg_test = np.array(test_eeg)
    noise_test = np.array(test_noise)

    EEG_test = []
    noise_EEG_test = []
    for i in range(10):
        noise_eeg_test = []
        for j in range(eeg_test.shape[0]):
            eeg = eeg_test[j]
            noise = noise_test[j]

            # Scale the noise according to the target SNR
            # 根据目标 SNR 缩放噪声
            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[i])
            noise = noise * coe
            neeg = noise + eeg

            noise_eeg_test.append(neeg)

        EEG_test.extend(eeg_test)
        noise_EEG_test.extend(noise_eeg_test)

    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)

    # Save std for each test sample (may be used to restore original scale)
    # 测试集同时保存每个样本的 std，用于可能的尺度恢复
    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    std_VALUE = []
    for i in range(noise_EEG_test.shape[0]):
        std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(std_value)

        eeg_test_all_std = EEG_test[i] / std_value
        EEG_test_end_standard.append(eeg_test_all_std)

        noiseeeg_test_end_standard = noise_EEG_test[i] / std_value
        noiseEEG_test_end_standard.append(noiseeeg_test_end_standard)

    std_VALUE = np.array(std_VALUE)
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)
    print('test data prepared, test data shape: ', noiseEEG_test_end_standard.shape, EEG_test_end_standard.shape)

    return noiseEEG_train_end_standard, EEG_train_end_standard, noiseEEG_val_end_standard, EEG_val_end_standard, noiseEEG_test_end_standard, EEG_test_end_standard, std_VALUE
import sklearn.model_selection as ms
import numpy as np
import scipy.io as sio
import math

# 作者: Haoming Zhang
# 本文件包含数据导入、标准化以及仿真噪声信号的生成，用于训练去噪模型。


def get_rms(records):
    """
    计算输入序列的均方根（RMS）。
    用于按 RMS 比例缩放噪声以达到期望的 SNR。
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def random_signal(signal, combin_num):
    """
    随机打乱并扩增信号集合。
    - signal: ndarray，形状 (N, T)
    - combin_num: 扩增次数（返回 combin_num 个打乱后的集合）
    返回: ndarray，shape (combin_num, N, T)
    """
    random_result = []

    for i in range(combin_num):
        # 对样本行做随机置换
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)

    random_result = np.array(random_result)
    return random_result


def prepare_data(EEG_all, noise_all, combin_num, train_per, noise_type):
    """
    生成训练/验证/测试数据：
    1) 随机重排样本
    2) 根据噪声类型调整样本数（EMG 重用 EEG，EOG 截断 EEG）
    3) 划分 train/val/test
    4) 在多个 SNR 下将噪声按 RMS 缩放并与 EEG 相加
    5) 对每个 epoch 按带噪信号的 std 做标准化

    返回值见函数末尾。
    """

    # 将输入打乱（仅一次）并去除冗余维度
    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    noise_all_random = np.squeeze(random_signal(signal=noise_all, combin_num=1))

    # 根据噪声类型平衡样本数量
    if noise_type == 'EMG':
        # EMG 噪声段可能比 EEG 多，复用部分 EEG 段以匹配数量
        reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
        EEG_reuse = EEG_all_random[0:reuse_num, :]
        EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
        print('EEG segments after reuse: ', EEG_all_random.shape[0])

    elif noise_type == 'EOG':
        # EOG 噪声段可能较少，裁剪 EEG 到噪声段数量
        EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]
        print('EEG segments after drop: ', EEG_all_random.shape[0])

    # 划分样本数量
    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    validation_num = round((EEG_all_random.shape[0] - train_num) / 2)

    train_eeg = EEG_all_random[0:train_num, :]
    validation_eeg = EEG_all_random[train_num:train_num + validation_num, :]
    test_eeg = EEG_all_random[train_num + validation_num:EEG_all_random.shape[0], :]

    train_noise = noise_all_random[0:train_num, :]
    validation_noise = noise_all_random[train_num:train_num + validation_num, :]
    test_noise = noise_all_random[train_num + validation_num:noise_all_random.shape[0], :]

    # 对训练集做 combin_num 倍的数据增强（每次打乱后堆叠）
    EEG_train = random_signal(signal=train_eeg, combin_num=combin_num).reshape(combin_num * train_eeg.shape[0], timepoint)
    NOISE_train = random_signal(signal=train_noise, combin_num=combin_num).reshape(combin_num * train_noise.shape[0], timepoint)

    #################################  模拟训练集噪声 ##############################

    # 生成训练集每个样本的随机 SNR（单位 dB）
    SNR_train_dB = np.random.uniform(-7, 2, (EEG_train.shape[0]))
    print(SNR_train_dB.shape)
    SNR_train = 10 ** (0.1 * (SNR_train_dB))

    noiseEEG_train = []
    NOISE_train_adjust = []
    for i in range(EEG_train.shape[0]):
        eeg = EEG_train[i].reshape(EEG_train.shape[1])
        noise = NOISE_train[i].reshape(NOISE_train.shape[1])

        # 根据目标 SNR 缩放噪声
        coe = get_rms(eeg) / (get_rms(noise) * SNR_train[i])
        noise = noise * coe
        neeg = noise + eeg

        NOISE_train_adjust.append(noise)
        noiseEEG_train.append(neeg)

    noiseEEG_train = np.array(noiseEEG_train)
    NOISE_train_adjust = np.array(NOISE_train_adjust)

    # 按每个带噪样本的标准差对数据进行标准化（方便模型训练）
    EEG_train_end_standard = []
    noiseEEG_train_end_standard = []
    for i in range(noiseEEG_train.shape[0]):
        eeg_train_all_std = EEG_train[i] / np.std(noiseEEG_train[i])
        EEG_train_end_standard.append(eeg_train_all_std)

        noiseeeg_train_end_standard = noiseEEG_train[i] / np.std(noiseEEG_train[i])
        noiseEEG_train_end_standard.append(noiseeeg_train_end_standard)

    noiseEEG_train_end_standard = np.array(noiseEEG_train_end_standard)
    EEG_train_end_standard = np.array(EEG_train_end_standard)
    print('training data prepared', noiseEEG_train_end_standard.shape, EEG_train_end_standard.shape)

    #################################  模拟验证集噪声 ##############################

    SNR_val_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_val = 10 ** (0.1 * (SNR_val_dB))

    eeg_val = np.array(validation_eeg)
    noise_val = np.array(validation_noise)

    EEG_val = []
    noise_EEG_val = []
    for i in range(10):
        noise_eeg_val = []
        for j in range(eeg_val.shape[0]):
            eeg = eeg_val[j]
            noise = noise_val[j]

            coe = get_rms(eeg) / (get_rms(noise) * SNR_val[i])
            noise = noise * coe
            neeg = noise + eeg

            noise_eeg_val.append(neeg)

        EEG_val.extend(eeg_val)
        noise_EEG_val.extend(noise_eeg_val)

    noise_EEG_val = np.array(noise_EEG_val)
    EEG_val = np.array(EEG_val)

    EEG_val_end_standard = []
    noiseEEG_val_end_standard = []
    for i in range(noise_EEG_val.shape[0]):
        std_value = np.std(noise_EEG_val[i])

        eeg_val_all_std = EEG_val[i] / std_value
        EEG_val_end_standard.append(eeg_val_all_std)

        noiseeeg_val_end_standard = noise_EEG_val[i] / std_value
        noiseEEG_val_end_standard.append(noiseeeg_val_end_standard)

    noiseEEG_val_end_standard = np.array(noiseEEG_val_end_standard)
    EEG_val_end_standard = np.array(EEG_val_end_standard)
    print('validation data prepared, validation data shape: ', noiseEEG_val_end_standard.shape, EEG_val_end_standard.shape)

    #################################  模拟测试集噪声 ##############################

    SNR_test_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_test = 10 ** (0.1 * (SNR_test_dB))

    eeg_test = np.array(test_eeg)
    noise_test = np.array(test_noise)

    EEG_test = []
    noise_EEG_test = []
    for i in range(10):
        noise_eeg_test = []
        for j in range(eeg_test.shape[0]):
            eeg = eeg_test[j]
            noise = noise_test[j]

            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[i])
            noise = noise * coe
            neeg = noise + eeg

            noise_eeg_test.append(neeg)

        EEG_test.extend(eeg_test)
        noise_EEG_test.extend(noise_eeg_test)

    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)

    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    std_VALUE = []
    for i in range(noise_EEG_test.shape[0]):
        std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(std_value)

        eeg_test_all_std = EEG_test[i] / std_value
        EEG_test_end_standard.append(eeg_test_all_std)

        noiseeeg_test_end_standard = noise_EEG_test[i] / std_value
        noiseEEG_test_end_standard.append(noiseeeg_test_end_standard)

    std_VALUE = np.array(std_VALUE)
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)
    print('test data prepared, test data shape: ', noiseEEG_test_end_standard.shape, EEG_test_end_standard.shape)

    return noiseEEG_train_end_standard, EEG_train_end_standard, noiseEEG_val_end_standard, EEG_val_end_standard, noiseEEG_test_end_standard, EEG_test_end_standard, std_VALUE
  
