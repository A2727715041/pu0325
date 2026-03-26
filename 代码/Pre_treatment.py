import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
import pywt
from scipy.io import savemat

# 滤波器设计函数
def design_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def design_notch(f0, Q, fs):
    w0 = f0 / (0.5 * fs)
    b, a = butter(2, [w0 - 0.01, w0 + 0.01], btype='bandstop')
    return b, a


# 预处理流程
def ecg_preprocessing(signal, fs=400):
    # 带通滤波 (0.05-100Hz)
    b, a = design_bandpass(0.05, 100, fs, order=4)
    filtered = filtfilt(b, a, signal)

    # 50Hz陷波滤波
    b_notch, a_notch = design_notch(50, 30, fs)
    filtered = filtfilt(b_notch, a_notch, filtered)

    # 去除基线漂移 (Savitzky-Golay)
    window_size = int(1.0 * fs)  # 1秒窗口
    if window_size % 2 == 0: window_size += 1  # 必须为奇数
    baseline = savgol_filter(filtered, window_size, 3)
    filtered = filtered - baseline

    # 小波去噪
    coeffs = pywt.wavedec(filtered, 'sym8', level=9)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(filtered)))
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    filtered = pywt.waverec(coeffs, 'sym8')

    return filtered[:len(signal)]  # 保持原长度


# 数据加载与处理
def process_dataset(file_path, output_path, labeled=True):
    with h5py.File(file_path, 'r') as f:
        signals = np.array(f['traindata' if 'train' in file_path else 'testdata'][:]).T

    processed_signals = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        processed_signals[i] = ecg_preprocessing(signals[i])

        # 绘制示例对比图
        if i < 10:  # 保存前10条信号的对比图
            plt.figure(figsize=(12, 6))
            plt.subplot(211)
            plt.plot(signals[i], 'b')
            plt.title(f'Raw ECG Signal (Sample {i + 1})')
            plt.subplot(212)
            plt.plot(processed_signals[i], 'r')
            plt.title(f'Processed ECG Signal (Sample {i + 1})')
            plt.tight_layout()
            plt.savefig(f'D:/CPSC2025/processing_sample_{i + 1}.png')
            plt.close()

    # 使用scipy保存为MATLAB兼容格式
    savemat(output_path,
           {'processed_traindata' if 'train' in file_path else 'processed_testdata': processed_signals},
           do_compression=True)

    return processed_signals


if __name__ == "__main__":
    # 路径配置
    train_path = 'D:\CPSC2025/traindata.mat'
    test_path = 'D:\CPSC2025/testdata.mat'
    output_train = 'D:\CPSC2025/processed_traindata.mat'
    output_test = 'D:\CPSC2025/processed_testdata.mat'

    # 处理训练集和测试集
    print("Processing training data...")
    _ = process_dataset(train_path, output_train)

    print("Processing test data...")
    _ = process_dataset(test_path, output_test, labeled=False)

    print("Preprocessing completed. Results saved to:")
    print(f"- {output_train}\n- {output_test}")
    print("Comparison plots saved to D:/CPSC2025/")