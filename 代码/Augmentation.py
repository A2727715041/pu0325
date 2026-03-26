# === 文件：Augmentation.py（增强方式扩展）===
import torch  # 导入PyTorch深度学习框架
from torch.utils.data import Dataset  # 导入Dataset基类，用于自定义数据集
import numpy as np  # 导入NumPy，用于数值计算
from data_loader import load_processed_mat  # 从数据加载模块导入处理mat文件的函数

class ECGMatDataset(Dataset):
    # 新增参数 start_idx 和 end_idx，用于指定数据范围（默认加载全部）
    def __init__(self, file_path, start_idx=0, end_idx=None, labeled=False):
        self.signals, _ = load_processed_mat(file_path, labeled=labeled)
        self.signals = self.signals.astype(np.float32)

        # 根据 start_idx 和 end_idx 切片，保留指定范围的数据
        if end_idx is None:
            self.signals = self.signals[start_idx:]  # 若未指定end_idx，从start_idx取到最后
        else:
            self.signals = self.signals[start_idx:end_idx]  # 取 [start_idx, end_idx) 范围的数据

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        return torch.tensor(x).unsqueeze(0)


# 基础增强（论文一致：高斯噪声 + 区段mask）
def ecg_augment_pair(x_np):
    def augment(x):  # 内部增强函数
        # 1. 添加高斯噪声：均值0，标准差0.1，形状与原信号一致
        x_aug = x + np.random.normal(0, 0.1, x.shape)
        # 计算mask长度为信号长度的15%
        mask_len = int(len(x_aug) * 0.15)
        # 随机选择mask的起始位置（确保不超出边界）
        start = np.random.randint(0, len(x_aug) - mask_len)
        # 将选中区段置为0（mask操作）
        x_aug[start:start+mask_len] = 0
        return x_aug

    # 对同一输入生成两个不同的增强版本
    x1 = augment(x_np)
    x2 = augment(x_np)
    # 转换为张量并增加通道维度后返回
    return torch.tensor(x1).unsqueeze(0), torch.tensor(x2).unsqueeze(0)

# 增强升级版（加上平移 + 缩放）
def ecg_augment_pair_advanced(x_np):
    def augment(x):  # 内部增强函数
        # 1. 高斯噪声
        x_aug = x + np.random.normal(0, 0.1, x.shape)

        # 2. 区段mask：mask长度为信号长度的10%
        mask_len = int(len(x_aug) * 0.1)
        start = np.random.randint(0, len(x_aug) - mask_len)
        x_aug[start:start+mask_len] = 0

        # 3. 时间平移：在-100到100之间随机选择平移量，循环平移
        shift = np.random.randint(-100, 100)
        x_aug = np.roll(x_aug, shift)

        # 4. 幅度缩放：在0.8到1.2之间随机选择缩放因子
        scale = np.random.uniform(0.8, 1.2)
        x_aug *= scale

        return x_aug

    # 生成两个不同的增强版本
    x1 = augment(x_np)
    x2 = augment(x_np)
    # 转换为张量并增加通道维度后返回
    return torch.tensor(x1).unsqueeze(0), torch.tensor(x2).unsqueeze(0)

# 训练时可在 main.py 中切换调用哪个版本：
# ecg_augment_pair(sig[0])  或  ecg_augment_pair_advanced(sig[0])