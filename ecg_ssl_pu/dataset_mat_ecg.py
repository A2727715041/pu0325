import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import random


class ECGTransform:
    """
    用于自监督的 1D ECG 增强：加噪、缩放、时间遮挡
    """

    def __init__(self, jitter_std=0.02, scale_std=0.1, drop_prob=0.2):
        self.jitter_std = jitter_std
        self.scale_std = scale_std
        self.drop_prob = drop_prob

    def _jitter(self, x):
        noise = np.random.normal(0, self.jitter_std, size=x.shape)
        return x + noise

    def _scaling(self, x):
        factor = np.random.normal(1.0, self.scale_std)
        return x * factor

    def _time_mask(self, x, max_ratio=0.1):
        L = x.shape[-1]
        mask_len = int(L * max_ratio)
        if mask_len < 1:
            return x
        start = random.randint(0, L - mask_len)
        x_masked = x.copy()
        x_masked[..., start:start + mask_len] = 0.0
        return x_masked

    def __call__(self, x):
        # x: numpy array, shape [1, L]
        x1 = x.copy()
        x2 = x.copy()
        # 视图1
        if random.random() < 0.8:
            x1 = self._jitter(x1)
        if random.random() < 0.8:
            x1 = self._scaling(x1)
        if random.random() < self.drop_prob:
            x1 = self._time_mask(x1)
        # 视图2
        if random.random() < 0.8:
            x2 = self._jitter(x2)
        if random.random() < 0.8:
            x2 = self._scaling(x2)
        if random.random() < self.drop_prob:
            x2 = self._time_mask(x2)
        # 显式转换为 float32，避免变成 float64
        return x1.astype("float32"), x2.astype("float32")


class ECGSSLDataSet(Dataset):
    """
    基于现有.mat文件的数据集类
    适配 processed_traindata.mat 和 processed_testdata.mat
    """
    def __init__(self, mat_path, mode="train", task="ssl"):
        """
        Args:
            mat_path: .mat文件路径
            mode: train/val/test
            task: ssl（无监督预训练，用无标签数据）/ nnpu（微调，用P+U数据）/ val（验证，用真实标签）
        """
        # 1. 读取.mat文件
        data = loadmat(mat_path)

        # 确定数据变量名
        if 'train' in mat_path:
            data_key = 'processed_traindata'
        else:
            data_key = 'processed_testdata'

        # 读取ECG数据
        self.ecg_data = data[data_key].astype(np.float32)  # (样本数, 4000)

        # 2. 创建标签（根据实施方案的分布）
        n_samples = self.ecg_data.shape[0]
        self.labels = np.zeros(n_samples, dtype=np.int32)

        if 'train' in mat_path:
            # 训练集标签分布：
            # 0-499: AF（1，正例）
            # 500-999: Non-AF（0，验证用真实负例）
            # 1000-19999: 无标签（-1，未标注）
            self.labels[:500] = 1      # AF
            if n_samples > 500:
                self.labels[500:1000] = 0  # Non-AF
            if n_samples > 1000:
                self.labels[1000:] = -1   # 无标签
        else:
            # 测试集：假设都有真实标签，暂时都设为0，实际使用时会根据真实情况调整
            # 这里可以根据需要修改
            pass

        # 3. 按任务划分数据
        if task == "ssl":
            # SSL预训练：只用无标签数据（标签=-1）
            mask = (self.labels == -1)
            self.ecg_data = self.ecg_data[mask]
            self.labels = self.labels[mask]
            print(f"SSL任务：使用 {len(self.ecg_data)} 个无标签样本")

        elif task == "nnpu":
            # NNPU微调：用P（标签=1）+ U（标签=-1）
            mask = (self.labels == 1) | (self.labels == -1)
            self.ecg_data = self.ecg_data[mask]
            self.labels = self.labels[mask]
            # 将U的标签从-1转为0（NNPU损失要求：P=1，U=0）
            self.labels[self.labels == -1] = 0
            print(f"NNPU任务：使用 {(self.labels == 1).sum()} 个正样本 + {(self.labels == 0).sum()} 个未标记样本")

        elif task == "val":
            # 验证：用P（标签=1）+ Non-AF（标签=0）
            mask = (self.labels == 1) | (self.labels == 0)
            self.ecg_data = self.ecg_data[mask]
            self.labels = self.labels[mask]
            print(f"验证任务：使用 {(self.labels == 1).sum()} 个AF样本 + {(self.labels == 0).sum()} 个Non-AF样本")

        # 4. 适配xResNet输入：(样本数, 1, 4000)（单导联）
        self.ecg_data = self.ecg_data[:, np.newaxis, :]  # 添加通道维度

        # 5. SimCLR数据增强（复用原有逻辑）
        self.augment = (mode == "train") and (task == "ssl")
        if self.augment:
            self.transform = ECGTransform()

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg = self.ecg_data[idx]  # [1, 4000]
        label = self.labels[idx].astype(np.float32)

        if self.augment:
            # SSL双增强
            ecg1, ecg2 = self.transform(ecg)
            return torch.tensor(ecg1), torch.tensor(ecg2)
        else:
            return torch.tensor(ecg), torch.tensor(label)


class ECGTestDataSet(Dataset):
    """
    测试数据集类，用于最终评估
    """
    def __init__(self, mat_path):
        # 读取测试数据
        data = loadmat(mat_path)
        self.ecg_data = data['processed_testdata'].astype(np.float32)

        # 添加通道维度
        self.ecg_data = self.ecg_data[:, np.newaxis, :]  # (10000, 1, 4000)

        # 为测试集创建真实标签（这里需要根据实际情况调整）
        # 暂时使用占位符，实际使用时应该从真实标签文件读取
        self.labels = np.zeros(self.ecg_data.shape[0], dtype=np.int64)

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg = self.ecg_data[idx]  # [1, 4000]
        label = self.labels[idx]
        return torch.tensor(ecg), torch.tensor(label, dtype=torch.long)