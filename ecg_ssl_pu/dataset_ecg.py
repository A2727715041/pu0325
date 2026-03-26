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


class ECGSSLPUTrainDataset(Dataset):
    """
    训练集：
        X: [N, 1, L]
        y_pu: [N]，取值 1（正类 AF）或 -1（未标记）
    来自 prepare_afdb_dataset.py 生成的 afdb_train.mat
    """

    def __init__(self, mat_path, transform=None):
        data = loadmat(mat_path)
        self.X = data["X"].astype(np.float32)
        self.y_pu = data["y_pu"].astype(np.int32).reshape(-1)
        assert self.X.shape[0] == self.y_pu.shape[0]
        self.transform = transform if transform is not None else ECGTransform()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [1, L]
        y = self.y_pu[idx]
        x1, x2 = self.transform(x)
        return (
            torch.from_numpy(x.astype("float32")),
            torch.from_numpy(x1),
            torch.from_numpy(x2),
            torch.tensor(y, dtype=torch.int64),
        )


class ECGTestDataset(Dataset):
    """
    测试/验证集：
        X: [N, 1, L]
        y: [N]，真实 AF/非AF 标签（0/1）
    来自 afdb_test.mat
    """

    def __init__(self, mat_path):
        data = loadmat(mat_path)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.int64).reshape(-1)
        assert self.X.shape[0] == self.y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)



