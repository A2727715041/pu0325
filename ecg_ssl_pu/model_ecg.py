import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGEncoder(nn.Module):
    """
    简单 1D-CNN 心电编码器，可根据需要自行加深或替换为 ResNet1D 等
    """

    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),  # [B, C, 1]
        )
        self.out_dim = base_channels * 4

    def forward(self, x):
        # x: [B, 1, L]
        h = self.net(x)
        h = h.squeeze(-1)  # [B, C]
        return h


class ProjectionHead(nn.Module):
    """
    SimCLR 风格投影头：编码特征 -> 对比学习空间
    """

    def __init__(self, in_dim, hid_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


class SSLPUModel(nn.Module):
    """
    共享编码器：
        - 自监督：两视图 -> 投影向量 z1, z2
        - PU 分类：原始视图 -> 标量打分 f(x)
    """

    def __init__(self, encoder: ECGEncoder, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.proj_head = ProjectionHead(encoder.out_dim, hid_dim=256, out_dim=proj_dim)
        self.classifier = nn.Linear(encoder.out_dim, 1)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x_raw, x1, x2):
        # x_raw/x1/x2: [B, 1, L]
        h_raw = self.encode(x_raw)
        h1 = self.encode(x1)
        h2 = self.encode(x2)

        z1 = self.proj_head(h1)
        z2 = self.proj_head(h2)

        logits = self.classifier(h_raw)  # [B,1]
        return logits.squeeze(-1), z1, z2



