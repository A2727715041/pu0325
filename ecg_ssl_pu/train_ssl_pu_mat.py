import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim
from tqdm import tqdm
import numpy as np

# 兼容直接运行脚本：将工程根目录加入 sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecg_ssl_pu.dataset_mat_ecg import ECGSSLDataSet, ECGTestDataSet
from 代码.xResNet_50 import xResNet50
from ecg_ssl_pu.ssl_loss import pu_aware_ssl_loss
from ecg_ssl_pu.pu_loss_torch import ImbalancedPULoss
from 代码.data_loader_mat import get_dataloaders, get_positive_prior

def prior_corrected_inference(logits, pi_prime, pi_true):
    """
    Prior-corrected inference: 从训练prior (π')修正到真实prior (π)
    """
    # 将logits转换为概率（在π'下的概率）
    probs_under_pi_prime = torch.sigmoid(logits)

    # 方法1：简化线性缩放（适用于π_true << 1的情况）
    # P(y=1|x)_true ≈ (π_true / π') * P(y=1|x)_π'
    corrected_probs = (pi_true / pi_prime) * probs_under_pi_prime
    corrected_probs = torch.clamp(corrected_probs, 0.0, 1.0)

    return corrected_probs

def compute_comprehensive_metrics(y_true, y_pred, y_probs, prior_true=None, pi_prime=None):
    """
    计算全面的评估指标
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        average_precision_score
    )

    metrics = {}

    # 基础指标（基于阈值0.5）
    metrics['accuracy'] = (y_true == y_pred).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 阈值无关指标
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
    except ValueError:
        metrics['roc_auc'] = np.nan

    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_probs)
    except ValueError:
        metrics['pr_auc'] = np.nan

    # Prior-corrected评估（如果提供了prior信息）
    if prior_true is not None and pi_prime is not None:
        # 将概率从π'修正到π_true
        corrected_probs = prior_corrected_inference(
            torch.from_numpy(y_probs).float(),
            pi_prime,
            prior_true
        ).numpy()

        # 使用修正后的概率重新计算预测（阈值0.5）
        corrected_preds = (corrected_probs > 0.5).astype(int)

        metrics['corrected_accuracy'] = (y_true == corrected_preds).mean()
        prec_corr, rec_corr, f1_corr, _ = precision_recall_fscore_support(
            y_true, corrected_preds, labels=[0, 1], average="binary", pos_label=1, zero_division=0
        )
        metrics['corrected_precision'] = prec_corr
        metrics['corrected_recall'] = rec_corr
        metrics['corrected_f1'] = f1_corr

        try:
            metrics['corrected_roc_auc'] = roc_auc_score(y_true, corrected_probs)
        except ValueError:
            metrics['corrected_roc_auc'] = np.nan

        try:
            metrics['corrected_pr_auc'] = average_precision_score(y_true, corrected_probs)
        except ValueError:
            metrics['corrected_pr_auc'] = np.nan

    return metrics

class SSLPUModel(nn.Module):
    """
    基于xResNet50的SSL+PU学习模型
    """
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        # 投影头用于SSL
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(encoder.feature_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, proj_dim)
        )
        # 分类器用于PU学习
        self.classifier = torch.nn.Linear(encoder.feature_dim, 1)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x_raw, x1=None, x2=None):
        # x_raw/x1/x2: [B, 1, L]
        h_raw = self.encode(x_raw)

        if x1 is not None and x2 is not None:
            # SSL模式
            h1 = self.encode(x1)
            h2 = self.encode(x2)
            z1 = F.normalize(self.proj_head(h1), dim=-1)
            z2 = F.normalize(self.proj_head(h2), dim=-1)
            logits = self.classifier(h_raw).squeeze(-1)
            return logits, z1, z2
        else:
            # 仅分类模式
            logits = self.classifier(h_raw).squeeze(-1)
            return logits

def train_ssl_pu_af():
    """
    基于现有.mat数据的SSL+PU学习训练流程
    """
    # ===================== 配置参数 =====================
    # 1. 数据与模型参数
    BATCH_SIZE = 32
    IN_CHANNELS = 1  # 单导联
    ECG_LENGTH = 4000
    POSITIVE_PRIOR = get_positive_prior()  # 从数据计算得到
    PI_PRIME = 0.5  # 目标平衡先验

    # 2. 训练参数
    SSL_EPOCHS = 100  # SSL预训练轮数
    NNPU_EPOCHS = 50  # NNPU微调轮数
    LR_SSL = 1e-3     # SSL学习率
    LR_NNPU = 1e-4    # NNPU学习率
    TEMPERATURE = 0.1 # SSL温度参数
    LAMBDA_PU = 5.0   # PU损失权重

    # 3. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ===================== 数据加载 =====================
    print("\n=== 加载数据 ===")
    ssl_loader, nnpu_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=4
    )

    # ===================== 模型初始化 =====================
    print("\n=== 初始化模型 ===")
    # 使用xResNet50作为编码器
    encoder = xResNet50()
    model = SSLPUModel(encoder, proj_dim=128).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ===================== SSL预训练 =====================
    print("\n=== 开始SSL预训练 ===")
    ssl_loss_fn = pu_aware_ssl_loss
    optimizer_ssl = optim.Adam(model.parameters(), lr=LR_SSL)

    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()
        total_ssl_loss = 0.0

        for batch in tqdm(ssl_loader, desc=f"SSL Epoch {epoch}/{SSL_EPOCHS}"):
            ecg1, ecg2 = batch
            ecg1, ecg2 = ecg1.to(device), ecg2.to(device)

            optimizer_ssl.zero_grad()

            # 前向传播：提取特征
            _, z1, z2 = model(None, ecg1, ecg2)

            # 计算SSL损失
            loss_ssl = ssl_loss_fn(z1, z2, temperature=TEMPERATURE)

            # 反向传播
            loss_ssl.backward()
            optimizer_ssl.step()
            total_ssl_loss += loss_ssl.item()

        avg_ssl_loss = total_ssl_loss / len(ssl_loader)
        print(f"SSL Epoch {epoch}, Loss: {avg_ssl_loss:.4f}")

        # 每20 epoch保存预训练权重
        if (epoch) % 20 == 0:
            checkpoint_path = f"ssl_pretrained_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.encoder.state_dict(),
                'loss': avg_ssl_loss,
            }, checkpoint_path)
            print(f"✓ 已保存SSL预训练权重: {checkpoint_path}")

    # ===================== NNPU微调 =====================
    print("\n=== 开始NNPU微调 ===")
    # 加载最优SSL预训练权重（这里使用最后的权重）
    # 在实际应用中，可以选择验证性能最好的权重

    # 先冻结编码器，仅训练分类头
    for param in model.encoder.parameters():
        param.requires_grad = False

    pu_loss_fn = ImbalancedPULoss(prior=POSITIVE_PRIOR, pi_prime=PI_PRIME, nnpu=True)
    optimizer_nnpu = optim.Adam(model.parameters(), lr=LR_NNPU)

    best_auc = 0.0
    best_model_path = "best_ssl_nnpu.pth"

    for epoch in range(1, NNPU_EPOCHS + 1):
        model.train()
        total_pu_loss = 0.0

        for batch in tqdm(nnpu_loader, desc=f"NNPU Epoch {epoch}/{NNPU_EPOCHS}"):
            ecg, labels = batch
            ecg, labels = ecg.to(device), labels.to(device)

            optimizer_nnpu.zero_grad()

            # 前向传播
            logits = model(ecg)

            # 计算PU损失
            loss_pu = pu_loss_fn(logits, labels)

            # 反向传播
            loss_pu.backward()
            optimizer_nnpu.step()
            total_pu_loss += loss_pu.item()

        avg_pu_loss = total_pu_loss / len(nnpu_loader)

        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for ecg, labels in val_loader:
                ecg = ecg.to(device)
                logits = model(ecg)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.numpy())

        # 计算AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_preds)
        print(f"NNPU Epoch {epoch}, Loss: {avg_pu_loss:.4f}, Val AUC: {val_auc:.4f}")

        # 保存最优模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'auc': best_auc,
            }, best_model_path)
            print(f"✓ 已保存最优模型: {best_model_path} (AUC: {best_auc:.4f})")

    # ===================== 最终测试 =====================
    print("\n=== 最终测试评估 ===")
    # 加载最优模型
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 在测试集上评估
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for ecg, labels in test_loader:
            ecg = ecg.to(device)
            logits = model(ecg)
            probs = torch.sigmoid(logits)
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())

    # 计算综合指标
    test_preds_binary = (np.array(test_preds) > 0.5).astype(int)
    metrics = compute_comprehensive_metrics(
        np.array(test_labels),
        test_preds_binary,
        np.array(test_preds),
        prior_true=POSITIVE_PRIOR,
        pi_prime=PI_PRIME
    )

    print(f"\n=== 测试结果 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"混淆矩阵:\n{metrics['confusion_matrix']}")

    print("\n=== 训练完成 ===")
    print(f"最优模型已保存到: {best_model_path}")

if __name__ == "__main__":
    train_ssl_pu_af()