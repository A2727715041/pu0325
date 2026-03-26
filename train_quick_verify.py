#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证训练流程 - 小样本验证模式
"""

import os
# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# 导入自定义模块
from ecg_ssl_pu.dataset_mat_ecg import ECGSSLDataSet, ECGTestDataSet
from 代码.xResNet_50 import xResNet50
from ecg_ssl_pu.pu_loss_torch import ImbalancedPULoss

def get_quick_verification_dataloaders():
    """获取快速验证用的极小数据加载器"""
    TRAIN_MAT = "data/processed_traindata.mat"
    TEST_MAT = "data/processed_testdata.mat"

    print("创建快速验证数据加载器...")
    print("使用极小样本集进行快速验证")

    # 创建完整数据集
    ssl_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="ssl")
    nnpu_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="nnpu")
    val_dataset = ECGSSLDataSet(TRAIN_MAT, mode="val", task="val")
    test_dataset = ECGTestDataSet(TEST_MAT)

    # 只取极小样本进行快速验证
    ssl_indices = list(range(50))      # 只取50个SSL样本
    nnpu_indices = list(range(30))     # 只取30个NNPU样本
    val_indices = list(range(20))      # 只取20个验证样本
    test_indices = list(range(20))     # 只取20个测试样本

    # 创建子集
    ssl_subset = Subset(ssl_dataset, ssl_indices)
    nnpu_subset = Subset(nnpu_dataset, nnpu_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    # 极小批次大小
    BATCH_SIZE = 8
    NUM_WORKERS = 0

    ssl_loader = DataLoader(ssl_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    nnpu_loader = DataLoader(nnpu_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"SSL样本数: {len(ssl_subset)}")
    print(f"NNPU样本数: {len(nnpu_subset)}")
    print(f"验证样本数: {len(val_subset)}")
    print(f"测试样本数: {len(test_subset)}")
    print(f"批次大小: {BATCH_SIZE}")

    return ssl_loader, nnpu_loader, val_loader, test_loader

class QuickVerifyModel(nn.Module):
    """快速验证模型"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.encoder = xResNet50()
        self.classifier = nn.Linear(feature_dim, 1)
        self.proj_head = nn.Sequential(
            nn.Linear(feature_dim, 128),  # 减小投影头维度
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)  # 进一步减小
        )

    def forward(self, x_raw=None, x1=None, x2=None):
        if x1 is not None and x2 is not None:
            # SSL模式
            h1 = self.encoder(x1)
            h2 = self.encoder(x2)
            z1 = self.proj_head(h1)
            z2 = self.proj_head(h2)
            if x_raw is not None:
                h_raw = self.encoder(x_raw)
                logits = self.classifier(h_raw).squeeze(-1)
            else:
                logits = None
            return logits, z1, z2
        elif x_raw is not None:
            # 分类模式
            features = self.encoder(x_raw)
            logits = self.classifier(features).squeeze(-1)
            return logits
        else:
            raise ValueError("必须提供x_raw或x1,x2")

def simple_ssl_loss(z1, z2, temperature=0.1):
    """简化的SSL损失函数"""
    # 归一化
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    # 计算相似度矩阵
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2*B, D]

    # 计算余弦相似度
    sim_matrix = torch.mm(z, z.t()) / temperature  # [2*B, 2*B]

    # 创建标签（正样本对）
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # 移除自身相似度
    mask = torch.eye(2 * batch_size).bool()
    labels = labels[~mask].view(2 * batch_size, -1)
    sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)

    # 计算InfoNCE损失
    loss = -torch.log(torch.exp(sim_matrix) / torch.sum(torch.exp(sim_matrix), dim=1, keepdim=True))
    loss = (labels * loss).sum(dim=1) / labels.sum(dim=1)

    return loss.mean()

def train_ssl_phase_quick(model, ssl_loader, device):
    """快速SSL预训练验证"""
    print("\n=== 快速SSL预训练验证 ===")
    print(f"训练轮数: 3")
    print(f"批次数量: {len(ssl_loader)}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 4):  # 只训练3轮
        model.train()
        total_loss = 0.0

        for batch_idx, (ecg1, ecg2) in enumerate(tqdm(ssl_loader, desc=f"SSL Epoch {epoch}")):
            ecg1, ecg2 = ecg1.to(device), ecg2.to(device)

            optimizer.zero_grad()

            # 前向传播
            _, z1, z2 = model(None, ecg1, ecg2)

            # 计算SSL损失
            loss = simple_ssl_loss(z1, z2, temperature=0.1)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        print(f"SSL Epoch {epoch} 完成, 平均损失: {avg_loss:.4f}")

def train_nnpu_phase_quick(model, nnpu_loader, val_loader, device):
    """快速NNPU微调验证"""
    print("\n=== 快速NNPU微调验证 ===")
    print(f"训练轮数: 5")
    print(f"训练批次: {len(nnpu_loader)}")
    print(f"验证批次: {len(val_loader)}")

    # 冻结编码器，只训练分类器
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # PU损失函数
    prior = 0.0256  # 正样本先验
    pu_loss_fn = ImbalancedPULoss(prior=prior, pi_prime=0.5, nnpu=True)

    for epoch in range(1, 6):  # 只训练5轮
        model.train()
        total_loss = 0.0

        # 训练
        for ecg, labels in tqdm(nnpu_loader, desc=f"NNPU Epoch {epoch}"):
            ecg, labels = ecg.to(device), labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            logits = model(ecg)

            # 计算PU损失
            loss = pu_loss_fn(logits, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(nnpu_loader)

        # 快速验证
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for ecg, labels in val_loader:
                ecg, labels = ecg.to(device), labels.to(device)
                logits = model(ecg)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算AUC
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.0

        print(f"NNPU Epoch {epoch}, 损失: {avg_loss:.4f}, 验证AUC: {val_auc:.4f}")

def final_test_quick(model, test_loader, device):
    """快速最终测试"""
    print("\n=== 快速最终测试 ===")

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for ecg, labels in test_loader:
            ecg, labels = ecg.to(device), labels.to(device)
            logits = model(ecg)
            probs = torch.sigmoid(logits)
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # 计算简单指标
    test_preds_binary = (np.array(test_preds) > 0.5).astype(int)
    accuracy = (np.array(test_labels) == test_preds_binary).mean()

    try:
        test_auc = roc_auc_score(test_labels, test_preds)
    except:
        test_auc = 0.0

    print(f"测试样本数: {len(test_preds)}")
    print(f"测试准确率: {accuracy:.4f}")
    print(f"测试AUC: {test_auc:.4f}")

    return accuracy, test_auc

def main():
    """快速验证主函数"""
    print("=== SSL+PU学习快速验证模式 ===")
    print("使用极小样本集验证训练流程")
    print("预计完成时间: 5-10分钟\n")

    # 设置设备
    device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 获取快速验证数据加载器
    ssl_loader, nnpu_loader, val_loader, test_loader = get_quick_verification_dataloaders()

    # 创建模型
    model = QuickVerifyModel().to(device)
    print(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # SSL预训练快速验证
    train_ssl_phase_quick(model, ssl_loader, device)

    # NNPU微调快速验证
    train_nnpu_phase_quick(model, nnpu_loader, val_loader, device)

    # 最终快速测试
    accuracy, auc = final_test_quick(model, test_loader, device)

    # 总结
    print("\n" + "="*50)
    print("✅ 快速验证完成！")
    print(f"最终准确率: {accuracy:.4f}")
    print(f"最终AUC: {auc:.4f}")
    print("\n训练流程验证成功！")
    print("可以开始完整训练。")
    print("\n建议完整训练命令:")
    print("python ecg_ssl_pu/train_ssl_pu_mat.py")

if __name__ == "__main__":
    main()