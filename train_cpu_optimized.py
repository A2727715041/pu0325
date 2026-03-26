#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对CPU环境的优化训练脚本
"""

import os
# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 优化CPU设置
os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
os.environ['MKL_NUM_THREADS'] = '4'  # 限制MKL线程数

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# 导入自定义模块
from ecg_ssl_pu.dataset_mat_ecg import ECGSSLDataSet, ECGTestDataSet
from 代码.xResNet_50 import xResNet50
from ecg_ssl_pu.pu_loss_torch import ImbalancedPULoss

def get_cpu_optimized_dataloaders():
    """获取针对CPU优化的数据加载器"""
    TRAIN_MAT = "data/processed_traindata.mat"
    TEST_MAT = "data/processed_testdata.mat"

    # CPU优化配置：较小的批次，较少的工作进程
    BATCH_SIZE = 16  # 减小批次大小以节省内存
    NUM_WORKERS = 0   # CPU训练时不使用多进程

    print("创建CPU优化的数据加载器...")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"工作进程: {NUM_WORKERS}")

    # 1. SSL预训练：无标签数据
    ssl_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="ssl")
    ssl_loader = DataLoader(ssl_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # 2. NNPU微调：P+U数据
    nnpu_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="nnpu")
    nnpu_loader = DataLoader(nnpu_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # 3. 验证集：P+Non-AF（真实标签）
    val_dataset = ECGSSLDataSet(TRAIN_MAT, mode="val", task="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 4. 测试集
    test_dataset = ECGTestDataSet(TEST_MAT)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return ssl_loader, nnpu_loader, val_loader, test_loader

class CPUOptimizedModel(nn.Module):
    """针对CPU优化的模型"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.encoder = xResNet50()
        self.classifier = nn.Linear(feature_dim, 1)
        self.proj_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
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

def train_ssl_phase(model, ssl_loader, device, epochs=50):
    """SSL预训练阶段（简化版，适合CPU）"""
    print("\n=== SSL预训练阶段 ===")
    print(f"训练轮数: {epochs}")
    print(f"批次数量: {len(ssl_loader)}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # 使用tqdm显示进度，但减少刷新频率
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

            # 每100个批次打印一次损失
            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  批次 {batch_idx}/{len(ssl_loader)}, 平均损失: {avg_loss:.4f}")

        avg_loss = total_loss / len(ssl_loader)
        print(f"SSL Epoch {epoch} 完成, 平均损失: {avg_loss:.4f}")

        # 每10轮保存一次
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, f'ssl_pretrained_epoch{epoch}.pth')
            print(f"✓ 已保存第{epoch}轮预训练权重")

def train_nnpu_phase(model, nnpu_loader, val_loader, device, epochs=30):
    """NNPU微调阶段（简化版，适合CPU）"""
    print("\n=== NNPU微调阶段 ===")
    print(f"训练轮数: {epochs}")
    print(f"训练批次: {len(nnpu_loader)}")
    print(f"验证批次: {len(val_loader)}")

    # 冻结编码器，只训练分类器
    for param in model.encoder.parameters():
        param.requires_grad = False

    # 使用较小的学习率
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # PU损失函数
    prior = 0.0256  # 正样本先验
    pu_loss_fn = ImbalancedPULoss(prior=prior, pi_prime=0.5, nnpu=True)

    best_auc = 0.0
    best_model_path = "best_nnpu_cpu_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # 训练
        for batch_idx, (ecg, labels) in enumerate(tqdm(nnpu_loader, desc=f"NNPU Epoch {epoch}")):
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

        # 验证
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

        # 保存最优模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'auc': best_auc,
            }, best_model_path)
            print(f"✓ 发现更优模型，已保存 (AUC: {best_auc:.4f})")

    return best_auc

def main():
    """主训练函数"""
    print("=== CPU优化的SSL+PU学习训练 ===")

    # 设置设备
    device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 获取数据加载器
    ssl_loader, nnpu_loader, val_loader, test_loader = get_cpu_optimized_dataloaders()

    # 创建模型
    model = CPUOptimizedModel().to(device)
    print(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # SSL预训练阶段
    train_ssl_phase(model, ssl_loader, device, epochs=50)

    # NNPU微调阶段
    best_auc = train_nnpu_phase(model, nnpu_loader, val_loader, device, epochs=30)

    # 最终测试
    print(f"\n=== 最终测试 ===")
    print(f"最佳验证AUC: {best_auc:.4f}")
    print("训练完成！模型已保存到当前目录。")

if __name__ == "__main__":
    main()