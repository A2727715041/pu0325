#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练测试，验证训练流程
"""

import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from 代码.xResNet_50 import xResNet50
from 代码.data_loader_mat import get_dataloaders, get_positive_prior
from ecg_ssl_pu.pu_loss_torch import ImbalancedPULoss

def test_model_forward():
    """测试模型前向传播"""
    print("=== 测试模型前向传播 ===")

    try:
        # 创建模型
        model = xResNet50()
        print(f"模型创建成功，特征维度: {model.feature_dim}")

        # 创建测试输入
        batch_size = 4
        test_input = torch.randn(batch_size, 1, 4000)
        print(f"测试输入形状: {test_input.shape}")

        # 前向传播
        output = model(test_input)
        print(f"模型输出形状: {output.shape}")
        print(f"预期输出形状: ({batch_size}, {model.feature_dim})")

        if output.shape == (batch_size, model.feature_dim):
            print("模型前向传播测试成功！")
            return True
        else:
            print("模型输出形状不符合预期")
            return False

    except Exception as e:
        print(f"模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pu_loss():
    """测试PU损失函数"""
    print("\n=== 测试PU损失函数 ===")

    try:
        # 创建PU损失函数
        prior = get_positive_prior()
        pu_loss_fn = ImbalancedPULoss(prior=prior, pi_prime=0.5, nnpu=True)
        print(f"PU损失函数创建成功，先验: {prior:.6f}")

        # 创建测试数据
        batch_size = 8
        logits = torch.randn(batch_size)
        labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0])  # 2个正样本，6个未标记

        print(f"测试logits形状: {logits.shape}")
        print(f"测试labels: {labels}")

        # 计算损失
        loss = pu_loss_fn(logits, labels)
        print(f"PU损失: {loss.item():.4f}")

        if not torch.isnan(loss) and not torch.isinf(loss):
            print("PU损失计算测试成功！")
            return True
        else:
            print("PU损失计算结果异常")
            return False

    except Exception as e:
        print(f"PU损失测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """测试训练步骤"""
    print("\n=== 测试训练步骤 ===")

    try:
        # 使用小批次数据
        ssl_loader, nnpu_loader, val_loader, test_loader = get_dataloaders(
            batch_size=8, num_workers=0
        )

        # 创建模型和优化器
        model = xResNet50()
        classifier = nn.Linear(model.feature_dim, 1)
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)

        # 创建损失函数
        prior = get_positive_prior()
        pu_loss_fn = ImbalancedPULoss(prior=prior, pi_prime=0.5, nnpu=True)

        print("开始训练步骤测试...")

        # 测试一个NNPU训练步骤
        model.train()
        classifier.train()

        # 获取一个批次
        batch = next(iter(nnpu_loader))
        ecg_data, labels = batch

        print(f"批次数据形状: {ecg_data.shape}")
        print(f"批次标签: {labels}")

        # 前向传播
        features = model(ecg_data)
        logits = classifier(features).squeeze(-1)

        print(f"特征形状: {features.shape}")
        print(f"logits形状: {logits.shape}")

        # 计算损失
        loss = pu_loss_fn(logits, labels)
        print(f"训练损失: {loss.item():.4f}")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("训练步骤测试成功！")
        return True

    except Exception as e:
        print(f"训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_step():
    """测试验证步骤"""
    print("\n=== 测试验证步骤 ===")

    try:
        # 使用小批次数据
        ssl_loader, nnpu_loader, val_loader, test_loader = get_dataloaders(
            batch_size=8, num_workers=0
        )

        # 创建模型
        model = xResNet50()
        classifier = nn.Linear(model.feature_dim, 1)

        print("开始验证步骤测试...")

        # 测试验证
        model.eval()
        classifier.eval()

        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 2:  # 只测试前2个批次
                    break

                ecg_data, labels = batch

                # 前向传播
                features = model(ecg_data)
                logits = classifier(features).squeeze(-1)
                probs = torch.sigmoid(logits)

                # 计算损失（使用标准BCE损失进行验证）
                bce_loss = nn.BCEWithLogitsLoss()(logits, labels.float())
                val_losses.append(bce_loss.item())

                # 收集预测和标签
                all_preds.extend(probs.numpy())
                all_labels.extend(labels.numpy())

        print(f"验证损失: {np.mean(val_losses):.4f}")
        print(f"收集到 {len(all_preds)} 个预测样本")

        if len(all_preds) > 0 and not np.any(np.isnan(all_preds)):
            print("验证步骤测试成功！")
            return True
        else:
            print("验证结果异常")
            return False

    except Exception as e:
        print(f"验证步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=== SSL+PU学习训练流程测试 ===")
    print("开始验证训练流程...\n")

    all_tests_passed = True

    # 1. 测试模型前向传播
    if not test_model_forward():
        all_tests_passed = False

    # 2. 测试PU损失函数
    if not test_pu_loss():
        all_tests_passed = False

    # 3. 测试训练步骤
    if not test_training_step():
        all_tests_passed = False

    # 4. 测试验证步骤
    if not test_validation_step():
        all_tests_passed = False

    # 总结
    print("\n" + "="*50)
    if all_tests_passed:
        print("所有训练流程测试通过！")
        print("可以进行完整的SSL+PU学习训练")
        print("\n建议训练命令:")
        print("python ecg_ssl_pu/train_ssl_pu_mat.py")
    else:
        print("部分训练流程测试失败，请检查相关模块")

    print("\n=== 测试完成 ===")

    return all_tests_passed

if __name__ == "__main__":
    main()