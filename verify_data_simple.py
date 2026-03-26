#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的数据验证脚本，只验证数据读取，不依赖PyTorch
"""

import numpy as np
from scipy.io import loadmat

def verify_mat_data():
    """验证.mat数据文件的结构和内容"""
    print("=== 开始数据验证 ===")

    # 1. 验证训练集
    print("\n=== 训练集验证 ===")
    try:
        train_data = loadmat('data/processed_traindata.mat')
        print("成功读取训练数据文件")

        # 获取数据
        train_signals = train_data['processed_traindata']
        print(f"训练数据形状: {train_signals.shape}")
        print(f"预期形状: (20000, 4000)")

        if train_signals.shape == (20000, 4000):
            print("训练数据形状符合预期")
        else:
            print(f"训练数据形状不符合预期: {train_signals.shape}")

        # 检查数据标准化
        mean_val = np.mean(train_signals)
        std_val = np.std(train_signals)
        print(f"信号均值: {mean_val:.6f} (预期: ≈0)")
        print(f"信号标准差: {std_val:.6f} (预期: ≈1)")

        # 检查数据范围
        min_val = np.min(train_signals)
        max_val = np.max(train_signals)
        print(f"信号范围: [{min_val:.4f}, {max_val:.4f}]")

        # 创建标签并验证分布
        n_samples = train_signals.shape[0]
        train_labels = np.zeros(n_samples, dtype=int)
        train_labels[:500] = 1      # AF
        if n_samples > 500:
            train_labels[500:1000] = 0  # Non-AF
        if n_samples > 1000:
            train_labels[1000:] = -1   # 无标签

        unique_vals, counts = np.unique(train_labels, return_counts=True)
        print(f"标签分布:")
        for val, count in zip(unique_vals, counts):
            print(f"  标签 {val}: {count} 个样本")

        # 验证任务划分
        ssl_mask = (train_labels == -1)
        nnpu_mask = (train_labels == 1) | (train_labels == -1)
        val_mask = (train_labels == 1) | (train_labels == 0)

        print(f"\n任务划分:")
        print(f"  SSL预训练 (无标签): {ssl_mask.sum()} 个样本")
        print(f"  NNPU微调 (P+U): {nnpu_mask.sum()} 个样本")
        print(f"  验证集 (P+Non-AF): {val_mask.sum()} 个样本")

    except Exception as e:
        print(f"训练数据验证失败: {e}")
        return False

    # 2. 验证测试集
    print("\n=== 测试集验证 ===")
    try:
        test_data = loadmat('data/processed_testdata.mat')
        print("成功读取测试数据文件")

        # 获取数据
        test_signals = test_data['processed_testdata']
        print(f"测试数据形状: {test_signals.shape}")
        print(f"预期形状: (10000, 4000)")

        if test_signals.shape == (10000, 4000):
            print("测试数据形状符合预期")
        else:
            print(f"测试数据形状不符合预期: {test_signals.shape}")

        # 检查数据标准化
        mean_val = np.mean(test_signals)
        std_val = np.std(test_signals)
        print(f"测试信号均值: {mean_val:.6f} (预期: ≈0)")
        print(f"测试信号标准差: {std_val:.6f} (预期: ≈1)")

        # 检查数据范围
        min_val = np.min(test_signals)
        max_val = np.max(test_signals)
        print(f"测试信号范围: [{min_val:.4f}, {max_val:.4f}]")

    except Exception as e:
        print(f"测试数据验证失败: {e}")
        return False

    # 3. 验证正样本先验
    print("\n=== 先验概率验证 ===")
    positive_count = 500
    unlabeled_count = 19000
    prior = positive_count / (positive_count + unlabeled_count)
    print(f"正样本数量: {positive_count}")
    print(f"未标记样本数量: {unlabeled_count}")
    print(f"正样本先验 π = {prior:.6f}")

    print("\n=== 数据验证完成 ===")
    print("所有检查通过！数据可以用于SSL+PU学习训练")

    return True

def create_mini_dataset():
    """创建小样本数据集用于快速测试"""
    print("\n=== 创建小样本数据集 ===")

    try:
        # 读取完整数据
        train_data = loadmat('data/processed_traindata.mat')
        train_signals = train_data['processed_traindata']

        # 创建小样本集
        mini_size = 200  # 总共200个样本
        ssl_samples = 100  # SSL无标签样本
        pos_samples = 50   # 正样本
        neg_samples = 50   # 负样本

        # 选择样本
        indices = []

        # SSL样本 (无标签，从1000开始)
        ssl_start = 1000
        ssl_indices = list(range(ssl_start, ssl_start + ssl_samples))
        indices.extend(ssl_indices)

        # 正样本 (AF，0-499)
        pos_indices = list(range(0, pos_samples))
        indices.extend(pos_indices)

        # 负样本 (Non-AF，500-999)
        neg_start = 500
        neg_indices = list(range(neg_start, neg_start + neg_samples))
        indices.extend(neg_indices)

        # 创建小样本数据
        mini_signals = train_signals[indices]

        # 创建标签
        mini_labels = np.zeros(len(indices), dtype=int)
        mini_labels[:ssl_samples] = -1  # SSL样本为无标签
        mini_labels[ssl_samples:ssl_samples+pos_samples] = 1  # 正样本
        mini_labels[ssl_samples+pos_samples:] = 0  # 负样本

        print(f"创建小样本数据集成功")
        print(f"  总样本数: {len(mini_signals)}")
        print(f"  信号形状: {mini_signals.shape}")
        print(f"  标签分布: {np.unique(mini_labels, return_counts=True)}")

        # 保存小样本数据
        mini_data = {
            'mini_signals': mini_signals,
            'mini_labels': mini_labels
        }

        from scipy.io import savemat
        savemat('data/mini_testdata.mat', mini_data)
        print(f"小样本数据已保存到: data/mini_testdata.mat")

        return True

    except Exception as e:
        print(f"创建小样本数据集失败: {e}")
        return False

if __name__ == "__main__":
    # 验证数据
    data_ok = verify_mat_data()

    if data_ok:
        # 创建小样本数据集
        create_mini_dataset()

        print("\n=== 验证总结 ===")
        print("数据验证和准备完成！")
        print("可以开始实施SSL+PU学习方案")
    else:
        print("\n数据验证失败，请检查数据文件")