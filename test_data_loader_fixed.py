#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载器（解决OpenMP冲突）
"""

import os
# 在导入任何可能使用OpenMP的库之前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from 代码.data_loader_mat import get_dataloaders, get_positive_prior

def test_data_loaders():
    """测试数据加载器"""
    print("=== 测试数据加载器 ===")

    try:
        # 获取正样本先验
        prior = get_positive_prior()
        print(f"正样本先验: {prior:.6f}")

        # 创建数据加载器
        ssl_loader, nnpu_loader, val_loader, test_loader = get_dataloaders(
            batch_size=16, num_workers=0  # 设置num_workers=0避免多进程问题
        )

        print("\n=== 数据加载器创建成功 ===")
        print(f"SSL预训练: {len(ssl_loader)} 个批次")
        print(f"NNPU微调: {len(nnpu_loader)} 个批次")
        print(f"验证集: {len(val_loader)} 个批次")
        print(f"测试集: {len(test_loader)} 个批次")

        # 测试批次数据
        print("\n=== 测试批次数据 ===")

        # SSL数据
        ssl_batch = next(iter(ssl_loader))
        print(f"SSL批次: 视图1形状={ssl_batch[0].shape}, 视图2形状={ssl_batch[1].shape}")

        # NNPU数据
        nnpu_batch = next(iter(nnpu_loader))
        print(f"NNPU批次: 数据形状={nnpu_batch[0].shape}, 标签形状={nnpu_batch[1].shape}")
        print(f"NNPU标签: 正样本={(nnpu_batch[1] == 1).sum().item()}, 未标记={(nnpu_batch[1] == 0).sum().item()}")

        # 验证数据
        val_batch = next(iter(val_loader))
        print(f"验证批次: 数据形状={val_batch[0].shape}, 标签形状={val_batch[1].shape}")
        print(f"验证标签: AF={(val_batch[1] == 1).sum().item()}, Non-AF={(val_batch[1] == 0).sum().item()}")

        print("\n数据加载器测试成功！")
        return True

    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loaders()