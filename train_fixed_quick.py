#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版快速验证训练
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

def get_fixed_dataloaders():
    """获取修复的数据加载器"""
    from ecg_ssl_pu.dataset_mat_ecg import ECGSSLDataSet, ECGTestDataSet

    TRAIN_MAT = "data/processed_traindata.mat"
    TEST_MAT = "data/processed_testdata.mat"

    print("创建修复版快速验证数据加载器...")

    # 创建数据集
    ssl_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="ssl")
    nnpu_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="nnpu")
    val_dataset = ECGSSLDataSet(TRAIN_MAT, mode="val", task="val")

    # 更小的样本集
    ssl_indices = list(range(20))   # 20个SSL样本
    nnpu_indices = list(range(10))  # 10个NNPU样本
    val_indices = list(range(10))   # 10个验证样本

    ssl_subset =