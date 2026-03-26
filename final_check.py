#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终环境检查
"""

import os
# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from scipy.io import loadmat

def final_environment_check():
    """最终环境检查"""
    print('=== 最终环境检查 ===')
    print(f'PyTorch版本: {torch.__version__}')
    print('设备: CPU (无GPU)')
    print(f'NumPy版本: {np.__version__}')

    # 检查数据文件
    try:
        train_data = loadmat('data/processed_traindata.mat')
        test_data = loadmat('data/processed_testdata.mat')
        print('数据文件: 可正常读取')
        print(f'训练数据形状: {train_data["processed_traindata"].shape}')
        print(f'测试数据形状: {test_data["processed_testdata"].shape}')
        print('数据文件检查: 通过')
    except Exception as e:
        print(f'数据文件: 读取失败: {e}')
        return False

    print('\n=== 环境准备完成 ===')
    print('可以开始SSL+PU学习训练！')
    print('\nCPU训练说明:')
    print('- SSL预训练: 预计需要2-4小时')
    print('- NNPU微调: 预计需要1-2小时')
    print('- 建议使用较小的批次大小以节省内存')
    print('- 训练过程中可以正常进行其他工作')

    return True

if __name__ == "__main__":
    success = final_environment_check()
    if success:
        print('\n所有检查通过，可以开始训练！')
    else:
        print('\n环境检查失败，请解决问题后重试。')