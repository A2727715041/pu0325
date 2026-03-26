#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证MATLAB数据文件的读取和结构
根据实施方案阶段1的要求进行验证
"""

import h5py
import numpy as np
from scipy.io import loadmat

def load_mat_file(mat_path):
    """读取MATLAB文件（支持v5和v7.3格式）"""
    print(f"正在读取文件: {mat_path}")

    # 首先尝试用scipy读取（v5格式）
    try:
        print("尝试使用scipy.io.loadmat读取...")
        data = loadmat(mat_path)
        print(f"✓ 成功使用scipy读取文件")
        print(f"文件中的变量: {list(data.keys())}")

        # 过滤掉系统变量（以__开头）
        user_vars = {k: v for k, v in data.items() if not k.startswith('__')}
        print(f"用户变量: {list(user_vars.keys())}")

        # 查找数据和标签
        ecg_data = None
        labels = None

        # 常见的数据变量名
        data_keys = ['processed_traindata', 'processed_testdata', 'data', 'X']
        label_keys = ['label', 'labels', 'y']

        for key in data_keys:
            if key in user_vars:
                print(f"找到数据变量: {key}")
                ecg_data = user_vars[key]
                print(f"数据形状: {ecg_data.shape}")
                break

        for key in label_keys:
            if key in user_vars:
                print(f"找到标签变量: {key}")
                labels = user_vars[key]
                print(f"标签形状: {labels.shape}")
                break

        # 如果没有找到标签，创建默认标签
        if labels is None and ecg_data is not None:
            print("未找到标签变量，创建默认标签")
            n_samples = ecg_data.shape[0] if ecg_data.shape[0] != 4000 else ecg_data.shape[1]
            labels = np.zeros(n_samples, dtype=int)
            # 根据实施方案设置标签
            if 'train' in mat_path:
                # 训练集：前500个为AF(1)，500-999为Non-AF(0)，其余为无标签(-1)
                labels[:500] = 1  # AF
                if len(labels) > 500:
                    labels[500:1000] = 0  # Non-AF
                if len(labels) > 1000:
                    labels[1000:] = -1  # 无标签
            else:
                # 测试集：假设都有真实标签
                pass
            print(f"创建标签形状: {labels.shape}")

        return ecg_data, labels

    except Exception as scipy_error:
        print(f"scipy读取失败: {scipy_error}")
        print("尝试使用h5py读取...")
        try:
            with h5py.File(mat_path, 'r') as f:
                # 查看文件内的变量名
                print(f"文件 {mat_path} 内的变量名：{list(f.keys())}")

                # 假设ECG信号变量为'data'，标签为'label'（按实际调整）
                # 先尝试查找可能的变量名
                for key in f.keys():
                    print(f"变量 '{key}' 的形状: {f[key].shape}")
                    print(f"变量 '{key}' 的数据类型: {f[key].dtype}")

                # 尝试常见的数据变量名
                data_keys = ['data', 'X', 'processed_traindata', 'processed_testdata']
                label_keys = ['label', 'y', 'labels']

                ecg_data = None
                labels = None

                # 查找数据变量
                for key in data_keys:
                    if key in f:
                        print(f"找到数据变量: {key}")
                        ecg_data = np.array(f[key])
                        print(f"数据形状: {ecg_data.shape}")
                        break

                # 查找标签变量
                for key in label_keys:
                    if key in f:
                        print(f"找到标签变量: {key}")
                        labels = np.array(f[key])
                        print(f"标签形状: {labels.shape}")
                        break

                # 如果没有找到标准变量名，使用第一个数组作为数据
                if ecg_data is None and len(f.keys()) > 0:
                    first_key = list(f.keys())[0]
                    print(f"使用第一个变量作为数据: {first_key}")
                    ecg_data = np.array(f[first_key])
                    print(f"数据形状: {ecg_data.shape}")

                    # 如果有第二个变量，作为标签
                    if len(f.keys()) > 1:
                        second_key = list(f.keys())[1]
                        print(f"使用第二个变量作为标签: {second_key}")
                        labels = np.array(f[second_key])
                        print(f"标签形状: {labels.shape}")

                return ecg_data, labels

        except Exception as h5py_error:
            print(f"h5py读取也失败: {h5py_error}")
            return None, None

def verify_data_structure():
    """验证数据结构是否符合预期"""
    # 路径配置
    TRAIN_MAT = "data/processed_traindata.mat"
    TEST_MAT = "data/processed_testdata.mat"

    print("=== 开始数据验证 ===")

    # 1. 验证训练集
    print("\n=== 训练集验证 ===")
    train_ecg, train_labels = load_mat_file(TRAIN_MAT)

    if train_ecg is not None:
        print(f"ECG数据形状：{train_ecg.shape}")
        print(f"预期形状：(20000, 4000) 或 (4000, 20000)")

        # 根据形状判断是否需要转置
        if train_ecg.shape == (20000, 4000):
            print("✓ 数据形状符合预期 (20000, 4000)")
        elif train_ecg.shape == (4000, 20000):
            print("✓ 数据形状为 (4000, 20000)，需要转置")
            train_ecg = train_ecg.T
            print(f"转置后形状：{train_ecg.shape}")
        else:
            print(f"⚠ 数据形状不符合预期: {train_ecg.shape}")

        # 检查数据标准化
        print(f"信号均值：{np.mean(train_ecg):.4f}")
        print(f"信号标准差：{np.std(train_ecg):.4f}")
        print(f"预期：均值≈0，标准差≈1")

        # 检查数据范围
        print(f"信号最小值：{np.min(train_ecg):.4f}")
        print(f"信号最大值：{np.max(train_ecg):.4f}")

    if train_labels is not None:
        print(f"标签形状：{train_labels.shape}")
        print(f"预期形状：(20000,) 或 (1, 20000)")

        # 调整标签形状
        if train_labels.shape == (1, 20000):
            train_labels = train_labels.squeeze()
            print(f"调整标签形状后：{train_labels.shape}")
        elif train_labels.shape == (20000, 1):
            train_labels = train_labels.squeeze()
            print(f"调整标签形状后：{train_labels.shape}")

        print(f"标签取值分布：")
        unique_vals, counts = np.unique(train_labels, return_counts=True)
        for val, count in zip(unique_vals, counts):
            print(f"  值 {val}: {count} 个样本")

        # 检查预期分布
        expected_af = 500  # AF样本
        expected_non_af = 500  # Non-AF样本
        expected_unlabeled = 19000  # 无标签样本

        if 1 in unique_vals and (-1 in unique_vals or 0 in unique_vals):
            print("✓ 找到预期的标签值")
        else:
            print("⚠ 标签值不符合预期")

    # 2. 验证测试集
    print("\n=== 测试集验证 ===")
    test_ecg, test_labels = load_mat_file(TEST_MAT)

    if test_ecg is not None:
        print(f"测试ECG形状：{test_ecg.shape}")
        print(f"预期形状：(10000, 4000) 或 (4000, 10000)")

        # 根据形状判断是否需要转置
        if test_ecg.shape == (10000, 4000):
            print("✓ 测试数据形状符合预期 (10000, 4000)")
        elif test_ecg.shape == (4000, 10000):
            print("✓ 测试数据形状为 (4000, 10000)，需要转置")
            test_ecg = test_ecg.T
            print(f"转置后形状：{test_ecg.shape}")
        else:
            print(f"⚠ 测试数据形状不符合预期: {test_ecg.shape}")

        # 检查数据标准化
        print(f"测试信号均值：{np.mean(test_ecg):.4f}")
        print(f"测试信号标准差：{np.std(test_ecg):.4f}")

    if test_labels is not None:
        print(f"测试标签形状：{test_labels.shape}")
        print(f"测试标签取值分布：")
        unique_vals, counts = np.unique(test_labels, return_counts=True)
        for val, count in zip(unique_vals, counts):
            print(f"  值 {val}: {count} 个样本")

    print("\n=== 验证总结 ===")
    print("如果以上检查都符合预期，数据验证通过！")
    print("接下来可以修改数据加载模块。")

if __name__ == "__main__":
    verify_data_structure()