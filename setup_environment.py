#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境设置和依赖安装脚本
"""

import subprocess
import sys
import os

def check_python_version():
    """检查Python版本"""
    print("=== 检查Python版本 ===")
    python_version = sys.version
    print(f"当前Python版本: {python_version}")

    # 检查是否为Python 3.7+
    if sys.version_info >= (3, 7):
        print("Python版本符合要求 (3.7+)")
        return True
    else:
        print("警告: Python版本低于3.7，建议升级到3.7或更高版本")
        return False

def install_dependencies():
    """安装必要的依赖包"""
    print("\n=== 安装依赖包 ===")

    # 依赖包列表
    dependencies = [
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "h5py"
    ]

    print("正在安装以下依赖包:")
    for dep in dependencies:
        print(f"  - {dep}")

    try:
        # 使用pip安装依赖
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + dependencies)

        print("✓ 依赖包安装成功")
        return True

    except subprocess.CalledProcessError as e:
        print(f"依赖包安装失败: {e}")
        return False

def verify_installation():
    """验证安装是否成功"""
    print("\n=== 验证安装 ===")

    required_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm")
    ]

    all_installed = True

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} 安装成功")
        except ImportError:
            print(f"❌ {package_name} 安装失败")
            all_installed = False

    return all_installed

def check_gpu_availability():
    """检查GPU是否可用"""
    print("\n=== 检查GPU可用性 ===")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ 检测到GPU: {gpu_name}")
            print(f"✓ GPU数量: {gpu_count}")
            print("✓ 可以使用GPU加速训练")
            return True
        else:
            print("⚠ 未检测到GPU，将使用CPU训练")
            print("注意: CPU训练速度较慢，建议使用GPU")
            return False
    except ImportError:
        print("❌ 无法导入torch，GPU检查失败")
        return False

def check_data_files():
    """检查数据文件是否存在"""
    print("\n=== 检查数据文件 ===")

    required_files = [
        "data/processed_traindata.mat",
        "data/processed_testdata.mat"
    ]

    all_files_exist = True

    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
            print(f"✓ {file_path} 存在 ({file_size:.2f} GB)")
        else:
            print(f"❌ {file_path} 不存在")
            all_files_exist = False

    return all_files_exist

def run_basic_tests():
    """运行基本测试"""
    print("\n=== 运行基本测试 ===")

    try:
        # 测试数据验证
        print("运行数据验证测试...")
        result = subprocess.run([
            sys.executable, "verify_data_simple.py"
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✓ 数据验证测试通过")
        else:
            print("❌ 数据验证测试失败")
            print("错误输出:", result.stderr)
            return False

        # 测试小样本流程
        print("运行小样本测试...")
        result = subprocess.run([
            sys.executable, "test_mini_pipeline.py"
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✓ 小样本测试通过")
        else:
            print("❌ 小样本测试失败")
            print("错误输出:", result.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        return False

def main():
    """主函数"""
    print("=== SSL+PU学习环境设置 ===")
    print("开始设置训练环境...\n")

    # 检查Python版本
    if not check_python_version():
        print("\n❌ Python版本检查失败")
        return False

    # 安装依赖
    if not install_dependencies():
        print("\n❌ 依赖安装失败")
        return False

    # 验证安装
    if not verify_installation():
        print("\n❌ 安装验证失败")
        return False

    # 检查GPU
    check_gpu_availability()

    # 检查数据文件
    if not check_data_files():
        print("\n❌ 数据文件检查失败")
        return False

    # 运行基本测试
    if not run_basic_tests():
        print("\n❌ 基本测试失败")
        return False

    # 总结
    print("\n" + "="*50)
    print("✓ 环境设置完成！")
    print("✓ 所有检查通过")
    print("✓ 可以开始SSL+PU学习训练")
    print("\n下一步:")
    print("1. 运行: python ecg_ssl_pu/train_ssl_pu_mat.py")
    print("2. 监控训练进度")
    print("3. 查看训练结果")
    print("\n=== 设置完成 ===")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)