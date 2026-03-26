# 导入PyTorch库，用于构建和训练神经网络
import torch
# 导入PyTorch的数据加载模块
from torch.utils.data import DataLoader
# 导入自定义的数据集类
from ecg_ssl_pu.dataset_mat_ecg import ECGSSLDataSet, ECGTestDataSet

def get_dataloaders(batch_size=32, num_workers=4):
    """
    获取SSL预训练、NNPU微调、验证的数据加载器
    适配现有的processed_traindata.mat和processed_testdata.mat
    """
    TRAIN_MAT = "data/processed_traindata.mat"
    TEST_MAT = "data/processed_testdata.mat"

    print("=== 数据加载器配置 ===")
    print(f"训练数据路径: {TRAIN_MAT}")
    print(f"测试数据路径: {TEST_MAT}")
    print(f"批次大小: {batch_size}")
    print(f"工作进程数: {num_workers}")

    # 1. SSL预训练：无标签数据
    print("\n正在创建SSL预训练数据集...")
    ssl_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="ssl")
    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f"SSL数据加载器创建完成，样本数: {len(ssl_dataset)}")

    # 2. NNPU微调：P+U数据
    print("\n正在创建NNPU微调数据集...")
    nnpu_dataset = ECGSSLDataSet(TRAIN_MAT, mode="train", task="nnpu")
    nnpu_loader = DataLoader(nnpu_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print(f"NNPU数据加载器创建完成，样本数: {len(nnpu_dataset)}")

    # 3. 验证集：P+Non-AF（真实标签）
    print("\n正在创建验证数据集...")
    val_dataset = ECGSSLDataSet(TRAIN_MAT, mode="val", task="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"验证数据加载器创建完成，样本数: {len(val_dataset)}")

    # 4. 测试集
    print("\n正在创建测试数据集...")
    test_dataset = ECGTestDataSet(TEST_MAT)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"测试数据加载器创建完成，样本数: {len(test_dataset)}")

    print("\n=== 数据加载器创建完成 ===")
    print(f"SSL预训练: {len(ssl_loader)} 个批次")
    print(f"NNPU微调: {len(nnpu_loader)} 个批次")
    print(f"验证集: {len(val_loader)} 个批次")
    print(f"测试集: {len(test_loader)} 个批次")

    return ssl_loader, nnpu_loader, val_loader, test_loader

def get_positive_prior():
    """
    计算正样本先验概率
    根据实施方案：positive_prior = 500 / (500 + 19000) ≈ 0.0256
    """
    # 正样本数量：500 (AF)
    # 未标记样本数量：19000 (1000-19999)
    positive_count = 500
    unlabeled_count = 19000
    prior = positive_count / (positive_count + unlabeled_count)
    print(f"\n正样本先验概率计算:")
    print(f"正样本数量: {positive_count}")
    print(f"未标记样本数量: {unlabeled_count}")
    print(f"先验概率 π = {prior:.4f}")
    return prior

if __name__ == "__main__":
    # 测试数据加载器
    print("测试数据加载器创建...")
    ssl_loader, nnpu_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

    # 显示一些批次信息
    print("\n=== 数据批次示例 ===")

    # SSL数据示例
    ssl_batch = next(iter(ssl_loader))
    print(f"SSL批次形状: {ssl_batch[0].shape}, {ssl_batch[1].shape}")

    # NNPU数据示例
    nnpu_batch = next(iter(nnpu_loader))
    print(f"NNPU批次形状: {nnpu_batch[0].shape}, 标签: {nnpu_batch[1].shape}")
    print(f"NNPU标签分布: 正样本={(nnpu_batch[1] == 1).sum().item()}, 未标记={(nnpu_batch[1] == 0).sum().item()}")

    # 验证数据示例
    val_batch = next(iter(val_loader))
    print(f"验证批次形状: {val_batch[0].shape}, 标签: {val_batch[1].shape}")
    print(f"验证标签分布: AF={(val_batch[1] == 1).sum().item()}, Non-AF={(val_batch[1] == 0).sum().item()}")

    print("\n✓ 数据加载器测试完成！")