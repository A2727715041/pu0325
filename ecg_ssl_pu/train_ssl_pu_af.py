import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)
import numpy as np

# 兼容直接运行脚本：将工程根目录加入 sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecg_ssl_pu.dataset_ecg import ECGSSLPUTrainDataset, ECGTestDataset, ECGTransform
from ecg_ssl_pu.model_ecg import ECGEncoder, SSLPUModel
from ecg_ssl_pu.ssl_loss import pu_aware_ssl_loss
from ecg_ssl_pu.pu_loss_torch import ImbalancedPULoss


def prior_corrected_inference(logits, pi_prime, pi_true):
    """
    Prior-corrected inference: 从训练prior (π')修正到真实prior (π)
    
    根据odds ratio公式：
        odds_true = (π_true / (1-π_true)) * (odds_π' * (1-π') / π')
    然后转换为概率：P(y=1|x)_true = odds_true / (1 + odds_true)
    
    更简化的形式（当π_true << 1时）：
        P(y=1|x)_true ≈ (π_true / π') * sigmoid(logit)
    
    Args:
        logits: 模型输出的logits（在π'下训练）
        pi_prime: 训练时使用的prior（例如0.5）
        pi_true: 真实数据的prior（例如0.01）
    
    Returns:
        corrected_probs: 修正后的概率
    """
    # 将logits转换为概率（在π'下的概率）
    probs_under_pi_prime = torch.sigmoid(logits)
    
    # 方法1：简化线性缩放（适用于π_true << 1的情况）
    # P(y=1|x)_true ≈ (π_true / π') * P(y=1|x)_π'
    corrected_probs = (pi_true / pi_prime) * probs_under_pi_prime
    corrected_probs = torch.clamp(corrected_probs, 0.0, 1.0)
    
    # 方法2：精确的odds ratio方法（如果需要更精确，可以取消注释）
    # odds_pi_prime = probs_under_pi_prime / (1 - probs_under_pi_prime + 1e-10)
    # odds_true = (pi_true / (1 - pi_true + 1e-10)) * (odds_pi_prime * (1 - pi_prime) / (pi_prime + 1e-10))
    # corrected_probs = odds_true / (1 + odds_true)
    # corrected_probs = torch.clamp(corrected_probs, 0.0, 1.0)
    
    return corrected_probs


def compute_comprehensive_metrics(y_true, y_pred, y_probs, prior_true=None, pi_prime=None):
    """
    计算全面的评估指标，包括阈值无关指标
    
    Args:
        y_true: 真实标签 [N]
        y_pred: 预测标签（0/1）[N]
        y_probs: 预测概率 [N]
        prior_true: 真实先验概率（用于prior-corrected评估）
        pi_prime: 训练prior（用于prior-corrected评估）
    
    Returns:
        metrics_dict: 包含所有指标的字典
    """
    metrics = {}
    
    # 基础指标（基于阈值0.5）
    metrics['accuracy'] = (y_true == y_pred).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # 阈值无关指标
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
    except ValueError:
        metrics['roc_auc'] = np.nan
    
    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_probs)
    except ValueError:
        metrics['pr_auc'] = np.nan
    
    # Prior-corrected评估（如果提供了prior信息）
    if prior_true is not None and pi_prime is not None:
        # 将概率从π'修正到π_true
        corrected_probs = prior_corrected_inference(
            torch.from_numpy(y_probs).float(),
            pi_prime,
            prior_true
        ).numpy()
        
        # 使用修正后的概率重新计算预测（阈值0.5）
        corrected_preds = (corrected_probs > 0.5).astype(int)
        
        metrics['corrected_accuracy'] = (y_true == corrected_preds).mean()
        prec_corr, rec_corr, f1_corr, _ = precision_recall_fscore_support(
            y_true, corrected_preds, labels=[0, 1], average="binary", pos_label=1, zero_division=0
        )
        metrics['corrected_precision'] = prec_corr
        metrics['corrected_recall'] = rec_corr
        metrics['corrected_f1'] = f1_corr
        
        try:
            metrics['corrected_roc_auc'] = roc_auc_score(y_true, corrected_probs)
        except ValueError:
            metrics['corrected_roc_auc'] = np.nan
        
        try:
            metrics['corrected_pr_auc'] = average_precision_score(y_true, corrected_probs)
        except ValueError:
            metrics['corrected_pr_auc'] = np.nan
    
    return metrics


def get_progressive_pi_prime(epoch, total_epochs, pi_prime_start=0.1, pi_prime_end=0.5, schedule='linear'):
    """
    渐进式prior调度
    
    Args:
        epoch: 当前epoch（从1开始）
        total_epochs: 总epoch数
        pi_prime_start: 起始π'
        pi_prime_end: 结束π'
        schedule: 'linear' 或 'cosine'
    
    Returns:
        pi_prime: 当前epoch的π'值
    """
    progress = (epoch - 1) / (total_epochs - 1) if total_epochs > 1 else 0.0
    
    if schedule == 'linear':
        pi_prime = pi_prime_start + (pi_prime_end - pi_prime_start) * progress
    elif schedule == 'cosine':
        import math
        pi_prime = pi_prime_start + (pi_prime_end - pi_prime_start) * (1 - math.cos(math.pi * progress)) / 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return pi_prime


def train(
    train_mat_path,
    test_mat_path,
    prior=None,
    batch_size=128,
    epochs=50,
    lr=1e-3,
    lambda_pu=5.0,
    temperature=0.1,
    min_prior=0.01,
    pi_prime=0.5,
    progressive_prior=False,
    pi_prime_schedule='linear',
    pi_prime_start=0.1,
    device="cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 如未显式给出 prior，则从 train_mat 里读取
    if prior is None:
        mat = loadmat(train_mat_path)
        if "prior" in mat:
            prior = float(mat["prior"].reshape(-1)[0])
        else:
            raise ValueError("未在训练 .mat 中找到 prior 字段，请在 prepare_afdb_dataset.py 中一并保存。")

    prior = max(prior, min_prior)
    pi_prime_final = pi_prime  # 保存最终值用于显示
    if progressive_prior:
        print(f"使用原始类先验 π = {prior:.4f}, 渐进式prior: {pi_prime_start:.2f} -> {pi_prime:.2f} ({pi_prime_schedule})")
    else:
        print(f"使用原始类先验 π = {prior:.4f}, 目标平衡先验 π' = {pi_prime:.2f} (固定)")

    # 数据集与 DataLoader
    transform = ECGTransform()
    train_dataset = ECGSSLPUTrainDataset(train_mat_path, transform=transform)
    test_dataset = ECGTestDataset(test_mat_path)

    # 为了缓解极端不平衡，使用 WeightedRandomSampler 提升正类采样概率
    y_pu_np = train_dataset.y_pu
    n_pos = (y_pu_np == 1).sum()
    n_u = (y_pu_np == -1).sum()
    if n_pos == 0:
        raise RuntimeError("训练集中没有标注正样本，无法进行 nnPU 训练，请在 prepare_afdb_dataset.py 提高 labeled_positive_ratio。")
    # 让正/未标记在采样期望上近似各占一半
    w_pos = (len(y_pu_np) / (2 * n_pos))
    w_u = (len(y_pu_np) / (2 * n_u))
    weights = torch.tensor([w_pos if y == 1 else w_u for y in y_pu_np], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型与损失
    encoder = ECGEncoder(in_channels=1, base_channels=32)
    model = SSLPUModel(encoder, proj_dim=128).to(device)

    # 初始损失函数（如果使用渐进式prior，会在每个epoch更新）
    current_pi_prime = pi_prime_start if progressive_prior else pi_prime
    pu_criterion = ImbalancedPULoss(prior=prior, pi_prime=current_pi_prime, nnpu=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 准备 CSV 结果文件（扩展列以包含所有指标）
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "ssl_pu_metrics.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "pi_prime",  # 当前使用的π'
                "ssl_loss",
                "pu_loss",
                "acc",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "corrected_acc",
                "corrected_precision",
                "corrected_recall",
                "corrected_f1",
                "corrected_roc_auc",
                "corrected_pr_auc",
            ]
        )

    for epoch in range(1, epochs + 1):
        # 如果使用渐进式prior，更新当前π'
        if progressive_prior:
            current_pi_prime = get_progressive_pi_prime(
                epoch, epochs, pi_prime_start, pi_prime, pi_prime_schedule
            )
            # 重新创建损失函数以使用新的π'
            pu_criterion = ImbalancedPULoss(prior=prior, pi_prime=current_pi_prime, nnpu=True)
        
        model.train()
        total_ssl_loss = 0.0
        total_pu_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (π'={current_pi_prime:.3f})"):
            x_raw, x1, x2, y_pu = batch
            x_raw = x_raw.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            y_pu = y_pu.to(device)

            optimizer.zero_grad()

            logits, z1, z2 = model(x_raw, x1, x2)

            # ====== PU-aware SSL：在已标注正样本上做 supervised contrastive ======
            loss_ssl = pu_aware_ssl_loss(z1, z2, y_pu, temperature=temperature, alpha=1.0)

            loss_pu = pu_criterion(logits, y_pu)

            loss = loss_ssl + lambda_pu * loss_pu
            loss.backward()
            optimizer.step()

            total_ssl_loss += loss_ssl.item()
            total_pu_loss += loss_pu.item()

        avg_ssl = total_ssl_loss / len(train_loader)
        avg_pu = total_pu_loss / len(train_loader)

        # 测试集评估：使用全面的评估指标
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                h = model.encoder(x)
                logits = model.classifier(h).squeeze(-1)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # 原始预测（基于训练prior π'，阈值0.5）
        all_probs_original = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
        all_preds_original = (all_probs_original > 0.5).astype(int)
        
        # 计算全面指标（原始和prior-corrected）
        metrics = compute_comprehensive_metrics(
            all_labels, 
            all_preds_original, 
            all_probs_original,
            prior_true=prior,
            pi_prime=current_pi_prime
        )

        print(f"Epoch {epoch} (π'={current_pi_prime:.3f}): SSL={avg_ssl:.4f}, PU={avg_pu:.4f}")
        print(f"  [原始评估] Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        print(f"  [阈值无关] ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
        
        if 'corrected_accuracy' in metrics:
            print(f"  [Prior-corrected] Acc={metrics['corrected_accuracy']:.4f}, "
                  f"Prec={metrics['corrected_precision']:.4f}, Rec={metrics['corrected_recall']:.4f}, "
                  f"F1={metrics['corrected_f1']:.4f}")
            print(f"  [Prior-corrected阈值无关] ROC-AUC={metrics['corrected_roc_auc']:.4f}, "
                  f"PR-AUC={metrics['corrected_pr_auc']:.4f}")
        
        print("  Confusion Matrix (rows=true, cols=pred, order=[non-AF, AF]):")
        print(metrics['confusion_matrix'])

        # 追加写入 CSV（包含所有指标）
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                current_pi_prime,
                avg_ssl,
                avg_pu,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['roc_auc'],
                metrics['pr_auc'],
                metrics.get('corrected_accuracy', np.nan),
                metrics.get('corrected_precision', np.nan),
                metrics.get('corrected_recall', np.nan),
                metrics.get('corrected_f1', np.nan),
                metrics.get('corrected_roc_auc', np.nan),
                metrics.get('corrected_pr_auc', np.nan),
            ])

    # 保存模型
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ssl_pu_ecg_af.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"模型已保存到: {ckpt_path}")


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = os.path.join(ROOT_DIR, "ecg_ssl_pu", "data")

    train_mat_path = os.path.join(DATA_DIR, "afdb_train.mat")
    test_mat_path = os.path.join(DATA_DIR, "afdb_test.mat")

    train(
        train_mat_path=train_mat_path,
        test_mat_path=test_mat_path,
        prior=None,  # None 表示从 .mat 中读取
        batch_size=128,
        epochs=50,
        lr=1e-3,
        lambda_pu=5.0,     # 放大 nnPU 项的权重
        temperature=0.1,
        min_prior=0.01,
        pi_prime=0.5,      # 目标"平衡"先验，ImbalancednnPU 关键参数（最终值）
        progressive_prior=False,  # 设置为True启用渐进式prior
        pi_prime_schedule='linear',  # 'linear' 或 'cosine'
        pi_prime_start=0.1,  # 渐进式prior的起始值
        device="cuda",
    )