"""
评估SSL+nnPU模型的脚本
支持Prior-corrected inference和全面的评估指标
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import numpy as np
import matplotlib.pyplot as plt

# 兼容直接运行脚本：将工程根目录加入 sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecg_ssl_pu.dataset_ecg import ECGTestDataset
from ecg_ssl_pu.model_ecg import ECGEncoder, SSLPUModel


def prior_corrected_inference(logits, pi_prime, pi_true):
    """
    Prior-corrected inference: 从训练prior (π')修正到真实prior (π)
    """
    probs_under_pi_prime = torch.sigmoid(logits)
    corrected_probs = (pi_true / pi_prime) * probs_under_pi_prime
    corrected_probs = torch.clamp(corrected_probs, 0.0, 1.0)
    return corrected_probs


def compute_comprehensive_metrics(y_true, y_pred, y_probs, prior_true=None, pi_prime=None):
    """计算全面的评估指标"""
    metrics = {}
    
    # 基础指标
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
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        metrics['roc_curve'] = (fpr, tpr)
    except ValueError:
        metrics['roc_auc'] = np.nan
        metrics['roc_curve'] = None
    
    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        metrics['pr_curve'] = (precision_curve, recall_curve)
    except ValueError:
        metrics['pr_auc'] = np.nan
        metrics['pr_curve'] = None
    
    # Prior-corrected评估
    if prior_true is not None and pi_prime is not None:
        corrected_probs = prior_corrected_inference(
            torch.from_numpy(y_probs).float(),
            pi_prime,
            prior_true
        ).numpy()
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
            fpr_corr, tpr_corr, _ = roc_curve(y_true, corrected_probs)
            metrics['corrected_roc_curve'] = (fpr_corr, tpr_corr)
        except ValueError:
            metrics['corrected_roc_auc'] = np.nan
            metrics['corrected_roc_curve'] = None
        
        try:
            metrics['corrected_pr_auc'] = average_precision_score(y_true, corrected_probs)
            prec_curve_corr, rec_curve_corr, _ = precision_recall_curve(y_true, corrected_probs)
            metrics['corrected_pr_curve'] = (prec_curve_corr, rec_curve_corr)
        except ValueError:
            metrics['corrected_pr_auc'] = np.nan
            metrics['corrected_pr_curve'] = None
    
    return metrics


def plot_curves(metrics, save_dir):
    """绘制ROC和PR曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    # ROC曲线
    if metrics['roc_curve'] is not None:
        fpr, tpr = metrics['roc_curve']
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"Original (AUC={metrics['roc_auc']:.4f})", linewidth=2)
        
        if 'corrected_roc_curve' in metrics and metrics['corrected_roc_curve'] is not None:
            fpr_corr, tpr_corr = metrics['corrected_roc_curve']
            plt.plot(fpr_corr, tpr_corr, label=f"Prior-corrected (AUC={metrics['corrected_roc_auc']:.4f})", 
                    linewidth=2, linestyle='--')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
        plt.close()
    
    # PR曲线
    if metrics['pr_curve'] is not None:
        precision_curve, recall_curve = metrics['pr_curve']
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f"Original (AUC={metrics['pr_auc']:.4f})", linewidth=2)
        
        if 'corrected_pr_curve' in metrics and metrics['corrected_pr_curve'] is not None:
            prec_curve_corr, rec_curve_corr = metrics['corrected_pr_curve']
            plt.plot(rec_curve_corr, prec_curve_corr, 
                    label=f"Prior-corrected (AUC={metrics['corrected_pr_auc']:.4f})", 
                    linewidth=2, linestyle='--')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300)
        plt.close()


def evaluate(model_path, test_mat_path, prior=None, pi_prime=0.5, device="cuda", 
             batch_size=128, save_plots=True, output_dir=None):
    """
    评估模型
    
    Args:
        model_path: 模型权重文件路径
        test_mat_path: 测试数据.mat文件路径
        prior: 真实先验概率（如果None，则从数据中估计）
        pi_prime: 训练时使用的prior
        device: 设备
        batch_size: 批次大小
        save_plots: 是否保存ROC/PR曲线图
        output_dir: 输出目录
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 加载测试数据
    test_dataset = ECGTestDataset(test_mat_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 如果未提供prior，从测试数据中估计
    if prior is None:
        data = loadmat(test_mat_path)
        y_test = data['y'].reshape(-1)
        prior = (y_test == 1).mean()
        print(f"从测试数据估计的真实prior: {prior:.4f}")
    else:
        print(f"使用提供的真实prior: {prior:.4f}")
    
    print(f"训练时使用的prior (π'): {pi_prime:.4f}")
    
    # 加载模型
    encoder = ECGEncoder(in_channels=1, base_channels=32)
    model = SSLPUModel(encoder, proj_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 评估
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
    
    # 原始预测
    all_probs_original = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    all_preds_original = (all_probs_original > 0.5).astype(int)
    
    # 计算指标
    metrics = compute_comprehensive_metrics(
        all_labels,
        all_preds_original,
        all_probs_original,
        prior_true=prior,
        pi_prime=pi_prime
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果（原始，基于训练prior π'）")
    print("="*60)
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    
    if 'corrected_accuracy' in metrics:
        print("\n" + "="*60)
        print("评估结果（Prior-corrected，修正到真实prior π）")
        print("="*60)
        print(f"Accuracy:     {metrics['corrected_accuracy']:.4f}")
        print(f"Precision:    {metrics['corrected_precision']:.4f}")
        print(f"Recall:       {metrics['corrected_recall']:.4f}")
        print(f"F1-Score:     {metrics['corrected_f1']:.4f}")
        print(f"ROC-AUC:      {metrics['corrected_roc_auc']:.4f}")
        print(f"PR-AUC:       {metrics['corrected_pr_auc']:.4f}")
    
    # 保存图表
    if save_plots:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "results", "evaluation")
        plot_curves(metrics, output_dir)
        print(f"\n曲线图已保存到: {output_dir}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估SSL+nnPU模型")
    parser.add_argument("--model", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据.mat文件路径")
    parser.add_argument("--prior", type=float, default=None, help="真实先验概率（如果None，则从数据中估计）")
    parser.add_argument("--pi_prime", type=float, default=0.5, help="训练时使用的prior")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        test_mat_path=args.test_data,
        prior=args.prior,
        pi_prime=args.pi_prime,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

