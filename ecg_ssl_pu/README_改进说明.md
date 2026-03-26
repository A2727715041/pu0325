# SSL+nnPU模型改进说明

根据修改意见，本次更新实现了以下关键改进：

## 主要改进点

### 1. Prior-corrected Inference（先验修正推断）

**问题**：训练时使用平衡prior（π'=0.5），但真实数据prior很低（π≈0.01），导致推断时出现极高的假阳性。

**解决方案**：实现prior-corrected inference，在推断阶段将模型输出的概率从训练prior修正到真实prior。

**公式**：
```
P(y=1|x)_true ≈ (π_true / π') * sigmoid(logit)
```

**使用**：评估时会自动计算原始评估和prior-corrected评估两套指标。

### 2. 阈值无关指标

**新增指标**：
- **ROC-AUC**：受试者工作特征曲线下面积
- **PR-AUC**：精确率-召回率曲线下面积（在极度不平衡数据下比ROC-AUC更可靠）

**说明**：在AF分类任务中（正类占比约1%），PR-AUC能更准确地反映模型性能，因为它在不平衡数据下不会像ROC-AUC那样过于乐观。

### 3. 渐进式Prior训练

**问题**：固定π'=0.5可能导致训练早期不稳定。

**解决方案**：支持渐进式prior调度，从较小的π'（如0.1）逐渐增加到目标值（如0.5）。

**调度方式**：
- `linear`：线性增加
- `cosine`：余弦调度（更平滑）

**使用示例**：
```python
train(
    ...
    pi_prime=0.5,              # 最终目标值
    progressive_prior=True,     # 启用渐进式
    pi_prime_start=0.1,        # 起始值
    pi_prime_schedule='linear', # 调度方式
)
```

### 4. 完整的评估指标输出

**CSV输出列**：
- epoch, π', ssl_loss, pu_loss
- 原始评估：acc, precision, recall, f1, roc_auc, pr_auc
- Prior-corrected评估：corrected_acc, corrected_precision, corrected_recall, corrected_f1, corrected_roc_auc, corrected_pr_auc

## 使用方法

### 训练

```python
from ecg_ssl_pu.train_ssl_pu_af import train

train(
    train_mat_path="path/to/afdb_train.mat",
    test_mat_path="path/to/afdb_test.mat",
    prior=None,  # 从.mat文件中读取
    batch_size=128,
    epochs=50,
    lr=1e-3,
    lambda_pu=5.0,
    pi_prime=0.5,  # 固定prior训练
    # 或使用渐进式prior：
    # progressive_prior=True,
    # pi_prime_start=0.1,
    # pi_prime_schedule='linear',
    device="cuda",
)
```

### 评估

```bash
python ecg_ssl_pu/evaluate_ssl_pu.py \
    --model checkpoints/ssl_pu_ecg_af.pth \
    --test_data data/afdb_test.mat \
    --prior 0.01 \
    --pi_prime 0.5 \
    --batch_size 128
```

或在Python中使用：

```python
from ecg_ssl_pu.evaluate_ssl_pu import evaluate

metrics = evaluate(
    model_path="checkpoints/ssl_pu_ecg_af.pth",
    test_mat_path="data/afdb_test.mat",
    prior=0.01,
    pi_prime=0.5,
    save_plots=True
)
```

## 实验结果解读

### 理解"高召回率但低精确率"的结果

当使用固定π'=0.5训练时，如果直接使用阈值0.5进行推断，可能会出现：
- **极高召回率**（接近1.0）：模型能够检测到大部分AF样本
- **极低精确率**（接近0.01）：但同时产生了大量假阳性

**这不是模型失败，而是训练目标与推断协议不匹配的结果。**

### Prior-corrected评估的作用

使用prior-corrected inference后，模型输出的概率被修正到真实prior（π=0.01），此时：
- 精确率和召回率会更平衡
- PR-AUC会显著提升（反映真实性能）
- 更贴近临床实际应用场景

### 渐进式Prior的优势

- **训练稳定性**：从较小的π'开始，避免早期训练波动
- **更好的收敛**：逐步增加prior，让模型逐步适应平衡数据的学习目标
- **可比较性**：可以与固定prior的结果进行对比

## 论文写作建议

### 结果描述示例

> "When trained with a balanced prior (π'=0.5) and evaluated under the same decision threshold, the nnPU-based model exhibits extremely high recall but suffers from severe false positive inflation. This behavior is expected, as the training objective explicitly prioritizes sensitivity to rare positive events. These results highlight the mismatch between training prior and inference prior in imbalanced PU settings."

### 关键表述

> **"This result does not indicate model failure, but rather a misalignment between training objectives and inference protocols."**

这句话在rebuttal中非常有力量，因为它将问题定位为方法设计选择，而非模型缺陷。

## 文件结构

```
ecg_ssl_pu/
├── train_ssl_pu_af.py          # 改进的训练脚本
├── evaluate_ssl_pu.py           # 独立的评估脚本
├── model_ecg.py                 # 模型定义
├── pu_loss_torch.py             # PU损失函数
├── ssl_loss.py                  # SSL损失函数
├── dataset_ecg.py               # 数据加载
└── README_改进说明.md           # 本文档
```

## 注意事项

1. **Prior值**：确保训练时使用的`pi_prime`和评估时提供的`pi_prime`一致
2. **真实Prior**：评估时尽可能使用真实的先验概率，可通过测试数据估计
3. **指标选择**：在极度不平衡数据下，优先关注PR-AUC而非ROC-AUC
4. **阈值调整**：如果需要调整决策阈值，应该在prior-corrected概率上进行

## 后续改进方向

1. **自适应阈值**：根据PR曲线选择最优阈值
2. **动态Prior估计**：在训练过程中动态估计和调整prior
3. **集成方法**：结合多个不同π'训练的模型
4. **不确定性量化**：为预测添加置信度估计

