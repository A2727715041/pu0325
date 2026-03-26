import torch
import torch.nn as nn


class PULoss(nn.Module):
    """
    标准 nnPU 损失：
        y_pred: [B]，模型输出 f(x)
        t: [B]，取值 1（正类 P）或 -1（未标记 U）
    公式参考 Kiryo et al., NIPS 2017
    """

    def __init__(self, prior, loss_fn=None, gamma=1.0, beta=0.0, nnpu=True):
        super().__init__()
        assert 0.0 < prior < 1.0
        self.prior = float(prior)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.nnpu = bool(nnpu)
        # 默认使用 sigmoid(-f(x))，与原 chainer 实现保持一致
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x: torch.sigmoid(-x))

    def forward(self, y_pred, t):
        """
        y_pred: [B] 或 [B,1]
        t: [B]，1 或 -1
        """
        device = y_pred.device
        y_pred = y_pred.view(-1, 1)
        t = t.view(-1, 1).to(device)

        positive = (t == 1).float()
        unlabeled = (t == -1).float()

        n_p = torch.clamp(positive.sum(), min=1.0)
        n_u = torch.clamp(unlabeled.sum(), min=1.0)

        y_positive = self.loss_fn(y_pred)      # l(f(x))
        y_unlabeled = self.loss_fn(-y_pred)    # l(-f(x))

        positive_risk = self.prior * (positive * y_positive).sum() / n_p
        negative_risk = (unlabeled / n_u - self.prior * positive / n_p) * y_unlabeled
        negative_risk = negative_risk.sum()

        objective = positive_risk + negative_risk

        if self.nnpu and negative_risk.item() < -self.beta:
            # non-negative 修正
            loss = positive_risk - self.beta
        else:
            loss = objective

        return loss


class ImbalancedPULoss(nn.Module):
    """
    Imbalanced nnPU（ImbalancednnPU）损失：
    通过对正风险 / 负风险重新加权，模拟在“平衡数据”上学习的效果。

    思路：
        - 原始数据类先验为 π (self.prior)
        - 目标“平衡”先验为 π' (pi_prime, 例如 0.5)
        - 在风险中对正/负项使用：
              w_p = π' / π
              w_n = (1-π') / (1-π)
          从而让学习目标更接近于在先验为 π' 的平衡数据上训练。
    """

    def __init__(self, prior, pi_prime=0.5, loss_fn=None, gamma=1.0, beta=0.0, nnpu=True):
        super().__init__()
        assert 0.0 < prior < 1.0
        assert 0.0 < pi_prime < 1.0
        self.prior = float(prior)          # 原始数据中的类先验 π
        self.pi_prime = float(pi_prime)    # 目标“平衡”先验 π'
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.nnpu = bool(nnpu)
        self.loss_fn = loss_fn if loss_fn is not None else (lambda x: torch.sigmoid(-x))

    def forward(self, y_pred, t):
        device = y_pred.device
        y_pred = y_pred.view(-1, 1)
        t = t.view(-1, 1).to(device)

        positive = (t == 1).float()
        unlabeled = (t == -1).float()

        n_p = torch.clamp(positive.sum(), min=1.0)
        n_u = torch.clamp(unlabeled.sum(), min=1.0)

        y_positive = self.loss_fn(y_pred)
        y_unlabeled = self.loss_fn(-y_pred)

        # 标准 nnPU 中的正/负风险
        positive_risk = self.prior * (positive * y_positive).sum() / n_p
        negative_risk = (unlabeled / n_u - self.prior * positive / n_p) * y_unlabeled
        negative_risk = negative_risk.sum()

        # 重新加权，模拟在先验为 pi_prime 的平衡数据上学习
        w_p = self.pi_prime / self.prior
        w_n = (1.0 - self.pi_prime) / (1.0 - self.prior)

        positive_risk_w = w_p * positive_risk
        negative_risk_w = w_n * negative_risk

        objective = positive_risk_w + negative_risk_w

        if self.nnpu and negative_risk_w.item() < -self.beta:
            loss = positive_risk_w - self.beta
        else:
            loss = objective

        return loss



