import torch
import torch.nn.functional as F


def simclr_loss(z1, z2, temperature=0.1):
    """
    标准 SimCLR InfoNCE 损失（完全不看标签）

    Args:
        z1, z2: [B, D]，来自同一批样本的两条增强视图
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    # 相似度矩阵 [2B,2B]
    sim_matrix = torch.matmul(z, z.T) / temperature

    # 屏蔽对角（自身与自身的相似度）
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)

    # 每个样本的正样本索引：同一批中另一视图的位置
    labels = torch.arange(2 * batch_size, device=z.device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def _supervised_contrastive_on_positives(z1, z2, y_pu, temperature=0.1):
    """
    仅在 batch 中的已标注正样本 (y_pu==1) 上做 supervised contrastive：
        - 所有正样本视图之间互为正对
        - 不使用负类标签信息

    Args:
        z1, z2: [B,D]
        y_pu: [B]，1 表示 P，-1 表示 U
    """
    device = z1.device
    y_pu = y_pu.view(-1)
    pos_mask = (y_pu == 1)
    n_pos = int(pos_mask.sum().item())
    if n_pos <= 1:
        # 没有或只有一个正样本，返回 0，不影响梯度
        return torch.tensor(0.0, device=device, dtype=z1.dtype)

    z_pos = torch.cat([z1[pos_mask], z2[pos_mask]], dim=0)  # [2P, D]
    z_pos = F.normalize(z_pos, dim=1)
    N = z_pos.size(0)

    sim = torch.matmul(z_pos, z_pos.T) / temperature  # [2P,2P]

    # mask 自身
    logits_mask = torch.ones((N, N), device=device, dtype=torch.bool)
    logits_mask.fill_(True)
    logits_mask.fill_diagonal_(False)

    # 所有其他视图都视为正对（因为都是同一类：P）
    positives_mask = logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    # 对每个 anchor：所有正对的相似度和 / 所有非自身样本的相似度和
    pos_sum = (exp_sim * positives_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1) + 1e-12

    log_prob = torch.log(pos_sum / all_sum + 1e-12)
    loss = -log_prob.mean()
    return loss


def pu_aware_ssl_loss(z1, z2, y_pu, temperature=0.1, alpha=1.0):
    """
    PU-aware SSL：
        L = L_simclr(所有样本) + alpha * L_supcon(仅在已标注正样本上)

    - 对 U 样本仍然是“纯无标签”的 SimCLR
    - 对 P 样本额外施加 supervised contrastive，使 P–P 之间表示更靠近
    - 不显式使用 P–U 或 U–U 的标签关系，保持 SSL 主体无监督，只注入弱先验
    """
    loss_unsup = simclr_loss(z1, z2, temperature=temperature)
    loss_sup = _supervised_contrastive_on_positives(z1, z2, y_pu, temperature=temperature)
    return loss_unsup + alpha * loss_sup

