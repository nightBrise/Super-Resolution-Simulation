"""L3b [S2] 损失：Poisson NLL + PSD 软一致罚项（spec line2/30 [S2]）。

率语义（D7）：rate = p · N_high；独立版与联合版共用同一 NLL 实现（C4）。
"""

from __future__ import annotations

import torch

from .unet import batch_sigma_from_counts

__all__ = ["poisson_nll", "psd_penalty", "total_loss"]


def poisson_nll(
    prob: torch.Tensor, target_counts: torch.Tensor, n_high: float
) -> torch.Tensor:
    """Poisson NLL（micro 平均）：L = mean_Ω [rate − c·ln(rate) + lgamma(c+1)]。

    rate = p·N_high（逐格期望计数）；c = 目标计数（2e5 全量直方图）。
    """
    rate = prob * float(n_high)
    c = target_counts.to(prob.dtype)
    nll = rate - c * torch.log(rate.clamp_min(1e-12)) + torch.lgamma(c + 1.0)
    return nll.mean()


def batch_lambda_min(prob: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """由输出概率图逐样本重构 Σ̂ 并返回 λ_min（(B,)）。"""
    sigma = batch_sigma_from_counts(prob, edges)
    return torch.linalg.eigvalsh(sigma)[:, 0]


def psd_penalty(prob: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """PSD 软一致罚项：mean_b relu(−λ_min(Σ̂_b))²（半正定 ⇒ 恰为 0）。"""
    lam_min = batch_lambda_min(prob, edges)
    return torch.relu(-lam_min).pow(2).mean()


def total_loss(
    prob: torch.Tensor,
    target_counts: torch.Tensor,
    edges: torch.Tensor,
    n_high: float,
    w_psd: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """总损失 = NLL + w_psd·P_PSD（联合版 w_psd>0；独立版 w_psd=0 且逐投影调用）。"""
    nll = poisson_nll(prob, target_counts, n_high)
    parts = {"nll": nll}
    total = nll
    if w_psd > 0:
        pen = psd_penalty(prob, edges)
        parts["psd"] = pen
        total = nll + w_psd * pen
    return total, parts
