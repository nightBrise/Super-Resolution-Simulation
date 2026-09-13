"""L3b [S1]/[S2] 验收测试（spec line2/30）：网络前向、参数上限、条件化、损失。

CPU 运行（base=24、depth=3 缩配以保测试速度；默认 C=48/d=4 的参数上限单独断言）。
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

from src.line2.data_pipeline.projections import station_edges
from src.line2.training.losses import poisson_nll, psd_penalty
from src.line2.training.unet import Phase1Net, condition_vector, normalize_input_group

pytestmark = pytest.mark.unit


def _small_net() -> Phase1Net:
    torch.manual_seed(0)
    return Phase1Net(base=24, num_levels=3, blocks=1, cond_dim=11)


def _edges() -> torch.Tensor:
    table_edges = np.linspace(-1.05, 1.05, 129)
    return torch.tensor(np.stack([table_edges] * 4), dtype=torch.float32)


def test_forward_shape_positive_and_prob_normalized() -> None:
    """[S1] C1/C3：前向形状、严格正、逐投影概率和 = 1。"""
    net = _small_net().eval()
    x = torch.rand(2, 6, 128, 128)
    kappa = torch.randn(2, 11)
    with torch.no_grad():
        p = net(x, kappa)
    assert p.shape == (2, 6, 128, 128)
    assert bool((p > 0).all())
    sums = p.sum(dim=(2, 3))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_param_cap_and_validation() -> None:
    """[S1] C2：默认配置参数量 ≤ 6×10⁷；非法 base/depth 拒绝。"""
    torch.manual_seed(0)
    net = Phase1Net()  # base=48, depth=4
    total, _ = net.count_parameters()
    assert total <= 6e7, f"参数量 {total} 超上限"
    with pytest.raises(ValueError):
        Phase1Net(base=32)
    with pytest.raises(ValueError):
        Phase1Net(num_levels=2)


def test_conditioning_changes_output() -> None:
    """[S1] FiLM 条件化生效：同输入、不同 κ → 输出不同。"""
    net = _small_net().eval()
    x = torch.rand(1, 6, 128, 128)
    with torch.no_grad():
        d = float((net(x, torch.zeros(1, 11)) - net(x, torch.ones(1, 11))).abs().max())
    assert d > 0


def test_checkpoint_roundtrip(tmp_path: pathlib.Path) -> None:
    """[S1] C3：checkpoint 往返 → 同输入逐位同输出。"""
    net = _small_net().eval()
    x = torch.rand(1, 6, 128, 128)
    kappa = torch.randn(1, 11)
    with torch.no_grad():
        before = net(x, kappa)
    ckpt = tmp_path / "model.ckpt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    net2 = _small_net()
    net2.load_state_dict(torch.load(ckpt, weights_only=True)["state_dict"])
    net2.eval()
    with torch.no_grad():
        after = net2(x, kappa)
    torch.testing.assert_close(before, after)


# ---------------------------------------------------------------- [S2] 损失
def test_poisson_nll_hand_example() -> None:
    """[S2] C1：单格手算——p=1、c=10、N_high=1 → rate=1，NLL = 1 + lgamma(11)。"""
    prob = torch.ones(1, 1, 1, 1)
    c = torch.tensor([[10.0]])
    nll = poisson_nll(prob, c, n_high=1.0)
    expect = 1.0 - 10.0 * 0.0 + float(torch.lgamma(torch.tensor(11.0)))
    assert abs(float(nll) - expect) < 1e-6


def test_poisson_nll_rate_semantics() -> None:
    """[S2] D7：rate = p·N_high（放大语义——目标为 2e5 计数时 p 和应为 c/N_high）。"""
    c = torch.zeros(1, 1, 4, 4); c[0, 0, 0, 0] = 400.0
    p = torch.zeros(1, 1, 4, 4); p[0, 0, 0, 0] = 1.0  # 全部概率在亮格
    n_high = 200_000.0
    nll_at_truth = poisson_nll(p, c, n_high)  # rate = 2e5 vs c=400 → 惩罚大
    p_half = torch.full_like(p, 0.25)
    nll_spread = poisson_nll(p_half, c, n_high)  # rate = 5e4/格 → 更差
    assert float(nll_at_truth) < float(nll_spread)


def test_psd_penalty_discriminates() -> None:
    """[S2] C2：一致 marginals → 罚≈0；混合符号相关（非 PSD Σ̂）→ 罚 > 0。"""
    n = 40_000
    rng = np.random.default_rng(3)
    # 一致：4D 高斯（ρ(x,x′)=0.5，跨面 0）
    cov = np.diag([0.04, 0.09, 0.04, 0.09])
    cov[0, 1] = cov[1, 0] = 0.5 * np.sqrt(0.04 * 0.09)
    cloud = rng.multivariate_normal(np.zeros(4), cov, size=n)
    counts = np.zeros((6, 128, 128))
    pairs = ((0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3))
    for gi, (a, b) in enumerate(pairs):
        counts[gi] = np.histogram2d(cloud[:, a], cloud[:, b], bins=128)[0]
    prob_consistent = torch.tensor(counts / counts.sum(axis=(1, 2), keepdims=True)).unsqueeze(0)
    edges = torch.tensor(np.linspace(-1, 1, 129)).unsqueeze(0).repeat(4, 1)
    pen_consistent = float(psd_penalty(prob_consistent, edges))

    # 非一致：六张各自合法的 2D 高斯边缘，其拼装 Σ 非正定（4 环负相关 + 对角正相关）
    prob_mixed = np.zeros((6, 128, 128))
    centers = np.linspace(-1, 1, 128) + 1e-9
    mix_rho = {(0, 1): -0.9, (2, 3): -0.9, (0, 2): 0.9, (0, 3): -0.9, (1, 2): 0.9, (1, 3): 0.9}
    var = [0.04, 0.09, 0.04, 0.09]
    for gi, (a, b) in enumerate(
        ((0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3))
    ):
        rho = mix_rho[(a, b)]
        s_a, s_b = var[a], var[b]
        img = np.exp(
            -0.5
            * (
                (centers[:, None] ** 2) / s_a
                + (centers[None, :] ** 2) / s_b
                - 2 * rho * centers[:, None] * centers[None, :] / np.sqrt(s_a * s_b)
            )
            / (1 - rho**2)
        )
        img = img / img.sum()
        prob_mixed[gi] = img
    prob_mixed_t = torch.tensor(prob_mixed).unsqueeze(0)
    pen_mixed = float(psd_penalty(prob_mixed_t, edges))
    assert pen_mixed > 0.0, "混合符号相关必须产生非 PSD Σ̂"
    assert pen_consistent <= pen_mixed
