"""L2 [S1] 度量族验收测试（spec line2/20 [S1]）。

M4 校验用已知 Σ 的 4D 高斯云投影重构（归一化坐标，分箱偏差 ~5% 容差）。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from src.line2.data_pipeline.projections import six_projections, station_edges
from src.line2.metrics.metrics import (
    MetricUnderflow,
    bucketed_rel_rmse,
    evaluate_group,
    halo_fraction_gap,
    hf_energy_ratio_gap,
    poisson_deviance,
    reconstruct_sigma,
    rel_rmse,
    sigma_errors,
)

pytestmark = pytest.mark.unit


def _synthetic_table(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.normal(0.0, 3.3e-4, n),
            rng.normal(0.0, 2.9e-5, n),
            rng.normal(0.0, 3.3e-4, n),
            rng.normal(0.0, 2.9e-5, n),
            rng.normal(0.0, 2.4e-12, n),
            rng.normal(254.16, 0.27, n),
        ],
        axis=1,
    )


# ---------------------------------------------------------------- M1
def test_m1_identity_scale_zero() -> None:
    r = np.random.default_rng(0).integers(0, 500, (128, 128)).astype(np.int32)
    assert rel_rmse(r.copy(), r) == 0.0
    x = r.copy(); x[0, 0] += 10
    base = rel_rmse(x, r)
    assert base > 0
    assert abs(rel_rmse(x * 2, r) - base) < 1e-12  # 总计数归一 → 尺度不变
    with pytest.raises(MetricUnderflow):
        rel_rmse(np.zeros_like(r), r)


def test_m1_hand_example() -> None:
    x = np.array([[1000, 3000]]); r = np.array([[2000, 2000]])
    expect = np.sqrt(np.mean((x - r) ** 2)) / np.mean(r)  # 总量相等 → 归一恒等
    assert abs(rel_rmse(x, r) - expect) < 1e-12


# ---------------------------------------------------------------- M2
def test_m2_buckets() -> None:
    r = np.zeros((128, 128), dtype=np.int32)
    r[1, :100] = 1     # B1
    r[2, :100] = 5     # B2
    r[3, :100] = 50    # B3（其余行全 0 → B0，≥100 格）
    x = r.copy()
    x[0, :10] += 1      # B0 内扰动（总计数守恒：成对增减）
    x[0, 10:20] -= 1
    x[3, :50] += 3      # B3 内扰动
    x[3, 50:100] -= 3
    out = bucketed_rel_rmse(x, r)
    assert out["B0"]["rel_rmse"] > 0 and not out["B0"]["insufficient"]
    assert out["B1"]["rel_rmse"] == 0.0 and not out["B1"]["insufficient"]
    assert out["B2"]["rel_rmse"] == 0.0
    assert out["B3"]["rel_rmse"] > 0


# ---------------------------------------------------------------- M3
def test_m3_identity_and_hand() -> None:
    r = np.random.default_rng(1).integers(0, 300, (64, 64)).astype(np.int32)
    assert poisson_deviance(r.copy(), r) == 0.0
    c = np.array([[100, 0]]); mu = np.array([[90, 10]])  # Σμ = Σc = 100 → 归一恒等
    expect = 2 * ((100 * np.log(100 / 90) - (100 - 90)) + (0 + 10))  # c=0 项贡献 +μ
    assert abs(poisson_deviance(mu, c) - expect) < 1e-9


# ---------------------------------------------------------------- M4
def _gaussian_cloud_and_group(n: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    sig = np.diag([0.04, 0.09, 0.04, 0.09])          # 方差（归一化坐标）
    sig[0, 1] = sig[1, 0] = 0.5 * np.sqrt(sig[0, 0] * sig[1, 1])  # ρ(x,x′)=0.5，跨面=0
    cloud = rng.multivariate_normal(np.zeros(4), sig, size=n)
    table = np.column_stack([cloud, np.zeros(n), np.full(n, 254.16)])
    edges = station_edges(table)
    group = six_projections(table, edges)
    return cloud, table, edges, group


def test_m4_sigma_recovery_and_errors() -> None:
    cloud, table, edges, group = _gaussian_cloud_and_group(100_000, seed=5)
    sigma_hat = reconstruct_sigma(group, edges)
    sigma_ref = reconstruct_sigma(group, edges)
    d = np.sqrt(np.diag(sigma_hat))
    rho_hat = sigma_hat / np.outer(d, d)
    assert abs(rho_hat[0, 1] - 0.5) < 0.05
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        assert abs(rho_hat[i, j]) < 0.05
    errors = sigma_errors(sigma_hat, sigma_ref)
    assert all(v == 0.0 for v in errors["diag_rel"])
    assert all(v == 0.0 for v in errors["abs_delta_rho"])
    assert errors["eps4d_rel"] == 0.0


def test_m4_independent_subsample_errors_bounded() -> None:
    """半样本重构 vs 全量重构：M4 报告非零但有界的偏差（分桶语义存在性）。"""
    cloud, table, edges, _ = _gaussian_cloud_and_group(100_000, seed=6)
    rng = np.random.default_rng(0)
    sub = table[rng.choice(len(table), 10_000, replace=False)]
    g_full = six_projections(table, edges)
    g_sub = six_projections(sub, edges)
    e = sigma_errors(reconstruct_sigma(g_sub, edges), reconstruct_sigma(g_full, edges))
    assert max(e["diag_rel"]) < 0.2
    assert e["eps4d_rel"] < 0.2


# ---------------------------------------------------------------- M5
def test_m5_halo_gap() -> None:
    rng = np.random.default_rng(9)
    core = np.column_stack([rng.normal(0, 2e-4, (20_000, 4)), np.zeros(20_000), np.full(20_000, 254.16)])
    halo = np.column_stack([rng.normal(0, 2e-3, (5_000, 4)), np.zeros(5_000), np.full(5_000, 254.16)])
    table_ref = np.vstack([core, halo])
    edges = station_edges(table_ref)
    g_ref = six_projections(table_ref, edges)
    g_core = six_projections(core, edges)
    sigma_ref = reconstruct_sigma(g_ref, edges)
    assert halo_fraction_gap(g_core, g_ref, sigma_ref, edges) > 0.0  # 丢晕 → 正差
    assert halo_fraction_gap(g_ref, g_ref, sigma_ref, edges) == 0.0  # 一致 → 0


# ---------------------------------------------------------------- M6
def test_m6_hf_gap() -> None:
    img = np.zeros((128, 128), dtype=np.int32); img[64, 64] = 50_000  # 亮尖峰
    img_smooth = np.zeros_like(img); img_smooth[64, 64] = 20_000; img_smooth[63:66, 63:66] += 7_000
    group_a = np.stack([img] * 6); group_b = np.stack([img_smooth] * 6)
    ref = np.stack([img] * 6)
    assert hf_energy_ratio_gap(group_a, ref) == 0.0
    assert hf_energy_ratio_gap(group_b, ref) > 0.0  # 平滑版高频比更低


# ---------------------------------------------------------------- evaluate_group
def test_evaluate_group_underflow_and_identity() -> None:
    table = _synthetic_table(5_000, seed=11)
    edges = station_edges(table)
    g = six_projections(table, edges)
    out = evaluate_group(g, g, edges)
    assert out["underflow"] is False
    assert out["M1"] == 0.0 and out["M3"] == 0.0 and out["M5"] == 0.0 and out["M6"] == 0.0
    zero = np.zeros_like(g)
    out_z = evaluate_group(zero, g, edges)
    assert out_z["underflow"] is True
