"""L2 [S4]/[S5] 验收测试（spec line2/20）：观测量向量、容差线、矩匹配基线。"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from src.line2.data_pipeline.dataset import build_dataset
from src.line2.data_pipeline.projections import (
    AXES,
    N_BINS,
    PROJECTION_ORDER,
    six_projections,
    station_edges,
)
from src.line2.metrics.baseline import moment_matching_baseline
from src.line2.metrics.metrics import reconstruct_sigma
from src.line2.metrics.n_eff import error_n_curve
from src.line2.metrics.tolerance import (
    deviation_table,
    nearest_node,
    observable_vector,
    tolerance_for,
)
from tests.conftest import derive_seed

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    from tests.unit.test_line2_dataset import _synthetic_table

    out = tmp_path_factory.mktemp("ds_tol")
    anchor = _synthetic_table(100_001, seed=99)
    return build_dataset(out, anchor, n_beams=4, data_version="test_tol")


# ---------------------------------------------------------------- [S4] 观测量向量
def test_observable_vector_dims_and_identity(tmp_path: pathlib.Path) -> None:
    from tests.unit.test_line2_dataset import _synthetic_table

    table = _synthetic_table(20_000, seed=11)
    edges = station_edges(table)
    g = six_projections(table, edges)
    v = observable_vector(g, g, edges)
    assert v.shape == (13,) and (v == 0).all()  # 自比较全零


def test_nearest_node_mapping() -> None:
    nodes = (1_000, 3_000, 10_000, 30_000, 100_000)
    assert nearest_node(nodes, 9_000) == 10_000
    assert nearest_node(nodes, 1_100) == 1_000
    assert nearest_node(nodes, 20_000) == 30_000
    assert nearest_node(nodes, 500) == 1_000      # 下溢 → 端点档
    assert nearest_node(nodes, 500_000) == 100_000  # 上溢 → 端点档


# ---------------------------------------------------------------- [S4] 容差线
def test_deviation_table_and_tolerance(small_dataset: pathlib.Path) -> None:
    curve = error_n_curve(small_dataset, beam_ids=list(range(4)))
    dev = deviation_table(
        small_dataset, beam_ids=list(range(4)), tiers=(1_000, 10_000, 100_000), k=20
    )
    assert set(dev) == {1_000, 10_000, 100_000}
    for tier, vec in dev.items():
        assert vec.shape == (13,) and (vec >= 0).all() and np.isfinite(vec).all()
    # 容差线：N_eff = 9e3 → 最近档 1e4
    tol = tolerance_for(dev, curve_nodes=tuple(p.n_low for p in curve.points), n_eff=9_000)
    assert tol.shape == (13,) and (tol > 0).all()
    assert np.array_equal(tol, dev[10_000])


# ---------------------------------------------------------------- [S5] 矩匹配基线
def test_moment_matching_baseline(tmp_path: pathlib.Path) -> None:
    from tests.unit.test_line2_dataset import _synthetic_table

    table = _synthetic_table(50_000, seed=21)
    edges = station_edges(table)
    g = six_projections(table, edges)

    b1 = moment_matching_baseline(g, g, edges)
    b2 = moment_matching_baseline(g, g, edges)
    np.testing.assert_array_equal(b1, b2)  # C1 确定性
    assert b1.shape == (6, N_BINS, N_BINS) and (b1 >= 0).all()  # C2 接口
    for gi in range(6):  # 渲染总计数 = 参照逐投影总计数
        assert int(round(b1[gi].sum())) == int(g[gi].sum())
    # 自洽性：基线渲染的 Σ 相关结构与输入 Σ 一致（高斯自洽）
    s_in = reconstruct_sigma(g, edges)
    s_base = reconstruct_sigma(b1.astype(np.int32), edges)
    d_in = np.sqrt(np.diag(s_in))
    d_base = np.sqrt(np.diag(s_base))
    rho_in, rho_base = s_in / np.outer(d_in, d_in), s_base / np.outer(d_base, d_base)
    assert np.abs(rho_in - rho_base).max() < 0.05  # 相关结构保持


def _index_moments(img: np.ndarray) -> tuple[float, float]:
    """图内索引空间的二阶中心矩 (v_axis0, v_axis1)（单位：bin²）。"""
    p = img.astype(np.float64) / img.sum()
    k = np.arange(img.shape[0], dtype=np.float64)
    m0 = float((p.sum(axis=1) * k).sum())
    m1 = float((p.sum(axis=0) * k).sum())
    v0 = float((p.sum(axis=1) * (k - m0) ** 2).sum())
    v1 = float((p.sum(axis=0) * (k - m1) ** 2).sum())
    return v0, v1


def test_moment_matching_baseline_axes_match_group_layout() -> None:
    """[S5] C2（+ `20` [S1] C3）：B-moment 渲染落在与图组相同的 (i,j) 轴序上。

    自洽性断言（渲染与输入互测）在上下游同错时会自抵消，故本条以 Σ 与站级网格为锚：
    存储布局 axis0 = aj（间距 h_j）、axis1 = ai（间距 h_i）⇒ 渲染图的索引空间二阶矩
    应为 v0 = Σ_jj/h_j²、v1 = Σ_ii/h_i²，且与输入图组自身的同轴矩一致（渲染是同一束的
    高斯拟合）。束取"核心高斯 + 6 个位于 +9σ_x 的远端粒子"：晕粒子把 x 的站级网格拉宽，
    使各轴 σ/h 之比互异（10.2 vs 18.3），从而"轴 0 = aj"与"轴 0 = ai"两种读法可分辨。

    实测（按存储布局 / 按"轴 0 = ai"误读）：与 Σ 期望的最大相对偏差 1×10⁻⁴ / 1.03
    （误读把 x 轴的宽度错放到另一存储轴上）；与输入图组的最大相对偏差 4×10⁻⁴ / 1.03。
    """
    table = _gaussian_core_with_far_halo()
    edges = station_edges(table)
    group = six_projections(table, edges)
    sigma = reconstruct_sigma(group, edges)
    render = moment_matching_baseline(group, group, edges)

    for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
        i, j = AXES.index(ai), AXES.index(aj)
        hi, hj = float(np.diff(edges[i])[0]), float(np.diff(edges[j])[0])
        expected = (sigma[j, j] / hj**2, sigma[i, i] / hi**2)  # 存储轴 0 = aj、轴 1 = ai
        got = _index_moments(render[gi])
        reference = _index_moments(group[gi])
        for axis_name, g, e, r in (("axis0", got[0], expected[0], reference[0]),
                                   ("axis1", got[1], expected[1], reference[1])):
            assert abs(g / e - 1.0) < 0.05, (
                f"投影 {gi} {PROJECTION_ORDER[gi]} 的 {axis_name} 索引矩 {g} 偏离 Σ 期望 {e}"
            )
            assert abs(g / r - 1.0) < 0.05, (
                f"投影 {gi} {PROJECTION_ORDER[gi]} 的 {axis_name} 索引矩 {g} 偏离输入图组 {r}"
            )


def _gaussian_core_with_far_halo(n: int = 200_000) -> np.ndarray:
    """核心 4D 高斯束 + 6 个 +9σ_x 远端粒子：使 x 的站级网格显著宽于其它轴。

    各轴 σ/h 之比互异是鉴别力的来源：若各轴 σ/h 相同，把渲染的高斯宽窄放到另一轴上
    会与 Σ 的互换恰好抵消（旧实现在 (x,x′) 与 (y,y′) 两投影上正是如此）。
    """
    corr = np.array(
        [[1.00, 0.60, 0.30, -0.20], [0.60, 1.00, 0.35, 0.10],
         [0.30, 0.35, 1.00, -0.45], [-0.20, 0.10, -0.45, 1.00]]
    )
    scales = np.array([3.3e-4, 4.0e-5, 2.2e-4, 5.0e-5])
    rng = np.random.default_rng(derive_seed(30, 4))
    table = (rng.standard_normal((n, 4)) @ np.linalg.cholesky(corr).T) * scales
    table[:6, 0] = 9.0 * scales[0]
    return table
