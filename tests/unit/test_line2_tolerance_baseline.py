"""L2 [S4]/[S5] 验收测试（spec line2/20）：观测量向量、容差线、矩匹配基线。"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from src.line2.data_pipeline.dataset import build_dataset
from src.line2.data_pipeline.projections import N_BINS, six_projections, station_edges
from src.line2.metrics.baseline import moment_matching_baseline
from src.line2.metrics.metrics import reconstruct_sigma
from src.line2.metrics.n_eff import error_n_curve
from src.line2.metrics.tolerance import (
    deviation_table,
    nearest_node,
    observable_vector,
    tolerance_for,
)

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
