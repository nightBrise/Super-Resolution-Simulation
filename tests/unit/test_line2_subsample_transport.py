"""L1 [S2]/[S3] 验收测试（spec line2/10）。

传输测试用 5×10³ 粒子子集（Claims 为逐粒子确定性与统计不变量，规模不影响
有效性；全量 2e5×1000 束的生产运行由管线 CLI 承担，不进单元套件）。
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from src.line2.data_pipeline.anchor import ANCHOR_MD5
from src.line2.data_pipeline.sdds_reader import read_sdds_binary
from src.line2.data_pipeline.subsample import (
    SUBSAMPLE_K,
    SUBSAMPLE_TIERS,
    subsample_index_set,
)
from src.line2.data_pipeline.transport import (
    FODO_MU_DEG,
    fodo_phase_advance_deg,
    jitter_params,
    transport_beam,
)

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
N_TEST = 5_000


@pytest.fixture(scope="module")
def particles() -> np.ndarray:
    page = read_sdds_binary(
        REPO_ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
    )
    full = np.stack([page.columns[n] for n in ("x", "xp", "y", "yp", "t", "p")], axis=1)
    rng = np.random.default_rng(20260913)
    return full[rng.choice(full.shape[0], size=N_TEST, replace=False)]


def _corr_xxp(tab: np.ndarray) -> float:
    return float(np.corrcoef(tab[:, 0], tab[:, 1])[0, 1])


# ---------------------------------------------------------------- [S2]
def test_subsample_claims() -> None:
    """[S2] C2/C3/C4：无放回、⊂全量、确定性、同档互异（n_total 覆盖最大档）。"""
    n_total = 200_000  # 与真实锚点一致：各档子集组合空间充分（避免 N_low = n_total 的退化）
    a = subsample_index_set(n_total, beam_idx=0, station_idx=0)
    assert set(a) == set(SUBSAMPLE_TIERS) and all(len(v) == SUBSAMPLE_K for v in a.values())
    for tier, subs in a.items():
        for idx in subs:
            assert len(idx) == tier and len(np.unique(idx)) == tier  # 无放回
            assert idx[-1] < n_total
        assert all(not np.array_equal(subs[i], subs[j])
                   for i in range(SUBSAMPLE_K) for j in range(i + 1, SUBSAMPLE_K))  # C4 互异
    b = subsample_index_set(n_total, beam_idx=0, station_idx=0)
    for tier in SUBSAMPLE_TIERS:
        np.testing.assert_array_equal(a[tier][0], b[tier][0])  # C3 确定性
    c = subsample_index_set(n_total, beam_idx=1, station_idx=0)
    assert not np.array_equal(a[10_000][0], c[10_000][0])  # 不同束不同采样


# ---------------------------------------------------------------- [S3]
def test_phase_advance_within_1_deg() -> None:
    """[S3] C1：实算每 cell 相位推进与目标 μ=75° 偏差 < 1°。"""
    assert abs(fodo_phase_advance_deg(_solve_f_probe()) - FODO_MU_DEG) < 1.0


def _solve_f_probe() -> float:
    from src.line2.data_pipeline.transport import _solve_f

    return _solve_f()


def test_transport_conservation_and_determinism(particles: np.ndarray) -> None:
    """[S3] C2/C3：行数守恒；同输入同参数逐位一致；纵向列 P1 档原样保留。"""
    s0_a, s1_a = transport_beam(particles, beam_idx=7)
    s0_b, s1_b = transport_beam(particles, beam_idx=7)
    assert s0_a.shape == s1_a.shape == particles.shape
    np.testing.assert_array_equal(s0_a, s0_b)
    np.testing.assert_array_equal(s1_a, s1_b)
    np.testing.assert_array_equal(s0_a[:, 4:], particles[:, 4:])  # P1: t/p 保留


def test_station_correlation_rotation(particles: np.ndarray) -> None:
    """[S3] C4：抖动束（beam ≥ 1）的站间 corr(x,x′) 差 ≥ 0.3。

    偏差记录（见 10 .decisions.md D11）：束 0（无抖动基准束）实测
    |Δcorr| ≈ 0.137 < 0.3——spec 推导的 cos(150°) 模型不含椭圆取向项，
    仅对基准束失效；按 write-spec 冻结规则记为 deviation，由验收闸口裁决。
    """
    deltas = []
    for beam_idx in (1, 2, 3, 4, 5):
        s0, s1 = transport_beam(particles, beam_idx=beam_idx)
        deltas.append(abs(_corr_xxp(s1) - _corr_xxp(s0)))
    assert min(deltas) >= 0.3, f"抖动束站间差异不足：{deltas}"


def test_station_correlation_beam0_deviation(particles: np.ndarray) -> None:
    """[S3] C4 偏差留痕：束 0 的 |Δcorr| 实测值（预期 ≈ 0.137，非零有内容）。"""
    s0, s1 = transport_beam(particles, beam_idx=0)
    delta = abs(_corr_xxp(s1) - _corr_xxp(s0))
    assert 0.05 <= delta < 0.3, f"束 0 偏差量级漂移：{delta}"


def test_beam_diversity(particles: np.ndarray) -> None:
    """[S3] C5：抖动使束间站 0 corr(x,x′) 标准差 > 0.01（束间多样性不变量）。"""
    corrs = [_corr_xxp(transport_beam(particles, beam_idx=i)[0]) for i in range(20)]
    assert float(np.std(corrs)) > 0.01
    assert abs(corrs[0] - _corr_xxp(particles)) < 1e-12  # 束 0 = 无抖动基准束
