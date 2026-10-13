"""L1 [S3] 束群构造：抖动线性变换 + 共同 FODO（P1 线性，spec line2/10 [S3]）。

抖动变换为精确线性映射（numpy 逐粒子实现）；共同 FODO 走 Ocelot 厚透镜跟踪
（``track(lat, p_array, calc_tws=False)``，v26 API），四极强度经二分求解使
实算每 cell 相位推进 = 目标 μ（偏差 < 1°，[S3] C1；相位推进标定用单 Particle
沿 ``lattice_track(lat, particle)`` 的传递矩阵迹 arccos(tr/2)——与生产传输同
一 Ocelot 实算路径）。

P1 档物理声明：传输仅作用于横向 4D (x, x′, y, y′)；纵向 (t, p) 列逐粒子原样
保留（线性横场的路径长度效应为二阶小量，纵向属本线 Out of Scope）——实现层
记录于本模块 docstring 与 .decisions.md。

两个等价入口：:func:`transport_beam`（锚点 + 束种子参数，生产路径）与
:func:`transport_with_params`（显式参数，单旋钮扫掠用）。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .seeds import JITTER_BRANCH, rng_for

__all__ = [
    "JITTER_NOMINAL_F", "JITTER_DELTA_RANGE", "JITTER_KICK_RAD", "JITTER_DRIFT_MM",
    "IDENTITY_JITTER",
    "FODO_N_CELLS", "FODO_MU_DEG", "FODO_HALF_DRIFT_M", "QUAD_LENGTH_M",
    "jitter_params", "beam_jitter_params", "apply_jitter",
    "transport_beam", "transport_with_params", "fodo_phase_advance_deg",
]

# 抖动分布（预注册，冻结于 spec line2/10 [S3] 批准）
JITTER_NOMINAL_F = 10.0          # m，前置薄透镜标称焦距
JITTER_DELTA_RANGE = (-0.05, +0.05)
JITTER_KICK_RAD = 10e-6          # ±10 µrad
JITTER_DRIFT_MM = (495.0, 505.0)

#: 无抖动基准束（`00` [S1] 的束 0）等价参数：零漂移、零踢角、无限大焦距
IDENTITY_JITTER: dict[str, float] = {
    "f_j": float("inf"),
    "theta_x_rad": 0.0,
    "theta_y_rad": 0.0,
    "drift_m": 0.0,
}

# 共同 FODO（定死，2026-09-14：M1 修复）
FODO_N_CELLS = 2
FODO_MU_DEG = 75.0
FODO_HALF_DRIFT_M = 2.5
QUAD_LENGTH_M = 0.1


def jitter_params(beam_idx: int) -> dict[str, float]:
    """束 i 的抖动参数（SeedSequence 派生，GC2 确定性）。束 0 由 transport_beam 特判为基准束。"""
    rng = rng_for(beam_idx, JITTER_BRANCH)
    delta = float(rng.uniform(*JITTER_DELTA_RANGE))
    return {
        "delta": delta,
        "f_j": JITTER_NOMINAL_F * (1.0 + delta),
        "theta_x_rad": float(rng.uniform(-JITTER_KICK_RAD, JITTER_KICK_RAD)),
        "theta_y_rad": float(rng.uniform(-JITTER_KICK_RAD, JITTER_KICK_RAD)),
        "drift_m": float(rng.uniform(*JITTER_DRIFT_MM)) / 1000.0,
    }


def apply_jitter(particles: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """精确线性映射：前置漂移 ℓ → 薄透镜 f_j（聚焦-x）→ 踢角。纵向列原样保留。"""
    out = particles.copy()
    x, xp, y, yp = (particles[:, i].copy() for i in range(4))
    x = x + xp * params["drift_m"]
    y = y + yp * params["drift_m"]
    xp = xp - x / params["f_j"]
    yp = yp + y / params["f_j"]
    xp = xp + params["theta_x_rad"]
    yp = yp + params["theta_y_rad"]
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = x, xp, y, yp
    return out


def _cell(f: float) -> list:
    from ocelot.cpbd.elements import Quadrupole, Drift

    k1 = 1.0 / (f * QUAD_LENGTH_M)
    return [
        Quadrupole(l=QUAD_LENGTH_M, k1=k1),
        Drift(l=FODO_HALF_DRIFT_M),
        Quadrupole(l=QUAD_LENGTH_M, k1=-k1),
        Drift(l=FODO_HALF_DRIFT_M),
    ]


@lru_cache(maxsize=1)
def fodo_lattice():
    """共同 FODO lattice（n_cells=2、μ=75°），模块级单例（确定性构建）。"""
    from ocelot.cpbd.magnetic_lattice import MagneticLattice

    elements: list = []
    for _ in range(FODO_N_CELLS):
        elements.extend(_cell(_solve_f()))
    return MagneticLattice(elements)


def fodo_phase_advance_deg(f: float) -> float:
    """给定聚焦焦距 f，Ocelot 实算的单 cell 相位推进（度）。

    单 Particle 笔形束两次过单 cell，提取 (x, x′) 传递矩阵对角元，
    μ = arccos(tr(M)/2)（μ ∈ [0°, 180°] 无歧义）。
    """
    from ocelot.cpbd.beam import Particle
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.track import lattice_track

    eps = 1e-9
    lat = MagneticLattice(_cell(f))
    m11 = m22 = None
    # 笔形束 A：x = ε 提取 m11；笔形束 B：x′ = ε 提取 m22（x 平面 2×2 块）
    pa = Particle()
    pa.x, pa.px, pa.y, pa.py = eps, 0.0, 0.0, 0.0
    lattice_track(lat, pa)
    m11 = pa.x / eps
    pb = Particle()
    pb.x, pb.px, pb.y, pb.py = 0.0, eps, 0.0, 0.0
    lattice_track(lat, pb)
    m22 = pb.px / eps
    mu = np.arccos(float(np.clip((m11 + m22) / 2.0, -1.0, 1.0)))
    return float(mu * 180.0 / np.pi)


def _solve_f(target_mu_deg: float = FODO_MU_DEG) -> float:
    """二分求解焦距：fodo_phase_advance_deg(f) = target（薄透镜解析式给初值与括号）。"""
    f0 = FODO_HALF_DRIFT_M / (2.0 * np.sin(np.radians(target_mu_deg) / 2.0))
    lo, hi = max(0.5 * FODO_HALF_DRIFT_M, f0 * 0.5), f0 * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fodo_phase_advance_deg(mid) > target_mu_deg:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def transport_beam(
    particles: np.ndarray, beam_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """单束两站快照（[S3] C2/C3）：站 0 = 抖动后，站 1 = FODO 后。

    束 0 为无抖动基准束（`00` [S1]）：恒等参数（零漂移/零踢角/无限大焦距）。
    输入/输出均为 (N, 6) 粒子表 (x, x′, y, y′, t, p)；行数守恒；确定性。
    """
    return transport_with_params(particles, beam_jitter_params(beam_idx))


def beam_jitter_params(beam_idx: int) -> dict[str, float]:
    """束 i 的两站传输参数：束 0 = 恒等基准，其余 = :func:`jitter_params` 的种子抽样。"""
    return dict(IDENTITY_JITTER) if beam_idx == 0 else jitter_params(beam_idx)


def transport_with_params(
    particles: np.ndarray, params: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """给定抖动参数的两站快照（站 0 = 抖动后，站 1 = 共同 FODO 后）。

    单旋钮扫掠（`50` [S2] C3/C4 的 θ/δ/ℓ 通道检验）经此入口注入参数；与
    :func:`transport_beam` 共用同一条传输代码路径。
    """
    from ocelot.cpbd.beam import ParticleArray
    from ocelot.cpbd.track import track

    s0 = apply_jitter(particles, params)
    p_arr = ParticleArray(s0.shape[0])
    p_arr.rparticles[:4, :] = s0[:, :4].T
    p_arr.rparticles[4:6, :] = 0.0
    p_arr.E = 130.0  # 参照动量（P1 横向线性传输无纵向依赖）
    _, p_arr = track(fodo_lattice(), p_arr, print_progress=False, calc_tws=False)
    s1 = s0.copy()
    s1[:, :4] = p_arr.rparticles[:4, :].T
    return s0, s1
