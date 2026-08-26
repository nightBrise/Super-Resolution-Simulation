"""测试全局配置：主种子、自洽性示例参数与共享 fixture。

覆盖规格：05 [S6] C1（TEST_MASTER_SEED=20260825 派生、禁止裸随机调用）、
20 [S9] 自洽性示例（σ_z=0.5、n=1.5、η=0.1、b₀=0.06、a₁=0.5、C=0.5 →
α=−1、a₂=0.05、β=1.2、a₃=−0.03、γ=−0.3、b₁=0.05；派生量
σ_δ≈0.257、S≈0.56、w_fine≈4.3px，满足 W1–W8）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 使项目根可导入（测试以 `src.` 包前缀引用生产代码）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.masks import DELTA_PX, fine_structure_width  # noqa: E402

#: 测试主种子；全部随机测试从该种子经 SeedSequence 派生（05 [S6] C1）。
TEST_MASTER_SEED = 20260825

#: 自洽性示例的模糊核宽度（20 [S9] 自洽性示例：σ_K ≈ 9 px）。
SIGMA_K_PX = 9.0


@pytest.fixture(scope="session")
def consistent_c() -> dict[str, float]:
    """20 [S9] 自洽性示例参数组合（满足 W1–W8，σ_K=9px）。"""
    return {
        "A": 1.0,
        "sigma_z": 0.5,
        "n": 1.5,
        "eta": 0.1,
        "b0": 0.06,
        "a1": 0.5,
        "alpha": -1.0,
        "a2": 0.05,
        "a3": -0.03,
        "beta": 1.2,
        "gamma": -0.3,
        "b1": 0.05,
        "C": 0.5,
    }


@pytest.fixture(scope="session")
def sigma_K_px() -> float:
    """自洽性示例配套的模糊核宽度（H 像素单位）。"""
    return SIGMA_K_PX


@pytest.fixture(scope="session")
def w_fine_px(consistent_c) -> float:
    """自洽性示例的精细结构宽度（H 像素单位，≈4.28）。"""
    return float(fine_structure_width(consistent_c) / DELTA_PX)


@pytest.fixture(scope="session")
def sigma_smooth_H_px(w_fine_px) -> float:
    """H 渲染平滑核 σ_smooth,H = 0.125×w_fine（20 [S3] C4 逐样本口径，
    2026-08-26 P0 修订：原 0.5× 废弃见 99 OQ-20-03）。"""
    return 0.125 * w_fine_px


@pytest.fixture(scope="session")
def sigma_smooth_P_px(sigma_smooth_H_px) -> float:
    """先验平滑核初始值 σ_smooth,P = 2×σ_smooth,H（40 [S5] C5 初始候选中心）。"""
    return 2.0 * sigma_smooth_H_px


@pytest.fixture(scope="session")
def beam_sample(consistent_c, sigma_smooth_H_px):
    """自洽性示例参数的 ``(H, m, c)`` 三元组（会话级缓存，20 [S4]）。"""
    from src.generators.f_beam import f_beam

    H, m, c = f_beam(consistent_c, sigma_smooth=sigma_smooth_H_px)
    return H, m, c


@pytest.fixture(scope="session")
def degraded_sample(beam_sample, sigma_K_px):
    """自洽性示例的退化四元组 ``(L, L_clean, d, m_L)``（30 [S11]，σ_n 取尾部信噪比≈3 档）。"""
    from src.generators.f_deg import f_deg

    H, _, _ = beam_sample
    sigma_n = float(H.sum() / 64.0**2 / 3.0)  # mean(L_clean)/σ_n ≈ 3
    L, L_clean, d, m_L = f_deg(H, sigma_K=sigma_K_px, sigma_n=sigma_n, seed=TEST_MASTER_SEED)
    return L, L_clean, d, m_L


@pytest.fixture()
def rng():
    """由主种子派生的独立随机数生成器（05 [S6]：禁止裸全局随机态）。"""
    return np.random.default_rng(np.random.SeedSequence(TEST_MASTER_SEED))


def derive_seed(*labels) -> int:
    """由主种子与标签派生确定性整数种子，供需要整型种子的接口使用。"""
    ss = np.random.SeedSequence([TEST_MASTER_SEED, *labels])
    return int(ss.generate_state(1, dtype=np.uint32)[0])
