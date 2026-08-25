"""种子派生测试：60 [S14] C4（SeedSequence 分支规则、生成器不自选随机源）。

覆盖规格：60 [S14] C4/C8、05 [S3.2] ★ test_no_self_random_source
（禁用全局 np.random 时生成函数仍正常——随机源全来自 SeedSequence 分支）、
05 [S6] C1（固定 TEST_MASTER_SEED 派生、禁止裸随机调用）。

测试铁律（05 [S1]）：只断言协议与不变量（种子派生可复现、分支独立、无自选
随机源），不断言研究结果。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generators.dataset_builder import _generate_sample, derive_sample_seed
from src.generators.f_beam import f_beam
from src.generators.f_deg import f_deg
from src.generators.f_prior import f_prior
from src.generators.sampling import sample_parameters

pytestmark = [pytest.mark.unit, pytest.mark.m2]

MASTER = 20260825
N_BRANCHES = 8


def _branch_states(master_seed: int, n: int = N_BRANCHES) -> list[np.ndarray]:
    """``SeedSequence(master_seed).spawn(n)`` 各分支的初始状态。"""
    return [
        ss.generate_state(4, dtype=np.uint64)
        for ss in np.random.SeedSequence(master_seed).spawn(n)
    ]


def test_same_master_seed_same_branches():
    """60 [S14] C4：同一 master_seed 派生的分支逐位一致（可复现）。"""
    a = _branch_states(MASTER)
    b = _branch_states(MASTER)
    for sa, sb in zip(a, b):
        assert np.array_equal(sa, sb)


def test_different_master_seed_different_branches():
    """不同 master_seed 派生的分支序列不同。"""
    a = _branch_states(MASTER)
    b = _branch_states(MASTER + 1)
    assert not any(np.array_equal(sa, sb) for sa, sb in zip(a, b))


def test_different_branches_different_sequences():
    """同一 master_seed 的不同分支产生不同的随机序列（划分间不共享）。"""
    states = _branch_states(MASTER)
    for i in range(1, N_BRANCHES):
        assert not np.array_equal(states[0], states[i])


def test_derive_sample_seed_reproducible():
    """样本种子派生可复现：同 (master_seed, split, index) 恒一致。"""
    s1 = derive_sample_seed(MASTER, "train", 7, 20_000)
    s2 = derive_sample_seed(MASTER, "train", 7, 20_000)
    assert s1 == s2
    assert isinstance(s1, int)


def test_derive_sample_seed_distinct_indices():
    """不同样本索引派生不同种子（同一 c 的噪声实现由种子区分）。"""
    seeds = {derive_sample_seed(MASTER, "train", i, 20_000) for i in range(64)}
    assert len(seeds) == 64


def test_derive_sample_seed_distinct_splits():
    """不同划分的种子流独立：同索引跨划分种子不同。"""
    seeds = {
        derive_sample_seed(MASTER, split, 0, 1_000)
        for split in ("train", "val", "test_id", "test_pb", "test_ood")
    }
    assert len(seeds) == 5


def test_sample_parameters_seed_reproducible():
    """同 master_seed 的 sample_parameters 两次调用输出逐位一致。"""
    a, stats_a = sample_parameters(8, master_seed=MASTER, sigma_K=11.0)
    b, stats_b = sample_parameters(8, master_seed=MASTER, sigma_K=11.0)
    assert a == b
    assert stats_a["n_accepted"] == stats_b["n_accepted"]


def test_fdeg_seed_reproducible():
    """同 seed 的 f_deg 噪声实现逐位一致（30 [S3] C4）。"""
    from src.generators.masks import DELTA_PX, fine_structure_width

    c = {
        "A": 1.0, "sigma_z": 0.5, "n": 1.5, "eta": 0.1, "b0": 0.06,
        "a1": 0.5, "alpha": -1.0, "a2": 0.05, "a3": -0.03, "beta": 1.2,
        "gamma": -0.3, "b1": 0.05, "C": 0.5,
    }
    H, _, _ = f_beam(c, sigma_smooth=0.5 * fine_structure_width(c) / DELTA_PX)
    L1 = f_deg(H, sigma_K=11.0, sigma_n=1.2e-4, seed=42)[0]
    L2 = f_deg(H, sigma_K=11.0, sigma_n=1.2e-4, seed=42)[0]
    assert np.array_equal(L1, L2)


def test_no_self_random_source(monkeypatch, consistent_c):
    """★ 禁用全局 np.random 时生成函数仍正常（60 [S14] C4）。

    随机源必须全部来自 SeedSequence 分支；若任何生成函数调用全局
    ``np.random.*``，本测试立即失败。
    """

    def _raise(*_args, **_kwargs):
        raise AssertionError("生成函数调用了全局 np.random 随机源")

    for name in (
        "normal", "rand", "randn", "uniform", "choice", "random",
        "shuffle", "permutation", "random_sample", "randint", "standard_normal",
    ):
        monkeypatch.setattr(np.random, name, _raise)

    from src.generators.masks import DELTA_PX, fine_structure_width

    c = dict(consistent_c)
    sigma_smooth_h = 0.5 * float(fine_structure_width(c) / DELTA_PX)
    H, m, c_rec = f_beam(c, sigma_smooth=sigma_smooth_h)          # 20：无随机源
    L, L_clean, d, m_L = f_deg(H, sigma_K=11.0, sigma_n=1.2e-4, seed=5)  # 30：SeedSequence
    P, meta = f_prior(c, level="P2", sigma_smooth=2.6 * sigma_smooth_h)   # 40：无随机源

    params, stats = sample_parameters(6, master_seed=MASTER, sigma_K=11.0)
    assert len(params) == 6
    assert stats["n_accepted"] == 6

    # 数据集样本级生成器（并行 worker 顶层函数）同样不触全局随机源
    rec = _generate_sample(
        ("train", 0, params[0], derive_sample_seed(MASTER, "train", 0, 100), 11.0,
         1.2e-4, "D2")
    )
    assert rec["H"].shape == (256, 256)
    assert rec["L"].shape == (64, 64)
    assert rec["P2"].shape == (256, 256)
    assert rec["seed_i"] == derive_sample_seed(MASTER, "train", 0, 100)
    assert rec["m_L"]["seed"] == rec["seed_i"]
