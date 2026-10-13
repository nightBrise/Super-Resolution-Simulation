"""`50` [S3] C3：v2 容差线重造（M6 分半去偏）与去偏程序的可证伪性验收。

两个层次：

1. **实测判据**（:func:`test_m6_tolerance_debiased_is_noise_line`）：读 v2 E0' 尺子产物
   （``results/tolerance_v2.json`` + M6 原始读数 npz），从**原始读数**独立重算分半去偏
   与验证判据——判据 = 验证半区去偏后 |中位|/σ < 1 且 95 分位/(1.96σ) ∈ [0.8, 1.2]；
   并核对落盘的 M6 行确实按去偏口径给出（远小于未去偏的机械偏移量）。
2. **程序性验收**（:func:`test_m6_debias_split_half_synthetic`）：合成分布上验证
   去偏量 = 估计半区中位数、共同偏移被扣除后判据成立，而重尾分布判据失败——
   该判据不是恒真的。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.line2.metrics.tolerance_v2 import (
    HALF_NORMAL_P95,
    M6_INDEX,
    M6_NOISE_P95_RANGE,
    m6_debias,
    split_half,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "studies/line2_particle_sr/results"
TOLERANCE_PATH = RESULTS / "tolerance_v2.json"
READINGS_PATH = RESULTS / "tolerance_v2_m6_train.npz"


def test_m6_tolerance_debiased_is_noise_line() -> None:
    """[S3] C3：v2 的 M6 容差行经分半去偏后是一条噪声线（判据在真实读数上可证伪）。

    证据链：从 npz 的 train 带符号读数独立重算分半 → 偏差中位数 → 验证半区判据；
    与落盘 JSON 的 ``m6_debias`` 段逐项对账（偏差、判据值、布尔结论）。产物的
    M6 行（13 维向量下标 12）须落在**去偏后的噪声尺度**上：小于未去偏的机械偏移
    量 |bias|——若该行退回 v1 的未去偏口径（v1 实测 |中位|/σ = 24.7，行的量级即
    偏移量本身），本条断言失败。
    """
    if not TOLERANCE_PATH.is_file() or not READINGS_PATH.is_file():
        pytest.skip(
            f"v2 容差线产物不在本机：{TOLERANCE_PATH.name} / {READINGS_PATH.name}"
            "（`50` [S5] C1 的 E0' 尺子产物，*.json 入 results/ 不入库）"
        )
    payload = json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))
    readings = np.load(READINGS_PATH)
    per_tier = payload["m6_debias"]["per_tier"]
    seed = int(payload["m6_debias"]["seed"])
    lo, hi = M6_NOISE_P95_RANGE

    assert payload["stations"] == [0], "`50` [S3] C3：v2 容差线为站 0"
    assert payload["quantile"] == 0.95
    assert len(payload["tiers"]) == 3
    for tier in payload["tiers"]:
        raw = np.asarray(readings[f"tier_{int(tier)}"], dtype=np.float64)
        rec = per_tier[str(int(tier))]
        assert raw.ndim == 1 and raw.size == 700 * 20, (tier, raw.shape)

        idx_est, idx_val = split_half(raw.size, seed=seed)
        bias = float(np.median(raw[idx_est]))
        sigma = float(np.std(raw[idx_val], ddof=1))
        debiased = raw[idx_val] - bias
        residual_ratio = abs(float(np.median(debiased))) / sigma
        p95_ratio = float(np.percentile(np.abs(debiased), 95.0)) / (
            HALF_NORMAL_P95 * sigma
        )

        # 落盘记录 = 独立重算（去偏量、判据数值、布尔结论）
        assert rec["bias"] == pytest.approx(bias, rel=1e-12, abs=1e-15)
        assert rec["val_abs_median_over_sigma"] == pytest.approx(residual_ratio, rel=1e-9)
        assert rec["val_p95_over_1p96sigma"] == pytest.approx(p95_ratio, rel=1e-9)
        assert rec["is_noise_line"] is True
        # 判据本体（不依赖落盘布尔，直接对原始读数成立）
        assert residual_ratio < 1.0
        assert lo <= p95_ratio <= hi
        # 反差：未去偏的 |中位|/σ 远超 1（v1 同口径实测 24.7，量的是机械偏移）
        assert rec["raw_abs_median_over_sigma"] > 1.0
        # 落盘的 M6 行 = 噪声尺度（远小于偏移量本身）
        row = float(payload["dev_table"][str(int(tier))][M6_INDEX])
        assert 0.0 < row < abs(bias), (tier, row, bias)
        assert len(payload["dev_table"][str(int(tier))]) == 13

    # 产物 ↔ 消费侧接口：门禁的容差读取器按整型档位还原整表（Phase 1.5 直接按档索引，
    # 不再经 N_eff 最近档映射）
    from src.line2.experiments.gates import _load_tolerance

    table = _load_tolerance(TOLERANCE_PATH)
    assert table is not None
    assert sorted(table) == [int(t) for t in payload["tiers"]]
    for tier, vec in table.items():
        assert vec.shape == (13,) and np.isfinite(vec).all() and (vec >= 0.0).all()
        assert vec[M6_INDEX] == payload["dev_table"][str(tier)][M6_INDEX]


def test_m6_debias_split_half_synthetic() -> None:
    """分半去偏的程序性验收：偏差 = 估计半区中位数；判据在半正态上成立、重尾上失败。

    若把判据写成"在估计半区自验证"（`50` D4 已否决的写法），(a) 的 |中位|/σ 恒为 0；
    若判据没有容忍带（如写成 p95/(1.96σ) == 1），(b) 的重尾样本会显得"通过"而
    实际的半正态一致性并不成立——两例共同钉住判据的鉴别力。
    """
    rng = np.random.default_rng(np.random.SeedSequence(20260825).spawn(1)[0])

    offset, sigma_true, n = 0.0275, 0.0011, 20_000
    gauss = offset + rng.normal(0.0, sigma_true, n)
    rec = m6_debias(gauss)
    assert rec["bias"] == pytest.approx(offset, abs=5.0 * sigma_true / np.sqrt(n / 2))
    assert rec["is_noise_line"] is True
    assert rec["raw_abs_median_over_sigma"] > 5.0     # 未去偏时"不是噪声线"
    assert rec["val_abs_median_over_sigma"] < 1.0

    heavy = rng.exponential(1.0, n)
    heavy_rec = m6_debias(heavy)
    assert heavy_rec["is_noise_line"] is False
    assert heavy_rec["val_p95_over_1p96sigma"] > 1.2  # 指数分布的 |X| 95 分位 ≈ 3.0σ

    idx_est, idx_val = split_half(n)
    assert idx_est.size == idx_val.size == n // 2
    assert sorted(np.concatenate([idx_est, idx_val]).tolist()) == list(range(n))
    assert np.array_equal(idx_est, split_half(n)[0])  # 同种子逐位一致

    with pytest.raises(ValueError, match="至少 4"):
        m6_debias(np.zeros(3))
    with pytest.raises(ValueError, match="一维"):
        m6_debias(np.zeros((2, 2)))
