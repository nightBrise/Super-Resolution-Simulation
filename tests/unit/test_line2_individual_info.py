"""`50` [S3] C5：个体信息仪器（增量对齐 + 输入打乱）的合成验收。

合成用例的偏差方向已知：预测的增量 d_out 与目标的增量 d_tgt 取同一束个体方向
（cos_true = 1），打乱侧改喂另一束的方向（与自身方向近似正交 ⇒ cos_scr ≈ 0）。
故本文件把"仪器能否把个体信息与模板化输出分开"变成可证伪的数值断言。

仪器只消费预测图组与束表（不训练、不推理），合成图用 16² 尺寸以控测试耗时
（生产尺寸 128²，形状契约是 ``(子样本, 6, H, W)``）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.line2.metrics.alignment import (
    alignment_instrument,
    bootstrap_median_ci,
    derangement,
    scramble_pairs,
)
from tests.conftest import derive_seed

pytestmark = pytest.mark.unit

H = W = 16
N_BEAMS = 12
N_SUBS = 3
N_HIGH = 1000.0      # 参照（全量）逐投影总计数
DELTA_AMP = 0.01     # 预测侧个体方向幅度（相对 prior）
TARGET_AMP = 0.02    # 目标侧个体方向幅度


def _prior_group() -> np.ndarray:
    """逐投影归一的 prior 图组（各投影和为 1；生产口径即 B-const 均值图）。"""
    grid = np.arange(H, dtype=np.float64)[:, None]
    img = np.exp(-((grid - 0.5 * H) ** 2 + (grid.T - 0.5 * H) ** 2) / (2.0 * 2.0**2))
    out = np.stack([img * (1.0 + 0.1 * gi) for gi in range(6)])
    return out / out.sum(axis=(1, 2), keepdims=True)


def _beam_directions(n: int) -> np.ndarray:
    """n 个"束个体方向"：逐投影零和（扣 prior 后总计数不变）、逐束单位范数。"""
    rng = np.random.default_rng(derive_seed(50, 7))
    dirs = rng.normal(size=(n, 6, H, W))
    dirs -= dirs.mean(axis=(2, 3), keepdims=True)
    dirs /= np.linalg.norm(dirs.reshape(n, -1), axis=1)[:, None, None, None]
    return dirs


def _synthetic_case() -> dict:
    """真输入/打乱输入的预测图组与束表（打乱配对取自仪器的固定 derangement）。"""
    prior = _prior_group()
    beam_ids = [100 + i for i in range(N_BEAMS)]
    dirs = _beam_directions(N_BEAMS)
    index = {b: i for i, b in enumerate(beam_ids)}
    pairs = scramble_pairs(beam_ids)
    predictions, scrambled, targets = {}, {}, {}
    for beam_id in beam_ids:
        i = index[beam_id]
        j = index[pairs[beam_id]]
        predictions[beam_id] = np.stack(
            [prior + DELTA_AMP * dirs[i]] * N_SUBS
        )
        scrambled[beam_id] = np.stack(
            [prior + DELTA_AMP * dirs[j]] * N_SUBS
        )
        targets[beam_id] = (prior + TARGET_AMP * dirs[i]) * N_HIGH
    return {
        "predictions": predictions,
        "scrambled": scrambled,
        "targets": targets,
        "prior": prior,
        "beam_ids": beam_ids,
        "pairs": pairs,
    }


# ---------------------------------------------------------------- C5 主判据
def test_alignment_instrument_synthetic() -> None:
    """[S3] C5：偏差方向已知的构造数据上，cos_true ≥ 0.999、打乱后 |cos_scr| ≤ 0.1。

    构造使 d_out 与 d_tgt 严格平行（同一单位方向、只差标量幅度）⇒ cos_true = 1
    （浮点级偏差）；打乱侧换到另一束方向，高维单位向量的内积近零 ⇒ cos_scr ≈ 0。
    同时核对由同一构造给出的闭式读数：drop_frac = (1 − cos_scr)/1 ≈ 1、
    disp_ratio = ‖Δ_π − Δ‖/‖Δ‖ = √2（正交单位方向）。
    """
    case = _synthetic_case()
    out = alignment_instrument(
        case["predictions"], case["scrambled"], case["targets"], case["prior"]
    )

    assert out["cos_true"] >= 0.999, out["cos_true"]
    assert abs(out["cos_scr"]) <= 0.1, out["cos_scr"]
    assert out["drop_frac"] >= 0.9, out["drop_frac"]
    assert abs(out["disp_ratio"] - np.sqrt(2.0)) < 0.1, out["disp_ratio"]
    # 束级读数逐束成立（不是中位数偶然落在阈值内）
    assert all(v >= 0.999 for v in out["cos_true_per_beam"].values())
    assert all(abs(v) <= 0.1 for v in out["cos_scr_per_beam"].values())
    # 契约键齐备（输出 = cos_true / cos_scr / drop_frac / disp_ratio）
    for key in ("cos_true", "cos_scr", "drop_frac", "disp_ratio"):
        assert key in out
    assert out["n_beams"] == N_BEAMS and out["n_subs"] == N_SUBS
    assert out["cos_true_ci95"][0] > 0.999  # 正对齐的 CI 亦排除 0


def test_alignment_instrument_is_deterministic() -> None:
    """同输入两次调用逐位一致（含 bootstrap CI；`20` [S1] C1 的确定性口径）。"""
    case = _synthetic_case()
    first = alignment_instrument(
        case["predictions"], case["scrambled"], case["targets"], case["prior"], n_boot=500
    )
    second = alignment_instrument(
        case["predictions"], case["scrambled"], case["targets"], case["prior"], n_boot=500
    )
    assert first == second


def test_alignment_instrument_pairing_is_the_instrument_derangement() -> None:
    """打乱配对 = 固定 derangement（无不动点），且随结论落盘供复核。"""
    case = _synthetic_case()
    pairs = case["pairs"]
    assert sorted(pairs) == sorted(case["beam_ids"])
    assert all(pairs[b] != b for b in pairs), "derangement 不应含不动点"
    assert sorted(pairs.values()) == sorted(case["beam_ids"]), "配对须是双射"
    out = alignment_instrument(
        case["predictions"], case["scrambled"], case["targets"], case["prior"], n_boot=200
    )
    assert out["pairing"] == pairs
    assert out["pairing"] == scramble_pairs(case["beam_ids"]), "换输入序不得改变配对口径"
    perm = derangement(N_BEAMS)
    assert sorted(int(v) for v in perm) == list(range(N_BEAMS))
    assert not np.any(perm == np.arange(N_BEAMS))


def test_alignment_instrument_prior_only_output_reports_undefined() -> None:
    """零信息输出（out ≡ prior）⇒ d_out ≡ 0：cos / drop / disp 无定义报 nan，不伪造 0 对齐。"""
    case = _synthetic_case()
    prior = case["prior"]
    prior_only = {b: np.stack([prior] * N_SUBS) for b in case["beam_ids"]}
    out = alignment_instrument(prior_only, prior_only, case["targets"], prior, n_boot=200)
    assert np.isnan(out["cos_true"]) and np.isnan(out["cos_scr"])
    assert np.isnan(out["drop_frac"]) and np.isnan(out["disp_ratio"])
    assert all(np.isnan(v) for v in out["cos_true_ci95"])
    assert all(np.isnan(out["cos_true_per_beam"][b]) for b in case["beam_ids"])


def test_alignment_instrument_rejects_misaligned_tables() -> None:
    """束集/形状不符时显式报错（不静默对齐到交集）。"""
    case = _synthetic_case()
    dropped = dict(case["scrambled"])
    dropped.pop(case["beam_ids"][0])
    with pytest.raises(ValueError, match="束集不一致"):
        alignment_instrument(case["predictions"], dropped, case["targets"], case["prior"])
    bad = dict(case["predictions"])
    bad[case["beam_ids"][0]] = bad[case["beam_ids"][0]][:, :3]
    with pytest.raises(ValueError, match="形状"):
        alignment_instrument(bad, case["scrambled"], case["targets"], case["prior"])


def test_bootstrap_median_ci_reports_n_and_ignores_nonfinite() -> None:
    """bootstrap 的 n 只计有限值；全非有限时返回 nan 而非抛错。"""
    values = np.array([1.0, 2.0, 3.0, np.nan])
    stats = bootstrap_median_ci(values, n_boot=200)
    assert stats["n"] == 3 and stats["median"] == 2.0
    empty = bootstrap_median_ci(np.array([np.nan, np.inf]), n_boot=10)
    assert empty["n"] == 0 and np.isnan(empty["median"])
