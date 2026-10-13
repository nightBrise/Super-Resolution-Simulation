"""`50` [S3] C5：个体信息仪器（增量对齐 + 输入打乱），种子 20260825。

读数（全部在 `20` [S1] 的总计数归一空间；`50` [S1] 定义）：

- ``cos_true`` = med cos(d_out, d_tgt)，d_out = out − prior、d_tgt = target − prior
  （prior = 该站 B-const，六投影展平）；
- ``cos_scr`` = 输入按**固定 derangement** 打乱后的同款读数——打乱后的输入对含该束的
  κ（由打乱后输入派生），故打乱预测须由调用方喂 π(b) 束的输入前向得到；
- ``drop_frac`` = (cos_true − cos_scr)/cos_true；
- ``disp_ratio`` = med‖out_scr − out_true‖ / med‖out_true − prior‖（两处 med 均按
  `50` [S1] 的聚合口径：先逐束取中位数、再对束取中位数，然后相除）。

聚合按 `50` [S1] 的统一口径：逐 (束, 子样本) 读数 → 逐束取中位数 → 对 150 个束级读数
取中位数（cos_true 的 95% CI = 束级 paired bootstrap，10,000 次，SeedSequence(根种子)）。
本仪器只消费**预测图组**与束表（不训练、不推理）：打乱侧的输出由调用方用 π(b) 的输入
前向得到，配对表由 :func:`scramble_pairs` 给出（同一根种子 ⇒ 同一置换）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from . import ANALYSIS_SEED
from .metrics import _normalized

__all__ = [
    "derangement",
    "scramble_pairs",
    "alignment_instrument",
    "bootstrap_median_ci",
]

N_BOOT = 10_000


def derangement(n: int, *, seed: int = ANALYSIS_SEED) -> np.ndarray:
    """确定性错位置换（无不动点）：固定种子流拒绝采样；期望 ~2.7 次抽取。"""
    if n < 2:
        raise ValueError(f"derangement 需要至少 2 个元素，收到 {n}")
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm.astype(np.int64)
    raise RuntimeError("derangement 构造失败（1000 次抽取仍有不动点）")


def scramble_pairs(beam_ids: Sequence[int], *, seed: int = ANALYSIS_SEED) -> dict[int, int]:
    """打乱配对的单一来源：束 b 的输入换成束 π(b) 的（π 为固定 derangement）。

    调用方按本表喂入 π(b) 束的输入（六图 + 由该输入派生的 κ）得到打乱侧预测，
    再把预测交给 :func:`alignment_instrument`。
    """
    ids = [int(b) for b in beam_ids]
    perm = derangement(len(ids), seed=seed)
    return {ids[i]: ids[int(perm[i])] for i in range(len(ids))}


def bootstrap_median_ci(
    values: np.ndarray, *, n_boot: int = N_BOOT, seed: int = ANALYSIS_SEED
) -> dict:
    """束级 paired bootstrap 的中位数 95% CI（非有限值按有限样本重抽）。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"median": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    idx = rng.integers(0, finite.size, size=(int(n_boot), finite.size))
    medians = np.median(finite[idx], axis=1)
    return {
        "median": float(np.median(finite)),
        "ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
        "n": int(finite.size),
    }


def _cos_flattened(a: np.ndarray, b: np.ndarray) -> float:
    """六投影展平的余弦相似度（任一侧零范数 → nan，不伪造 0）。"""
    flat_a, flat_b = a.ravel(), b.ravel()
    na, nb = float(np.linalg.norm(flat_a)), float(np.linalg.norm(flat_b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(np.dot(flat_a, flat_b) / (na * nb))


def _nanmedian(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _normalized_group(group: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """总计数归一（`20` [S1] C1）：逐投影各按自身参照总计数重标后拼回图组。

    M1 图组口径的同一实现点（``group_rel_rmse`` 逐投影调 ``rel_rmse``）：把整组一次性
    重标会与逐投影口径分叉（各投影总计数失衡时），故此处逐投影调用同一归一函数。
    """
    return np.stack([_normalized(group[gi], ref[gi]) for gi in range(group.shape[0])])


def _group_stack(value: np.ndarray, beam_id: int, name: str) -> np.ndarray:
    """预测图组集合的形状守卫：``(子样本, 6, H, W)``（生产尺寸 128²）。"""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != 6:
        raise ValueError(
            f"{name}[{beam_id}] 形状 {arr.shape} ≠ (子样本, 6, H, W)"
        )
    return arr


def alignment_instrument(
    predictions: Mapping[int, np.ndarray],
    scrambled: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    prior: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    """个体信息仪器：预测图组集合 + prior + 束表 → cos_true / cos_scr / drop_frac / disp_ratio。

    Args:
        predictions: 束号 → (子样本, 6, H, W) 真输入下的预测图组（计数或概率图组，
            逐投影总计数在仪器内按 `20` [S1] 归一契约重标到该束参照总计数；
            生产尺寸 H = W = 128）。
        scrambled: 束号 → 打乱侧预测图组（同形状），其输入为 π(束)（见
            :func:`scramble_pairs`）；与 ``predictions`` 逐键对齐。
        targets: 束表——束号 → (6, H, W) 全量目标图组（参照 R）。
        prior: (6, H, W) 该站 B-const 均值图（逐投影归一后为 1；仪器内按束重标）。
        n_boot: bootstrap 次数（`50` [S1] 冻结 10,000）。
        seed: 分析根种子（`50` [S1] C4：20260825）。

    Returns:
        ``{"cos_true", "cos_scr", "drop_frac", "disp_ratio", "cos_true_ci95",
        "cos_scr_ci95", "cos_true_per_beam", "cos_scr_per_beam", "n_beams",
        "n_subs", "seed", "pairing", "bootstrap"}``。

    Raises:
        ValueError: 三张表的键集不一致、形状不符或子样本数不一致。
        src.line2.metrics.metrics.MetricUnderflow: 某束的预测图组总计数为 0
            （`20` [S1] C5 的下溢情形）。
    """
    prior_group = np.asarray(prior, dtype=np.float64)
    if prior_group.ndim != 3 or prior_group.shape[0] != 6:
        raise ValueError(f"prior 形状 {prior_group.shape} ≠ (6, H, W)")
    beam_ids = sorted(int(b) for b in predictions)
    if not beam_ids:
        raise ValueError("alignment_instrument：预测图组集合为空")
    if sorted(int(b) for b in scrambled) != beam_ids:
        raise ValueError("alignment_instrument：打乱侧预测与真输入预测的束集不一致")
    if sorted(int(b) for b in targets) != beam_ids:
        raise ValueError("alignment_instrument：束表与预测图组的束集不一致")

    cos_true_per_beam: dict[int, float] = {}
    cos_scr_per_beam: dict[int, float] = {}
    disp_num: list[list[float]] = []
    disp_den: list[list[float]] = []
    n_subs = None
    for beam_id in beam_ids:
        ref = np.asarray(targets[beam_id], dtype=np.float64)
        if ref.shape != prior_group.shape:
            raise ValueError(
                f"束表[{beam_id}] 形状 {ref.shape} ≠ prior 形状 {prior_group.shape}"
            )
        out_true = _group_stack(predictions[beam_id], beam_id, "predictions")
        out_scr = _group_stack(scrambled[beam_id], beam_id, "scrambled")
        if out_true.shape[0] != out_scr.shape[0]:
            raise ValueError(f"束 {beam_id}：真输入与打乱侧的子样本数不一致")
        if n_subs is None:
            n_subs = int(out_true.shape[0])
        elif int(out_true.shape[0]) != n_subs:
            raise ValueError(f"束 {beam_id}：子样本数与其它束不一致")
        prior_n = _normalized_group(prior_group, ref)
        d_tgt = ref - prior_n
        cos_true, cos_scr, num, den = [], [], [], []
        for sub_idx in range(out_true.shape[0]):
            d_out = _normalized_group(out_true[sub_idx], ref) - prior_n
            d_scr = _normalized_group(out_scr[sub_idx], ref) - prior_n
            cos_true.append(_cos_flattened(d_out, d_tgt))
            cos_scr.append(_cos_flattened(d_scr, d_tgt))
            den.append(float(np.linalg.norm(d_out)))
            num.append(float(np.linalg.norm(d_scr - d_out)))
        cos_true_per_beam[beam_id] = _nanmedian(np.asarray(cos_true))
        cos_scr_per_beam[beam_id] = _nanmedian(np.asarray(cos_scr))
        disp_num.append(num)
        disp_den.append(den)

    true_stats = bootstrap_median_ci(
        np.asarray(list(cos_true_per_beam.values())), n_boot=n_boot, seed=seed
    )
    scr_stats = bootstrap_median_ci(
        np.asarray(list(cos_scr_per_beam.values())), n_boot=n_boot, seed=seed
    )
    # 逐 (束, 子样本) 读数按 [S1] 聚合口径先取束级中位数，再对束取中位数后相除
    num_median = _nanmedian(np.asarray([np.median(v) for v in disp_num]))
    den_median = _nanmedian(np.asarray([np.median(v) for v in disp_den]))
    disp_ratio = float(num_median / den_median) if den_median > 0.0 else float("nan")
    cos_true = float(true_stats["median"])
    drop_frac = (
        float((cos_true - float(scr_stats["median"])) / cos_true)
        if np.isfinite(cos_true) and cos_true != 0.0
        else float("nan")
    )
    return {
        "cos_true": cos_true,
        "cos_scr": float(scr_stats["median"]),
        "drop_frac": drop_frac,
        "disp_ratio": disp_ratio,
        "cos_true_ci95": true_stats["ci95"],
        "cos_scr_ci95": scr_stats["ci95"],
        "cos_true_per_beam": cos_true_per_beam,
        "cos_scr_per_beam": cos_scr_per_beam,
        "n_beams": len(beam_ids),
        "n_subs": int(n_subs or 0),
        "seed": int(seed),
        "pairing": scramble_pairs(beam_ids, seed=seed),
        "bootstrap": {"n_boot": int(n_boot), "root_seed": int(seed), "unit": "beam"},
    }
