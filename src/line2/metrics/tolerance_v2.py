"""`50` [S3] C3：v2 容差线重造（13 项 × 3 训练档 × 站 0）与 M6 分半去偏。

沿用 `20` [S4] 的口径：观测量 q 的容差 = 该档裸子样本 |q_裸 − q_参照| 分布的 95% 分位。
v2 重造的三处变更（其余不变）：

1. 档集 = 训练档 {3×10³, 1×10⁴, 1×10⁵}（`50` [S4] C2）、站 = 站 0（`50` [S3] C3）；
2. 数据 = v2 公共域（`50` [S2]），故容差线随边层重建；
3. **M6 行去偏**（`50` [S3] C3 / D4）：M6 的容差量在 v1 实测不是噪声线——裸子样本
   的带符号偏差 |中位|/σ = 24.7，度量的是子采样的机械高通偏移。故以一半 train
   子样本的符号偏差中位数为去偏量 b（只吃 train 束的读数，`00` GC3），容差 = 去偏后
   |读数 − b| 的 95 分位；b 的迁移性在另一半 train 读数上按 :func:`m6_debias` 判据验证。

M6 在 13 维向量里先取 |·|（符号被抹掉），去偏需要符号信息，故本模块另用
:func:`src.line2.metrics.metrics.signed_hf_energy_ratio_gap` 取带符号读数；其余 12 个
分量仍走 :func:`src.line2.metrics.tolerance.observable_vector`（单一实现点）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from . import ANALYSIS_SEED
from .metrics import signed_hf_energy_ratio_gap
from .n_eff import _beam_split, _dataset_layout
from .tolerance import N_OBSERVABLES, observable_vector

__all__ = [
    "V2_TIERS",
    "QUANTILE",
    "M6_INDEX",
    "M6_NOISE_MEDIAN_LIMIT",
    "M6_NOISE_P95_RANGE",
    "HALF_NORMAL_P95",
    "split_half",
    "m6_debias",
    "m6_signed_readings",
    "deviation_table_v2",
]

#: `50` [S4] C2 的训练档 = `50` [S3] C3 的容差线档集（三档各一条线）
V2_TIERS = (3_000, 10_000, 100_000)
QUANTILE = 0.95
#: 13 维观测量向量的 M6 分量下标（`20` [S4]：M4 的 11 维 + M5 + M6）
M6_INDEX = N_OBSERVABLES - 1
#: 半正态一致性：|X| 的 95 分位 = 1.96σ（X ~ N(0, σ²)）
HALF_NORMAL_P95 = 1.96
#: M6 验证判据（`50` [S3] C3）：去偏后 |中位|/σ < 1，95 分位/(1.96σ) ∈ [0.8, 1.2]
M6_NOISE_MEDIAN_LIMIT = 1.0
M6_NOISE_P95_RANGE = (0.8, 1.2)


def split_half(n: int, *, seed: int = ANALYSIS_SEED) -> tuple[np.ndarray, np.ndarray]:
    """把 n 个读数分成估计半区与验证半区（返回下标数组，确定性）。

    以 SeedSequence(seed) 派生置换取前/后半——不用奇偶或相邻序号切分，避免与束序、
    子样本序等生产端顺序发生系统耦合。n 为奇数时验证半区多取一个。
    """
    if n < 2:
        raise ValueError(f"分半需要至少 2 个读数，收到 {n}")
    perm = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]).permutation(n)
    half = n // 2
    return perm[:half], perm[half:]


def m6_debias(
    train_signed: np.ndarray, *, seed: int = ANALYSIS_SEED, quantile: float = QUANTILE
) -> dict:
    """M6 的分半去偏与"是噪声线"验证（`50` [S3] C3）。

    估计半区取符号偏差中位数 b；验证半区扣除 b 后按下述两条判据判定：

    - ``|median(验证半区 − b)| / σ < 1``（残差偏移不超过一个 σ）；
    - ``P95(|验证半区 − b|) / (1.96σ) ∈ [0.8, 1.2]``（半正态一致性 = 噪声线形状）。

    σ 取验证半区**带符号**读数的标准差（ddof=1，与前置④的 |中位|/σ 口径一致）。
    判据在估计半区自身上按构造恒真（减去自身中位数后中位数恒为 0），分半是该判据
    可证伪的前提。未去偏的 |中位|/σ 一并留痕作为反差（v1 同口径实测 24.7）。

    Args:
        train_signed: 一维数组，train 束的 M6 带符号读数（逐 (束, 子样本)）。
        seed: 分半用的根种子（`50` [S1] C4）。
        quantile: 分位口径（`20` [S4] 冻结 0.95）。

    Returns:
        落盘用字典：``bias``（去偏量）、``raw_abs_median_over_sigma``、
        ``val_abs_median_over_sigma``、``val_p95_over_1p96sigma``、``is_noise_line``、
        样本量与阈值快照。

    Raises:
        ValueError: 读数不足 4 个或不是一维。
    """
    values = np.asarray(train_signed, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"M6 读数须为一维，收到 shape={values.shape}")
    if values.size < 4:
        raise ValueError(f"M6 去偏需要至少 4 个读数，收到 {values.size}")
    idx_est, idx_val = split_half(values.size, seed=seed)
    est, val = values[idx_est], values[idx_val]
    bias = float(np.median(est))
    sigma = float(np.std(val, ddof=1))
    debiased = val - bias
    if sigma > 0.0:
        raw_ratio = abs(float(np.median(val))) / sigma
        residual_ratio = abs(float(np.median(debiased))) / sigma
        p95_ratio = float(np.percentile(np.abs(debiased), quantile * 100)) / (
            HALF_NORMAL_P95 * sigma
        )
    else:
        raw_ratio = residual_ratio = p95_ratio = float("nan")
    lo, hi = M6_NOISE_P95_RANGE
    is_noise_line = bool(
        np.isfinite(residual_ratio)
        and np.isfinite(p95_ratio)
        and residual_ratio < M6_NOISE_MEDIAN_LIMIT
        and lo <= p95_ratio <= hi
    )
    return {
        "bias": bias,
        "n_est": int(idx_est.size),
        "n_val": int(idx_val.size),
        "raw_abs_median_over_sigma": raw_ratio,
        "val_median": float(np.median(debiased)),
        "val_sigma": sigma,
        "val_p95_abs": float(np.percentile(np.abs(debiased), quantile * 100)),
        "val_abs_median_over_sigma": residual_ratio,
        "val_p95_over_1p96sigma": p95_ratio,
        "is_noise_line": is_noise_line,
        "thresholds": {
            "abs_median_over_sigma_max": M6_NOISE_MEDIAN_LIMIT,
            "p95_over_1p96sigma_range": [float(lo), float(hi)],
        },
        "seed": int(seed),
    }


def _split_beams(manifest: dict, scope: str) -> list[int]:
    """按口径列出束号（``valtest`` = val+test 池化，`20` D15）。"""
    if scope == "valtest":
        return [int(b) for s in ("val", "test") for b in manifest["splits"][s]]
    if scope not in ("train", "val", "test"):
        raise ValueError(f"beams_scope 未知：{scope!r}（∈ train/val/test/valtest）")
    return [int(b) for b in manifest["splits"][scope]]


def m6_signed_readings(
    data_dir: str | Path,
    beam_ids: list[int],
    *,
    tier: int,
    station: int = 0,
    k: int = 20,
    progress_every: int | None = None,
) -> np.ndarray:
    """逐 (束, 子样本) 的 M6 带符号读数（该档该站裸子样本 vs 全量目标图组）。"""
    manifest, data_dir = _dataset_layout(data_dir)
    files: dict[str, h5py.File] = {}
    out: list[float] = []
    try:
        for n_done, beam_id in enumerate(beam_ids, start=1):
            split = _beam_split(manifest, beam_id)
            h5 = files.setdefault(split, h5py.File(data_dir / f"{split}.h5", "r"))
            stg = h5[f"beam_{beam_id:04d}"][f"station{station}"]
            ref = stg["target_full"][...]
            for sub_idx in range(k):
                img = stg[f"tier_{tier:06d}_sub_{sub_idx:02d}"][...]
                out.append(signed_hf_energy_ratio_gap(img, ref))
            if progress_every and n_done % progress_every == 0:
                print(
                    f"[tol-v2] M6 读数 {n_done}/{len(beam_ids)} 束（档 {tier}）",
                    file=sys.stderr, flush=True,
                )
    finally:
        for fh in files.values():
            fh.close()
    return np.asarray(out, dtype=np.float64)


def deviation_table_v2(
    data_dir: str | Path,
    *,
    tiers: tuple[int, ...] = V2_TIERS,
    station: int = 0,
    k: int = 20,
    beams_scope: str = "valtest",
    seed: int = ANALYSIS_SEED,
    quantile: float = QUANTILE,
    progress_every: int | None = None,
) -> dict:
    """v2 容差线表（13 项 × 各档）与 M6 去偏记录（`50` [S3] C3）。

    13 维向量的 12 个非 M6 分量取 ``observable_vector`` 原口径的 95 分位；M6 分量取
    ``|带符号读数 − b|`` 的 95 分位，b = train 束估计半区的中位数偏差。M6 的 b 按档
    独立估计（`20` [S4]：容差线按档独立构造）。

    Returns:
        ``{"dev_table": {tier: (13,)}, "m6_debias": {tier: dict},
        "m6_signed_train": {tier: (n,)}, "tiers", "scope", "station", "k", "n_beams",
        "n_train_beams", "quantile", "seed"}``；原始 train 读数随结果返回供落盘复核。
    """
    manifest, data_dir = _dataset_layout(data_dir)
    beams = _split_beams(manifest, beams_scope)
    train_beams = _split_beams(manifest, "train")
    m6_signed_train: dict[int, np.ndarray] = {}
    m6_records: dict[int, dict] = {}
    for tier in tiers:
        signed = m6_signed_readings(
            data_dir, train_beams, tier=int(tier), station=station, k=k,
            progress_every=progress_every,
        )
        m6_signed_train[int(tier)] = signed
        m6_records[int(tier)] = m6_debias(signed, seed=seed, quantile=quantile)

    readings: dict[int, list[np.ndarray]] = {int(t): [] for t in tiers}
    files: dict[str, h5py.File] = {}
    try:
        for n_done, beam_id in enumerate(beams, start=1):
            split = _beam_split(manifest, beam_id)
            h5 = files.setdefault(split, h5py.File(data_dir / f"{split}.h5", "r"))
            stg = h5[f"beam_{beam_id:04d}"][f"station{station}"]
            ref = stg["target_full"][...]
            edges = np.asarray(stg["target_full"].attrs["bin_edges"], dtype=np.float64)
            for tier in tiers:
                bias = m6_records[int(tier)]["bias"]
                for sub_idx in range(k):
                    img = stg[f"tier_{int(tier):06d}_sub_{sub_idx:02d}"][...]
                    vec = observable_vector(img, ref, edges)
                    vec[M6_INDEX] = abs(signed_hf_energy_ratio_gap(img, ref) - bias)
                    readings[int(tier)].append(vec)
            if progress_every and n_done % progress_every == 0:
                print(f"[tol-v2] 观测量 {n_done}/{len(beams)} 束", file=sys.stderr, flush=True)
    finally:
        for fh in files.values():
            fh.close()

    return {
        "dev_table": {
            tier: np.percentile(np.asarray(vals), quantile * 100, axis=0)
            for tier, vals in readings.items()
        },
        "m6_debias": m6_records,
        "m6_signed_train": m6_signed_train,
        "quantile": float(quantile),
        "scope": "valtest_pooled" if beams_scope == "valtest" else f"{beams_scope}_pooled",
        "tiers": [int(t) for t in tiers],
        "station": int(station),
        "k": int(k),
        "n_beams": len(beams),
        "n_train_beams": len(train_beams),
        "seed": int(seed),
    }
