"""L2 [S2]/[S3] 裸误差–N 曲线协议与等效放大因子 N_eff（spec line2/20）。

- 曲线消费 `10` [S5] 数据集（五档 × k=20 子样本，目标 = 2e5 全量）；
- 门禁曲线跨 **val/test 束** 中位数/IQR（train 束曲线仅诊断，不进门禁）；
- 互差地板 = 同档两两独立子样本 M1 差的中位数；参照噪声地板 = 1e5 档
  两两配对 M1 差中位数 / √2（dist(H_1e5, ρ) 的估计，`20` [S2]）；
- N_eff：log-N 线性插值，报告域 [1e3, 1e5]，上/下溢显式标记（[S3] C2）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .metrics import MetricUnderflow, rel_rmse

__all__ = ["CurvePoint", "CurveResult", "error_n_curve", "reference_noise_floor", "n_eff"]

_CURVE_TIERS = (1_000, 3_000, 10_000, 30_000, 100_000)
_DOMAIN = (1_000.0, 100_000.0)


@dataclass
class CurvePoint:
    n_low: int
    median: float
    iqr_lo: float
    iqr_hi: float
    n_samples: int


@dataclass
class CurveResult:
    points: list[CurvePoint]
    mutual_floor: dict[int, float]          # 同档两两 M1 互差中位数
    reference_noise_floor: float            # 1e5 配对 M1 中位数 / √2
    raw_samples: dict[int, list[float]]     # 复核用原始值

    def median_at(self, n_low: int) -> float:
        for p in self.points:
            if p.n_low == n_low:
                return p.median
        raise KeyError(n_low)


def _dataset_layout(data_dir: str | Path) -> tuple[dict, Path]:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest, data_dir


def _beam_split(manifest: dict, beam_id: int) -> str:
    for split, ids in manifest["splits"].items():
        if beam_id in ids:
            return split
    raise KeyError(f"beam {beam_id} 不在任何切分中")


def _group_m1(img: np.ndarray, ref: np.ndarray) -> float:
    from .metrics import group_rel_rmse

    return group_rel_rmse(img, ref)


def error_n_curve(
    data_dir: str | Path,
    beam_ids: list[int] | None = None,
    stations: tuple[int, ...] = (0, 1),
    tiers: tuple[int, ...] = _CURVE_TIERS,
    k: int = 20,
) -> CurveResult:
    """[S2] 裸误差–N 曲线（默认门禁口径 = val+test 束；train 束诊断曲线另行传 beam_ids）。"""
    manifest, data_dir = _dataset_layout(data_dir)
    if beam_ids is None:
        beam_ids = [b for s in ("val", "test") for b in manifest["splits"][s]]
    files: dict[str, h5py.File] = {}
    samples: dict[int, list[float]] = {t: [] for t in tiers}
    pair_m1: dict[int, list[float]] = {t: [] for t in tiers}
    try:
        for beam_id in beam_ids:
            split = _beam_split(manifest, beam_id)
            h5 = files.setdefault(
                split, h5py.File(data_dir / f"{split}.h5", "r")
            )
            grp = h5[f"beam_{beam_id:04d}"]
            for station_idx in stations:
                stg = grp[f"station{station_idx}"]
                ref = stg["target_full"][...]
                for tier in tiers:
                    imgs = [
                        stg[f"tier_{tier:06d}_sub_{sub_idx:02d}"][...]
                        for sub_idx in range(k)
                    ]
                    samples[tier].extend(_group_m1(img, ref) for img in imgs)
                    for a in range(0, k - 1, 2):  # 预注册配对：(0,1),(2,3),…
                        pair_m1[tier].append(_group_m1(imgs[a], imgs[a + 1]))
    finally:
        for f in files.values():
            f.close()

    points = []
    for tier in tiers:
        vals = np.asarray(samples[tier], dtype=np.float64)
        med = float(np.median(vals))
        q_lo, q_hi = np.percentile(vals, [25, 75])
        points.append(CurvePoint(tier, med, float(q_lo), float(q_hi), len(vals)))
    noise_floor = float(np.median(pair_m1[100_000]) / np.sqrt(2.0)) if pair_m1[100_000] else float("nan")
    return CurveResult(points, {t: float(np.median(v)) for t, v in pair_m1.items()}, noise_floor, samples)


def reference_noise_floor(curve: CurveResult) -> float:
    """参照噪声地板（dist(H_1e5, ρ) 估计）——曲线对象的预注册字段。"""
    return curve.reference_noise_floor


def n_eff_from_err(err: float, curve: CurveResult) -> tuple[float, str | None]:
    """[S3] 由误差值在裸曲线上 log-N 线性插值反解 N_eff（确定性）。

    - err 高于 1e3 档中位数 → (1e3, "underflow")；
    - err 低于 1e5 档中位数 → (1e5, "overflow")（语义：达到可验证域顶，1e4 工况 10×）。
    """
    xs = np.log10(np.asarray([p.n_low for p in curve.points], dtype=np.float64))
    ys = np.asarray([p.median for p in curve.points], dtype=np.float64)
    lo, hi = float(_DOMAIN[0]), float(_DOMAIN[1])
    if err > ys[0]:
        return lo, "underflow"
    if err < ys[-1]:
        return hi, "overflow"
    for i in range(len(ys) - 1):
        if ys[i + 1] <= err <= ys[i]:
            t = (ys[i] - err) / (ys[i] - ys[i + 1])
            return float(10.0 ** (xs[i] + t * (xs[i + 1] - xs[i]))), None
    return hi, "overflow"


def n_eff(
    output_group: np.ndarray,
    reference_group: np.ndarray,
    curve: CurveResult,
) -> tuple[float, str | None]:
    """图组级入口：err(O) = 六投影平均 M1（总计数归一内置），再插值 N_eff。

    O 总计数为 0 → (1e3, "underflow")（`20` [S3] C4 的消费端）。
    """
    errs = []
    try:
        for gi in range(output_group.shape[0]):
            errs.append(rel_rmse(output_group[gi], reference_group[gi]))
    except MetricUnderflow:
        return float(_DOMAIN[0]), "underflow"
    return n_eff_from_err(float(np.mean(errs)), curve)
