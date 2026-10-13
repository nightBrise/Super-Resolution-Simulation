"""L2 [S5] 基线族（冻结成员见 `20` [S5]）：矩匹配 B-moment + 零信息常数解 B-const。

- B-moment：从输入图组估计 Σ（reconstruct_sigma）+ 投影均值，渲染 2D 高斯期望计数图
  （总计数 = 参照总计数，与 M1 比较语义一致）。确定性（同输入逐位一致）。
- B-const（两站，`50` [S1]）：站 s 的 train 束该站全量目标图在 v2 公共域下逐投影归一的
  均值图；不含任何 test/束个体信息，作为 G1/G4/G5 的零信息参照。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np

from .metrics import group_rel_rmse, reconstruct_sigma
from .n_eff import _dataset_layout

__all__ = [
    "BASELINE_FAMILY",
    "baseline_family",
    "moment_matching_baseline",
    "bconst_group",
    "bconst_beam_errors",
    "bconst_records",
]

#: 冻结基线族（`20` [S5] 四成员 → 实现点）；B-const 由 `50` [S3] C4 入族
BASELINE_FAMILY: dict[str, str] = {
    "B-raw": "src.line2.metrics.n_eff.error_n_curve",
    "B-const": "src.line2.metrics.baseline.bconst_group",
    "B-curve": "src.line2.metrics.n_eff.error_n_curve",
    "B-moment": "src.line2.metrics.baseline.moment_matching_baseline",
}


def baseline_family() -> dict[str, str]:
    """基线族成员 → 实现点（只读返回副本，防调用方就地改写冻结族）。"""
    return dict(BASELINE_FAMILY)


def _per_projection_normalized(group: np.ndarray) -> np.ndarray:
    """逐投影归一（每投影除以自身总计数）——`50` [S1] B-const 定义的归一。"""
    totals = group.sum(axis=(1, 2), keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("图组含总计数为 0 的投影，无法逐投影归一")
    return group / totals


def bconst_group(
    data_dir: str | Path,
    station: int,
    *,
    split: str = "train",
    beam_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """B-const 均值图（`50` [S1]）：split 束该站全量目标图逐投影归一后的均值。

    Args:
        data_dir: v2 数据集目录（manifest + 各划分 h5）。
        station: 站号（0 或 1）。
        split: 取均值所用的划分（冻结口径 = train）。
        beam_ids: 显式束集（缺省 = manifest 的该划分）。

    Returns:
        (6, 128, 128) float64，逐投影总计数各为 1。
    """
    manifest, data_dir = _dataset_layout(data_dir)
    if beam_ids is None:
        beam_ids = [int(b) for b in manifest["splits"][split]]
    acc: np.ndarray | None = None
    n_beams = 0
    with h5py.File(data_dir / f"{split}.h5", "r") as h5:
        for beam_id in beam_ids:
            grp = h5[f"beam_{int(beam_id):04d}"][f"station{int(station)}"]["target_full"][...]
            normalized = _per_projection_normalized(grp.astype(np.float64))
            acc = normalized if acc is None else acc + normalized
            n_beams += 1
    if acc is None or n_beams == 0:
        raise ValueError(f"B-const：{split} 划分无束可平均")
    return acc / n_beams


def bconst_beam_errors(
    group: np.ndarray,
    data_dir: str | Path,
    station: int,
    *,
    split: str = "test",
    beam_ids: Sequence[int] | None = None,
) -> dict[int, float]:
    """B-const 的逐束 M1 误差向量（`20` [S1] M1：六投影各算后平均）。"""
    manifest, data_dir = _dataset_layout(data_dir)
    if beam_ids is None:
        beam_ids = [int(b) for b in manifest["splits"][split]]
    errors: dict[int, float] = {}
    with h5py.File(data_dir / f"{split}.h5", "r") as h5:
        for beam_id in beam_ids:
            ref = h5[f"beam_{int(beam_id):04d}"][f"station{int(station)}"]["target_full"][...]
            errors[int(beam_id)] = group_rel_rmse(group, ref)
    return errors


def bconst_records(
    data_dir: str | Path,
    *,
    stations: Iterable[int] = (0, 1),
    eval_split: str = "test",
    train_split: str = "train",
) -> dict:
    """B-const 两站的落盘载荷（组图 + 逐束误差向量 + 溯源字段）。

    ``groups`` 为 (6, 128, 128) 数组（落盘走 npz，不入 JSON）；``records`` 全部为
    JSON 原生标量（逐束误差键为字符串束号）。
    """
    manifest, data_dir = _dataset_layout(data_dir)
    train_beams = [int(b) for b in manifest["splits"][train_split]]
    eval_beams = [int(b) for b in manifest["splits"][eval_split]]
    groups: dict[int, np.ndarray] = {}
    records: dict[int, dict] = {}
    for station in stations:
        group = bconst_group(
            data_dir, int(station), split=train_split, beam_ids=train_beams
        )
        errors = bconst_beam_errors(
            group, data_dir, int(station), split=eval_split, beam_ids=eval_beams
        )
        values = np.asarray([errors[b] for b in eval_beams], dtype=np.float64)
        groups[int(station)] = group
        records[int(station)] = {
            "group_projection_sums": [float(v) for v in group.sum(axis=(1, 2))],
            "group_sha256": _array_sha256(group),
            "n_eval_beams": len(eval_beams),
            "eval_split": eval_split,
            "per_beam_m1": {str(b): float(errors[b]) for b in eval_beams},
            "err_median": float(np.median(values)),
            "err_min": float(values.min()),
            "err_max": float(values.max()),
        }
    return {
        "groups": groups,
        "records": {
            "data_version": manifest.get("data_version"),
            "spec_version": manifest.get("spec_version"),
            "data_dir": str(data_dir),
            "station": {str(s): r for s, r in records.items()},
            "n_train_beams": len(train_beams),
            "train_split": train_split,
            "reference": "`50` [S1] B-const（零信息常数解，两站）",
        },
    }


def _array_sha256(values: np.ndarray) -> str:
    """组图的小端 float64 连续字节 SHA256（溯源用，独立于落盘编码）。"""
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def moment_matching_baseline(
    input_group: np.ndarray, ref_group: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """输入六图 → Σ̂ → 渲染六张 2D 高斯期望计数图（和 = 参照逐投影总计数）。

    图组按存储布局 [aj, ai] 给出（见 metrics.py 模块 docstring）：输入图先转置回
    [ai, aj] 取投影均值、渲染后转回存储布局，否则渲染与数据集图组的轴序不一致。
    """
    sigma = reconstruct_sigma(input_group, edges)
    out = np.zeros_like(input_group, dtype=np.float64)
    pairs = (("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"))
    axis_of = {a: i for i, a in enumerate(("x", "xp", "y", "yp"))}
    for gi, (ai, aj) in enumerate(pairs):
        i, j = axis_of[ai], axis_of[aj]
        sub = np.array([[sigma[i, i], sigma[i, j]], [sigma[j, i], sigma[j, j]]])
        det = sub[0, 0] * sub[1, 1] - sub[0, 1] * sub[1, 0]
        if det <= 0:
            # Σ̂ 非正定（输入退化）：回退为对角化
            sub = np.diag(np.clip(np.diag(sub), 1e-12, None))
            det = sub[0, 0] * sub[1, 1]
        inv = np.array([[sub[1, 1], -sub[0, 1]], [-sub[1, 0], sub[0, 0]]]) / det
        ui = 0.5 * (edges[i][:-1] + edges[i][1:])
        uj = 0.5 * (edges[j][:-1] + edges[j][1:])
        # 投影均值：由输入图组的边缘分布估计（比 0 假设更贴近真实束心）
        img_in = input_group[gi].astype(np.float64).T
        tot_in = max(img_in.sum(), 1.0)
        mean_i = float((img_in.sum(axis=1) * ui).sum() / tot_in)
        mean_j = float((img_in.sum(axis=0) * uj).sum() / tot_in)
        du0 = ui - mean_i
        du1 = uj - mean_j
        quad = (
            inv[0, 0] * du0[:, None] ** 2
            + 2.0 * inv[0, 1] * np.outer(du0, du1)
            + inv[1, 1] * du1[None, :] ** 2
        )
        w = np.exp(-0.5 * quad)
        n_high = ref_group[gi].sum()  # 渲染总计数 = 参照（N_high 语义）
        rate = w / w.sum() * n_high
        out[gi] = rate.T
    return out
