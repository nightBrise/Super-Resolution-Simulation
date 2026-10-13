"""`50` [S3] C1/C2：判定输入白名单与选型读数 val_nll。

Phase 1.5 的判定分两类，各只有一条输入通道（`50` [S3] C1）：

- **选型与路由**只消费 **val 段** 的 val_nll（[S3] C2 定义）；
- **出口门禁 G1–G5** 只消费 **test 段** 的落盘读数（逐束 M1 err、13 维 obs、
  对齐仪器读数、B-const 逐束误差）。

两段不交叉（沿用 `40` D6/D12 两段式）；**N_eff 在两类判定中 SHALL 均不出现**——它保留
计算与落盘，但降级为叙事尺度（``DISCLOSURE_ONLY`` 的披露字段）。本模块是该白名单的
单一裁定点：判定侧代码 SHALL 经 :func:`assert_inputs` 校核其输入键，不得自行判读。

命名：选型读数在落盘字段里按 `50` [S3] C2 的名字叫 ``val_nll``（白名单键名同），但本
模块的**函数名一律避开** ``nll``——`30` [S2] C4 的"Poisson 项唯一实现点"测试按函数名
扫描 ``src/line2``（唯一实现点是 ``training.losses.poisson_nll``），聚合函数不得撞名。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from . import ANALYSIS_SEED

__all__ = [
    "KIND_SELECTION",
    "KIND_ROUTING",
    "KIND_GATE",
    "DECISION_KINDS",
    "DECISION_INPUTS",
    "DECISION_SEGMENTS",
    "DISCLOSURE_ONLY",
    "DecisionInputError",
    "inputs_for",
    "segment_for",
    "assert_inputs",
    "VAL_NLL_TIERS",
    "SELECTION_STATION",
    "SELECTION_SUBS",
    "tier_reading",
    "selection_reading",
    "selection_payload",
    "route_by_selection_reading",
]

KIND_SELECTION = "selection"
KIND_ROUTING = "routing"
KIND_GATE = "gate"
DECISION_KINDS = (KIND_SELECTION, KIND_ROUTING, KIND_GATE)

#: 逐类判定输入白名单（`50` [S3] C1 冻结）：键名即落盘读数的字段名
DECISION_INPUTS: dict[str, tuple[str, ...]] = {
    KIND_SELECTION: ("val_nll",),
    KIND_ROUTING: ("val_nll",),
    KIND_GATE: ("beam_err_m1", "obs13", "alignment", "bconst_beam_err"),
}
#: 各类判定消费的段（`50` [S3] C1：两段不交叉）
DECISION_SEGMENTS: dict[str, str] = {
    KIND_SELECTION: "val",
    KIND_ROUTING: "val",
    KIND_GATE: "test",
}
#: 只可披露、不得进判定的字段（N_eff 保留计算与落盘，降级为叙事尺度）
DISCLOSURE_ONLY = ("n_eff", "n_eff_flag")

#: N_eff 类字段名（含 best_val_n_eff / n_eff_median 等派生写法）
_N_EFF_PATTERN = re.compile(r"(^|_)n_eff(_|$)")


class DecisionInputError(ValueError):
    """判定输入越出白名单（含 N_eff 字段）时抛出。"""


def inputs_for(kind: str) -> tuple[str, ...]:
    """该类判定的输入白名单（冻结内容，按声明序返回）。"""
    if kind not in DECISION_INPUTS:
        raise DecisionInputError(f"未知判定类别：{kind!r}（∈ {DECISION_KINDS}）")
    return DECISION_INPUTS[kind]


def segment_for(kind: str) -> str:
    """该类判定消费的数据段（``val`` 或 ``test``）。"""
    if kind not in DECISION_SEGMENTS:
        raise DecisionInputError(f"未知判定类别：{kind!r}（∈ {DECISION_KINDS}）")
    return DECISION_SEGMENTS[kind]


def assert_inputs(kind: str, keys: Sequence[str]) -> tuple[str, ...]:
    """校核该类判定的输入键：白名单子集且不含 N_eff 字段。

    Args:
        kind: 判定类别（选型/路由/门禁）。
        keys: 该判定**实际消费**的读数键名。

    Returns:
        白名单元组（与 :func:`inputs_for` 同序）——调用方按它取数即可保证白名单化。

    Raises:
        DecisionInputError: 键集合含 N_eff 类字段，或出现白名单外的键。
    """
    allowed = inputs_for(kind)
    for key in keys:
        if _N_EFF_PATTERN.search(str(key)):
            raise DecisionInputError(
                f"{kind} 判定消费了 N_eff 字段 {key!r}：`50` [S3] C1 规定 N_eff 在"
                f"选型/路由/门禁中均不出现，仅作披露字段（∈ {DISCLOSURE_ONLY}）"
            )
        if key not in allowed:
            raise DecisionInputError(
                f"{kind} 判定输入 {key!r} 不在白名单 {allowed} 内（`50` [S3] C1）"
            )
    return allowed


# ------------------------------------------------------------------ [S3] C2
#: 选型读数的档集（`50` [S3] C2 冻结）：三档等权
VAL_NLL_TIERS = (3_000, 10_000, 100_000)
#: 选型读数的站与子样本范围（`50` [S3] C2：站 0、subs 0–4）
SELECTION_STATION = 0
SELECTION_SUBS = (0, 1, 2, 3, 4)


def tier_reading(per_tier: Mapping[int, np.ndarray], tier: int) -> float:
    """单档的 val_nll 分量：逐 (束, 子样本) → 逐束中位数 → 全束中位数。

    [S1] 的聚合惯性是"先逐束收子样本、再对束取统计量"（`50` [S1] 聚合口径第一条），
    故本函数对 (n_beams, n_subs) 矩阵先按束取中位数、再对束取中位数；池化读法
    （把该档全部 (束, 子样本) 读数一次取中位数）在束内读数非对称时与前者分叉。
    """
    if tier not in per_tier:
        raise ValueError(f"val_nll 缺档 {tier}（`50` [S3] C2 档集 {VAL_NLL_TIERS}）")
    arr = np.asarray(per_tier[tier], dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"档 {tier} 的 NLL 须为 (束, 子样本) 矩阵，收到 shape={arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"档 {tier} 的 NLL 矩阵为空：shape={arr.shape}")
    return float(np.median(np.median(arr, axis=1)))


def selection_reading(per_tier: Mapping[int, np.ndarray]) -> float:
    """选型读数 val_nll（`50` [S3] C2）：三档分量的等权算术平均。

    Args:
        per_tier: 档位 → (束, 子样本) NLL 矩阵；档集须覆盖 :data:`VAL_NLL_TIERS`。

    Returns:
        三档等权平均的 val_nll（越小越好；选型与路由的唯一准绳）。
    """
    return float(np.mean([tier_reading(per_tier, int(t)) for t in VAL_NLL_TIERS]))


def selection_payload(
    per_tier: Mapping[int, np.ndarray],
    *,
    station: int = SELECTION_STATION,
    subs: Sequence[int] = SELECTION_SUBS,
) -> dict:
    """选型读数的落盘载荷（`50` [S3] C2：SHALL 附各档分项）。

    Raises:
        ValueError: 站或子样本范围偏离 [S3] C2 的冻结定义（该读数只在站 0、subs 0–4
            定义；偏离即不是 val_nll，不得沿用该名字）。
    """
    if int(station) != SELECTION_STATION:
        raise ValueError(
            f"val_nll 只在站 {SELECTION_STATION} 定义（`50` [S3] C2），收到站 {station}"
        )
    if tuple(int(s) for s in subs) != SELECTION_SUBS:
        raise ValueError(
            f"val_nll 只在 subs {SELECTION_SUBS} 定义（`50` [S3] C2），收到 {tuple(subs)}"
        )
    return {
        "val_nll": selection_reading(per_tier),
        "per_tier": {
            str(int(t)): {
                "median_nll": tier_reading(per_tier, int(t)),
                "n_beams": int(np.asarray(per_tier[int(t)]).shape[0]),
                "n_subs": int(np.asarray(per_tier[int(t)]).shape[1]),
            }
            for t in VAL_NLL_TIERS
        },
        "tiers": [int(t) for t in VAL_NLL_TIERS],
        "station": int(station),
        "subs": [int(s) for s in subs],
        "segment": DECISION_SEGMENTS[KIND_SELECTION],
        "aggregation": "per_beam_median_then_beam_median_then_tier_equal_mean",
        "seed": int(ANALYSIS_SEED),
    }


def route_by_selection_reading(per_arm: Mapping[str, float]) -> str:
    """按选型读数（val_nll，越小越好）排序取获胜臂（`50` [S5] 的次级准绳）。

    同值并列时按臂名字典序取小者（确定性；避免输入顺序影响结论）。喂入哪个读数
    （站 0 @1×10⁴ 的 val 段 val_nll）由 `50` [S5] 的路由条款决定，本函数只做排序。
    """
    if not per_arm:
        raise ValueError("route_by_selection_reading：无臂可比")
    for arm, value in per_arm.items():
        if value is None or not np.isfinite(float(value)):
            raise ValueError(
                f"route_by_selection_reading：臂 {arm!r} 的 val_nll 非有限（{value!r}）"
            )
    return min(sorted(per_arm), key=lambda arm: float(per_arm[arm]))
