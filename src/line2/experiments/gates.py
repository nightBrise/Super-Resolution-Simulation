"""L4 E1/E3 门禁评估（spec line2/40 [S2]/[S3]）。

- H1（放大存在性）：获胜变体 @1e4 的 5 种子 N_eff 分布
  - 成立：5/5 种子 N_eff > 1e4 且中位数 ≥ 2×1e4
  - 不成立：中位数 ≤ 1e4
  - 不可判定：其它情况
- H3（联合优于独立）：同种子值成对比较（5 对）；两臂种子集合不同则判不可判定
  （``reason_code="seed_sets_differ"``），不得退化为槽位配对
  - 成立：5/5 对联合 N_eff ≥ 独立
  - 方向成立（弱）：4/5
  - 不成立：≤ 2/5
  - 不可判定：3/5
- H7（跨站泛化，[S2]）：ρ = N_eff(站1)/N_eff(站0) 对预注册边界 {1.0, 0.5} 的三分类
- H8（细结构可恢复性，[S3]）：13 项观测量对容差线的达标率 ≥80% 成立 / ≤50% 不成立 /
  之间不可判定；聚合 = test 束 × 站 0 × 1e4 × 5 种子**中位数输出**（`20` [S4]）
- `evaluate_gates_from_disk`：只读 run 目录产物出 `e1_gates.json`（不训练、不推理，[S3] C2）

两段式消费边界（[S3]、D6）：变体路由只吃 **val** 段（run 目录 summary.json 的
best_val_n_eff）；H1/H3/H8 出口门禁只吃 **test** 段（test_eval.json，split 断言强制）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.line2.data_pipeline.dataset import git_head
from src.line2.metrics.tolerance import N_OBSERVABLES, tolerance_for

from .evaluate import curve_from_json

__all__ = [
    "evaluate_h1",
    "evaluate_h3",
    "evaluate_e1_gates",
    "winner_variant",
    "pair_by_seed",
    "EVAL_FN",
    "VERDICT_PASS",
    "VERDICT_FAIL",
    "VERDICT_INDETERMINATE",
    "VERDICT_DIRECTIONAL",
    "OBSERVABLE_NAMES",
    "h8_deviation_matrix",
    "evaluate_h8",
    "evaluate_h7_cross_station",
    "evaluate_gates_from_disk",
]

# `40` [S3]：门禁输出恒为下列合法枚举（"方向成立"仅 H3 使用），不得自由文本
VERDICT_PASS = "成立"
VERDICT_FAIL = "不成立"
VERDICT_INDETERMINATE = "不可判定"
VERDICT_DIRECTIONAL = "方向成立"
VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_INDETERMINATE, VERDICT_DIRECTIONAL)

H8_PASS_RATE = 0.80   # `40` [S3] H8 成立线（冻结）
H8_FAIL_RATE = 0.50   # `40` [S3] H8 不成立线（冻结）
H7_RHO_STRONG = 1.0   # `40` [S2] ρ 边界（冻结）
H7_RHO_WEAK = 0.5
GATE_STATION = 0      # `40` [S3] H8 聚合维度：站 0
GATE_N_LOW = 10_000   # `40` [S3] H8 聚合维度：N_low = 1e4
E1_SEEDS = 5          # `40` [S1] E1 每变体 5 种子

EVAL_GLOB = "test_eval*.json"
SUMMARY_FILENAME = "summary.json"
CURVE_FILENAME = "e0_curve.json"
TOLERANCE_FILENAME = "tolerance.json"
GATES_FILENAME = "e1_gates.json"

_AXES = ("x", "xp", "y", "yp")
_RHO_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
# 13 项观测量名（顺序 = `20` [S4] observable_vector：M4 的 11 维 + M5 + M6）
OBSERVABLE_NAMES: tuple[str, ...] = (
    *(f"sigma_{a}{a}_rel" for a in _AXES),
    *(f"abs_delta_rho_{_AXES[i]}_{_AXES[j]}" for i, j in _RHO_PAIRS),
    "eps4d_rel",
    "m5_halo_fraction_gap",
    "m6_hf_energy_ratio_gap",
)


def winner_variant(
    joint: Sequence[float], independent: Sequence[float]
) -> str:
    """根据 5 对同种子比较推算 E2 路由的"获胜变体"。

    - 5/5 对联合 ≥ 独立 → joint
    - 4/5 → joint（方向成立）
    - 3/5 → 中位数较高者
    - ≤ 2/5 → independent
    """
    if len(joint) != len(independent):
        raise ValueError("joint 与 independent 种子数不匹配")
    pairs = [(j, i) for j, i in zip(joint, independent) if j is not None and i is not None]
    if not pairs:
        return "indeterminate"
    n = len(pairs)
    joint_wins = sum(1 for j, i in pairs if j >= i)
    if joint_wins >= 5:
        return "joint"
    if joint_wins >= 4:
        return "joint"  # 方向成立
    if joint_wins >= 3:
        return "joint" if np.median([j for j, _ in pairs]) >= np.median([i for _, i in pairs]) else "independent"
    return "independent"


def evaluate_h1(n_eff_values: Sequence[float]) -> str:
    """H1（spec 40 [S3]：放大存在性，获胜变体 @1e4 的 5 种子 N_eff 分布）。

    成立：5/5 N_eff > 1e4 且中位数 ≥ 2e4
    不成立：中位数 ≤ 1e4
    不可判定：其它
    """
    if len(n_eff_values) != 5:
        raise ValueError("H1 需要恰好 5 种子 N_eff")
    n_pos = sum(1 for n in n_eff_values if n > 1e4)
    median = float(np.median(n_eff_values))
    if n_pos == 5 and median >= 2e4:
        return "成立"
    if median <= 1e4:
        return "不成立"
    return "不可判定"


def evaluate_h3(
    joint: Sequence[float], independent: Sequence[float]
) -> str:
    """H3（spec 40 [S3]：联合优于独立，同种子成对 5 对）。

    成立：5/5 对联合 ≥ 独立
    方向成立（弱）：4/5
    不成立：≤ 2/5
    不可判定：3/5
    """
    if len(joint) != 5 or len(independent) != 5:
        raise ValueError("H3 需要各 5 种子 N_eff")
    pairs = [(j, i) for j, i in zip(joint, independent)]
    joint_wins = sum(1 for j, i in pairs if j >= i)
    if joint_wins == 5:
        return "成立"
    if joint_wins == 4:
        return "方向成立"
    if joint_wins <= 2:
        return "不成立"
    return "不可判定"


def evaluate_e1_gates(
    joint: Sequence[float], independent: Sequence[float]
) -> dict:
    """E1 完整门禁：H1（联合） + H3（联合 vs 独立） + 路由推算。"""
    h1 = evaluate_h1(joint)
    h3 = evaluate_h3(joint, independent)
    variant = winner_variant(joint, independent)
    return {"H1": h1, "H3": h3, "winner_variant": variant}


EVAL_FN = evaluate_e1_gates


# --------------------------------------------------------------------------- H8
def h8_deviation_matrix(
    per_run_obs: Sequence[dict], beam_ids: Sequence[int]
) -> np.ndarray:
    """多 run（= 多种子）的 ``per_beam[].obs`` → (len(beam_ids), 13) 矩阵。

    聚合次序预注册于 `20` [S4]：**逐观测量先对种子取中位数**，达标判定在其后
    （`evaluate_h8`）——不是"逐种子判达标再计票"，二者对偶数种子或跨束混合
    失效模式会给出不同达标率。行序 = ``beam_ids`` 给定顺序。
    """
    if not per_run_obs:
        raise ValueError("h8_deviation_matrix：无 run 可聚合")
    ids = [int(b) for b in beam_ids]
    if not ids:
        raise ValueError("h8_deviation_matrix：beam_ids 为空")
    per_run: list[np.ndarray] = []
    for run in per_run_obs:
        run_id = run.get("run_id", "?")
        obs_by_beam = {
            int(b["beam_id"]): np.asarray(b["obs"], dtype=np.float64)
            for b in run["per_beam"]
        }
        rows = []
        for beam_id in ids:
            if beam_id not in obs_by_beam:
                raise ValueError(f"run {run_id}：per_beam 缺 beam {beam_id} 的观测量")
            obs = obs_by_beam[beam_id]
            if obs.shape != (N_OBSERVABLES,):
                raise ValueError(
                    f"run {run_id} beam {beam_id}：观测量维度 {obs.shape} ≠ ({N_OBSERVABLES},)"
                )
            rows.append(obs)
        per_run.append(np.stack(rows))
    return np.median(np.stack(per_run), axis=0)


def evaluate_h8(dev_matrix: np.ndarray, tol_matrix: np.ndarray) -> dict:
    """H8（`40` [S3]）：13 项观测量对容差线的达标率三分类。

    达标率 = 达标的 (束, 观测量) 对数 / (n_beams × 13)；达标 = 偏差 ≤ 容差线
    （容差 = 裸子样本 |Δ| 分布的 95% 分位，故取闭区间）。≥80% 成立、≤50% 不成立、
    之间不可判定。``tol_matrix`` 可为 (n_beams, 13) 或广播用的 (13,)。
    """
    dev = np.asarray(dev_matrix, dtype=np.float64)
    tol = np.asarray(tol_matrix, dtype=np.float64)
    if dev.ndim != 2 or dev.shape[1] != N_OBSERVABLES:
        raise ValueError(f"dev_matrix 形状 {dev.shape} ≠ (n_beams, {N_OBSERVABLES})")
    if dev.size == 0:
        raise ValueError("dev_matrix 为空：无 (束, 观测量) 对可判")
    if tol.shape == (N_OBSERVABLES,):
        tol = np.broadcast_to(tol, dev.shape)
    if tol.shape != dev.shape:
        raise ValueError(f"tol_matrix 形状 {tol.shape} 与 dev_matrix {dev.shape} 不可比")
    # 非有限偏差按不达标计（observable_vector 已拒绝非有限值，此处为守卫）
    ok = dev <= tol
    n_beams = dev.shape[0]
    pass_rate = float(ok.sum()) / float(ok.size)
    if pass_rate >= H8_PASS_RATE:
        verdict = VERDICT_PASS
    elif pass_rate <= H8_FAIL_RATE:
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_INDETERMINATE
    per_observable = [
        {
            "index": i,
            "name": OBSERVABLE_NAMES[i],
            "n_pass": int(ok[:, i].sum()),
            "n_beams": int(n_beams),
            "pass_rate": float(ok[:, i].sum()) / float(n_beams),
        }
        for i in range(N_OBSERVABLES)
    ]
    return {"verdict": verdict, "pass_rate": pass_rate, "per_observable": per_observable}


# --------------------------------------------------------------------------- H7
def _coerce_n_eff(value: object) -> tuple[float | None, str | None]:
    """接受 N_eff 的三种落盘形态：裸数值 / (值, 标记) / test_eval.json 的束级字典。"""
    flag: str | None = None
    if value is None:
        return None, None
    if isinstance(value, dict):
        flag = value.get("n_eff_flag")
        value = value.get("n_eff")
    elif isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"N_eff 二元组长度异常：{value!r}")
        value, flag = value
    if value is None:
        return None, flag
    return float(value), flag


def evaluate_h7_cross_station(n_eff_s0: object, n_eff_s1: object) -> dict:
    """H7（`40` [S2]）：ρ = N_eff(站1)/N_eff(站0) 对预注册边界 {1.0, 0.5} 的三分类。

    ρ ≥ 1 → 成立；0.5 ≤ ρ < 1 → 成立（弱，``weak=True`` + ``decay`` 报衰减系数）；
    ρ < 0.5 → 不成立；ρ 无法计算 → 不可判定。

    "无法计算"的界定（依据 `40` [S2] C2：结论 SHALL 由 ρ 与 {1.0, 0.5} **唯一**决定，
    不得人工裁量）：上/下溢标记只说明 N_eff 触及报告域端点（1e3 / 1e5），端点本身
    是有限正数，比值仍可算——两侧同触端点时 ρ = 1（例：双上溢 = 双域顶，跨站无衰减）。
    因此标记不增设不可判定分支，只随结论输出供报告披露饱和；真正不可判定的是
    任一侧缺失、非有限或非正（除零/无定义）的情形。
    """
    v0, flag0 = _coerce_n_eff(n_eff_s0)
    v1, flag1 = _coerce_n_eff(n_eff_s1)
    out: dict = {
        "verdict": VERDICT_INDETERMINATE,
        "rho": None,
        "weak": False,
        "decay": None,
        "n_eff_station0": v0,
        "n_eff_station1": v1,
        "flag_station0": flag0,
        "flag_station1": flag1,
        "thresholds": {"strong": H7_RHO_STRONG, "weak": H7_RHO_WEAK},
    }
    if v0 is None or v1 is None or not (math.isfinite(v0) and math.isfinite(v1)) or v0 <= 0.0:
        return out
    rho = v1 / v0
    out["rho"] = float(rho)
    if rho >= H7_RHO_STRONG:
        out["verdict"] = VERDICT_PASS
    elif rho >= H7_RHO_WEAK:
        out["verdict"] = VERDICT_PASS
        out["weak"] = True
        out["decay"] = float(rho)  # 衰减系数 = ρ（[S2]"成立（弱，报告衰减系数）"）
    else:
        out["verdict"] = VERDICT_FAIL
    return out


# --------------------------------------------------- 落盘产物驱动的出口门禁
def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺产物：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_test_evals(results_dir: Path) -> list[dict]:
    """收集 run 目录的 test_eval*.json，并强制 val/test 消费边界（`40` [S3] C2、`00` GC3）。

    glob 覆盖 ``test_eval.json`` 与 ``--out`` 自定义名（E3 的同一 ckpt 跨站评估需另起名，
    否则与站 0 产物同名相覆）；凡被门禁消费的 eval 文件其 ``split`` 必须为 test。
    """
    paths = sorted(results_dir.glob(f"*/{EVAL_GLOB}"))
    if not paths:
        raise FileNotFoundError(f"{results_dir}：未找到任何 */{EVAL_GLOB}，门禁无输入")
    evals = []
    for path in paths:
        payload = _read_json(path)
        split = payload.get("split")
        if split != "test":
            raise ValueError(
                f"{path}：split={split!r}——出口门禁只消费 test 段"
                "（`40` [S3] 两段式判据消费、`00` GC3 test 只评一次）"
            )
        payload["eval_path"] = str(path)
        evals.append(payload)
    return evals


def _load_val_n_eff(results_dir: Path, run_ids: Sequence[str]) -> dict[str, dict[int, float]]:
    """val 段 N_eff（**仅供变体路由**）：run 目录 summary.json 的 best_val_n_eff。"""
    out: dict[str, dict[int, float]] = {}
    for run_id in run_ids:
        path = results_dir / run_id / SUMMARY_FILENAME
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        value = summary.get("best_val_n_eff")
        if value is None or not math.isfinite(float(value)):
            continue
        variant = str(summary.get("variant", "?"))
        out.setdefault(variant, {})[int(summary["seed"])] = float(value)
    return out


def _load_tolerance(path: Path) -> dict[int, np.ndarray] | None:
    """读 `20` [S4] 容差线产物；缺失返回 None（H8 降级不可判定，不现场猜容差）。

    格式 = `tolerance.deviation_table` 的 dump（键为字符串档位）：
    ``{"quantile": 0.95, "scope": "valtest_pooled", "dev_table": {"10000": [13 个分位值]}}``
    """
    path = Path(path)
    if not path.is_file():
        print(f"[gates] 容差线产物缺失：{path}——H8 判不可判定", file=sys.stderr)
        return None
    dump = _read_json(path)
    table = dump.get("dev_table", dump)
    return {int(tier): np.asarray(vec, dtype=np.float64) for tier, vec in table.items()}


def _beam_records(run: dict) -> dict[int, dict]:
    return {int(b["beam_id"]): b for b in run["per_beam"]}


def pair_by_seed(
    joint: Mapping[int, float], independent: Mapping[int, float]
) -> list[tuple[int, int]]:
    """H3 成对比较的配对表 = 两臂**同种子值**字面配对（[S3]"同种子成对比较（5 对）"）。

    两臂种子集合不同时抛 ``ValueError``（不返回空表、不退化为升序槽位配对）：`30` [S4] C2
    要求两臂共享同一种子派生链，集合不同意味着两份 run 不可配对，槽位配对会把不同种子
    当成一对。调用方须显式降级为不可判定（见 :func:`evaluate_gates_from_disk`），
    配对口径不得静默改变。
    """
    joint_seeds, indep_seeds = set(joint), set(independent)
    if joint_seeds != indep_seeds:
        raise ValueError(
            f"H3 同种子配对不成立：joint 种子 {sorted(joint_seeds)} "
            f"≠ independent 种子 {sorted(indep_seeds)}"
        )
    return [(seed, seed) for seed in sorted(joint_seeds)]


def _pairing_record(
    pairs: Sequence[tuple[int, int]], seed_sets: Mapping[str, Sequence[int]] | None = None
) -> dict:
    """配对表落盘供审计：``basis="seed_value"`` + 成对的种子值（含每对的 seed）。"""
    record = {"basis": "seed_value", "pairs": [[int(a), int(b)] for a, b in pairs]}
    if seed_sets is not None:
        record["seed_sets"] = {arm: [int(s) for s in seeds] for arm, seeds in seed_sets.items()}
    return record


def _pairs_by_seed_or_reason(
    joint: Mapping[int, float], independent: Mapping[int, float]
) -> tuple[list[tuple[int, int]], str | None, dict[str, list[int]]]:
    """(pairs, reason_code, 两臂种子集)——种子集合不同时 pairs 为空并给出显式原因。"""
    seed_sets = {"joint": sorted(joint), "independent": sorted(independent)}
    try:
        return pair_by_seed(joint, independent), None, seed_sets
    except ValueError as exc:
        print(f"[gates] {exc}", file=sys.stderr)
        return [], "seed_sets_differ", seed_sets


def _h8_from_disk(
    runs_by_seed: dict[int, dict],
    curve_nodes: tuple[int, ...],
    tolerance_path: Path,
) -> dict:
    """获胜变体的 test 束 × 站 0 × 1e4 观测量 → 种子中位数 → 达标率（`40` [S3] H8）。"""
    seeds = sorted(runs_by_seed)
    runs = [runs_by_seed[s] for s in seeds]
    base = {"seeds": [int(s) for s in seeds], "n_seeds": len(seeds), "tolerance": None}
    if not runs:
        return {
            **base, "verdict": VERDICT_INDETERMINATE, "pass_rate": None,
            "per_observable": [], "n_beams": 0, "beam_ids": [], "reason_code": "no_runs",
        }
    records = [_beam_records(r) for r in runs]
    beam_ids = sorted(set.intersection(*[{int(b) for b in rec} for rec in records]))
    dev = h8_deviation_matrix(runs, beam_ids)
    dev_table = _load_tolerance(tolerance_path)
    if dev_table is None:
        return {
            **base, "verdict": VERDICT_INDETERMINATE, "pass_rate": None,
            "per_observable": [], "n_beams": len(beam_ids), "beam_ids": beam_ids,
            "reason_code": "tolerance_missing",
        }
    missing = [int(n) for n in curve_nodes if int(n) not in dev_table]
    if missing:
        raise ValueError(f"{tolerance_path}：容差表缺曲线档 {missing}，`20` [S4] 最近档映射无法执行")
    # 容差节点 = 与该束 N_eff 对数距离最近的曲线档（`20` [S4]）；束级 N_eff 取种子中位数
    tol_rows = [
        tolerance_for(
            dev_table,
            curve_nodes,
            float(np.median([rec[beam_id]["n_eff"] for rec in records])),
        )
        for beam_id in beam_ids
    ]
    return {
        **base,
        **evaluate_h8(dev, np.stack(tol_rows)),
        "n_beams": len(beam_ids),
        "beam_ids": beam_ids,
        "tolerance": str(tolerance_path),
    }


def evaluate_gates_from_disk(
    results_dir: str | Path, tolerance: str | Path | None = None
) -> dict:
    """只读 run 目录产物 → `<results_dir>/e1_gates.json`（`40` [S3] C1/C2：不训练、不推理）。

    消费面：
    - `<run_id>/test_eval*.json`——**test 段**，H1/H3/H7/H8 的唯一判据来源（split 断言强制）；
    - `<run_id>/summary.json` 的 ``best_val_n_eff``——**val 段**，仅供获胜变体路由（[S3] 两段式）；
    - `e0_curve.json`——曲线溯源与容差节点档位；
    - `tolerance.json`（可选）——`20` [S4] 容差线，缺失则 H8 判不可判定。

    聚合维度按 [S3] 预注册：站 0、N_low = 1e4、每变体 5 种子；不满足则相应门禁判
    不可判定并记录 ``n_seeds``/``n_pairs``。H3 与变体路由的配对口径 = **同种子值字面
    配对**（``pairing.basis="seed_value"``）；两臂种子集合不同时判不可判定并显式记
    ``reason_code="seed_sets_differ"`` / ``routing_basis="val_seed_mismatch"``。
    每条结论附判定表达式的输入数值快照（`40` [S4] C2）。同输入两次调用输出逐位一致。
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在：{results_dir}")
    curve_path = results_dir / CURVE_FILENAME
    if not curve_path.is_file():
        raise FileNotFoundError(f"缺产物：{curve_path}（E0 曲线；`40` [S1] C5 尺子先行）")
    curve = curve_from_json(curve_path)
    curve_nodes = tuple(int(p.n_low) for p in curve.points)

    evals = _load_test_evals(results_dir)
    gate_evals: list[dict] = []
    cross_evals: list[dict] = []
    for payload in evals:
        n_low = int(payload.get("n_low", GATE_N_LOW))
        station = int(payload.get("station", GATE_STATION))
        if n_low != GATE_N_LOW:
            continue  # E2 的 3e3/1e5 档不进 E1 门禁（[S3] 聚合维度预注册）
        if station == GATE_STATION:
            gate_evals.append(payload)
        elif station == 1:
            cross_evals.append(payload)

    by_variant: dict[str, dict[int, dict]] = {}
    for payload in gate_evals:
        variant = str(payload.get("variant"))
        seed = int(payload["seed"])
        bucket = by_variant.setdefault(variant, {})
        if seed in bucket:
            raise ValueError(
                f"变体 {variant} 种子 {seed} 存在多份站 0 @1e4 test 评估"
                f"（{bucket[seed]['run_id']} / {payload['run_id']}）——test 只评一次（`00` GC3）"
            )
        bucket[seed] = payload

    # val 段：路由（5 对成对计数，[S3] 获胜变体路由全分支由 winner_variant 承担）
    val_n_eff = _load_val_n_eff(results_dir, sorted({str(p["run_id"]) for p in evals}))
    val_joint, val_indep = val_n_eff.get("joint", {}), val_n_eff.get("independent", {})
    val_pairs, val_reason, val_seed_sets = _pairs_by_seed_or_reason(val_joint, val_indep)
    routing_basis = "val"
    if val_reason is not None:
        winner = "indeterminate"
        routing_basis = "val_seed_mismatch"
    elif len(val_pairs) == E1_SEEDS:
        winner = winner_variant(
            [val_joint[a] for a, _ in val_pairs], [val_indep[b] for _, b in val_pairs]
        )
    else:
        winner = "indeterminate"
        routing_basis = "val_insufficient"

    # test 段：H3（同种子值成对 5 对）
    joint_test = {s: by_variant["joint"][s]["n_eff"] for s in sorted(by_variant.get("joint", {}))}
    indep_test = {
        s: by_variant["independent"][s]["n_eff"] for s in sorted(by_variant.get("independent", {}))
    }
    test_pairs, test_reason, test_seed_sets = _pairs_by_seed_or_reason(joint_test, indep_test)
    if len(test_pairs) == E1_SEEDS:
        h3: dict = {
            "verdict": evaluate_h3(
                [joint_test[a] for a, _ in test_pairs], [indep_test[b] for _, b in test_pairs]
            ),
            "n_pairs": len(test_pairs),
            "joint_wins": int(
                sum(1 for a, b in test_pairs if joint_test[a] >= indep_test[b])
            ),
            "pairing": _pairing_record(test_pairs),
            "inputs": {
                "joint": {str(a): joint_test[a] for a, _ in test_pairs},
                "independent": {str(b): indep_test[b] for _, b in test_pairs},
            },
        }
    else:
        h3 = {
            "verdict": VERDICT_INDETERMINATE, "n_pairs": len(test_pairs), "joint_wins": None,
            "pairing": _pairing_record(test_pairs, test_seed_sets if test_reason else None),
            "inputs": {"joint": {str(k): v for k, v in joint_test.items()},
                       "independent": {str(k): v for k, v in indep_test.items()}},
            "reason_code": test_reason or "pairs_insufficient",
        }

    # test 段：H1（获胜变体 5 种子分布）
    winner_runs = by_variant.get(winner, {})
    winner_seeds = sorted(winner_runs)
    h1_inputs = {str(s): winner_runs[s]["n_eff"] for s in winner_seeds}
    if len(winner_seeds) == E1_SEEDS:
        values = [winner_runs[s]["n_eff"] for s in winner_seeds]
        h1: dict = {
            "verdict": evaluate_h1(values),
            "variant": winner,
            "n_seeds": len(values),
            "median_n_eff": float(np.median(np.asarray(values, dtype=np.float64))),
            "n_above_n_low": int(sum(1 for v in values if v > GATE_N_LOW)),
            "thresholds": {"n_low": float(GATE_N_LOW), "median_min": 2.0 * GATE_N_LOW},
            "inputs": h1_inputs,
        }
    else:
        h1 = {
            "verdict": VERDICT_INDETERMINATE, "variant": winner, "n_seeds": len(winner_seeds),
            "median_n_eff": None, "n_above_n_low": None,
            "thresholds": {"n_low": float(GATE_N_LOW), "median_min": 2.0 * GATE_N_LOW},
            "inputs": h1_inputs, "reason_code": "winner_or_seeds_insufficient",
        }

    h8 = _h8_from_disk(
        winner_runs, curve_nodes, Path(tolerance) if tolerance else results_dir / TOLERANCE_FILENAME
    )
    h8["variant"] = winner

    # test 段：H7（E3 跨站，站 1 的 test 束评估）
    cross_by_seed = {
        int(p["seed"]): p for p in cross_evals if str(p.get("variant")) == winner
    }
    h7_seeds = sorted(set(cross_by_seed) & set(winner_runs))
    if h7_seeds:
        per_seed = {
            str(s): evaluate_h7_cross_station(winner_runs[s], cross_by_seed[s]) for s in h7_seeds
        }
        s0 = [float(winner_runs[s]["n_eff"]) for s in h7_seeds]
        s1 = [float(cross_by_seed[s]["n_eff"]) for s in h7_seeds]
        overall = evaluate_h7_cross_station(
            float(np.median(np.asarray(s0, dtype=np.float64))),
            float(np.median(np.asarray(s1, dtype=np.float64))),
        )
        h7: dict = {
            "verdict": overall["verdict"], "rho": overall["rho"], "weak": overall["weak"],
            "decay": overall["decay"], "variant": winner, "n_seeds": len(h7_seeds),
            "aggregation": "seed_median_then_ratio",
            "n_eff_station0_median": float(np.median(np.asarray(s0, dtype=np.float64))),
            "n_eff_station1_median": float(np.median(np.asarray(s1, dtype=np.float64))),
            "thresholds": {"strong": H7_RHO_STRONG, "weak": H7_RHO_WEAK},
            "per_seed": per_seed,
        }
    else:
        h7 = {
            "verdict": VERDICT_INDETERMINATE, "rho": None, "weak": False, "decay": None,
            "variant": winner, "n_seeds": 0, "aggregation": "seed_median_then_ratio",
            "thresholds": {"strong": H7_RHO_STRONG, "weak": H7_RHO_WEAK}, "per_seed": {},
            "reason_code": "station1_missing",
        }

    snapshot = {}
    for variant, runs in sorted(by_variant.items()):
        records = [_beam_records(runs[s]) for s in sorted(runs)]
        rows = []
        for beam_id in sorted(set.intersection(*[{int(b) for b in rec} for rec in records])):
            values = [rec[beam_id]["n_eff"] for rec in records]
            rows.append(
                {
                    "beam_id": int(beam_id),
                    "n_eff_median": float(np.median(np.asarray(values, dtype=np.float64))),
                    "n_eff_per_seed": {
                        str(s): records[i][beam_id]["n_eff"] for i, s in enumerate(sorted(runs))
                    },
                    "flag_per_seed": {
                        str(s): records[i][beam_id]["n_eff_flag"] for i, s in enumerate(sorted(runs))
                    },
                }
            )
        snapshot[variant] = rows

    payload = {
        "results_dir": str(results_dir),
        "runs": sorted({str(p["run_id"]) for p in evals}),
        "code_version": git_head(),
        "gate_dims": {
            "split": "test", "station": GATE_STATION, "n_low": GATE_N_LOW,
            "n_seeds_expected": E1_SEEDS,
        },
        "curve": {
            "path": str(curve_path),
            "beams_scope": getattr(curve, "beams_scope", None),
            "reference_noise_floor": float(curve.reference_noise_floor),
            "nodes": [int(n) for n in curve_nodes],
        },
        "val_segment": {
            "source": f"{SUMMARY_FILENAME}:best_val_n_eff",
            "consumed_by": "variant_routing",
            "n_pairs": len(val_pairs),
            "pairing_basis": "seed_value",
            "seed_sets": val_seed_sets,
            "per_seed_n_eff": {
                v: {str(s): val_n_eff[v][s] for s in sorted(val_n_eff[v])} for v in sorted(val_n_eff)
            },
        },
        "test_segment": {
            "consumed_by": "H1/H3/H7/H8",
            "n_eval_files": len(evals),
            "gate_inputs": [str(p["eval_path"]) for p in gate_evals],
            "cross_station_inputs": [str(p["eval_path"]) for p in cross_evals],
            "per_seed_n_eff": {
                v: {
                    str(s): {
                        "run_id": by_variant[v][s]["run_id"],
                        "n_eff": by_variant[v][s]["n_eff"],
                        "n_eff_flag": by_variant[v][s]["n_eff_flag"],
                        "err_median": by_variant[v][s]["err_median"],
                        "n_beams": by_variant[v][s]["n_beams"],
                        "n_groups": by_variant[v][s]["n_groups"],
                        "code_version": by_variant[v][s].get("code_version"),
                    }
                    for s in sorted(by_variant[v])
                }
                for v in sorted(by_variant)
            },
        },
        "winner_variant": winner,
        "routing_basis": routing_basis,
        "H1": h1,
        "H3": h3,
        "H7": h7,
        "H8": h8,
        "beam_n_eff_snapshot": snapshot,
    }
    out_path = results_dir / GATES_FILENAME
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="line2 E1/E3 出口门禁（spec 40 [S3]；只读落盘产物）"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=Path, default=None,
                        help=f"`20` [S4] 容差线产物（默认 <results-dir>/{TOLERANCE_FILENAME}）")
    args = parser.parse_args(argv)
    payload = evaluate_gates_from_disk(args.results_dir, tolerance=args.tolerance)
    print(
        json.dumps(
            {
                "winner_variant": payload["winner_variant"],
                "H1": payload["H1"]["verdict"],
                "H3": payload["H3"]["verdict"],
                "H7": payload["H7"]["verdict"],
                "H8": payload["H8"]["verdict"],
                "H8_pass_rate": payload["H8"]["pass_rate"],
                "out": str(Path(args.results_dir) / GATES_FILENAME),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
