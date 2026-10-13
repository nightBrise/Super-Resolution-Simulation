"""`50` [S5]：Phase 1.5 出口门禁 G1–G5 的判定与落盘（只读产物，脚本化，无人工干预）。

五条门禁各由一个**独立纯函数**消费落盘 JSON 判定（便于正/反例测试），顶层
:func:`evaluate_e15_gates_from_disk` 只负责取数、调用与落盘 ``e15_gates.json``
（`50` [S5] C7：五门禁结论 + 每条输入快照 + routing + 暗桶 M2 分项）。

判定输入白名单（`50` [S3] C1；两段式为 `40` D6/D12 的沿用）：
- **routing** 只消费 **val 段** 的 val_nll（``<run>/summary.json``），入口调
  :func:`src.line2.metrics.decisions.assert_inputs`，并断言路径名就是 ``summary.json``；
- **G1–G5** 只消费 **test 段** 的落盘读数（``<run>/eval_tier*_station*.json`` 的
  ``per_beam[].err`` / ``.obs`` / ``.m2_buckets``、``bconst_v2.json`` 的逐束误差、
  ``e15_align_*.json`` 的对齐读数），入口断言 eval 文件自带 ``split="test"`` 且路径名
  以 ``eval_`` 开头；**N_eff 类字段不进任何判定**（仅在 eval 文件里披露，不被读取）。

统计口径（`50` [S1]，G1/G2/G4/G5 统一）：逐 (束, 子样本) 读数取中位数（eval 文件的
``per_beam[].err`` 已是该步的产物）→ 逐束对 5 种子取中位数 → 对所得 150 个束级读数做
束级统计；CI = 束级 paired bootstrap 95%（10,000 次、SeedSequence(20260825)，实现点
= :func:`src.line2.metrics.alignment.bootstrap_median_ci`）。B-const 无种子维（其束级
读数即逐束值）。

容差线（`50` [S3] C3）：G3 按**评估档直接索引** ``tolerance_v2.json``（整型档位键）；
M6 行已去偏，消费侧**不得**再次减 bias；`20` [S4] 的 N_eff 最近档映射在本批废止。

用法：
    python -m src.line2.experiments.phase15_gates --results-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from src.line2.data_pipeline.dataset import git_head
from src.line2.metrics import ANALYSIS_SEED, decisions
from src.line2.metrics.alignment import N_BOOT
from src.line2.metrics.alignment import bootstrap_median_ci as _bootstrap_median_ci
from src.line2.metrics.tolerance import N_OBSERVABLES

from . import gates as _gates

__all__ = [
    "E15_GATES_FILENAME",
    "E15_ROUTING_FILENAME",
    "EVAL_GLOB",
    "SUMMARY_FILENAME",
    "CURVE_V2_FILENAME",
    "TOLERANCE_V2_FILENAME",
    "BCONST_FILENAME",
    "BCONST_GROUPS_FILENAME",
    "ALIGN_PREFIX",
    "GATE_STATION",
    "CROSS_STATION",
    "GATE_TIER",
    "G5_TIERS",
    "N_SEEDS",
    "N_BEAMS_EXPECTED",
    "VERDICTS",
    "G1_WIN_RATE",
    "G2_DROP_FRAC",
    "G3_PASS_RATE",
    "G3_FAIL_RATE",
    "G4_STRONG",
    "G4_WEAK",
    "eval_filename",
    "eval_path",
    "align_filename",
    "assert_val_segment_paths",
    "assert_test_segment_paths",
    "beam_level_readings",
    "paired_advantage",
    "g1_verdict",
    "g5_combine",
    "gate_g1",
    "gate_g2",
    "gate_g3",
    "gate_g4",
    "gate_g5",
    "load_summaries",
    "route_arms",
    "write_routing",
    "load_routing",
    "load_eval_files",
    "load_bconst",
    "tolerance_v2_table",
    "load_alignments",
    "evaluate_e15_gates_from_disk",
    "main",
]

E15_GATES_FILENAME = "e15_gates.json"
E15_ROUTING_FILENAME = "e15_routing.json"
SUMMARY_FILENAME = "summary.json"
#: v2 尺子产物（E0' 三件；`50` [S5] C1 的时间戳门禁对象）
CURVE_V2_FILENAME = "e0_curve_v2.json"
TOLERANCE_V2_FILENAME = "tolerance_v2.json"
BCONST_FILENAME = "bconst_v2.json"
BCONST_GROUPS_FILENAME = "bconst_v2_groups.npz"
#: 对齐仪器读数（G2 的输入，逐 run 一份）
ALIGN_PREFIX = "e15_align_"
#: 评估产物 glob（`50` [S5] C8：每 run × 6 评估）
EVAL_GLOB = "eval_*.json"

VERDICT_PASS = _gates.VERDICT_PASS
VERDICT_FAIL = _gates.VERDICT_FAIL
VERDICT_INDETERMINATE = _gates.VERDICT_INDETERMINATE
VERDICT_DIRECTIONAL = _gates.VERDICT_DIRECTIONAL
VERDICTS = _gates.VERDICTS

#: 门禁聚合维度（`50` [S5]：站 0、@1×10⁴；G4 的跨站侧 = 站 1）
GATE_STATION = 0
CROSS_STATION = 1
GATE_TIER = 10_000
#: G5 的两档（站 0、参照同一 B-const 均值图）
G5_TIERS: tuple[int, ...] = (3_000, 100_000)
N_SEEDS = 5
#: test 束数（`50` [S1] 的 150 束统计口径；数据契约冻结，落盘供审计）
N_BEAMS_EXPECTED = 150

#: G1/G5 的赢率线（`50` [S5] C2/C6 预注册）
G1_WIN_RATE = 0.90
#: G2 的打乱落差线（`50` [S5] C3："drop_frac ≥ 0.80"）
G2_DROP_FRAC = 0.80
#: G3 的达标率双线（`50` [S5] C4：≥0.80 成立 / ≤0.50 不成立 / 之间方向成立）
G3_PASS_RATE = 0.80
G3_FAIL_RATE = 0.50
#: G4 的难度归一残差比（`50` [S5] C5：≤1.10 成立 / (1.10,1.25] 方向成立 / >1.25 不成立）
G4_STRONG = 1.10
G4_WEAK = 1.25

#: 判定表达式里用到的输入键名（[S3] C1 白名单；本模块的四处取数各占一名）
GATE_INPUT_KEYS: tuple[str, ...] = ("beam_err_m1", "obs13", "alignment", "bconst_beam_err")


def eval_filename(tier: int, station: int) -> str:
    """评估产物文件名（每 run × 3 档 × 2 站各一份；站标记既在文件名也在 payload）。"""
    return f"eval_tier{int(tier)}_station{int(station)}.json"


def eval_path(results_dir: str | Path, run_id: str, tier: int, station: int) -> Path:
    return Path(results_dir) / str(run_id) / eval_filename(int(tier), int(station))


def align_filename(run_id: str) -> str:
    return f"{ALIGN_PREFIX}{run_id}.json"


def _read_json(path: Path) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"缺产物：{path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------- 两段式路径断言
def assert_val_segment_paths(paths: Iterable[str | Path]) -> None:
    """val 段取数只能来自 ``summary.json``（`40` D6/D12 两段式，[S5] 自带实现点）。"""
    for path in paths:
        if Path(path).name != SUMMARY_FILENAME:
            raise ValueError(
                f"路由（val 段）消费了 {path}：只许读 <run>/{SUMMARY_FILENAME} 的 val_nll"
                "（`50` [S3] C1 两段不交叉）"
            )


def assert_test_segment_paths(paths: Iterable[str | Path]) -> None:
    """门禁（test 段）取数只能来自 ``eval_*.json``（两段式，[S5] 自带实现点）。"""
    for path in paths:
        name = Path(path).name
        if not name.startswith("eval_") or not name.endswith(".json"):
            raise ValueError(
                f"门禁（test 段）消费了 {path}：只许读 <run>/{EVAL_GLOB} 的 test 段读数"
                "（`50` [S3] C1 两段不交叉）"
            )


# ------------------------------------------------------------------ 束级读数
def beam_level_readings(evals: Sequence[Mapping], key: str = "err") -> dict[int, float]:
    """多 run（= 多种子）的 ``per_beam[].<key>`` → 逐束对种子取中位数（[S1] 第二步）。

    任一 run 的束集与交集不同即报错（不静默收窄统计域）：150 束口径是门禁的定义域，
    缺束的 run 不是"少一点样本"，而是另一批数据。
    """
    if not evals:
        raise ValueError("beam_level_readings：无 run 可聚合")
    per_run: list[dict[int, float]] = []
    for payload in evals:
        run_id = payload.get("run_id", "?")
        rows: dict[int, float] = {}
        for entry in payload.get("per_beam", ()):
            beam_id = int(entry["beam_id"])
            if key not in entry:
                raise ValueError(f"run {run_id}：per_beam 缺字段 {key!r}（`50` [S5] C8 的 schema）")
            rows[beam_id] = float(entry[key])
        if not rows:
            raise ValueError(f"run {run_id}：per_beam 为空")
        per_run.append(rows)
    beam_ids = sorted(per_run[0])
    for payload, rows in zip(evals, per_run):
        if sorted(rows) != beam_ids:
            raise ValueError(
                f"run {payload.get('run_id', '?')} 的束集与其它 run 不一致"
                "（门禁统计域 = 全部 run 共有的束集，不得静默收窄）"
            )
    return {
        beam_id: float(np.median(np.asarray([rows[beam_id] for rows in per_run], dtype=np.float64)))
        for beam_id in beam_ids
    }


def paired_advantage(
    model_errors: Mapping[int, float],
    reference_errors: Mapping[int, float],
    *,
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    """配对逐束优势（`50` [S1]）：delta = err_ref − err_model，win_rate / advantage / CI。

    CI = 束级 paired bootstrap 的中位数 95% 区间（10,000 次、根种子 20260825）；
    实现点复用 :func:`src.line2.metrics.alignment.bootstrap_median_ci`（单一实现点）。
    """
    model_keys, ref_keys = set(model_errors), set(reference_errors)
    if model_keys != ref_keys:
        raise ValueError(
            "配对逐束统计失败：模型与参照的束集不一致"
            f"（模型独有 {sorted(model_keys - ref_keys)[:5]}，参照独有 {sorted(ref_keys - model_keys)[:5]}）"
        )
    if not model_keys:
        raise ValueError("配对逐束统计失败：束集为空")
    beam_ids = sorted(model_keys)
    deltas = np.asarray(
        [float(reference_errors[b]) - float(model_errors[b]) for b in beam_ids], dtype=np.float64
    )
    stats = _bootstrap_median_ci(deltas, n_boot=n_boot, seed=seed)
    return {
        "n_beams": len(beam_ids),
        "beam_ids": beam_ids,
        "model_err": [float(model_errors[b]) for b in beam_ids],
        "reference_err": [float(reference_errors[b]) for b in beam_ids],
        "delta": [float(d) for d in deltas],
        "win_rate": float(np.mean(deltas > 0.0)),
        "advantage": float(stats["median"]),
        "ci95": [float(stats["ci95"][0]), float(stats["ci95"][1])],
        "aggregation": "per_beam_median_over_subsamples_then_seed_median_then_beam_stats",
        "bootstrap": {"n_boot": int(n_boot), "root_seed": int(seed), "unit": "beam"},
    }


# ------------------------------------------------------------------ 四值映射
def g1_verdict(win_rate: float, ci95: Sequence[float]) -> str:
    """G1/G5 单档的四值映射（`50` [S5] C2）。

    成立 = win_rate ≥ 0.90 且 CI 下界 > 0；方向成立 = CI 下界 > 0 且 win_rate < 0.90；
    不成立 = CI 上界 ≤ 0；不可判定 = CI 跨 0（下界 ≤ 0 < 上界）。四类互斥完备。
    """
    lo, hi = float(ci95[0]), float(ci95[1])
    if lo > 0.0 and float(win_rate) >= G1_WIN_RATE:
        return VERDICT_PASS
    if lo > 0.0:
        return VERDICT_DIRECTIONAL
    if hi <= 0.0:
        return VERDICT_FAIL
    return VERDICT_INDETERMINATE


def g5_combine(tier_verdicts: Mapping[int, str]) -> str:
    """G5 的两档组合规则（`50` [S5] C6）：①不成立 > ②不可判定 > ③两档成立 > ④方向成立。"""
    if not tier_verdicts:
        raise ValueError("g5_combine：无档位结论可组合（G5 的档集 = {3×10³, 1×10⁵}）")
    values = [str(v) for v in tier_verdicts.values()]
    if VERDICT_FAIL in values:
        return VERDICT_FAIL
    if VERDICT_INDETERMINATE in values:
        return VERDICT_INDETERMINATE
    if all(v == VERDICT_PASS for v in values):
        return VERDICT_PASS
    return VERDICT_DIRECTIONAL


def _missing_gate(name: str, reason: str, **extra) -> dict:
    return {"gate": name, "verdict": VERDICT_INDETERMINATE, "reason_code": reason, **extra}


# ------------------------------------------------------------------ G1–G5
def gate_g1(
    evals: Sequence[Mapping],
    bconst_station: Mapping | None,
    *,
    tier: int = GATE_TIER,
    station: int = GATE_STATION,
) -> dict:
    """G1 密度反超先验（获胜臂 @1×10⁴ 站 0，参照 = B-const 站 0）。"""
    if bconst_station is None:
        return _missing_gate("G1", "bconst_missing", tier=int(tier), station=int(station))
    reference = _bconst_errors(bconst_station)
    if not evals:
        return _missing_gate("G1", "eval_missing", tier=int(tier), station=int(station))
    model = beam_level_readings(evals, "err")
    stats = paired_advantage(model, reference)
    return {
        "gate": "G1",
        "verdict": g1_verdict(stats["win_rate"], stats["ci95"]),
        "reason_code": None,
        "tier": int(tier),
        "station": int(station),
        "n_beams": stats["n_beams"],
        "win_rate": stats["win_rate"],
        "advantage": stats["advantage"],
        "ci95": stats["ci95"],
        "thresholds": {
            "win_rate_min": G1_WIN_RATE,
            "ci_lower": "advantage 的 95% CI 下界 > 0",
        },
        "aggregation": stats["aggregation"],
        "bootstrap": stats["bootstrap"],
        "source": {
            "eval_paths": [str(e.get("eval_path", "")) for e in evals],
            "reference_path": str(bconst_station.get("source_path", "")),
            "reference_station": int(bconst_station.get("station", GATE_STATION)),
        },
        "inputs": {
            "beam_ids": stats["beam_ids"],
            "model_err": stats["model_err"],
            "reference_err": stats["reference_err"],
            "delta": stats["delta"],
        },
    }


def gate_g2(
    alignments: Mapping[int, Mapping],
    *,
    tier: int = GATE_TIER,
    station: int = GATE_STATION,
) -> dict:
    """G2 个体信息（获胜臂 5 种子，选型 ckpt，@1×10⁴ 站 0，prior = B-const 站 0）。

    k_a = 满足 [cos_true 的 CI > 0] 的种子数、k_and = 满足 [CI > 0 且 drop_frac ≥ 0.80]
    的种子数；四值映射见 `50` [S5] C3。读数不足 5 种子时判不可判定（不猜值）。
    """
    seeds = sorted(int(s) for s in alignments)
    if len(seeds) < N_SEEDS:
        return _missing_gate(
            "G2", "alignment_missing", tier=int(tier), station=int(station),
            seeds=seeds, n_seeds=len(seeds), n_seeds_expected=N_SEEDS,
        )
    per_seed: dict[str, dict] = {}
    k_a = 0
    k_and = 0
    for seed in seeds:
        entry = alignments[seed]
        if int(entry.get("n_low", tier)) != int(tier) or int(entry.get("station", station)) != int(station):
            raise ValueError(
                f"对齐读数（seed {seed}）的档位/站不符：n_low={entry.get('n_low')!r} "
                f"station={entry.get('station')!r}（G2 定义在 @{tier} 站 {station}）"
            )
        ci = [float(v) for v in entry["cos_true_ci95"]]
        drop = float(entry["drop_frac"])
        positive = ci[0] > 0.0
        k_a += int(positive)
        k_and += int(positive and drop >= G2_DROP_FRAC)
        per_seed[str(seed)] = {
            "run_id": entry.get("run_id"),
            "cos_true": float(entry["cos_true"]),
            "cos_true_ci95": ci,
            "cos_scr": float(entry.get("cos_scr", float("nan"))),
            "drop_frac": drop,
            "disp_ratio": entry.get("disp_ratio"),
            "ci_positive": bool(positive),
        }
    if k_a >= 4 and k_and >= 4:
        verdict = VERDICT_PASS
    elif k_a >= 4 and k_and == 3:
        verdict = VERDICT_DIRECTIONAL
    elif k_a == 3 or (k_a >= 4 and k_and <= 2):
        verdict = VERDICT_INDETERMINATE
    elif k_a <= 2:
        verdict = VERDICT_FAIL
    else:  # pragma: no cover - 四支已覆盖 k_a ∈ {0..5} 的全部分区
        verdict = VERDICT_INDETERMINATE
    return {
        "gate": "G2",
        "verdict": verdict,
        "reason_code": None,
        "tier": int(tier),
        "station": int(station),
        "n_seeds": len(seeds),
        "k_a": int(k_a),
        "k_and": int(k_and),
        "thresholds": {"drop_frac_min": G2_DROP_FRAC, "ci_lower": "cos_true 的 95% CI 下界 > 0"},
        "per_seed": per_seed,
        "source": {"alignment_paths": [str(alignments[s].get("align_path", "")) for s in seeds],
                   "reference_path": str(alignments[seeds[0]].get("bconst_groups", ""))},
        "inputs": {
            "cos_true_ci_lower": [per_seed[str(s)]["cos_true_ci95"][0] for s in seeds],
            "drop_frac": [per_seed[str(s)]["drop_frac"] for s in seeds],
        },
    }


def gate_g3(
    evals: Sequence[Mapping],
    tolerance_row: Sequence[float] | None,
    *,
    tier: int = GATE_TIER,
    station: int = GATE_STATION,
) -> dict:
    """G3 观测量达标（获胜臂 @1×10⁴ 站 0，v2 容差线、M6 去偏，逐观测量先对种子取中位数）。

    容差线按**评估档直接索引**（`50` [S3] C3 的取代条款），不做 N_eff 最近档映射。
    达标率 = 达标的 (束, 观测量) 对数占比（达标 = 偏差 ≤ 容差线，与 `20` [S4] 同口径）。
    """
    if tolerance_row is None:
        return _missing_gate("G3", "tolerance_missing", tier=int(tier), station=int(station))
    if not evals:
        return _missing_gate("G3", "eval_missing", tier=int(tier), station=int(station))
    tol = np.asarray(tolerance_row, dtype=np.float64)
    if tol.shape != (N_OBSERVABLES,):
        raise ValueError(f"容差线形状 {tol.shape} ≠ ({N_OBSERVABLES},)")
    beam_ids = sorted({int(b["beam_id"]) for b in evals[0]["per_beam"]})
    dev = _gates.h8_deviation_matrix(evals, beam_ids)
    scored = _gates.evaluate_h8(dev, tol)
    pass_rate = float(scored["pass_rate"])
    if pass_rate >= G3_PASS_RATE:
        verdict = VERDICT_PASS
    elif pass_rate <= G3_FAIL_RATE:
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_DIRECTIONAL
    return {
        "gate": "G3",
        "verdict": verdict,
        "reason_code": None,
        "tier": int(tier),
        "station": int(station),
        "n_beams": len(beam_ids),
        "pass_rate": pass_rate,
        "n_observables": int(N_OBSERVABLES),
        "thresholds": {"pass_min": G3_PASS_RATE, "fail_max": G3_FAIL_RATE},
        "aggregation": "per_observable_seed_median_then_pass_rate",
        "per_observable": scored["per_observable"],
        "source": {"eval_paths": [str(e.get("eval_path", "")) for e in evals],
                   "tolerance": str(evals[0].get("tolerance_path", ""))},
        "inputs": {
            "tolerance": [float(v) for v in tol],
            "beam_ids": beam_ids,
            "deviation": [[float(v) for v in row] for row in dev.tolist()],
        },
    }


def gate_g4(
    evals_s0: Sequence[Mapping],
    evals_s1: Sequence[Mapping],
    bconst_s0: Mapping | None,
    bconst_s1: Mapping | None,
    *,
    tier: int = GATE_TIER,
) -> dict:
    """G4 跨站真迁移（获胜臂站 1 @1×10⁴；限于 [S1] 晶格边界的站 0 → 站 1）。

    R_diff = (err_s1 − B_s1)/(err_s0 − B_s0)，式中各量取对应 150 个束级读数的**中位数**；
    原始比 err_s1/err_s0 并列落盘披露、不进判定（`50` [S5] C5 / D5）。
    """
    missing = [
        name for name, value in (
            ("eval_s0", bool(evals_s0)), ("eval_s1", bool(evals_s1)),
            ("bconst_s0", bconst_s0 is not None), ("bconst_s1", bconst_s1 is not None),
        ) if not value
    ]
    if missing:
        return _missing_gate("G4", "input_missing", tier=int(tier), missing=missing)
    err_s0 = float(np.median(np.asarray(list(beam_level_readings(evals_s0, "err").values()))))
    err_s1 = float(np.median(np.asarray(list(beam_level_readings(evals_s1, "err").values()))))
    b_s0 = float(np.median(np.asarray(list(_bconst_errors(bconst_s0).values()), dtype=np.float64)))
    b_s1 = float(np.median(np.asarray(list(_bconst_errors(bconst_s1).values()), dtype=np.float64)))
    denominator = err_s0 - b_s0
    raw_ratio = (err_s1 / err_s0) if err_s0 != 0.0 else float("nan")
    if denominator == 0.0 or not np.isfinite(denominator):
        return _missing_gate(
            "G4", "denominator_zero", tier=int(tier), err_s0=err_s0, b_s0=b_s0,
            err_s1=err_s1, b_s1=b_s1, raw_err_ratio=raw_ratio,
        )
    r_diff = (err_s1 - b_s1) / denominator
    if r_diff <= G4_STRONG:
        verdict = VERDICT_PASS
    elif r_diff <= G4_WEAK:
        verdict = VERDICT_DIRECTIONAL
    else:
        verdict = VERDICT_FAIL
    return {
        "gate": "G4",
        "verdict": verdict,
        "reason_code": None,
        "tier": int(tier),
        "r_diff": float(r_diff),
        "raw_err_ratio": float(raw_ratio),
        "err_median": {"station0": err_s0, "station1": err_s1},
        "bconst_median": {"station0": b_s0, "station1": b_s1},
        "excess": {"station0": denominator, "station1": err_s1 - b_s1},
        "thresholds": {"strong_max": G4_STRONG, "weak_max": G4_WEAK},
        "note": "迁移限于 [S1] 晶格边界的站 0 → 站 1；原始比仅作披露、不进判定",
        "source": {
            "eval_paths": {
                "station0": [str(e.get("eval_path", "")) for e in evals_s0],
                "station1": [str(e.get("eval_path", "")) for e in evals_s1],
            },
            "reference_path": str((bconst_s0 or {}).get("source_path", "")),
        },
        "inputs": {
            "err_s0_median": err_s0, "err_s1_median": err_s1,
            "bconst_s0_median": b_s0, "bconst_s1_median": b_s1,
        },
    }


def gate_g5(
    evals_by_tier: Mapping[int, Sequence[Mapping]],
    bconst_station: Mapping | None,
    *,
    station: int = GATE_STATION,
) -> dict:
    """G5 N 泛化（获胜臂 @3×10³ 与 @1×10⁵ 站 0，参照 = B-const 站 0——同一均值图两档共用）。

    两档分别按 G1 同款四值判定，再按 `50` [S5] C6 的组合规则合成；两档结论并列落盘。
    """
    tier_verdicts: dict[int, str] = {}
    per_tier: dict[str, dict] = {}
    for tier in G5_TIERS:
        evals = evals_by_tier.get(int(tier), ())
        if bconst_station is None or not evals:
            verdict = VERDICT_INDETERMINATE
            reason = "bconst_missing" if bconst_station is None else "eval_missing"
            per_tier[str(int(tier))] = {"verdict": verdict, "reason_code": reason}
            tier_verdicts[int(tier)] = verdict
            continue
        stats = paired_advantage(beam_level_readings(evals, "err"), _bconst_errors(bconst_station))
        verdict = g1_verdict(stats["win_rate"], stats["ci95"])
        tier_verdicts[int(tier)] = verdict
        per_tier[str(int(tier))] = {
            "verdict": verdict,
            "reason_code": None,
            "n_beams": stats["n_beams"],
            "win_rate": stats["win_rate"],
            "advantage": stats["advantage"],
            "ci95": stats["ci95"],
            "source": {"eval_paths": [str(e.get("eval_path", "")) for e in evals]},
            "inputs": {
                "beam_ids": stats["beam_ids"],
                "model_err": stats["model_err"],
                "reference_err": stats["reference_err"],
                "delta": stats["delta"],
            },
        }
    return {
        "gate": "G5",
        "verdict": g5_combine(tier_verdicts),
        "reason_code": None,
        "station": int(station),
        "tiers": [int(t) for t in G5_TIERS],
        "tier_verdicts": {str(int(t)): v for t, v in tier_verdicts.items()},
        "thresholds": {"win_rate_min": G1_WIN_RATE, "ci_lower": "advantage 的 95% CI 下界 > 0"},
        "combine_rule": "①任一档不成立→不成立；②否则任一档不可判定→不可判定；③两档均成立→成立；④其余→方向成立",
        "per_tier": per_tier,
        "source": {"reference_path": str((bconst_station or {}).get("source_path", "")),
                   "reference_station": int((bconst_station or {}).get("station", GATE_STATION))},
    }


def _bconst_errors(record: Mapping) -> dict[int, float]:
    """B-const 记录的逐束误差向量（JSON 键为字符串束号，`50` [S3] C4 的落盘格式）。"""
    values = record.get("per_beam_m1")
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"B-const 记录缺 per_beam_m1（来源 {record.get('source_path')!r}）")
    return {int(k): float(v) for k, v in values.items()}


# ------------------------------------------------------------------ 路由（val 段）
def load_summaries(
    results_dir: str | Path, runs_by_arm: Mapping[str, Sequence[str]]
) -> dict[str, dict[int, dict]]:
    """读两臂各 run 的 ``summary.json``（**val 段唯一取数口**），逐臂按种子升序。

    断言：路径名 = ``summary.json``（[S5] 自带的 val 段路径断言）、payload 的 variant
    与臂一致、`50` [S4] 的三档/站 0 口径（val_nll 的定义域）、``best_val_nll`` 有限。
    入口调 :func:`src.line2.metrics.decisions.assert_inputs`（白名单：val_nll）。
    """
    from src.line2.training.config_check import PHASE15_STATIONS, PHASE15_TIERS

    results_dir = Path(results_dir)
    decisions.assert_inputs(decisions.KIND_ROUTING, ("val_nll",))
    out: dict[str, dict[int, dict]] = {}
    for arm, run_ids in runs_by_arm.items():
        bucket: dict[int, dict] = {}
        for run_id in run_ids:
            path = results_dir / str(run_id) / SUMMARY_FILENAME
            assert_val_segment_paths([path])
            payload = _read_json(path)
            payload["summary_path"] = str(path)
            payload["run_id"] = str(run_id)
            if str(payload.get("variant")) != str(arm):
                raise ValueError(
                    f"{path}：variant={payload.get('variant')!r} 与臂 {arm!r} 不符"
                )
            if list(payload.get("tiers", ())) != list(PHASE15_TIERS):
                raise ValueError(
                    f"{path}：tiers={payload.get('tiers')!r} ≠ {list(PHASE15_TIERS)}"
                    "（val_nll 的三档定义域，`50` [S3] C2）"
                )
            if list(payload.get("stations", ())) != list(PHASE15_STATIONS):
                raise ValueError(
                    f"{path}：stations={payload.get('stations')!r} ≠ {list(PHASE15_STATIONS)}"
                    "（val_nll 只在站 0 定义，`50` [S3] C2）"
                )
            value = payload.get("best_val_nll")
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"{path}：best_val_nll={value!r} 非有限——路由无输入")
            seed = int(payload["seed"])
            if seed in bucket:
                raise ValueError(f"臂 {arm} 种子 {seed} 有多份 summary（{bucket[seed]['run_id']} / {run_id}）")
            bucket[seed] = payload
        out[str(arm)] = {seed: bucket[seed] for seed in sorted(bucket)}
    return out


def route_arms(summaries: Mapping[str, Mapping[int, Mapping]]) -> dict:
    """获胜臂路由（`50` [S5]）：两臂按 val_nll 排序取优，整体取代 `40` [S3] 的计数分支。

    臂级读数 = 该臂 5 种子 ``best_val_nll`` 的中位数（[S1] 的中位数口径；spec 未另行
    规定臂级聚合，取与全批一致的中位数哲学）；排序与获胜判定由
    :func:`src.line2.metrics.decisions.route_by_selection_reading` 承担（单一实现点）。
    并列时该函数按臂名字典序取小者，确定性。

    1×10⁴ 单档分量（``best_val_nll_per_tier["10000"]``）一并落盘并对同一排序求解，
    两者的胜者不一致时置 ``reading_disagreement``（供报告披露，不静默择一）。
    """
    arms = {str(arm): rows for arm, rows in summaries.items()}
    for arm, rows in arms.items():
        seeds = sorted(int(s) for s in rows)
        if len(seeds) != N_SEEDS:
            raise ValueError(
                f"臂 {arm} 的 summary 只有 {len(seeds)} 个种子 {seeds}；"
                f"`50` [S5] 矩阵要求每臂 {N_SEEDS} 个"
            )
    def _median_reading(rows: Mapping[int, Mapping], key: str) -> dict[int, float]:
        values: dict[int, float] = {}
        for seed, payload in sorted(rows.items()):
            if key == "val_nll":
                value = payload["best_val_nll"]
            else:
                per_tier = payload.get("best_val_nll_per_tier") or {}
                if key not in per_tier:
                    raise ValueError(
                        f"summary（seed {seed}）缺 best_val_nll_per_tier[{key!r}]"
                        "——1×10⁴ 单档分量不可得"
                    )
                value = per_tier[key]
            values[int(seed)] = float(value)
        return values

    per_arm: dict[str, dict] = {}
    readings: dict[str, float] = {}
    tier_readings: dict[str, float] = {}
    for arm, rows in arms.items():
        values = _median_reading(rows, "val_nll")
        tier_values = _median_reading(rows, str(GATE_TIER))
        readings[arm] = float(np.median(np.asarray(list(values.values()), dtype=np.float64)))
        tier_readings[arm] = float(np.median(np.asarray(list(tier_values.values()), dtype=np.float64)))
        per_arm[arm] = {
            "per_seed_val_nll": {str(s): v for s, v in values.items()},
            "median_val_nll": readings[arm],
            "per_seed_tier_1e4": {str(s): v for s, v in tier_values.items()},
            "median_tier_1e4": tier_readings[arm],
            "summary_paths": {str(s): str(rows[s].get("summary_path", "")) for s in sorted(rows)},
        }
    winner = decisions.route_by_selection_reading(readings)
    winner_by_tier = decisions.route_by_selection_reading(tier_readings)
    return {
        "basis": "val_nll",
        "spec": "`50` [S5] 路由：两臂按 val_nll 排序取优（整体取代 `40` [S3] 的计数分支）",
        "source": {"summary": f"{SUMMARY_FILENAME}:best_val_nll", "segment": "val"},
        "seeds": sorted(int(s) for s in next(iter(arms.values()))),
        "per_arm": per_arm,
        "arm_reading": {"aggregation": "per_seed_median", **readings},
        "winner_arm": winner,
        "tier_1e4_reading": tier_readings,
        "winner_by_tier_1e4": winner_by_tier,
        "reading_disagreement": bool(winner != winner_by_tier),
        "tie_break": "同值并列时按臂名字典序取小者（route_by_selection_reading）",
    }


def write_routing(
    results_dir: str | Path,
    runs_by_arm: Mapping[str, Sequence[str]],
    *,
    out: str | Path | None = None,
) -> dict:
    """读 val 段 summary → 落 ``e15_routing.json``（返回同一 payload）。"""
    results_dir = Path(results_dir)
    summaries = load_summaries(results_dir, runs_by_arm)
    path = Path(out) if out is not None else results_dir / E15_ROUTING_FILENAME
    payload = route_arms(summaries)
    payload.update({
        "results_dir": str(results_dir),
        "runs_by_arm": {arm: [str(r) for r in runs] for arm, runs in runs_by_arm.items()},
        "code_version": git_head(),
        "routing_path": str(path),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_routing(results_dir: str | Path) -> dict:
    """读 ``e15_routing.json``（门禁的获胜臂输入；缺文件即报错，不降级猜臂）。"""
    results_dir = Path(results_dir)
    payload = _read_json(results_dir / E15_ROUTING_FILENAME)
    for key in ("winner_arm", "per_arm", "basis"):
        if key not in payload:
            raise ValueError(f"{results_dir / E15_ROUTING_FILENAME}：缺字段 {key!r}")
    return payload


# ------------------------------------------------------------ test 段取数（门禁输入）
def load_eval_files(
    results_dir: str | Path,
    run_ids: Sequence[str],
    *,
    tier: int,
    station: int,
    require: Sequence[str] = ("err",),
) -> list[dict]:
    """读某（档, 站）的全部 run 评估产物；缺文件时返回空表（由门禁判"评估缺失"）。

    断言（[S5] 自带的 test 段路径断言 + `00` GC3 的 test 只评一次）：
    ``split == "test"``、payload 的 station/n_low/run_id 与请求一致、``per_beam`` 含所需字段。
    N_eff 类字段**不被读取**（仅随文件披露）。
    """
    results_dir = Path(results_dir)
    out: list[dict] = []
    for run_id in run_ids:
        path = eval_path(results_dir, run_id, tier, station)
        if not path.is_file():
            continue
        assert_test_segment_paths([path])
        payload = _read_json(path)
        payload["eval_path"] = str(path)
        if payload.get("split") != "test":
            raise ValueError(
                f"{path}：split={payload.get('split')!r}——出口门禁只消费 test 段"
                "（`50` [S3] C1 两段不交叉）"
            )
        if int(payload.get("station", -1)) != int(station):
            raise ValueError(f"{path}：station={payload.get('station')!r} ≠ 请求的 {station}")
        if int(payload.get("n_low", -1)) != int(tier):
            raise ValueError(f"{path}：n_low={payload.get('n_low')!r} ≠ 请求的 {tier}")
        if str(payload.get("run_id")) != str(run_id):
            raise ValueError(f"{path}：run_id={payload.get('run_id')!r} ≠ 目录 {run_id!r}")
        for entry in payload.get("per_beam", ()):
            for field in require:
                if field not in entry:
                    raise ValueError(
                        f"{path}：per_beam 缺字段 {field!r}（`50` [S5] C8 的 v1 schema）"
                    )
        out.append(payload)
    return out


def load_bconst(results_dir: str | Path, *, path: str | Path | None = None) -> tuple[dict, dict]:
    """读 ``bconst_v2.json`` → ``({station: 记录}, 文件级元信息)``（含 source_path 溯源）。"""
    results_dir = Path(results_dir)
    bconst_path = Path(path) if path is not None else results_dir / BCONST_FILENAME
    payload = _read_json(bconst_path)
    stations = payload.get("station")
    if not isinstance(stations, Mapping) or not stations:
        raise ValueError(f"{bconst_path}：缺 station 段（B-const 两站口径，`50` [S3] C4）")
    records: dict[int, dict] = {}
    for key, record in stations.items():
        key_int = int(key)
        record = dict(record)
        record["source_path"] = str(bconst_path)
        record["station"] = key_int
        records[key_int] = record
    meta = {
        "path": str(bconst_path),
        "data_version": payload.get("data_version"),
        "data_dir": payload.get("data_dir"),
        "completed_at": payload.get("completed_at"),
        "groups_file": payload.get("groups_file"),
        "stations": sorted(records),
        # 逐站输入快照（G1/G4/G5 的参照读数；报告第 4 章直接渲染，读者不必回读产物）
        "per_station": {
            str(station): {
                "group_sha256": record.get("group_sha256"),
                "err_median": record.get("err_median"),
                "err_min": record.get("err_min"),
                "err_max": record.get("err_max"),
                "n_eval_beams": record.get("n_eval_beams"),
                "eval_split": record.get("eval_split"),
            }
            for station, record in sorted(records.items())
        },
    }
    return records, meta


def tolerance_v2_table(
    results_dir: str | Path, *, path: str | Path | None = None
) -> tuple[dict[int, np.ndarray], str]:
    """读 ``tolerance_v2.json``（**经 `gates._load_tolerance` 直读**，整型档位键）。

    返回 ``({档位: 13 维向量}, 路径)``；文件缺失时返回 ``({}, 路径)``（由门禁判不可判定，
    不现场猜容差）。M6 行已在产物侧去偏，此处**不再**做任何 bias 修正。
    """
    results_dir = Path(results_dir)
    tol_path = Path(path) if path is not None else results_dir / TOLERANCE_V2_FILENAME
    table = _gates._load_tolerance(tol_path)
    if not table:
        return {}, str(tol_path)
    for tier, row in table.items():
        if np.asarray(row).shape != (N_OBSERVABLES,):
            raise ValueError(
                f"{tol_path}：档 {tier} 的容差向量形状 {np.asarray(row).shape} ≠ ({N_OBSERVABLES},)"
            )
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in table.items()}, str(tol_path)


def load_alignments(results_dir: str | Path, run_ids: Sequence[str]) -> dict[int, dict]:
    """读获胜臂各 run 的 ``e15_align_<run_id>.json``（缺文件即从映射中缺席，由 G2 判缺）。"""
    results_dir = Path(results_dir)
    out: dict[int, dict] = {}
    for run_id in run_ids:
        path = results_dir / align_filename(run_id)
        if not path.is_file():
            continue
        payload = _read_json(path)
        payload["align_path"] = str(path)
        out[int(payload["seed"])] = payload
    return out


# --------------------------------------------------------------- 顶层装配与落盘
def evaluate_e15_gates_from_disk(
    results_dir: str | Path,
    *,
    runs_by_arm: Mapping[str, Sequence[str]] | None = None,
    tolerance: str | Path | None = None,
    out: str | Path | None = None,
) -> dict:
    """只读落盘产物 → ``<results_dir>/e15_gates.json``（`50` [S5] C7）。

    消费面与判定域：routing（val 段 = summary.json 的 val_nll）确定获胜臂；该臂的
    test 段评估读数（@1×10⁴ 站 0/站 1、@3×10³ 与 @1×10⁵ 站 0）、容差线
    （``tolerance_v2.json``，按评估档索引）、B-const 两站逐束误差、对齐读数
    （``e15_align_*``）分别驱动 G1–G5；暗桶 M2 由 :func:`gates.h8_dark_bucket_report`
    单独报告（`40` [S3] 条款沿用）。同输入两次调用输出逐位一致。
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在：{results_dir}")
    decisions.assert_inputs(decisions.KIND_GATE, GATE_INPUT_KEYS)

    routing = load_routing(results_dir)
    winner = str(routing["winner_arm"])
    runs_by_arm = routing.get("runs_by_arm") if runs_by_arm is None else dict(runs_by_arm)
    if not runs_by_arm or winner not in runs_by_arm:
        raise ValueError(
            f"e15_routing.json 未记录获胜臂 {winner!r} 的 run 列表（runs_by_arm），门禁无取数口"
        )
    winner_runs = [str(r) for r in runs_by_arm[winner]]

    bconst_records, bconst_meta = load_bconst(results_dir)
    bconst_s0 = bconst_records.get(GATE_STATION)
    bconst_s1 = bconst_records.get(CROSS_STATION)
    tolerance_table, tolerance_path = tolerance_v2_table(results_dir, path=tolerance)
    tolerance_row = tolerance_table.get(GATE_TIER)

    evals_gate = load_eval_files(
        results_dir, winner_runs, tier=GATE_TIER, station=GATE_STATION, require=("err", "obs", "m2_buckets")
    )
    for payload in evals_gate:  # G3 的容差线溯源随读数落盘（读者不必回读本模块）
        payload["tolerance_path"] = tolerance_path
    evals_cross = load_eval_files(results_dir, winner_runs, tier=GATE_TIER, station=CROSS_STATION)
    evals_g5 = {
        int(tier): load_eval_files(results_dir, winner_runs, tier=int(tier), station=GATE_STATION)
        for tier in G5_TIERS
    }
    alignments = load_alignments(results_dir, winner_runs)

    g1 = gate_g1(evals_gate, bconst_s0)
    g2 = gate_g2(alignments)
    g3 = gate_g3(evals_gate, tolerance_row)
    g4 = gate_g4(evals_gate, evals_cross, bconst_s0, bconst_s1)
    g5 = gate_g5(evals_g5, bconst_s0)

    beam_ids = sorted({int(b["beam_id"]) for p in evals_gate for b in p["per_beam"]})
    runs_by_seed = {int(p["seed"]): p for p in evals_gate}
    dark_bucket = _gates.h8_dark_bucket_report(runs_by_seed, beam_ids)

    out_path = Path(out) if out is not None else results_dir / E15_GATES_FILENAME
    payload = {
        "results_dir": str(results_dir),
        "spec": "`50` [S5]：实验矩阵、评估与出口门禁 G1–G5",
        "code_version": git_head(),
        "gates_path": str(out_path),
        "gate_dims": {
            "split": "test", "station": GATE_STATION, "cross_station": CROSS_STATION,
            "tier": GATE_TIER, "g5_tiers": [int(t) for t in G5_TIERS],
            "n_seeds_expected": N_SEEDS, "n_beams_expected": N_BEAMS_EXPECTED,
        },
        "routing": routing,
        "arms": {
            "winner": winner,
            "run_ids": winner_runs,
            "seeds": sorted(runs_by_seed),
            "n_runs": len(winner_runs),
        },
        "products": {
            "bconst": bconst_meta,
            "bconst_groups": str(results_dir / BCONST_GROUPS_FILENAME),
            "bconst_groups_present": (results_dir / BCONST_GROUPS_FILENAME).is_file(),
            "tolerance": tolerance_path,
            "tolerance_tiers": sorted(int(t) for t in tolerance_table),
            "curve": str(results_dir / CURVE_V2_FILENAME),
        },
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "G4": g4,
        "G5": g5,
        "verdicts": {name: payload_gate["verdict"] for name, payload_gate in
                     (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4), ("G5", g5))},
        "dark_bucket_m2": dark_bucket,
        "inputs_snapshot": {
            "gate_evals": [str(p["eval_path"]) for p in evals_gate],
            "cross_station_evals": [str(p["eval_path"]) for p in evals_cross],
            "g5_evals": {str(t): [str(e["eval_path"]) for e in evals] for t, evals in evals_g5.items()},
            "alignment": {str(s): str(p["align_path"]) for s, p in sorted(alignments.items())},
            "routing": str(results_dir / E15_ROUTING_FILENAME),
            "bconst": bconst_meta["path"],
            "tolerance": tolerance_path,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="line2 Phase 1.5 出口门禁 G1–G5（spec 50 [S5]）")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=Path, default=None,
                        help=f"v2 容差线产物（默认 <results-dir>/{TOLERANCE_V2_FILENAME}）")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = evaluate_e15_gates_from_disk(
        args.results_dir, tolerance=args.tolerance, out=args.out
    )
    print(
        json.dumps(
            {
                "winner_arm": payload["arms"]["winner"],
                **payload["verdicts"],
                "routing_reading_disagreement": payload["routing"]["reading_disagreement"],
                "out": payload["gates_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
