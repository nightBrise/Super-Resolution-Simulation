"""T0 实验 4：H8 十三项观测量误差的偏差/方差解剖（只读产物，纯 CPU）。

口径与 H8 官方判定一致（`src/line2/experiments/gates.py`、`src/line2/metrics/tolerance.py`）：

- 聚合 = test 束 × 站 0 × 1e4 × 种子**中位数**（`20` [S4]）：逐束、逐观测量先对 run 取中位数，
  再判 ``偏差 ≤ 容差线``；
- 容差线 = 该束 N_eff 中位数在对数距离上最近的曲线档的 ``dev_table`` 行（`20` [S4]）；
  容差行本身 = 该档裸子样本 |Δ| 分布的 95% 分位（val+test 300 束池化）。

**关键定义边界**：`per_beam[].obs` 是观测量偏差的**绝对值**向量（`observable_vector` 对
M4 的 11 个分量与 M5/M6 都先取 abs，见 `metrics.sigma_errors`/`halo_fraction_gap`/
`hf_energy_ratio_gap`），落盘产物不含符号。故本实验的"偏移"只能按**幅值共同成分**度量：
若所有束的误差同号，幅值中位数即等于符号偏移；符号混杂时 ``|m − median(m)|`` 是真实
校准残差的下界。据此，oracle 校准通过率是**上界**（与"oracle 用 test 信息"的方向一致）。

偏差/方差判据（本实验预注册，脚本内实现）：
- ``r = MAD(|Δ|) / median(|Δ|)``；零均值高斯误差经 abs 折叠后（半正态）的参考值
  ``r_ref ≈ 0.914`` 由脚本内数值实验给出；
- ``r < 0.5`` → 偏移主导；``0.5 ≤ r < 0.9`` → 混合；``r ≥ 0.9`` → 方差主导；
- 交叉核对：折叠正态 MLE 拟合 ``δ ~ N(b, σ)``（b 为跨束共同偏移）后的 ``b²/(b²+σ²)``；
- 另有阈值敏感性表（0.4/0.5/0.6/0.75/0.9）供读者自行重判。

运行：``python exp4_decompose.py``（只读 ``results/``，输出写本目录）。
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
if str(REPO) not in sys.path:  # 以脚本路径直接运行时补项目根（`src.` 导入）
    sys.path.insert(0, str(REPO))

from src.line2.experiments.gates import (  # noqa: E402
    OBSERVABLE_NAMES,
    evaluate_h8,
    h8_deviation_matrix,
    winner_variant,
)
from src.line2.metrics.tolerance import nearest_node, tolerance_for  # noqa: E402

RESULTS = REPO / "studies" / "line2_particle_sr" / "results"

# H8 官方聚合维度（`40` [S3]，冻结）
GATE_STATION = 0
GATE_N_LOW = 10_000
E1_SEEDS = 5

#: 幅值分布形状判据（折叠正态 δ ~ N(b, σ) 的数值参考，见 :func:`folded_reference`）：
#: β=b/σ=0（纯零均值噪声）时 r=MAD/median≈0.592；β=1（偏差份额 0.5）时 r≈0.238。
#: 主判据 = ``r < R_BIAS_HALF``（等价于共用偏移份额 > 0.5）。
R_PURE_NOISE = 0.592
R_BIAS_HALF = 0.238
RATIO_SENSITIVITY = (0.1, 0.15, 0.2, R_BIAS_HALF, 0.3, 0.5, R_PURE_NOISE)

#: 观测量物理含义（读自 `metrics.py` M4/M5/M6 与 `HALO_NSIGMA=4`、`_HF_SIGMA_BINS=1`）
OBS_MEANING = {
    "sigma_xx_rel": "水平方向 RMS 尺寸 σ_xx 的相对误差（Σ 对角元）",
    "sigma_xpxp_rel": "水平发散角 RMS σ_xp xp 的相对误差",
    "sigma_yy_rel": "竖直方向 RMS 尺寸 σ_yy 的相对误差",
    "sigma_ypyp_rel": "竖直发散角 RMS σ_yp yp 的相对误差",
    "abs_delta_rho_x_xp": "相关系数 ρ(x,xp) 的绝对变化 |Δρ|",
    "abs_delta_rho_x_y": "相关系数 ρ(x,y) 的绝对变化 |Δρ|",
    "abs_delta_rho_x_yp": "相关系数 ρ(x,yp) 的绝对变化 |Δρ|",
    "abs_delta_rho_xp_y": "相关系数 ρ(xp,y) 的绝对变化 |Δρ|",
    "abs_delta_rho_xp_yp": "相关系数 ρ(xp,yp) 的绝对变化 |Δρ|",
    "abs_delta_rho_y_yp": "相关系数 ρ(y,yp) 的绝对变化 |Δρ|",
    "eps4d_rel": "4D 相空间体积 √det Σ 的相对误差",
    "m5_halo_fraction_gap": "六投影 4σ 椭圆外计数占比差的平均（M5 晕占比）",
    "m6_hf_energy_ratio_gap": "六投影 σ_smooth=1 bin 高通能量比 |Δ| 的平均（M6 细结构）",
}


# --------------------------------------------------------------------------- IO
def load_tolerance(path: Path) -> dict[int, np.ndarray]:
    dump = json.loads(path.read_text(encoding="utf-8"))
    table = dump.get("dev_table", dump)
    return {int(t): np.asarray(v, dtype=np.float64) for t, v in table.items()}


def load_evals() -> list[dict]:
    evals = []
    for path in sorted(RESULTS.glob("*/test_eval*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["eval_path"] = str(path.relative_to(REPO))
        payload["dir"] = path.parent.name
        evals.append(payload)
    return evals


def obs_matrix(payload: dict) -> tuple[list[int], np.ndarray]:
    ids = [int(b["beam_id"]) for b in payload["per_beam"]]
    mat = np.stack([np.asarray(b["obs"], dtype=np.float64) for b in payload["per_beam"]])
    return ids, mat


def seed_median_matrix(payloads: list[dict], beam_ids: list[int]) -> np.ndarray:
    """与 `h8_deviation_matrix` 同口径：逐束、逐观测量对 run 取中位数。"""
    per_run = []
    for payload in payloads:
        by_beam = {int(b["beam_id"]): np.asarray(b["obs"], dtype=np.float64) for b in payload["per_beam"]}
        per_run.append(np.stack([by_beam[b] for b in beam_ids]))
    return np.median(np.stack(per_run), axis=0)


def tol_matrix_for(payloads: list[dict], beam_ids: list[int], dev_table, nodes) -> np.ndarray:
    """与 `_h8_from_disk` 同口径：逐束 N_eff 中位数 → 对数最近档 → 该档容差行。"""
    rows = []
    for beam_id in beam_ids:
        n_eff = float(np.median([float(p["_obs_by_beam"][beam_id]["n_eff"]) for p in payloads]))
        rows.append(tolerance_for(dev_table, nodes, n_eff))
    return np.stack(rows)


# ---------------------------------------------------------------------- 统计量
def folded_reference(n: int = 4_000_000, seed: int = 20261012) -> dict:
    """折叠正态 δ ~ N(b, σ) 的 (b/σ → MAD/median) 数值参考曲线。

    观测 |δ| 在纯零均值高斯误差下（b=0，半正态）的 ``r = MAD/median ≈ 0.592``——
    这是"方差主导"的参考水平；r 越低说明幅值越集中（共同成分越强）。
    β=b/σ=1 对应偏差份额 ``b²/(b²+σ²)=0.5``，其 r 即本实验"偏移主导"的判据线。
    """
    rng = np.random.default_rng(seed)
    base = np.abs(rng.standard_normal(n))
    curve = {}
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        m = beta + base
        med = float(np.median(m))
        mad = float(np.median(np.abs(m - med)))
        curve[str(beta)] = {"r": mad / med, "bias_share": beta ** 2 / (1.0 + beta ** 2)}
    return {"curve": curve, "r_pure_noise": curve["0.0"]["r"], "r_bias_half": curve["1.0"]["r"]}


def stats_block(dev: np.ndarray, tol: np.ndarray) -> dict:
    """单观测量的描述统计 + 通过率（tol 已按束给出）。"""
    x = np.asarray(dev, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    tol_med = float(np.median(tol))
    n = x.size
    return {
        "n": int(n),
        "median": med,
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if n > 1 else 0.0,
        "mad": mad,
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "tol": tol_med,
        "median_over_tol": med / tol_med if tol_med > 0 else math.nan,
        "mad_over_median": mad / med if med > 0 else math.nan,
        "pass_rate": float(np.mean(x <= tol)),
        "n_pass": int(np.sum(x <= tol)),
    }


def folded_fit(x: np.ndarray) -> dict:
    """折叠正态 MLE：δ ~ N(b, σ)（b = 跨束共同偏移），观测 |δ|。

    若各束误差符号一致，b 即符号偏移；返回 b、σ 与偏差占均方误差的份额。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    scale = float(np.median(x)) or 1.0

    def nll(logp):
        b, s = math.exp(logp[0]), math.exp(logp[1])
        f = (norm.pdf((x - b) / s) + norm.pdf((x + b) / s)) / s
        return float(-np.sum(np.log(np.maximum(f, 1e-300))))

    best = None
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p0 = np.log([max(frac * scale, 1e-12), max(1.4826 * float(np.median(np.abs(x - np.median(x)))), 1e-12)])
        res = minimize(nll, p0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    b, s = math.exp(best.x[0]), math.exp(best.x[1])
    mse = b * b + s * s
    return {
        "b_fit": b,
        "sigma_fit": s,
        "bias_share": b * b / mse if mse > 0 else math.nan,
        "nll": float(best.fun),
        "b_over_sigma": b / s if s > 0 else math.inf,
    }


def classify(ratio: float, r_bias_half: float = R_BIAS_HALF) -> str:
    """幅值集中度 → 偏差/方差判定（阈值 = 折叠正态参考曲线的两个锚点）。"""
    if not math.isfinite(ratio):
        return "n/a"
    if ratio < r_bias_half:
        return "偏移主导"
    if ratio < R_PURE_NOISE:
        return "混合"
    return "方差主导"


def oracle_calibration(dev: np.ndarray, tol: np.ndarray) -> dict:
    """完美校准共同偏移后的达标率（诊断上界）。

    - ``center``：校准残差 = |m − median(m)|（符号一致时即真实校准残差；符号混杂时为下界
      → 达标率为上界）。按 (束, 观测量) 的官方口径计达标率；
    - ``shift``：字面的"偏差减中位数"（等价于把容差线整体抬高中位数）——**不是校准**，
      只用于显示该操作的量级，随结论标注。
    """
    m = np.asarray(dev, dtype=np.float64)
    c = np.median(m, axis=0)
    resid = np.abs(m - c)
    shifted = m - c
    return {
        "centered_pass_rate": float(np.mean(resid <= tol)),
        "centered_pass_rate_per_obs": [float(np.mean(resid[:, i] <= tol[:, i])) for i in range(m.shape[1])],
        "shift_pass_rate_invalid": float(np.mean(shifted <= tol)),
        "center": c,
    }


# ------------------------------------------------------------------------ 主流程
def build_scopes(evals: list[dict]) -> dict[str, dict]:
    """各分析口径的 (run 集合, 束集合, 容差行)。"""
    for payload in evals:
        payload["_obs_by_beam"] = {int(b["beam_id"]): b for b in payload["per_beam"]}
    e1 = [p for p in evals if p["dir"].startswith("e1_") and int(p["n_low"]) == GATE_N_LOW]
    e1_s0 = [p for p in e1 if int(p["station"]) == GATE_STATION]
    scopes: dict[str, dict] = {}

    def add(name: str, payloads: list[dict], note: str) -> None:
        if not payloads:
            return
        common = sorted(set.intersection(*[{int(b["beam_id"]) for b in p["per_beam"]} for p in payloads]))
        scopes[name] = {
            "payloads": payloads,
            "beam_ids": common,
            "note": note,
            "variants": sorted({str(p["variant"]) for p in payloads}),
            "n_low": sorted({int(p["n_low"]) for p in payloads}),
            "station": sorted({int(p["station"]) for p in payloads}),
        }

    add("E1_joint_5seeds", sorted([p for p in e1_s0 if p["variant"] == "joint"], key=lambda p: p["seed"]),
        "H8 官方口径：获胜变体 joint 的 5 种子 × 站 0 × 1e4")
    add("E1_indep_5seeds", sorted([p for p in e1_s0 if p["variant"] == "independent"], key=lambda p: p["seed"]),
        "对照：未被路由的 independent 臂 5 种子 × 站 0 × 1e4（同口径，换臂）")
    add("E1_all10_seedmedian", sorted(e1_s0, key=lambda p: (p["variant"], p["seed"])),
        "主分析：E1 全部 10 run（joint+independent）× 站 0 × 1e4，种子中位数")
    add("E1_joint_station1", sorted([p for p in e1 if p["variant"] == "joint" and int(p["station"]) == 1],
                                    key=lambda p: p["seed"]),
        "次要：站 1（H7 侧），joint 5 种子")
    add("E2_joint_n3000", sorted([p for p in evals if p["dir"].startswith("e2_") and int(p["n_low"]) == 3000],
                                 key=lambda p: p["seed"]),
        "次要：E2 输入档 3e3，joint 5 种子")
    add("E2_joint_n100000", sorted([p for p in evals if p["dir"].startswith("e2_") and int(p["n_low"]) == 100_000],
                                   key=lambda p: p["seed"]),
        "次要：E2 输入档 1e5，joint 5 种子")
    return scopes


def main() -> int:
    dev_table = load_tolerance(RESULTS / "tolerance.json")
    gates = json.loads((RESULTS / "e1_gates.json").read_text(encoding="utf-8"))
    curve_nodes = tuple(int(n) for n in gates["curve"]["nodes"])
    evals = load_evals()
    scopes = build_scopes(evals)

    ref = folded_reference()
    summary: dict = {
        "reference_folded_normal": ref,
        "curve_nodes": list(curve_nodes),
        "thresholds": {"r_pure_noise": R_PURE_NOISE, "r_bias_half": R_BIAS_HALF,
                       "ratio_sensitivity": list(RATIO_SENSITIVITY)},
        "observables": list(OBSERVABLE_NAMES),
        "observable_meaning": OBS_MEANING,
        "scopes": {},
    }

    # ---- 官方 H8 复现（口径校验） ----
    official = scopes["E1_joint_5seeds"]
    dev_official = h8_deviation_matrix(official["payloads"], official["beam_ids"])
    tol_official = tol_matrix_for(official["payloads"], official["beam_ids"], dev_table, curve_nodes)
    h8_repro = evaluate_h8(dev_official, tol_official)
    tier_per_beam = [
        nearest_node(curve_nodes, float(np.median([float(p["_obs_by_beam"][b]["n_eff"]) for p in official["payloads"]])))
        for b in official["beam_ids"]
    ]
    summary["official_reproduction"] = {
        "variant": gates["winner_variant"],
        "routing_basis": gates["routing_basis"],
        "n_beams": len(official["beam_ids"]),
        "pass_rate_reproduced": h8_repro["pass_rate"],
        "pass_rate_landed": gates["H8"]["pass_rate"],
        "abs_diff": abs(h8_repro["pass_rate"] - gates["H8"]["pass_rate"]),
        "verdict_reproduced": h8_repro["verdict"],
        "verdict_landed": gates["H8"]["verdict"],
        "per_observable_n_pass_match": [
            int(o["n_pass"]) == int(g["n_pass"])
            for o, g in zip(h8_repro["per_observable"], gates["H8"]["per_observable"])
        ],
        "tolerance_tier_counts": {str(t): tier_per_beam.count(t) for t in sorted(set(tier_per_beam))},
        "tolerance_rows_used": sorted({int(t) for t in tier_per_beam}),
    }
    # 获胜变体独立复算（val 段 best_val_n_eff）
    val = {}
    for payload in evals:
        sp = RESULTS / payload["dir"] / "summary.json"
        if sp.is_file():
            s = json.loads(sp.read_text(encoding="utf-8"))
            val.setdefault(str(s["variant"]), {})[int(s["seed"])] = float(s["best_val_n_eff"])
    if set(val.get("joint", {})) == set(val.get("independent", {})):
        seeds = sorted(val["joint"])
        summary["official_reproduction"]["winner_recomputed_from_val"] = winner_variant(
            [val["joint"][s] for s in seeds], [val["independent"][s] for s in seeds]
        )
        summary["official_reproduction"]["val_n_eff"] = {k: {str(s): v for s, v in sorted(d.items())}
                                                         for k, d in sorted(val.items())}

    # ---- 逐口径 × 逐观测量统计 ----
    obs_rows: list[dict] = []
    oracle_rows: list[dict] = []
    per_scope: dict[str, dict] = {}
    for name, scope in scopes.items():
        ids = scope["beam_ids"]
        dev = seed_median_matrix(scope["payloads"], ids)
        tol = tol_matrix_for(scope["payloads"], ids, dev_table, curve_nodes)
        # 束级 N_eff 中位数 → 档位分布
        tiers = [nearest_node(curve_nodes, float(np.median([float(p["_obs_by_beam"][b]["n_eff"]) for p in scope["payloads"]])))
                 for b in ids]
        entry = {
            "note": scope["note"],
            "variants": scope["variants"],
            "n_low": scope["n_low"],
            "station": scope["station"],
            "n_runs": len(scope["payloads"]),
            "n_beams": len(ids),
            "tolerance_tier_counts": {str(t): tiers.count(t) for t in sorted(set(tiers))},
            "pass_rate_overall": float(np.mean(dev <= tol)),
            "per_observable": {},
        }
        cal = oracle_calibration(dev, tol)
        entry["oracle_centered_pass_rate_overall"] = cal["centered_pass_rate"]
        entry["oracle_shift_pass_rate_invalid"] = cal["shift_pass_rate_invalid"]
        n_offset = 0
        for i, obs_name in enumerate(OBSERVABLE_NAMES):
            st = stats_block(dev[:, i], tol[:, i])
            fit = folded_fit(dev[:, i])
            verdict = classify(st["mad_over_median"])
            if verdict == "偏移主导":
                n_offset += 1
            st.update({
                "scope": name, "index": i, "name": obs_name,
                "run": "-", "variant": "+".join(scope["variants"]),
                "seed": "-", "tier": sorted(set(tiers))[0] if len(set(tiers)) == 1 else "mixed",
                "aggregation": "seed_median_then_pass",
                "bias_variance": verdict,
                "oracle_centered_pass_rate": cal["centered_pass_rate_per_obs"][i],
                **fit,
            })
            obs_rows.append(dict(st))
            entry["per_observable"][obs_name] = st
            oracle_rows.append({
                "scope": name, "index": i, "name": obs_name,
                "baseline_pass_rate": st["pass_rate"],
                "oracle_centered_pass_rate": cal["centered_pass_rate_per_obs"][i],
                "oracle_shift_pass_rate_invalid": float(np.mean(dev[:, i] - cal["center"][i] <= tol[:, i])),
                "median": st["median"], "tol": st["tol"], "center_offset": cal["center"][i],
            })
        entry["n_offset_dominant"] = n_offset
        entry["n_bias_share_over_half"] = sum(
            1 for i in range(13) if entry["per_observable"][OBSERVABLE_NAMES[i]]["bias_share"] > 0.5)
        entry["ratio_sensitivity_counts"] = {
            str(t): sum(1 for i in range(13)
                        if entry["per_observable"][OBSERVABLE_NAMES[i]]["mad_over_median"] < t)
            for t in RATIO_SENSITIVITY
        }
        per_scope[name] = entry

    # ---- 逐 run 统计（10 个 E1 + 5 站1 + 10 E2） ----
    for payload in evals:
        ids, dev = obs_matrix(payload)
        tol = tol_matrix_for([payload], ids, dev_table, curve_nodes)
        tiers = [nearest_node(curve_nodes, float(payload["_obs_by_beam"][b]["n_eff"])) for b in ids]
        scope = ("E1" if payload["dir"].startswith("e1_") else "E2")
        for i, obs_name in enumerate(OBSERVABLE_NAMES):
            st = stats_block(dev[:, i], tol[:, i])
            st.update({
                "run": payload["dir"], "scope": scope, "variant": payload["variant"],
                "seed": payload["seed"], "station": payload["station"], "n_low": payload["n_low"],
                "tier": sorted(set(tiers))[0] if len(set(tiers)) == 1 else "mixed",
                "index": i, "name": obs_name, "aggregation": "per_run_beams",
                "bias_variance": classify(st["mad_over_median"]),
                **folded_fit(dev[:, i]),
            })
            obs_rows.append(st)

    # ---- 合并 (run, 束) 池化：E1 10 run × 150 束 = 1500 样本 ----
    # L1 层（单次推理）：校准实际作用的对象；L2 层（种子中位数）是 H8 的判定口径。
    # 两层的偏差/方差结构不同，故分别给基线通过率与 oracle 上界。
    pooled_rows: list[dict] = []
    e1_pool = [p for p in evals if p["dir"].startswith("e1_") and int(p["n_low"]) == GATE_N_LOW and int(p["station"]) == 0]
    tol_row = dev_table[curve_nodes[-1]]  # 全部束落在 1e5 档（见官方复现的档位分布）
    levels: dict = {"per_run_beam": {}, "seed_median_official": {}, "seed_median_all10": {}}
    pooled_pass, pooled_oracle_global, pooled_oracle_per_run = [], [], []
    pooled_base: dict[str, list[float]] = {v: [] for v in ("joint", "independent")}
    for i, obs_name in enumerate(OBSERVABLE_NAMES):
        vals = np.concatenate([np.asarray([b["obs"][i] for b in p["per_beam"]], dtype=np.float64) for p in e1_pool])
        tol_i = float(tol_row[i])
        st = stats_block(vals, np.full_like(vals, tol_i))
        st.update({
            "run": "E1_pooled_(run,beam)", "scope": "E1", "variant": "joint+independent",
            "seed": "-", "station": 0, "n_low": GATE_N_LOW, "tier": int(curve_nodes[-1]),
            "index": i, "name": obs_name, "aggregation": "pooled_run_beam",
            "bias_variance": classify(st["mad_over_median"]),
            **folded_fit(vals),
        })
        pooled_rows.append(st)
        # oracle 两档：单一常数/观测量（13 参数，全部从 test 拟合）与逐 run 常数（10×13 参数）
        c_global = float(np.median(vals))
        tol_all = np.full_like(vals, tol_i)
        resid_global = np.abs(vals - c_global)
        resid_per_run = np.concatenate([
            np.abs(np.asarray([b["obs"][i] for b in p["per_beam"]], dtype=np.float64)
                   - float(np.median([b["obs"][i] for b in p["per_beam"]])))
            for p in e1_pool])
        levels["per_run_beam"][obs_name] = {
            "baseline_pass_rate": st["pass_rate"],
            "oracle_centered_pass_rate": float(np.mean(resid_global <= tol_all)),
            "oracle_per_run_centered_pass_rate": float(np.mean(resid_per_run <= tol_all)),
            "median": st["median"], "tol": tol_i, "mad_over_median": st["mad_over_median"],
            "bias_share": st["bias_share"], "offset": c_global,
        }
        pooled_pass.append(st["pass_rate"])
        pooled_oracle_global.append(float(np.mean(resid_global <= tol_all)))
        pooled_oracle_per_run.append(float(np.mean(resid_per_run <= tol_all)))
        for variant in ("joint", "independent"):
            sub = [p for p in e1_pool if p["variant"] == variant]
            v = np.concatenate([np.asarray([b["obs"][i] for b in p["per_beam"]], dtype=np.float64) for p in sub])
            pooled_base[variant].append(float(np.mean(v <= tol_i)))
    for level, entry in (("seed_median_official", per_scope["E1_joint_5seeds"]),
                         ("seed_median_all10", per_scope["E1_all10_seedmedian"])):
        for obs_name in OBSERVABLE_NAMES:
            o = entry["per_observable"][obs_name]
            levels[level][obs_name] = {
                "baseline_pass_rate": o["pass_rate"],
                "oracle_centered_pass_rate": o["oracle_centered_pass_rate"],
                "median": o["median"], "tol": o["tol"],
                "mad_over_median": o["mad_over_median"], "bias_share": o["bias_share"],
            }
    summary["levels"] = levels
    # ---- 种子中位数的选档行为：逐 run 观测量向量与种子中位数向量的相关 ----
    seed_selection = {}
    for obs_i, obs_name in enumerate(OBSERVABLE_NAMES):
        mat = np.stack([np.asarray([b["obs"][obs_i] for b in p["per_beam"]], dtype=np.float64)
                        for p in official["payloads"]])          # (n_seeds, n_beams)
        med = np.median(mat, axis=0)
        cors = [float(np.corrcoef(mat[k], med)[0, 1]) for k in range(mat.shape[0])]
        seed_selection[obs_name] = {
            "seeds": [int(p["seed"]) for p in official["payloads"]],
            "per_run_beam_median": [float(np.median(mat[k])) for k in range(mat.shape[0])],
            "corr_with_seed_median": cors,
            "argmax_seed": int(official["payloads"][int(np.argmax(cors))]["seed"]),
            "seed_median": float(np.median(med)),
            "level_ratio_max_over_min": float(max(np.median(mat, axis=1)) / min(np.median(mat, axis=1))),
        }
    summary["seed_selection"] = seed_selection
    summary["per_run_beam_overall"] = {
        "n_pairs": len(e1_pool) * len(e1_pool[0]["per_beam"]),
        "baseline_pass_rate": float(np.mean(pooled_pass)),
        "baseline_pass_rate_by_variant": {k: float(np.mean(v)) for k, v in pooled_base.items()},
        "oracle_centered_pass_rate": float(np.mean(pooled_oracle_global)),
        "oracle_per_run_centered_pass_rate": float(np.mean(pooled_oracle_per_run)),
        "note": "L1 层：单次推理的 (run, 束) 对等权；与 H8 的种子中位数口径不是同一统计量",
    }

    # ---- 偏移稳定性：逐 run 中位数（10 个 E1 run）× 变体对照 ----
    # 共同成分若为方案/数据接口的系统性质，则应跨种子、跨变体复现；若为训练随机性，
    # 则逐 run 中位数应散布。判据读法：逐 run 中位数的极差 / 容差线。
    e1_runs = [p for p in evals if p["dir"].startswith("e1_") and int(p["n_low"]) == GATE_N_LOW
               and int(p["station"]) == GATE_STATION]
    per_run_medians: dict[str, dict[str, list[float]]] = {}
    for payload in e1_runs:
        _, dev = obs_matrix(payload)
        for i, obs_name in enumerate(OBSERVABLE_NAMES):
            per_run_medians.setdefault(obs_name, {"joint": [], "independent": []})[payload["variant"]].append(
                float(np.median(dev[:, i])))
    stability = {}
    variant_split = {}
    for i, obs_name in enumerate(OBSERVABLE_NAMES):
        tol_i = float(tol_row[i])
        rec = per_run_medians[obs_name]
        all_med = np.asarray(rec["joint"] + rec["independent"])
        stability[obs_name] = {
            "per_run_medians_joint": rec["joint"],
            "per_run_medians_independent": rec["independent"],
            "median": float(np.median(all_med)),
            "spread": float(np.max(all_med) - np.min(all_med)),
            "spread_over_tol": float((np.max(all_med) - np.min(all_med)) / tol_i),
            "max_over_min": float(np.max(all_med) / np.min(all_med)),
        }
        variant_split[obs_name] = {
            "joint_median": float(np.median(rec["joint"])),
            "independent_median": float(np.median(rec["independent"])),
            "joint_over_independent": float(np.median(rec["joint"]) / np.median(rec["independent"])),
            "tol": tol_i,
        }
    summary["offset_stability_across_runs"] = stability
    summary["variant_split"] = variant_split

    # ---- 判定（预注册） ----
    def judgement(entry: dict) -> dict:
        n_off = entry["n_offset_dominant"]
        rate = entry["oracle_centered_pass_rate_overall"]
        if n_off >= 7 and rate > 0.5:
            verdict = "偏移主导可救（便宜校准可将 H8 达标率抬过 50%）"
        elif n_off < 7 and rate < 0.5:
            verdict = "方差主导需真改进（校准上界仍 < 50%）"
        else:
            verdict = "混合（偏移项数或校准上界仅一侧达线，见明细）"
        return {"n_offset_dominant": n_off, "oracle_centered_pass_rate": rate,
                "baseline_pass_rate": entry["pass_rate_overall"], "verdict": verdict}

    summary["scopes"] = per_scope
    summary["judgement"] = {
        "primary_scope": "E1_all10_seedmedian",
        "primary": judgement(per_scope["E1_all10_seedmedian"]),
        "official_scope": judgement(per_scope["E1_joint_5seeds"]),
    }
    # 主分析口径的 13 项逐项判定表（回报用）
    summary["primary_table"] = [
        {k: per_scope["E1_all10_seedmedian"]["per_observable"][n][k]
         for k in ("index", "name", "median", "tol", "median_over_tol", "mad", "mad_over_median",
                   "bias_share", "b_fit", "sigma_fit", "pass_rate", "bias_variance",
                   "oracle_centered_pass_rate")}
        for n in OBSERVABLE_NAMES
    ]

    write_csv(HERE / "exp4_observable_table.csv", obs_rows)
    write_csv(HERE / "exp4_oracle_calibration.csv", oracle_rows)
    write_csv(HERE / "exp4_pooled_run_beam.csv", pooled_rows)
    (HERE / "exp4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=float), encoding="utf-8")

    print(json.dumps({
        "official_reproduction": summary["official_reproduction"],
        "judgement": summary["judgement"],
        "reference_folded_normal": ref,
    }, ensure_ascii=False, indent=2, default=float))
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []  # 并集（首见顺序），混合口径行不得丢列
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


if __name__ == "__main__":
    raise SystemExit(main())
