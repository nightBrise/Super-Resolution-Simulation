"""评估 CLI：``python -m src.evaluation.evaluate --config <path> --split <...>``。

接口按 60 [S15] 15.7：评估测试集输出 ``metrics.csv``（长表，80 [S8]
列名规范）+ ``summary.json``（字段规范）；评估默认读
``checkpoints/best_val.ckpt``（60 [S12] C3），可用 ``--checkpoint`` 覆盖；
三方案由 ``--config`` 指定（config 内含 ``scheme`` 字段）。启动前先跑
``scripts/check_env.py``（仅警告）。

单次调用评估一个方案（config 指定）；多次调用（不同方案）写入同一输出
目录时按 ``(sample_id, split, scheme)`` 合并行，summary.json 依据已合并
的行计算：指标均值/标准差、先验增益（配对差 + Wilcoxon + bootstrap CI +
Holm + 三分类，70 [S7]）、一票否决（70 [S4]）与 L_up 退化基线（G1(b)
依赖）。预注册三分类主统计仅在 ``test_id`` 上执行；``test_pb`` 同法计算
但标注补充/探索性（70 [S7.1] C7，不与 test_id 合并）。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    PRIMARY_METRIC_COL,
    SECONDARY_METRIC_COL,
    OVER_OVERSHOOT,
    OVER_SMOOTH,
    bootstrap_ci,
    c_high_component_breakdown,
    c_high_mask_from_hp,
    dog_sigma_outer,
    e_high_doG,
    evaluate_sample,
    hallucination_flag,
    high_freq_energy_ratio,
    high_pass_fft,
    holm_correction,
    normalize_density,
    overshoot_smooth_class,
    prior_gain_stats,
    prior_leak_index,
    veto_verdict,
)
from src.generators.f_beam import render_level1_density
from src.models.schemes import SchemeC, build_scheme_model, build_scheme_model_from_checkpoint, forward_scheme
from src.training.loss import F_C
from src.utils.checkpoint import load_checkpoint
from src.utils.config_utils import (
    load_config,
    resolve_data_version,
    resolve_device,
    run_output_dir,
)
from src.utils.h5data import C_PRIOR_KEYS, H5Dataset, collate_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: metrics.csv 列（80 [S8] 列名规范 + split 列 + 过冲分类内部列）。
#: ``ssim_vis``/``cne`` 为感知一致性指标（70 [S3] C3，2026-08-26 新增）；
#: ``ch_in_mask``/``b_in_mask``/``pi_leak`` 为掩膜成分分解与先验泄漏指数
#: （70 [S7.1] C2，样本级诊断量，旧数据无 H_neg_ch 时记 nan）。
METRIC_COLUMNS: list[str] = [
    "sample_id",
    "split",
    "scheme",
    "a_3",
    "gamma",
    "b_1",
    "psnr",
    "mae",
    "mse",
    "ssim",
    "ssim_vis",
    "cne",
    "e_eps_z",
    "e_I_peak",
    "e_sigma_z",
    "e_sigma_delta",
    "e_h_eff",
    "e_high_doG",
    "R_E",
    "e_high_mask",
    "e_peak",
    "e_profile_I",
    "e_profile_S",
    "Q_hat",
    "R_E_class",
    "F_i",
    "e_I_peak_signed",
    "n_peaks",
    "peak_positions",
    "ch_in_mask",
    "b_in_mask",
    "pi_leak",
]

#: summary 中计算均值/标准差的数值指标列。
SUMMARY_METRICS = [
    "psnr",
    "mae",
    "mse",
    "ssim",
    "ssim_vis",
    "cne",
    "e_eps_z",
    "e_I_peak",
    "e_sigma_z",
    "e_sigma_delta",
    "e_h_eff",
    "e_high_doG",
    "R_E",
    "e_high_mask",
    "e_peak",
    "e_profile_I",
    "e_profile_S",
]

#: 先验增益必报的三组对比（70 [S6] C2）。
GAIN_PAIRS = (("A", "B"), ("A", "C"), ("B", "C"))
GAIN_KEYS = {"A_B": "M_A_minus_M_B", "A_C": "M_A_minus_M_C", "B_C": "M_B_minus_M_C"}


def _run_check_env() -> None:
    """运行 scripts/check_env.py（60 [S15] 15.5/15.7：仅警告，不阻塞）。"""
    script = PROJECT_ROOT / "scripts" / "check_env.py"
    if not script.exists():
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,
        )
        if proc.returncode != 0:
            print("[check_env] 校验未通过（仅警告）：\n" + (proc.stdout + proc.stderr), file=sys.stderr)
    except Exception as exc:  # pragma: no cover - 环境异常仅警告
        print(f"[check_env] 无法运行：{exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def load_model(config: dict, checkpoint_path: str | Path, device: str):
    """按 checkpoint 构造模型并加载权重与方案 C 标准化统计量。

    优先用 checkpoint 记录的模型标识与网络配置（训练产物自洽）；
    c_prior 统计量来自训练集（60 [S5] C3，评估不重算，★ 无泄漏）。

    双空间契约加载断言（60 [S15] 项 4）：checkpoint 持久化的 ``work_scale``
    与当前运行 config 的 ``network.work_scale`` SHALL 一致；不一致或旧版
    checkpoint 未持久化时 SHALL 报错终止并提示 R2 流程（80 [S12]）。
    """
    ckpt = load_checkpoint(checkpoint_path)
    config_ws = config.get("network", {}).get("work_scale")
    if config_ws is None:
        raise ValueError(
            "config 缺 network.work_scale（60 [S15] 双空间契约：加载前须显式"
            "配置工作尺度 S）"
        )
    ckpt_ws = ckpt.get("work_scale")
    if ckpt_ws is None:
        raise ValueError(
            "checkpoint 缺少 work_scale（60 [S15] 双空间契约：旧版 checkpoint "
            "未持久化尺度参数，需按 80 [S12] R2 流程重训后重新评估）"
        )
    if float(ckpt_ws) != float(config_ws):
        raise ValueError(
            f"checkpoint work_scale={float(ckpt_ws)} 与 config "
            f"network.work_scale={float(config_ws)} 不一致（60 [S15] 加载一致性"
            f"断言；按 80 [S12] R2 流程重训后重新评估）"
        )
    if ckpt.get("model_class"):
        model = build_scheme_model_from_checkpoint(ckpt)
    else:
        model = build_scheme_model(config)
    model.load_state_dict(ckpt["model_state"])
    if isinstance(model, SchemeC):
        if "c_prior_mu" not in ckpt or "c_prior_sigma" not in ckpt:
            raise ValueError("方案 C checkpoint 缺少 c_prior 标准化统计量（60 [S5] C3）")
        model.set_c_prior_stats(ckpt["c_prior_mu"], ckpt["c_prior_sigma"])
    model = model.to(device)
    return model, ckpt


def infer_predictions(
    model,
    dataset: H5Dataset,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """推理整个数据集，返回 ``(N, 256, 256)`` 的 Ĥ（Σ=1 空间，已 ÷S 还原）。

    模型输出为工作尺度（Softplus(S·Base+R)，50 [S12] C5），评估/推理还原为
    ``Ĥ = Ĥ_work / S`` 以便与总强度归一化真值直接比较。
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_samples)
    scale = float(getattr(model, "work_scale", 65536.0))
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            H_hat = forward_scheme(model, batch, device)
            preds.append(H_hat.cpu().numpy()[:, 0] / scale)
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# metrics.csv（长表，80 [S8]）
# ---------------------------------------------------------------------------
def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = METRIC_COLUMNS + [c for c in rows[0] if c not in METRIC_COLUMNS] if rows else METRIC_COLUMNS
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _merge_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """按 ``(sample_id, split, scheme)`` 合并（新行覆盖旧行）。"""
    index = {(r["sample_id"], r["split"], r["scheme"]): i for i, r in enumerate(existing)}
    merged = list(existing)
    for row in new_rows:
        key = (row["sample_id"], row["split"], row["scheme"])
        if key in index:
            merged[index[key]] = row
        else:
            index[key] = len(merged)
            merged.append(row)
    return merged


def _render_neg_b(sample: dict) -> np.ndarray:
    """β 清零版真值 H_neg_b（与 H 同 σ_smooth,H 渲染，70 [S7.1] C2 差分基准）。

    数据集只缓存 H_neg_ch（a₃=γ=b₁=0）；β 清零版按归档脚本
    （``scripts_tmp/g2_c_high_leak.py``）口径由内容参数重渲染得到——
    ``render_level1_density(..., sigma_smooth=None)`` 即取 20 [S3] C4 的
    σ_smooth,H = 0.125×w_fine 单样本回退值（2026-08-26 P0 修订口径）。
    """
    c: dict[str, float] = dict(
        zip(C_PRIOR_KEYS, (float(v) for v in sample["c_prior_raw"].numpy()))
    )
    c["A"] = 1.0
    c.update(
        {
            key: float(v)
            for key, v in zip(("a3", "gamma", "b1"), sample["c_high"].numpy())
        }
    )
    c["beta"] = 0.0
    return render_level1_density(c)


def build_rows(
    dataset: H5Dataset,
    preds: np.ndarray,
    scheme: str,
    split: str,
    sigma_outer: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """构造单方案逐样本指标行 + 该划分的 L_up 退化基线汇总（G1(b) 依赖）。"""
    rows: list[dict[str, Any]] = []
    baseline_e_high: list[float] = []
    baseline_r_e: list[float] = []
    baseline_emask: list[float] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        H = sample["H"].numpy()[0]
        L_up = sample["L_up"].numpy()[0]
        H_hat = preds[i]
        H_norm = normalize_density(H)
        L_up_norm = normalize_density(L_up)
        baseline_e_high.append(e_high_doG(L_up_norm, H_norm, sigma_outer))
        baseline_r_e.append(high_freq_energy_ratio(L_up_norm, H_norm))
        cmask = c_high_mask_from_hp(high_pass_fft(H_norm))
        baseline_emask.append(e_high_mask(L_up_norm, H_norm, cmask))
        met = evaluate_sample(
            H_hat, H, sample["m"], sigma_outer, e_high_baseline=baseline_e_high[-1]
        )
        c_high = sample["c_high"].numpy()
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "split": split,
            "scheme": scheme,
            "a_3": float(c_high[0]),
            "gamma": float(c_high[1]),
            "b_1": float(c_high[2]),
        }
        for key in METRIC_COLUMNS:
            if key in met:
                row[key] = met[key]

        # ---- G1(b) 零学习基线（L_up 直出，70 [S7] 后备分支第二部分）：逐样本
        # L_up 在预注册主指标 ε_high^mask 上的误差，供配对差 bootstrap CI
        # （A vs L_up 正增益 + CI 下界>0）；可选列，80 [S8] 列名规范不破坏。
        row["e_high_mask_lup"] = baseline_emask[-1]

        # ---- 掩膜成分分解 + 先验泄漏指数（70 [S7.1] C2，2026-08-26 P0 修订）
        # 样本级诊断量（只依赖真值 H 与先验 P2，与方案无关）；H_neg_ch 由
        # 数据集提供，旧版本数据无该字段时记 nan 并跳过（C 类缺省行为）。
        row["ch_in_mask"] = float("nan")
        row["b_in_mask"] = float("nan")
        row["pi_leak"] = float("nan")
        if "H_neg_ch" in sample:
            H_neg_ch = normalize_density(sample["H_neg_ch"].numpy()[0])
            P2_norm = normalize_density(sample["P2"].numpy()[0])
            breakdown = c_high_component_breakdown(
                H_norm, H_neg_ch, cmask, H_neg_b=_render_neg_b(sample)
            )
            row["ch_in_mask"] = breakdown["ch_in"]
            row["b_in_mask"] = breakdown["b_in"]
            row["pi_leak"] = prior_leak_index(P2_norm, H_neg_ch, H_norm, cmask)
        rows.append(row)
    baseline = {
        "e_high_doG_mean": float(np.mean(baseline_e_high)),
        "R_E_mean": float(np.mean(baseline_r_e)),
        "e_high_mask_mean": float(np.mean(baseline_emask)),  # G1(b) 主指标基线
    }
    return rows, baseline


def compute_f_i(rows: list[dict[str, Any]], tau: float = 0.05) -> None:
    """为 B/C 行计算样本级幻觉标志 F_i（70 [S4]，需同划分 A 行；原地更新）。

    ``F_i`` 只对先验方案（B/C）计算，方案 A 行为空（70 [S4] C4）。
    """
    a_by_sample = {r["sample_id"]: r for r in rows if r["scheme"] == "A"}
    for row in rows:
        row["F_i"] = None
        if row["scheme"] == "A":
            continue
        ref = a_by_sample.get(row["sample_id"])
        if ref is None:
            continue
        row["F_i"] = hallucination_flag(
            float(row["psnr"]), float(ref["psnr"]),
            float(row["e_eps_z"]), float(ref["e_eps_z"]),
            float(row["e_I_peak"]), float(ref["e_I_peak"]),
            tau=tau,
        )


# ---------------------------------------------------------------------------
# summary.json（80 [S8] 字段规范）
# ---------------------------------------------------------------------------
def _metric_stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def build_summary(
    rows: list[dict[str, Any]],
    config: dict,
    split: str,
    baseline: dict[str, float] | None,
) -> dict[str, Any]:
    """由合并后的 metrics.csv 行构造 summary.json（80 [S8] 字段规范）。"""
    schemes = sorted({r["scheme"] for r in rows})
    seed = int(np.random.SeedSequence([int(config["master_seed"]), 0xE9A1]).generate_state(1, dtype=np.uint32)[0])  # "eval" 分支

    # -- metrics：各方案各指标 mean/std -------------------------------------
    metrics: dict[str, Any] = {}
    for scheme in schemes:
        sub = [r for r in rows if r["scheme"] == scheme]
        metrics[scheme] = {}
        for col in SUMMARY_METRICS:
            vals = []
            for r in sub:
                try:
                    vals.append(float(r.get(col)))
                except (TypeError, ValueError):
                    continue
            if vals:
                metrics[scheme][col] = _metric_stats(vals)

    # -- 先验增益（70 [S6][S7.2]；Holm 校正族 = 方案对 × 测试集，仅主+次）---
    primary = PRIMARY_METRIC_COL
    secondary = SECONDARY_METRIC_COL
    prior_gain: dict[str, Any] = {}
    three_class: dict[str, Any] = {}
    for a, b in GAIN_PAIRS:
        if a not in schemes or b not in schemes:
            continue
        pair_key = GAIN_KEYS[f"{a}_{b}"]
        a_rows = {r["sample_id"]: r for r in rows if r["scheme"] == a}
        b_rows = {r["sample_id"]: r for r in rows if r["scheme"] == b}
        common = sorted(set(a_rows) & set(b_rows))
        if len(common) < 2:
            continue
        pair_gain: dict[str, Any] = {}
        p_values: list[float] = []
        p_metrics: list[str] = []
        for metric in (primary, secondary):
            d = np.array([float(a_rows[s][metric]) - float(b_rows[s][metric]) for s in common])
            stat = prior_gain_stats(d, seed=seed)
            p_values.append(float(stat["wilcoxon_p"]))
            p_metrics.append(metric)
            pair_gain[metric] = stat
        # Holm 校正（70 [S7.1] C7b：仅主+次指标入族）
        adj = holm_correction(np.asarray(p_values))
        for metric, p_adj in zip(p_metrics, adj):
            pair_gain[metric]["holm_p"] = float(p_adj)
        prior_gain[pair_key] = pair_gain
        primary_stat = pair_gain[primary]
        three_class[pair_key] = {
            "metric": primary,
            "verdict": primary_stat["verdict"],
            "ci95": primary_stat["ci95"],
            "wilcoxon_p": primary_stat["wilcoxon_p"],
            "holm_p": primary_stat["holm_p"],
        }

    # -- 一票否决（70 [S4]：B/C vs A，两层判据 + 过冲/平滑分类）-------------
    one_veto: dict[str, Any] = {}
    if "A" in schemes:
        a_rows = {r["sample_id"]: r for r in rows if r["scheme"] == "A"}
        for scheme in ("B", "C"):
            if scheme not in schemes:
                continue
            x_rows = {r["sample_id"]: r for r in rows if r["scheme"] == scheme}
            common = sorted(set(a_rows) & set(x_rows))
            if len(common) < 2:
                continue
            fi = [1 if x_rows[s].get("F_i") == 1 else 0 for s in common]
            d_eps = np.array([float(x_rows[s]["e_eps_z"]) - float(a_rows[s]["e_eps_z"]) for s in common])
            d_ipeak = np.array([float(x_rows[s]["e_I_peak"]) - float(a_rows[s]["e_I_peak"]) for s in common])
            ci_eps = bootstrap_ci(d_eps, seed=seed)
            ci_ipeak = bootstrap_ci(d_ipeak, seed=seed)
            gain_eps = float(-d_eps.mean())
            gain_ipeak = float(-d_ipeak.mean())
            p_f = float(np.mean(fi)) if fi else 0.0
            verdict = veto_verdict(
                p_f, ci_eps[0], ci_ipeak[0], gain_eps, gain_ipeak,
                trigger_rate=float(config.get("evaluation", {}).get("trigger_rate", 0.20)),
            )
            triggered = [s for s in common if x_rows[s].get("F_i") == 1]
            classes = [overshoot_smooth_class(float(x_rows[s]["e_I_peak_signed"])) for s in triggered]
            one_veto[scheme] = {
                "P_F": p_f,
                "n_triggered": len(triggered),
                "ci_lower_eps_z": ci_eps[0],
                "ci_lower_ipeak": ci_ipeak[0],
                "gain_eps_z": gain_eps,
                "gain_ipeak": gain_ipeak,
                "verdict": verdict,
                "overshoot": classes.count(OVER_OVERSHOOT),
                "smooth": classes.count(OVER_SMOOTH),
            }

    # -- 掩膜成分分解 + 先验泄漏指数（70 [S7.1] C2；样本级诊断量）---------
    mask_composition: dict[str, Any] = {}
    for col in ("ch_in_mask", "b_in_mask", "pi_leak"):
        vals = []
        for r in rows:
            try:
                v = float(r.get(col))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        if vals:
            mask_composition[col] = {
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "n": len(vals),
            }
        else:
            mask_composition[col] = {"median": None, "mean": None, "n": 0}
    ch_med = mask_composition["ch_in_mask"].get("median")
    b_med = mask_composition["b_in_mask"].get("median")
    pi_med = mask_composition["pi_leak"].get("median")
    mask_composition["b_in_exceeds_ch_in_x1.5"] = bool(
        ch_med is not None and b_med is not None and b_med > 1.5 * ch_med
    )
    mask_composition["pi_leak_gt_0.5"] = bool(pi_med is not None and pi_med > 0.5)
    mask_composition["notes"] = [
        "ch_in/b_in 为掩膜内 c_high/β 差分能量敏感度读数（70 [S7.1] C2，"
        "口径同 scripts_tmp/g2_c_high_leak.py 归档版）",
        "任一方案 b_in > 1.5×ch_in 时 MUST 附先验能量级泄漏分析；"
        "Π_leak > 0.5 视为先验在主指标区域结构性占优，final_report MUST 显式标注",
        "旧版本数据集无 H_neg_ch 字段时三字段记 nan（n=0）并跳过",
    ]

    # -- R_E 守卫（80 [S4] C3b，2026-08-26 P0 修订第 13 项）-----------------
    r_e_max = float(config.get("evaluation", {}).get("r_e_max", 10.0))
    re_guard: dict[str, Any] = {"r_e_max": r_e_max, "per_scheme": {}}
    for scheme in schemes:
        vals = []
        for r in rows:
            if r["scheme"] != scheme:
                continue
            try:
                v = float(r.get("R_E"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        if vals:
            median = float(np.median(vals))
            maxv = float(np.max(vals))
            passed = median <= r_e_max
            re_guard["per_scheme"][scheme] = {
                "median": median,
                "max": maxv,
                "passed": passed,
                "label": "normal" if passed else "sharpening_artifact",
            }
    re_guard["any_sharpening_artifact"] = any(
        not v["passed"] for v in re_guard["per_scheme"].values()
    )
    re_guard["note"] = (
        "R_E 超限不否决实验（80 [S4] C3b）：超限方案标注'锐化伪影档'，"
        "登记 99 并发起二级咨询评估是否影响主指标解读"
    )

    summary = {
        "version": {
            "code_version": str(config.get("code_version", "")),
            "data_version": resolve_data_version(config),
            "spec_version": str(config.get("spec_version", "")),
        },
        "split": split,
        "note": (
            "primary stats on test_id; test_pb supplementary/exploratory, "
            "not merged, not in Holm family (70 [S7.1] C7)"
            if split != "test_id"
            else "pre-registered primary stats (70 [S7.1] C7)"
        ),
        "metrics": metrics,
        "baseline": {"L_up": baseline} if baseline else {},
        "prior_gain": prior_gain,
        "three_class": three_class,
        "one_veto": one_veto,
        "mask_composition": mask_composition,
        "re_guard": re_guard,
    }
    return summary


def run_evaluate(
    config: dict,
    split: str,
    out_dir: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 16,
    indices: list[int] | None = None,
) -> dict[str, Any]:
    """评估一个方案（config 指定）并写 metrics.csv + summary.json，返回 summary。

    训练/评估/推理共用的评估入口（smoke 测试与 CLI 同路径）。
    ``indices`` 只用于测试的样本子集钩子（生产为 None）。
    """
    out_dir = Path(out_dir)
    scheme = str(config["scheme"]).upper()
    dataset_dir = PROJECT_ROOT / "data" / str(config["dataset"]["version"])
    # 60 [S14]：EXP-03/04 测试工件文件名为 test_exp03.h5 / test_exp04.h5
    # （--split exp03/exp04 映射到 test_exp03/test_exp04，T3 对齐）
    filename = {"exp03": "test_exp03", "exp04": "test_exp04",
                "exp07": "test_exp07", "exp08": "test_exp08"}.get(split, split)
    split_file = dataset_dir / f"{filename}.h5"
    if not split_file.exists():
        raise FileNotFoundError(f"数据集不存在：{split_file}")

    device = resolve_device(config)
    model, _ = load_model(config, checkpoint_path, device)
    dataset = H5Dataset(split_file, split)
    if indices is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(indices))  # type: ignore[assignment]
    preds = infer_predictions(model, dataset, device, batch_size=batch_size)

    dog = config.get("evaluation", {}).get("dog", {})
    sigma_outer = float(dog.get("sigma_outer") or 0.0)
    if sigma_outer <= 0.0:
        sigma_outer = dog_sigma_outer(F_C)

    rows, baseline = build_rows(dataset, preds, scheme, split, sigma_outer)
    merged = _merge_rows(_read_metrics(out_dir / "metrics.csv"), rows)
    compute_f_i(merged, tau=float(config.get("evaluation", {}).get("tau", 0.05)))
    _write_metrics(out_dir / "metrics.csv", merged)

    summary = build_summary(merged, config, split, baseline)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI（60 [S15] 15.7）
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluate",
        description="评估测试集（输出 metrics.csv + summary.json，80 [S8]）",
    )
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument(
        "--split",
        required=True,
        choices=["test_id", "test_pb", "test_ood", "exp03", "exp04", "exp07", "exp08"],
        help="测试划分（HDF5 文件名：<split>.h5）",
    )
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径（默认 best_val.ckpt）")
    parser.add_argument("--out", default=None, help="输出目录（默认按 15.2 命名）")
    parser.add_argument("--batch-size", type=int, default=16, help="推理 batch size")
    args = parser.parse_args(argv)

    _run_check_env()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config 不存在：{config_path}", file=sys.stderr)
        return 1
    config = load_config(config_path)
    scheme = str(config["scheme"]).upper()
    out_dir = Path(args.out) if args.out else run_output_dir(config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoints" / "best_val.ckpt"
    if not checkpoint_path.exists():
        print(f"checkpoint 不存在：{checkpoint_path}（评估默认读 best_val.ckpt，60 [S12] C3）", file=sys.stderr)
        return 1

    try:
        summary = run_evaluate(config, args.split, out_dir, checkpoint_path, batch_size=args.batch_size)
    except Exception as exc:
        print(f"评估失败：{exc}", file=sys.stderr)
        return 1

    n_samples = 0
    scheme_metrics = summary.get("metrics", {}).get(scheme, {})
    psnr_stats = scheme_metrics.get("psnr")
    if isinstance(psnr_stats, dict) and isinstance(psnr_stats.get("mean"), (int, float)):
        n_samples = int(np.isfinite(psnr_stats["mean"])) or 0
    print(
        f"[evaluate] scheme={scheme} split={args.split} "
        f"metrics.csv={out_dir / 'metrics.csv'} summary.json={out_dir / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
