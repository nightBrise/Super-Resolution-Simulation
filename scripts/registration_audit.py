"""预注册对账工具（90 [S5] N8，M6 验收用）。

扫描 ``results/`` 下各实验 run 的 ``config.yaml`` 与 ``summary.json``，核对预注册判据与
阈值（判据符号表，90 [S6] V1）在各载体间一致；扫描 ``99_change_log.md`` 中触及判据
模块（00/30/40/70/80/90）的批准批次输出对账表；核对 ``final_report.md`` 预注册骨架
标记存在。任何不一致（FAIL）视为验收失败（90 [S5] N8）；产物缺失记为 WARN（M6
前属预期，M6 时须齐全）。

用法：:

    python scripts/registration_audit.py [--results-dir <results 根>]
                                         [--99 <99_change_log.md>]
                                         [--report <final_report.md>]
                                         [--out <对账表.md>]

退出码：0 = 无 FAIL；1 = 存在 FAIL（预注册值不一致）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 判据符号表（90 [S6] V1 预注册，2026-08-26）：config.yaml 可机检字段。
#: 键为 yaml 点路径，值为 (期望值, 标签)；期望值为 str 时全等比较，数字按数值比较。
PRE_REGISTERED_CONFIG: dict[str, tuple[object, str]] = {
    "evaluation.tau": (0.05, "τ=5%"),
    "evaluation.trigger_rate": (0.2, "触发率 20%"),
    "evaluation.primary_metric": ("ε_high^mask", "主指标 ε_high^mask"),
    "evaluation.rho_threshold": (0.1, "高频存活比 ρ≥0.1"),
    "evaluation.r_e_max": (10.0, "R_E^max=10（锐化伪影守卫）"),
    "evaluation.ood_degradation_threshold": (0.2, "OOD 退化阈值 ≤20%"),
    "evaluation.ci_width_min": (0.05, "最小可信 CI 宽度 5%"),
    "evaluation.mde": (0.05, "最小可检出效应量 MDE 5%"),
    "evaluation.max_expansion_factor": (2.0, "扩集上限 2.0"),
    "degradation.snr_hf_threshold": (0.1, "SNR_hf<0.1"),
    "network.work_scale": (65536.0, "工作尺度 S=N²=65536"),
    "training.lambda_spec": (1.0, "λ=1.0（冻结）"),
    "calibration.sigma_K": (11.0, "标定采用值 σ_K=11.0"),
    "calibration.sigma_n": (0.000122, "标定采用值 σ_n=1.22e-4"),
}

CODE_VERSION_RE = re.compile(r"^[0-9a-f]{40}$")
SPEC_VERSION_RE = re.compile(r"^v1\.0\+20\d{2}-\d{2}-\d{2}$")

THREE_CLASS_VALUES = {"significant_positive", "equivalent", "significant_negative"}
ONE_VETO_VALUES = {"no_veto", "veto_B", "veto_C", "noise_fluctuation"}

#: 99 变更日志中触及预注册判据的模块（对账表只保留这些行）。
CRITERIA_MODULES = re.compile(r"\b(00|30|40|70|80|90)\b")

#: final_report.md 预注册骨架必含标记（90 [S6] V1 五要素）。
REPORT_MARKERS = ("判据符号表", "分支树", "预注册")


def _get_path(data: dict, dotted: str) -> object:
    """按点路径取嵌套字典值；任一层缺失返回 None。"""
    cur: object = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _values_equal(expected: object, actual: object) -> bool:
    """数值按数值比较（容忍 0.05 vs 0.05000000000000001），字符串全等。"""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 1e-9
    if isinstance(expected, str):
        return expected == actual
    return expected == actual


def audit_config(config_path: Path) -> list[dict]:
    """核对单个 config.yaml 的预注册字段；返回 [(severity, message)]。"""
    findings: list[dict] = []
    try:
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        return [{"severity": "FAIL", "message": f"config 无法解析：{exc}"}]

    for dotted, (expected, label) in PRE_REGISTERED_CONFIG.items():
        actual = _get_path(cfg, dotted)
        if actual is None:
            findings.append({"severity": "WARN", "message": f"{dotted}（{label}）缺失"})
        elif not _values_equal(expected, actual):
            findings.append(
                {
                    "severity": "FAIL",
                    "message": f"{dotted}（{label}）：期望 {expected}，实际 {actual}",
                }
            )

    for dotted, regex, label in (
        ("code_version", CODE_VERSION_RE, "code_version 完整 40 位 git hash"),
        ("spec_version", SPEC_VERSION_RE, "spec_version v1.0+YYYY-MM-DD"),
    ):
        actual = str(_get_path(cfg, dotted) or "")
        if not regex.match(actual):
            findings.append(
                {"severity": "FAIL", "message": f"{dotted}（{label}）：实际 {actual!r}"}
            )
    if not _get_path(cfg, "data_version"):
        findings.append({"severity": "WARN", "message": "data_version 缺失"})
    return findings


def audit_summary(summary_path: Path) -> list[dict]:
    """核对单个 summary.json 的结构与判定枚举；返回 [(severity, message)]。"""
    findings: list[dict] = []
    try:
        with open(summary_path, encoding="utf-8") as fh:
            s = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return [{"severity": "FAIL", "message": f"summary 无法解析：{exc}"}]

    version = s.get("version", {})
    for key, regex, label in (
        ("code_version", CODE_VERSION_RE, "version.code_version 40 位 hash"),
        ("spec_version", SPEC_VERSION_RE, "version.spec_version v1.0+YYYY-MM-DD"),
    ):
        val = str(version.get(key) or "")
        if not regex.match(val):
            findings.append({"severity": "FAIL", "message": f"{label}：实际 {val!r}"})
    if not version.get("data_version"):
        findings.append({"severity": "WARN", "message": "version.data_version 缺失"})

    for gain_key in ("M_A_minus_M_B", "M_A_minus_M_C"):
        entry = s.get("prior_gain", {}).get(gain_key)
        if entry is None:
            findings.append({"severity": "WARN", "message": f"prior_gain.{gain_key} 缺失"})
            continue
        if entry.get("verdict") not in THREE_CLASS_VALUES:
            findings.append(
                {"severity": "FAIL", "message": f"prior_gain.{gain_key}.verdict 非法：{entry.get('verdict')!r}"}
            )
        if not isinstance(entry.get("ci95"), list) or len(entry.get("ci95", [])) != 2:
            findings.append({"severity": "WARN", "message": f"prior_gain.{gain_key}.ci95 缺失/非二元组"})

    if s.get("three_class", {}).get("verdict") not in THREE_CLASS_VALUES:
        findings.append(
            {"severity": "FAIL", "message": f"three_class.verdict 非法：{s.get('three_class', {}).get('verdict')!r}"}
        )
    if s.get("one_veto", {}).get("verdict") not in ONE_VETO_VALUES:
        findings.append(
            {"severity": "FAIL", "message": f"one_veto.verdict 非法：{s.get('one_veto', {}).get('verdict')!r}"}
        )
    return findings


def scan_99(change_log_path: Path) -> list[dict]:
    """解析 99 变更日志中触及判据模块的批次，返回对账表行（信息性，不计 FAIL）。"""
    rows: list[dict] = []
    try:
        lines = change_log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [{"severity": "WARN", "message": f"99 无法读取：{exc}"}]
    for line in lines:
        if not line.startswith("| 20"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 8:
            continue
        date, version, module, typ, desc, reason, impact, status = cols[:8]
        if typ not in {"Added", "Changed", "Removed", "Fixed"}:
            continue
        if not CRITERIA_MODULES.search(module):
            continue
        rows.append(
            {
                "severity": "INFO",
                "message": f"{date} | {module} | {typ} | {status} | {desc[:80]}",
            }
        )
    return rows


def audit_report(report_path: Path) -> list[dict]:
    """核对 final_report.md 预注册骨架标记；返回 [(severity, message)]。"""
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError:
        return [{"severity": "WARN", "message": f"final_report.md 不存在：{report_path}"}]
    missing = [m for m in REPORT_MARKERS if m not in text]
    if missing:
        return [
            {
                "severity": "FAIL",
                "message": f"final_report.md 缺预注册骨架标记：{missing}",
            }
        ]
    return [{"severity": "INFO", "message": "final_report.md 预注册骨架标记齐备"}]


def run_audit(
    results_dir: Path,
    change_log_path: Path,
    report_path: Path | None,
    experiment_prefixes: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """执行全部对账，返回 (findings, 对账表行)。

    ``experiment_prefixes`` 非空时只对账 run 目录名匹配任一前缀的 config/summary
    （M6 用例如 ``["EXP-02", "EXP-07"]``；EXP-01 校准期产物为旧 schema，不在 N8 对账范围）。
    run 目录名须匹配 ``EXP-<NN>[a-z]?_<scheme>_seed<N>_`` 模式——数据生成目录
    （如 ``EXP-02_data_v1``，code_version 为 N4 前短 hash、判据符号表字段缺失属历史产物）
    与汇总目录（``*_summary``）不纳入对账。
    """
    run_dir_pattern = re.compile(r"^EXP-\d+[a-z]?_([ABC])_seed\d+_")
    findings: list[dict] = []
    table: list[str] = []

    configs = sorted(results_dir.rglob("config.yaml"))
    if not configs:
        findings.append({"severity": "WARN", "message": f"{results_dir} 下无 config.yaml"})
    for cfg in configs:
        run_dir = cfg.parent
        if not run_dir_pattern.match(run_dir.name):
            continue
        if experiment_prefixes and not any(
            run_dir.name.startswith(p) for p in experiment_prefixes
        ):
            continue
        summary = run_dir / "summary.json"
        for f in audit_config(cfg):
            findings.append({"severity": f["severity"], "message": f"[config {cfg.parent.name}] {f['message']}"})
        if summary.exists():
            for f in audit_summary(summary):
                findings.append({"severity": f["severity"], "message": f"[summary {run_dir.name}] {f['message']}"})
        else:
            findings.append(
                {"severity": "WARN", "message": f"[summary {run_dir.name}] summary.json 缺失（评估未跑）"}
            )

    for f in scan_99(change_log_path):
        findings.append(f)

    if report_path is not None:
        for f in audit_report(report_path):
            findings.append(f)

    # 对账表行：99 判据模块批次（信息性）+ FAIL/WARN 汇总。
    table.append("| 类型 | 数量 |")
    table.append("|---|---|")
    table.append(f"| FAIL（不一致，验收失败） | {sum(1 for f in findings if f['severity'] == 'FAIL')} |")
    table.append(f"| WARN（缺失/待补） | {sum(1 for f in findings if f['severity'] == 'WARN')} |")
    table.append(f"| INFO（99 判据模块批次等） | {sum(1 for f in findings if f['severity'] == 'INFO')} |")
    return findings, table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/registration_audit.py",
        description="预注册对账（90 [S5] N8，M6 验收用）",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results", help="results/ 根目录")
    parser.add_argument("--99", dest="change_log", type=Path, default=PROJECT_ROOT / "docs" / "specs" / "99_change_log.md")
    parser.add_argument("--report", type=Path, default=None, help="final_report.md 路径（可选）")
    parser.add_argument("--experiments", default=None,
                        help="对账范围：run 目录名前缀，逗号分隔（如 EXP-02,EXP-07）；缺省对账全部")
    parser.add_argument("--out", type=Path, default=None, help="对账表输出路径（可选）")
    args = parser.parse_args(argv)

    prefixes = [p.strip() for p in args.experiments.split(",") if p.strip()] if args.experiments else None
    findings, table = run_audit(args.results_dir, args.change_log, args.report, prefixes)

    lines = ["# 预注册对账表（90 [S5] N8）", ""]
    lines.extend(table)
    lines.append("")
    lines.append("## 明细")
    lines.append("")
    for f in findings:
        lines.append(f"- **{f['severity']}** {f['message']}")

    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")

    n_fail = sum(1 for f in findings if f["severity"] == "FAIL")
    if n_fail:
        print(f"\n[registration_audit] 存在 {n_fail} 处 FAIL（预注册值不一致）——按 90 [S5] N8 视为验收失败", file=sys.stderr)
        return 1
    print("\n[registration_audit] 无 FAIL：预注册判据与阈值在载体间一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
