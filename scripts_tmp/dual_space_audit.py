#!/usr/bin/env python
"""双空间契约审计：静态扫描 src/ 中工作尺度（S 空间）与 Σ=1 空间的边界。

目的
----
坍缩修复（50 [S12] C5）后项目存在两个数值空间：
  * Σ=1 空间：磁盘上的 H、predictions.npz、全部指标与哨兵口径；
  * 工作尺度空间（×S，S=N²=65536）：模型前向输出、损失目标、训练日志。
契约：凡消费模型输出必须显式 ÷S 还原；凡构造损失目标必须显式 ×S；
checkpoint 必须持久化 work_scale 并在加载时校验。任何「忘了 /S 或 ×S」
的位置都会产生 10⁴ 倍量级的静默错误。本脚本做系统性静态扫描。

依赖
----
仅 stdlib（ast + re）。不 import torch/项目模块，可在任何环境运行。

用法
----
    python scripts_tmp/dual_space_audit.py [--root src]

预期输出表
----------
    check | file:line                | status | detail
    C1    | src/...py:NN             | PASS/FAIL | 模型输出消费点是否配对 ÷S
    ...
    SUMMARY: n_fail_p0=... n_fail_p1=...

检查项与解读规则
----------------
- C1 输出配对：任何调用 `forward_scheme(...)` 的函数体内必须出现
  `/ scale` 或 `/ model.work_scale`（即还原）。
  FAIL（无配对）→ **P0**：EXP-02 前必须修复（漏洞 N2 关联）。
- C2 目标配对：`src/training/` 内使用 `batch["H"]` 的行必须同现
  `work_scale`（构造损失目标 ×S）。
  FAIL → **P0**。src/training 之外的 `batch["H"]/["H"]` 使用按 Σ=1 真值
  处理，仅 INFO 列出（评估真值消费，正确行为）。
- C3 checkpoint 绑定：(a) 保存侧 state 必须含 "config" 或顶层
  "work_scale" 键；(b) 加载侧（evaluate.load_model）必须存在
  work_scale 一致性断言。
  (b) 缺失 → **P1**（报告 §3 提案 9：当前已知缺口）。
- C4 日志空间标注：训练日志中 out_min/out_max/out_sum 若直接取自
  工作尺度张量且无 `/` 还原、字段名无 `_work` 后缀 → **P1**
  （报告 §3 提案 19，train.py 已知命中）。
- C5 评估还原哨兵：`infer_predictions` 必须含 `/ scale` 还原
  （回归保护，预期恒 PASS）。
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINDINGS: list[tuple[str, str, str, str]] = []


def add(check: str, loc: str, status: str, detail: str) -> None:
    FINDINGS.append((check, loc, status, detail))


def iter_funcs_with_call(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if (isinstance(fn, ast.Name) and fn.id == name) or (
                        isinstance(fn, ast.Attribute) and fn.attr == name
                    ):
                        yield node, sub.lineno
                        break


def c1_output_pairing(files: dict[Path, str]) -> None:
    """C1：forward_scheme 输出消费点必须配对 ÷S。"""
    for path, src in files.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            add("C1", f"{path}:?", "FAIL", "语法错误，无法解析")
            continue
        for func, lineno in iter_funcs_with_call(tree, "forward_scheme"):
            seg = ast.get_source_segment(src, func) or ""
            paired = re.search(
                r"/\s*(scale|model\.work_scale|work_scale)\b|\*\s*model\.work_scale", seg
            )
            rel = path.relative_to(PROJECT_ROOT)
            if paired:
                add("C1", f"{rel}:{func.lineno}", "PASS",
                    f"{func.name}() 输出已配对（÷S 还原或 ×S 目标同空间）")
            else:
                add("C1", f"{rel}:{lineno}", "FAIL",
                    f"{func.name}() 消费 forward_scheme 输出但函数体内未见 ÷S/×S 配对")


def c2_target_pairing(files: dict[Path, str]) -> None:
    """C2：训练侧 batch["H"] 必须与 work_scale 同现（损失目标 ×S）。"""
    pat = re.compile(r"batch\[[\"']H[\"']\]")
    for path, src in files.items():
        rel = path.relative_to(PROJECT_ROOT)
        in_training = rel.parts[1] == "training" if len(rel.parts) > 1 else False
        for i, line in enumerate(src.splitlines(), 1):
            if not pat.search(line):
                continue
            if in_training:
                if "work_scale" in line:
                    add("C2", f"{rel}:{i}", "PASS", "损失目标已配对 ×S")
                else:
                    add("C2", f"{rel}:{i}", "FAIL",
                        "训练侧使用 batch[\"H\"] 但同行未见 work_scale")
            else:
                add("C2", f"{rel}:{i}", "INFO", "非训练侧真值消费（按 Σ=1 空间，预期行为）")


def c3_checkpoint_binding(files: dict[Path, str]) -> None:
    """C3：checkpoint 持久化 work_scale + 加载侧一致性断言。"""
    save_ok, save_loc, found_call = False, "", False
    for path, src in files.items():
        rel = path.relative_to(PROJECT_ROOT)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            calls_util = any(
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "save_checkpoint"
                for sub in ast.walk(func)
            )
            if not calls_util or func.name == "save_checkpoint":
                continue
            found_call = True
            seg = ast.get_source_segment(src, func) or ""
            if re.search(r"[\"']config[\"']\s*:|[\"']work_scale[\"']\s*:", seg):
                save_ok, save_loc = True, f"{rel}:{func.lineno}"
            else:
                save_loc = save_loc or f"{rel}:{func.lineno}"
    if found_call:
        add("C3a", save_loc or "-", "PASS" if save_ok else "FAIL",
            "保存函数 state 含 config/work_scale 键" if save_ok
            else "保存函数 state 未见 config/work_scale 键")
    else:
        add("C3a", "-", "INFO", "未发现 save_checkpoint 调用点")

    load_src = next((s for p, s in files.items() if p.name == "evaluate.py"), "")
    has_assert = bool(re.search(r"work_scale.*(assert|!=|raise)|assert.*work_scale", load_src))
    add("C3b", "src/evaluation/evaluate.py", "PASS" if has_assert else "FAIL",
        "load_model 存在 work_scale 一致性断言" if has_assert
        else "load_model 无 work_scale 一致性断言（已知缺口，报告 §3 提案 9）")


def c4_log_space(files: dict[Path, str]) -> None:
    """C4：训练日志统计量若取自工作尺度张量，需 ÷S 还原或 _work 后缀。"""
    for path, src in files.items():
        if not (len(path.parts) >= 2 and path.parts[-2:] == ("training", "train.py")):
            continue
        rel = path.relative_to(PROJECT_ROOT)
        lines = src.splitlines()
        for i, line in enumerate(lines, 1):
            m = re.search(r"(out_min|out_max|out_sum)\s*=\s*float\(([^)]*)", line)
            if not m:
                continue
            expr = m.group(2)
            divided = "/" in expr or re.search(r"\bwork_scale\b|\bscale\b", expr)
            renamed = re.search(rf"[\"']{m.group(1)}_work[\"']", "\n".join(lines[i - 1:i + 20]))
            if divided:
                add("C4", f"{rel}:{i}", "PASS", f"{m.group(1)} 已还原到 Σ=1 空间")
            elif renamed:
                add("C4", f"{rel}:{i}", "PASS", f"{m.group(1)} 以 _work 后缀标注空间")
            else:
                add("C4", f"{rel}:{i}", "FAIL",
                    f"{m.group(1)} 直接取自工作尺度张量，无还原无后缀（报告 §3 提案 19）")


def c5_infer_restore(files: dict[Path, str]) -> None:
    """C5：infer_predictions 必须 ÷S 还原（回归哨兵）。"""
    for path, src in files.items():
        if path.name != "evaluate.py":
            continue
        rel = path.relative_to(PROJECT_ROOT)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "infer_predictions":
                seg = ast.get_source_segment(src, node) or ""
                if re.search(r"/\s*scale\b", seg):
                    add("C5", f"{rel}:{node.lineno}", "PASS", "infer_predictions 已 ÷S 还原")
                else:
                    add("C5", f"{rel}:{node.lineno}", "FAIL", "infer_predictions 未见 ÷S")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="双空间契约静态审计")
    ap.add_argument("--root", default="src")
    args = ap.parse_args(argv)

    root = PROJECT_ROOT / args.root
    files = {p: p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.py"))
             if "__pycache__" not in p.parts}
    if not files:
        print(f"未在 {root} 找到 .py 文件")
        return 1

    c1_output_pairing(files)
    c2_target_pairing(files)
    c3_checkpoint_binding(files)
    c4_log_space(files)
    c5_infer_restore(files)

    print(f"{'check':6s} | {'location':38s} | {'status':4s} | detail")
    print("-" * 110)
    for check, loc, status, detail in FINDINGS:
        print(f"{check:6s} | {loc:38s} | {status:4s} | {detail}")

    fail_p0 = [f for f in FINDINGS if f[2] == "FAIL" and f[0] in ("C1", "C2")]
    fail_p1 = [f for f in FINDINGS if f[2] == "FAIL" and f[0] in ("C3b", "C4", "C5")]
    print("\n解读规则：")
    print("- C1/C2 FAIL = P0（EXP-02 前必须修复：模型输出未还原或损失目标未放大）")
    print("- C3b/C4/C5 FAIL = P1（报告 §3 提案 9/19；C5 为回归哨兵，FAIL 即数据损坏风险）")
    print(f"SUMMARY: n_fail_p0={len(fail_p0)} n_fail_p1={len(fail_p1)} "
          f"total_findings={len(FINDINGS)}")
    return 1 if fail_p0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
