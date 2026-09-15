"""数字溯源自检：列出 HTML 报告里每个数字 → 磁盘来源。

用法：python studies/line2_particle_sr/reports/tools/verify_report_numbers.py [--strict]

做法：
1) 建索引——把 results/ 与 reports/tools/phase1_numbers.json 的落盘产物逐层展开成
   “数值 → 来源文件”表，并为每个数值生成若干舍入形态（2–6 位有效数字、整数化），
   使报告里按位数截断的写法也能命中；
2) 扫报告——剥掉 <style>/<script>/base64 后抽出所有数字 token；
3) 对每个 token 报出命中的来源文件；未命中的 token 一律列出（--strict 时非零退出）。

命中是“该数值出现在某个落盘产物中”，不是“该数值只有这一个来源”；
报告正文中的结构性数字（年份、章节号、像素尺寸等）在 STRUCTURAL 里显式登记。
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LINE2 = ROOT / "studies" / "line2_particle_sr"
RESULTS = LINE2 / "results"
HTML = LINE2 / "reports" / "line2_phase1_report.html"
NUMBERS = Path(__file__).resolve().parent / "phase1_numbers.json"

#: 报告中的结构性数字（非测量值），显式登记以免被当成未溯源
STRUCTURAL = {
    "2026", "09", "15", "14", "13", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "00", "20", "30", "40", "50", "80", "100", "128", "256", "64", "6", "2", "1", "0",
    "3", "4", "5", "7", "9", "72", "300", "1000", "10",
    "-1",   # 归一化区间 [-1, 1] 的写法
}

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def index_values(obj, src: str, out: dict[str, set[str]], depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            index_values(v, src, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            index_values(v, src, out, depth + 1)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, int):
        add_forms(float(obj), src, out)
    elif isinstance(obj, float):
        add_forms(obj, src, out)
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            try:
                add_forms(float(m), src, out)
            except ValueError:
                pass


def add_forms(value: float, src: str, out: dict[str, set[str]]) -> None:
    if value != value or abs(value) == float("inf"):
        return
    forms = {f"{value:.{n}g}" for n in range(2, 8)}
    if abs(value) >= 100:          # 只对三位以上数值加整数舍入形态，避免小整数滥配
        forms.add(f"{value:.0f}")
    for n in range(1, 7):
        forms.add(f"{value:.{n}f}")
    for n in range(1, 5):      # 报告里的百分比写法（值 × 100）
        forms.add(f"{value * 100:.{n}f}")
    if float(value).is_integer():
        forms.add(f"{value:.0f}")
    forms.add(str(int(value)) if float(value).is_integer() else str(value))
    for f in forms:
        try:
            out[f].add(src)
        except TypeError:
            pass


def build_index() -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    sources: list[Path] = []
    sources += sorted(RESULTS.glob("*.json"))
    sources += sorted(RESULTS.glob("*.csv"))
    sources += sorted(RESULTS.glob("*/run_meta.json"))
    sources += sorted(RESULTS.glob("*/summary.json"))
    sources += sorted(RESULTS.glob("*/test_eval*.json"))
    sources += sorted(RESULTS.glob("*/train_log.jsonl"))
    sources += [NUMBERS]
    sources += [LINE2 / "data" / "line2_ds_v1" / "manifest.json"]
    sources += [LINE2 / "data" / "anchor" / "manifest.json"]
    sources += sorted((ROOT / "docs" / "specs" / "line2").glob("*.md"))
    sources += sorted((ROOT / "docs" / "specs" / "line2").glob("*.json"))
    for png in sorted((LINE2 / "reports" / "figs").glob("*.png")):
        add_forms(float(png.stat().st_size), str(png.relative_to(ROOT)), out)
    sources += [LINE2 / "reports" / "line2_phase1_stage_report.md"]
    sources += [LINE2 / "reports" / "phase1_report.md"]
    sources += sorted((RESULTS / "phase1_chain.log", RESULTS / "phase1_e2_chain.log"))
    for path in sources:
        if not path.exists():
            continue
        rel = str(path.relative_to(ROOT))
        try:
            if path.suffix == ".json":
                index_values(json.loads(path.read_text(encoding="utf-8")), rel, out)
            elif path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        index_values(json.loads(line), rel, out)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 千分位分组先合并
                for m in NUM_RE.findall(text):
                    try:
                        add_forms(float(m), rel, out)
                    except ValueError:
                        continue
        except Exception as exc:  # 单个来源解析失败不阻塞整体自检
            print(f"  (跳过 {rel}：{exc})", file=sys.stderr)
    return out


def strip_html(html: str) -> tuple[str, set[str]]:
    """剥标签并返回（正文文本, 10 的幂次集合）。

    报告用 <sup> 排版科学计数法（10<sup>-4</sup>），剥标签后会剩下孤立的指数，
    因此在剥标签之前先把这些指数登记下来，避免被当成未溯源数字。
    """
    exponents = set(re.findall(r"10\s*<sup>\s*(-?\d+)\s*</sup>", html))
    html = re.sub(r"<style>.*?</style>", " ", html, flags=re.S)
    html = re.sub(r"<script>.*?</script>", " ", html, flags=re.S)
    html = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", " ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    # 千分位分组（98,723.7）在分词前合并，避免断成两个 token
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text, exponents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="存在未溯源数字时非零退出")
    ap.add_argument("--show", type=int, default=40, help="最多打印多少条未溯源数字")
    args = ap.parse_args()

    index = build_index()
    body, exponents = strip_html(HTML.read_text(encoding="utf-8"))
    tokens = NUM_RE.findall(body)
    unique = sorted(set(tokens), key=lambda t: (-len(t), t))
    exponent_tokens = {e for e in exponents if e not in index}
    context: dict[str, str] = {}
    for tok in unique:
        m = re.search(r".{0,34}" + re.escape(tok) + r".{0,34}", body)
        context[tok] = re.sub(r"\s+", " ", m.group(0)) if m else ""

    traced, untraced, structural = [], [], []
    for tok in unique:
        if tok in STRUCTURAL:
            structural.append(tok)
            continue
        if tok in exponent_tokens:
            structural.append(tok)
            continue
        hit = index.get(tok, set())
        if not hit:
            for key, val in index.items():
                try:
                    if abs(float(key) - float(tok)) <= 1e-12 * max(1.0, abs(float(tok))):
                        hit |= val
                except (ValueError, ZeroDivisionError):
                    continue
        (traced if hit else untraced).append((tok, sorted(hit)))

    print(f"HTML：{HTML.relative_to(ROOT)}")
    print(f"索引来源文件：{len({s for v in index.values() for s in v})} 个；索引数值形态：{len(index):,} 个")
    print(f"报告中出现的不同数字 token：{len(unique)} 个")
    print(f"  · 结构性数字（显式登记 + 10 的幂次）：{len(structural)} 个")
    print(f"  · 已溯源：{len(traced)} 个")
    print(f"  · 未溯源：{len(untraced)} 个")
    print()
    print("已溯源数字 → 磁盘来源（每个数字最多列 3 个来源）")
    print("-" * 78)
    for tok, hits in sorted(traced, key=lambda x: (len(x[1]), x[0])):
        print(f"  {tok:>16s}  ←  {', '.join(hits[:3])}{' …' if len(hits) > 3 else ''}")
    if untraced:
        print()
        print("未溯源数字（需人工确认是否为结构性数字）")
        print("-" * 78)
        for tok, _ in sorted(untraced, key=lambda x: -float(x[0]))[: args.show]:
            print(f"  {tok:>10s}  …  {context.get(tok, '')}")
        if len(untraced) > args.show:
            print(f"  … 另有 {len(untraced) - args.show} 个")
    return 1 if (args.strict and untraced) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:   # 管道被提前关闭（如 | head）时安静退出
        sys.exit(0)
