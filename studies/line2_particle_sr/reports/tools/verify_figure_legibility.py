"""图件可测清晰度审查（独立复核 make_phase1_figures.py 的产物与自证）。

用法：python studies/line2_particle_sr/reports/tools/verify_figure_legibility.py [--strict]

审查项（全部给出实测数字，方法与本仓库出图脚本相互独立）：
1. **屏幕字号**：按 .layout 实测栏宽 864 px 换算 screen_px = fontsize × 3600 / png_width_px，
   要求每张图的最小值 ≥ 11 px；同时核对 PNG 为 300 dpi。
2. **文字重叠（hide-and-diff）**：重建图对象，按 AABB 粗筛候选对，再逐对隐藏其中一条文字
   重新栅格化，取真实字形掩码的交集像素数；要求为 0。
3. **无底衬文字压深色墨迹**：同上差分，统计普通文字（无自底衬）落在深色像素（<150）上的面积。
4. **灰度可辨性**：用 WCAG 相对亮度公式独立复算调色板两两 ΔL，并打印每图采用的
   非颜色编码（线型 / 标记 / 阴影线 / 组名文字）。
5. **缺字检查**：重渲染时捕获 matplotlib 的 missing-glyph 警告，要求为 0。
6. **HTML 内联一致性**：解出 report HTML 里的 base64，与磁盘 PNG 逐字节比对。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_phase1_figures as mk  # noqa: E402

LINE2 = HERE.parent.parent
FIGS = LINE2 / "reports" / "figs"
HTML = LINE2 / "reports" / "line2_phase1_report.html"
COLUMN_PX = 864
MIN_SCREEN_PX = 11.0
DIFF_DPI = 100


# ---------------------------------------------------------------- 独立实现
def relative_luminance(hex_color: str) -> float:
    """WCAG 相对亮度（sRGB → 线性 → Y）。"""
    h = hex_color.lstrip("#")
    chans = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in chans]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def grey(im: np.ndarray) -> np.ndarray:
    return (0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]).astype(np.int16)


def raster(fig) -> np.ndarray:
    buf = io.BytesIO()
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(buf, format="png", dpi=DIFF_DPI)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def texts_of(fig) -> list:
    """实际绘出的文本：排除视图区间之外的刻度标签。"""
    hidden_ids: set[int] = set()
    live_ids: set[int] = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                hidden_ids.update((id(tick.label1), id(tick.label2)))
                loc = tick.get_loc()
                if loc is not None and (lo - 1e-9) <= float(loc) <= (hi + 1e-9):
                    live_ids.update((id(tick.label1), id(tick.label2)))
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        if id(t) in hidden_ids and id(t) not in live_ids:
            continue
        out.append(t)
    return out


def audit_figure(fig) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ts = texts_of(fig)
    boxes = []
    for t in ts:
        bb = t.get_window_extent(renderer=renderer)
        if bb.width <= 2 or bb.height <= 2:
            continue
        rot = abs(float(t.get_rotation())) % 180.0
        if 15.0 < rot < 165.0:              # 旋转文本的轴对齐框远大于字面，按中心收缩
            cx, cy = bb.x0 + bb.width / 2, bb.y0 + bb.height / 2
            bb = matplotlib.transforms.Bbox.from_extents(cx - bb.width * 0.3, cy - bb.height * 0.3,
                                                         cx + bb.width * 0.3, cy + bb.height * 0.3)
        boxes.append((t, bb))
    cands = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
            oy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
            if ox > 0 and oy > 0 and ox * oy / min(a.width * a.height, b.width * b.height) >= 0.10:
                cands.append((boxes[i][0], boxes[j][0]))
    base = raster(fig) if cands else None
    base_g = grey(base) if base is not None else None
    overlaps = []
    for ta, tb in cands:
        masks = []
        for t in (ta, tb):
            t.set_visible(False)
            m = np.abs(grey(raster(fig)) - base_g) > 8
            t.set_visible(True)
            masks.append(m)
        n = int(np.logical_and(masks[0], masks[1]).sum())
        if n:
            overlaps.append({"a": ta.get_text()[:44], "b": tb.get_text()[:44], "px": n})
    # 普通文字压深色墨迹
    plain = [t for t in ts if t.get_bbox_patch() is None]
    if plain:
        top = raster(fig)
        for t in plain:
            t.set_visible(False)
        under = grey(raster(fig))
        for t in plain:
            t.set_visible(True)
        pressed = int(np.logical_and(np.abs(grey(top) - under) > 8, under < 150).sum())
    else:
        pressed = 0
    return {"n_texts": len(ts), "n_candidate_pairs": len(cands), "overlaps": overlaps,
            "text_over_dark_ink_px": pressed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    metrics = json.loads((HERE / "fig_metrics.json").read_text(encoding="utf-8"))
    recorded = {f["png"]: f for f in metrics["figures"]}
    palette = metrics["palette"]
    encodings = metrics["noncolor_encodings"]

    print("=" * 100)
    print(f"图件可测清晰度审查 · 栏宽 {COLUMN_PX} px · 硬约束 最小屏幕字号 ≥ {MIN_SCREEN_PX} px")
    print("=" * 100)

    failures = []
    audit_rows = []
    print("\n[1–5] 逐图复核（重建图对象后独立测量）\n")
    print(f"{'图件':<32}{'PNG(px)':>10}{'dpi':>7}{'最小屏上字号':>13}{'重叠':>6}{'压深墨':>8}{'缺字':>6}")
    print("-" * 100)
    for name, fig in mk.build_figures():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = audit_figure(fig)
        missing = [w for w in caught if "missing from font" in str(w.message)]
        with Image.open(FIGS / name) as im:
            w_px, h_px = im.size
            dpi = im.info.get("dpi", (0, 0))[0]
        fonts = [t.get_fontsize() for t in texts_of(fig)]
        min_pt = min(fonts) if fonts else 0.0
        min_px = min_pt * 3600.0 / w_px
        rec = recorded.get(name, {})
        audit_rows.append({
            "png": name, "png_px": [w_px, h_px], "dpi": round(float(dpi), 1),
            "min_font_pt": round(float(min_pt), 2), "min_screen_px": round(min_px, 2),
            "n_texts": res["n_texts"], "overlap_px": sum(o["px"] for o in res["overlaps"]),
            "text_over_dark_ink_px": res["text_over_dark_ink_px"],
            "missing_glyph_warnings": len(missing),
            "recorded_min_screen_px": rec.get("min_screen_px"),
            "noncolor_encoding": encodings.get(name, ""),
        })
        flag = ""
        if min_px < MIN_SCREEN_PX:
            failures.append(f"{name}: 最小屏上字号 {min_px:.2f} px < {MIN_SCREEN_PX}")
            flag = "  ← 未达标"
        if res["overlaps"]:
            failures.append(f"{name}: 文字重叠 {len(res['overlaps'])} 处")
        if missing:
            failures.append(f"{name}: 缺字警告 {len(missing)} 条")
        if round(float(dpi)) != 300:
            failures.append(f"{name}: dpi = {dpi}")
        print(f"{name:<32}{w_px:>5}x{h_px:<5}{dpi:>7.0f}{min_px:>12.2f}px"
              f"{len(res['overlaps']):>6}{res['text_over_dark_ink_px']:>8}{len(missing):>6}{flag}")
        plt.close(fig)

    print("\n[3'] 灰度可辨性：调色板两两相对亮度差 ΔL（WCAG 公式独立复算）\n")
    names = list(palette)
    lum = {k: relative_luminance(v) for k, v in palette.items()}
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if palette[names[i]].lower() == palette[names[j]].lower():
                continue          # 同名颜色的别名（如 pine = tier10000）不算一对
            pairs.append((names[i], names[j], abs(lum[names[i]] - lum[names[j]])))
    pairs.sort(key=lambda p: p[2])
    print(f"  调色板 {len(names)} 色（含 1 组同名别名）、去重后 {len(pairs)} 对；ΔL 最小 {pairs[0][2]:.4f}"
          f"（{pairs[0][0]} ↔ {pairs[0][1]}）、最大 {pairs[-1][2]:.4f}")
    print("  ΔL 最小的 6 对：")
    for a, b, d in pairs[:6]:
        print(f"    {palette[a]} {a:<7} ↔ {palette[b]} {b:<7} ΔL = {d:.4f}"
              f"{'   < 0.15 ⇒ 必须另加非颜色编码' if d < 0.15 else ''}")
    n_below = sum(1 for *_, d in pairs if d < 0.15)
    print(f"  低于 0.15 的对：{n_below}/{len(pairs)} —— 结论：灰度可辨性由非颜色编码承担，逐图声明如下：")
    for name in [r["png"] for r in audit_rows]:
        print(f"    {name:<32} {encodings.get(name, '（无颜色编码需求）')}")

    print("\n[6] HTML 内联一致性\n")
    html = HTML.read_text(encoding="utf-8")
    embeds = re.findall(r'<img src="data:image/png;base64,([A-Za-z0-9+/=]+)" alt="([^"]+)"', html)
    print(f"  HTML 内联图：{len(embeds)} 张；磁盘 PNG：{len(list(FIGS.glob('*.png')))} 张")
    for b64, alt in embeds:
        raw = base64.b64decode(b64)
        disk = (FIGS / alt).read_bytes()
        same = hashlib.sha256(raw).hexdigest() == hashlib.sha256(disk).hexdigest()
        if not same:
            failures.append(f"HTML 内联与磁盘不一致：{alt}")
        print(f"    {alt:<32} {len(raw):>8,} B  sha256 一致={same}")
    if len(embeds) != len(list(FIGS.glob("*.png"))):
        failures.append("HTML 内联图数量与 figs/ 不一致")

    print("\n" + "=" * 100)
    worst = min(audit_rows, key=lambda r: r["min_screen_px"])
    print(f"汇总：{len(audit_rows)} 张图；最小屏上字号 {worst['min_screen_px']:.2f} px"
          f"（{worst['png']}）；文字重叠 {sum(r['overlap_px'] for r in audit_rows)} px（应为 0）；"
          f"无底衬文字压深墨 {sum(r['text_over_dark_ink_px'] for r in audit_rows)} px；"
          f"缺字警告 {sum(r['missing_glyph_warnings'] for r in audit_rows)} 条")
    print("=" * 100)
    out = HERE / "fig_legibility_audit.json"
    out.write_text(json.dumps({
        "column_px": COLUMN_PX, "min_screen_px_required": MIN_SCREEN_PX,
        "figures": audit_rows,
        "palette_delta_l": [{"a": a, "b": b, "dL": round(d, 4)} for a, b, d in pairs],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"审查记录：{out}")
    if failures:
        print("\n未通过项：")
        for f in failures:
            print("  -", f)
    else:
        print("\n全部审查项通过。")
    return 1 if (args.strict and failures) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
