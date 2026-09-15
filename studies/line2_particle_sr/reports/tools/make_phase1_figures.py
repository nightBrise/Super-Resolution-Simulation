"""生成 line2 Phase 1 阶段报告的图件（300 dpi PNG，落 reports/figs/）+ 可测清晰度自证。

设计约束（按 864 px 栏宽反推）：
- 屏幕上某段文字的像素高 = fontsize_pt × 3600 / png_width_px
  （300 dpi 下 1 pt = 4.1667 px，显示时按 864/width 缩放），
  故本脚本把"最小屏幕字号 ≥ 11 px"作为硬约束，逐图实测并写入 fig_metrics.json。
- 只靠颜色区分的系列（曲线 / 分组）一律追加非颜色编码（线型 + 标记 + 阴影线 + 组名文字）。
- 自由浮动的注释文字一律带半透明白底，避免压线。

运行：python studies/line2_particle_sr/reports/tools/make_phase1_figures.py
只读 studies/line2_particle_sr/{data,results} 与同目录 phase1_numbers.json；只写 reports/figs/ 与 fig_metrics.json。
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.line2.data_pipeline.projections import normalize_axis  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
METRICS_OUT = HERE / "fig_metrics.json"
NUMBERS = json.loads((HERE / "phase1_numbers.json").read_text(encoding="utf-8"))
D = NUMBERS["derived"]
LINE2 = ROOT / "studies" / "line2_particle_sr"
RESULTS = LINE2 / "results"
DATA = LINE2 / "data" / "line2_ds_v1"

CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(CJK)
CJK_NAME = fm.FontProperties(fname=CJK).get_name()

#: 报告栏宽（.layout max-width 1200 − padding 48 − sidebar 240 − gap 48）与硬约束
COLUMN_PX = 864
MIN_SCREEN_PX = 11.0

PINE = "#2D5A4A"
SADDLE = "#8B4513"
INK = "#1A1A1A"
GREY = "#666666"
VIOLET = "#6A4C93"
TIER_COLORS = {
    1000: "#7F7F7F", 3000: "#2B6CB0", 10000: "#2D5A4A", 30000: "#B7791F", 100000: "#A02020",
}
TIER_LABEL = {1000: "1e3", 3000: "3e3", 10000: "1e4", 30000: "3e4", 100000: "1e5"}
#: 只靠颜色区分的系列必须同时携带这些非颜色编码
STYLE_ENC = {
    "joint@3e3": dict(color=TIER_COLORS[3000], ls="-", marker="o", hatch="///"),
    "joint@1e4": dict(color=PINE, ls="--", marker="s", hatch="\\\\"),
    "indep@1e4": dict(color=TIER_COLORS[30000], ls="-.", marker="^", hatch="..."),
    "joint@1e5": dict(color=TIER_COLORS[100000], ls=":", marker="D", hatch="xxx"),
}
#: 供审查者独立复算灰度对比的调色板（sRGB 十六进制）
PALETTE = {"pine": PINE, "saddle": SADDLE, "ink": INK, "grey": GREY, "violet": VIOLET,
           **{f"tier{n}": c for n, c in TIER_COLORS.items()}}
DISP_AXIS = ("x", "x′", "y", "y′")

SEEDS = [100, 101, 102, 103, 104]
JOINT_RUNS = [f"e1_joint_s{i}" for i in range(5)]
INDEP_RUNS = [f"e1_indep_s{i}" for i in range(5)]
E2_RUNS = [f"e2_joint_n{t}_s{i}" for t in (3000, 100000) for i in range(5)]
ALL_RUNS = JOINT_RUNS + INDEP_RUNS + E2_RUNS
GROUP_OF_RUN = {
    **{r: "joint@3e3" for r in E2_RUNS[:5]},
    **{r: "joint@1e4" for r in JOINT_RUNS},
    **{r: "indep@1e4" for r in INDEP_RUNS},
    **{r: "joint@1e5" for r in E2_RUNS[5:]},
}
GROUP_ORDER = ["joint@3e3", "joint@1e4", "indep@1e4", "joint@1e5"]
GROUP_LABEL = {
    "joint@3e3": "joint @ 3e3",
    "joint@1e4": "joint @ 1e4",
    "indep@1e4": "indep @ 1e4",
    "joint@1e5": "joint @ 1e5",
}
RUNS_BY_GROUP = {g: [r for r in ALL_RUNS if GROUP_OF_RUN[r] == g] for g in GROUP_ORDER}
NUMS_SEED = {run: str(SEEDS[int(run[-1])]) for run in ALL_RUNS}
#: 只靠颜色区分的元素所追加的非颜色编码（审查脚本逐图打印）
NONCOLOR_ENC = {
    "fig05_training_dynamics.png":
        "4 组曲线用「实线+圆 / 虚线+方 / 点划线+三角 / 点线+菱形」区分——线型与标记均不同，图例按线型给出样本",
    "fig06_neff_and_err.png":
        "三档柱形用阴影线区分（///=3e3、\\\\=1e4、xxx=1e5），N_eff 面板同档用不同 marker；图例标题写明“阴影线 = 档位”",
    "fig08a_h8_passrate.png":
        "达标档用阴影线区分（xx=全通过、\\\\=部分通过、//=全不通过），且每根柱旁直接给出百分比与通过束数",
    "fig08b_dark_buckets.png":
        "裸输入用斜纹、模型用实心，图例文字点明“斜纹 / 实心”",
    "fig09a_zero_info_errs.png":
        "四个档位分组有背景色带 + 行末组名文字；箱体再叠加阴影线，y 轴标签只留种子号",
    "fig09b_zero_info_wins.png":
        "同上：分组色带 + 行末组名文字；柱长数值与误差比直接写在柱旁",
    "fig03_e0_curve.png":
        "四条落点系列用不同 marker（圆/方/三角/菱形）+ 深色描边；曲线用实线、地板线用虚线",
    "fig04_amplification.png":
        "两臂用不同 marker（圆/方）；域内为实心、上溢为空心（形状之外再编码填充）",
    "fig07_h3_pairing.png":
        "两臂用不同 marker（圆/方）+ 上溢空心，哑铃连线给出配对关系",
    "fig10_h7_cross_station.png":
        "两站用不同 marker（圆/菱形）+ 站 1 上溢空心；连线给出逐种子配对",
    "fig01_pipeline.png": "流程节点为独立方框 + 箭头方向，不依赖颜色",
    "fig02_lattice.png": "两站两投影为四个独立面板 + 面板标题文字，不依赖颜色",
}
METRICS: list[dict] = []


def style() -> None:
    """字号按“864 px 栏宽下 ≥ 11 px”反推：宽度 ≤ 3000 px 的图，正文 ≥ 10.5 pt 即可。"""
    plt.rcParams.update(
        {
            "font.family": CJK_NAME,
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#BBBBBB",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


# ------------------------------------------------------------------ 度量工具
def visible_texts(fig) -> list:
    """实际会被绘出的文本（视图外的刻度标签不算）。"""
    all_ids: set[int] = set()
    drawn: set[int] = set()
    for ax in fig.axes:
        for axis, lims in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lims), max(lims)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                for lab in (tick.label1, tick.label2):
                    all_ids.add(id(lab))
                loc = tick.get_loc()
                if loc is not None and (lo - 1e-9) <= float(loc) <= (hi + 1e-9):
                    drawn.add(id(tick.label1))
                    drawn.add(id(tick.label2))
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        if id(t) in all_ids and id(t) not in drawn:
            continue
        out.append(t)
    return out


def shrink_bbox(bb, frac: float):
    cx, cy = bb.x0 + bb.width / 2, bb.y0 + bb.height / 2
    hw, hh = bb.width * frac / 2, bb.height * frac / 2
    return matplotlib.transforms.Bbox.from_extents(cx - hw, cy - hh, cx + hw, cy + hh)


def candidate_pairs(texts, renderer, min_frac: float = 0.10):
    """旋转感知的 AABB 粗筛：给出需要做像素级复核的文本对。"""
    items = []
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        if bb.width <= 2 or bb.height <= 2:
            continue
        rot = abs(float(t.get_rotation())) % 180.0
        if 15.0 < rot < 165.0:
            bb = shrink_bbox(bb, 0.6)
        items.append((t, bb))
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ox = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
            oy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
            if ox <= 0 or oy <= 0:
                continue
            frac = ox * oy / min(a.width * a.height, b.width * b.height)
            if frac >= min_frac:
                pairs.append((items[i][0], items[j][0], frac))
    return sorted(pairs, key=lambda x: -x[2])


def _gray(fig, dpi: int = 100) -> np.ndarray:
    """非紧裁渲染成灰度图：所有差分渲染必须用同一画布，否则掩码无法对齐。"""
    buf = io.BytesIO()
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.int16)


def hide_and_diff(fig, texts) -> dict:
    """像素级复核：逐候选对隐藏其一做差分，得到真实字形掩码并计交叠像素。"""
    renderer = fig.canvas.get_renderer()
    pairs = candidate_pairs(texts, renderer)
    base = None
    overlaps = []
    for ta, tb, frac in pairs:
        if base is None:
            base = _gray(fig)
        masks = {}
        for key, t in (("a", ta), ("b", tb)):
            t.set_visible(False)
            img = _gray(fig)
            t.set_visible(True)
            masks[key] = np.abs(img - base) > 8
        n = int(np.logical_and(masks["a"], masks["b"]).sum())
        if n > 0:
            overlaps.append(
                {"a": ta.get_text()[:48], "b": tb.get_text()[:48], "overlap_px_100dpi": n,
                 "aabb_frac": round(float(frac), 3)}
            )
    return {"n_candidates": len(pairs), "overlaps": overlaps, "base": base}


def text_over_nontext(fig, texts) -> int:
    """“无底衬文字”压在深色墨迹（曲线 / 虚线 / 柱）上的像素数。

    带不透明底衬（bbox patch）的注释按设计即可读，不参与该统计；浅色带底与网格
    不算“被压住的对象”（阈值 < 150）。
    """
    plain = [t for t in texts if t.get_bbox_patch() is None]
    base_all = _gray(fig)
    for t in plain:
        t.set_visible(False)
    hidden = _gray(fig)
    for t in plain:
        t.set_visible(True)
    text_ink = np.abs(base_all - hidden) > 8
    dark_ink = hidden < 150
    return int(np.logical_and(text_ink, dark_ink).sum())


def localize_text_over_ink(fig, texts, top: int = 4) -> list[dict]:
    """逐条定位“无底衬文字压在深色墨迹”上：给出文字与像素数。"""
    plain = [t for t in texts if t.get_bbox_patch() is None]
    base = _gray(fig)
    out = []
    for t in plain:
        t.set_visible(False)
        hidden = _gray(fig)
        t.set_visible(True)
        n = int(np.logical_and(np.abs(base - hidden) > 8, hidden < 150).sum())
        if n:
            out.append({"text": t.get_text().replace("\n", "⏎")[:56], "px_100dpi": n})
    return sorted(out, key=lambda d: -d["px_100dpi"])[:top]


def measure(fig, name: str) -> dict:
    fig.canvas.draw()
    texts = visible_texts(fig)
    diff = hide_and_diff(fig, texts)
    pressed_raw = text_over_nontext(fig, texts)
    pressed_detail = localize_text_over_ink(fig, texts) if pressed_raw else []
    fig.savefig(FIGS / name, dpi=300)
    with Image.open(FIGS / name) as im:
        w_px, h_px = im.size
        dpi = im.info.get("dpi")
    rows = []
    for t in texts:
        rows.append({"text": t.get_text().replace("\n", "⏎")[:72],
                     "fontsize": round(float(t.get_fontsize()), 2)})
    for r in rows:
        r["screen_px"] = round(r["fontsize"] * 3600.0 / w_px, 2)
    smallest = min(rows, key=lambda r: r["screen_px"])
    return {
        "png": name,
        "png_px": [w_px, h_px],
        "dpi": round(float(dpi[0]), 1) if dpi else None,
        "n_texts": len(rows),
        "min_screen_px": smallest["screen_px"],
        "min_screen_px_element": f"{smallest['text']}（{smallest['fontsize']} pt）",
        "min_fontsize": min(r["fontsize"] for r in rows),
        "text_px_histogram": sorted({r["screen_px"] for r in rows}),
        "aabb_candidates": diff["n_candidates"],
        "pixel_overlaps": diff["overlaps"],
        "text_over_nontext_px_100dpi": pressed_raw,
        "text_over_nontext_detail": pressed_detail,
    }


def save(fig, name: str) -> None:
    rec = measure(fig, name)
    METRICS.append(rec)
    flag = "OK " if rec["min_screen_px"] >= MIN_SCREEN_PX else "!! "
    extra = ""
    if rec["pixel_overlaps"]:
        extra += f"  重叠 {len(rec['pixel_overlaps'])} 处"
    if rec["text_over_nontext_px_100dpi"]:
        d = rec["text_over_nontext_detail"]
        extra += f"  压线 {rec['text_over_nontext_px_100dpi']}px（{d[0]['text'][:22] if d else ''}）"
    print(f"  {flag}{name:32s} {rec['png_px'][0]:>5}px  最小屏上字号 {rec['min_screen_px']:>5.2f} px "
          f"（{rec['min_screen_px_element'][:34]}）{extra}")
    plt.close(fig)


# ------------------------------------------------------------------ 绘图基元
def box(ax, x, y, w, h, text, fc="#FAF8F5", ec=PINE, fs=11.0, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.5)


def arrow(ax, p0, p1, color=PINE, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))


NOTE_BBOX = dict(boxstyle="round,pad=0.36", facecolor="white", edgecolor="none", alpha=0.88)


# ------------------------------------------------------------------ 图 1
def fig01_pipeline():
    """节点框交给 matplotlib 按文字自适应尺寸（boxstyle pad），杜绝文字溢框压边框。"""
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    nodes = {
        "A": (0.115, 0.55, "锚点粒子表\n2×10^5 行 × 6 列\n(x, x′, y, y′)\n束群 1000（抖动 ×FODO）",
              "#FAF8F5", PINE),
        "B": (0.345, 0.55, "六投影分箱\n6 × 128 × 128\nint32 计数直方图", "#FAF8F5", PINE),
        "C": (0.585, 0.80, "低计数输入\nN_low ∈ {$10^3$…$10^5$}\n五档 × 每档 20 个子样本",
              "#FAF8F5", TIER_COLORS[3000]),
        "D": (0.585, 0.24, "同束同站的目标\n2×10^5 全量直方图\n= 参照 R", "#FAF8F5", TIER_COLORS[100000]),
        "E": (0.835, 0.55, "网络\n6 × 128² → 6 × 128²\n（联合 FiLM U-Net 或\n6 个独立子网）", "#FFF6EC", SADDLE),
        "F": (0.975, 0.55, "误差 →\nN_eff", "#FFF6EC", SADDLE),
    }
    texts = {}
    for key, (x, y, txt, fc, ec) in nodes.items():
        texts[key] = ax.text(x, y, txt, ha="center", va="center", fontsize=11.0, color=INK,
                             linespacing=1.55,
                             bbox=dict(boxstyle="round,pad=0.55", facecolor=fc, edgecolor=ec,
                                       linewidth=1.2))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    ext = {}
    for key, t in texts.items():
        bb = t.get_window_extent(renderer=renderer)
        (x0, y0) = inv.transform((bb.x0, bb.y0)); (x1, y1) = inv.transform((bb.x1, bb.y1))
        ext[key] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    for a, b in (("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"), ("E", "F")):
        arrow(ax, ((ext[a][2] + 0.006), (ext[a][1] + ext[a][3]) / 2),
              ((ext[b][0] - 0.006), (ext[b][1] + ext[b][3]) / 2))
    ax.text(0.5, 1.06, "图 1 · Phase 1 任务与数据管线：粒子表 → 六投影 → 低计数输入 → 2×$10^5$ 目标",
            ha="center", va="top", fontsize=13, color=INK, weight="bold")
    ax.text(0.5, 0.02,
            "关键约定：输入与输出都是 128×128 像素的同一张图——“放大”指等效粒子数（N_eff / N_low），"
            "不是像素分辨率",
            ha="center", va="bottom", fontsize=11.0, color=SADDLE,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAF8F5", edgecolor=SADDLE, linewidth=1.1))
    return fig


# ------------------------------------------------------------------ 图 2
def fig02_lattice():
    beam = "beam_0006"
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8))
    stats = {}
    with h5py.File(DATA / "val.h5", "r") as fh:
        for row, station in enumerate((0, 1)):
            grp = fh[beam][f"station{station}"]
            edges = np.asarray(grp["target_full"].attrs["bin_edges"], dtype=np.float64)
            img = grp["target_full"][:]
            for col, (gi, ai, aj) in enumerate(((0, 0, 1), (1, 2, 3))):
                ax = axes[row][col]
                sub = img[gi].astype(np.float64)
                ax.pcolormesh(normalize_axis(edges[ai], edges[ai]),
                              normalize_axis(edges[aj], edges[aj]),
                              np.maximum(sub, 0.5), shading="auto", cmap="magma",
                              norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=sub.max()))
                ui = 0.5 * (edges[ai][:-1] + edges[ai][1:]); uj = 0.5 * (edges[aj][:-1] + edges[aj][1:])
                pi = sub.sum(axis=1) / sub.sum(); pj = sub.sum(axis=0) / sub.sum()
                mi = float((pi * ui).sum()); mj = float((pj * uj).sum())
                si = float(np.sqrt((pi * (ui - mi) ** 2).sum()))
                sj = float(np.sqrt((pj * (uj - mj) ** 2).sum()))
                stats[(station, ai)] = (si, sj)
                ax.tick_params(axis="both", pad=6)
                ax.set_title(f"站 {station} · ({DISP_AXIS[ai]}, {DISP_AXIS[aj]})", fontsize=12)
                ax.set_xlabel(f"归一化 {DISP_AXIS[ai]}")
                ax.set_ylabel(f"归一化 {DISP_AXIS[aj]}")
                ax.grid(False)
                ax.text(0.03, 0.965,
                        f"物理 RMS：σ({DISP_AXIS[ai]}) = {si:.3e}\n"
                        f"                σ({DISP_AXIS[aj]}) = {sj:.3e}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=10.5, color="white",
                        linespacing=1.5,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000099", edgecolor="none"))
    cb = fig.colorbar(axes[0][1].collections[0],
                      ax=[axes[r][c] for r in range(2) for c in range(2)],
                      fraction=0.026, pad=0.02)
    cb.set_label("计数（log 色标）", fontsize=11)
    cb.ax.tick_params(labelsize=10.5)
    fig.suptitle("图 2 · 同一束在 FODO 两站的相空间分布并不相同", fontsize=13, color=INK, weight="bold")
    fig.text(0.5, 0.008,
             f"两轴各自归一到 [-1, 1]；σ(x′) 由站 0 的 {stats[(0, 0)][1]:.3e} 变为站 1 的 "
             f"{stats[(1, 0)][1]:.3e}",
             ha="center", va="bottom", fontsize=11.0, color=INK)
    fig.subplots_adjust(top=0.925, bottom=0.150, hspace=0.30, wspace=0.34)
    return fig


# ------------------------------------------------------------------ 图 3
def fig03_e0_curve():
    e0 = NUMBERS["e0"]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ns = [p["n_low"] for p in e0["points"]]
    med = np.array([p["median"] for p in e0["points"]])
    lo = np.array([p["iqr_lo"] for p in e0["points"]])
    hi = np.array([p["iqr_hi"] for p in e0["points"]])
    ax.fill_between(ns, lo, hi, color=PINE, alpha=0.16)
    ax.plot(ns, med, "o-", color=PINE, lw=2.2, ms=7)
    floor = e0["reference_noise_floor"]
    ax.axhline(floor, color=SADDLE, ls="--", lw=1.6)
    tier_med = e0["points"][-1]["median"]
    rows = NUMBERS["n_eff_rows"]
    for key, tier in (("joint@3e3", 3000), ("joint@1e4", 10000), ("indep@1e4", 10000),
                      ("joint@1e5", 100000)):
        pts = [r for r in rows if r["n_low"] == tier and r["station"] == 0
               and GROUP_OF_RUN[r["run_id"]] == key]
        xs = np.full(len(pts), tier) * np.exp(np.linspace(-0.10, 0.10, len(pts)))
        ax.plot(xs, [r["err_median"] for r in pts], STYLE_ENC[key]["marker"], ls="none", ms=7,
                mfc=STYLE_ENC[key]["color"], mec=INK, mew=0.6)
    ax.axvspan(1000, 100000, color="#E8E4DC", alpha=0.35, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("输入子样本粒子数 N_low（log）")
    ax.set_ylabel("六投影平均相对 RMSE（M1，log）")
    ax.set_title("图 3 · “尺子”：裸模拟的误差–粒子数曲线与噪声地板", fontsize=13, color=INK, weight="bold")
    handles = [
        Line2D([], [], color=PINE, lw=2.2, marker="o", ms=6, label="M1 中位数（E0 裸误差–N 曲线）"),
        Patch(facecolor=PINE, alpha=0.25, label="IQR 带（跨 300 束 × 20 子样本）"),
        Line2D([], [], color=SADDLE, ls="--", lw=1.6, label=f"参照噪声地板 {floor:.6f}"),
    ]
    for key in GROUP_ORDER:
        e = STYLE_ENC[key]
        handles.append(Line2D([], [], ls="none", marker=e["marker"], ms=7, mfc=e["color"],
                              mec=INK, mew=0.6, label=f"{GROUP_LABEL[key]} 各 run 的 err 落点"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.95)
    ax.annotate("误差随粒子数近似幂律下降：\n1e3 → 1e4 档，4.030 → 1.246（约 1/3.2）",
                xy=(10000, 1.2455), xytext=(1150, 0.38), fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0), linespacing=1.5,
                va="bottom", ha="left", bbox=NOTE_BBOX)
    ax.annotate(f"1e5 档裸误差中位数 {tier_med:.6f}\n"
                f"参照噪声地板 {floor:.6f}（1e5 档配对差/√2）\n"
                f"两者相差仅 {D['floor_vs_tier_gap']:.1e}——同一可分辨性下界的两种估计",
                xy=(70000, floor), xytext=(3100, 0.148), fontsize=10.5, color=SADDLE,
                arrowprops=dict(arrowstyle="->", color=SADDLE, lw=1.0), linespacing=1.5,
                va="bottom", ha="left", bbox=NOTE_BBOX)
    ax.set_xlim(880, 150000); ax.set_ylim(0.135, 6.5)
    return fig


# ------------------------------------------------------------------ 图 4
def fig04_amplification():
    rows = NUMBERS["n_eff_rows"]
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    xs = np.logspace(np.log10(2200), np.log10(145000), 200)
    ax.plot(xs, 1e5 / xs, ls="--", color=SADDLE, lw=1.7)
    ax.plot(xs, np.ones_like(xs), ls=":", color=GREY, lw=1.7)
    ax.axhspan(1, 10, color=PINE, alpha=0.10)
    ax.axhspan(10, 60, color=SADDLE, alpha=0.07)
    ax.text(2600, 1.75, "可验证放大域（≤ 10×，`00` [S1] 声明的叙事上限）", fontsize=10.5, color=PINE,
            bbox=NOTE_BBOX)
    ax.text(2600, 11.5, "超出 10× 声明边界", fontsize=10.5, color=SADDLE, bbox=NOTE_BBOX)
    series = [
        ("joint @ 站 0",
         [r for r in rows if r["station"] == 0 and (r["run_id"] in JOINT_RUNS
                                                    or r["run_id"].startswith("e2_joint_n"))],
         PINE, "o", "-"),
        ("independent @ 站 0",
         [r for r in rows if r["station"] == 0 and r["run_id"] in INDEP_RUNS],
         TIER_COLORS[30000], "s", "--"),
    ]
    for label, rs, color, marker, ls in series:
        by_tier: dict[int, list[float]] = {}
        for r in rs:
            by_tier.setdefault(r["n_low"], []).append(r["n_eff"] / r["n_low"])
        for tier in sorted(by_tier):
            vals = np.asarray(by_tier[tier])
            xs_t = np.full(len(vals), tier) * np.exp(np.linspace(-0.11, 0.11, len(vals)))
            overflow = np.isclose(vals, 1e5 / tier, rtol=1e-9)
            if (~overflow).any():
                ax.plot(xs_t[~overflow], vals[~overflow], marker=marker, ls="none", ms=10,
                        mfc=color, mec=color,
                        label=f"{label} · {TIER_LABEL[tier]}（实心 = 域内真实插值）")
            if overflow.any():
                ax.plot(xs_t[overflow], vals[overflow], marker=marker, ls="none", ms=10,
                        mfc="none", mec=color, mew=2.1,
                        label=f"{label} · {TIER_LABEL[tier]}（空心 = 上溢 · 域顶）")
    ax.plot([3000, 10000, 100000], [1e5 / 3000, 1e5 / 10000, 1.0], "-", color=PINE, lw=1.3, alpha=0.55)
    for txt, xy, xytext in (
        ("33.3× = $10^5$/$3\\times10^3$\n由域顶构造得出，分辨力为零", (3000, 33.33), (3900, 15.0)),
        ("10× = $10^5$/$10^4$：同上", (10000, 10.0), (14500, 4.4)),
        ("1e5 档：唯一非饱和点\n（s2 的 98,723.7，未触顶）", (100000, 1.0), (42000, 1.75)),
    ):
        ax.annotate(txt, xy=xy, xytext=xytext, fontsize=10.5, color=SADDLE, linespacing=1.5,
                    arrowprops=dict(arrowstyle="->", color=SADDLE, lw=1.0), bbox=NOTE_BBOX)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("输入粒子数 $N_{low}$（log）")
    ax.set_ylabel("放大比 $N_{eff}/N_{low}$（log）")
    ax.set_title("图 4 · 三点放大主曲线：只有右端那一点具有分辨力", fontsize=13, color=INK, weight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(0.75, 95); ax.set_xlim(2200, 150000)
    return fig


# ------------------------------------------------------------------ 图 5（训练动力学）
def fig05_training_dynamics():
    tr = NUMBERS["training"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    ax = axes[0]
    for key in GROUP_ORDER:
        e = STYLE_ENC[key]
        for k, run in enumerate(RUNS_BY_GROUP[key]):
            ax.plot(tr[run]["steps"], tr[run]["val_n_eff"], color=e["color"], ls=e["ls"], lw=1.3,
                    marker=e["marker"], ms=3.6, markevery=3, alpha=0.85,
                    label=GROUP_LABEL[key] if k == 0 else None)
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.7)
    ax.text(400, 1.055e5, "选型准绳 $N_{eff}$ 在域顶 $10^5$ 饱和 ⇒ 之后不再更新 best",
            fontsize=10.5, color=SADDLE, va="bottom", bbox=NOTE_BBOX)
    ax.set_ylim(2.5e4, 1.34e5)
    ax.set_xlabel("训练步数"); ax.set_ylabel("val 段 N_eff（选型准绳）")
    ax.legend(loc="lower right", framealpha=0.95, ncol=2)
    ax.set_title("(a) 20 个 run 的 val N_eff 轨迹：多数在首 1–3 个\n评估周期即触顶",
                 fontsize=12, color=INK)
    ax = axes[1]
    run = "e1_joint_s0"
    ax.plot(tr[run]["steps"], tr[run]["train_nll"], "-o", color=PINE, ms=4.4, lw=1.8,
            label="train NLL")
    ax.plot(tr[run]["steps"], tr[run]["val_nll"], "--s", color=SADDLE, ms=4.4, lw=1.8,
            label="val NLL")
    best = tr[run]["best_step"]; total = tr[run]["steps_total"]
    ax.axvline(best, color=VIOLET, ls=":", lw=1.8)
    ax.axvline(total, color=GREY, ls=":", lw=1.8)
    ax.text(best + 400, 0.975, f"选型点 best_step = {best}\n（准绳并列最优中最早到达者）",
            fontsize=10.5, color=VIOLET, linespacing=1.45, va="top", ha="left", bbox=NOTE_BBOX)
    ax.text(total - 400, 0.795,
            f"停机 step = {total:,}\n被丢弃的 {total - best:,} 步：train NLL\n"
            f"从 {tr[run]['train_nll'][2]:.3f} 继续降到 {tr[run]['train_nll'][-1]:.3f}",
            fontsize=10.5, color=GREY, ha="right", va="top", linespacing=1.45, bbox=NOTE_BBOX)
    ax.set_xlabel("训练步数"); ax.set_ylabel("Poisson NLL")
    ax.set_ylim(0.585, 1.00)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_title("(b) e1_joint_s0 的 train/val NLL 曲线", fontsize=12, color=INK)
    fig.suptitle("图 5 · 训练动力学：为什么 20 个 run 都提前停机", fontsize=13, color=INK, weight="bold")
    fig.subplots_adjust(wspace=0.26, top=0.80)
    return fig


# ------------------------------------------------------------------ 图 6（20 个 run 的读数）
def fig06_neff_and_err():
    rows = NUMBERS["n_eff_rows"]
    st0 = [r for r in rows if r["station"] == 0]
    order = []
    for key in GROUP_ORDER:
        for r in sorted([x for x in st0 if GROUP_OF_RUN[x["run_id"]] == key],
                        key=lambda x: int(x["seed"])):
            order.append((key, r))
    x = np.arange(len(order))
    labels = [f"s{r['seed']}" for _, r in order]
    bounds = []
    start = 0
    for key in GROUP_ORDER:
        n = len(RUNS_BY_GROUP[key])
        bounds.append((key, start, start + n - 1))
        start += n
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.2), sharex=True)
    ax = axes[0]
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.7)
    for gi, (key, i0, i1) in enumerate(bounds):
        if gi:
            ax.axvline(i0 - 0.5, color="#BBBBBB", ls=":", lw=1.0)
        ax.text((i0 + i1) / 2, 1.92e5, GROUP_LABEL[key], ha="center", fontsize=11, color=INK,
                bbox=NOTE_BBOX)
        for i in range(i0, i1 + 1):
            v = float(order[i][1]["n_eff"]); flag = order[i][1]["n_eff_flag"]
            over = flag == "overflow"
            ax.plot(i, v, marker=STYLE_ENC[key]["marker"], ms=9,
                    mfc="none" if over else STYLE_ENC[key]["color"],
                    mec=STYLE_ENC[key]["color"], mew=2.0, ls="none")
    ax.text(0.4, 1.075e5, "空心 = 上溢（域顶截断）", fontsize=10.5, color=SADDLE, bbox=NOTE_BBOX)
    idx = [i for i, (_, r) in enumerate(order) if r["n_eff_flag"] != "overflow"][0]
    ax.annotate(f"{float(order[idx][1]['n_eff']):,.1f}（未触顶）", (idx, float(order[idx][1]["n_eff"])),
                textcoords="offset points", xytext=(-6, 22), ha="center", fontsize=10.5,
                color=STYLE_ENC[order[idx][0]]["color"], weight="bold", bbox=NOTE_BBOX)
    ax.set_yscale("log"); ax.set_ylim(2.4e4, 2.6e5)
    ax.set_ylabel("N_eff（log）")
    ax.set_title("图 6 · 20 个 run 的 N_eff 与形状误差：19 个点贴在域顶，1 个落在域内",
                 fontsize=13, color=INK, weight="bold")
    ax = axes[1]
    floor = NUMBERS["e0"]["points"][-1]["median"]
    const = NUMBERS["zero_information"]["const_median"]
    ax.axhline(const, color=VIOLET, ls="-.", lw=1.7)
    ax.axhline(floor, color=SADDLE, ls="--", lw=1.7)
    for gi, (key, i0, i1) in enumerate(bounds):
        if gi:
            ax.axvline(i0 - 0.5, color="#BBBBBB", ls=":", lw=1.0)
        e = STYLE_ENC[key]
        ax.bar(x[i0:i1 + 1], [float(order[i][1]["err_median"]) for i in range(i0, i1 + 1)],
               color=e["color"], alpha=0.92, hatch=e["hatch"], edgecolor="white", linewidth=0.6,
               width=0.68, label=GROUP_LABEL[key])
        for i in range(i0, i1 + 1):
            ax.text(i, float(order[i][1]["err_median"]) + 0.008, f"{float(order[i][1]['err_median']):.3f}",
                    ha="center", va="bottom", rotation=90, fontsize=10.5, color=INK,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none",
                              alpha=0.92))
    handles, labels_ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=VIOLET, ls="-.", lw=1.7,
                       label=f"零信息常数解 {const:.6f}"),
                Line2D([], [], color=SADDLE, ls="--", lw=1.7,
                       label=f"1e5 档裸误差 {floor:.6f}")]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, 0.44)
    ax.set_ylabel("test err 中位数（M1）")
    ax.legend(handles=handles, loc="upper left", ncol=3, framealpha=0.95, fontsize=10.5,
              title="阴影线 = 档位（非颜色编码）")
    axes[0].set_xticks(x); axes[0].set_xticklabels([])
    fig.subplots_adjust(hspace=0.12, top=0.90)
    return fig


# ------------------------------------------------------------------ 图 7
def fig07_h3_pairing():
    rows = {r["run_id"]: r for r in NUMBERS["n_eff_rows"] if r["station"] == 0}
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    xs = np.arange(5)
    for i in xs:
        j = rows[f"e1_joint_s{i}"]; d = rows[f"e1_indep_s{i}"]
        ax.plot([i, i], [d["n_eff"], j["n_eff"]], color=GREY, lw=1.8, zorder=1)
        ax.plot(i - 0.12, d["n_eff"], "s", ms=12, color=TIER_COLORS[30000],
                mfc=TIER_COLORS[30000] if not d["n_eff_flag"] else "none", mew=2.0, zorder=2)
        ax.plot(i + 0.12, j["n_eff"], "o", ms=12, color=PINE,
                mfc=PINE if not j["n_eff_flag"] else "none", mew=2.0, zorder=2)
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.7)
    ax.text(-0.42, 1.008e5, "报告域顶 $10^5$：两臂 10 个 N_eff 全部落在这里（空心 = 上溢标记）",
            fontsize=10.5, color=SADDLE, va="bottom", bbox=NOTE_BBOX)
    ax.plot([], [], "s", ms=11, color=TIER_COLORS[30000], label="independent（6 个单投影网）")
    ax.plot([], [], "o", ms=11, color=PINE, label="joint（1 个 6 进 6 出网）")
    ax.set_xticks(xs); ax.set_xticklabels([f"种子 {s}" for s in SEEDS])
    ax.set_ylabel("test N_eff（站 0，@ N_low = 1e4）")
    ax.set_ylim(4e4, 1.40e5); ax.set_xlim(-0.5, 4.5)
    ax.set_title("图 7 · H3 的 5 对同种子比较：逐对连线长度为零", fontsize=13, color=INK, weight="bold")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), framealpha=0.95)
    ax.text(0.0, 0.03,
            "判定表达式输入：joint_wins = 5/5 → 结论“成立”；但 n_ties = 5、n_strict_joint_wins = 0、\n"
            "tied_at_domain_edge = true —— “联合严格更优”并未被观测到。",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FAF8F5", edgecolor=SADDLE, linewidth=1.1))
    return fig


# ------------------------------------------------------------------ 图 8a（H8 达标率）
def fig08a_h8_passrate():
    gates = NUMBERS["gates"]
    obs = gates["H8"]["per_observable"]
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    y = np.arange(len(obs))[::-1]
    rates = [o["pass_rate"] for o in obs]
    for yi, o in zip(y, obs):
        hatch = "//" if o["pass_rate"] == 0 else ("xx" if o["pass_rate"] >= 0.8 else "\\\\")
        color = PINE if o["pass_rate"] >= 0.8 else (SADDLE if o["pass_rate"] > 0 else "#9A9A9A")
        ax.barh(yi, o["pass_rate"], color=color, height=0.66, hatch=hatch, edgecolor="white",
                linewidth=0.8)
        ax.text(max(o["pass_rate"], 0.0) + 0.018, yi, f"{o['pass_rate'] * 100:.1f}%  ({o['n_pass']}/150)",
                va="center", fontsize=10.5, color=INK, bbox=NOTE_BBOX)
    ax.axvline(0.50, color=SADDLE, ls="--", lw=1.6)
    ax.axvline(0.80, color=PINE, ls="--", lw=1.6)
    ax.text(0.505, len(obs) - 0.35, "不成立阈值 50%", fontsize=10.5, color=SADDLE, va="top",
            bbox=NOTE_BBOX)
    ax.text(0.808, len(obs) - 0.35, "成立阈值 80%", fontsize=10.5, color=PINE, va="top",
            bbox=NOTE_BBOX)
    ax.set_yticks(y); ax.set_yticklabels([o["name"] for o in obs], fontsize=10.5)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("达标率（13 项观测量，容差线按每束 N_eff 映射到 1e5 档）")
    ax.set_title(f"图 8a · H8：13 项物理量整体达标率 {gates['H8']['pass_rate']:.4f} ≤ 0.50 → 不成立",
                 fontsize=13, color=INK, weight="bold")
    ax.legend(handles=[
        Patch(facecolor=PINE, hatch="xx", edgecolor="white", label="全部通过（≥ 80%）"),
        Patch(facecolor=SADDLE, hatch="\\\\", edgecolor="white", label="部分通过（0–80%）"),
        Patch(facecolor="#9A9A9A", hatch="//", edgecolor="white", label="全不通过（0/150）"),
    ], loc="lower right", framealpha=0.95)
    return fig


# ------------------------------------------------------------------ 图 8b（暗桶 M2）
def fig08b_dark_buckets():
    db = NUMBERS["dark_buckets"]
    buckets = ["B0", "B1", "B2", "B3"]
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    x = np.arange(4)
    bare = [db[b]["bare_median"] for b in buckets]
    model = [db[b]["model_median"] for b in buckets]
    ax.bar(x - 0.19, bare, width=0.36, color=GREY, hatch="///", edgecolor="white", linewidth=0.8,
           label="裸输入（1e4 子样本直出，斜纹）")
    ax.bar(x + 0.19, model, width=0.36, color=PINE, edgecolor="white", linewidth=0.8,
           label="模型（joint，5 种子中位，实心）")
    for xi, bucket in enumerate(buckets):
        b, m = db[bucket]["bare_median"], db[bucket]["model_median"]
        ax.text(xi - 0.19, b + 0.08, f"{b:.3f}", ha="center", fontsize=10.5, color=INK)
        ax.text(xi + 0.19, m + 0.08, f"{m:.3f}", ha="center", fontsize=10.5, color=INK)
        imp = db[bucket]["relative_improvement"]
        txt = "相对改进无定义" if imp is None else f"相对改进 {imp * 100:.1f}%"
        ax.text(xi, -0.85, txt, ha="center", fontsize=10.5,
                color=SADDLE if imp is None else PINE, bbox=NOTE_BBOX)
    ax.text(1.5, -1.55,
            "B0 的裸输入结构性恒零：子样本 ⊂ 全量 ⇒ 参照零格在子样本中亦为零，故该桶无“相对改进”可定义",
            ha="center", fontsize=10.5, color=SADDLE, bbox=NOTE_BBOX)
    ax.set_xticks(x)
    ax.set_xticklabels(["B0\n参照 < 0.5", "B1\n0.5 ≤ 参照 < 2", "B2\n2 ≤ 参照 < 10", "B3\n参照 ≥ 10"],
                       fontsize=10.5)
    ax.set_ylim(-2.0, 5.6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("M2 分桶相对 RMSE 中位数")
    ax.set_title("图 8b · 暗桶 M2：裸输入 vs 模型（单独报告，不并入达标率）", fontsize=13,
                 color=INK, weight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.subplots_adjust(bottom=0.16, top=0.90)
    return fig


# ------------------------------------------------------------------ 图 9a（零信息对照：误差分布）
def _zero_info_rows():
    z = NUMBERS["zero_information"]
    return z, [f"{GROUP_OF_RUN[r]}|{r}" for r in ALL_RUNS]


def fig09a_zero_info_errs():
    z = NUMBERS["zero_information"]
    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    ypos = []
    y = 0.0
    group_bands = []
    for key in GROUP_ORDER:
        i0 = y
        for run in RUNS_BY_GROUP[key]:
            ypos.append((key, run, y))
            y += 1.0
        group_bands.append((key, i0, y - 1.0))
        y += 0.6
    const_y = y + 0.2
    data = [np.asarray(z["per_run"][run]["model_errs"]) for _, run, _ in ypos]
    bp = ax.boxplot(data, positions=[p for _, _, p in ypos], vert=False, widths=0.62,
                    patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.4))
    for i, (key, _, _) in enumerate(ypos):
        bp["boxes"][i].set(facecolor=STYLE_ENC[key]["color"], alpha=0.55,
                           edgecolor=STYLE_ENC[key]["color"], hatch=STYLE_ENC[key]["hatch"])
    for key, i0, i1 in group_bands:
        ax.axhspan(i0 - 0.5, i1 + 0.5, color=STYLE_ENC[key]["color"], alpha=0.06, zorder=0)
        ax.text(0.505, (i0 + i1) / 2, GROUP_LABEL[key], fontsize=11, color=INK, va="center",
                ha="right", bbox=NOTE_BBOX)
        if key != GROUP_ORDER[0]:
            ax.axhline(i0 - 0.8, color="#CCCCCC", ls=":", lw=1.0)
    const = np.asarray(z["const_errs"])
    ax.plot(const, np.full(len(const), const_y) + np.linspace(-0.24, 0.24, len(const)),
            ".", color=VIOLET, ms=3.0, alpha=0.75)
    ax.plot([np.median(const)] * 2, [const_y - 0.42, const_y + 0.42], color=VIOLET, lw=3.0)
    ax.axhline(const_y - 0.75, color=VIOLET, ls=":", lw=1.0)
    ax.axvline(NUMBERS["e0"]["points"][-1]["median"], color=SADDLE, ls="--", lw=1.6)
    ax.text(NUMBERS["e0"]["points"][-1]["median"] + 0.004, -2.4,
            "1e5 档裸误差 0.285636（上溢判据线）", fontsize=10.5, color=SADDLE, va="top",
            bbox=NOTE_BBOX)
    ax.set_yticks([p for _, _, p in ypos] + [const_y])
    ax.set_yticklabels([f"s{NUMS_SEED[run]}" for _, run, _ in ypos] + ["均值束（零信息常数解）"])
    ax.get_yticklabels()[-1].set_color(VIOLET)
    ax.set_ylim(const_y + 0.9, -3.0)
    ax.set_xlim(0.05, 0.52)
    ax.set_xlabel("逐束 M1 误差（150 个 test 束）")
    ax.set_title("图 9a · 逐束误差分布：20 个 run 的箱体整体位于常数解右侧", fontsize=13,
                 color=INK, weight="bold")
    fit = f"常数解中位 {z['const_median']:.6f}；20 个 run 的中位区间 0.2218–0.2898"
    ax.text(0.052, const_y + 0.55, fit, fontsize=10.5, color=VIOLET, va="bottom", bbox=NOTE_BBOX)
    return fig


# ------------------------------------------------------------------ 图 9b（零信息对照：胜出束数）
def fig09b_zero_info_wins():
    z = NUMBERS["zero_information"]
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    ypos = []
    y = 0.0
    bands = []
    for key in GROUP_ORDER:
        i0 = y
        for run in [r for r in ALL_RUNS if GROUP_OF_RUN[r] == key]:
            ypos.append((key, run, y))
            y += 1.0
        bands.append((key, i0, y - 1.0))
        y += 0.6
    wins = [z["per_run"][run]["n_wins_vs_const"] for _, run, _ in ypos]
    ratios = [z["per_run"][run]["median_ratio_model_over_const"] for _, run, _ in ypos]
    ax.barh([p for _, _, p in ypos], wins, height=0.62, color=PINE, alpha=0.9,
            label="模型优于常数解的束数（共 150）")
    ax.axvline(75, color=INK, ls=":", lw=1.5)
    ax.text(75, -2.35, "75/150 = 50% 平局线", fontsize=10.5, color=INK, ha="center", bbox=NOTE_BBOX)
    for key, i0, i1 in bands:
        ax.axhspan(i0 - 0.5, i1 + 0.5, color=STYLE_ENC[key]["color"], alpha=0.06, zorder=0)
        ax.text(298, (i0 + i1) / 2, GROUP_LABEL[key], fontsize=11, color=INK, va="center",
                ha="right", bbox=NOTE_BBOX)
        if key != GROUP_ORDER[0]:
            ax.axhline(i0 - 0.8, color="#CCCCCC", ls=":", lw=1.0)
    for i, (w, rt) in enumerate(zip(wins, ratios)):
        ax.text(w + 3.0, ypos[i][2], f"{w}/150 · 中位误差比 {rt:.2f}", va="center", fontsize=10.5,
                color=INK, bbox=NOTE_BBOX)
    ax.set_yticks([p for _, _, p in ypos])
    ax.set_yticklabels([f"s{NUMS_SEED[run]}" for _, run, _ in ypos])
    ax.set_ylim(ypos[-1][2] + 1.2, -3.0)
    ax.set_xlim(0, 300)
    ax.set_xlabel("胜出束数（共 150 束）")
    ax.set_title("图 9b · 胜出束数与逐束误差比：全部 20 个 run 都在 50% 平局线以下",
                 fontsize=13, color=INK, weight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    return fig


# ------------------------------------------------------------------ 图 10
def fig10_h7_cross_station():
    rows = {(r["run_id"], r["station"]): r for r in NUMBERS["n_eff_rows"]}
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    xs = np.arange(5)
    s0 = [rows[("e1_joint_s%d" % i, 0)]["n_eff"] for i in range(5)]
    s1 = [rows[("e1_joint_s%d" % i, 1)]["n_eff"] for i in range(5)]
    e1 = [rows[("e1_joint_s%d" % i, 1)]["err_median"] for i in range(5)]
    for i in xs:
        ax.plot([i - 0.12, i + 0.12], [s0[i], s1[i]], color=GREY, lw=1.8)
        ax.plot(i - 0.12, s0[i], "o", ms=12, mfc=PINE, mec=PINE)
        ax.plot(i + 0.12, s1[i], "D", ms=11, mfc="none", mec=TIER_COLORS[100000], mew=2.1)
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.7)
    ax.text(-0.44, 1.004e5, "域顶 $10^5$：站 0 与站 1 的 10 个 N_eff 全部触顶（空心菱形 = 站 1 上溢）",
            fontsize=10.5, color=SADDLE, va="bottom", bbox=NOTE_BBOX)
    for i, v in enumerate(e1):
        ax.text(i + 0.17, 0.85e5, f"站 1 err {v:.4f}", fontsize=10.5,
                color=TIER_COLORS[100000], ha="left", va="top", bbox=NOTE_BBOX)
    ax.plot([], [], "o", ms=11, color=PINE, label="站 0（FODO 入口）的 N_eff")
    ax.plot([], [], "D", ms=10, mfc="none", mec=TIER_COLORS[100000], mew=2.1,
            label="站 1（FODO 出口）的 N_eff")
    ax.set_xticks(xs); ax.set_xticklabels([f"种子 {s}" for s in SEEDS])
    ax.set_ylim(0.55e5, 1.34e5); ax.set_xlim(-0.5, 4.95)
    ax.set_ylabel("test N_eff（获胜变体 joint，5 个 ckpt 直接跨站应用）")
    ax.set_title("图 10 · H7 跨站泛化：ρ 恰为 1.0，但两侧同时上溢使该结论不可辨识",
                 fontsize=13, color=INK, weight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.20), framealpha=0.95)
    ax.text(0.0, 0.025,
            "判定表达式输入：ρ = N_eff(站1)/N_eff(站0) = 1.0（种子中位数之比 rho_median = 1.0），\n"
            "但 reason_code = n_eff_flag_present（两侧均携带 overflow）→ 按预注册规则判“不可判定”。\n"
            "站 1 的误差中位数（0.2087–0.2739）与站 0 同量级，原始证据指向无明显跨站衰减。",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FAF8F5", edgecolor=PINE, linewidth=1.1))
    return fig


def build_figures() -> list[tuple[str, object]]:
    """按文档出现顺序返回 (文件名, Figure)。审查脚本也调用本函数做独立复核。"""
    style()
    return [
        ("fig01_pipeline.png", fig01_pipeline()),
        ("fig02_lattice.png", fig02_lattice()),
        ("fig03_e0_curve.png", fig03_e0_curve()),
        ("fig04_amplification.png", fig04_amplification()),
        ("fig05_training_dynamics.png", fig05_training_dynamics()),
        ("fig06_neff_and_err.png", fig06_neff_and_err()),
        ("fig07_h3_pairing.png", fig07_h3_pairing()),
        ("fig08a_h8_passrate.png", fig08a_h8_passrate()),
        ("fig08b_dark_buckets.png", fig08b_dark_buckets()),
        ("fig09a_zero_info_errs.png", fig09a_zero_info_errs()),
        ("fig09b_zero_info_wins.png", fig09b_zero_info_wins()),
        ("fig10_h7_cross_station.png", fig10_h7_cross_station()),
    ]


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    METRICS.clear()
    print(f"渲染图件（硬约束：{COLUMN_PX} px 栏宽下最小屏幕字号 ≥ {MIN_SCREEN_PX} px）：")
    for name, fig in build_figures():
        save(fig, name)
    METRICS_OUT.write_text(json.dumps({
        "column_px": COLUMN_PX,
        "min_screen_px_required": MIN_SCREEN_PX,
        "palette": PALETTE,
        "noncolor_encodings": NONCOLOR_ENC,
        "figures": METRICS,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [m for m in METRICS if m["min_screen_px"] < MIN_SCREEN_PX]
    ov = sum(len(m["pixel_overlaps"]) for m in METRICS)
    pressed = sum(m["text_over_nontext_px_100dpi"] for m in METRICS)
    print(f"\n汇总：{len(METRICS)} 张图；最小屏上字号 {min(m['min_screen_px'] for m in METRICS):.2f} px；"
          f"未达标图 {len(bad)} 张；像素级重叠 {ov} 处；文字压非文字墨迹 {pressed} px(100dpi)")
    print(f"度量落盘：{METRICS_OUT}")


if __name__ == "__main__":
    main()
