"""生成 line2 Phase 1 阶段报告的 10 张图（300 dpi，PNG 落 reports/figs/）。

运行：python studies/line2_particle_sr/reports/tools/make_phase1_figures.py
只读 studies/line2_particle_sr/{data,results} 与同目录 phase1_numbers.json；只写 reports/figs/。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.line2.data_pipeline.projections import normalize_axis  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
NUMBERS = json.loads((HERE / "phase1_numbers.json").read_text(encoding="utf-8"))
LINE2 = ROOT / "studies" / "line2_particle_sr"
RESULTS = LINE2 / "results"
DATA = LINE2 / "data" / "line2_ds_v1"

CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(CJK)
CJK_NAME = fm.FontProperties(fname=CJK).get_name()

PINE = "#2D5A4A"
SADDLE = "#8B4513"
INK = "#1A1A1A"
GREY = "#666666"
TIER_COLORS = {1000: "#7F7F7F", 3000: "#2B6CB0", 10000: "#2D5A4A", 30000: "#B7791F", 100000: "#A02020"}
TIER_LABEL = {1000: r"$1\times10^3$", 3000: r"$3\times10^3$", 10000: r"$1\times10^4$", 30000: r"$3\times10^4$", 100000: r"$1\times10^5$"}
SEEDS = [100, 101, 102, 103, 104]
JOINT_RUNS = [f"e1_joint_s{i}" for i in range(5)]
INDEP_RUNS = [f"e1_indep_s{i}" for i in range(5)]
E2_RUNS = [f"e2_joint_n{t}_s{i}" for t in (3000, 100000) for i in range(5)]
ALL_RUNS = JOINT_RUNS + INDEP_RUNS + E2_RUNS
ROW_BY_RUN = {r["run_id"] + "::" + str(r["station"]): r for r in NUMBERS["n_eff_rows"]}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": CJK_NAME,
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
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


def save(fig, name: str) -> None:
    path = FIGS / name
    fig.canvas.draw()
    report = text_overlaps(fig)
    if report:
        print(f"  !! {name} 文本重叠 {len(report)} 处：")
        for a, b, frac in report[:8]:
            print(f"     {frac*100:4.0f}%  {a[:46]!r}  <->  {b[:46]!r}")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  wrote {name} ({path.stat().st_size} bytes)")


def text_overlaps(fig, min_frac: float = 0.10):
    """列出相互重叠面积超过自身面积 min_frac 的文本对（清晰度自检）。

    只计入实际会被绘出的文本：超出坐标轴视图区间（视图外）的刻度标签不参与，
    否则 log 轴在视图外仍持有位置的刻度文本会产生假警报。
    """
    renderer = fig.canvas.get_renderer()
    all_tick_ids: set[int] = set()
    drawn_tick_ids: set[int] = set()
    for ax in fig.axes:
        for axis, lims in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lims), max(lims)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                for label in (tick.label1, tick.label2):
                    all_tick_ids.add(id(label))
                loc = tick.get_loc()
                if loc is not None and (lo - 1e-9) <= float(loc) <= (hi + 1e-9):
                    drawn_tick_ids.add(id(tick.label1))
                    drawn_tick_ids.add(id(tick.label2))
    items = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        if id(t) in all_tick_ids and id(t) not in drawn_tick_ids:
            continue
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width <= 2 or bb.height <= 2:
            continue
        rot = abs(float(t.get_rotation())) % 180.0
        if 15.0 < rot < 165.0:
            # 旋转文本的轴对齐包围盒远大于字面，按中心收缩近似字面足迹
            cx, cy = bb.x0 + bb.width / 2, bb.y0 + bb.height / 2
            bb = matplotlib.transforms.Bbox.from_extents(
                cx - bb.width * 0.3, cy - bb.height * 0.3,
                cx + bb.width * 0.3, cy + bb.height * 0.3,
            )
        items.append((t.get_text().replace("\n", "⏎")[:60], bb))
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ox = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
            oy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
            if ox <= 0 or oy <= 0:
                continue
            inter = ox * oy
            smaller = min(a.width * a.height, b.width * b.height)
            if smaller <= 0:
                continue
            frac = inter / smaller
            if frac >= min_frac:
                out.append((items[i][0], items[j][0], frac))
    return sorted(out, key=lambda x: -x[2])


def box(ax, x, y, w, h, text, fc="#FAF8F5", ec=PINE, fs=9.5, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            linespacing=1.5)


def arrow(ax, p0, p1, color=PINE, style="-|>", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- fig 01
def fig01_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    box(ax, 0.02, 0.42, 0.17, 0.20,
        "锚点粒子表\n2×10^5 行 × 6 列\n(x, x′, y, y′, t, p)\n束群 1000（抖动 ×FODO）", fc="#EDE8DF")
    box(ax, 0.24, 0.42, 0.16, 0.20,
        "六投影分箱\n6 × 128 × 128\n计数直方图\n(int32，∑ = 粒子数)")
    arrow(ax, (0.19, 0.52), (0.24, 0.52))
    box(ax, 0.45, 0.70, 0.20, 0.20,
        "低计数输入（训练/评测入口）\n无放回抽 N_low ∈\n{$10^3$, 3×$10^3$, $10^4$, 3×$10^4$, $10^5$}\n每档 k = 20 个子样本",
        fc="#EDE8DF", ec=TIER_COLORS[3000])
    box(ax, 0.45, 0.14, 0.20, 0.20,
        "同束同站的目标\n2×10^5 全量直方图\n= 参照 R\n（不是另一台机器拍的）",
        fc="#EDE8DF", ec=TIER_COLORS[100000])
    arrow(ax, (0.40, 0.55), (0.45, 0.76)); arrow(ax, (0.40, 0.49), (0.45, 0.28))
    box(ax, 0.70, 0.42, 0.17, 0.20,
        "网络\n6 × 128² → 6 × 128²\n（联合 FiLM U-Net，\n或 6 个独立子网）",
        fc="#FFF6EC", ec=SADDLE)
    arrow(ax, (0.65, 0.80), (0.785, 0.62))
    box(ax, 0.89, 0.42, 0.10, 0.20, "误差 →\nN_eff", fc="#FFF6EC", ec=SADDLE)
    arrow(ax, (0.87, 0.52), (0.89, 0.52))
    ax.text(0.5, 0.975, "图 1 · Phase 1 任务与数据管线：粒子表 → 六投影图 → 低计数输入 → 2×$10^5$ 目标",
            ha="center", va="top", fontsize=12.5, color=INK, weight="bold")
    ax.text(0.5, 0.045,
            "关键约定：输入与输出都是 128×128 像素的同一张图——本阶段的“超分 / 放大”指的是等效粒子数\n"
            "（N_eff / N_low），不是像素分辨率。像素分辨率的扩展（4096² 等）属 Phase 3，不在本阶段范围内。",
            ha="center", va="bottom", fontsize=9.7, color=SADDLE, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAF8F5", edgecolor=SADDLE, linewidth=1.0))
    save(fig, "fig01_pipeline.png")


# ---------------------------------------------------------------- fig 02
def fig02_lattice() -> None:
    beam = "beam_0006"
    panels = [(0, 0, "(x, x′) 水平面", SADDLE), (0, 1, "(y, y′) 垂直面", PINE)]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))
    stats = {}
    with h5py.File(DATA / "val.h5", "r") as fh:
        for row, station in enumerate((0, 1)):
            grp = fh[beam][f"station{station}"]
            edges = np.asarray(grp["target_full"].attrs["bin_edges"], dtype=np.float64)
            img = grp["target_full"][:]
            for col, (gi, ai, aj) in enumerate(((0, 0, 1), (1, 2, 3))):
                ax = axes[row][col]
                sub = img[gi].astype(np.float64)
                # 存储布局：axis0 = 后一坐标、axis1 = 前一坐标
                e_j = normalize_axis(edges[aj], edges[aj])
                e_i = normalize_axis(edges[ai], edges[ai])
                shown = sub
                mesh = ax.pcolormesh(e_i, e_j, np.maximum(shown, 0.5), shading="auto",
                                     cmap="magma", norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=shown.max()))
                ui = 0.5 * (edges[ai][:-1] + edges[ai][1:]); uj = 0.5 * (edges[aj][:-1] + edges[aj][1:])
                pi = sub.sum(axis=1) / sub.sum(); pj = sub.sum(axis=0) / sub.sum()
                mi = float((pi * ui).sum()); mj = float((pj * uj).sum())
                si = float(np.sqrt((pi * (ui - mi) ** 2).sum())); sj = float(np.sqrt((pj * (uj - mj) ** 2).sum()))
                stats[(station, ai)] = (si, sj)
                ax.set_title(f"站 {station} · {panels[col][2]}", fontsize=11.5)
                ax.set_xlabel(f"归一化 {('x','xp','y','yp')[ai]}")
                ax.set_ylabel(f"归一化 {('x','xp','y','yp')[aj]}")
                ax.grid(False)
                ax.text(0.03, 0.97,
                        f"物理 RMS：σ = {si:.3e} ({('x','xp','y','yp')[ai]})\n"
                        f"             {sj:.3e} ({('x','xp','y','yp')[aj]})",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9.0, color="white",
                        bbox=dict(boxstyle="round,pad=0.32", facecolor="#00000088", edgecolor="none"))
                if row == 0 and col == 0:
                    mesh0 = mesh
    cb = fig.colorbar(mesh0, ax=[axes[r][c] for r in range(2) for c in range(2)],
                      fraction=0.024, pad=0.02)
    cb.set_label("计数（log 色标）", fontsize=9.5)
    cb.ax.tick_params(labelsize=9.0)
    fig.suptitle("图 2 · 同一束在 FODO 两站的相空间分布并不相同", fontsize=13, color=INK, weight="bold", y=0.995)
    d = {k: (v[0], v[1]) for k, v in stats.items()}
    fig.text(0.5, 0.005,
             "轴向已各自归一到 [-1, 1] 便于看形状；FODO 传输同时改变两个横向面的形状与物理尺度——"
             f"例如 σ_x′ 由站 0 的 {d[(0,0)][1]:.3e} 变为站 1 的 {d[(1,0)][1]:.3e}。\n"
             "因此站 1 的输入图不是站 0 那张图的平移或缩放，跨站泛化必须单独检验（H7）。",
             ha="center", va="bottom", fontsize=9.7, color=INK, linespacing=1.55)
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.28, wspace=0.22)
    save(fig, "fig02_lattice.png")


# ---------------------------------------------------------------- fig 03
def fig03_e0_curve() -> None:
    e0 = NUMBERS["e0"]
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    ns = [p["n_low"] for p in e0["points"]]
    med = np.array([p["median"] for p in e0["points"]])
    lo = np.array([p["iqr_lo"] for p in e0["points"]]); hi = np.array([p["iqr_hi"] for p in e0["points"]])
    ax.fill_between(ns, lo, hi, color=PINE, alpha=0.16, label="IQR 带（跨 300 束 × 20 子样本）")
    ax.plot(ns, med, "o-", color=PINE, lw=2.0, ms=6, label="M1 中位数（E0 裸误差–N 曲线）")
    floor = e0["reference_noise_floor"]
    ax.axhline(floor, color=SADDLE, ls="--", lw=1.4,
               label=f"参照噪声地板 {floor:.6f}（1e5 档配对待出）")
    rows = NUMBERS["n_eff_rows"]
    for tier in (3000, 10000, 100000):
        pts = [r for r in rows if r["n_low"] == tier and r["station"] == 0]
        xs = np.full(len(pts), tier) * np.exp(np.linspace(-0.09, 0.09, len(pts)))
        ax.plot(xs, [r["err_median"] for r in pts], "D", color=TIER_COLORS[tier], ms=5.5, alpha=0.9,
                label=f"{TIER_LABEL[tier]} 档 · 各 run 的模型 err 落点（n={len(pts)}）")
    ax.axvspan(1000, 100000, color="#E8E4DC", alpha=0.35, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("输入子样本粒子数 N_low（log）")
    ax.set_ylabel("六投影平均相对 RMSE（M1，log）")
    ax.set_title("图 3 · “尺子”：裸模拟的误差–粒子数曲线与噪声地板", fontsize=12.5, color=INK, weight="bold")
    ax.legend(loc="lower left", framealpha=0.94)
    ax.annotate("误差随粒子数近似幂律下降：\n1e3 → 1e4 档，4.030 → 1.246（约 1/3.2）",
                xy=(10000, 1.2455), xytext=(2100, 0.62), fontsize=9.3, color=INK,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0), linespacing=1.5)
    ax.annotate("地板以下：再多的裸粒子数\n也换不来可分辨的改善",
                xy=(70000, floor), xytext=(13500, 0.36), fontsize=9.3, color=SADDLE,
                arrowprops=dict(arrowstyle="->", color=SADDLE, lw=1.0), linespacing=1.5)
    ax.set_xlim(880, 160000)
    save(fig, "fig03_e0_curve.png")


# ---------------------------------------------------------------- fig 04
def fig04_amplification() -> None:
    rows = NUMBERS["n_eff_rows"]
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    xs = np.logspace(np.log10(2200), np.log10(145000), 200)
    ax.plot(xs, 1e5 / xs, ls="--", color=SADDLE, lw=1.5, label="域顶线 $N_{eff}=10^5$（= $10^5/N_{low}$）")
    ax.plot(xs, np.ones_like(xs), ls=":", color=GREY, lw=1.5, label="无放大线 $N_{eff}=N_{low}$（比值 = 1）")
    ax.axhspan(1, 10, color=PINE, alpha=0.10)
    ax.text(2600, 2.0, "可验证放大域（≤ 10×，`00` [S1] 声明的叙事上限）", fontsize=9.2, color=PINE)
    ax.axhspan(10, 60, color=SADDLE, alpha=0.07)
    ax.text(2600, 13.0, "超出 10× 声明边界", fontsize=9.2, color=SADDLE)

    series = [
        ("joint @ 站 0",
         [r for r in rows
          if r["station"] == 0 and (r["run_id"] in JOINT_RUNS
                                    or r["run_id"].startswith("e2_joint_n"))],
         PINE, "o"),
        ("independent @ 站 0",
         [r for r in rows if r["station"] == 0 and r["run_id"] in INDEP_RUNS],
         TIER_COLORS[30000], "s"),
    ]
    for label, rs, color, marker in series:
        by_tier: dict[int, list[float]] = {}
        for r in rs:
            by_tier.setdefault(r["n_low"], []).append(r["n_eff"] / r["n_low"])
        for tier in sorted(by_tier):
            vals = np.asarray(by_tier[tier])
            xs_t = np.full(len(vals), tier) * np.exp(np.linspace(-0.10, 0.10, len(vals)))
            overflow = np.isclose(vals, 1e5 / tier, rtol=1e-9)
            if (~overflow).any():
                ax.plot(xs_t[~overflow], vals[~overflow], marker=marker, ls="none", ms=9,
                        mfc=color, mec=color, label=f"{label} · {TIER_LABEL[tier]}（实心 = 域内真实插值）")
            if overflow.any():
                ax.plot(xs_t[overflow], vals[overflow], marker=marker, ls="none", ms=9,
                        mfc="none", mec=color, mew=1.9,
                        label=f"{label} · {TIER_LABEL[tier]}（空心 = 上溢 · 域顶）")
    ax.plot([3000, 10000, 100000], [1e5 / 3000, 1e5 / 10000, 1.0], "-", color=PINE, lw=1.2, alpha=0.55)
    ax.annotate("33.3× = $10^5$/$3\\times10^3$：由域顶构造得出，\n不是一次独立的放大测量（分辨力为零）",
                xy=(3000, 33.33), xytext=(4150, 20.0), fontsize=9.2, color=SADDLE, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=SADDLE, lw=1.0))
    ax.annotate("10× = $10^5$/$10^4$：同上",
                xy=(10000, 10.0), xytext=(14000, 5.2), fontsize=9.2, color=SADDLE, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=SADDLE, lw=1.0))
    ax.annotate("1e5 档：唯一出现非饱和点的档位\n（s2 的 98,724，未触顶）",
                xy=(100000, 1.0), xytext=(46000, 1.95), fontsize=9.2, color=PINE, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=PINE, lw=1.0))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("输入粒子数 $N_{low}$（log）")
    ax.set_ylabel("放大比 $N_{eff}/N_{low}$（log）")
    ax.set_title("图 4 · 三点放大主曲线：观测越稀疏，等效放大越大——但只有右端有分辨力",
                 fontsize=12.5, color=INK, weight="bold")
    ax.legend(loc="upper right", framealpha=0.95, ncol=1)
    ax.set_ylim(0.75, 90); ax.set_xlim(2200, 145000)
    save(fig, "fig04_amplification.png")


# ---------------------------------------------------------------- fig 05
def fig05_neff_and_err() -> None:
    rows = NUMBERS["n_eff_rows"]
    st0 = [r for r in rows if r["station"] == 0]
    order = [(3000, "e2_joint_n3000"), (10000, "e1_joint"), (10000, "e1_indep"),
             (100000, "e2_joint_n100000")]
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 9.0), sharex=True)
    labels, neffs, flags, errs, colors = [], [], [], [], []
    for tier, prefix in order:
        for r in sorted([x for x in st0 if x["n_low"] == tier and x["run_id"].startswith(prefix)],
                        key=lambda x: x["seed"]):
            labels.append(f"{r['run_id']}\n(seed {r['seed']})")
            neffs.append(r["n_eff"]); flags.append(r["n_eff_flag"]); errs.append(r["err_median"])
            colors.append(TIER_COLORS[tier])
    x = np.arange(len(labels))
    ax = axes[0]
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.4, label="报告域顶 $10^5$（上溢 = 达到可验证域顶）")
    for xi, (v, f, c) in enumerate(zip(neffs, flags, colors)):
        overflow = f == "overflow"
        ax.plot(xi, v, marker="o", ms=8.5, mfc="none" if overflow else c, mec=c, mew=1.9, ls="none")
        if not overflow:
            ax.annotate(f"{v:,.0f}（未触顶）", (xi, v), textcoords="offset points", xytext=(0, -30),
                        ha="center", fontsize=9.0, color=c, weight="bold")
    ax.plot([], [], "o", ms=8.5, mfc="none", mec=SADDLE, mew=1.9, label="空心 = 上溢（域顶截断，20 个 run 里的 19 个）")
    ax.plot([], [], "o", ms=8.5, mfc=SADDLE, mec=SADDLE, label="实心 = 域内真实插值（20 个 run 里的 1 个）")
    ax.set_yscale("log"); ax.set_ylim(2.6e4, 2.2e5)
    ax.set_ylabel("N_eff（log）")
    ax.set_title("图 5 · 20 个 run 的 N_eff 与形状误差：19 个点贴在域顶，1 个落在域内",
                 fontsize=12.5, color=INK, weight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    ax = axes[1]
    floor = NUMBERS["e0"]["points"][-1]["median"]
    const = NUMBERS["zero_information"]["const_median"]
    ax.axhline(const, color="#6A4C93", ls="-.", lw=1.4,
               label=f"零信息常数解（train 均值束）{const:.6f}")
    ax.axhline(floor, color=SADDLE, ls="--", lw=1.4, label=f"1e5 档裸误差 {floor:.6f}")
    ax.bar(x, errs, color=colors, alpha=0.88, width=0.66)
    for xi, v in enumerate(errs):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9.0, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9.0)
    ax.set_ylabel("test err 中位数（M1）")
    ax.set_ylim(0, 0.40)
    ax.legend(loc="upper left", framealpha=0.95)
    axes[0].set_xticks(x); axes[0].set_xticklabels([])
    fig.subplots_adjust(hspace=0.10, bottom=0.28)
    save(fig, "fig05_neff_and_err.png")


# ---------------------------------------------------------------- fig 06
def fig06_training_dynamics() -> None:
    tr = NUMBERS["training"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    ax = axes[0]
    groups = [
        (E2_RUNS[:5], TIER_COLORS[3000], r"joint @ $3\times10^3$（5 run）"),
        (JOINT_RUNS, PINE, r"joint @ $1\times10^4$（5 run）"),
        (INDEP_RUNS, TIER_COLORS[30000], r"independent @ $1\times10^4$（5 run）"),
        (E2_RUNS[5:], TIER_COLORS[100000], r"joint @ $1\times10^5$（5 run）"),
    ]
    for runs, color, label in groups:
        for k, run in enumerate(runs):
            ax.plot(tr[run]["steps"], tr[run]["val_n_eff"], color=color, lw=1.0, alpha=0.75,
                    label=label if k == 0 else None)
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.5, label="报告域顶 $10^5$")
    ax.text(1400, 1.005e5, "选型准绳 $N_{eff}$ 在此饱和 ⇒ 之后不再更新 best", fontsize=9.0,
            color=SADDLE, va="bottom")
    ax.set_ylim(2.6e4, 1.30e5)
    ax.set_xlabel("训练步数"); ax.set_ylabel("val 段 N_eff（选型准绳）")
    ax.legend(loc="lower right", fontsize=9.0, framealpha=0.95)
    ax.set_title("(a) 20 个 run 的 val N_eff 轨迹：多数在首 1–3 个\n评估周期即触顶",
                 fontsize=11.5, color=INK)
    ax = axes[1]
    run = "e1_joint_s0"
    ax.plot(tr[run]["steps"], tr[run]["train_nll"], "-o", color=PINE, ms=3.6, lw=1.6, label="train NLL")
    ax.plot(tr[run]["steps"], tr[run]["val_nll"], "-s", color=SADDLE, ms=3.6, lw=1.6, label="val NLL")
    best = tr[run]["best_step"]; total = tr[run]["steps_total"]
    ax.axvline(best, color="#6A4C93", ls=":", lw=1.6)
    ax.axvline(total, color=GREY, ls=":", lw=1.6)
    ax.text(best + 300, 0.845, f"选型点 best_step = {best}", fontsize=9.2, color="#6A4C93")
    ax.text(total - 300, 0.845, f"实际停机 step = {total:,}", fontsize=9.2, color=GREY, ha="right")
    ax.annotate("", xy=(total, 0.70), xytext=(best, 0.70),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text((best + total) / 2, 0.712, f"被丢弃的 {total-best:,} 步：train NLL 从 "
            f"{tr[run]['train_nll'][2]:.3f} 继续降到 {tr[run]['train_nll'][-1]:.3f}",
            fontsize=9.0, color=INK, ha="center")
    ax.set_xlabel("训练步数"); ax.set_ylabel("Poisson NLL")
    ax.set_ylim(0.58, 0.90)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_title("(b) e1_joint_s0 的 train/val NLL：选型停在最不充分的时刻\n（预算 50,000，实际只用 "
                 f"{100*total/50000:.0f}%）", fontsize=11.5, color=INK)
    fig.suptitle("图 6 · 训练动力学：为什么 20 个 run 都提前停机", fontsize=13, color=INK, weight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.24, top=0.82)
    save(fig, "fig06_training_dynamics.png")


# ---------------------------------------------------------------- fig 07
def fig07_h3_pairing() -> None:
    rows = {r["run_id"]: r for r in NUMBERS["n_eff_rows"] if r["station"] == 0}
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    xs = np.arange(5)
    j = [rows[f"e1_joint_s{i}"]["n_eff"] for i in range(5)]
    d = [rows[f"e1_indep_s{i}"]["n_eff"] for i in range(5)]
    ov_j = [rows[f"e1_joint_s{i}"]["n_eff_flag"] for i in range(5)]
    ov_d = [rows[f"e1_indep_s{i}"]["n_eff_flag"] for i in range(5)]
    for i in xs:
        ax.plot([i, i], [d[i], j[i]], color=GREY, lw=1.6, zorder=1)
        ax.plot(i - 0.11, d[i], "s", ms=11, color=TIER_COLORS[30000],
                mfc=TIER_COLORS[30000] if ov_d[i] else "none", mew=1.8, zorder=2)
        ax.plot(i + 0.11, j[i], "o", ms=11, color=PINE,
                mfc=PINE if ov_j[i] else "none", mew=1.8, zorder=2)
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.5)
    ax.text(-0.42, 1.012e5, "报告域顶 $10^5$：两臂 10 个 N_eff 全部落在这里（含空心的上溢标记）",
            fontsize=9.3, color=SADDLE)
    ax.plot([], [], "s", ms=10, color=TIER_COLORS[30000], label="independent（6 个单投影网）")
    ax.plot([], [], "o", ms=10, color=PINE, label="joint（1 个 6 进 6 出网）")
    ax.set_xticks(xs); ax.set_xticklabels([f"种子 {s}" for s in SEEDS])
    ax.set_ylabel("test N_eff（站 0，@ N_low=1e4）")
    ax.set_ylim(4e4, 1.35e5); ax.set_xlim(-0.5, 4.5)
    ax.set_title("图 7 · H3 的 5 对同种子比较：逐对连线长度为零", fontsize=12.5, color=INK, weight="bold")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), framealpha=0.95)
    ax.text(0.0, 0.03,
            "判定表达式输入：joint_wins = 5/5 → 结论“成立”；但 n_ties = 5、n_strict_joint_wins = 0、\n"
            "tied_at_domain_edge = true —— “联合严格更优”并未被观测到。",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.4, color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FAF8F5", edgecolor=SADDLE, linewidth=1.0))
    save(fig, "fig07_h3_pairing.png")


# ---------------------------------------------------------------- fig 08
def fig08_h8_and_dark() -> None:
    gates = NUMBERS["gates"]
    obs = gates["H8"]["per_observable"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    names = [o["name"] for o in obs]
    rates = [o["pass_rate"] for o in obs]
    y = np.arange(len(obs))[::-1]
    ax.barh(y, rates, color=[PINE if r >= 0.8 else (SADDLE if r > 0 else "#B0B0B0") for r in rates],
            height=0.62)
    ax.axvline(0.50, color=SADDLE, ls="--", lw=1.4, label="“不成立”阈值 50%")
    ax.axvline(0.80, color=PINE, ls="--", lw=1.4, label="“成立”阈值 80%")
    for yi, (r, o) in zip(y, zip(rates, obs)):
        ax.text(max(r, 0.0) + 0.015, yi, f"{r*100:.1f}%  ({o['n_pass']}/150)", va="center", fontsize=9.0, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.0)
    ax.set_xlim(0, 1.14); ax.set_xlabel("达标率（13 项观测量，容差线按每束 N_eff 映射到 1e5 档）")
    ax.set_title(f"(a) H8：整体达标率 {gates['H8']['pass_rate']:.4f} ≤ 0.50 → 不成立",
                 fontsize=11.5, color=INK)
    ax.legend(loc="lower right", framealpha=0.95)
    ax = axes[1]
    db = NUMBERS["dark_buckets"]
    buckets = ["B0", "B1", "B2", "B3"]
    x = np.arange(4)
    bare = [db[b]["bare_median"] for b in buckets]
    model = [db[b]["model_median"] for b in buckets]
    ax.bar(x - 0.19, bare, width=0.36, color=GREY, label="裸输入（1e4 子样本直出）")
    ax.bar(x + 0.19, model, width=0.36, color=PINE, label="模型（joint，5 种子中位）")
    for xi, bucket in enumerate(buckets):
        b, m = db[bucket]["bare_median"], db[bucket]["model_median"]
        ax.text(xi - 0.19, b + 0.07, f"{b:.3f}", ha="center", fontsize=9.0, color=INK)
        ax.text(xi + 0.19, m + 0.07, f"{m:.3f}", ha="center", fontsize=9.0, color=INK)
        imp = db[bucket]["relative_improvement"]
        txt = "相对改进无定义" if imp is None else f"相对改进 {imp*100:.1f}%"
        ax.text(xi, -0.80, txt, ha="center", fontsize=9.0, color=SADDLE if imp is None else PINE)
    ax.text(1.5, -1.5,
            "B0 的裸输入结构性恒零：子样本 ⊂ 全量 ⇒ 参照零格在子样本中亦为零，故该桶无“相对改进”可定义",
            ha="center", fontsize=9.0, color=SADDLE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B0\n参照 < 0.5", "B1\n0.5 ≤ 参照 < 2", "B2\n2 ≤ 参照 < 10", "B3\n参照 ≥ 10"], fontsize=9.2)
    ax.set_ylim(-1.9, 5.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("M2 分桶相对 RMSE 中位数")
    ax.set_title("(b) 暗桶 M2：裸输入 vs 模型（单独报告，不并入达标率）", fontsize=11.5, color=INK)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.suptitle("图 8 · H8 的 13 项达标率与暗桶分项：整体形状好 ≠ 每项细结构精度都够",
                 fontsize=13, color=INK, weight="bold", y=0.99)
    fig.subplots_adjust(wspace=0.42, bottom=0.16, top=0.88)
    save(fig, "fig08_h8_and_dark.png")


# ---------------------------------------------------------------- fig 09
def fig09_zero_info_control() -> None:
    z = NUMBERS["zero_information"]
    runs = ALL_RUNS
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 8.6), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ypos = np.arange(len(runs), dtype=float)

    def short(run: str) -> str:
        v = "joint" if "joint" in run else "independent"
        tier = "3e3" if "_n3000_" in run else ("1e5" if "_n100000_" in run else "1e4")
        return f"{v} · {tier} · seed {SEEDS[int(run[-1])] - 100}"

    ax = axes[0]
    data = [np.asarray(z["per_run"][r]["model_errs"]) for r in runs]
    bp = ax.boxplot(data, positions=ypos, vert=False, widths=0.62,
                    patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.2))
    for i, r in enumerate(runs):
        tier = 3000 if "_n3000_" in r else (100000 if "_n100000_" in r else 10000)
        bp["boxes"][i].set(facecolor=TIER_COLORS[tier], alpha=0.5, edgecolor=TIER_COLORS[tier])
    const = np.asarray(z["const_errs"])
    ax.plot(const, np.full(len(const), -1.15) + np.linspace(-0.22, 0.22, len(const)),
            ".", color="#6A4C93", ms=2.6, alpha=0.75)
    ax.plot([np.median(const)] * 2, [-1.5, -0.8], color="#6A4C93", lw=2.8)
    ax.set_yticks(list(ypos) + [-1.15])
    ax.set_yticklabels([short(r) for r in runs] + ["均值束（零信息常数解）"], fontsize=9.0)
    ax.get_yticklabels()[-1].set_color("#6A4C93")
    ax.axhline(-0.55, color="#6A4C93", ls=":", lw=1.0)
    ax.axvline(NUMBERS["e0"]["points"][-1]["median"], color=SADDLE, ls="--", lw=1.4)
    ax.text(NUMBERS["e0"]["points"][-1]["median"] + 0.004, -2.55,
            "1e5 档裸误差 0.285636（上溢判据线）", fontsize=9.0, color=SADDLE, va="top")
    ax.set_xlabel("逐束 M1 误差（150 个 test 束）")
    ax.set_xlim(0.05, 0.42)
    ax.set_ylim(-2.75, len(runs) - 0.3)
    ax.invert_yaxis()
    ax.set_title("(a) 逐束误差分布：20 个 run 的箱体整体位于常数解右侧", fontsize=11.5, color=INK)

    ax = axes[1]
    wins = [z["per_run"][r]["n_wins_vs_const"] for r in runs]
    ratios = [z["per_run"][r]["median_ratio_model_over_const"] for r in runs]
    ax.barh(ypos, wins, height=0.62, color=PINE, alpha=0.85,
            label="模型优于常数解的束数（/150）")
    ax.axvline(75, color=INK, ls=":", lw=1.3, label="50% 平局线（75/150）")
    for i, (w, rt) in enumerate(zip(wins, ratios)):
        ax.text(w + 2.0, i, f"{w}/150   ·   中位误差比 {rt:.2f}", va="center", fontsize=9.0, color=INK)
    ax.set_yticks(list(ypos) + [-1.15])
    ax.set_yticklabels([short(r) for r in runs] + ["均值束（零信息常数解）"], fontsize=9.0)
    ax.get_yticklabels()[-1].set_color("#6A4C93")
    ax.axhline(-0.55, color="#6A4C93", ls=":", lw=1.0)
    ax.set_xlim(0, 230)
    ax.set_ylim(-2.75, len(runs) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("胜出束数（共 150 束）")
    ax.set_title(f"(b) 胜出 {min(wins)}–{max(wins)}/150，中位误差比 {min(ratios):.2f}–{max(ratios):.2f}",
                 fontsize=11.5, color=INK)
    ax.text(75, -2.52, "75/150 = 50% 平局线", fontsize=9.0, color=INK, ha="center")
    fig.suptitle("图 9 · 核心对照：一个零信息常数解给出比全部 20 个 run 更低的误差",
                 fontsize=13, color=INK, weight="bold", y=0.985)
    fig.subplots_adjust(wspace=0.52, left=0.14, right=0.98, top=0.92, bottom=0.08)
    save(fig, "fig09_zero_info_control.png")


# ---------------------------------------------------------------- fig 10
def fig10_h7_cross_station() -> None:
    rows = {(r["run_id"], r["station"]): r for r in NUMBERS["n_eff_rows"]}
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    xs = np.arange(5)
    s0 = [rows[("e1_joint_s%d" % i, 0)]["n_eff"] for i in range(5)]
    s1 = [rows[("e1_joint_s%d" % i, 1)]["n_eff"] for i in range(5)]
    e1 = [rows[("e1_joint_s%d" % i, 1)]["err_median"] for i in range(5)]
    for i in xs:
        ax.plot([i - 0.12, i + 0.12], [s0[i], s1[i]], color=GREY, lw=1.6)
        ax.plot(i - 0.12, s0[i], "o", ms=11, mfc=PINE, mec=PINE)
        ax.plot(i + 0.12, s1[i], "D", ms=10, mfc="none", mec=TIER_COLORS[100000], mew=1.9)
    ax.axhline(1e5, color=SADDLE, ls="--", lw=1.5)
    ax.text(-0.44, 1.005e5, "域顶 $10^5$：站 0 与站 1 的 10 个 N_eff 全部触顶（空心菱形 = 站 1 的上溢标记）",
            fontsize=9.2, color=SADDLE, va="bottom")
    for i, v in enumerate(e1):
        ax.text(i + 0.16, 0.845e5, f"站 1 err {v:.4f}", fontsize=9.0, color=TIER_COLORS[100000],
                ha="left", va="top")
    ax.plot([], [], "o", ms=10, color=PINE, label="站 0（FODO 入口）的 N_eff")
    ax.plot([], [], "D", ms=9, mfc="none", mec=TIER_COLORS[100000], mew=1.9, label="站 1（FODO 出口）的 N_eff")
    ax.set_xticks(xs); ax.set_xticklabels([f"种子 {s}" for s in SEEDS])
    ax.set_ylim(0.55e5, 1.30e5); ax.set_xlim(-0.5, 4.9)
    ax.set_ylabel("test N_eff（获胜变体 joint，5 个 ckpt 直接跨站应用）")
    ax.set_title("图 10 · H7 跨站泛化：ρ 恰为 1.0，但两侧同时上溢使该结论不可辨识",
                 fontsize=12.5, color=INK, weight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.185), framealpha=0.95)
    ax.text(0.0, 0.025,
            "判定表达式输入：ρ = N_eff(站1)/N_eff(站0) = 1.0（种子中位数之比 rho_median = 1.0），\n"
            "但 reason_code = n_eff_flag_present（两侧均携带 overflow）→ 按预注册规则判“不可判定”。\n"
            "站 1 的误差中位数（0.2087–0.2739）与站 0 同量级，原始证据指向无明显跨站衰减。",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.3, color=INK, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FAF8F5", edgecolor=PINE, linewidth=1.0))
    save(fig, "fig10_h7_cross_station.png")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    style()
    print("渲染图件：")
    fig01_pipeline()
    fig02_lattice()
    fig03_e0_curve()
    fig04_amplification()
    fig05_neff_and_err()
    fig06_training_dynamics()
    fig07_h3_pairing()
    fig08_h8_and_dark()
    fig09_zero_info_control()
    fig10_h7_cross_station()


if __name__ == "__main__":
    main()
