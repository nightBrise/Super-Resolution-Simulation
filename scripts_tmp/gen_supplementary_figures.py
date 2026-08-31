"""Generate 11 supplementary figures with ENGLISH labels (no CJK fonts needed).
Saves to results/EXP-02_summary/figures_extra/ as PNG 300dpi.
"""
from __future__ import annotations
from pathlib import Path
import csv, json, re
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

ROOT = Path("/home/zhangny/Super-Resolution-Simulation")
OUT = ROOT / "results/EXP-02_summary/figures_extra"
OUT.mkdir(parents=True, exist_ok=True)

# Warm academic palette (same as before)
PRIMARY = "#1A1A1A"; BODY = "#333333"; SUB = "#4A4A4A"; SOFT = "#666666"
FAINT = "#E8E4DC"; CARD = "#FAF8F5"; PINE = "#2D5A4A"; SADDLE = "#8B4513"
BLUE = "#457B9d"; RED = "#e63946"; AMBER = "#e9c46a"; TEAL = "#2a9d8f"

DPI = 300
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "savefig.dpi": DPI, "figure.dpi": DPI,
    "savefig.bbox": "tight",
})


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")


# =========================================================================
# Fig 1: M1 generator sample (H + DoG-HF detail + CSR profile)
# =========================================================================
def fig1():
    with h5py.File(ROOT / "data/v1/test_id.h5", "r") as f:
        H = f["H"][0]
    blur = gaussian_filter(H, sigma=8)
    H_hp = H - blur

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    im = axes[0].imshow(np.log1p(H * 100), cmap="magma", origin="lower")
    axes[0].set_title("Ground truth H  (256x256, log scale)", fontsize=12, color=PRIMARY)
    axes[0].set_xlabel("z axis  (longitudinal position)", fontsize=10, color=SUB)
    axes[0].set_ylabel("delta axis  (energy spread)", fontsize=10, color=SUB)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="log(1+100*H)")

    H_hp_abs = np.abs(H_hp)
    cy, cx = np.unravel_index(np.argmax(H_hp_abs), H_hp_abs.shape)
    half = 32
    y0, y1 = max(0, cy - half), min(H.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(H.shape[1], cx + half)
    crop = H_hp_abs[y0:y1, x0:x1]
    im2 = axes[1].imshow(crop, cmap="viridis", origin="lower")
    axes[1].set_title(f"H high-freq detail (DoG sigma=8)\ncenter ({cy},{cx}) 64x64 crop", fontsize=12, color=PRIMARY)
    axes[1].set_xlabel("z", fontsize=10, color=SUB)
    axes[1].set_ylabel("delta", fontsize=10, color=SUB)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    mid_row = H[cy, :]
    axes[2].plot(mid_row, color=PINE, lw=1.5, label="H(z) raw center row")
    axes[2].plot(gaussian_filter(mid_row, sigma=2), color=AMBER, lw=1.5, ls="--", label="sigma_smooth=2 low-freq envelope")
    axes[2].set_title("CSR microbunching ripples\n(difference = high-freq fine structure)", fontsize=12, color=PRIMARY)
    axes[2].set_xlabel("z position  (pixel)", fontsize=10, color=SUB)
    axes[2].set_ylabel("Current density H(z)", fontsize=10, color=SUB)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Fig 1  |  M1 generator sample  (H / DoG-HF detail / CSR profile)",
                 fontsize=14, color=PRIMARY, y=1.02, fontweight="bold")
    save(fig, "fig01_m1_sample.png")


# =========================================================================
# Fig 2: Degradation protocol (4 steps)
# =========================================================================
def fig2():
    with h5py.File(ROOT / "data/v1/test_id.h5", "r") as f:
        H = f["H"][0]
        L_up = f["L_up"][0]
    sigma_K = 11.0; sigma_n = 1.22e-4; r = 4
    H_blur = gaussian_filter(H, sigma=sigma_K)
    H_down = H_blur.reshape(H_blur.shape[0] // r, r, H_blur.shape[1] // r, r).sum(axis=(1, 3))
    rng = np.random.default_rng(42)
    H_noisy = H_down + rng.normal(0, sigma_n, H_down.shape)
    H_clipped = np.maximum(H_noisy, 0)
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)
    axes = [fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            ]

    titles = [
        "(a) Ground truth H\n256x256, sum(H)=1",
        f"(b) Gaussian blur\nsigma_K={sigma_K:.1f} px",
        f"(c) 4x block-sum downsample\n256 -> 64",
        f"(d) Gaussian noise\nsigma_n={sigma_n:.2e}",
        "(e) Clip+bilinear upsample\nL_up 256x256 (network input)",
    ]
    images = [H, H_blur, H_down, H_noisy, L_up]
    cmaps = ["magma", "magma", "magma", "RdBu_r", "magma"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        if cmap == "RdBu_r":
            vmax = max(abs(img.min()), abs(img.max()))
            im = ax.imshow(img, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(np.log1p(img * 100) if img.max() > 0 else img, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=12, color=PRIMARY)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Fig 2  |  Degradation protocol  (H -> L_up : blur -> downsample -> noise -> clip)",
                 fontsize=15, color=PRIMARY, y=0.98, fontweight="bold")
    save(fig, "fig02_degradation.png")


def fig3():
    fig, axes = plt.subplots(3, 1, figsize=(16, 11))
    for ax in axes:
        ax.set_xlim(0, 16); ax.set_ylim(0, 3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

    def box(ax, x, y, w, h, text, color=PRIMARY, fc=CARD, fontsize=10, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               edgecolor=color, facecolor=fc, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight="bold" if bold else "normal")

    def arrow(ax, x0, x1, y, color=SOFT):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

    # Scheme A
    axes[0].set_title("Scheme A  |  No prior (baseline)", loc="left",
                      fontsize=18, color=PRIMARY, fontweight="bold", x=0.01)
    box(axes[0], 0.3, 1.2, 1.7, 0.9, "L_up\n(64->256)", color=PINE, fontsize=15)
    arrow(axes[0], 2.1, 3.0, 1.65)
    box(axes[0], 3.0, 1.2, 3.5, 0.9, "U-Net backbone\n(5-level residual)", color=PRIMARY, fontsize=14)
    arrow(axes[0], 6.6, 7.5, 1.65)
    box(axes[0], 7.5, 1.2, 3.2, 0.9, "Softplus(S*L_up+R)\n(work scale S=N^2)", color=PRIMARY, fontsize=14)
    arrow(axes[0], 10.8, 11.7, 1.65)
    box(axes[0], 11.7, 1.2, 4.0, 0.9, "Predicted H\n(256x256)", color=SADDLE, fontsize=17, bold=True)

    # Scheme B
    axes[1].set_title("Scheme B  |  Image prior P2 + early fusion + residual", loc="left",
                      fontsize=18, color=PRIMARY, fontweight="bold", x=0.01)
    box(axes[1], 0.3, 1.6, 1.7, 0.7, "L_up", color=PINE, fontsize=15)
    box(axes[1], 0.3, 0.6, 1.7, 0.7, "P2\n(sm=15x)", color=BLUE, fontsize=14)
    arrow(axes[1], 2.1, 3.0, 1.95)
    arrow(axes[1], 2.1, 3.0, 0.95)
    box(axes[1], 3.0, 0.7, 2.8, 1.7, "Early fusion\n(2-channel)\nconcat", color=TEAL, fontsize=14)
    arrow(axes[1], 5.9, 6.8, 1.55)
    box(axes[1], 6.8, 1.2, 3.5, 0.9, "U-Net learns\nresidual R", color=PRIMARY, fontsize=14)
    arrow(axes[1], 10.4, 11.3, 1.65)
    box(axes[1], 11.3, 1.2, 4.0, 0.9, "H = Softplus\n(S*L_up + R)", color=SADDLE, fontsize=10, bold=True)

    # Scheme C
    axes[2].set_title("Scheme C  |  Param prior c_prior + FiLM", loc="left",
                      fontsize=18, color=PRIMARY, fontweight="bold", x=0.01)
    box(axes[2], 0.3, 1.6, 1.7, 0.7, "L_up", color=PINE, fontsize=15)
    box(axes[2], 0.3, 0.6, 1.7, 0.7, "c_prior\n(8-dim)", color=BLUE, fontsize=14)
    arrow(axes[2], 2.1, 3.0, 1.95)
    arrow(axes[2], 2.1, 3.0, 0.95)
    box(axes[2], 3.0, 0.7, 2.6, 1.7, "z-score + MLP\n(128 hidden)\nparam encode", color=TEAL, fontsize=9)
    box(axes[2], 5.7, 0.7, 3.4, 1.7, "FiLM injection\n(bottleneck+decoder)\ngamma*x + beta", color=AMBER, fontsize=9)
    arrow(axes[2], 9.2, 10.1, 1.55)
    box(axes[2], 10.1, 1.2, 2.5, 0.9, "U-Net\n(FiLM)", color=PRIMARY, fontsize=14)
    arrow(axes[2], 12.7, 13.6, 1.65)
    box(axes[2], 13.6, 1.2, 2.0, 0.9, "Predicted H", color=SADDLE, fontsize=17, bold=True)

    fig.suptitle("Fig 3  |  Three-scheme architecture  (fair comparison: only prior injection differs)",
                 fontsize=19, color=PRIMARY, fontweight="bold")
    save(fig, "fig03_schemes.png")


def fig4():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = {"A": BLUE, "B": TEAL, "C": SADDLE}
    ls_map = {0: "-", 1: "--"}
    for run in ["A_seed0", "B_seed0", "A_seed1", "B_seed1", "C_seed0", "C_seed1"]:
        log_path = ROOT / f"results/EXP-02_{run}_run1_D2/logs/train.log"
        if not log_path.exists(): continue
        steps, vals = [], []
        for line in log_path.read_text(errors="ignore").splitlines():
            m = re.search(r"step=(\d+)\s.*val_loss=([\d.eE+-]+)", line)
            if m:
                steps.append(int(m.group(1))); vals.append(float(m.group(2)))
        if not steps: continue
        arm = run.split("_")[0]; seed = int(run.split("_seed")[1])
        label = f"{arm}_seed{seed} (best={min(vals):.4f})"
        ax.plot(steps, vals, color=colors[arm], ls=ls_map[seed], lw=1.4, label=label, alpha=0.85)
    ax.set_xlabel("Training step", fontsize=12, color=SUB)
    ax.set_ylabel("Validation loss (val_loss)", fontsize=12, color=SUB)
    ax.set_title("Fig 4  |  EXP-02 training loss curves  (val_loss vs step, 6 runs)",
                 fontsize=14, color=PRIMARY, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50000)
    save(fig, "fig04_training_curves.png")


# =========================================================================
# Fig 5: G1(b) A vs L_up histogram
# =========================================================================
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for seed in [0, 1]:
        ax = axes[seed]
        with open(ROOT / f"results/EXP-02_summary/seed{seed}/metrics.csv") as f:
            rdr = csv.DictReader(f)
            a_lup, a_hat = [], []
            for r in rdr:
                if r["split"] != "test_id" or r["scheme"] != "A": continue
                try:
                    a_lup.append(float(r["e_high_mask_lup"]))
                    a_hat.append(float(r["e_high_mask"]))
                except (ValueError, KeyError): pass
        ax.hist(a_lup, bins=50, alpha=0.55, color=BLUE,
                label=f"L_up baseline (mean={np.mean(a_lup):.4f})")
        ax.hist(a_hat, bins=50, alpha=0.55, color=SADDLE,
                label=f"Scheme A (mean={np.mean(a_hat):.4f})")
        d = np.array(a_lup) - np.array(a_hat)
        ax.axvline(np.mean(a_lup), color=BLUE, ls="--", lw=1.2)
        ax.axvline(np.mean(a_hat), color=SADDLE, ls="--", lw=1.2)
        # Subplot title: shorter + lower position to avoid overlap
        ax.set_title(f"seed{seed}: d = L_up - A = {np.mean(d):+.4f}\n(positive = A beats L_up)",
                     fontsize=10, color=PRIMARY, pad=10)
        ax.set_xlabel("eps_high_mask", fontsize=10, color=SUB)
        ax.set_ylabel("Sample count", fontsize=10, color=SUB)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    # Main suptitle: shorter text + smaller font + more top padding
    fig.suptitle("Fig 5  |  G1(b) zero-learning baseline: A vs L_up (test_id, n=1000/seed)",
                 fontsize=12, color=PRIMARY, fontweight="bold", y=0.99)
    # Reserve top space for suptitle (avoid overlap with subplot titles)
    fig.subplots_adjust(top=0.88, wspace=0.25)
    save(fig, "fig05_g1b_A_vs_Lup.png")


# =========================================================================
# Fig 6: G2 verdict heatmap
# =========================================================================
def fig6():
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    seeds = [0, 1]
    cells = np.zeros((len(pairs), len(seeds)), dtype=int)
    labels = np.empty_like(cells, dtype=object)
    cis = np.empty_like(cells, dtype=object)
    for i, (y, x) in enumerate(pairs):
        for j, s in enumerate(seeds):
            with open(ROOT / f"results/EXP-02_summary/seed{s}/summary_test_id.json") as f:
                tc = json.load(f)["three_class"][f"M_{y}_minus_M_{x}"]
            verdict = tc["verdict"]
            labels[i, j] = verdict
            cis[i, j] = f"[{tc['ci95'][0]:.1e}, {tc['ci95'][1]:.1e}]"
            cells[i, j] = {"equivalent": 0, "significant_positive": 1, "significant_negative": 2}[verdict]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.matplotlib.colors.ListedColormap(["#FAF8F5", "#c0392b", "#27ae60"])  # [equivalent, sig_pos, sig_neg]
    im = ax.imshow(cells, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed{s}" for s in seeds], fontsize=12)
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels([f"{y} - {x}" for y, x in pairs], fontsize=12)
    for i in range(len(pairs)):
        for j in range(len(seeds)):
            color = "white" if cells[i, j] != 0 else PRIMARY
            weight = "bold" if cells[i, j] != 0 else "normal"
            ax.text(j, i - 0.15, labels[i, j], ha="center", va="center",
                    color=color, fontsize=10, fontweight=weight)
            ax.text(j, i + 0.18, cis[i, j], ha="center", va="center",
                    color=color, fontsize=8, fontweight=weight)
    legend_elements = [
        Patch(facecolor="#c0392b", label="sig.positive (y error > x error => x wins)"),
        Patch(facecolor="#FAF8F5", edgecolor="#E8E4DC", label="equivalent (CI crosses zero)"),
        Patch(facecolor="#27ae60", label="sig.negative (y error < x error => y wins)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=10)
    ax.set_title("Fig 6  |  G2 three-class verdict heatmap  (test_id, 6 verdicts)\n* ALL 3 pair comparisons are cross-seed INCONSISTENT -> G2 'diagnostically uncertain'",
                 fontsize=13, color=PRIMARY, fontweight="bold")
    save(fig, "fig06_g2_verdicts.png")


# =========================================================================
# Fig 7: Cross-seed distribution boxplot
# =========================================================================
def fig7():
    data = {}
    with open(ROOT / "results/EXP-02_summary/seed0/metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test_id": continue
            try: data.setdefault(("seed0", r["scheme"]), []).append(float(r["e_high_mask"]))
            except (ValueError, KeyError): pass
    with open(ROOT / "results/EXP-02_summary/seed1/metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] != "test_id": continue
            try: data.setdefault(("seed1", r["scheme"]), []).append(float(r["e_high_mask"]))
            except (ValueError, KeyError): pass

    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions, vals, colors_box = [], [], []
    labels_x = []
    arms_order = ["A", "B", "C"]
    for j, seed in enumerate([0, 1]):
        for i, arm in enumerate(arms_order):
            pos = j * 4 + i
            positions.append(pos); vals.append(np.log10(np.array(data.get((f"seed{seed}", arm), [])) + 1e-9))
            colors_box.append({"A": BLUE, "B": TEAL, "C": SADDLE}[arm])
            labels_x.append(f"{arm}_seed{seed}")
    bp = ax.boxplot(vals, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for median, c in zip(bp["medians"], colors_box):
        median.set_color(c); median.set_linewidth(2)
    ax.set_xticks(positions); ax.set_xticklabels(labels_x, fontsize=11, color=PRIMARY)
    ax.set_ylabel("log10(eps_high_mask)", fontsize=12, color=SUB)
    ax.set_title("Fig 7  |  Cross-seed eps_high_mask distribution  (boxplot, log scale)\n* Scheme C: std nearly constant (0.0006) BUT mean shifts 51% (0.00106 -> 0.00160) -- distribution shift, not outliers",
                 fontsize=13, color=PRIMARY, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    c0_mean = np.mean(data[("seed0", "C")]); c1_mean = np.mean(data[("seed1", "C")])
    ax.annotate(f"C seed0\nmean {c0_mean:.4f}", xy=(2, np.log10(c0_mean)),
                xytext=(2.5, np.log10(c0_mean) - 0.3), fontsize=9, color=SADDLE,
                arrowprops=dict(arrowstyle="->", color=SADDLE, lw=1.2))
    ax.annotate(f"C seed1\nmean {c1_mean:.4f}", xy=(6, np.log10(c1_mean)),
                xytext=(6.5, np.log10(c1_mean) + 0.3), fontsize=9, color=SADDLE,
                arrowprops=dict(arrowstyle="->", color=SADDLE, lw=1.2))
    save(fig, "fig07_cross_seed_boxplot.png")


# =========================================================================
# Fig 8: Mask composition
# =========================================================================
def fig8():
    with open(ROOT / "results/EXP-02_summary/seed0/summary_test_id.json") as f:
        mc = json.load(f)["mask_composition"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    metrics = ["ch_in (c_high frac)", "b_in (beta frac)", "Pi_leak (prior leak)"]
    values = [mc["ch_in_mask"]["mean"], mc["b_in_mask"]["mean"], mc["pi_leak"]["mean"]]
    colors = [PINE, SADDLE, SADDLE]
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor=PRIMARY, lw=1.5)
    ax.axhline(1.5 * mc["ch_in_mask"]["mean"], color=PINE, ls="--", lw=1.2,
               label=f"b_in trigger threshold = 1.5 x ch_in = {1.5*mc['ch_in_mask']['mean']:.3f}")
    ax.axhline(0.5, color=SADDLE, ls="--", lw=1.2, label="Pi_leak trigger threshold = 0.5")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.3f}",
                ha="center", fontsize=11, color=PRIMARY, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12, color=SUB)
    ax.set_title("Fig 8  |  Primary-metric mask composition + prior leak index\n* b_in > 1.5x ch_in AND Pi_leak > 0.5: prior structurally dominates the primary-metric region",
                 fontsize=13, color=PRIMARY, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "fig08_mask_composition.png")


# =========================================================================
# Fig 9: Merged CI coverage warning
# =========================================================================
def fig9():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods = ["Single-seed bootstrap CI\n(n=1000, 1000 reps)",
               "Merged 2-seed CI\n(n=2000, 3000 reps, seed-distinct)"]
    coverages = [95.0, 5.0]
    nominal = 95
    colors_b = [PINE, SADDLE]
    bars = ax.bar(methods, coverages, color=colors_b, alpha=0.7,
                  edgecolor=PRIMARY, lw=1.5, width=0.5)
    ax.axhline(nominal, color=PRIMARY, ls="--", lw=1.5,
               label=f"Nominal coverage = {nominal}%")
    for bar, v in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v}%",
                ha="center", fontsize=14, color=PRIMARY, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Empirical coverage (%)", fontsize=12, color=SUB)
    ax.set_title("Fig 9  |  Merged bootstrap CI coverage warning  (Qwen 3.8 Max simulation)\n* Merged CI across seeds: coverage only 5% -- descriptive only, NOT for three-class verdict",
                 fontsize=13, color=PRIMARY, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "fig09_merged_ci_coverage.png")


# =========================================================================
# Fig 10: Milestone timeline M0-M6
# =========================================================================
def fig10():
    milestones = [
        ("M0", "Spec\nfrozen", "2026-08-25", "done"),
        ("M1", "Generator\nready", "2026-08-26", "done"),
        ("M2", "Dataset\nbuilt", "2026-08-26", "done"),
        ("M3", "Baseline\n(A scheme)", "2026-08-26", "done"),
        ("M4", "Main exp\n+ G2 verdict", "2026-08-28", "active"),
        ("M5", "Ablation\nEXP-03/04/07/08", "TBD", "pending"),
        ("M6", "Final report\n+ exit decision", "TBD", "pending"),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    n = len(milestones); xs = list(range(n))
    status_color = {"done": PINE, "active": SADDLE, "pending": SOFT}
    ax.plot(xs, [0]*n, "-", color=FAINT, lw=2, zorder=1)
    for i, (mid, name, date, status) in enumerate(milestones):
        c = status_color[status]
        ax.scatter(i, 0, s=300, color=c, zorder=3, edgecolors=PRIMARY, lw=2)
        ax.text(i, 0.08, mid, ha="center", va="bottom", fontsize=13,
                color=PRIMARY, fontweight="bold")
        ax.text(i, -0.08, name, ha="center", va="top", fontsize=10, color=SUB)
        ax.text(i, -0.20, date, ha="center", va="top", fontsize=9, color=SOFT)
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.35, 0.35)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    legend_elements = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor=PINE, markersize=12, lw=0, label="completed"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor=SADDLE, markersize=12, lw=0,
               label="in progress (M4 seed expansion)"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor=SOFT, markersize=12, lw=0, label="pending"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.55),
              ncol=3, fontsize=11)
    ax.set_title("Fig 10  |  M0-M6 milestone timeline  (currently at M4; G2 failure path: seed expansion approved)",
                 fontsize=14, color=PRIMARY, fontweight="bold")
    save(fig, "fig10_milestones.png")


# =========================================================================
# Fig 11: sigma_smooth,H revision before/after
# =========================================================================
def fig11():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = ["Before\nsm_H = 0.5 x w_fine", "After\nsm_H = 0.125 x w_fine"]
    values = [1.90, 7.16]  # from P0 revision package, n=30 measurement
    colors_b = [SADDLE, PINE]
    bars = ax.bar(labels, values, color=colors_b, alpha=0.75,
                  edgecolor=PRIMARY, lw=1.5, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.15, f"{v:.2f}%",
                ha="center", fontsize=14, color=PRIMARY, fontweight="bold")
    ax.annotate("3.8x increase", xy=(1, 7.16), xytext=(0.5, 6),
                fontsize=12, color=PINE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PINE, lw=1.5))
    ax.set_ylabel("H high-freq energy fraction  ( ||H_hp||_1 / sum(H) )", fontsize=12, color=SUB)
    ax.set_title("Fig 11  |  sigma_smooth,H revision: H high-freq energy before/after  (n=30 measurement)\n* Root-cause fix: 1.90% -> 7.16% recovers 54% of fine structure (vs old wiped out by over-smoothing)",
                 fontsize=13, color=PRIMARY, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "fig11_sigma_smooth_revision.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9(); fig10(); fig11()
    print("\n=== all 11 figures (ENGLISH labels) generated ===")