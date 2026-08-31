"""M6 三方案五图生成（90 [S3]，PNG only 300 dpi）。

Fig1/2/5：取自 pred_{A,B,C} test_id 第 1 个样本（sample_id=test_id-000）。
Fig3/4：合并 test_id metrics.csv（A/B/C × 4 种子，n=4000/方案）。

排版说明（2026-08-31 修订）：Fig1/2/5 原为单行多列并排，缩放进报告后被压窄、
文字偏小。现将 Fig1 改 2×3 网格、Fig5 改 2×2 网格（余一空格）、Fig2 放大字号，
保证每个子图缩放后仍可读。数据内容与样本编号不变。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from src.generators.f_beam import pixel_center_coordinates
from src.evaluation.metrics import bootstrap_ci
from src.evaluation.plots import _coords

OUT = pathlib.Path('assets')
OUT.mkdir(parents=True, exist_ok=True)

PRED = {s: np.load(f'results/EXP-02_summary/pred_{s}/predictions_test_id.npz', allow_pickle=True)
        for s in ['A','B','C']}
i = 0
H   = PRED['A']['H'][i]
LU  = PRED['A']['L_up'][i]
HA  = PRED['A']['H_hat'][i]
HB  = PRED['B']['H_hat'][i]
HC  = PRED['C']['H_hat'][i]
SID = PRED['A']['sample_id'][i]
if isinstance(SID, bytes): SID = SID.decode()

# ---------- Fig 1: 2D phasespace (log)  [2×3 网格，5 面板 + 1 空] ----------
fig, axes = plt.subplots(2, 3, figsize=(4.4*3, 4.4*2))
axes = axes.flatten()
panels = [("L_up", LU), ("H", H), ("Ĥ_A", HA), ("Ĥ_B", HB), ("Ĥ_C", HC)]
for k,(title,img) in enumerate(panels):
    ax = axes[k]
    img=np.asarray(img,dtype=np.float64)
    vmin=max(img[img>0].min(),1e-8)
    norm=LogNorm(vmin=vmin, vmax=img.max())
    im=ax.imshow(img, origin="lower", extent=(-1,1,-1,1), norm=norm, cmap="viridis")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("δ", fontsize=11); ax.set_ylabel("z", fontsize=11)
    ax.tick_params(labelsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for ax in axes[len(panels):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(OUT/'figure_01_2d_phasespace.png', dpi=300)
plt.close(fig)
print("Fig1 done")

# ---------- Fig 2: 1D profiles I(z) & S(δ)  [放大字号] ----------
coords,_ = pixel_center_coordinates(256)
prof = {
    "I(z)": {"H": H.sum(axis=1), "Ĥ_A": HA.sum(axis=1), "Ĥ_B": HB.sum(axis=1), "Ĥ_C": HC.sum(axis=1)},
    "S(δ)": {"H": H.sum(axis=0), "Ĥ_A": HA.sum(axis=0), "Ĥ_B": HB.sum(axis=0), "Ĥ_C": HC.sum(axis=0)},
}
fig, axes = plt.subplots(1, 2, figsize=(6.2*2, 5.0))
for ax,(title,curves) in zip(axes, prof.items()):
    for label,v in curves.items():
        ax.plot(coords, v, label=label, linewidth=1.3)
    ax.set_title(title, fontsize=13); ax.set_xlabel("z / δ", fontsize=11)
    ax.legend(fontsize=9); ax.set_ylabel("intensity", fontsize=11)
    ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(OUT/'figure_02_1d_profile.png', dpi=300)
plt.close(fig)
print("Fig2 done")

# ---------- Fig 3: physics error bar (bootstrap 95% CI) [不变] ----------
import csv
rows=list(csv.DictReader(open('results/EXP-02_summary/test_id_combined.csv')))
schemes=["A","B","C"]
metric_keys=["e_eps_z","e_I_peak"]
metric_labels=["ε_z (emittance)","I_peak (peak current)"]
fig, ax = plt.subplots(figsize=(6,4.2))
width=0.25; x=np.arange(len(metric_keys))
colors=["#4c72b0","#dd8452","#55a868"]
for j,sc in enumerate(schemes):
    means=[]; los=[]; his=[]
    for key in metric_keys:
        d=np.array([float(r[key]) for r in rows if r['scheme']==sc],dtype=np.float64)
        mean=float(d.mean())
        lo,hi=bootstrap_ci(d,n_boot=1000,seed=20260825)
        means.append(mean); los.append(mean-lo); his.append(hi-mean)
    ax.bar(x+(j-1)*width, means, width, yerr=[los,his], capsize=3, label=f"Scheme {sc}", color=colors[j])
ax.set_xticks(x); ax.set_xticklabels(metric_labels); ax.set_ylabel("Relative error")
ax.legend(); ax.set_title("Physics metric relative error (bootstrap 95% CI)")
fig.tight_layout(); fig.savefig(OUT/'figure_03_physics_error_bar.png', dpi=300); plt.close(fig)
print("Fig3 done (n=4000/scheme)")

# ---------- Fig 4: error vs γ scatter [不变] ----------
fig, ax = plt.subplots(figsize=(6,4.2))
for j,sc in enumerate(schemes):
    xs=[float(r['gamma']) for r in rows if r['scheme']==sc]
    ys=[float(r['e_eps_z']) for r in rows if r['scheme']==sc]
    ax.scatter(xs, ys, s=10, alpha=0.35, color=colors[j], label=f"Scheme {sc}")
ax.set_xlabel("γ (fine-structure parameter)"); ax.set_ylabel("ε_z relative error")
ax.legend(markerscale=2); ax.set_title("Error vs γ (test_id, 4 seeds)")
fig.tight_layout(); fig.savefig(OUT/'figure_04_error_vs_gamma_scatter.png', dpi=300); plt.close(fig)
print("Fig4 done")

# ---------- Fig 5: residual maps (diverging)  [2×2 网格，3 面板 + 1 空] ----------
def resid(h,hhat):
    return hhat/hhat.sum() - h/h.sum()
resA, resB, resC = resid(H,HA), resid(H,HB), resid(H,HC)
vmax=max(abs(np.concatenate([resA,resB,resC])).max(),1e-12)
fig, axes = plt.subplots(2,2, figsize=(4.6*2,4.6*2))
axes = axes.flatten()
rpanels=[("Ĥ_A − H",resA),("Ĥ_B − H",resB),("Ĥ_C − H",resC)]
for k,(title,rz) in enumerate(rpanels):
    ax = axes[k]
    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im=ax.imshow(rz, origin="lower", extent=(-1,1,-1,1), norm=norm, cmap="RdBu_r")
    ax.set_title(title, fontsize=13); ax.set_xlabel("δ", fontsize=11); ax.set_ylabel("z", fontsize=11)
    ax.tick_params(labelsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for ax in axes[len(rpanels):]:
    ax.axis('off')
fig.tight_layout(); fig.savefig(OUT/'figure_05_residual_map.png', dpi=300); plt.close(fig)
print("Fig5 done")
