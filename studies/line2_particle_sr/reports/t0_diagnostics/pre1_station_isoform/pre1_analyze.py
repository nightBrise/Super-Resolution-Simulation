"""前置① 第 2/3/4 层：理论层（R Σ 传播）+ 粒子级两两差异分解 + 分箱口径复算。

只读 src/、data/ 与同目录 `pre1_regenerated.npz`；只写本目录。CPU only。

判据（预注册）：
- 物理  = 理论收缩解释观测 + 粒子表一致塌缩 + 管线审计无异常；
- 缺陷  = 审计发现抖动/传输/快照路径错误，或粒子表差异显著大于直方图差异；
- 伪影  = 固定公共 edges 后两两 M1 恢复到与站 0 同量级。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.line2.data_pipeline.transport import (  # noqa: E402
    _cell, _solve_f, fodo_lattice, jitter_params,
)

OUT = Path(__file__).resolve().parent
N_BINS = 128
AXES = ("x", "xp", "y", "yp")
PROJ = (("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"))


# ---------------------------------------------------------------- 理论层
def transfer_matrix(lattice) -> np.ndarray:
    """与生产 transport_beam 同一 ocelot 路径的 4×4 线性传输矩阵。"""
    from ocelot.cpbd.beam import ParticleArray
    from ocelot.cpbd.track import track

    pa = ParticleArray(4)
    pa.rparticles[:4, :] = np.eye(4)
    pa.rparticles[4:6, :] = 0.0
    pa.E = 130.0
    _, out = track(lattice, pa, print_progress=False, calc_tws=False)
    return out.rparticles[:4, :].copy()


def jitter_matrix(p: dict) -> np.ndarray:
    """站 0 的抖动线性映射（不含平移项：踢角只动均值，不影响 Σ）。"""
    ell, f = p["drift_m"], p["f_j"]
    j = np.zeros((4, 4))
    j[0, 0], j[0, 1] = 1.0, ell
    j[1, 0], j[1, 1] = -1.0 / f, 1.0 - ell / f
    j[2, 2], j[2, 3] = 1.0, ell
    j[3, 2], j[3, 3] = 1.0 / f, 1.0 + ell / f
    return j


def corr_of(sigma: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(sigma))
    return sigma / np.outer(d, d)


def rho_x(sigma: np.ndarray) -> float:
    return float(sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1]))


def pair_stats(x: np.ndarray) -> dict:
    """1D 样本的两两绝对差中位数/分位。"""
    n = len(x)
    iu = np.triu_indices(n, 1)
    d = np.abs(x[iu[0]] - x[iu[1]])
    return {"median": float(np.median(d)), "p05": float(np.quantile(d, 0.05)),
            "p95": float(np.quantile(d, 0.95)), "min": float(d.min()), "max": float(d.max())}


# ---------------------------------------------------------------- 图像层
def pairwise_m1(imgs: np.ndarray) -> np.ndarray:
    """逐对 M1（= 生产 `group_rel_rmse`；各束总计数相同 ⇒ 归一因子为常数）。"""
    n, nproj, nb, _ = imgs.shape
    flat = imgs.reshape(n, nproj, nb * nb).astype(np.float64)
    mean_r = flat.sum(axis=2) / (nb * nb)                       # (n, 6)
    gram = np.einsum("ipb,jpb->ijp", flat, flat)
    sq = np.einsum("ipb,ipb->ip", flat, flat)
    d2 = sq[:, None, :] + sq[None, :, :] - 2.0 * gram           # (n,n,6)
    np.maximum(d2, 0.0, out=d2)
    m1_g = np.sqrt(d2 / (nb * nb)) / mean_r[None, :, :]
    return np.mean(m1_g, axis=2)


def pair_summary(mat: np.ndarray) -> dict:
    iu = np.triu_indices(mat.shape[0], 1)
    d = mat[iu]
    return {"median": float(np.median(d)), "mean": float(d.mean()),
            "p05": float(np.quantile(d, 0.05)), "p95": float(np.quantile(d, 0.95))}


def main() -> int:
    z = np.load(OUT / "pre1_regenerated.npz")
    beams = z["beams"]
    covs, means, stats = z["covs"], z["means"], z["stats"]
    report: dict = {"n_beams": int(len(beams))}

    # ---------------- 理论层 ----------------
    r4 = transfer_matrix(fodo_lattice())
    report["theory"] = {
        "R_4x4": r4.tolist(),
        "det_R": float(np.linalg.det(r4)),
        "sing_vals_x_block": np.linalg.svd(r4[:2, :2], compute_uv=False).tolist(),
        "sing_vals_y_block": np.linalg.svd(r4[2:, 2:], compute_uv=False).tolist(),
    }
    # 锚点 Σ：束 0 无抖动 ⇒ Σ_a = Σ0(束 0)。测试集不含束 0，用抖动矩阵反解。
    j0 = jitter_matrix(jitter_params(int(beams[0])))
    sigma_a = np.linalg.inv(j0) @ covs[0][0] @ np.linalg.inv(j0).T

    sig0_t, sig1_t = [], []
    for b in beams:
        s0 = jitter_matrix(jitter_params(int(b))) @ sigma_a @ jitter_matrix(jitter_params(int(b))).T
        sig0_t.append(s0)
        sig1_t.append(r4 @ s0 @ r4.T)
    sig0_t, sig1_t = np.stack(sig0_t), np.stack(sig1_t)
    report["theory"]["analytic_vs_particle_max_rel_err"] = {
        "station0": float(np.abs(sig0_t - covs[0]).max() / np.abs(covs[0]).max()),
        "station1": float(np.abs(sig1_t - covs[1]).max() / np.abs(covs[1]).max()),
    }

    rho0 = np.array([rho_x(c) for c in covs[0]])
    rho1 = np.array([rho_x(c) for c in covs[1]])
    rho0_t = np.array([rho_x(c) for c in sig0_t])
    rho1_t = np.array([rho_x(c) for c in sig1_t])
    alpha1 = -covs[1][:, 0, 1] / np.sqrt(np.linalg.det(covs[1][:, :2, :2]))
    report["shape_parameter"] = {
        "rho_x_station0": {"median": float(np.median(rho0)), "min": float(rho0.min()),
                           "max": float(rho0.max()), "pairwise_abs_delta": pair_stats(rho0)},
        "rho_x_station1": {"median": float(np.median(rho1)), "min": float(rho1.min()),
                           "max": float(rho1.max()), "pairwise_abs_delta": pair_stats(rho1)},
        "alpha_x_station1": {"median": float(np.median(alpha1)), "min": float(alpha1.min()),
                             "max": float(alpha1.max())},
        "theory_rho_x_station0": [float(rho0_t.min()), float(rho0_t.max())],
        "theory_rho_x_station1": [float(rho1_t.min()), float(rho1_t.max())],
    }
    d_rho0 = pair_stats(rho0)["median"]
    d_rho1 = pair_stats(rho1)["median"]
    report["shape_parameter"]["contraction_ratio_drho"] = float(d_rho0 / max(d_rho1, 1e-30))

    # ---------------- 粒子级差异分解 ----------------
    def decompose(st: int) -> dict:
        sd = stats[st]
        rel_pos, rel_size, d_rho, d_rho_y = {}, {}, {}, {}
        iu = np.triu_indices(len(beams), 1)
        for i, a in enumerate(AXES):
            s = sd[:, 4 + i]
            rel_pos[a] = float(np.median(np.abs(sd[iu[0], i] - sd[iu[1], i]) / np.sqrt((s**2).mean())))
            rel_size[a] = float(np.median(np.abs(np.log(sd[iu[0], 4 + i]) - np.log(sd[iu[1], 4 + i]))))
        for (i, j, name) in ((0, 1, "rho_x"), (2, 3, "rho_y")):
            r = covs[st][:, i, j] / np.sqrt(covs[st][:, i, i] * covs[st][:, j, j])
            d_rho[name] = float(np.median(np.abs(r[iu[0]] - r[iu[1]])))
        return {"position_delta_over_sigma": rel_pos, "rel_size_delta": rel_size,
                "abs_delta_rho": d_rho}
    report["particle_decomposition"] = {f"station{st}": decompose(st) for st in (0, 1)}

    # ---------------- 图像层：四种分箱口径 ----------------
    m1 = {}
    for v in ("V1", "V2", "V2b", "V2c"):
        for st in (0, 1):
            m1[f"{v}_s{st}"] = pairwise_m1(z[f"{v}_s{st}"])
    report["pairwise_M1"] = {k: pair_summary(v) for k, v in m1.items()}
    report["pairwise_M1_ratio_s1_over_s0"] = {
        v: float(report["pairwise_M1"][f"{v}_s1"]["median"]
                 / report["pairwise_M1"][f"{v}_s0"]["median"]) for v in ("V1", "V2", "V2b", "V2c")}

    # ---------------- 与既有 h5 图组的逐对 M1（V1 全 150 束，交叉核对 exp5） ----------------
    np.save(OUT / "pre1_pairwise_m1_V1_s0.npy", m1["V1_s0"])
    np.save(OUT / "pre1_pairwise_m1_V1_s1.npy", m1["V1_s1"])

    # ---------------- 常数解标尺（留一均值图；交叉核对 exp5 的 0.1944 / 0.0352） ----------------
    const = {}
    for v in ("V1", "V2", "V2b", "V2c"):
        for st in (0, 1):
            x = z[f"{v}_s{st}"].astype(np.float64)
            n = x.shape[0]
            total = x.sum(axis=0)
            loo = (total[None] - x) / (n - 1)                     # (n,6,128,128)
            num = np.sqrt(((x - loo) ** 2).mean(axis=(2, 3)))
            den = loo.mean(axis=(2, 3))
            const[f"{v}_s{st}"] = float(np.median((num / den).mean(axis=1)))
    report["constant_solution_loo_M1"] = const

    # ---------------- 格点周期扫描：同形是否为该 lattice 设定特有 ----------------
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    scan = {}
    for n_cells in (1, 2, 3, 4, 6, 8):
        elements: list = []
        for _ in range(n_cells):
            elements.extend(_cell(_solve_f()))
        rx = transfer_matrix(MagneticLattice(elements))[:2, :2]
        rs0 = [jitter_matrix(jitter_params(int(b)))[:2, :2] @ sigma_a[:2, :2]
               @ jitter_matrix(jitter_params(int(b)))[:2, :2].T for b in beams]
        rs1 = [rx @ s @ rx.T for s in rs0]
        r0 = np.array([rho_x(s) for s in rs0])
        r1 = np.array([rho_x(s) for s in rs1])
        b1 = np.array([s[0, 0] / np.sqrt(np.linalg.det(s)) for s in rs1])
        scan[int(n_cells)] = {
            "abs_delta_rho_s0_median": pair_stats(r0)["median"],
            "abs_delta_rho_s1_median": pair_stats(r1)["median"],
            "contraction_ratio": float(pair_stats(r0)["median"] / max(pair_stats(r1)["median"], 1e-30)),
            "beta1_variation_rel": float((b1.max() - b1.min()) / b1.mean()),
        }
    report["lattice_period_scan"] = scan

    (OUT / "pre1_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
