"""L1 [S1] 锚点入库与元数据（spec line2/10 [S1]）。

职责：锚点副本校验（MD5/大小）、SDDS 解析、Twiss/高阶矩提取（twiss.json）、
manifest 写入。验收 Claims：[S1] C1–C4。
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from .sdds_reader import read_sdds_binary

__all__ = [
    "ANCHOR_MD5",
    "ANCHOR_SIZE",
    "read_anchor_particles",
    "compute_twiss",
    "verify_anchor_copy",
    "ingest_anchor",
]

ANCHOR_MD5 = "f39d905f96abd5588091dad6dd3884b2"
ANCHOR_SIZE = 9_600_397
_ANCHOR_DEFAULT_SOURCE = Path(
    "/home/zhangny/pyaml-main/examples/SXFEL/elegant_files/injector.sdds"
)
_TRANSVERSE = ("x", "xp", "y", "yp")


def md5_of(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_anchor_copy(copy_path: str | Path) -> None:
    """[S1] C1：副本存在、大小、MD5 三重校验，不符抛 FileNotFoundError/ValueError。"""
    path = Path(copy_path)
    if not path.is_file():
        raise FileNotFoundError(f"锚点副本不存在：{path}")
    size = path.stat().st_size
    if size != ANCHOR_SIZE:
        raise ValueError(f"锚点副本大小 {size} != 期望 {ANCHOR_SIZE}")
    digest = md5_of(path)
    if digest != ANCHOR_MD5:
        raise ValueError(f"锚点副本 MD5 {digest} != 期望 {ANCHOR_MD5}")


def compute_twiss(particles: np.ndarray) -> dict:
    """[S1] C2 的主实现：由 (N,6) 粒子表 (x,xp,y,yp,t,p) 计算 Twiss/统计量。

    独立重算路径（测试侧）用另一套实现互检，相对误差 < 1e-12。
    协方差取样本协方差（ddof=1，sddsanalyzebeam 惯例）。
    """
    x, xp, y, yp, t, p = (particles[:, i] for i in range(6))
    cov = np.cov(np.vstack([x, xp, y, yp]), ddof=1)
    eps_x = float(np.sqrt(np.linalg.det(cov[:2, :2])))
    eps_y = float(np.sqrt(np.linalg.det(cov[2:, 2:])))
    beta_gamma = float(np.mean(p))
    eps_nx, eps_ny = beta_gamma * eps_x, beta_gamma * eps_y
    beta_x = float(cov[0, 0] / eps_x)
    alpha_x = float(-cov[0, 1] / eps_x)
    beta_y = float(cov[2, 2] / eps_y)
    alpha_y = float(-cov[2, 3] / eps_y)

    def _skew(v: np.ndarray) -> float:
        z = (v - v.mean()) / v.std()
        return float(np.mean(z**3))

    def _kurt(v: np.ndarray) -> float:
        z = (v - v.mean()) / v.std()
        return float(np.mean(z**4) - 3.0)

    skew = {n: _skew(v) for n, v in zip(("x", "xp", "y", "yp", "t", "p"), (x, xp, y, yp, t, p))}
    kurt = {n: _kurt(v) for n, v in zip(("x", "xp", "y", "yp", "t", "p"), (x, xp, y, yp, t, p))}
    sd_t, sd_p = float(t.std(ddof=1)), float(p.std(ddof=1))
    return {
        "n_particles": int(particles.shape[0]),
        "cov_ddof": 1,
        "sigma_matrix": {
            "order": ["x", "xp", "y", "yp"],
            "xx": float(cov[0, 0]), "xxp": float(cov[0, 1]),
            "xy": float(cov[0, 2]), "xyp": float(cov[0, 3]),
            "xpxp": float(cov[1, 1]), "xpy": float(cov[1, 2]),
            "xpyp": float(cov[1, 3]), "yy": float(cov[2, 2]),
            "yyp": float(cov[2, 3]), "ypyp": float(cov[3, 3]),
        },
        "alpha": {"x": alpha_x, "y": alpha_y},
        "beta": {"x": beta_x, "y": beta_y},
        "emit_geom": {"x": eps_x, "y": eps_y},
        "emit_n": {"x": eps_nx, "y": eps_ny},
        "beta_gamma": beta_gamma,
        "sigma_t_s": sd_t,
        "sigma_dp_over_p": sd_p / beta_gamma,
        "corr_t_p": float(np.corrcoef(t, p)[0, 1]),
        "skew": skew,
        "kurtosis": kurt,
    }


def read_anchor_particles(sdds_path: str | Path) -> np.ndarray:
    """SDDS → (N, 6) 粒子表 (x, x′, y, y′, t, p)，float64。"""
    page = read_sdds_binary(sdds_path)
    return np.stack(
        [page.columns[n] for n in ("x", "xp", "y", "yp", "t", "p")], axis=1
    )


def write_anchor_metadata(
    anchor_dir: str | Path,
    source_path: str | Path,
    master_seed: int = 20260913,
) -> dict:
    """解析锚点副本 → 写 twiss.json 与 manifest.json（[S1] C2/C3）。"""
    anchor_dir = Path(anchor_dir)
    copy = anchor_dir / "injector.sdds"
    verify_anchor_copy(copy)
    page = read_sdds_binary(copy)
    if "x" not in page.columns or page.row_count != 200_001:
        raise ValueError(f"锚点解析异常：row_count={page.row_count}")
    particles = np.stack(
        [page.columns[n] for n in ("x", "xp", "y", "yp", "t", "p")], axis=1
    )
    twiss = compute_twiss(particles)
    twiss["charge_c"] = float(page.parameter_values.get("Charge", float("nan")))
    twiss["units_p"] = "beta_gamma"
    (anchor_dir / "twiss.json").write_text(
        json.dumps(twiss, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "source_path": str(source_path),
        "md5": ANCHOR_MD5,
        "size_bytes": ANCHOR_SIZE,
        "units_p": "beta_gamma",
        "extraction_script": "src/line2/data_pipeline/anchor.py",
        "extraction_date": date.today().isoformat(),
        "master_seed": int(master_seed),
        "twiss": "twiss.json",
    }
    (anchor_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"twiss": twiss, "manifest": manifest, "row_count": page.row_count}


def ingest_anchor(
    anchor_dir: str | Path = "studies/line2_particle_sr/data/anchor",
    source_path: str | Path = _ANCHOR_DEFAULT_SOURCE,
    master_seed: int = 20260913,
) -> dict:
    """[S1] 入库：复制源文件 → 校验 → 写元数据。twiss.json 逐位幂等；manifest 同日幂等（extraction_date = 运行日）。"""
    anchor_dir = Path(anchor_dir)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    copy = anchor_dir / "injector.sdds"
    if not copy.is_file():
        copy.write_bytes(Path(source_path).read_bytes())
    verify_anchor_copy(copy)
    return write_anchor_metadata(anchor_dir, source_path, master_seed)
