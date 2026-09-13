"""L1 [S1] 锚点入库与元数据的验收测试（spec line2/10 [S1] C1–C4、[S2] C1）。

测试只断言协议/契约/不变量（line1 铁律延续），不依赖仓库外的源文件路径
（reviewer m5：以 manifest/硬编码 MD5 为准）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from src.line2.data_pipeline.anchor import (
    ANCHOR_MD5,
    ANCHOR_SIZE,
    compute_twiss,
    ingest_anchor,
)
from src.line2.data_pipeline.sdds_reader import read_sdds_binary

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_COPY = REPO_ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
EXPECTED_MD5 = ANCHOR_MD5
EXPECTED_SIZE = ANCHOR_SIZE


# ---------------------------------------------------------------- [S1] C1
def test_anchor_copy_integrity() -> None:
    """[S1] C1：副本存在、大小 9,600,397、实算 MD5 与硬编码值一致（不依赖外部路径）。"""
    assert ANCHOR_COPY.is_file(), f"锚点副本缺失：{ANCHOR_COPY}"
    assert ANCHOR_COPY.stat().st_size == EXPECTED_SIZE
    digest = hashlib.md5(ANCHOR_COPY.read_bytes()).hexdigest()
    assert digest == EXPECTED_MD5


# ---------------------------------------------------------------- [S2] C1
def test_sdds_dual_parse_bitwise() -> None:
    """[S2] C1：sdds_reader 解析与手工页布局解析逐位一致（双实现互检）。"""
    raw = ANCHOR_COPY.read_bytes()
    # 手工路径：&data 行后 → int32 行数 → double 参数 → 行主序 double 列
    marker = b"&data mode=binary, &end\n"
    offset = raw.index(marker) + len(marker)
    row_count = int(np.frombuffer(raw[offset : offset + 4], dtype="<i4")[0])
    assert row_count == 200_001
    charge = float(np.frombuffer(raw[offset + 4 : offset + 12], dtype="<f8")[0])
    body = np.frombuffer(raw[offset + 12 : offset + 12 + row_count * 48], dtype="<f8")
    manual = body.reshape(row_count, 6)

    page = read_sdds_binary(ANCHOR_COPY)
    assert page.row_count == row_count
    assert page.parameter_values["Charge"] == charge
    assert page.column_names == ("x", "xp", "y", "yp", "t", "p")
    parsed = np.stack([page.columns[n] for n in page.column_names], axis=1)
    np.testing.assert_array_equal(parsed, manual)  # 逐位一致


# ---------------------------------------------------------------- [S1] C2
def test_twiss_independent_recompute() -> None:
    """[S1] C2：twiss 主实现与测试侧独立重算一致（相对误差 < 1e-12）。"""
    page = read_sdds_binary(ANCHOR_COPY)
    particles = np.stack([page.columns[n] for n in ("x", "xp", "y", "yp", "t", "p")], axis=1)
    twiss = compute_twiss(particles)

    x, xp, y, yp, t, p = (particles[:, i] for i in range(6))
    # 独立实现：手写中心化积、样本协方差口径（分母 N-1，与 np.cov(ddof=1) 同义）
    def var(v: np.ndarray) -> float:
        m = v.mean()
        return float(((v - m) ** 2).sum() / (len(v) - 1))

    def cov(a: np.ndarray, b: np.ndarray) -> float:
        am, bm = a.mean(), b.mean()
        return float(((a - am) * (b - bm)).sum() / (len(a) - 1))

    sxx, sxpxp = var(x), var(xp)
    sxy = cov(x, xp)
    eps_x = (sxx * sxpxp - sxy**2) ** 0.5
    assert abs(twiss["sigma_matrix"]["xx"] - sxx) < abs(sxx) * 1e-12
    assert abs(twiss["emit_geom"]["x"] - eps_x) < eps_x * 1e-12
    assert abs(twiss["beta"]["x"] - sxx / eps_x) < 1e-9
    assert abs(twiss["alpha"]["x"] + sxy / eps_x) < 1e-9
    # 峰度/偏度与 scipy 独立实现互检
    assert abs(twiss["kurtosis"]["xp"] - (sps.kurtosis(xp, fisher=True))) < 1e-10
    assert abs(twiss["skew"]["p"] - sps.skew(p)) < 1e-12
    # 物理画像抽查（2026-09-13 实测值，硬编码于大纲 v0.8）
    assert abs(twiss["beta_gamma"] - 254.1622) < 0.01
    assert abs(twiss["emit_n"]["x"] - 1.26e-6) < 0.02e-6


# ---------------------------------------------------------------- [S1] C3
def test_ingest_manifest_fields(tmp_path: Path) -> None:
    """[S1] C3：ingest 在 tmp 目录产出 manifest（字段完整）与 twiss.json。"""
    result = ingest_anchor(tmp_path, ANCHOR_COPY, master_seed=20260913)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for key in (
        "source_path",
        "md5",
        "size_bytes",
        "units_p",
        "extraction_script",
        "extraction_date",
        "master_seed",
    ):
        assert key in manifest, f"manifest 缺字段 {key}"
    assert manifest["md5"] == EXPECTED_MD5
    assert manifest["units_p"] == "beta_gamma"
    assert manifest["master_seed"] == 20260913
    twiss = json.loads((tmp_path / "twiss.json").read_text(encoding="utf-8"))
    assert twiss["n_particles"] == 200_001
    assert result["row_count"] == 200_001
    # 幂等：重复入库产物一致（extraction_date 同日）
    ingest_anchor(tmp_path, ANCHOR_COPY, master_seed=20260913)
    assert (tmp_path / "twiss.json").read_bytes() == (tmp_path / "twiss.json").read_bytes()


# ---------------------------------------------------------------- [S1] C4
def test_anchor_not_git_ignored() -> None:
    """[S1] C4：锚点文件不被 .gitignore 忽略（白名单生效；git check-ignore 返回 1）。"""
    proc = subprocess.run(
        ["git", "check-ignore", "-v", str(ANCHOR_COPY.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, f"锚点被忽略：{proc.stdout}"
