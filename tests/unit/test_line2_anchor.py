"""L1 [S1] 锚点入库与元数据的验收测试（spec line2/10 [S1] C1–C4、[S2] C1）。

测试只断言协议/契约/不变量（line1 铁律延续），不依赖仓库外的源文件路径
（reviewer m5：以 manifest/硬编码 MD5 为准）。[S1] C2 的覆盖分两层：归档件
`anchor/twiss.json` 的字段完整性 + 同公式独立重算（含 σ_t/σ_dp,p/corr(t,p)/
六列偏度峰度/ε_n），相对误差按 spec 口径 < 1e-12。
"""

from __future__ import annotations

import hashlib
import json
import re
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
    read_anchor_particles,
)
from src.line2.data_pipeline.sdds_reader import read_sdds_binary

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_DIR = REPO_ROOT / "studies/line2_particle_sr/data/anchor"
ANCHOR_COPY = ANCHOR_DIR / "injector.sdds"
EXPECTED_MD5 = ANCHOR_MD5
EXPECTED_SIZE = ANCHOR_SIZE

#: 粒子列顺序（`10` [S2] C1）
COLUMNS = ("x", "xp", "y", "yp", "t", "p")
#: Σ 的存储键 → (x, x′, y, y′) 的矩阵下标（`10` [S1] C2 的 4×4 顺序）
SIGMA_INDEX = {
    "xx": (0, 0), "xxp": (0, 1), "xy": (0, 2), "xyp": (0, 3),
    "xpxp": (1, 1), "xpy": (1, 2), "xpyp": (1, 3),
    "yy": (2, 2), "yyp": (2, 3), "ypyp": (3, 3),
}
#: spec [S1] C2 点名的 twiss.json 顶层量
TWISS_QUANTITY_KEYS = (
    "sigma_matrix", "alpha", "beta", "emit_geom", "emit_n", "beta_gamma",
    "sigma_t_s", "sigma_dp_over_p", "corr_t_p", "skew", "kurtosis",
)


@pytest.fixture(scope="module")
def anchor_particles() -> np.ndarray:
    """锚点副本的 (200001, 6) 粒子表 (x, x′, y, y′, t, p)（模块级缓存）。"""
    return read_anchor_particles(ANCHOR_COPY)


def _sample_cov(left: np.ndarray, right: np.ndarray) -> float:
    left_mean, right_mean = left.mean(), right.mean()
    return float(((left - left_mean) * (right - right_mean)).sum() / (left.size - 1))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left_centered, right_centered = left - left.mean(), right - right.mean()
    return float(
        (left_centered * right_centered).sum()
        / np.sqrt((left_centered**2).sum() * (right_centered**2).sum())
    )


def _rel(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


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
    assert abs(charge - 5.0e-10) < 1e-15  # [S2] C1 契约值
    body = np.frombuffer(raw[offset + 12 : offset + 12 + row_count * 48], dtype="<f8")
    manual = body.reshape(row_count, 6)

    page = read_sdds_binary(ANCHOR_COPY)
    assert page.row_count == row_count
    assert page.parameter_values["Charge"] == charge
    assert page.column_names == ("x", "xp", "y", "yp", "t", "p")
    parsed = np.stack([page.columns[n] for n in page.column_names], axis=1)
    assert parsed.dtype == np.float64
    np.testing.assert_array_equal(parsed, manual)  # 逐位一致


# ---------------------------------------------------------------- [S1] C2
def test_twiss_independent_recompute(anchor_particles: np.ndarray) -> None:
    """[S1] C2：twiss 主实现与独立重算逐量一致（相对误差 < 1e-12，spec 口径）。

    覆盖 spec 点名的全部量：Σ(4×4、(x,x′,y,y′) 顺序)、α/β 两平面、ε_geom/ε_n
    两平面、σ_t、σ_dp/p、corr(t,p)、六列偏度与峰度。偏度/峰度以 scipy 的独立
    实现对拍（population 矩，与 compute_twiss 的 z 分数定义同义）。
    """
    particles = anchor_particles
    twiss = compute_twiss(particles)
    col = dict(zip(COLUMNS, (particles[:, i] for i in range(6))))

    # 独立实现：手写两遍中心化积（样本协方差，分母 N−1）
    sigma = np.array(
        [
            [_sample_cov(col[a], col[b]) for b in ("x", "xp", "y", "yp")]
            for a in ("x", "xp", "y", "yp")
        ]
    )
    emit = {
        "x": float(np.sqrt(sigma[0, 0] * sigma[1, 1] - sigma[0, 1] ** 2)),
        "y": float(np.sqrt(sigma[2, 2] * sigma[3, 3] - sigma[2, 3] ** 2)),
    }
    beta_gamma = float(col["p"].mean())

    assert twiss["n_particles"] == 200_001
    assert twiss["cov_ddof"] == 1
    assert twiss["sigma_matrix"]["order"] == ["x", "xp", "y", "yp"]
    for key, (i, j) in SIGMA_INDEX.items():
        assert _rel(twiss["sigma_matrix"][key], sigma[i, j]) < 1e-12, key
    for plane in ("x", "y"):
        assert _rel(twiss["emit_geom"][plane], emit[plane]) < 1e-12, plane
        assert _rel(twiss["emit_n"][plane], beta_gamma * emit[plane]) < 1e-12, plane
    assert _rel(twiss["beta_gamma"], beta_gamma) < 1e-12
    assert _rel(twiss["beta"]["x"], sigma[0, 0] / emit["x"]) < 1e-12
    assert _rel(twiss["alpha"]["x"], -sigma[0, 1] / emit["x"]) < 1e-12
    assert _rel(twiss["beta"]["y"], sigma[2, 2] / emit["y"]) < 1e-12
    assert _rel(twiss["alpha"]["y"], -sigma[2, 3] / emit["y"]) < 1e-12
    # 纵向量（1e-12 相对口径在同量级可达成：σ_t/σ_dp,p 为同式重算，corr 为两遍积）
    assert _rel(twiss["sigma_t_s"], float(col["t"].std(ddof=1))) < 1e-12
    assert _rel(twiss["sigma_dp_over_p"], float(col["p"].std(ddof=1)) / beta_gamma) < 1e-12
    assert _rel(twiss["corr_t_p"], _corr(col["t"], col["p"])) < 1e-12
    # 偏度/峰度（六列）：scipy 独立库实现对拍
    for column in COLUMNS:
        assert _rel(twiss["skew"][column], sps.skew(col[column])) < 1e-12, column
        assert _rel(twiss["kurtosis"][column], sps.kurtosis(col[column], fisher=True)) < 1e-12, column
    # 物理画像抽查（00 [S1] 锚点参照值：pc ≈ 130 MeV、ε_n ≈ 1.26 mm·mrad、
    # σ_t = 2.41 ps、xp/yp 峰度 ≈ +2.6）
    assert abs(twiss["beta_gamma"] - 254.1622) < 0.01
    assert abs(twiss["emit_n"]["x"] - 1.26e-6) < 0.02e-6
    assert abs(twiss["sigma_t_s"] - 2.41e-12) < 0.01e-12
    assert 2.5 < twiss["kurtosis"]["xp"] < 2.7
    assert 2.5 < twiss["kurtosis"]["yp"] < 2.7


def test_archived_metadata_files_contract(anchor_particles: np.ndarray) -> None:
    """[S1] C2/C3（归档件）：twiss.json/manifest.json 落盘件字段完整且与独立重算一致。

    与上一条的分工：本测试核的是**入库产物本身**（`data/anchor/` 下已入库的两份
    JSON），上一条核的是主实现与独立重算的一致性——实现漂移或产物未同步重写
    都会在此暴露。
    """
    twiss = json.loads((ANCHOR_DIR / "twiss.json").read_text(encoding="utf-8"))
    manifest = json.loads((ANCHOR_DIR / "manifest.json").read_text(encoding="utf-8"))

    for key in TWISS_QUANTITY_KEYS:
        assert key in twiss, f"twiss.json 缺字段 {key}"
    for column in COLUMNS:
        assert column in twiss["skew"], f"twiss.json 缺 skew[{column}]"
        assert column in twiss["kurtosis"], f"twiss.json 缺 kurtosis[{column}]"
    for family in ("emit_geom", "emit_n", "beta", "alpha"):
        for plane in ("x", "y"):
            assert plane in twiss[family], f"twiss.json 缺 {family}[{plane}]"

    fresh = compute_twiss(anchor_particles)
    for key in ("beta_gamma", "sigma_t_s", "sigma_dp_over_p", "corr_t_p"):
        assert _rel(twiss[key], fresh[key]) < 1e-12, key
    for key in SIGMA_INDEX:
        assert _rel(twiss["sigma_matrix"][key], fresh["sigma_matrix"][key]) < 1e-12, key
    for family in ("emit_geom", "emit_n", "beta", "alpha"):
        for plane in ("x", "y"):
            assert _rel(twiss[family][plane], fresh[family][plane]) < 1e-12, (family, plane)
    for column in COLUMNS:
        assert _rel(twiss["skew"][column], fresh["skew"][column]) < 1e-12, column
        assert _rel(twiss["kurtosis"][column], fresh["kurtosis"][column]) < 1e-12, column
    assert twiss["n_particles"] == 200_001
    assert twiss["units_p"] == "beta_gamma"
    assert abs(twiss["charge_c"] - 5.0e-10) < 1e-15

    # [S1] C3：manifest 字段与取值（MD5/大小硬编码于 spec）
    assert manifest["md5"] == "f39d905f96abd5588091dad6dd3884b2"
    assert manifest["size_bytes"] == 9_600_397
    assert manifest["units_p"] == "beta_gamma"
    assert manifest["master_seed"] == 20260913
    assert manifest["extraction_script"] == "src/line2/data_pipeline/anchor.py"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", manifest["extraction_date"])


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
    # 幂等：重复入库产物逐位一致（twiss.json 无日期字段；manifest 仅 extraction_date 随运行日）
    twiss_before = (tmp_path / "twiss.json").read_bytes()
    ingest_anchor(tmp_path, ANCHOR_COPY, master_seed=20260913)
    assert (tmp_path / "twiss.json").read_bytes() == twiss_before
    manifest_after = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in manifest_after.items() if k != "extraction_date"} == {
        k: v for k, v in manifest.items() if k != "extraction_date"
    }


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
