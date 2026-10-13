"""L2 [S1] 度量族验收测试（spec line2/20 [S1]）。

M4 的主校验用已知 Σ 的 4D 高斯云投影重构（分箱偏差 ~5% 容差）：该合成云各轴
同形状同尺度（均为高斯，σ/bin 宽比例相同），轴错配的伪值恰落在真值附近，故它只
作量级护栏；**轴向配对**的鉴别力由各轴形状/量级互异的 copula 束承担（见 M4 段落
末的测试），与 `10` [S1] twiss.json 的真实锚点双实现互检见
``tests/integration/test_line2_metric_reconciliation.py``。
"""

from __future__ import annotations

import inspect
import json
import pathlib
import re
from collections.abc import Callable

import h5py
import numpy as np
import pytest
from scipy import stats as sps

from src.line2.data_pipeline.projections import (
    AXES,
    PROJECTION_ORDER,
    axis_edges,
    six_projections,
    station_edges,
)
from src.line2.metrics import alignment, baseline, decisions
from src.line2.metrics.metrics import (
    BUCKET_EDGES,
    HALO_NSIGMA,
    MetricUnderflow,
    _halo_fraction_per_projection,
    bucketed_rel_rmse,
    evaluate_group,
    group_rel_rmse,
    halo_fraction_gap,
    hf_energy_ratio_gap,
    poisson_deviance,
    reconstruct_sigma,
    rel_rmse,
    sigma_errors,
)
from tests.conftest import derive_seed

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
V2_DIR = REPO_ROOT / "studies/line2_particle_sr/data/line2_ds_v2"
V2_RESULTS = REPO_ROOT / "studies/line2_particle_sr/results"


def _synthetic_table(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.normal(0.0, 3.3e-4, n),
            rng.normal(0.0, 2.9e-5, n),
            rng.normal(0.0, 3.3e-4, n),
            rng.normal(0.0, 2.9e-5, n),
            rng.normal(0.0, 2.4e-12, n),
            rng.normal(254.16, 0.27, n),
        ],
        axis=1,
    )


# ---------------------------------------------------------------- M1
def test_m1_identity_scale_zero() -> None:
    r = np.random.default_rng(0).integers(0, 500, (128, 128)).astype(np.int32)
    assert rel_rmse(r.copy(), r) == 0.0
    x = r.copy(); x[0, 0] += 10
    base = rel_rmse(x, r)
    assert base > 0
    assert abs(rel_rmse(x * 2, r) - base) < 1e-12  # 总计数归一 → 尺度不变
    with pytest.raises(MetricUnderflow):
        rel_rmse(np.zeros_like(r), r)


def test_m1_hand_example() -> None:
    x = np.array([[1000, 3000]]); r = np.array([[2000, 2000]])
    expect = np.sqrt(np.mean((x - r) ** 2)) / np.mean(r)  # 总量相等 → 归一恒等
    assert abs(rel_rmse(x, r) - expect) < 1e-12


# ---------------------------------------------------------------- M2
def test_m2_buckets() -> None:
    r = np.zeros((128, 128), dtype=np.int32)
    r[1, :100] = 1     # B1
    r[2, :100] = 5     # B2
    r[3, :100] = 50    # B3（其余行全 0 → B0，≥100 格）
    x = r.copy()
    x[0, :10] += 1      # B0 内扰动（总计数守恒：成对增减）
    x[0, 10:20] -= 1
    x[3, :50] += 3      # B3 内扰动
    x[3, 50:100] -= 3
    out = bucketed_rel_rmse(x, r)
    assert out["B0"]["rel_rmse"] > 0 and not out["B0"]["insufficient"]
    assert out["B1"]["rel_rmse"] == 0.0 and not out["B1"]["insufficient"]
    assert out["B2"]["rel_rmse"] == 0.0
    assert out["B3"]["rel_rmse"] > 0


def _boundary_probe_reference() -> np.ndarray:
    """每条桶界两侧各 100 格的参照（其余全 0），桶界归属由格数唯一暴露。

    预期归属（`20` [S1] M2 原文：B0 r<0.5；B1 0.5≤r<2；B2 2≤r<10；B3 r≥10）：
    0.499→B0、0.5→B1、1.999999→B1、2.0→B2、9.999999→B2、10.0→B3。
    """
    r = np.zeros((128, 128), dtype=np.float64)
    for row, value in enumerate((0.499, 0.5, 1.999999, 2.0, 9.999999, 10.0)):
        r[row, :100] = value
    return r


def test_m2_bucket_edges_frozen_and_boundary_membership() -> None:
    """[S1] C2：桶界恰为 {0.5, 2, 10}，且四条边界点的归属按半开区间 [lo, hi) 划分。

    格数断言对每个边界点都可证伪：任一边界写成开区间（如 r>0.5 而非 r≥0.5）
    都会把 100 格从一桶挪到相邻桶，四个格数中至少两个随之改变。
    """
    assert BUCKET_EDGES == (0.5, 2.0, 10.0)
    assert isinstance(BUCKET_EDGES, tuple)

    out = bucketed_rel_rmse(_boundary_probe_reference().copy(), _boundary_probe_reference())
    # 全 16384 格：B0 = 16384 − 5×100（后五组非 B0），B1/B2 各含两组，B3 含一组
    assert {k: out[k]["n_cells"] for k in ("B0", "B1", "B2", "B3")} == {
        "B0": 15884, "B1": 200, "B2": 200, "B3": 100,
    }
    # 每桶格数 ≥100 → 不得被标记 insufficient（M2 的桶内有效判据）
    assert all(out[k]["insufficient"] is False for k in ("B0", "B1", "B2", "B3"))
    # 被评 == 参照 → 各桶相对 RMSE 恰为 0（归一后逐格相等）
    assert all(out[k]["rel_rmse"] == 0.0 for k in ("B0", "B1", "B2", "B3"))


def test_m2_bucket_membership_decided_by_reference_only() -> None:
    """[S1] C2：分桶归属由参照 r 唯一决定，与被评图组取值无关。

    两条证据：① 被评值在每条桶界两侧与参照**反向**摆放（参照 ≥2 的格给 0、
    其余给 50），四桶格数不变 ⇒ 分桶不读被评值；② 只在 B3 内部做总量守恒的
    成对扰动，只有 B3 的相对 RMSE 变正 ⇒ 逐格误差按其**参照**值所在桶归集。
    """
    r = _boundary_probe_reference()
    baseline = bucketed_rel_rmse(r.copy(), r)
    counts = {k: baseline[k]["n_cells"] for k in ("B0", "B1", "B2", "B3")}

    crossed = np.where(r >= 2.0, 0.0, 50.0)  # 被评值跨越全部桶界，方向与参照相反
    out_crossed = bucketed_rel_rmse(crossed, r)
    assert {k: out_crossed[k]["n_cells"] for k in ("B0", "B1", "B2", "B3")} == counts

    x = r.copy()
    x[5, 0], x[5, 1] = 13.0, 7.0  # r=10.0 的两格（B3）成对增减 → Σx == Σr
    assert x.sum() == r.sum()
    out_b3 = bucketed_rel_rmse(x, r)
    assert {k: out_b3[k]["n_cells"] for k in ("B0", "B1", "B2", "B3")} == counts
    assert out_b3["B3"]["rel_rmse"] > 0.0
    for k in ("B0", "B1", "B2"):
        assert out_b3[k]["rel_rmse"] == 0.0, f"{k} 桶不应因 B3 内扰动而非零"


# ---------------------------------------------------------------- M3
def test_m3_identity_and_hand() -> None:
    r = np.random.default_rng(1).integers(0, 300, (64, 64)).astype(np.int32)
    assert poisson_deviance(r.copy(), r) == 0.0
    c = np.array([[100, 0]]); mu = np.array([[90, 10]])  # Σμ = Σc = 100 → 归一恒等
    expect = 2 * ((100 * np.log(100 / 90) - (100 - 90)) + (0 + 10))  # c=0 项贡献 +μ
    assert abs(poisson_deviance(mu, c) - expect) < 1e-9


# ---------------------------------------------------------------- M4
def _gaussian_cloud_and_group(n: int, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    sig = np.diag([0.04, 0.09, 0.04, 0.09])          # 方差（归一化坐标）
    sig[0, 1] = sig[1, 0] = 0.5 * np.sqrt(sig[0, 0] * sig[1, 1])  # ρ(x,x′)=0.5，跨面=0
    cloud = rng.multivariate_normal(np.zeros(4), sig, size=n)
    table = np.column_stack([cloud, np.zeros(n), np.full(n, 254.16)])
    edges = station_edges(table)
    group = six_projections(table, edges)
    return cloud, table, edges, group


def test_m4_sigma_recovery_and_errors() -> None:
    cloud, table, edges, group = _gaussian_cloud_and_group(100_000, seed=5)
    sigma_hat = reconstruct_sigma(group, edges)
    sigma_ref = reconstruct_sigma(group, edges)
    d = np.sqrt(np.diag(sigma_hat))
    rho_hat = sigma_hat / np.outer(d, d)
    assert abs(rho_hat[0, 1] - 0.5) < 0.05
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        assert abs(rho_hat[i, j]) < 0.05
    errors = sigma_errors(sigma_hat, sigma_ref)
    assert all(v == 0.0 for v in errors["diag_rel"])
    assert all(v == 0.0 for v in errors["abs_delta_rho"])
    assert errors["eps4d_rel"] == 0.0
    # F1 修复：对角已知真值断言（抓对角互换/均值错位——ρ 断言对此不变）
    truth = [0.04, 0.09, 0.04, 0.09]
    for got, want in zip(np.diag(sigma_hat), truth):
        assert abs(got - want) / want < 0.05, f"对角分量偏离真值：{got} vs {want}"


def test_m4_independent_subsample_errors_bounded() -> None:
    """半样本重构 vs 全量重构：M4 报告非零但有界的偏差（分桶语义存在性）。"""
    cloud, table, edges, _ = _gaussian_cloud_and_group(100_000, seed=6)
    rng = np.random.default_rng(0)
    sub = table[rng.choice(len(table), 10_000, replace=False)]
    g_full = six_projections(table, edges)
    g_sub = six_projections(sub, edges)
    e = sigma_errors(reconstruct_sigma(g_sub, edges), reconstruct_sigma(g_full, edges))
    assert max(e["diag_rel"]) < 0.2
    assert e["eps4d_rel"] < 0.2


#: 各轴 σ 相差 ~40× 的 copula 束相关阵（六对相关系数互异，防对称性盲区）。
_ANISOTROPIC_CORR = np.array(
    [
        [1.00, 0.60, 0.30, -0.20],
        [0.60, 1.00, 0.35, 0.10],
        [0.30, 0.35, 1.00, -0.45],
        [-0.20, 0.10, -0.45, 1.00],
    ]
)


def _anisotropic_truth_table(n: int = 400_000) -> np.ndarray:
    """四轴**量级与分布形状均互异**的合成束：(n, 4) 物理坐标表。

    x/yp 取宽均匀、xp/y 取窄高斯（各轴 σ 相差 ~40×），由高斯 copula + 逐轴边缘变换
    生成（相关性结构照 _ANISOTROPIC_CORR 落位）。形状差是鉴别力的来源：若各轴同为
    高斯，生产端 `h.T` 的轴错配会把对角算成 Σ_jj·(h_i/h_j)²，而 h_i/h_j ≈ σ_i/σ_j
    恰使伪值落在真值附近；换成"均匀 vs 高斯"后 range/σ 比不同，抵消不再成立。
    """
    rng = np.random.default_rng(derive_seed(30, 1))
    z = rng.standard_normal((n, 4)) @ np.linalg.cholesky(_ANISOTROPIC_CORR).T
    u = sps.norm.cdf(z)
    x = -3.0e-4 + 6.0e-4 * u[:, 0]
    xp = 1.5e-5 + 4.0e-6 * z[:, 1]
    y = -1.2e-5 + 3.0e-6 * z[:, 2]
    yp = -4.0e-4 + 8.0e-4 * u[:, 3]
    return np.column_stack([x, xp, y, yp])


def _scaled_rel_error(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """逐分量相对误差，分母取 √(Σ_ii·Σ_jj)（非对角近零时仍有界良定义）。"""
    scale = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    return np.abs(got - ref) / scale


def test_m4_image_sigma_recovers_anisotropic_particle_moments() -> None:
    """[S1] C3/C1：各轴互异的束上，图路径 Σ 复现粒子表实测 Σ（含对角与相关分量）。

    鉴别的是生产/消费两端的**图内轴向配对**：错配时对角被算成 Σ_jj^q·(h_i/h_j)²，
    在各轴 range/σ 比互异的束上偏差达 10⁰ 量级（实测 −0.87/+7.0/+5.9/−0.86）；
    配对一致时只剩 bin 量化偏差（Sheppard h²/12 ≤ 3×10⁻⁴ 相对），实测 ≤ 1×10⁻³。
    """
    table = _anisotropic_truth_table()
    edges = station_edges(table)
    sigma_img = reconstruct_sigma(six_projections(table, edges), edges)
    sigma_ref = np.cov(table.T, ddof=0)

    rel = _scaled_rel_error(sigma_img, sigma_ref)
    diag_ref = np.diag(sigma_ref)
    diag_rel = np.abs(np.diag(sigma_img) - diag_ref) / diag_ref
    assert diag_rel.max() < 1e-2, f"Σ 对角偏离粒子表实测：{diag_rel}"
    assert rel.max() < 1e-2, f"Σ 分量偏离粒子表实测（按 √(Σ_iiΣ_jj) 归一）：{rel.max()}"
    # 非空洞守卫：真值对角互异且量级拉开 ⇒ 上面的比较不是常数对常数
    assert sigma_ref[0, 0] / sigma_ref[3, 3] < 0.7
    assert sigma_ref[1, 1] < 1e-3 * sigma_ref[0, 0]


# ---------------------------------------------------------------- M5
def test_m5_halo_gap() -> None:
    rng = np.random.default_rng(9)
    core = np.column_stack([rng.normal(0, 2e-4, (20_000, 4)), np.zeros(20_000), np.full(20_000, 254.16)])
    halo = np.column_stack([rng.normal(0, 2e-3, (5_000, 4)), np.zeros(5_000), np.full(5_000, 254.16)])
    table_ref = np.vstack([core, halo])
    edges = station_edges(table_ref)
    g_ref = six_projections(table_ref, edges)
    g_core = six_projections(core, edges)
    sigma_ref = reconstruct_sigma(g_ref, edges)
    assert halo_fraction_gap(g_core, g_ref, sigma_ref, edges) > 0.0  # 丢晕 → 正差
    assert halo_fraction_gap(g_ref, g_ref, sigma_ref, edges) == 0.0  # 一致 → 0


@pytest.fixture(scope="module")
def halo_case() -> dict:
    """窄核（2×10⁴）+ 宽晕（5×10³）参照与其纯核被评图组——M5 区域定义探针。"""
    rng = np.random.default_rng(derive_seed(20, 5))
    core = np.column_stack(
        [rng.normal(0, 2e-4, (20_000, 4)), np.zeros(20_000), np.full(20_000, 254.16)]
    )
    halo = np.column_stack(
        [rng.normal(0, 2e-3, (5_000, 4)), np.zeros(5_000), np.full(5_000, 254.16)]
    )
    table_ref = np.vstack([core, halo])
    edges = station_edges(table_ref)
    g_ref = six_projections(table_ref, edges)
    g_core = six_projections(core, edges)
    return {
        "edges": edges,
        "g_ref": g_ref,
        "g_core": g_core,
        "sigma_ref": reconstruct_sigma(g_ref, edges),
        "sigma_core": reconstruct_sigma(g_core, edges),
    }


def test_m5_region_follows_reference_sigma(halo_case: dict) -> None:
    """[S1] C4 证据①（数值）：同一被评图组配不同参照 Σ → 晕占比不同。

    区域若不由传入的 ``sigma_ref`` 构造，两条断言同时失效：缩放参照 Σ 不改变结果、
    换成被评图组自身的 Σ 也不改变结果。
    实测：gap(Σ_ref)=0.0379、gap(0.25·Σ_ref)=0.131、gap(Σ_core)=0.184。
    """
    g_core, g_ref = halo_case["g_core"], halo_case["g_ref"]
    sigma_ref, sigma_core = halo_case["sigma_ref"], halo_case["sigma_core"]
    edges = halo_case["edges"]

    gap_ref = halo_fraction_gap(g_core, g_ref, sigma_ref, edges)
    gap_shrunk = halo_fraction_gap(g_core, g_ref, 0.25 * sigma_ref, edges)
    gap_core = halo_fraction_gap(g_core, g_ref, sigma_core, edges)
    assert abs(gap_shrunk - gap_ref) > 1e-6, "区域未随参照 Σ 缩放而变"
    assert abs(gap_core - gap_ref) > 1e-6, "sigma_ref 实参未被真正消费"


def test_m5_region_is_not_translated_by_evaluated_group(halo_case: dict) -> None:
    """[S1] C4 证据①（数值·平移探针）：区域固定在参照系，不随被评图组的质心移动。

    把被评图组整体平移 30 bin（质心随之移出参照椭圆中心；二阶中心矩对平移不变，
    故鉴别力来自一阶矩）：区域若由被评矩定义，则区域跟着平移、晕占比回到未平移
    值；区域由参照定义时，被评质量被推出固定椭圆，晕占比升到 ~0.96（实测；
    未平移时 0.038）。
    """
    g_core, g_ref = halo_case["g_core"], halo_case["g_ref"]
    sigma_ref, edges = halo_case["sigma_ref"], halo_case["edges"]

    def centroid_bins(img: np.ndarray, axis: int) -> float:
        p = img.astype(np.float64)
        p /= p.sum()
        idx = np.arange(img.shape[axis], dtype=np.float64)
        return float((p.sum(axis=1 - axis) * idx).sum())

    gap_base = halo_fraction_gap(g_core, g_ref, sigma_ref, edges)
    shift = 30
    rolled = np.roll(g_core, shift, axis=(1, 2))
    # 平移确实移动了被评图组的质心（否则本探针无鉴别力）
    assert abs(centroid_bins(rolled[0], 0) - centroid_bins(g_core[0], 0) - shift) < 1e-9
    assert abs(centroid_bins(rolled[0], 1) - centroid_bins(g_core[0], 1) - shift) < 1e-9

    gap_rolled = halo_fraction_gap(rolled, g_ref, sigma_ref, edges)
    assert gap_rolled > 0.5 > gap_base, (
        f"区域疑似随被评质心平移：未平移 {gap_base}、平移后 {gap_rolled}"
    )


def test_m5_region_axes_match_stored_group_layout() -> None:
    """[S1] C4：M5 的椭圆按 (i,j) 与**存储布局**的图内轴配对（轴 0 = aj、轴 1 = ai）。

    构造：解析 4D 高斯参照束 + 逐轴**不等倍数**网格（±8σ/±8σ/±12σ/±8σ）——各轴 σ/h
    之比互异，故"轴 0 = aj"与"轴 0 = ai"两种读法不再互相抵消；被评图组 = 参照束沿
    y′（轴 3）平移 1.5σ（λ ≈ 2.3–2.8，该量级对轴错配最敏感），区域仍由参照 Σ 定义
    ⇒ 每投影的椭圆外占比有闭式解（非中心 χ²₂ 的尾概率，λ = 平移的 Mahalanobis 范数²；
    不含 y′ 的三个投影 λ = 0），度量 = 六投影 |占比_被评 − 占比_参照| 的平均，
    占比_参照 = e⁻⁸（4σ 椭圆外）。

    实测：按存储布局 0.00651（解析 0.00637，偏差 2%——来自分箱 Σ 与离散化）；
    按"轴 0 = ai"误读 0.01218（+91%）。网格取 ±8σ 以上而非站级 ±3.5σ：
    4σ 椭圆的近轴外沿超出站级域，域外计数被丢弃会使闭式解失效。
    """
    sigma = np.array([1.0, 0.05, 0.6, 0.1])
    multiplies = (8.0, 8.0, 12.0, 8.0)
    edges = np.stack([axis_edges(-m * s, m * s) for m, s in zip(multiplies, sigma)])
    corr = np.array(
        [[1.00, 0.60, 0.30, -0.20], [0.60, 1.00, 0.35, 0.10],
         [0.30, 0.35, 1.00, -0.45], [-0.20, 0.10, -0.45, 1.00]]
    )
    rng = np.random.default_rng(derive_seed(30, 3))
    table = (rng.standard_normal((400_000, 4)) @ np.linalg.cholesky(corr).T) * sigma

    axis, delta = 3, 1.5 * float(sigma[3])
    shifted = table.copy()
    shifted[:, axis] += delta
    group_ref = six_projections(table, edges)
    group_eval = six_projections(shifted, edges)

    lam = []
    for ai, aj in PROJECTION_ORDER:
        i, j = AXES.index(ai), AXES.index(aj)
        if axis in (i, j):
            partner = j if i == axis else i
            lam.append((delta / sigma[axis]) ** 2 / (1.0 - corr[axis, partner] ** 2))
        else:
            lam.append(0.0)
    per_proj = [
        abs(float(sps.ncx2.sf(HALO_NSIGMA**2, 2, lam_i)) - np.exp(-8.0)) for lam_i in lam
    ]
    expected = float(np.mean(per_proj))
    assert sum(l > 0 for l in lam) == 3, f"平移轴未落到预期的三个投影：{lam}"
    assert expected > 1e-3, "解析期望过小，探针退化"

    sigma_ref = reconstruct_sigma(group_ref, edges)
    gap = halo_fraction_gap(group_eval, group_ref, sigma_ref, edges)
    assert abs(gap - expected) < 2.5e-3, (
        f"M5 椭圆轴序与图组存储布局不符：gap={gap}、解析={expected}"
    )


def test_m5_evaluated_moments_have_no_entry_point() -> None:
    """[S1] C4 证据②（结构）：被评图组的矩在 M5 的区域定义中没有入口。

    签名层：``halo_fraction_gap`` 只接受 (group, group_ref, sigma_ref, edges)——
    没有 sigma_eval / moments 之类的被评侧区域参数；区域原语
    ``_halo_fraction_per_projection`` 的区域由 (sigma2d, mean2d, centers0,
    centers1) 四个数据实参给定，被评图组只以 ``img``（计数权重）身份进入。
    源码层：区域 2×2 子块取自 ``sigma_ref``、区域均值取自 ``group_ref``，且函数
    体内不出现 ``reconstruct_sigma``（被评矩的唯一计算入口）。

    若有人把被评矩塞进区域定义，会被以下断言抓住：
    - 改成 ``reconstruct_sigma(group, edges)`` 定义区域 → 本测试的源码断言
      （"reconstruct_sigma" not in src）直接失败；同时
      :func:`test_m5_region_follows_reference_sigma` 的 gap(Σ_core) ≠ gap(Σ_ref)
      与 :func:`test_m5_region_is_not_translated_by_evaluated_group` 的平移探针
      同时失效（区域会随被评组缩放/平移，两条断言都翻红）。
    - 新增被评侧区域形参（如 sigma_eval）→ 本测试的签名断言失败。
    """
    outer = list(inspect.signature(halo_fraction_gap).parameters)
    assert outer == ["group", "group_ref", "sigma_ref", "edges"], outer
    inner = list(inspect.signature(_halo_fraction_per_projection).parameters)
    assert inner == ["img", "sigma2d", "mean2d", "centers0", "centers1"], inner

    src = inspect.getsource(halo_fraction_gap)
    assert "reconstruct_sigma" not in src, "M5 内部不得计算被评图组的矩"
    assert "sigma_ref[i, i]" in src and "sigma_ref[j, j]" in src, "区域子块须取自参照 Σ"
    assert "group_ref[gi]" in src, "区域均值须取自参照图组"


# ---------------------------------------------------------------- M6
def test_m6_hf_gap() -> None:
    img = np.zeros((128, 128), dtype=np.int32); img[64, 64] = 50_000  # 亮尖峰
    img_smooth = np.zeros_like(img); img_smooth[64, 64] = 20_000; img_smooth[63:66, 63:66] += 7_000
    group_a = np.stack([img] * 6); group_b = np.stack([img_smooth] * 6)
    ref = np.stack([img] * 6)
    assert hf_energy_ratio_gap(group_a, ref) == 0.0
    assert hf_energy_ratio_gap(group_b, ref) > 0.0  # 平滑版高频比更低


# ---------------------------------------------------------------- evaluate_group
def test_evaluate_group_underflow_and_identity() -> None:
    table = _synthetic_table(5_000, seed=11)
    edges = station_edges(table)
    g = six_projections(table, edges)
    out = evaluate_group(g, g, edges)
    assert out["underflow"] is False
    assert out["M1"] == 0.0 and out["M3"] == 0.0 and out["M5"] == 0.0 and out["M6"] == 0.0
    zero = np.zeros_like(g)
    out_z = evaluate_group(zero, g, edges)
    assert out_z["underflow"] is True


def test_evaluate_group_m1_is_six_projection_average() -> None:
    """[S1] M1：图组级 M1 = 六投影**各自归一后**各算再平均（`20` [S1] M1 原文），
    不是"整组一次性归一 + 整体 RMSE"。

    两口径只在逐投影总计数成比例失衡时才分叉：把被评图组第 0 投影放大 3× 后，
    逐投影归一使该投影仍是精确匹配（M1 降到浮点噪声级），整组归一则会留下非零残差。
    故本断言对被消费的是哪一个口径具有唯一性（整组口径在此必为正值）。
    """
    table = _synthetic_table(5_000, seed=12)
    edges = station_edges(table)
    g_ref = six_projections(table, edges)
    g_eval = g_ref.copy()
    g_eval[0] = g_eval[0] * 3

    out = evaluate_group(g_eval, g_ref, edges)
    whole = rel_rmse(g_eval, g_ref)
    assert whole > 0.0, "整组口径在此必须非零，否则本条无鉴别力"
    assert out["M1"] == group_rel_rmse(g_eval, g_ref)
    assert out["M1"] < 1e-9 * whole, (
        f"图组 M1 = {out['M1']} 未按逐投影归一（整组口径给 {whole}）"
    )


# ---------------------------------------------------------------- [S1] C1
@pytest.fixture(scope="module")
def metric_inputs() -> dict:
    """确定性/归一探针的共享输入：参照图组（2×10⁴）与被评图组（5×10³ 无放回子样本）。"""
    table = _synthetic_table(20_000, seed=derive_seed(20, 1))
    edges = station_edges(table)
    g_ref = six_projections(table, edges)
    rng = np.random.default_rng(derive_seed(20, 2))
    g_eval = six_projections(table[rng.choice(len(table), 5_000, replace=False)], edges)
    return {
        "edges": edges,
        "g_ref": g_ref,
        "g_eval": g_eval,
        "ref_img": g_ref[0],
        "eval_img": g_eval[0],
        "sigma_ref": reconstruct_sigma(g_ref, edges),
    }


#: 六个预注册度量的单一函数入口（M1 含单图与图组两级；M4 含 Σ 重构与 11 维误差）。
_METRIC_CALLS: dict[str, Callable[[dict], object]] = {
    "M1_rel_rmse": lambda d: rel_rmse(d["eval_img"], d["ref_img"]),
    "M1_group_rel_rmse": lambda d: group_rel_rmse(d["g_eval"], d["g_ref"]),
    "M2_bucketed_rel_rmse": lambda d: bucketed_rel_rmse(d["eval_img"], d["ref_img"]),
    "M3_poisson_deviance": lambda d: poisson_deviance(d["eval_img"], d["ref_img"]),
    "M4_reconstruct_sigma": lambda d: reconstruct_sigma(d["g_eval"], d["edges"]),
    "M4_sigma_errors": lambda d: sigma_errors(
        reconstruct_sigma(d["g_eval"], d["edges"]), d["sigma_ref"]
    ),
    "M5_halo_fraction_gap": lambda d: halo_fraction_gap(
        d["g_eval"], d["g_ref"], d["sigma_ref"], d["edges"]
    ),
    "M6_hf_energy_ratio_gap": lambda d: hf_energy_ratio_gap(d["g_eval"], d["g_ref"]),
    "evaluate_group": lambda d: evaluate_group(d["g_eval"], d["g_ref"], d["edges"]),
}

_INPUT_KEYS = ("g_eval", "g_ref", "edges", "ref_img", "eval_img", "sigma_ref")


def _canonical(value: object) -> object:
    """把度量返回值折成可 ``==`` 逐位比较的结构（ndarray 取原始字节，浮点不装箱）。"""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.tobytes())
    if isinstance(value, dict):
        return ("dict", {k: _canonical(v) for k, v in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_canonical(v) for v in value))
    return ("scalar", value)


@pytest.mark.parametrize("name", sorted(_METRIC_CALLS))
def test_metric_repeat_call_bitwise_identical(name: str, metric_inputs: dict) -> None:
    """[S1] C1（确定性）：同输入两次连续调用逐位一致，且度量不得就地改写输入。

    逐位比较用 :func:`_canonical` 的原始字节/精确浮点相等（非 approx）；
    输入快照断言抓住「首次调用把归一化写回实参」这类使第二次调用变值的实现。
    """
    call = _METRIC_CALLS[name]
    arrays = [metric_inputs[k] for k in _INPUT_KEYS]
    snapshots = [np.array(a, copy=True) for a in arrays]

    first = _canonical(call(metric_inputs))
    second = _canonical(call(metric_inputs))
    assert first == second, f"{name} 两次调用结果不逐位一致"
    for key, arr, snap in zip(_INPUT_KEYS, arrays, snapshots):
        np.testing.assert_array_equal(arr, snap, err_msg=f"{name} 就地改写了输入 {key}")


def _m2_summary(x: np.ndarray, r: np.ndarray) -> tuple:
    out = bucketed_rel_rmse(x, r)
    vals = tuple(
        out[k]["rel_rmse"] for k in ("B0", "B1", "B2", "B3") if not out[k]["insufficient"]
    )
    assert len(vals) == 4, f"参照桶覆盖不足，M2 归一探针退化：{out}"
    return vals


#: 总计数归一内置的度量（`20` [S1] C1：M1–M5；M6 为比值型，不在归一契约内）。
_SCALED_CALLS: dict[str, Callable[[dict, np.ndarray], object]] = {
    "M1": lambda d, g: group_rel_rmse(g, d["g_ref"]),
    "M2": lambda d, g: _m2_summary(g[0], d["ref_img"]),
    "M3": lambda d, g: poisson_deviance(g[0], d["ref_img"]),
    "M4": lambda d, g: reconstruct_sigma(g, d["edges"]),
    "M5": lambda d, g: halo_fraction_gap(g, d["g_ref"], d["sigma_ref"], d["edges"]),
}


@pytest.mark.parametrize("name", sorted(_SCALED_CALLS))
def test_total_count_normalization_is_internal(name: str, metric_inputs: dict) -> None:
    """[S1] C1（归一内置）：被评图组整体缩放 3× 后度量值不变——调用方无归一入口。"""
    call = _SCALED_CALLS[name]
    g_eval = metric_inputs["g_eval"]
    base = call(metric_inputs, g_eval)
    scaled = call(metric_inputs, g_eval.astype(np.float64) * 3.0)
    assert np.asarray(base, dtype=np.float64).size > 0, f"{name} 无有效返回值（空断言）"
    np.testing.assert_allclose(
        np.asarray(scaled, dtype=np.float64).ravel(),
        np.asarray(base, dtype=np.float64).ravel(),
        rtol=1e-12,
        atol=0.0,
        err_msg=f"{name} 未内置总计数归一",
    )


@pytest.mark.parametrize(
    "func",
    [
        rel_rmse,
        group_rel_rmse,
        bucketed_rel_rmse,
        poisson_deviance,
        reconstruct_sigma,
        halo_fraction_gap,
        hf_energy_ratio_gap,
        evaluate_group,
    ],
)
def test_metric_signatures_expose_no_normalization_entry(func: Callable) -> None:
    """[S1] C1（归一内置·结构）：度量签名不得出现归一/缩放/权重类实参。"""
    forbidden = ("norm", "scale", "total", "weight", "factor")
    leaked = [
        n for n in inspect.signature(func).parameters if any(w in n.lower() for w in forbidden)
    ]
    assert not leaked, f"{func.__name__} 暴露归一入口：{leaked}"


# ---------------------------------------------------------------- [S3] C1
#: N_eff 类字段名的探针（白名单校核必须逐类拒绝它们）
_N_EFF_FIELDS = ("n_eff", "n_eff_flag", "best_val_n_eff", "n_eff_median", "val_n_eff")


def test_neff_not_in_gate_inputs() -> None:
    """`50` [S3] C1：判定输入按类别白名单化，选型/路由/门禁三类均不含 N_eff。

    白名单是判定侧的唯一取数入口，本测试对它做静态断言（不依赖任何一次具体判定的
    运行）：① 三类白名单的冻结内容（选型/路由 = val 段 val_nll；门禁 = test 段的
    逐束 M1 err / 13 维 obs / 对齐仪器读数 / B-const 逐束误差）；② 任何键名命中
    N_eff 类模式即不在白名单，N_eff 只出现在 ``DISCLOSURE_ONLY``；③ 两段不交叉
    （沿用 `40` D6/D12）；④ 校核函数对三类都拒绝 N_eff 字段与白名单外键；
    ⑤ 三类各自的消费点（选型 ``selection_reading``、路由
    ``route_by_selection_reading``、门禁的
    对齐仪器与 B-const 读数）在签名与源码层面都不出现 N_eff。
    """
    # ① 冻结白名单
    assert decisions.DECISION_KINDS == ("selection", "routing", "gate")
    assert decisions.inputs_for("selection") == ("val_nll",)
    assert decisions.inputs_for("routing") == ("val_nll",)
    assert decisions.inputs_for("gate") == (
        "beam_err_m1", "obs13", "alignment", "bconst_beam_err",
    )
    # ② N_eff 只作披露字段
    for kind in decisions.DECISION_KINDS:
        for key in decisions.inputs_for(kind):
            assert not re.search(r"(^|_)n_eff(_|$)", key), (kind, key)
    assert "n_eff" in decisions.DISCLOSURE_ONLY
    assert set(decisions.DISCLOSURE_ONLY) == {"n_eff", "n_eff_flag"}
    # ③ 两段不交叉
    assert decisions.segment_for("selection") == "val"
    assert decisions.segment_for("routing") == "val"
    assert decisions.segment_for("gate") == "test"
    assert "val_nll" not in decisions.inputs_for("gate")
    assert not any(k.startswith("test_") for k in decisions.inputs_for("selection"))
    # ④ 校核函数：合法键通过、N_eff 与白名单外键一律拒绝（三类全覆盖）
    for kind in decisions.DECISION_KINDS:
        allowed = decisions.inputs_for(kind)
        assert decisions.assert_inputs(kind, list(allowed)) == allowed
        assert decisions.assert_inputs(kind, []) == allowed
        for field in _N_EFF_FIELDS:
            with pytest.raises(decisions.DecisionInputError, match="N_eff"):
                decisions.assert_inputs(kind, [*allowed, field])
        with pytest.raises(decisions.DecisionInputError, match="不在白名单"):
            decisions.assert_inputs(kind, ["mystery_reading"])
    # ⑤ 消费点的静态断言（签名 + 源码）
    consumers = {
        "selection": (decisions.selection_reading, decisions.selection_payload),
        "routing": (decisions.route_by_selection_reading,),
        "gate": (alignment.alignment_instrument, alignment.scramble_pairs,
                 baseline.bconst_group, baseline.bconst_records),
    }
    for kind, funcs in consumers.items():
        for func in funcs:
            params = list(inspect.signature(func).parameters)
            assert not any(re.search(r"(^|_)n_eff(_|$)", p) for p in params), (kind, func.__name__)
            assert "n_eff" not in inspect.getsource(func).lower(), (kind, func.__name__)
    assert list(inspect.signature(decisions.route_by_selection_reading).parameters) == ["per_arm"]
    # 选型读数的唯一准绳是 val_nll：签名不接受任何其它判定读数
    assert list(inspect.signature(decisions.selection_reading).parameters) == ["per_tier"]


def test_route_by_selection_reading_is_the_only_routing_input() -> None:
    """`50` [S3] C1/S5：路由只吃 val 段 val_nll；同值按臂名定序（确定性）。"""
    assert decisions.route_by_selection_reading({"joint": 1.0, "independent": 2.0}) == "joint"
    assert decisions.route_by_selection_reading({"joint": 2.0, "independent": 1.0}) == "independent"
    assert decisions.route_by_selection_reading({"indep": 1.0, "independent": 1.0}) == "indep"
    # 输入顺序不影响结论（排序键 = (val_nll, 臂名)）
    shuffled = {"b_arm": 0.5, "a_arm": 0.5}
    assert decisions.route_by_selection_reading(shuffled) == "a_arm"
    assert decisions.route_by_selection_reading(dict(reversed(list(shuffled.items())))) == "a_arm"
    with pytest.raises(ValueError, match="无臂"):
        decisions.route_by_selection_reading({})
    with pytest.raises(ValueError, match="非有限"):
        decisions.route_by_selection_reading({"joint": float("nan"), "independent": 1.0})


# ---------------------------------------------------------------- [S3] C2
def _tier_matrix(beam_values: list[list[float]]) -> np.ndarray:
    return np.asarray(beam_values, dtype=np.float64)


def test_val_nll_three_tier_mean() -> None:
    """`50` [S3] C2：val_nll = 逐 (束, 子样本) NLL → 每档逐束中位数 → 全束中位数 → 三档等权平均。

    构造把三种读法分开（每档各 5 束 × 5 子样本）：

    - 档 3×10³ 的束中位数为 [0, 1, 5, 6, 7]：**先逐束取中位数**再对束取中位数给 5；
      把该档 25 个读数池化后取中位数给 7（`50` [S1] 的聚合惯性是"先逐束收子样本、
      再对束取统计量"，池化读法是本断言的鉴别对象）；
    - 三档档内中位数 = 5 / 2 / 7 ⇒ 等权算术平均 14/3；跨档池化读法与对全部读数取
      均值都得不到该值（分别是 7.0 与 4.0 量级）。
    """
    tier_3000 = _tier_matrix([
        [0.0, 0.0, 0.0, 50.0, 50.0],
        [1.0, 1.0, 1.0, 50.0, 50.0],
        [-100.0, -100.0, 5.0, 1000.0, 1000.0],
        [6.0, 6.0, 6.0, 1000.0, 1000.0],
        [7.0, 7.0, 7.0, 1000.0, 1000.0],
    ])
    tier_10000 = _tier_matrix([[1.0] * 5, [2.0] * 5, [3.0] * 5])
    tier_100000 = _tier_matrix([[7.0] * 5])
    per_tier = {3_000: tier_3000, 10_000: tier_10000, 100_000: tier_100000}

    # 逐档分量：逐束中位数 → 全束中位数
    np.testing.assert_array_equal(np.median(tier_3000, axis=1), [0.0, 1.0, 5.0, 6.0, 7.0])
    assert decisions.tier_reading(per_tier, 3_000) == 5.0
    assert decisions.tier_reading(per_tier, 10_000) == 2.0
    assert decisions.tier_reading(per_tier, 100_000) == 7.0
    # 池化读法在本构造上给 7（3e3 档）——先把两读法分开，再断言用的是哪一支
    assert float(np.median(tier_3000)) == 7.0
    assert decisions.selection_reading(per_tier) == pytest.approx((5.0 + 2.0 + 7.0) / 3.0)
    assert decisions.selection_reading(per_tier) != (7.0 + 2.0 + 7.0) / 3.0
    pooled_all = float(np.median(np.concatenate([v.ravel() for v in per_tier.values()])))
    assert decisions.selection_reading(per_tier) != pytest.approx(pooled_all)

    # 落盘载荷：附各档分项（`50` [S3] C2 明文）与冻结口径快照
    payload = decisions.selection_payload(per_tier)
    assert payload["val_nll"] == pytest.approx(decisions.selection_reading(per_tier))
    assert payload["tiers"] == [3_000, 10_000, 100_000] == list(decisions.VAL_NLL_TIERS)
    assert payload["station"] == 0 and payload["subs"] == [0, 1, 2, 3, 4]
    assert payload["segment"] == "val"
    assert payload["per_tier"]["3000"] == {"median_nll": 5.0, "n_beams": 5, "n_subs": 5}
    assert payload["per_tier"]["10000"] == {"median_nll": 2.0, "n_beams": 3, "n_subs": 5}
    assert json.loads(json.dumps(payload)) == payload  # 落盘可往返

    # 契约护栏：缺档 / 非矩阵 / 站与子样本偏离冻结口径一律显式报错
    with pytest.raises(ValueError, match="缺档"):
        decisions.selection_reading({3_000: tier_3000, 10_000: tier_10000})
    with pytest.raises(ValueError, match="矩阵"):
        decisions.tier_reading({3_000: np.zeros(5)}, 3_000)
    with pytest.raises(ValueError, match="站 0"):
        decisions.selection_payload(per_tier, station=1)
    with pytest.raises(ValueError, match="subs"):
        decisions.selection_payload(per_tier, subs=(0, 1, 2, 3))


# ---------------------------------------------------------------- [S3] C4
def test_bconst_in_baseline_family() -> None:
    """`50` [S3] C4：B-const（两站）入冻结基线族，并在 v2 公共域上可复算。

    四层证据：① 族成员与实现点冻结（B-const → ``baseline.bconst_group``，且
    ``baseline_family()`` 返回副本、改不动冻结族）；② 两站均值图 = train 束全量目标图
    逐投影归一后的均值（手写循环独立重算，逐位一致），且所消费的 bin 边界逐位等于
    manifest 的 v2 公共域边界（确认是 v2 口径而非 v1 逐束自定标）；③ 150 个 test 束
    的逐束 M1 误差向量（两站）与抽样独立复算一致；④ 落盘产物与重算对账（组图、
    逐束误差中位数）。

    v1 边界的同口径读数为站 0 ≈ 0.194、站 1 ≈ 0.035（pre3/exp3）；v2 公共域下实测
    中位数为站 0 ≈ 1.575、站 1 ≈ 1.510——本测试不断言该数值（属研究结果），只断言
    结构、复算一致与 v2 边界口径。
    """
    # ① 冻结族
    assert set(baseline.BASELINE_FAMILY) == {"B-raw", "B-const", "B-curve", "B-moment"}
    assert baseline.BASELINE_FAMILY["B-const"] == "src.line2.metrics.baseline.bconst_group"
    family = baseline.baseline_family()
    family["B-const"] = "tampered"
    assert baseline.BASELINE_FAMILY["B-const"] == "src.line2.metrics.baseline.bconst_group"

    if not (V2_DIR / "manifest.json").is_file():
        pytest.skip(f"v2 数据集不在本机：{V2_DIR}（*.h5 不入库）")
    manifest = json.loads((V2_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_version"] == "line2_ds_v2"
    train_beams = [int(b) for b in manifest["splits"]["train"]]
    test_beams = [int(b) for b in manifest["splits"]["test"]]
    assert len(train_beams) == 700 and len(test_beams) == 150

    # ② 两站均值图：可复算 + 消费 v2 公共域边界
    groups: dict[int, np.ndarray] = {}
    for station in (0, 1):
        group = baseline.bconst_group(V2_DIR, station, beam_ids=train_beams)
        assert group.shape == (6, 128, 128) and (group >= 0).all()
        np.testing.assert_allclose(group.sum(axis=(1, 2)), 1.0, rtol=0.0, atol=1e-12)
        acc = np.zeros_like(group)
        edges = np.stack([
            np.asarray(manifest["edges"][f"station{station}"][axis], dtype=np.float64)
            for axis in AXES
        ])
        with h5py.File(V2_DIR / "train.h5", "r") as h5:
            for beam_id in train_beams:
                stg = h5[f"beam_{beam_id:04d}"][f"station{station}"]
                np.testing.assert_array_equal(
                    stg["target_full"].attrs["bin_edges"], edges
                )
                raw = stg["target_full"][...].astype(np.float64)
                acc += raw / raw.sum(axis=(1, 2), keepdims=True)
        np.testing.assert_array_equal(group, acc / len(train_beams))
        groups[station] = group

    # ③ 逐束误差向量（150 test 束 × 两站）与抽样独立复算
    for station in (0, 1):
        errors = baseline.bconst_beam_errors(
            groups[station], V2_DIR, station, beam_ids=test_beams
        )
        assert sorted(errors) == sorted(test_beams)
        values = np.asarray([errors[b] for b in test_beams], dtype=np.float64)
        assert np.isfinite(values).all() and (values > 0.0).all()
        with h5py.File(V2_DIR / "test.h5", "r") as h5:
            for beam_id in test_beams[:3]:
                ref = h5[f"beam_{beam_id:04d}"][f"station{station}"]["target_full"][...]
                assert errors[beam_id] == group_rel_rmse(groups[station], ref)

    # ④ 落盘产物对账（另见 :func:`test_bconst_product_matches_recomputation`）
    artifact = V2_RESULTS / "bconst_v2.json"
    if not artifact.is_file():
        pytest.skip(f"B-const 产物不在本机：{artifact.name}（results/ 不入库）")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["n_train_beams"] == 700
    assert sorted(int(k) for k in payload["station"]) == [0, 1]
    # 门禁输入的 C1 白名单性质：B-const 读数不含任何 N_eff 字段
    assert "n_eff" not in artifact.read_text(encoding="utf-8")


def test_bconst_product_matches_recomputation() -> None:
    """`50` [S3] C4 的落盘对账：两站均值图与 150×2 个逐束误差向量可由数据复算。

    产物（``bconst_v2.json`` + ``bconst_v2_groups.npz``）是 G1/G4/G5 的判定参照，
    故其组图与逐束误差须与从 v2 数据重算的值逐位（组图）／1e-12 相对（误差）一致。
    """
    artifact = V2_RESULTS / "bconst_v2.json"
    npz_path = V2_RESULTS / "bconst_v2_groups.npz"
    if not artifact.is_file() or not npz_path.is_file():
        pytest.skip(f"B-const 产物不在本机：{artifact.name}（results/ 不入库）")
    if not (V2_DIR / "manifest.json").is_file():
        pytest.skip(f"v2 数据集不在本机：{V2_DIR}（*.h5 不入库）")
    manifest = json.loads((V2_DIR / "manifest.json").read_text(encoding="utf-8"))
    train_beams = [int(b) for b in manifest["splits"]["train"]]
    test_beams = [int(b) for b in manifest["splits"]["test"]]
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    stored = np.load(npz_path)
    for station in (0, 1):
        group = baseline.bconst_group(V2_DIR, station, beam_ids=train_beams)
        np.testing.assert_allclose(
            stored[f"station{station}"], group, rtol=1e-12, atol=0.0
        )
        errors = baseline.bconst_beam_errors(
            group, V2_DIR, station, beam_ids=test_beams
        )
        per_beam = payload["station"][str(station)]["per_beam_m1"]
        assert sorted(int(b) for b in per_beam) == sorted(test_beams)
        np.testing.assert_allclose(
            np.asarray([per_beam[str(b)] for b in test_beams], dtype=np.float64),
            np.asarray([errors[b] for b in test_beams], dtype=np.float64),
            rtol=1e-12,
            atol=0.0,
        )
        assert payload["station"][str(station)]["err_median"] == pytest.approx(
            float(np.median([errors[b] for b in test_beams])), rel=1e-12
        )


# ---------------------------------------------------------------- [S3] C3 支撑
def test_signed_hf_gap_is_the_signed_m6_reading() -> None:
    """`50` [S3] C3 的符号量口径：逐投影符号差的平均，且 |signed| ≤ M6 的 |Δ| 平均。

    去偏需要符号（M6 在 13 维向量里先取 |·|），本测试钉住两个口径的关系：符号量按
    逐投影差的平均独立重算；三角不等式给出 |signed| ≤ gap（各投影同号时取等）。
    """
    from src.line2.metrics.metrics import hf_energy_ratios, signed_hf_energy_ratio_gap

    rng = np.random.default_rng(derive_seed(50, 11))
    group_ref = rng.integers(0, 40, (6, 128, 128)).astype(np.int32)
    assert signed_hf_energy_ratio_gap(group_ref, group_ref) == 0.0
    for shift in (-3, 0, 3):
        group = (group_ref + shift).clip(0).astype(np.int32)
        ratios_g, ratios_r = hf_energy_ratios(group), hf_energy_ratios(group_ref)
        signed = signed_hf_energy_ratio_gap(group, group_ref)
        assert signed == pytest.approx(float(np.mean(ratios_g - ratios_r)), rel=1e-12)
        gap = hf_energy_ratio_gap(group, group_ref)
        assert gap == pytest.approx(float(np.mean(np.abs(ratios_g - ratios_r))), rel=1e-12)
        assert abs(signed) <= gap + 1e-15
        # 同号构造（整体更平滑 ⇒ 六个投影的高频比一致下降）下两口径取等
        if np.all(np.sign(ratios_g - ratios_r) == np.sign(signed)) and signed != 0.0:
            assert abs(signed) == pytest.approx(gap, rel=1e-12)


def test_m6_gap_matches_the_v1_per_projection_formula() -> None:
    """`50` Global Constraints（代码复用与语义不变）：M6 的数值与 v1 逐投影写法逐位一致。

    ``hf_energy_ratios`` 抽出后，``hf_energy_ratio_gap`` 改为"先各算比值再取平均"；
    本测试以 v1 的原始写法（逐投影 `gaussian_filter` 高通 + 逐投影取 |Δ| 后再平均）
    作为独立 oracle，随机图组上要求逐位相等——防"重构顺手改了 M6 语义"。
    """
    from scipy.ndimage import gaussian_filter

    from src.line2.metrics.metrics import _HF_SIGMA_BINS

    def legacy(group: np.ndarray, group_ref: np.ndarray) -> float:
        gaps = []
        for gi in range(group.shape[0]):
            ratios = []
            for img in (group[gi].astype(np.float64), group_ref[gi].astype(np.float64)):
                hp = img - gaussian_filter(img, _HF_SIGMA_BINS)
                ratios.append(float(np.abs(hp).sum() / max(img.sum(), 1e-12)))
            gaps.append(abs(ratios[0] - ratios[1]))
        return float(np.mean(gaps))

    rng = np.random.default_rng(derive_seed(50, 12))
    for scale in (10, 1000):
        group = rng.integers(0, scale, (6, 128, 128)).astype(np.int32)
        group_ref = rng.integers(0, scale, (6, 128, 128)).astype(np.int32)
        assert hf_energy_ratio_gap(group, group_ref) == legacy(group, group_ref)
        assert hf_energy_ratio_gap(group, group) == 0.0
