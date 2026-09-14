"""L1 [S2]/[S3] 验收测试（spec line2/10）。

传输测试用 5×10³ 粒子子集（Claims 为逐粒子确定性与统计不变量，规模不影响
有效性；全量 2e5×1000 束的生产运行由管线 CLI 承担，不进单元套件）。

[S3] C1 的相位推进标定分两条互补路径验证：`_solve_f → fodo_phase_advance_deg`
自洽链（只证二分收敛）+ 本文件的独立重算（生产 lattice 元件参数 → 闭式厚透镜
矩阵 / Ocelot `optics.twiss` 周期解 / 全 2 cell 传递矩阵）。[S2] C3 的多进程
一致性由子进程重跑索引摘要验证。
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from src.line2.data_pipeline.sdds_reader import read_sdds_binary
from src.line2.data_pipeline.subsample import (
    SUBSAMPLE_K,
    SUBSAMPLE_TIERS,
    subsample_index_set,
)
from src.line2.data_pipeline.transport import (
    FODO_HALF_DRIFT_M,
    FODO_MU_DEG,
    FODO_N_CELLS,
    QUAD_LENGTH_M,
    fodo_lattice,
    fodo_phase_advance_deg,
    transport_beam,
)

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
N_TEST = 5_000
ANCHOR_COPY = REPO_ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"

#: 跨进程一致性测试的采样规模（覆盖五档 × k、多束多站；摘要计算 <0.1 s）
XPROC_N_TOTAL = 200_000
XPROC_BEAMS = (0, 3, 7)
XPROC_STATIONS = (0, 1)
XPROC_K = 2

#: 子进程脚本：导入本模块的摘要实现（同一实现、全新解释器），打印 sha256
_CHILD_SCRIPT = (
    "import sys; sys.path.insert(0, {root!r})\n"
    "from tests.unit.test_line2_subsample_transport import _cross_process_digest\n"
    "print(_cross_process_digest())\n"
)


@pytest.fixture(scope="module")
def particles() -> np.ndarray:
    page = read_sdds_binary(
        REPO_ROOT / "studies/line2_particle_sr/data/anchor/injector.sdds"
    )
    full = np.stack([page.columns[n] for n in ("x", "xp", "y", "yp", "t", "p")], axis=1)
    rng = np.random.default_rng(20260913)
    return full[rng.choice(full.shape[0], size=N_TEST, replace=False)]


def _corr_xxp(tab: np.ndarray) -> float:
    return float(np.corrcoef(tab[:, 0], tab[:, 1])[0, 1])


def _index_digest(beam_idx: int, station_idx: int, k: int = XPROC_K) -> str:
    """单 (beam, station) 的五档 × k 子样本索引 sha256（确定性摘要）。"""
    digest = hashlib.sha256()
    index_set = subsample_index_set(
        XPROC_N_TOTAL, beam_idx=beam_idx, station_idx=station_idx, k=k
    )
    for tier in SUBSAMPLE_TIERS:
        for index in index_set[tier]:
            digest.update(np.ascontiguousarray(index, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _cross_process_digest() -> str:
    """跨进程可比摘要：多束多站索引摘要的 sha256（子进程导入本函数重算）。"""
    digest = hashlib.sha256()
    for beam_idx in XPROC_BEAMS:
        for station_idx in XPROC_STATIONS:
            digest.update(_index_digest(beam_idx, station_idx).encode())
    return digest.hexdigest()


# ---------------------------------------------------------------- [S2]
def test_subsample_claims() -> None:
    """[S2] C2/C3/C4：无放回、⊂全量、确定性、同档互异（n_total 覆盖最大档）。"""
    n_total = 200_000  # 与真实锚点一致：各档子集组合空间充分（避免 N_low = n_total 的退化）
    a = subsample_index_set(n_total, beam_idx=0, station_idx=0)
    assert set(a) == set(SUBSAMPLE_TIERS) and all(len(v) == SUBSAMPLE_K for v in a.values())
    for tier, subs in a.items():
        for idx in subs:
            assert len(idx) == tier and len(np.unique(idx)) == tier  # 无放回
            assert idx[-1] < n_total
        assert all(not np.array_equal(subs[i], subs[j])
                   for i in range(SUBSAMPLE_K) for j in range(i + 1, SUBSAMPLE_K))  # C4 互异
    b = subsample_index_set(n_total, beam_idx=0, station_idx=0)
    for tier in SUBSAMPLE_TIERS:
        np.testing.assert_array_equal(a[tier][0], b[tier][0])  # C3 确定性
    c = subsample_index_set(n_total, beam_idx=1, station_idx=0)
    assert not np.array_equal(a[10_000][0], c[10_000][0])  # 不同束不同采样


def test_subsample_indices_cross_process_identical() -> None:
    """[S2] C3（多进程，`00` GC2）：子进程在同一 master_seed 下重算子样本索引逐位相同。

    与 `test_subsample_claims` 的分工：该测试只证**同进程**两次调用一致；本测试把同一
    摘要实现放进全新解释器（`sys.executable`、同一 conda 环境）重算——进程级全局态、
    哈希种子（PYTHONHASHSEED）或隐式随机源造成的偏差在此暴露。
    """
    per_pair = {
        (beam_idx, station_idx): _index_digest(beam_idx, station_idx)
        for beam_idx in XPROC_BEAMS
        for station_idx in XPROC_STATIONS
    }
    # 鉴别力前置：采样确实依赖 (beam, station)——否则"跨进程一致"可被常量实现冒充
    assert len(set(per_pair.values())) == len(per_pair), f"不同 (beam, station) 摘要重合：{per_pair}"

    expected = _cross_process_digest()
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT.format(root=str(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"子进程失败：{proc.stderr[-2000:]}"
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"子进程无输出：{proc.stderr[-2000:]}"
    assert lines[-1] == expected, f"跨进程索引不一致：父 {expected} ≠ 子 {lines[-1]}"


# ---------------------------------------------------------------- [S3]
def test_phase_advance_within_1_deg() -> None:
    """[S3] C1：实算每 cell 相位推进与目标 μ=75° 偏差 < 1°（自洽链：二分收敛）。"""
    assert abs(fodo_phase_advance_deg(_solve_f_probe()) - FODO_MU_DEG) < 1.0


def _solve_f_probe() -> float:
    from src.line2.data_pipeline.transport import _solve_f

    return _solve_f()


def _thick_quad_block(k1: float, length: float, plane: str) -> np.ndarray:
    """闭式厚透镜 2×2 块（Ocelot `uni_matrix` 定义：kx² = k1、ky² = −k1，k1>0 聚焦 x）。"""
    k = float(np.sqrt(abs(k1)))
    if (k1 > 0.0) == (plane == "x"):
        return np.array(
            [
                [np.cos(k * length), np.sin(k * length) / k],
                [-k * np.sin(k * length), np.cos(k * length)],
            ]
        )
    return np.array(
        [
            [np.cosh(k * length), np.sinh(k * length) / k],
            [k * np.sinh(k * length), np.cosh(k * length)],
        ]
    )


def _mu_deg(matrix: np.ndarray) -> float:
    """μ = arccos(tr(M)/2)（度）——单平面 2×2 传递矩阵。"""
    return float(np.degrees(np.arccos(np.clip((matrix[0, 0] + matrix[1, 1]) / 2.0, -1.0, 1.0))))


def test_phase_advance_independent_recompute() -> None:
    """[S3] C1：三条独立路径重算每 cell 相位推进，与 μ=75° 的偏差均 < 1°。

    与 `test_phase_advance_within_1_deg` 的分工：该测试用 `_solve_f()` 解出的 f 回代
    同一 `fodo_phase_advance_deg`，只证二分收敛；本测试**不复用该自洽链**，三条路径为

    (a) 生产 lattice 的元件参数（k1/l/L_d）→ 闭式厚透镜矩阵（纯 numpy，不调 Ocelot 跟踪）；
    (b) 生产 lattice → Ocelot `optics.twiss` 周期解（与 `track.lattice_track` 不同的 API）；
    (c) 生产 lattice 全 2 cell 的笔形束 4×4 传递矩阵 → 迹与特征值两条提取路径。

    鉴别力：`_solve_f()` 的二分只要收敛，自洽链恒成立；若二分符号/括号写错、或
    `fodo_lattice()` 与探针 cell 不一致（元件顺序、k1 反号、漏乘 l），(a)(b)(c) 全部
    偏离 75° 而自洽测试仍绿——例如 k1 误用 1/f 而非 1/(f·L) 时 (a) 给出 6.98°。
    """
    from ocelot.cpbd.beam import Particle
    from ocelot.cpbd.elements import Quadrupole
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.optics import twiss
    from ocelot.cpbd.track import lattice_track

    assert FODO_MU_DEG == 75.0  # spec 冻结目标（`10` [S3]）
    lattice = fodo_lattice()
    sequence = list(lattice.sequence)
    assert len(sequence) == 4 * FODO_N_CELLS
    for cell in range(FODO_N_CELLS):
        focus, drift_a, defocus, drift_b = sequence[4 * cell : 4 * cell + 4]
        assert isinstance(focus, Quadrupole) and isinstance(defocus, Quadrupole)
        assert focus.l == QUAD_LENGTH_M and defocus.l == QUAD_LENGTH_M
        assert focus.k1 > 0.0 and defocus.k1 == -focus.k1  # 正 k1 聚焦 x
        assert drift_a.l == FODO_HALF_DRIFT_M and drift_b.l == FODO_HALF_DRIFT_M
    cell_signature = [
        [(type(e).__name__, e.l, getattr(e, "k1", None)) for e in sequence[4 * c : 4 * c + 4]]
        for c in range(FODO_N_CELLS)
    ]
    assert all(cell == cell_signature[0] for cell in cell_signature), cell_signature

    # (a) 闭式厚透镜矩阵（不调 Ocelot 跟踪）
    def cell_matrix(elements, plane: str) -> np.ndarray:
        matrix = np.eye(2)
        for element in elements:
            block = (
                _thick_quad_block(element.k1, element.l, plane)
                if isinstance(element, Quadrupole)
                else np.array([[1.0, element.l], [0.0, 1.0]])
            )
            matrix = block @ matrix
        return matrix

    analytic = {plane: cell_matrix(sequence[:4], plane) for plane in ("x", "y")}
    for plane in ("x", "y"):
        assert abs(np.linalg.det(analytic[plane]) - 1.0) < 1e-12, plane  # 辛性
        assert abs(_mu_deg(analytic[plane]) - FODO_MU_DEG) < 1.0, plane
    assert abs(_mu_deg(analytic["x"]) - _mu_deg(analytic["y"])) < 1e-9  # FODO 两平面同相移

    # (b) Ocelot 光学模块的周期解（mux 为弧度）
    total_twiss = twiss(lattice)
    cell_twiss = twiss(MagneticLattice(sequence[:4]))
    mu_total = float(np.degrees(total_twiss[-1].mux))
    mu_cell = float(np.degrees(cell_twiss[-1].mux))
    assert abs(mu_cell - FODO_MU_DEG) < 1.0
    assert abs(mu_total - FODO_N_CELLS * mu_cell) < 1e-9  # 相移加性（与单位约定无关的自洽）
    assert abs(float(np.degrees(cell_twiss[-1].muy)) - FODO_MU_DEG) < 1.0

    # (c) 全 2 cell 笔形束 4×4 传递矩阵：迹与特征值两条提取路径
    pencil_eps = 1e-9
    columns = []
    for axis in ("x", "px", "y", "py"):
        particle = Particle()
        particle.x = particle.px = particle.y = particle.py = 0.0
        setattr(particle, axis, pencil_eps)
        lattice_track(lattice, particle)
        columns.append(
            np.array([particle.x, particle.px, particle.y, particle.py]) / pencil_eps
        )
    matrix = np.stack(columns, axis=1)
    assert abs(np.linalg.det(matrix[:2, :2]) - 1.0) < 1e-12
    assert abs(np.linalg.det(matrix[2:, 2:]) - 1.0) < 1e-12
    assert np.abs(matrix[2:, :2]).max() < 1e-12  # 横向零耦合
    mu_trace = _mu_deg(matrix[:2, :2]) / FODO_N_CELLS
    mu_eig = float(np.degrees(np.abs(np.angle(np.linalg.eigvals(matrix[:2, :2])[0])))) / FODO_N_CELLS
    assert abs(mu_trace - FODO_MU_DEG) < 1.0
    assert abs(mu_eig - FODO_MU_DEG) < 1.0
    assert abs(mu_trace - mu_eig) < 1e-9

    # 三条独立路径互洽（实测差异 ~1e-13 度）
    assert abs(mu_cell - _mu_deg(analytic["x"])) < 1e-6
    assert abs(mu_cell - mu_trace) < 1e-6


def test_transport_conservation_and_determinism(particles: np.ndarray) -> None:
    """[S3] C2/C3：行数守恒；同输入同参数逐位一致；纵向列 P1 档原样保留。"""
    s0_a, s1_a = transport_beam(particles, beam_idx=7)
    s0_b, s1_b = transport_beam(particles, beam_idx=7)
    assert s0_a.shape == s1_a.shape == particles.shape
    np.testing.assert_array_equal(s0_a, s0_b)
    np.testing.assert_array_equal(s1_a, s1_b)
    np.testing.assert_array_equal(s0_a[:, 4:], particles[:, 4:])  # P1: t/p 保留


def test_station_correlation_rotation(particles: np.ndarray) -> None:
    """[S3] C4：抖动束（beam ≥ 1）的站间 corr(x,x′) 差 ≥ 0.3。

    偏差记录（见 10 .decisions.md D11）：束 0（无抖动基准束）实测
    |Δcorr| ≈ 0.137 < 0.3——spec 推导的 cos(150°) 模型不含椭圆取向项，
    仅对基准束失效；按 write-spec 冻结规则记为 deviation，由验收闸口裁决
    （D11 已裁决：接受）。束 0 的偏差由下一条测试以 xfail 显式留痕。
    """
    deltas = []
    for beam_idx in (1, 2, 3, 4, 5):
        s0, s1 = transport_beam(particles, beam_idx=beam_idx)
        deltas.append(abs(_corr_xxp(s1) - _corr_xxp(s0)))
    assert min(deltas) >= 0.3, f"抖动束站间差异不足：{deltas}"


def test_station_correlation_beam0_deviation(particles: np.ndarray) -> None:
    """[S3] C4 偏差（D11 已登记、验收闸口已裁决接受）：以 xfail 显式声明，不以弱断言冒充通过。

    束 0 无抖动，站间 |Δcorr| 实测 ≈ 0.137 < 阈值 0.3。量级带（D11 记录 0.05–0.3）成立时
    `pytest.xfail` 记录实测值 ⇒ 本测试报 XFAIL（不是 PASS）；若实现/数据使 |Δcorr| ≥ 0.3，
    断言先失败 ⇒ 报红，提示 D11 可销账、本测试应删除。
    """
    s0, s1 = transport_beam(particles, beam_idx=0)
    delta = abs(_corr_xxp(s1) - _corr_xxp(s0))
    assert 0.05 <= delta < 0.3, f"束 0 偏差量级漂移出 D11 记录带（0.05–0.3）：{delta}"
    pytest.xfail(
        f"D11 已登记偏差：束 0（无抖动基准束）|Δcorr| = {delta:.4f} < spec 10 [S3] C4 阈值 0.3"
    )


def test_beam_diversity(particles: np.ndarray) -> None:
    """[S3] C5：抖动使束间站 0 corr(x,x′) 标准差 > 0.01（束间多样性不变量）。"""
    corrs = [_corr_xxp(transport_beam(particles, beam_idx=i)[0]) for i in range(20)]
    assert float(np.std(corrs)) > 0.01
    assert abs(corrs[0] - _corr_xxp(particles)) < 1e-12  # 束 0 = 无抖动基准束


def test_beam0_is_unjittered_base(particles: np.ndarray) -> None:
    """[S3] 束群契约（`00` [S1]「束 0 为无抖动基准束」）：束 0 站 0 与锚点表逐位相同。

    该不变量是 C4 束 0 偏差（D11）与 C5 基准参照的解释前提——若基准束被施加了抖动，
    两条对照的语义失效，而只比较 corr 值（上一条）未必能发现幅度小的误抖动。
    """
    s0, _ = transport_beam(particles, beam_idx=0)
    np.testing.assert_array_equal(s0, particles)

