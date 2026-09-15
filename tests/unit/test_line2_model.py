"""L3b [S1]/[S2]/[S4] 验收测试（spec line2/30）：前向契约、参数上限、条件化防泄露、损失、变体对照。

CPU 运行：前向类用例取缩配网络（base=24、num_levels=3、blocks=1）；只构造不前向的
参数量用例按生产档取值（[S1] C2 上限、[S4] C1 两版对齐）。合成粒子表逐轴量级互异
（x/y 为 m 级、xp/yp 为 rad 级）且六对相关系数互异——使"通道 ↔ 投影 ↔ 轴对"的
任何错置都会改变可观测输出，而非落在对称性盲区里。
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import numpy as np
import pytest
import torch

from src.line2.data_pipeline.projections import (
    AXES,
    PROJECTION_ORDER,
    six_projections,
    station_edges,
)
from src.line2.training.independent import (
    PROJ_PAIRS,
    IndependentEnsemble,
    proj_condition,
    proj_condition_batch,
)
from src.line2.training.losses import batch_lambda_min, poisson_nll, psd_penalty
from src.line2.training.train import _forward_variant, build_model
from src.line2.training.unet import (
    Phase1Net,
    batch_sigma_from_counts,
    condition_vector,
    normalize_input_group,
)
from tests.conftest import derive_seed

pytestmark = pytest.mark.unit

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_E1_CONFIGS = _REPO_ROOT / "docs" / "specs" / "line2" / "e1_configs"
_SRC_LINE2 = _REPO_ROOT / "src" / "line2"

#: 四轴量级（x, xp, y, yp）——互异 ⇒ 四轴 bin 边界互异。
_AXIS_SCALES = (3.3e-4, 2.9e-5, 2.2e-4, 5.0e-5)
#: 六投影在四轴下的索引对，由 `00` [S1] 枚举经 `AXES` 导出（不硬编码第二份顺序）。
_AXIS_PAIRS = tuple((AXES.index(a), AXES.index(b)) for a, b in PROJECTION_ORDER)
#: [S2] C2 单调性用的三档非半正定构造强度（β 越大 λ_min 越负）。
_PSD_BETAS = (0.40, 0.55, 0.70)
#: [S2] C2 半正定档（同一构造族中 λ_min ≥ 0 的一侧）。
_PSD_BETA_POSITIVE = 0.30


def _small_net() -> Phase1Net:
    """缩配联合版网络（前向类用例；生产档参数量单独断言）。"""
    torch.manual_seed(derive_seed(1))
    return Phase1Net(base=24, num_levels=3, blocks=1, cond_dim=11)


def _small_ensemble() -> IndependentEnsemble:
    """缩配独立版（6 个互不共享权重的 1 进 1 出子网）。"""
    torch.manual_seed(derive_seed(2))
    return IndependentEnsemble(base=24, num_levels=3, blocks=1)


def _edges() -> torch.Tensor:
    """四轴同界的 (4,129) 边界矩阵：Σ̂ 的 λ_min 由相关结构主导，不被轴间量级差左右。"""
    table_edges = np.linspace(-1.05, 1.05, 129)
    return torch.tensor(np.stack([table_edges] * 4), dtype=torch.float32)


def _particle_table(seed_label: int, n: int = 8_000) -> np.ndarray:
    """合成 4D 粒子表 (n, 4)：六对相关系数互异 + 逐轴量级互异（随机源由主种子派生）。"""
    rng = np.random.default_rng(derive_seed(seed_label))
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.60    # (x, xp)
    corr[2, 3] = corr[3, 2] = -0.45   # (y, yp)
    corr[0, 2] = corr[2, 0] = 0.30    # (x, y)
    corr[0, 3] = corr[3, 0] = -0.20   # (x, yp)
    corr[1, 2] = corr[2, 1] = 0.35    # (xp, y)
    corr[1, 3] = corr[3, 1] = 0.10    # (xp, yp)
    z = rng.standard_normal((n, 4)) @ np.linalg.cholesky(corr).T
    return z * np.asarray(_AXIS_SCALES)


def _station_edges(table: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """站级边界 (4,129)：返回 (float64 numpy, float32 tensor) 两份同源视图。"""
    edges_np = station_edges(table)
    return edges_np, torch.from_numpy(edges_np.astype(np.float32))


def _group(table: np.ndarray, edges_np: np.ndarray, batch: int = 1) -> torch.Tensor:
    """粒子表 → (batch, 6, 128, 128) float32 计数图组（走 `10` [S4] 数据管线同一入口）。"""
    counts = six_projections(table, edges_np).astype(np.float32)
    return torch.from_numpy(counts).unsqueeze(0).repeat(batch, 1, 1, 1)


def _subsample(table: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
    """站级全量表 → n 粒子无放回子样本（`10` [S2] C2：子样本 ⊂ 全量表）。

    等距步进取样、`offset` 区分不同实现，故子样本必然落在站级 edges 之内——
    这是 `six_projections` 零截断守卫（`10` [S4] C1）所要求的前置条件。
    """
    step = max(1, table.shape[0] // n)
    return table[offset::step][:n]


def _centers(edges: torch.Tensor) -> np.ndarray:
    """(4,129) 边界 → (4,128) bin 中心（float64，供解析构造图组）。"""
    e = edges.numpy().astype(np.float64)
    return 0.5 * (e[:, :-1] + e[:, 1:])


def _psd_cloud_group(edges: torch.Tensor) -> torch.Tensor:
    """同一 4D 高斯云的六投影直方图（(1,6,128,128) 概率图）——Σ̂ 半正定的可控样例。"""
    rng = np.random.default_rng(derive_seed(71))
    e = edges.numpy().astype(np.float64)
    cov = np.diag([0.04, 0.09, 0.04, 0.09])
    cov[0, 1] = cov[1, 0] = 0.5 * np.sqrt(cov[0, 0] * cov[1, 1])
    cloud = rng.multivariate_normal(np.zeros(4), cov, size=40_000)
    # 与生产端 `six_projections` 同布局：h[bin_ai, bin_aj].T ⇒ [bin_aj, bin_ai]
    counts = np.stack(
        [np.histogram2d(cloud[:, a], cloud[:, b], bins=[e[a], e[b]])[0].T
         for a, b in _AXIS_PAIRS]
    )
    prob = counts / counts.sum(axis=(1, 2), keepdims=True)
    return torch.from_numpy(prob.astype(np.float32)).unsqueeze(0)


def _mixed_signature_group(edges: torch.Tensor, beta: float) -> tuple[torch.Tensor, float]:
    """六张各自合法的 2D 高斯图，其拼装 Σ 的相关阵 = I + β·P（P 的 λ_min = −3）。

    β < 1/3 ⇒ Σ 半正定；β > 1/3 ⇒ 非半正定且 λ_min 随 β 单调下降。
    返回 (概率图组, 目标 Σ 的解析 λ_min)。
    """
    centers = _centers(edges)
    var = np.array([0.04, 0.09, 0.04, 0.09])
    signs = {(0, 1): -1.0, (2, 3): -1.0, (0, 2): 1.0, (0, 3): 1.0, (1, 2): 1.0, (1, 3): 1.0}
    pattern = np.zeros((4, 4))
    for (a, b), sign in signs.items():
        pattern[a, b] = pattern[b, a] = sign
    corr = np.eye(4) + beta * pattern
    sigma_target = np.outer(np.sqrt(var), np.sqrt(var)) * corr
    out = np.zeros((len(PROJECTION_ORDER), 128, 128))
    for gi, (a, b) in enumerate(_AXIS_PAIRS):
        rho, ca, cb = corr[a, b], centers[a], centers[b]
        quad = -0.5 * (
            ca[:, None] ** 2 / var[a]
            + cb[None, :] ** 2 / var[b]
            - 2 * rho * ca[:, None] * cb[None, :] / np.sqrt(var[a] * var[b])
        ) / (1 - rho**2)
        img = np.exp(quad)  # [bin_ai, bin_aj]
        out[gi] = (img / img.sum()).T  # 存为生产端布局 [bin_aj, bin_ai]
    lam_target = float(np.linalg.eigvalsh(sigma_target)[0])
    return torch.from_numpy(out.astype(np.float32)).unsqueeze(0), lam_target


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _calls_named(nodes: list[ast.stmt], name: str) -> list[ast.Call]:
    """收集语句列表内对 ``name`` 的调用（Name 或 Attribute 形式）。"""
    found = []
    for node in nodes:
        for sub in ast.walk(node):
            func = getattr(sub, "func", None) if isinstance(sub, ast.Call) else None
            if func is None:
                continue
            callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if callee == name:
                found.append(sub)
    return found


def _compares_to(test: ast.expr, value: str) -> bool:
    """判断 if 条件里是否出现字符串常量 ``value``（结构判定，非源码文本匹配）。"""
    return any(
        isinstance(node, ast.Constant) and node.value == value for node in ast.walk(test)
    )


def test_forward_shape_positive_and_prob_normalized() -> None:
    """[S1] C1/C3 + [S2] C3：前向形状、严格正、逐投影概率和 = 1（spec 口径 1×10⁻⁶）。"""
    net = _small_net().eval()
    gen = torch.Generator().manual_seed(derive_seed(3))
    x = torch.rand(2, 6, 128, 128, generator=gen)
    kappa = torch.randn(2, 11, generator=gen)
    with torch.no_grad():
        p = net(x, kappa)
    assert p.shape == (2, 6, 128, 128)
    assert bool((p > 0).all())
    sums = p.sum(dim=(2, 3))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_param_cap_and_validation() -> None:
    """[S1] C2：默认配置参数量 ≤ 6×10⁷；非法 base/depth 拒绝。"""
    torch.manual_seed(0)
    net = Phase1Net()  # base=48, depth=4
    total, _ = net.count_parameters()
    assert total <= 6e7, f"参数量 {total} 超上限"
    with pytest.raises(ValueError):
        Phase1Net(base=32)
    with pytest.raises(ValueError):
        Phase1Net(num_levels=2)


def test_conditioning_changes_output() -> None:
    """[S1] FiLM 条件化生效：同输入、不同 κ → 输出不同。"""
    net = _small_net().eval()
    gen = torch.Generator().manual_seed(derive_seed(4))
    x = torch.rand(1, 6, 128, 128, generator=gen)
    with torch.no_grad():
        d = float((net(x, torch.zeros(1, 11)) - net(x, torch.ones(1, 11))).abs().max())
    assert d > 0


def test_checkpoint_roundtrip(tmp_path: pathlib.Path) -> None:
    """[S1] C3：checkpoint 往返 → 同输入输出**逐位一致**（`torch.equal`，非容差比较）。"""
    net = _small_net().eval()
    gen = torch.Generator().manual_seed(derive_seed(5))
    x = torch.rand(1, 6, 128, 128, generator=gen)
    kappa = torch.randn(1, 11, generator=gen)
    with torch.no_grad():
        before = net(x, kappa)
    ckpt = tmp_path / "model.ckpt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    net2 = _small_net()
    net2.load_state_dict(torch.load(ckpt, weights_only=True)["state_dict"])
    net2.eval()
    with torch.no_grad():
        after = net2(x, kappa)
    assert torch.equal(before, after), (
        f"ckpt 往返后输出有位差：max|Δ|={float((before - after).abs().max())}"
    )


def test_input_channel_order_is_projection_enumeration() -> None:
    """[S1] C4：输入通道序 = `00` [S1] 六投影枚举，且 κ 的 Σ 分量按该枚举配对轴。

    三层可证伪断言：枚举同序（数据侧/模型侧）、通道数 = 枚举长度、以及
    "只扰动通道 gi ⇒ 只有 PROJECTION_ORDER[gi] 两轴所在的 Σ 分量改变"——
    枚举重排或轴对映射错置都会让改变集越界。
    """
    assert AXES == ("x", "xp", "y", "yp")
    assert PROJECTION_ORDER == (
        ("x", "xp"), ("y", "yp"), ("x", "y"), ("x", "yp"), ("xp", "y"), ("xp", "yp"),
    )
    assert PROJ_PAIRS == PROJECTION_ORDER

    joint = _small_net()
    assert joint.backbone.stem[0].in_channels == len(PROJECTION_ORDER)
    assert joint.backbone.head.out_channels == len(PROJECTION_ORDER)
    ensemble = _small_ensemble()
    assert len(ensemble.nets) == len(PROJECTION_ORDER)
    assert all(
        net.backbone.stem[0].in_channels == 1 and net.backbone.head.out_channels == 1
        for net in ensemble.nets
    )

    table = _particle_table(21)
    edges_np, edges = _station_edges(table)
    counts = _group(table, edges_np)
    sigma = batch_sigma_from_counts(counts, edges)
    assert torch.equal(sigma[0], sigma[0].T), "Σ 必须对称"

    kappa = condition_vector(counts, edges)
    iu, ju = np.triu_indices(4, k=0)
    # κ 前 10 维 = Σ 的 10 个唯一分量（排序后比较：不锁定实现内部的分量次序）
    torch.testing.assert_close(
        torch.sort(kappa[0, :10]).values,
        torch.sort(sigma[0][iu, ju]).values,
        rtol=1e-4,
        atol=0.0,
    )
    n_low = counts.sum(dim=(1, 2, 3)).clamp_min(1.0)
    torch.testing.assert_close(kappa[:, 10], torch.log10(n_low), rtol=1e-6, atol=1e-6)

    for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
        axes_idx = {AXES.index(ai), AXES.index(aj)}
        perturbed = counts.clone()
        perturbed[0, gi] = 0.0
        perturbed[0, gi, 40:88, 40:88] = 500.0
        sigma_p = batch_sigma_from_counts(perturbed, edges)[0]
        changed = {
            (p, q)
            for p in range(4)
            for q in range(4)
            if not torch.equal(sigma_p[p, q], sigma[0, p, q])
        }
        allowed = {(p, q) for p in range(4) for q in range(4) if p in axes_idx or q in axes_idx}
        assert changed, f"通道 {gi} {PROJECTION_ORDER[gi]} 未影响 Σ 的任何分量"
        assert changed <= allowed, (
            f"通道 {gi} {PROJECTION_ORDER[gi]} 越界影响 Σ 分量 {sorted(changed - allowed)}"
        )
        ia, ib = AXES.index(ai), AXES.index(aj)
        assert not torch.equal(sigma_p[ia, ib], sigma[0, ia, ib])


def test_batch_sigma_from_counts_recovers_anisotropic_particle_moments() -> None:
    """[S1] C4/κ 摘要：counts → Σ 须复现粒子表实测 Σ（与 `20` [S1] M4 同源同口径）。

    计数图组由 `six_projections` 以 h.T 存储，故 Σ 的边缘也必须按 (ai, aj) 语义取。
    本束各轴形状/量级互异（均匀 vs 高斯、σ 相差 ~40×）——按"轴 0 = ai"误读时对角偏差
    达 10⁰ 量级（实测 −0.87/+7.0/+5.9/−0.86），配对一致时只剩 bin 量化偏差（≤ 1×10⁻³）。
    """
    from tests.unit.test_line2_metrics import _anisotropic_truth_table

    table = _anisotropic_truth_table()
    edges_np = station_edges(table)
    counts_np = six_projections(table, edges_np).astype(np.float32)
    counts = torch.from_numpy(counts_np).unsqueeze(0)
    edges = torch.from_numpy(edges_np.astype(np.float32))

    sigma_t = batch_sigma_from_counts(counts, edges)[0]
    assert torch.equal(sigma_t, sigma_t.T), "Σ 必须对称"
    sigma = sigma_t.double().numpy()
    sigma_ref = np.cov(table.T, ddof=0)
    scale = np.sqrt(np.outer(np.diag(sigma_ref), np.diag(sigma_ref)))
    diag_ref = np.diag(sigma_ref)
    diag_rel = np.abs(np.diag(sigma) - diag_ref) / diag_ref
    assert diag_rel.max() < 1e-2, f"Σ 对角偏离粒子表实测：{diag_rel}"
    assert float((np.abs(sigma - sigma_ref) / scale).max()) < 1e-2, "Σ 分量偏离粒子表实测"
    # 真值互异（防"常数对常数"的空断言）
    assert sigma_ref[1, 1] < 1e-3 * sigma_ref[0, 0]


def test_independent_channel_and_axis_binding() -> None:
    """[S1] C4（独立版）：输出通道 gi 只消费输入通道 gi，其条件只用该投影两轴的 edges。

    轴依赖在 **κ 层**逐位可判：改某轴 edges ⇔ 恰好该投影对含此轴的通道的 κ 变化。
    输出层的连接性只在 Δκ 大于 float32 分辨率时成立——κ 未做尺度归一（见
    `.decisions.md` D13），Σ 分量为物理单位²（~1e-4…1e-9）而 log₁₀N_low ~4.8，
    FiLM 线性层对 <1e-8 的 κ 变化在 float32 下不响应。故输出层断言只覆盖
    Δκ > 1e-8 的轴（x、y），其余轴由 κ 层断言承载——避免把一条设计局限写成红灯。
    """
    ensemble = _small_ensemble().eval()
    table = _particle_table(31)
    edges_np, edges = _station_edges(table)
    counts = _group(table, edges_np, batch=2)
    shapes, _totals = normalize_input_group(counts)
    log_n = torch.log10(counts.sum(dim=(1, 2, 3)).clamp_min(1.0))
    with torch.no_grad():
        base = ensemble(shapes, edges, log_n)

    for perturbed_channel in (0, 3):
        shapes_p = shapes.clone()
        shapes_p[:, perturbed_channel] = shapes_p[:, perturbed_channel].flip(-1)
        with torch.no_grad():
            out = ensemble(shapes_p, edges, log_n)
        for gi in range(len(PROJECTION_ORDER)):
            assert torch.equal(out[:, gi], base[:, gi]) == (gi != perturbed_channel), (
                f"改输入通道 {perturbed_channel} 时输出通道 {gi} 的响应不符合 1 进 1 出隔离"
            )

    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
    for axis_name, axis_idx in (("x", 0), ("xp", 1), ("y", 2), ("yp", 3)):
        edges_p = edges.clone()
        edges_p[axis_idx] = edges_p[axis_idx] * 1.5
        centers_p = 0.5 * (edges_p[:, :-1] + edges_p[:, 1:])
        expected = {gi for gi, pair in enumerate(PROJECTION_ORDER) if axis_name in pair}
        # κ 层：改该轴 ⇒ 恰好含该轴的投影通道的 κ 变化（逐位比较，不受 float32 分辨率影响）
        changed_kappa = set()
        deltas: dict[int, float] = {}
        for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
            i, j = AXES.index(ai), AXES.index(aj)
            k0 = proj_condition(shapes[0, gi], centers[i], centers[j], float(log_n[0]))
            k1 = proj_condition(shapes[0, gi], centers_p[i], centers_p[j], float(log_n[0]))
            delta = float((k1[:3] - k0[:3]).abs().max())
            deltas[gi] = delta
            if delta > 0.0:
                changed_kappa.add(gi)
        assert changed_kappa == expected, (
            f"改 {axis_name} 轴 edges：κ 受影响通道 {sorted(changed_kappa)} ≠ 枚举预期 {sorted(expected)}"
        )
        if max(deltas.values()) <= 1e-8:  # 低于 FiLM 的 float32 分辨率：只断言 κ 层
            continue
        with torch.no_grad():
            out = ensemble(shapes, edges_p, log_n)
        changed = {
            gi for gi in range(len(PROJECTION_ORDER)) if not torch.equal(out[:, gi], base[:, gi])
        }
        assert changed == expected, (
            f"改 {axis_name} 轴 edges：受影响通道 {sorted(changed)} ≠ 枚举预期 {sorted(expected)}"
        )


def test_conditioning_and_output_invariant_to_target() -> None:
    """[S1] C5 ★：换目标不改输入 ⇒ κ 与网络输出逐位不变（走训练同款调用路径）。

    非空洞性由同一用例内的正向对照保证：换输入 ⇒ κ 与输出必变（说明"不变"不是
    因为输出恒定或 κ 与数据无关）。
    """
    joint = _small_net().eval()
    ensemble = _small_ensemble().eval()
    device = torch.device("cpu")
    station = _particle_table(41, n=60_000)
    edges_np, edges = _station_edges(station)
    # 输入与两个目标都取自同一站级全量表的互斥子样本（A ⊂ B，`10` [S2] C2）——
    # 满足 six_projections 零截断守卫的前置，同时保持"同一输入、两个不同目标"的语义
    counts = _group(_subsample(station, 8_000, offset=0), edges_np, batch=2)
    target_a = _group(_subsample(station, 20_000, offset=1), edges_np, batch=2)
    target_b = _group(_subsample(station, 20_000, offset=2), edges_np, batch=2)
    assert not torch.equal(target_a, target_b), "两个目标图组必须真的不同"
    assert bool((target_a >= 0).all()) and bool((target_b >= 0).all())

    batches = [
        {"counts": counts, "target": target_a, "edges": edges},
        {"counts": counts, "target": target_b, "edges": edges},
    ]
    kappa_before = condition_vector(counts, edges)
    counts_before = counts.clone()
    for variant, net in (("joint", joint), ("independent", ensemble)):
        with torch.no_grad():
            probs = [_forward_variant(net, variant, b, device)[0] for b in batches]
        assert torch.equal(probs[0], probs[1]), f"{variant}：只换目标却改变了网络输出（泄露）"
    # κ 逐位不变：前向路径既拿不到目标，也不得就地改写输入图组来间接引入目标信息
    assert torch.equal(condition_vector(counts, edges), kappa_before)
    assert torch.equal(counts, counts_before), "前向路径就地改写了输入计数图组"

    # 正向对照：换输入（同 edges、同站级全量表内的另一子样本）⇒ κ 变、两变体输出均变
    counts_other = _group(_subsample(station, 8_000, offset=3), edges_np, batch=2)
    assert not torch.equal(
        condition_vector(counts, edges), condition_vector(counts_other, edges)
    ), "κ 未随输入图组变化——防泄露断言失去意义"
    batch_other = {"counts": counts_other, "target": target_a, "edges": edges}
    for variant, net in (("joint", joint), ("independent", ensemble)):
        with torch.no_grad():
            other = _forward_variant(net, variant, batch_other, device)[0]
            reference = _forward_variant(net, variant, batches[0], device)[0]
        assert not torch.equal(other, reference), f"{variant}：换输入后输出未变"


def test_variant_forward_signatures_exclude_target() -> None:
    """[S1] C5（结构面）：条件化与前向入口都不接收目标/参照——结构上拿不到 target。"""
    assert list(inspect.signature(normalize_input_group).parameters) == ["counts"]
    assert list(inspect.signature(condition_vector).parameters) == ["counts", "edges", "log_n_low"]
    assert list(inspect.signature(Phase1Net.forward).parameters) == ["self", "shape_maps", "kappa"]
    assert list(inspect.signature(IndependentEnsemble.forward).parameters) == [
        "self", "shape_maps", "edges", "log_n_low",
    ]


def test_normalize_input_group_reversible_and_degenerate() -> None:
    """[S1] C6：形状图 × 总计数还原计数（相对误差 < 1×10⁻⁶）；Σx=0 投影 → 全零形状图且无 NaN。"""
    table = _particle_table(51)
    edges_np, edges = _station_edges(table)
    counts = _group(table, edges_np, batch=2)
    shapes, totals = normalize_input_group(counts)
    assert torch.equal(totals, counts.sum(dim=(2, 3)))
    reconstructed = shapes * totals[:, :, None, None]
    occupied = counts > 0
    relative = (reconstructed - counts).abs()[occupied] / counts[occupied]
    assert float(relative.max()) < 1e-6
    assert bool((reconstructed[~occupied] == 0).all())
    per_proj = shapes.sum(dim=(2, 3))
    assert torch.allclose(per_proj, torch.ones_like(per_proj), atol=1e-6)

    degenerate = counts.clone()
    degenerate[:, 1] = 0.0
    shapes_d, totals_d = normalize_input_group(degenerate)
    assert bool((shapes_d[:, 1] == 0).all()), "Σx=0 的投影必须给全零形状图"
    assert bool(torch.isfinite(shapes_d).all())
    assert float(totals_d[:, 1].abs().max()) == 0.0
    kappa_d = condition_vector(degenerate, edges)
    assert bool(torch.isfinite(kappa_d).all()), "退化投影不得让 κ 出现 NaN/Inf"
    n_low = degenerate.sum(dim=(1, 2, 3)).clamp_min(1.0)
    torch.testing.assert_close(kappa_d[:, 10], torch.log10(n_low), rtol=1e-6, atol=1e-6)

    zeros = torch.zeros_like(counts)
    shapes_z, totals_z = normalize_input_group(zeros)
    assert bool((shapes_z == 0).all()) and bool(torch.isfinite(shapes_z).all())
    assert float(totals_z.abs().max()) == 0.0
    assert bool(torch.isfinite(condition_vector(zeros, edges)).all())


# ---------------------------------------------------------------- [S2] 损失
def test_poisson_nll_hand_example() -> None:
    """[S2] C1：单格手算——p=1、c=10、N_high=1 → rate=1，NLL = 1 + lgamma(11)。"""
    prob = torch.ones(1, 1, 1, 1)
    c = torch.tensor([[10.0]])
    nll = poisson_nll(prob, c, n_high=1.0)
    expect = 1.0 - 10.0 * 0.0 + float(torch.lgamma(torch.tensor(11.0)))
    assert abs(float(nll) - expect) < 1e-6


def test_poisson_nll_rate_semantics() -> None:
    """[S2] D7：rate = p·N_high（放大语义——目标为 2e5 计数时 p 和应为 c/N_high）。"""
    c = torch.zeros(1, 1, 4, 4); c[0, 0, 0, 0] = 400.0
    p = torch.zeros(1, 1, 4, 4); p[0, 0, 0, 0] = 1.0  # 全部概率在亮格
    n_high = 200_000.0
    nll_at_truth = poisson_nll(p, c, n_high)  # rate = 2e5 vs c=400 → 惩罚大
    p_half = torch.full_like(p, 0.25)
    nll_spread = poisson_nll(p_half, c, n_high)  # rate = 5e4/格 → 更差
    assert float(nll_at_truth) < float(nll_spread)


def test_psd_penalty_zero_when_sigma_psd() -> None:
    """[S2] C2 前半：Σ̂ 半正定（λ_min ≥ 0）⇒ P_PSD 恰为 0（不是"足够小"）。"""
    edges = _edges()
    samples = [
        _psd_cloud_group(edges),
        _mixed_signature_group(edges, _PSD_BETA_POSITIVE)[0],
    ]
    for prob in samples:
        lam_min = batch_lambda_min(prob, edges)
        assert float(lam_min.min()) >= 0.0, f"前置条件不成立：λ_min = {lam_min.tolist()}"
        assert float(psd_penalty(prob, edges)) == 0.0
    batched = torch.cat(samples, dim=0)
    assert float(batch_lambda_min(batched, edges).min()) >= 0.0
    assert float(psd_penalty(batched, edges)) == 0.0


def test_psd_penalty_monotone_in_minus_lambda_min() -> None:
    """[S2] C2 后半：λ_min < 0 ⇒ P_PSD > 0、随 −λ_min 单调增，且 = max(0, −λ_min)²。"""
    edges = _edges()
    previous: tuple[float, float] | None = None
    for beta in _PSD_BETAS:
        prob, lam_target = _mixed_signature_group(edges, beta)
        lam_min = batch_lambda_min(prob, edges)
        penalty = psd_penalty(prob, edges)
        assert float(lam_min[0]) < 0.0, f"β={beta} 未构造出非半正定 Σ̂"
        # 构造可控：实测 λ_min 跟随解析目标（同一 β 序列下依次更负）
        np.testing.assert_allclose(float(lam_min[0]), lam_target, rtol=5e-2)
        assert float(penalty) > 0.0
        torch.testing.assert_close(penalty, torch.relu(-lam_min).pow(2).mean(), rtol=1e-6, atol=0.0)
        if previous is not None:
            assert float(lam_min[0]) < previous[0], f"β={beta}：λ_min 未随构造强度下降"
            assert float(penalty) > previous[1], f"β={beta}：P_PSD 未随 −λ_min 单调增"
        previous = (float(lam_min[0]), float(penalty))


def test_nll_single_implementation_point() -> None:
    """[S2] C4：Poisson NLL 在 src/line2 内唯一实现点，联合/独立两臂共用之（AST 静态检查）。"""
    nll_defs: list[tuple[str, str]] = []
    lgamma_modules: set[str] = set()
    variant_dispatch: list[ast.If] = []
    for path in sorted(_SRC_LINE2.rglob("*.py")):
        relative = path.relative_to(_SRC_LINE2).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "nll" in node.name.lower():
                nll_defs.append((relative, node.name))
            if isinstance(node, ast.Attribute) and node.attr == "lgamma":
                lgamma_modules.add(relative)
        if relative == "training/train.py":
            variant_dispatch = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.If) and _compares_to(node.test, "joint")
            ]

    assert nll_defs == [("training/losses.py", "poisson_nll")], f"NLL 实现点不唯一：{nll_defs}"
    assert lgamma_modules == {"training/losses.py"}, f"别处自拼 Poisson 项：{sorted(lgamma_modules)}"
    assert variant_dispatch, "训练循环内找不到 variant 分派点"
    arms_with_nll = []
    for node in variant_dispatch:
        in_body = bool(_calls_named(node.body, "poisson_nll"))
        in_else = bool(_calls_named(node.orelse, "poisson_nll"))
        assert in_body == in_else, "variant 两臂必须调用同一 NLL 实现（一臂自算即违约）"
        arms_with_nll.append(in_body)
    assert any(arms_with_nll), "训练循环的 variant 分派两臂都没有调用 NLL"


# ------------------------------------------------- [S4] 独立版/联合版对照与回归
def test_param_alignment_calibrated_design_point() -> None:
    """[S4] C1（设计标定档：C_ind=16、num_levels=4、blocks=3）：两版参数量差 ≤ 30%。"""
    joint = build_model({"variant": "joint", "base": 48, "num_levels": 4, "blocks": 2})
    independent = build_model(
        {"variant": "independent", "ind_base": 16, "num_levels": 4, "ind_blocks": 3}
    )
    n_joint, n_independent = _n_params(joint), _n_params(independent)
    relative = abs(n_independent - n_joint) / n_joint
    assert relative <= 0.30, f"独立 {n_independent} vs 联合 {n_joint} → {relative:.1%}"


def test_param_alignment_e1_production_configs() -> None:
    """[S4] C1（E1 生产档）：按 e1_configs 实际取值构造两臂，参数量差 ≤ 30%。

    生产档取值：联合臂 `base=48, num_levels=4, blocks=2`（10,708,710 参数）；独立臂
    由 `ind_base=16, ind_blocks=3` 决定（`build_model` 对 independent 只读
    `ind_base`/`ind_blocks`，故独立臂 config 不携带 `base`/`blocks`）→ 10,092,870
    = −5.8%，标定来源见 `30` D11。
    """
    cfg_joint = json.loads((_E1_CONFIGS / "e1_joint_s0.json").read_text(encoding="utf-8"))
    cfg_independent = json.loads((_E1_CONFIGS / "e1_indep_s0.json").read_text(encoding="utf-8"))
    n_joint = _n_params(build_model(cfg_joint))
    n_independent = _n_params(build_model(cfg_independent))
    relative = (n_independent - n_joint) / n_joint
    assert abs(relative) <= 0.30, (
        f"独立 {n_independent} vs 联合 {n_joint} → {relative:+.1%}"
        f"（ind_base={cfg_independent['ind_base']}、num_levels={cfg_independent['num_levels']}、"
        f"ind_blocks={cfg_independent['ind_blocks']}）"
    )


def test_independent_forward_matches_per_sample_reference() -> None:
    """[S4] 回归：逐投影整批 forward ≡ 逐样本 (proj_condition + SingleProjNet) 参考路径。

    整批化只允许改变调度，不允许改变数值：FiLM/GroupNorm/Softplus 与逐图归一化都
    逐样本独立，故两条路径必须等价（atol 1×10⁻⁶；CPU 实测量级 ~1×10⁻¹⁰）。
    参考路径的轴索引由 `AXES`/`PROJECTION_ORDER` 独立导出，不复用实现内部的映射表。
    """
    ensemble = _small_ensemble().eval()
    # 混合 batch：逐样本各用自己的站级边界（轴量级互异）且 N_low 互异——
    # 逐样本 κ 的四个分量都不同，任何"把样本 0 的条件套用到整个 batch"的实现都会被抓出。
    tables = [
        _particle_table(61, n=8_000),
        _particle_table(62, n=5_000) * np.array([1.8, 0.55, 1.0, 1.0]),
        _particle_table(63, n=11_000) * np.array([0.7, 1.0, 1.4, 0.8]),
    ]
    edges_list = [station_edges(table) for table in tables]
    counts = torch.cat(
        [_group(table, edges_np) for table, edges_np in zip(tables, edges_list)], dim=0
    )
    edges_batch = torch.from_numpy(np.stack([e.astype(np.float32) for e in edges_list]))
    shapes, _totals = normalize_input_group(counts)
    log_n = torch.log10(counts.sum(dim=(1, 2, 3)).clamp_min(1.0))
    assert len(set(log_n.tolist())) == 3, "逐样本 N_low 必须互异，否则条件化的逐样本性未被检验"
    centers = 0.5 * (edges_batch[:, :, :-1] + edges_batch[:, :, 1:])

    with torch.no_grad():
        out = ensemble(shapes, edges_batch, log_n)
        reference = torch.zeros_like(out)
        for b in range(shapes.shape[0]):
            for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
                kappa = proj_condition(
                    shapes[b, gi], centers[b, AXES.index(ai)], centers[b, AXES.index(aj)],
                    float(log_n[b]),
                )
                reference[b, gi] = ensemble.nets[gi](
                    shapes[b, gi].reshape(1, 1, 128, 128), kappa.reshape(1, -1)
                )[0, 0]

    assert out.shape == (3, 6, 128, 128)
    torch.testing.assert_close(out, reference, rtol=0.0, atol=1e-6)
    # 概率图量级 ~6×10⁻⁵/格，atol=1e-6 对其相对宽松——补一条与量级无关的判据：
    # 两路径最大绝对差必须远小于概率图平均值（实测比值 ~4×10⁻⁶）。
    assert float((out - reference).abs().max()) < 1e-3 * float(reference.mean())
    assert bool((out > 0).all())
    sums = out.sum(dim=(2, 3))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_independent_forward_broadcasts_edges_and_log_n() -> None:
    """[S4] 回归：edges (4,129) 与 (B,4,129) 复制体结果逐位相同；标量/单元素 log_n 等价于常量向量。"""
    ensemble = _small_ensemble().eval()
    table = _particle_table(64)
    edges_np, edges = _station_edges(table)
    counts = _group(table, edges_np, batch=2)
    shapes, _totals = normalize_input_group(counts)
    log_n = torch.log10(counts.sum(dim=(1, 2, 3)).clamp_min(1.0))
    with torch.no_grad():
        shared = ensemble(shapes, edges, log_n)
        repeated = ensemble(shapes, edges.unsqueeze(0).repeat(2, 1, 1), log_n)
        scalar = ensemble(shapes, edges, float(log_n[0]))
        single = ensemble(shapes, edges, log_n[:1])
    assert torch.equal(shared, repeated), "共享 edges 与逐样本复制 edges 必须给出同一结果"
    constant = torch.full_like(log_n, float(log_n[0]))
    with torch.no_grad():
        expected = ensemble(shapes, edges, constant)
    assert torch.equal(scalar, expected)
    assert torch.equal(single, expected)


def test_proj_condition_single_sample_matches_batch() -> None:
    """[S4] 回归：proj_condition 是 proj_condition_batch 的单样本薄封装（六投影逐位一致）。"""
    table = _particle_table(65)
    edges_np, edges = _station_edges(table)
    counts = _group(table, edges_np, batch=2)
    shapes, _totals = normalize_input_group(counts)
    log_n = torch.log10(counts.sum(dim=(1, 2, 3)).clamp_min(1.0))
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
    for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
        ci, cj = centers[AXES.index(ai)], centers[AXES.index(aj)]
        batched = proj_condition_batch(
            shapes[:, gi], ci.unsqueeze(0).repeat(2, 1), cj.unsqueeze(0).repeat(2, 1), log_n
        )
        assert batched.shape == (2, 4)
        for b in range(2):
            single = proj_condition(shapes[b, gi], ci, cj, float(log_n[b]))
            torch.testing.assert_close(single, batched[b], rtol=0.0, atol=0.0)


def test_proj_condition_matches_metric_sigma_convention() -> None:
    """[S4]/[S1]：独立臂 κ 的 (var_i, var_j, cov_ij) 与 `20` [S1] M4 的 Σ 同口径。

    第三分量是协方差而非相关系数（`30` [S4] 的文字写 ρ_ij，差异见 `30` D17），
    故此处直接与 Σ 的对应分量比，不做 σ_i·σ_j 归一。

    图组存储布局为 `[后一坐标 a_j, 前一坐标 a_i]`（生产端 `six_projections` 的
    `h.T`）。κ 若按 `[a_i, a_j]` 误读边缘，会把"另一轴边缘在本轴网格上的展宽"
    当成 σ_i（各轴 range/σ 相同的合成云上自抵消、测不出——故本用例的粒子表
    逐轴量级与形状互异）。此不变量跨 `metrics.reconstruct_sigma` 与
    `training.independent.proj_condition` 两个模块，任一侧的轴序改动都会被抓住。
    """
    from src.line2.metrics.metrics import reconstruct_sigma

    table = _particle_table(67)
    edges_np, edges = _station_edges(table)
    group = six_projections(table, edges_np)
    sigma = reconstruct_sigma(group, edges_np)
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
    for gi, (ai, aj) in enumerate(PROJECTION_ORDER):
        i, j = AXES.index(ai), AXES.index(aj)
        cond = proj_condition(
            torch.from_numpy(group[gi].astype(np.float32)), centers[i], centers[j], 4.0
        )
        var_i, var_j, cov_ij = (float(cond[0]), float(cond[1]), float(cond[2]))
        assert abs(var_i - sigma[i, i]) / sigma[i, i] < 1e-4, (ai, aj, var_i, sigma[i, i])
        assert abs(var_j - sigma[j, j]) / sigma[j, j] < 1e-4, (ai, aj, var_j, sigma[j, j])
        assert abs(cov_ij - sigma[i, j]) / abs(sigma[i, j]) < 1e-4
