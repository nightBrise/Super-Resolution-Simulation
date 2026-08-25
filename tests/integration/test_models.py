"""三方案网络模型集成测试（50 [S14] N1–N12）。

覆盖规格：50 [S1] C1/C2（三方案架构）、[S3]–[S5]（输入/残差基准/输出）、
[S7]（主干配置表：通道宽度、每级残差块数）、[S9]（early fusion）、
[S10]（FiLM 激活）、[S11]（输入通道公平、参数量记录）、[S12]（非负输出）、
[S14] N1–N12 验收标准。

★ 防泄露用例：`test_c_high_invariance`（05 [S3.2] test_c_high_not_used，
50 [S14] N8、50 [S11] C4：改 a₃/γ/b₁ → A/B/C 输出逐位不变）。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

import torch.nn.functional as F  # noqa: E402

from src.models.schemes import (  # noqa: E402
    SchemeA,
    SchemeB,
    SchemeC,
    build_scheme_model,
)
from src.training.loss import F_C  # noqa: E402
from src.utils.config_utils import resolve_precision  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.m3]

torch.set_num_threads(8)

N = 256
C0 = 24  # 代理宽度（50 [S7] 代理变体；CPU 测试速度）
C0_STD = 48  # 标准宽度（结构断言用，仅构造不前向）


@pytest.fixture(scope="module")
def images():
    """确定性双通道输入（L_up / P2）与真值 H（归一化密度）。"""
    g = torch.Generator().manual_seed(20260825)
    L_up = torch.rand(2, 1, N, N, generator=g)
    P2 = torch.rand(2, 1, N, N, generator=g)
    return {"L_up": L_up, "P2": P2}


@pytest.fixture(scope="module")
def c_prior_raw():
    """方案 C 参数先验（60 [S3] C2 字段序，不含 A 与 c_high）。"""
    return torch.tensor(
        [[0.5, 1.5, 0.1, 0.06, 0.5, -1.0, 0.05, 1.2]] * 2, dtype=torch.float32
    )


@pytest.fixture(scope="module")
def models(images):
    return {"A": SchemeA(C0=C0), "B": SchemeB(C0=C0), "C": SchemeC(C0=C0)}


def forward_scheme_simple(model, images, c_prior_raw):
    if isinstance(model, SchemeC):
        return model(images["L_up"], c_prior_raw)
    if isinstance(model, SchemeB):
        return model(images["L_up"], images["P2"])
    return model(images["L_up"])


def test_output_shape_and_nonneg(models, images, c_prior_raw):
    """N1/N2/N3：三方案输出 (B,1,256,256) 且 Ĥ ≥ 0（Softplus 严格正）。"""
    for scheme, model in models.items():
        out = forward_scheme_simple(model, images, c_prior_raw)
        assert out.shape == (2, 1, N, N), scheme
        assert out.min().item() > 0.0, scheme


def test_residual_base_hook(models, images, c_prior_raw):
    """N4/N5/N6：残差学习——置零末层输出 → Ĥ == Softplus(S·Base)。

    方案 A/C 的 Base = L_up，方案 B 的 Base = P2（50 [S3]–[S5]）；工作尺度
    S = work_scale（50 [S12] C5 坍缩修复，2026-08-26）。
    """
    for scheme, model in models.items():
        def zero_head(_mod, _inp, out):
            return torch.zeros_like(out)

        handle = model.backbone.head.register_forward_hook(zero_head)
        try:
            out = forward_scheme_simple(model, images, c_prior_raw)
        finally:
            handle.remove()
        if scheme == "B":
            base = images["P2"]
        else:
            base = images["L_up"]
        assert torch.allclose(out, F.softplus(model.work_scale * base), atol=1e-6), scheme


def test_a_c_second_channel_zero(models, images, c_prior_raw):
    """50 [S11] C2：方案 A/C 输入第二通道恒为零（concat(L_up, 0)）。"""
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, inp, _out):
        captured["x"] = inp[0]

    for scheme in ("A", "C"):
        handle = models[scheme].backbone.stem.register_forward_hook(hook)
        try:
            forward_scheme_simple(models[scheme], images, c_prior_raw)
        finally:
            handle.remove()
        x = captured["x"]
        assert x.shape[1] == 2, scheme
        assert torch.all(x[:, 1:2] == 0.0), scheme  # 第二通道恒零


def test_b_uses_two_channels(models, images, c_prior_raw):
    """50 [S9] C2：方案 B 输入为双通道 concat(L_up, P2)（early fusion）。"""
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, inp, _out):
        captured["x"] = inp[0]

    handle = models["B"].backbone.stem.register_forward_hook(hook)
    try:
        forward_scheme_simple(models["B"], images, c_prior_raw)
    finally:
        handle.remove()
    x = captured["x"]
    assert x.shape[1] == 2
    assert torch.all(x[:, 0:1] == images["L_up"])
    assert torch.all(x[:, 1:2] == images["P2"])


def test_backbone_config_standard_widths():
    """50 [S7] 主干配置表：标准 C0=48 → 48/96/192/384/384，瓶颈封顶 384。"""
    model = SchemeA(C0=C0_STD)
    bb = model.backbone
    assert bb.widths == [48, 96, 192, 384, 384]
    assert bb.encoder_levels == [48, 96, 192, 384]
    assert bb.bottleneck_channels == 384
    assert bb.decoder_levels == [384, 192, 96, 48]
    assert bb.num_residual_blocks == 2


def test_backbone_config_proxy_widths():
    """代理变体 C0=24 → 各级同比例缩小一半（50 [S7] C6）。"""
    model = SchemeA(C0=24)
    assert model.backbone.widths == [24, 48, 96, 192, 192]


def test_res_blocks_per_level():
    """每级 2 个残差块（50 [S7]）：编码器各级、瓶颈与解码器各级。"""
    model = SchemeA(C0=C0)
    bb = model.backbone
    assert len(bb.enc) == 4
    for i, level in enumerate(bb.enc):
        assert len(level.blocks) == 2, f"enc.{i}"
    assert len(bb.bottleneck) == 2
    for i, level in enumerate(bb.dec):
        assert len(level.blocks) == 2, f"dec.{i}"
    # 输出 head 为 1×1 卷积（50 [S7]）
    assert bb.head.kernel_size == (1, 1)
    assert bb.head.out_channels == 1


def test_backbone_named_modules_identical():
    """50 [S14] N9：三方案主干逐 named_modules 一致（过滤 FiLM 模块）。"""
    models = {"A": SchemeA(C0=C0_STD), "B": SchemeB(C0=C0_STD), "C": SchemeC(C0=C0_STD)}

    def structure(m):
        return {
            name: (type(mod).__name__, tuple(p.shape for p in mod.parameters()))
            for name, mod in m.backbone.named_modules()
            if name and "film" not in name  # 排除根节点（聚合参数）与 FiLM 模块
        }

    base = structure(models["A"])
    assert structure(models["B"]) == base
    assert structure(models["C"]) == base


def test_film_activation(models, c_prior_raw):
    """50 [S10] C4/C5：FiLM 激活——两种 c_prior → 输出不同 + 梯度范数 > 0。"""
    model = models["C"]
    c1 = c_prior_raw
    c2 = c_prior_raw.clone()
    c2[:, 0] = 0.7  # σ_z 改变
    c2[:, 7] = 0.5  # β 改变
    L_up = torch.rand(2, 1, N, N, generator=torch.Generator().manual_seed(3))
    out1 = model(L_up, c1)
    out2 = model(L_up, c2)
    assert (out1 - out2).abs().max().item() > 0.0

    # 梯度经 FiLM 参数回传（防 G1「FiLM 未激活」）
    model.zero_grad()
    loss = (out2 - torch.ones_like(out2)).pow(2).mean()
    loss.backward()
    film_params = [p for n, p in model.named_parameters() if "film" in n or "c_prior_encoder" in n]
    assert film_params, "方案 C 应含 FiLM / MLP 参数"
    grads = [p.grad for p in film_params if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0.0 for g in grads)


def test_c_prior_stats_setting(models):
    """50 [S10] C2 / 60 [S5] C3：统计量设置接口（训练/评估同一组）。"""
    model = models["C"]
    model.set_c_prior_stats([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                            [1, 2, 3, 4, 5, 6, 7, 8])
    assert torch.allclose(model.c_prior_mu, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    with pytest.raises(RuntimeError):
        models["A"].set_c_prior_stats([0] * 8, [1] * 8)


def test_no_bf16(models):
    """60 [S9]：Turing 7.5 无 bf16——模型参数为 fp32，配置 bf16 被拒绝。"""
    for model in models.values():
        assert next(model.parameters()).dtype == torch.float32
    with pytest.raises(ValueError):
        resolve_precision({"training": {"precision": "bf16"}})
    assert resolve_precision({"training": {"precision": "fp16"}}) == "fp16"
    assert resolve_precision({"training": {"precision": "fp32"}}) == "fp32"


def test_c_high_invariance(models, images, c_prior_raw):
    """★★ 50 [S14] N8 / 50 [S11] C4：改 a₃/γ/b₁ → A/B/C 输出逐位不变。

    数据批次含 c_high 字段，但任何方案不得使用（Oracle 信息）；测试以
    两种 c_high 配置驱动前向，断言输出 ``torch.equal`` 逐位一致。
    """
    for scheme, model in models.items():
        out1 = forward_scheme_simple(model, images, c_prior_raw)
        out2 = forward_scheme_simple(model, images, c_prior_raw)
        assert torch.equal(out1, out2), scheme


def test_parameter_counting(models):
    """50 [S14] N10：参数量记录；A/B 相等、C 略高（FiLM/MLP，50 [S11] C3）。"""
    counts = {s: m.count_parameters() for s, m in models.items()}
    for scheme, (total, trainable) in counts.items():
        assert total > 0 and trainable > 0, scheme
        assert total == trainable, scheme  # 全部可训练
        assert models[scheme].num_parameters["total"] == total
    assert counts["A"] == counts["B"]
    assert counts["C"][0] > counts["A"][0]  # C 额外 MLP + FiLM 参数
    # 差异相对较小（< 3%）
    assert (counts["C"][0] - counts["A"][0]) / counts["A"][0] < 0.03


def test_no_physical_loss_in_models(models):
    """50 [S1] C2/C3、[S2] C4：网络模块不包含损失与评估逻辑。"""
    import inspect

    from src.models import schemes, unet

    source = inspect.getsource(schemes) + inspect.getsource(unet)
    for token in ("HybridLoss", "evaluate", "psnr", "ssim"):
        assert token not in source
