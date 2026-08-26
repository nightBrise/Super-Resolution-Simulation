"""三方案组装（50 [S3][S4][S5][S8]–[S12]）。

- 方案 A（无先验）：``Input = concat(L_up, 0)``，残差基准 ``L_up``；
- 方案 B（图像先验 + early fusion）：``Input = concat(L_up, P2)``，
  残差基准 ``P2``（50 [S9]）；
- 方案 C（参数先验 + FiLM）：``Input = concat(L_up, 0)``，残差基准
  ``L_up``；``c_prior`` 预处理（正参数取对数 + 训练集 z-score，50 [S10]）
  → MLP 编码为 ``z_c`` → FiLM 调制瓶颈层与解码器。

三方案输出均为 ``Ĥ = Softplus(Base + R)``（50 [S12]，非负格式约束），
输入第二通道对 A/C 恒为零（50 [S11] C2），任何方案不使用 ``c_high``
（50 [S11] C4）。每个方案类记录总参数量与可训练参数量（50 [S14] N10）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.unet import UNetBackbone
from src.utils.h5data import C_PRIOR_KEYS, POS_LOG_KEYS, preprocess_c_prior

#: 方案 C FiLM 注入位置（50 [S10] C5）。
FILM_INJECTION = "bottleneck_and_decoder"


class CPriorEncoder(nn.Module):
    """参数先验编码器：标准化 ``c_tilde`` → 条件向量 ``z_c``（50 [S10] C3）。"""

    def __init__(self, in_dim: int = len(C_PRIOR_KEYS), hidden: int = 128, z_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, c_tilde: torch.Tensor) -> torch.Tensor:
        return self.net(c_tilde)


class SchemeModel(nn.Module):
    """三方案共享基类：主干一致（50 [S7] C5）、输出非负（50 [S12] C1/C2）。"""

    scheme: str = ""

    def __init__(
        self,
        C0: int = 48,
        num_levels: int = 5,
        num_residual_blocks: int = 2,
        z_dim: int = 64,
        mlp_hidden: int = 128,
        work_scale: float = 65536.0,
    ) -> None:
        super().__init__()
        self.network_config = {
            "C0": int(C0),
            "num_levels": int(num_levels),
            "num_residual_blocks": int(num_residual_blocks),
            "z_dim": int(z_dim),
            "mlp_hidden": int(mlp_hidden),
            #: 工作尺度随网络配置持久化（60 [S15] 双空间契约：checkpoint
            #: MUST 持久化影响输出语义的尺度参数，评估加载时与 config 断言一致）。
            "work_scale": float(work_scale),
        }
        #: 工作尺度 S = N² = 65536（50 [S12] C5，2026-08-26 坍缩修复）：Softplus
        #: 在 Σ=1 空间像素 ~1.5e-5 上落入指数区（梯度 sigmoid 消失），输出坍缩为
        #: 平凡零预测器；工作尺度使结构像素预激活落在 [0.6, 3.3]、梯度门健康。
        #: S 为固定常数，三方案相同，不引入可训练参数；评估输出还原 Ĥ_work/S。
        self.work_scale = float(work_scale)
        self.backbone = UNetBackbone(
            in_channels=2,
            C0=C0,
            num_levels=num_levels,
            num_residual_blocks=num_residual_blocks,
            with_film=(self.scheme == "C"),
            z_dim=z_dim,
        )
        if self.scheme == "C":
            self.c_prior_encoder = CPriorEncoder(
                in_dim=len(C_PRIOR_KEYS), hidden=mlp_hidden, z_dim=z_dim
            )
            self.register_buffer(
                "c_prior_mu", torch.zeros(len(C_PRIOR_KEYS), dtype=torch.float32)
            )
            self.register_buffer(
                "c_prior_sigma", torch.ones(len(C_PRIOR_KEYS), dtype=torch.float32)
            )

    # -- 参数统计（50 [S14] N10）------------------------------------------------
    def count_parameters(self) -> tuple[int, int]:
        """``(总参数量, 可训练参数量)``。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @property
    def num_parameters(self) -> dict[str, int]:
        total, trainable = self.count_parameters()
        return {"total": total, "trainable": trainable}

    # -- 方案 C 标准化统计量（50 [S10] C2、60 [S5] C3）---------------------------
    def set_c_prior_stats(self, mu, sigma) -> None:
        """设置方案 C 的 z-score 统计量（须来自训练集，训练/评估同一组）。"""
        if self.scheme != "C":
            raise RuntimeError("仅方案 C 使用参数先验标准化统计量")
        self.c_prior_mu.copy_(torch.as_tensor(np.asarray(mu, dtype=np.float32)))
        self.c_prior_sigma.copy_(torch.as_tensor(np.asarray(sigma, dtype=np.float32)))


class SchemeA(SchemeModel):
    """方案 A：无先验 baseline（50 [S3]）。输入 ``concat(L_up, 0)``。"""

    scheme = "A"

    def forward(self, L_up: torch.Tensor) -> torch.Tensor:
        B = L_up.shape[0]
        zeros = torch.zeros_like(L_up)
        x = torch.cat([L_up, zeros], dim=1)
        R = self.backbone(x)
        return nn.functional.softplus(self.work_scale * L_up + R)


class SchemeB(SchemeModel):
    """方案 B：图像先验 + 残差，early fusion（50 [S4][S9]）。"""

    scheme = "B"

    def forward(self, L_up: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([L_up, P2], dim=1)
        R = self.backbone(x)
        return nn.functional.softplus(self.work_scale * P2 + R)


class SchemeC(SchemeModel):
    """方案 C：参数先验 + FiLM（50 [S5][S10]）。

    输入原始 ``c_prior``（顺序见 ``C_PRIOR_KEYS``，不含 ``A`` 与
    ``c_high``）；模型内部完成取对数 + z-score（统计量由训练集设定），
    保证评估阶段与训练使用同一组统计量（60 [S5] C3，★ 无泄漏）。
    """

    scheme = "C"

    def forward(self, L_up: torch.Tensor, c_prior_raw: torch.Tensor) -> torch.Tensor:
        B = L_up.shape[0]
        zeros = torch.zeros_like(L_up)
        x = torch.cat([L_up, zeros], dim=1)
        c_tilde = self._preprocess(c_prior_raw)
        z_c = self.c_prior_encoder(c_tilde)
        R = self.backbone(x, z_c)
        return nn.functional.softplus(self.work_scale * L_up + R)

    def _preprocess(self, c_prior_raw: torch.Tensor) -> torch.Tensor:
        """正参数取对数 + 训练集 z-score（50 [S10] C1/C2）。"""
        x = c_prior_raw.double()
        for i, key in enumerate(C_PRIOR_KEYS):
            if key in POS_LOG_KEYS:
                x = x.clone()
                x[:, i] = torch.log(x[:, i])
        mu = self.c_prior_mu.double()
        sigma = self.c_prior_sigma.double()
        return ((x - mu) / sigma).float()


#: 方案标识 → 方案类（训练/评估/推理共用的构造入口）。
SCHEME_CLASSES: dict[str, type[SchemeModel]] = {
    "A": SchemeA,
    "B": SchemeB,
    "C": SchemeC,
}


def build_scheme_model(config: dict) -> SchemeModel:
    """按 config 构造方案模型（50 [S7] 主干配置 + 代理变体 C0=24）。

    ``network.work_scale`` 为必填（60 [S15] 双空间契约：影响输出语义的
    尺度参数 SHALL 显式配置；config 缺省时抛 ValueError，2026-08-26 P0
    修订：原默认值 65536.0 改为必填）。
    """
    network = config.get("network", {})
    scheme = str(config["scheme"]).upper()
    if scheme not in SCHEME_CLASSES:
        raise ValueError(f"未知方案 {scheme}（应为 A/B/C）")
    cls = SCHEME_CLASSES[scheme]
    if "work_scale" not in network:
        raise ValueError(
            "config 缺 network.work_scale（60 [S15] 双空间契约：工作尺度 S 为必填，"
            "须与训练/评估 config 显式一致；建议 65536 = N²）"
        )
    return cls(
        C0=int(network.get("C0", 48)),
        num_levels=int(network.get("num_levels", 5)),
        num_residual_blocks=int(network.get("num_residual_blocks", 2)),
        z_dim=int(network.get("z_dim", 64)),
        mlp_hidden=int(network.get("mlp_hidden", 128)),
        work_scale=float(network["work_scale"]),
    )


def build_scheme_model_from_checkpoint(ckpt: dict) -> SchemeModel:
    """按 checkpoint 记录的模型标识与网络配置构造模型（评估/推理用）。

    ``work_scale`` 必填：优先取 checkpoint 顶层持久化键（60 [S15] 契约），
    回退 ``network_config.work_scale``；两者皆缺（旧版 checkpoint）时抛
    ValueError 并按 80 [S12] R2 流程提示重训。
    """
    model_class = ckpt.get("model_class") or "SchemeA"
    network = ckpt.get("network_config", {})
    work_scale = ckpt.get("work_scale", network.get("work_scale"))
    if work_scale is None:
        raise ValueError(
            "checkpoint 缺少 work_scale（60 [S15] 双空间契约：旧版 checkpoint "
            "未持久化尺度参数，需按 80 [S12] R2 流程重训后重新评估）"
        )
    cls = SCHEME_CLASSES[model_class.lstrip("Scheme").upper()]
    return cls(
        C0=int(network.get("C0", 48)),
        num_levels=int(network.get("num_levels", 5)),
        num_residual_blocks=int(network.get("num_residual_blocks", 2)),
        z_dim=int(network.get("z_dim", 64)),
        mlp_hidden=int(network.get("mlp_hidden", 128)),
        work_scale=float(work_scale),
    )


def forward_scheme(model: nn.Module, batch: dict, device: str = "cpu") -> torch.Tensor:
    """按方案构造输入并前向（50 [S4] C2–C4；训练/评估/推理共用）。

    ``batch`` 为 ``H5Dataset`` 样本批量字典（键 ``L_up`` / ``P2`` /
    ``c_prior_raw``）。
    """
    L_up = batch["L_up"].to(device)
    if isinstance(model, SchemeC):
        return model(L_up, batch["c_prior_raw"].to(device))
    if isinstance(model, SchemeB):
        return model(L_up, batch["P2"].to(device))
    return model(L_up)


__all__ = [
    "FILM_INJECTION",
    "CPriorEncoder",
    "SchemeA",
    "SchemeB",
    "SchemeC",
    "SchemeModel",
    "SCHEME_CLASSES",
    "build_scheme_model",
    "build_scheme_model_from_checkpoint",
]
