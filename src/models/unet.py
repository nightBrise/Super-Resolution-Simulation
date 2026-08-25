"""残差 U-Net 主干（50 [S7]）。

三方案共享同一主干家族：输入 stem（2 通道）、5 级下采样编码器
（256→128→64→32→16）、瓶颈层（通道封顶 384）、对称解码器 + skip
融合、1×1 卷积输出 head（单通道残差 ``R``）。基础通道宽 ``C0`` 可配置
（标准 48 / 代理 24，50 [S7] 代理变体），每级 2 个残差块。

FiLM 注入（50 [S10] C5：瓶颈层 + 解码器）仅当 ``with_film=True`` 时创建
（方案 C）；方案 A/B 的主干不携带 FiLM 参数，保证公平比较时 A/B 的
参数量不混入条件参数（50 [S11] C3）。FiLM 模块名以 ``film`` 前缀命名，
公平性测试据此过滤后比较三方案主干结构（50 [S14] N9）。
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """基础残差块：``conv3×3 → GroupNorm → SiLU → conv3×3 → GroupNorm`` + skip。

    输入输出通道不同时用 1×1 卷积对齐 skip（50 [S7] 残差块）。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class FilmLayer(nn.Module):
    """FiLM 调制层：``F' = γ(z_c)·F + β(z_c)``（50 [S10] C4）。

    ``γ``、``β`` 由条件向量 ``z_c`` 经线性映射生成（每通道一个标量）。
    """

    def __init__(self, z_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma_gen = nn.Linear(z_dim, channels)
        self.beta_gen = nn.Linear(z_dim, channels)

    def forward(self, x: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_gen(z_c).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_gen(z_c).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class _EncoderLevel(nn.Module):
    """编码器级：``n_blocks`` 个残差块（保持本级通道宽）+ 步长 2 下采样。

    本级特征（skip）通道宽 = ``widths[i]``，下采样卷积把通道转为下一级
    宽度 ``next_channels``（50 [S7] 通道序列 48/96/192/384/384）。
    """

    def __init__(self, channels: int, n_blocks: int, next_channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock(channels, channels) for _ in range(n_blocks)]
        )
        self.downsample = nn.Conv2d(channels, next_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        skip = x  # 下采样前的特征作为 skip connection（50 [S7]）
        return self.downsample(x), skip


class _DecoderLevel(nn.Module):
    """解码器级：2 倍上采样 + 与对应编码器 skip 拼接 + ``n_blocks`` 个残差块。"""

    def __init__(self, in_channels: int, skip_channels: int, n_blocks: int) -> None:
        super().__init__()
        fused = in_channels + skip_channels
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.blocks = nn.ModuleList(
            [ResBlock(fused if j == 0 else skip_channels, skip_channels) for j in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x)
        return x


class UNetBackbone(nn.Module):
    """残差 U-Net 主干（50 [S7] 主干配置表）。

    通道宽度序列：``C0, 2C0, 4C0, 8C0, min(16C0, 384)``——瓶颈层封顶
    ``384``（标准 ``C0=48`` 时为 48/96/192/384/384）。编码器 4 级下采样
    （256→128→64→32）+ 瓶颈（16），解码器对称 4 级 + 1×1 head。

    前向返回单通道残差 ``R = G(x)``（``(B, 1, H, W)``）；``z_c`` 非空且
    主干带 FiLM 时对瓶颈与各解码器级输出施加调制（50 [S10] C5）。
    """

    def __init__(
        self,
        in_channels: int = 2,
        C0: int = 48,
        num_levels: int = 5,
        num_residual_blocks: int = 2,
        bottleneck_cap: int = 384,
        with_film: bool = False,
        z_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels 至少为 2")
        # 瓶颈封顶随 C0 同比例缩放：标准 C0=48 → 384，代理 C0=24 → 192
        # （50 [S7]：代理变体各级通道宽度同比例缩小一半）。
        eff_cap = min(int(bottleneck_cap), 8 * int(C0))
        widths = [min(int(C0) * 2**i, eff_cap) for i in range(num_levels)]
        self.widths = widths
        #: 编码器各级通道（skip 特征宽度），标准配置为 [48, 96, 192, 384]。
        self.encoder_levels = list(widths[:-1])
        #: 瓶颈层通道（封顶 384）。
        self.bottleneck_channels = int(widths[-1])
        #: 解码器各级通道（与编码器 skip 对称），标准配置为 [384, 192, 96, 48]。
        self.decoder_levels = list(reversed(self.encoder_levels))
        self.num_residual_blocks = int(num_residual_blocks)

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), widths[0], 3, padding=1),
            nn.SiLU(),
        )
        self.enc = nn.ModuleList(
            [
                _EncoderLevel(widths[i], num_residual_blocks, widths[i + 1])
                for i in range(num_levels - 1)
            ]
        )
        self.bottleneck = nn.Sequential(
            *[ResBlock(widths[-1], widths[-1]) for _ in range(num_residual_blocks)]
        )
        self.dec = nn.ModuleList(
            [
                _DecoderLevel(
                    widths[-1] if j == 0 else self.decoder_levels[j - 1],
                    skip_channels,
                    num_residual_blocks,
                )
                for j, skip_channels in enumerate(self.decoder_levels)
            ]
        )
        self.head = nn.Conv2d(self.decoder_levels[-1], 1, 1)

        self.with_film = bool(with_film)
        self.z_dim = int(z_dim)
        if with_film:
            self.film_bottleneck = FilmLayer(z_dim, widths[-1])
            self.film_decoder = nn.ModuleList(
                [FilmLayer(z_dim, w) for w in self.decoder_levels]
            )

    def forward(
        self, x: torch.Tensor, z_c: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for level in self.enc:
            x, skip = level(x)
            skips.append(skip)
        x = self.bottleneck(x)
        if z_c is not None and self.with_film:
            x = self.film_bottleneck(x, z_c)
        for j, level in enumerate(self.dec):
            x = level(x, skips[-(j + 1)])
            if z_c is not None and self.with_film:
                x = self.film_decoder[j](x, z_c)
        return self.head(x)

    def count_parameters(self) -> tuple[int, int]:
        """``(总参数量, 可训练参数量)``（50 [S14] N10）。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
