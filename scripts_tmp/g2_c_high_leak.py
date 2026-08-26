#!/usr/bin/env python
"""G2 验证：σ_smooth,H=0.125× 下 c_high_mask 的能量成分分解。

目的
----
70 [S7.1] 主指标掩膜 $M_{c_{high}}$ 定义为 |H_hp| 幅值降序累计能量达 90% 的
最小像素集，并宣称其为 c_high={a3,γ,b1} 精细结构的空间区域。本脚本用差分
能量法检验该宣称：分别渲染完整 H、c_high 清零版 H_noch（a3=γ=b1=0）与
β 清零版 H_nob（β=0），计算掩膜内各成分的差分高通能量占比。

隐藏假设检验
------------
若掩膜内能量主要由 β 折叠（β∈c_mid）贡献，则主指标在能量意义上测的是
「折叠恢复」而非「c_high 恢复」，方案 B（先验含折叠信息）结构性占优。

依赖
----
stdlib + numpy + scipy + src.generators（已安装）。不依赖 torch。

用法
----
    python scripts_tmp/g2_c_high_leak.py [--n 12] [--seed 9999] [--factor 0.125]
                                         [--compare-old]

    --n            样本数（默认 12，建议 ≥12 取中位数稳定）
    --seed         审计专用主种子（默认 9999，与数据生成 20260825 隔离）
    --factor       σ_smooth,H = factor × w_fine（默认 0.125 = 报批包第 1 项）
    --compare-old  同时计算旧口径 0.5× 作对照

预期输出表
----------
    idx | beta  | w_fine_px | mask_px% | ch_in_mask | b_in_mask | ch_full | b_full
    ...（逐样本）
    MEDIAN ...
    VERDICT: ...

解读规则
--------
- ch_in_mask（掩膜内「c_high 移除敏感度」= c_high 差分能量 / |H_hp| 能量）
  中位数：
  * < 40%  → G2 坐实：掩膜对 c_high 不敏感（主要选中其他结构）→ 升无条件
             P0，70 [S7.1] 按报告 §3 提案 2 分支甲修订（差分能量掩膜）。
  * ≥ 40%  → 掩膜语义大体保持 → G2 降 P1，按提案 2 分支乙登记效度说明
             与成分分解表。
- 差分法口径说明：|H_hp − H_nox_hp| 不是正交分量分解，其值可超过 |H_hp|
  （b_in_mask > 1.0 表示移除该参数引起大幅结构重排/相消干涉，属敏感度
  读数而非能量份额），因此 ch_in_mask 应读作「掩膜能量对 c_high 的
  敏感度」，与语义宣称检验的目的相符；b_in_mask > 0.5 仍作 β 主导的
  方向性证据（与 ch_in_mask < 40% 互证）。
- ch_full 为全视场口径（不排序取掩膜），对照掩膜排序的放大/稀释效应。
- 渲染为纯数值操作（≈0.1s/样本），建议 --n ≥ 30 取稳定中位数。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.f_beam import render_level1_density
from src.generators.masks import DELTA_PX, fine_structure_width
from src.generators.sampling import sample_parameters

F_C = 1.0 / 8.0          # 高通截止（周期/像素，与 30 [S6] C8 / 70 [S5.3] 一致）
ENERGY_FRACTION = 0.90   # 70 [S7.1] 累计能量 90% 掩膜
C_HIGH_KEYS = ("a3", "gamma", "b1")


def highpass(img: np.ndarray, f_c: float = F_C) -> np.ndarray:
    """傅里叶高通分量（径向频率 > f_c），返回实空间场。"""
    n = img.shape[0]
    fy = np.fft.fftfreq(n)
    fx = np.fft.fftfreq(n)
    fr = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
    spec = np.fft.fft2(img)
    spec[fr <= f_c] = 0.0
    return np.real(np.fft.ifft2(spec))


def energy_mask(hp: np.ndarray, frac: float = ENERGY_FRACTION) -> np.ndarray:
    """|hp| 幅值降序取累计能量达 frac 的最小像素集（70 [S7.1] 操作定义）。"""
    mag = np.abs(hp).ravel()
    order = np.argsort(mag)[::-1]
    cum = np.cumsum(mag[order])
    k = int(np.searchsorted(cum, frac * cum[-1]) + 1)
    mask = np.zeros(mag.shape, dtype=bool)
    mask[order[:k]] = True
    return mask.reshape(hp.shape)


def render(c: dict, factor: float) -> np.ndarray:
    sigma_px = factor * float(fine_structure_width(c)) / DELTA_PX
    return render_level1_density(c, sigma_smooth=sigma_px)


def zero(c: dict, keys) -> dict:
    out = dict(c)
    for k in keys:
        out[k] = 0.0
    return out


def analyze(c: dict, factor: float) -> dict:
    H = render(c, factor)
    H_noch = render(zero(c, C_HIGH_KEYS), factor)
    H_nob = render(zero(c, ("beta",)), factor)

    hp, hp_noch, hp_nob = highpass(H), highpass(H_noch), highpass(H_nob)
    mask = energy_mask(hp)

    m_h = np.abs(hp)[mask].sum()
    d_ch = np.abs(hp - hp_noch)
    d_b = np.abs(hp - hp_nob)
    return {
        "beta": float(c["beta"]),
        "w_fine_px": float(fine_structure_width(c)) / DELTA_PX,
        "mask_px_pct": 100.0 * mask.sum() / mask.size,
        "ch_in_mask": d_ch[mask].sum() / m_h,
        "b_in_mask": d_b[mask].sum() / m_h,
        "ch_full": d_ch.sum() / np.abs(hp).sum(),
        "b_full": d_b.sum() / np.abs(hp).sum(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G2 验证：c_high_mask 能量成分分解")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9999)
    ap.add_argument("--factor", type=float, default=0.125)
    ap.add_argument("--compare-old", action="store_true")
    args = ap.parse_args(argv)

    samples, _ = sample_parameters(args.n, master_seed=args.seed, sigma_K=11.0)

    header = (
        "idx | beta   | w_fine_px | mask_px% | ch_in_mask | b_in_mask | ch_full | b_full"
    )
    for factor in ((args.factor, 0.5) if args.compare_old else (args.factor,)):
        rows = [analyze(c, factor) for c in samples]
        print(f"\n=== sigma_smooth,H = {factor} x w_fine | n={args.n} ===")
        print(header)
        for i, r in enumerate(rows):
            print(
                f"{i:3d} | {r['beta']:+.2f} | {r['w_fine_px']:9.2f} | "
                f"{r['mask_px_pct']:8.2f} | {r['ch_in_mask']:10.3f} | "
                f"{r['b_in_mask']:9.3f} | {r['ch_full']:7.3f} | {r['b_full']:6.3f}"
            )
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        print(
            f"MEDIAN ch_in_mask={med['ch_in_mask']:.3f} "
            f"b_in_mask={med['b_in_mask']:.3f} "
            f"ch_full={med['ch_full']:.3f} b_full={med['b_full']:.3f}"
        )
        if factor == args.factor:
            if med["ch_in_mask"] < 0.40:
                print(
                    "VERDICT: ch_in_mask < 40% -> G2 坐实，升无条件 P0，"
                    "按报告 §3 提案 2 分支甲修订 70 [S7.1] 掩膜。"
                )
            else:
                print(
                    "VERDICT: ch_in_mask >= 40% -> 掩膜语义大体保持，G2 降 P1，"
                    "按提案 2 分支乙登记效度说明与成分分解表。"
                )
            if med["b_in_mask"] > 0.50:
                print("NOTE: b_in_mask > 50%，β 折叠主导掩膜能量的直接证据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
