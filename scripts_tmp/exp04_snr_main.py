#!/usr/bin/env python
"""EXP-04 可行性校验：主束区域信噪比硬下限（30 [S7] C7）。

目的
----
30 [S7] C5 定义 EXP-04（高噪声档）：σ_n 调至尾部区域信噪比 ≈1、σ_K = D2；
30 [S7] C7 是硬门：生成后主束区域平均信噪比
    SNR_main = mean_M(L_clean) / σ_n ≥ 2，
其中 M = {(i,j): L_clean ≥ 0.1·max(L_clean)}，尾部区域
T = {(i,j): L_clean < 0.1·max(L_clean)}。
关键观察：SNR_main / SNR_tail = mean_M / mean_T **与 σ_n 无关**，
因此可行性可以在生成前判定。本脚本在当前 D2 标定（σ_K=11.0、
σ_n=1.22e-4，尾部 2 档）的 L_clean 结构上计算该比值，并外推
EXP-04 档（尾部 ≈1）下的 SNR_main 是否仍 ≥2。

依赖
----
stdlib + numpy + scipy + src.generators。不依赖 torch。

用法
----
    python scripts_tmp/exp04_snr_main.py [--n 8] [--seed 8888]
                                         [--factor 0.5] [--sigma-k 11.0]

    --factor   σ_smooth,H = factor × w_fine（默认 0.5 = 当前 v1 口径；
               P0 批准后建议 --factor 0.125 复算 v2 口径）
    --sigma-k  D2 模糊核（默认 11.0，M2 登记值）

预期输出表
----------
    idx | snr_tail(cur) | snr_main(cur) | ratio_M/T
    MEDIAN ...
    EXP-04: sigma_n*（尾部=1）、SNR_main(EXP-04) 中位数
    VERDICT: ...

解读规则
--------
- snr_full(cur)（全图均值口径）中位数应落在 2–5——σ_n=1.22e-4 正是按此
  口径标定的（OQ-30-02）；snr_tail(cur) 低约 ratio 倍属预期（尾部口径已
  随 OQ-30-02 分支降为诊断量）。snr_full 偏离 2–5 才需核查标定。
- SNR_main(EXP-04 档) 中位数 ≥2 → C7 可行，EXP-04 按 30 [S7] C5 选参。
- <2 → C7 与 C5（尾部 ≈1）冲突，按报告 §3 提案 15 回退：σ_n 取使
  SNR_main=2 恰好成立的值，此时实际尾部信噪比 = 2 / ratio 中位数，
  偏离值登记 99。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.f_beam import render_level1_density
from src.generators.f_deg import f_deg
from src.generators.masks import DELTA_PX, fine_structure_width
from src.generators.sampling import sample_parameters

SIGMA_N_CURRENT = 1.22e-4   # M2 登记值（OQ-30-02：尾部 2 档妥协口径）
TAIL_FRACTION = 0.1         # 30 [S7] C7：M/T 以 10% 峰值为界


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EXP-04 SNR_main 可行性校验（30 [S7] C7）")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=8888)
    ap.add_argument("--factor", type=float, default=0.5)
    ap.add_argument("--sigma-k", type=float, default=11.0)
    args = ap.parse_args(argv)

    samples, _ = sample_parameters(args.n, master_seed=args.seed, sigma_K=args.sigma_k)

    ratios, tail_cur, main_cur, full_cur, mean_m, mean_t = [], [], [], [], [], []
    print(f"=== sigma_K={args.sigma_k} | sigma_n(current)={SIGMA_N_CURRENT} "
          f"| sigma_smooth,H={args.factor}x w_fine | n={args.n} ===")
    print("idx | snr_tail(cur) | snr_main(cur) | snr_full(cur) | ratio_M/T")
    for i, c in enumerate(samples):
        sigma_px = args.factor * float(fine_structure_width(c)) / DELTA_PX
        H = render_level1_density(c, sigma_smooth=sigma_px)
        _, L_clean, _, _ = f_deg(H, sigma_K=args.sigma_k, sigma_n=SIGMA_N_CURRENT)
        peak = L_clean.max()
        m_mask = L_clean >= TAIL_FRACTION * peak
        t_mask = ~m_mask
        mm = float(L_clean[m_mask].mean())
        mt = float(L_clean[t_mask].mean()) if t_mask.any() else 0.0
        ratios.append(mm / mt if mt > 0 else float("inf"))
        tail_cur.append(mt / SIGMA_N_CURRENT)
        main_cur.append(mm / SIGMA_N_CURRENT)
        full_cur.append(float(L_clean.mean()) / SIGMA_N_CURRENT)
        mean_m.append(mm)
        mean_t.append(mt)
        print(f"{i:3d} | {tail_cur[-1]:13.2f} | {main_cur[-1]:13.2f} | "
              f"{full_cur[-1]:13.2f} | {ratios[-1]:9.2f}")

    med_ratio = float(np.median(ratios))
    med_tail = float(np.median(tail_cur))
    med_main = float(np.median(main_cur))
    med_full = float(np.median(full_cur))
    print(f"MEDIAN snr_tail(cur)={med_tail:.2f} snr_main(cur)={med_main:.2f} "
          f"snr_full(cur)={med_full:.2f} ratio_M/T={med_ratio:.2f}")
    print(f"口径说明：σ_n=1.22e-4 按『全图均值口径 ≈2』标定（OQ-30-02），"
          f"故 snr_full≈2 而 snr_tail 低约 ratio 倍属预期（尾部口径已降诊断量，"
          f"OQ-30-02 分支）；若 snr_full 也显著偏离 2–5 才需核查标定。")
    print(f"参考：SNR_main/SNR_tail = mean_M/mean_T = {med_ratio:.2f}，"
          f"该比值与 sigma_n 无关（L_clean 结构量）。")

    # EXP-04 外推：尾部信噪比 = 1 档 → sigma_n* = mean_T，逐样本
    exp04_main = np.array(ratios)  # SNR_main(尾部=1) = mean_M/mean_T
    med_exp04 = float(np.median(exp04_main))
    print(f"EXP-04（尾部≈1 档）：SNR_main 中位数 = {med_exp04:.2f}")

    ok = True
    if not (2.0 <= med_full <= 5.0):
        print(f"WARN: 全图均值口径中位数 {med_full:.2f} 不在 2–5，"
              "σ_n=1.22e-4 的标定登记与实际结构不一致，先核查标定口径。")
        ok = False
    if med_exp04 >= 2.0:
        print("VERDICT: SNR_main(EXP-04) >= 2 -> C7 可行，EXP-04 按 30 [S7] C5 选参。")
    else:
        fallback_tail = 2.0 / med_ratio
        print(f"VERDICT: SNR_main(EXP-04) = {med_exp04:.2f} < 2 -> C7 与尾部≈1 冲突。"
              f"按报告 §3 提案 15 回退：尾部信噪比目标改为 {fallback_tail:.2f}"
              f"（即 sigma_n = {np.median(mean_t) * fallback_tail:.3e} 量级），偏离值登记 99。")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
