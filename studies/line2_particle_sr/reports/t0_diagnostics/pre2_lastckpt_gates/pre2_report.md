# 前置②：last.ckpt 口径的门禁复算、三点主曲线与常数解正式对照（2026-10-13）

> 性质：Phase 1.5 预注册提案（99）的**附件证据**，纯 CPU 只读复算，零训练。判定一律走官方代码路径
> `src/line2/experiments/gates.py` 的判定函数与 `src/line2/metrics/tolerance.py` 的档位映射；
> 官方判定输出（`results/e1_gates.json`）**未改动**，正式采纳任何口径变更须走 line2 的 99 登记。
> 输入只有两处：best_val 列 = 既有 `results/<run>/test_eval.json`；last 列 = exp1 已落盘的
> `exp1_early_stop/<run>_last_eval.json`（同一 `evaluate.main`、同一 `--station 0 --subs 0-19 --split test`，
> 只换 `--ckpt`）。

## 一、三个最重要答案

1. **H8（last，joint 臂，官方聚合口径）= 0.7118**（best_val 0.1318）。判定由"不成立"升为**"不可判定"**
   （0.50 < 0.7118 < 0.80）；距 80% 判定线还差 **172/1950 个 (束, 观测量) 对**（1388/1950 已达标，需 1560）。
   同口径**换 independent 臂 = 0.9518（成立）**——充分训练后 H8 的"不达标"结论对臂路由不稳健。
   容差档映射两口径完全相同（150 束全落 1e5 档），改善全部来自观测量本身而非换容差线。
2. **N_eff 是否仍全饱和：是，两口径都饱和。** @1e4 两臂 5 种子 × 两口径共 20 个 N_eff 全是 1e5 域顶；
   三点曲线各档 N_eff 中位数也恒为 1e5（last 在 1e5 档连唯一一个非饱和点也消失：上溢 5/5，best_val 4/5）。
   故官方 H1/H3 的**字面判定一个字都没变**（成立 / 成立），但其输入全是截断值——现有 N_eff 准绳对这次
   66% 级的改善**完全不可分辨**。
3. **模型 vs 常数解（@1e4 joint 5 种子，last）：彻底反转。** 逐束赢率 **150/150（5 种子束中位数口径；
   逐种子 130–150/150）**，中位 advantage +0.0993（const − model），bootstrap 95% CI [+0.0802, +0.1403]，
   Wilcoxon p < 2.3e-26。best_val 口径同法复算为 **47/150 = 0.3133**、中位 advantage −0.0687
   （即旧结论"常数解更优"）。逐种子明细见 `pre2_const_pairs.csv`。

## 二、前后对照表

主表：**`pre2_compare.csv`**（机器可读，35 行 × `quantity, best_val, last, note`）；
同内容 Markdown 版 `pre2_compare_table.md`。关键行摘录：

| 量 | best_val（官方判定） | last | 备注 |
|---|---|---|---|
| H1 判定（@1e4 joint） | 成立 | 成立 | 5/5 上溢、中位数 1e5 ≥ 2e4；饱和成立 |
| H3 判定（N_eff 口径） | 成立 | 成立 | 平局 5/5、严格联合胜出 0/5、`tied_at_domain_edge=True` |
| H3 err 口径（次级准绳） | 方向成立（4/5） | **不成立（1/5）** | 见 §三 |
| H8 达标率（joint） | 0.1318（不成立） | **0.7118（不可判定）** | 差 172 对到 80% 线 |
| H8 达标率（independent） | 0.5174（不可判定） | **0.9518（成立）** | 臂路由披露 |
| 主曲线 3e3：err / N_eff 中位 / 上溢 | 0.2780 / 1e5 / 5 | 0.1017 / 1e5 / 5 | 相对 1e5 裸误差 0.285636：0.973 → 0.356 |
| 主曲线 1e4：err / N_eff 中位 / 上溢 | 0.2632 / 1e5 / 5 | 0.0976 / 1e5 / 5 | 0.921 → 0.342 |
| 主曲线 1e5：err / N_eff 中位 / 上溢 | 0.2633 / 1e5 / 4 | 0.0858 / 1e5 / 5 | 0.922 → 0.300 |
| 模型 vs 常数解：逐束赢率 | 47/150 = 0.313 | **150/150 = 1.000** | 5 种子束中位数口径 |
| 模型 vs 常数解：中位 advantage（CI） | −0.0687（[−0.0837, −0.0302]） | **+0.0993（[+0.0802, +0.1403]）** | >0 = 模型更好 |
| 模型 vs 常数解：Wilcoxon p | 1.19e-10 | 2.30e-26 | 双尾；2.3e-26 为 scipy 最小可表示值 |

**复算自校验**（本表的可信度前提，全部通过）：best_val 列由 `results/<run>/test_eval.json` 重算的
H1/H3/H8 判定与数值与官方 `results/e1_gates.json` **逐位一致**（H8 pass_rate `0.1317948717948718`）；
与 exp3 的常数解对照在 5/5 个 joint run 上 `n_wins` 与中位 advantage **逐位一致**；exp1 的
`<run>_bestval_eval.json` 与官方 `test_eval.json` 的束级 err 与 13 维 obs **最大绝对差 = 0**。
详见 `pre2_gates_compare.json` 的 `official_gates_reproduction` / `exp3_cross_check` / `control_rerun_vs_official`。

## 三、H3 在 err 口径下的新读数（次级准绳，**非**预注册判据）

官方 H3 判据吃 N_eff，@1e4 两臂全部上溢 ⇒ 5 对全是域顶平局，无区分力。把同一判定函数作用在连续
`err` 上（取负号以保持 gates "越大越好"的取向），结果如下（逐对明细 `pre2_h3_err_pairs.csv`）：

| 种子 | best_val：joint / indep err 中位 | 胜者 | last：joint / indep err 中位 | 胜者 |
|---|---|---|---|---|
| 100 | 0.2548 / 0.2849 | joint | 0.0976 / 0.0754 | independent |
| 101 | 0.2811 / 0.2818 | joint | 0.0994 / 0.0812 | independent |
| 102 | 0.2664 / 0.2751 | joint | 0.1056 / 0.0731 | independent |
| 103 | 0.2259 / 0.2218 | independent | 0.0817 / 0.1256 | joint |
| 104 | 0.2632 / 0.2755 | joint | 0.0884 / 0.0707 | independent |
| 计票 | **4/5 joint** → "方向成立" | | **1/5 joint** → "不成立" | |

- **best_val**：pooled 中位 Δerr（joint − indep）= −0.00960，CI [−0.00985, −0.00901]，150/150 束同向。
- **last**：pooled 中位 Δerr = **+0.00979**，CI [+0.00923, +0.01011]，**150/150 束全部指向 independent**；
  逐种子 CI 全部排除 0（5/5），Wilcoxon p 全部 ≤ 6.0e-24。
- **读法**：把准绳从 N_eff 换成 err 后，1e4 档的"联合 ≥ 独立"在**充分训练下反向**。原 H3"成立"完全来自
  域顶平局计入胜的字面规则，不携带证据；因此**"联合优于独立"这一条在两种口径下都没有稳健支持**
  （N_eff 口径无区分力、err 口径方向相反）。附带地，独立臂在 err 与 H8 两个连续量上都更好，
  与 exp2"κ 条件化是旁路"互补：joint 臂多出的参数先验路径在充分训练下是负担。

## 四、H8：改善来自哪些观测量，还差什么

13 项观测量逐项达标率（150 束，官方聚合口径；`pre2_gates_compare.json → gates.*.H8_*.per_observable`）：

| 观测量 | best_val joint | last joint | best_val indep | last indep |
|---|---|---|---|---|
| sigma_xx_rel | 0.000 | 0.300 | 0.393 | **1.000** |
| sigma_xpxp_rel | 0.453 | 0.460 | 0.167 | 0.780 |
| sigma_yy_rel | 0.000 | **0.000** | 0.873 | **1.000** |
| sigma_ypyp_rel | 0.000 | 0.520 | 0.500 | 0.867 |
| abs_delta_rho_x_xp | 0.000 | 0.780 | 0.513 | 0.980 |
| abs_delta_rho_x_y | 0.000 | **1.000** | 1.000 | **1.000** |
| abs_delta_rho_x_yp | 0.000 | **1.000** | 1.000 | **1.000** |
| abs_delta_rho_xp_y | 0.120 | **1.000** | 1.000 | **1.000** |
| abs_delta_rho_xp_yp | 0.140 | **1.000** | 0.980 | **1.000** |
| abs_delta_rho_y_yp | 0.000 | 0.987 | 0.147 | **1.000** |
| eps4d_rel | 1.000 | 0.887 | 0.033 | 0.747 |
| m5_halo_fraction_gap | 0.000 | 0.320 | 0.120 | **1.000** |
| m6_hf_energy_ratio_gap | 0.000 | **1.000** | 0.000 | **1.000** |

- joint 臂残余缺口集中在 **sigma_yy_rel（0/150，完全没修好）、sigma_xx_rel（0.30）、m5_halo（0.32）、
  sigma_xpxp（0.46）、sigma_ypyp（0.52）**——即 4 个二阶矩对角项与 halo 分数；6 项相关变化与 M6 已被修复。
- independent 臂在 last 口径下 11/13 项 ≥0.86，仅 eps4d_rel（0.747）与 sigma_xpxp_rel（0.780）低于 0.8。
- 种子间离散很大（披露口径，非官方聚合）：last joint 逐种子达标率 0.327–0.897（中位 0.690），
  best_val joint 0.060–0.505（中位 0.170）；last independent 0.764–0.926（中位 0.881）。

## 五、三点主曲线（joint，best_val vs last）

数值源 `pre2_curve_points.csv`；err = 5 个 run 的 `err_median` 再取中位数，N_eff 同法。

| N_low | err 中位（best_val → last） | N_eff 中位 / 上溢数（best_val → last） | 放大比 N_eff/N_low | 相对 1e5 档裸误差 0.285636 |
|---|---|---|---|---|
| 3×10³ | 0.2780 → **0.1017** | 1e5 / 5 → 1e5 / 5 | 33.33（两口径同值） | 0.973 → 0.356 |
| 1×10⁴ | 0.2632 → **0.0976** | 1e5 / 5 → 1e5 / 5 | 10.00（两口径同值） | 0.921 → 0.342 |
| 1×10⁵ | 0.2633 → **0.0858** | 1e5 / 4 → 1e5 / 5 | 1.00（两口径同值） | 0.922 → 0.300 |

（independent @1e4：0.2755 → **0.0754**，N_eff 1e5 / 5 不变。）

- **N_eff 在两口径下都饱和**：三点、两臂、两口径的 N_eff 中位数恒为域顶 1e5，因此"放大比 33.3×/10×"
  是 `1e5 / N_low` 的构造值，last 口径下连**唯一一个非饱和点**（best_val 的 `e2_joint_n100000_s2`，
  N_eff = 98,724）也消失——曲线右端原有的"可信度直接证据"不复存在。
- last 口径下三点 err **全部低于 1e5 档裸误差**（0.356/0.342/0.300 倍）⇒ 网络已越过尺子右端。
  用裸曲线 3e4→1e5 段局部 log-log 斜率（−0.7205）外推，三个点对应 N ≈ 4.2e5 / 4.4e5 / 5.3e5
  （≈ 42×/44×/53× 的 1e4 工作点）——**这是域外外推，不是测量**（`20` [S3] 报告域顶 = 1e5），
  只说明"现有尺子量不到"，要真测需向右扩尺（更密/更大的裸曲线档）。

## 六、可复现命令行

```bash
cd /home/zhangny/Super-Resolution-Simulation
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sr-sim
python studies/line2_particle_sr/reports/t0_diagnostics/pre2_lastckpt_gates/run_pre2_gates.py
```

脚本只读 `studies/line2_particle_sr/{results,data}`、`reports/t0_diagnostics/exp1_early_stop`、
`reports/t0_diagnostics/exp3_constant_control`、`reports/tools/phase1_numbers.json`，全部产物写本目录；
随机性经 `np.random.SeedSequence(20260825)` 三类命名子流（H3 err / 常数解 / 曲线），无裸随机调用；
同输入两次运行输出一致。运行时间约 3 s。

## 七、产物清单

| 文件 | 内容 |
|---|---|
| `run_pre2_gates.py` | 复算脚本（自写 loader + 官方判定函数；不写 `results/`） |
| `pre2_compare.csv` / `pre2_compare_table.md` | 前后对照主表（CSV / Markdown，同内容） |
| `pre2_gates_compare.json` | 全量载荷：H1/H3/H8（两臂 × 两口径）、官方复现对账、exp3 对账、逐束/逐种子细节、H8 逐观测量达标率、暗桶 M2、曲线外推披露 |
| `pre2_curve_points.csv` | 三点主曲线（含 independent @1e4），err / N_eff / 放大比 / 上溢数 |
| `pre2_h3_err_pairs.csv` | H3 err 口径逐种子对逐对读数 |
| `pre2_const_pairs.csv` | 模型 vs 常数解逐种子读数 |
| `pre2_report.md` | 本文 |

## 八、偏差与边界

- **exp1 产物字段与 gates 预期完全兼容**：`split="test"`、15 个束级字段齐全（含 `obs` 13 维与 `m2_buckets`），
  H8 暗桶报告可正常产出（`runs_without_m2` 为空），无字段缺失或口径不一致。
- **H8 last 仍不达标是本任务最重要的单点发现**：0.7118 落在"不可判定"档而非"成立"，且**同一训练充分的
  臂路由（independent）已达 0.9518**。现路由是 val 段 N_eff 域顶平局的产物（99 已有登记），
  last 口径把它的代价显性化：门禁被推向了较差的一条臂。
- **last.ckpt 是"训练最充分的在盘 ckpt"，不是"最优 ckpt"**：预算只用了 22–26%，val_nll 在末步仍下降
  ⇒ 上表所有 last 数字都是**下界性质**，不是该选型准绳的收敛值。
- **Wilcoxon p 在 150 束全集上撞 scipy 最小值 2.3e-26**（数值下溢，非精确 p）。
- **H8 官方聚合口径下的种子压缩**：逐种子达标率（last joint 0.327–0.897）被"逐观测量先取种子中位数"
  压成单一 0.7118；该聚合是 `20` [S4] 预注册次序，本任务未改动，披露值仅作稳健性参考。
- **未覆盖**：站 1（跨站 H7）、OOD、3e3/1e5 档的 independent 臂（E2 只训了获胜变体）均不在本复算范围；
  H8 的容差线构造问题（exp4 已证 M6 1e5 档容差线非噪声线）未重新裁定，本任务按官方规则如实映射。
