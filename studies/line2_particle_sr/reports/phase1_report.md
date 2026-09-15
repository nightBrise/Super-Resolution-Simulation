# line2 Phase 1 实验报告（方法可用域地图 + N_eff 表 + 门禁结论）

> Spec：`docs/specs/line2/40_experiment_spec.md`（Frozen v1.0）[S4] 产出物；判据与 N_eff 见 `20`，模型与训练见 `30`，总纲与全局约束见 `00`。
> 生成方式：`python -m src.line2.experiments.map_report --results-dir <dir>`（只读；同输入逐位一致，[S4] C1）。
> 输入产物：`e0_curve.json`、`e1_gates.json`、25 份 `*/test_eval*.json`、20 份 `*/run_meta.json`。
> 门禁产物 code_version：306d3964f60a03bc3fd3cc57eef4bbdd2b289c56（产生 `e1_gates.json` 的判定代码版本；`00` GC7 要求的每个 run 的三元组见 §1.2）｜数据版本：line2_ds_v1｜spec 版本：line2-spec-v1.0
> 配套产出物：地图 `studies/line2_particle_sr/reports/figure_avail_domain.png`、N_eff 表 `studies/line2_particle_sr/results/n_eff_table.csv`。

## 1. 数据

### 1.1 E0 裸误差–N 曲线（`20` [S2]，N_eff 的唯一插值基底）

| 项 | 值 |
|---|---|
| 曲线路径 | studies/line2_particle_sr/results/e0_curve.json |
| 档位 tiers | 1000, 3000, 10000, 30000, 100000 |
| 每档子样本数 k | 20 |
| 束数 | 300 |
| 束范围 beams_scope | valtest |
| E0 started_at | 2026-09-14T04:09:42.433167+00:00 |
| E0 completed_at | 2026-09-14T04:11:22.933516+00:00 |
| E0 data_version | line2_ds_v1 |
| E0 code_version | 77486ae42490a2035a1a64c058ba1b8cd8f9619e |
| E0 spec_version | line2-spec-v1.0 |
| 参照噪声地板（`20` [S2] C3） | 0.285597 |

五档曲线（跨束 M1 中位数与 IQR；互差地板 = 同档两两子样本 M1 差中位数）：

| N | M1 中位数 | IQR 下 | IQR 上 | n_samples | 互差地板 |
|---|---|---|---|---|---|
| 1000 | 4.03003 | 3.9906 | 4.07032 | 12000 | 5.70119 |
| 3000 | 2.3161 | 2.29342 | 2.33792 | 12000 | 3.27519 |
| 10000 | 1.24553 | 1.23348 | 1.25764 | 12000 | 1.76137 |
| 30000 | 0.680086 | 0.673579 | 0.686695 | 12000 | 0.961663 |
| 100000 | 0.285636 | 0.282938 | 0.288472 | 12000 | 0.403895 |

### 1.2 GC7 三元组（run 级溯源）

| run_id | data_version | code_version | spec_version | variant | seed | n_low | max_steps |
|---|---|---|---|---|---|---|---|
| e1_indep_s0 | line2_ds_v1 | 21aaedbd02ee4353acd7342d0836c4df4029f104 | line2-spec-v1.0 | independent | 100 | 10000 | 50000 |
| e1_indep_s1 | line2_ds_v1 | 3cb412bb68265102ec80c76d864a75902dc545f1 | line2-spec-v1.0 | independent | 101 | 10000 | 50000 |
| e1_indep_s2 | line2_ds_v1 | a122ec57fe9ad81143f6be4e04069fdbf3900473 | line2-spec-v1.0 | independent | 102 | 10000 | 50000 |
| e1_indep_s3 | line2_ds_v1 | a7b3530c8bdd1eb56ae9f8c3fc660e17c9597015 | line2-spec-v1.0 | independent | 103 | 10000 | 50000 |
| e1_indep_s4 | line2_ds_v1 | a7b3530c8bdd1eb56ae9f8c3fc660e17c9597015 | line2-spec-v1.0 | independent | 104 | 10000 | 50000 |
| e1_joint_s0 | line2_ds_v1 | 21aaedbd02ee4353acd7342d0836c4df4029f104 | line2-spec-v1.0 | joint | 100 | 10000 | 50000 |
| e1_joint_s1 | line2_ds_v1 | 3cb412bb68265102ec80c76d864a75902dc545f1 | line2-spec-v1.0 | joint | 101 | 10000 | 50000 |
| e1_joint_s2 | line2_ds_v1 | 3cb412bb68265102ec80c76d864a75902dc545f1 | line2-spec-v1.0 | joint | 102 | 10000 | 50000 |
| e1_joint_s3 | line2_ds_v1 | a7b3530c8bdd1eb56ae9f8c3fc660e17c9597015 | line2-spec-v1.0 | joint | 103 | 10000 | 50000 |
| e1_joint_s4 | line2_ds_v1 | a7b3530c8bdd1eb56ae9f8c3fc660e17c9597015 | line2-spec-v1.0 | joint | 104 | 10000 | 50000 |
| e2_joint_n100000_s0 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 100 | 100000 | 50000 |
| e2_joint_n100000_s1 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 101 | 100000 | 50000 |
| e2_joint_n100000_s2 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 102 | 100000 | 50000 |
| e2_joint_n100000_s3 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 103 | 100000 | 50000 |
| e2_joint_n100000_s4 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 104 | 100000 | 50000 |
| e2_joint_n3000_s0 | line2_ds_v1 | a7b3530c8bdd1eb56ae9f8c3fc660e17c9597015 | line2-spec-v1.0 | joint | 100 | 3000 | 50000 |
| e2_joint_n3000_s1 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 101 | 3000 | 50000 |
| e2_joint_n3000_s2 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 102 | 3000 | 50000 |
| e2_joint_n3000_s3 | line2_ds_v1 | a7b3530c8bdd1eb56ae9f8c3fc660e17c9597015 | line2-spec-v1.0 | joint | 103 | 3000 | 50000 |
| e2_joint_n3000_s4 | line2_ds_v1 | 306d3964f60a03bc3fd3cc57eef4bbdd2b289c56 | line2-spec-v1.0 | joint | 104 | 3000 | 50000 |

## 2. 设置

### 2.1 门禁聚合维度（`40` [S3] 预注册）

| 项 | 值 |
|---|---|
| n_low | 10000 |
| n_seeds_expected | 5 |
| split | test |
| station | 0 |

### 2.2 变体路由（`40` [S1] C4、[S3]；只消费 val 段）

- 获胜变体：**joint**
- 路由依据：val

| 项 | 值 |
|---|---|
| source | summary.json:best_val_n_eff |
| consumed_by | variant_routing |
| n_pairs | 5 |
| val N_eff[independent][100] | 100000 |
| val N_eff[independent][101] | 100000 |
| val N_eff[independent][102] | 100000 |
| val N_eff[independent][103] | 100000 |
| val N_eff[independent][104] | 100000 |
| val N_eff[joint][100] | 100000 |
| val N_eff[joint][101] | 100000 |
| val N_eff[joint][102] | 100000 |
| val N_eff[joint][103] | 100000 |
| val N_eff[joint][104] | 100000 |

### 2.3 尺子先行审计（`40` [S1] C5）

- E0 completed_at：2026-09-14T04:11:22.933516+00:00
- 已核验 run 数：20（全部 started_at 晚于 E0 completed_at）

| run_id | started_at |
|---|---|
| e1_indep_s0 | 2026-09-14T06:23:30.573935+00:00 |
| e1_indep_s1 | 2026-09-14T08:14:31.224819+00:00 |
| e1_indep_s2 | 2026-09-14T10:05:11.191885+00:00 |
| e1_indep_s3 | 2026-09-14T11:55:34.942008+00:00 |
| e1_indep_s4 | 2026-09-14T13:55:56.511873+00:00 |
| e1_joint_s0 | 2026-09-14T06:23:30.572453+00:00 |
| e1_joint_s1 | 2026-09-14T07:55:29.228397+00:00 |
| e1_joint_s2 | 2026-09-14T09:20:30.810315+00:00 |
| e1_joint_s3 | 2026-09-14T10:47:18.931852+00:00 |
| e1_joint_s4 | 2026-09-14T12:17:37.368205+00:00 |
| e2_joint_n100000_s0 | 2026-09-14T20:31:35.581057+00:00 |
| e2_joint_n100000_s1 | 2026-09-14T21:58:00.424828+00:00 |
| e2_joint_n100000_s2 | 2026-09-14T19:03:01.468354+00:00 |
| e2_joint_n100000_s3 | 2026-09-14T20:21:15.566076+00:00 |
| e2_joint_n100000_s4 | 2026-09-14T21:43:16.102283+00:00 |
| e2_joint_n3000_s0 | 2026-09-14T16:10:45.357370+00:00 |
| e2_joint_n3000_s1 | 2026-09-14T17:34:17.850831+00:00 |
| e2_joint_n3000_s2 | 2026-09-14T19:04:24.110913+00:00 |
| e2_joint_n3000_s3 | 2026-09-14T16:10:45.365345+00:00 |
| e2_joint_n3000_s4 | 2026-09-14T17:37:19.174550+00:00 |

### 2.4 run 清单（test 段评估，`00` GC3：只评一次）

| run_id | variant | seed | station | n_low | split | n_beams |
|---|---|---|---|---|---|---|
| e2_joint_n3000_s0 | joint | 100 | 0 | 3000 | test | 150 |
| e2_joint_n3000_s1 | joint | 101 | 0 | 3000 | test | 150 |
| e2_joint_n3000_s2 | joint | 102 | 0 | 3000 | test | 150 |
| e2_joint_n3000_s3 | joint | 103 | 0 | 3000 | test | 150 |
| e2_joint_n3000_s4 | joint | 104 | 0 | 3000 | test | 150 |
| e1_indep_s0 | independent | 100 | 0 | 10000 | test | 150 |
| e1_indep_s1 | independent | 101 | 0 | 10000 | test | 150 |
| e1_indep_s2 | independent | 102 | 0 | 10000 | test | 150 |
| e1_indep_s3 | independent | 103 | 0 | 10000 | test | 150 |
| e1_indep_s4 | independent | 104 | 0 | 10000 | test | 150 |
| e1_joint_s0 | joint | 100 | 0 | 10000 | test | 150 |
| e1_joint_s1 | joint | 101 | 0 | 10000 | test | 150 |
| e1_joint_s2 | joint | 102 | 0 | 10000 | test | 150 |
| e1_joint_s3 | joint | 103 | 0 | 10000 | test | 150 |
| e1_joint_s4 | joint | 104 | 0 | 10000 | test | 150 |
| e2_joint_n100000_s0 | joint | 100 | 0 | 100000 | test | 150 |
| e2_joint_n100000_s1 | joint | 101 | 0 | 100000 | test | 150 |
| e2_joint_n100000_s2 | joint | 102 | 0 | 100000 | test | 150 |
| e2_joint_n100000_s3 | joint | 103 | 0 | 100000 | test | 150 |
| e2_joint_n100000_s4 | joint | 104 | 0 | 100000 | test | 150 |
| e1_joint_s0 | joint | 100 | 1 | 10000 | test | 150 |
| e1_joint_s1 | joint | 101 | 1 | 10000 | test | 150 |
| e1_joint_s2 | joint | 102 | 1 | 10000 | test | 150 |
| e1_joint_s3 | joint | 103 | 1 | 10000 | test | 150 |
| e1_joint_s4 | joint | 104 | 1 | 10000 | test | 150 |

## 3. 结果

### 3.1 N_eff 明细表

落盘：`studies/line2_particle_sr/results/n_eff_table.csv`（列序 = run_id, variant, seed, n_low, station, split, n_eff, n_eff_flag, err_median, n_beams）。

| run_id | variant | seed | n_low | station | split | n_eff | n_eff_flag | err_median | n_beams |
|---|---|---|---|---|---|---|---|---|---|
| e2_joint_n3000_s0 | joint | 100 | 3000 | 0 | test | 100000 | overflow | 0.269786 | 150 |
| e2_joint_n3000_s1 | joint | 101 | 3000 | 0 | test | 100000 | overflow | 0.281423 | 150 |
| e2_joint_n3000_s2 | joint | 102 | 3000 | 0 | test | 100000 | overflow | 0.262211 | 150 |
| e2_joint_n3000_s3 | joint | 103 | 3000 | 0 | test | 100000 | overflow | 0.277993 | 150 |
| e2_joint_n3000_s4 | joint | 104 | 3000 | 0 | test | 100000 | overflow | 0.281208 | 150 |
| e1_indep_s0 | independent | 100 | 10000 | 0 | test | 100000 | overflow | 0.284933 | 150 |
| e1_indep_s1 | independent | 101 | 10000 | 0 | test | 100000 | overflow | 0.281785 | 150 |
| e1_indep_s2 | independent | 102 | 10000 | 0 | test | 100000 | overflow | 0.275088 | 150 |
| e1_indep_s3 | independent | 103 | 10000 | 0 | test | 100000 | overflow | 0.221848 | 150 |
| e1_indep_s4 | independent | 104 | 10000 | 0 | test | 100000 | overflow | 0.275505 | 150 |
| e1_joint_s0 | joint | 100 | 10000 | 0 | test | 100000 | overflow | 0.254818 | 150 |
| e1_joint_s1 | joint | 101 | 10000 | 0 | test | 100000 | overflow | 0.281061 | 150 |
| e1_joint_s2 | joint | 102 | 10000 | 0 | test | 100000 | overflow | 0.266393 | 150 |
| e1_joint_s3 | joint | 103 | 10000 | 0 | test | 100000 | overflow | 0.225916 | 150 |
| e1_joint_s4 | joint | 104 | 10000 | 0 | test | 100000 | overflow | 0.263158 | 150 |
| e2_joint_n100000_s0 | joint | 100 | 100000 | 0 | test | 100000 | overflow | 0.263266 | 150 |
| e2_joint_n100000_s1 | joint | 101 | 100000 | 0 | test | 100000 | overflow | 0.258587 | 150 |
| e2_joint_n100000_s2 | joint | 102 | 100000 | 0 | test | 98723.7 | — | 0.289844 | 150 |
| e2_joint_n100000_s3 | joint | 103 | 100000 | 0 | test | 100000 | overflow | 0.284655 | 150 |
| e2_joint_n100000_s4 | joint | 104 | 100000 | 0 | test | 100000 | overflow | 0.25309 | 150 |
| e1_joint_s0 | joint | 100 | 10000 | 1 | test | 100000 | overflow | 0.244437 | 150 |
| e1_joint_s1 | joint | 101 | 10000 | 1 | test | 100000 | overflow | 0.273902 | 150 |
| e1_joint_s2 | joint | 102 | 10000 | 1 | test | 100000 | overflow | 0.236621 | 150 |
| e1_joint_s3 | joint | 103 | 10000 | 1 | test | 100000 | overflow | 0.208691 | 150 |
| e1_joint_s4 | joint | 104 | 10000 | 1 | test | 100000 | overflow | 0.236336 | 150 |

### 3.2 主曲线聚合（N_low = 3e3, 1e4, 1e5；方法可用域地图面板 A 的数值）

放大比 = N_eff / N_low；N_eff 取 `20` [S3] 报告域 [1e3, 1e5]，上/下溢行以域端点计入并在表中计数（图内以空心叉号标注）。

| station | variant | n_low | n_runs | ratio 中位数 | ratio [min, max] | 上溢数 | 下溢数 | err_median 中位数 |
|---|---|---|---|---|---|---|---|---|
| 0 | independent | 10000 | 5 | 10 | [10, 10] | 5 | 0 | 0.275505 |
| 0 | joint | 3000 | 5 | 33.3333 | [33.3333, 33.3333] | 5 | 0 | 0.277993 |
| 0 | joint | 10000 | 5 | 10 | [10, 10] | 5 | 0 | 0.263158 |
| 0 | joint | 100000 | 5 | 1 | [0.987237, 1] | 4 | 0 | 0.263266 |
| 1 | joint | 10000 | 5 | 10 | [10, 10] | 5 | 0 | 0.236621 |

### 3.3 跨站泛化（`40` [S2]，站 1 的 test 束）

| run_id | variant | seed | n_low | n_eff | n_eff_flag | err_median |
|---|---|---|---|---|---|---|
| e1_joint_s0 | joint | 100 | 10000 | 100000 | overflow | 0.244437 |
| e1_joint_s1 | joint | 101 | 10000 | 100000 | overflow | 0.273902 |
| e1_joint_s2 | joint | 102 | 10000 | 100000 | overflow | 0.236621 |
| e1_joint_s3 | joint | 103 | 10000 | 100000 | overflow | 0.208691 |
| e1_joint_s4 | joint | 104 | 10000 | 100000 | overflow | 0.236336 |

## 4. 失败模式与不可验证项

- 20/20 个 run 目录缺 `config.yaml`（`40` [S1] C2 契约文件）：e1_indep_s0, e1_indep_s1, e1_indep_s2, e1_indep_s3, e1_indep_s4, e1_joint_s0, e1_joint_s1, e1_joint_s2, e1_joint_s3, e1_joint_s4, e2_joint_n100000_s0, e2_joint_n100000_s1, e2_joint_n100000_s2, e2_joint_n100000_s3, e2_joint_n100000_s4, e2_joint_n3000_s0, e2_joint_n3000_s1, e2_joint_n3000_s2, e2_joint_n3000_s3, e2_joint_n3000_s4——配置实由 `docs/specs/line2/e1_configs/*.json` 冻结副本与 `run_meta.json` 的 `config`/`cmd` 字段承载；该缺口登记于 `40` 的 decisions，本报告只披露、不补写产物。
- H7：n_eff_flag_present（判定降级为 不可判定）。
- 24 行 N_eff 触及报告域端点（上溢 24 / 下溢 0）——放大比在端点处为截断值，只声明达到域顶/域底，不可外推（`20` [S3] C2）。

## 5. 门禁结论（每条附判定表达式的输入数值快照，`40` [S4] C2）

- 获胜变体：**joint**（路由依据 val；`40` [S1] C4）

### 5.1 H1（放大存在性）

- **结论**：成立
- **判定表达式**（`40` [S3]，阈值冻结）：放大存在性：获胜变体 @1e4 的 5 种子 test N_eff 分布——成立 ⟺ 5/5 种子 N_eff > 1e4 且中位数 ≥ 2×1e4；不成立 ⟺ 中位数 ≤ 1e4；其余不可判定。

**判定表达式的输入数值快照**（取自门禁产物，本脚本不重算判定）：

| 项 | 值 |
|---|---|
| variant | joint |
| n_seeds | 5 |
| median_n_eff | 100000 |
| n_above_n_low | 5 |
| thresholds | median_min=20000, n_low=10000 |
| inputs.100 | 100000 |
| inputs.101 | 100000 |
| inputs.102 | 100000 |
| inputs.103 | 100000 |
| inputs.104 | 100000 |

### 5.2 H3（联合优于独立）

- **结论**：成立
- **判定表达式**（`40` [S3]，阈值冻结）：联合优于独立：同种子成对 5 对——成立 ⟺ 5/5 对联合 N_eff ≥ 独立；方向成立 ⟺ 4/5；不成立 ⟺ ≤ 2/5；3/5 不可判定。

**判定表达式的输入数值快照**（取自门禁产物，本脚本不重算判定）：

| 项 | 值 |
|---|---|
| n_pairs | 5 |
| joint_wins | 5 |
| pairing.basis | seed_value |
| pairing.pairs | 100, 100; 101, 101; 102, 102; 103, 103; 104, 104 |
| inputs.independent[100] | 100000 |
| inputs.independent[101] | 100000 |
| inputs.independent[102] | 100000 |
| inputs.independent[103] | 100000 |
| inputs.independent[104] | 100000 |
| inputs.joint[100] | 100000 |
| inputs.joint[101] | 100000 |
| inputs.joint[102] | 100000 |
| inputs.joint[103] | 100000 |
| inputs.joint[104] | 100000 |
| n_ties | 5 |
| n_strict_joint_wins | 0 |
| tied_at_domain_edge | 是 |
| n_eff_flags.independent[100] | overflow |
| n_eff_flags.independent[101] | overflow |
| n_eff_flags.independent[102] | overflow |
| n_eff_flags.independent[103] | overflow |
| n_eff_flags.independent[104] | overflow |
| n_eff_flags.joint[100] | overflow |
| n_eff_flags.joint[101] | overflow |
| n_eff_flags.joint[102] | overflow |
| n_eff_flags.joint[103] | overflow |
| n_eff_flags.joint[104] | overflow |

- **域顶平局提示**：5/5 对为域端点平局（两侧 N_eff 均携带上/下溢标记，见上表 `n_eff_flags` 与 3.1 的 N_eff 表），该档**不具备区分力**；`40` [S3] H3 把平局计入胜，故上面的结论是**字面结论**，不表示联合严格更优（严格更优仅 0 对）。

### 5.3 H7（跨站泛化）

- **结论**：不可判定
- **判定表达式**（`40` [S3]，阈值冻结）：跨站泛化：ρ = N_eff(站1)/N_eff(站0)——成立 ⟺ ρ ≥ 1.0；成立（弱，报衰减系数）⟺ 0.5 ≤ ρ < 1.0；不成立 ⟺ ρ < 0.5；N_eff 触发上/下溢标记使 ρ 无法计算 ⟺ 不可判定（任一侧携带标记即降级为不可判定，此时 ρ 为截断值之比、仅作披露）。

**判定表达式的输入数值快照**（取自门禁产物，本脚本不重算判定）：

| 项 | 值 |
|---|---|
| rho_median | 1 |
| weak | 否 |
| decay | — |
| variant | joint |
| n_seeds | 5 |
| aggregation | seed_median_then_ratio |
| thresholds | strong=1, weak=0.5 |
| n_eff_station0_median | 100000 |
| n_eff_station1_median | 100000 |
| reason_code | n_eff_flag_present |

逐种子跨站比值：

| seed | rho | verdict | reason_code | flag(站0) | flag(站1) | N_eff(站0) | N_eff(站1) |
|---|---|---|---|---|---|---|---|
| 100 | 1 | 不可判定 | n_eff_flag_present | overflow | overflow | 100000 | 100000 |
| 101 | 1 | 不可判定 | n_eff_flag_present | overflow | overflow | 100000 | 100000 |
| 102 | 1 | 不可判定 | n_eff_flag_present | overflow | overflow | 100000 | 100000 |
| 103 | 1 | 不可判定 | n_eff_flag_present | overflow | overflow | 100000 | 100000 |
| 104 | 1 | 不可判定 | n_eff_flag_present | overflow | overflow | 100000 | 100000 |

- 上/下溢导致的不可判定：5/5 个种子任一侧携带标记（100=station0=overflow, station1=overflow, 101=station0=overflow, station1=overflow, 102=station0=overflow, station1=overflow, 103=station0=overflow, station1=overflow, 104=station0=overflow, station1=overflow）——该侧的 N_eff 是 `20` [S3] 报告域端点（1e3 / 1e5）的截断值，其 ρ 不构成跨站泛化的结论依据（`40` [S2]）；聚合条目因此同样判不可判定（`rho_median` 仅作披露）。

### 5.4 H8（细结构可恢复性）

- **结论**：不成立
- **判定表达式**（`40` [S3]，阈值冻结）：细结构可恢复性：13 项观测量对容差线的达标率（test 束 × 站 0 × 1e4 × 5 种子中位数输出）——成立 ⟺ ≥ 80%；不成立 ⟺ ≤ 50%；之间不可判定。

**判定表达式的输入数值快照**（取自门禁产物，本脚本不重算判定）：

| 项 | 值 |
|---|---|
| pass_rate | 0.131795 |
| n_beams | 150 |
| beam_ids | 2, 13, 21, 24, 25, 41, 42, 46, 54, 55, 56, 67, 70, 88, 96, 105, 113, 114, 134, 157, 184, 186, 190, 193, 197, 208, 213, 219, 220, 223, 224, 235, 241, 242, 244, 245, 247, 253, 264, 283, 285, 287, 292, 296, 299, 303, 324, 328, 343, 360, 370, 379, 387, 396, 404, 406, 411, 416, 424, 425, 427, 428, 429, 430, 446, 454, 459, 460, 467, 492, 493, 500, 502, 503, 506, 510, 511, 522, 530, 533, 534, 540, 559, 564, 567, 569, 578, 582, 590, 597, 601, 611, 615, 616, 629, 630, 632, 636, 645, 649, 650, 651, 666, 670, 680, 682, 691, 696, 701, 705, 725, 728, 731, 733, 743, 752, 759, 787, 807, 815, 828, 835, 840, 843, 844, 848, 850, 859, 863, 870, 874, 875, 877, 879, 881, 884, 890, 929, 940, 942, 946, 947, 950, 956, 957, 973, 980, 982, 987, 993 |
| seeds | 100, 101, 102, 103, 104 |
| n_seeds | 5 |
| tolerance | studies/line2_particle_sr/results/tolerance.json |
| 容差档映射 | 1e5（每束 N_eff 中位数最近档，`20` [S4]；档位分布 1e5×150） |
| variant | joint |

逐观测量达标明细（13 项，`20` [S4]）：

| index | name | n_pass | n_beams | pass_rate |
|---|---|---|---|---|
| 0 | sigma_xx_rel | 0 | 150 | 0 |
| 1 | sigma_xpxp_rel | 68 | 150 | 0.453333 |
| 2 | sigma_yy_rel | 0 | 150 | 0 |
| 3 | sigma_ypyp_rel | 0 | 150 | 0 |
| 4 | abs_delta_rho_x_xp | 0 | 150 | 0 |
| 5 | abs_delta_rho_x_y | 0 | 150 | 0 |
| 6 | abs_delta_rho_x_yp | 0 | 150 | 0 |
| 7 | abs_delta_rho_xp_y | 18 | 150 | 0.12 |
| 8 | abs_delta_rho_xp_yp | 21 | 150 | 0.14 |
| 9 | abs_delta_rho_y_yp | 0 | 150 | 0 |
| 10 | eps4d_rel | 150 | 150 | 1 |
| 11 | m5_halo_fraction_gap | 0 | 150 | 0 |
| 12 | m6_hf_energy_ratio_gap | 0 | 150 | 0 |

### 5.5 束级 N_eff 快照（门禁产物的中间量）

- **independent**

| beam_id | n_eff_median | n_eff_per_seed | flag_per_seed |
|---|---|---|---|
| 2 | 100000 | 100=99988.2, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 13 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 21 | 100000 | 100=100000, 101=99878.5, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 24 | 100000 | 100=99368.7, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 25 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 41 | 100000 | 100=99292.6, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 42 | 100000 | 100=99714.4, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 46 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 54 | 100000 | 100=99080.4, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 55 | 100000 | 100=98780, 101=99679.5, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 56 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 67 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 70 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 88 | 100000 | 100=98810, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 96 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 105 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 113 | 100000 | 100=99329.3, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 114 | 100000 | 100=99607.3, 101=99433.9, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 134 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 157 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 184 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 186 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 190 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 193 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 197 | 100000 | 100=98939.2, 101=99474.6, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 208 | 100000 | 100=98823.9, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 213 | 100000 | 100=99442, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 219 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 220 | 100000 | 100=99476.6, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 223 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 224 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 235 | 100000 | 100=99061.7, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 241 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 242 | 100000 | 100=99529.8, 101=99846.1, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 244 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 245 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 247 | 100000 | 100=99948.8, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 253 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 264 | 100000 | 100=99772, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 283 | 100000 | 100=99197.6, 101=99383.6, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 285 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 287 | 100000 | 100=98273.8, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 292 | 100000 | 100=99340.1, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 296 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 299 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 303 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 324 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 328 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 343 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 360 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 370 | 100000 | 100=98997.3, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 379 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 387 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 396 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 404 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 406 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 411 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 416 | 100000 | 100=99234.6, 101=99906.4, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 424 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 425 | 100000 | 100=98853.2, 101=99647.1, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 427 | 100000 | 100=98801.3, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 428 | 100000 | 100=99389.4, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 429 | 100000 | 100=99175.7, 101=99681.9, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 430 | 100000 | 100=99673, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 446 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 454 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 459 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 460 | 100000 | 100=99585.7, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 467 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 492 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 493 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 500 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 502 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 503 | 100000 | 100=98752.3, 101=99953.6, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 506 | 100000 | 100=99570.8, 101=99285.7, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 510 | 100000 | 100=98887.8, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 511 | 100000 | 100=99688.1, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 522 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 530 | 100000 | 100=99259.5, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 533 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 534 | 100000 | 100=98728.3, 101=99444.2, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 540 | 100000 | 100=99292.7, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 559 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 564 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 567 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 569 | 100000 | 100=99966.5, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 578 | 100000 | 100=98552, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 582 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 590 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 597 | 100000 | 100=99336.4, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 601 | 100000 | 100=99687.2, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 611 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 615 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 616 | 100000 | 100=99981, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 629 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 630 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 632 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 636 | 100000 | 100=99828.6, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 645 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 649 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 650 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 651 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 666 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 670 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 680 | 100000 | 100=99512.3, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 682 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 691 | 100000 | 100=99939.5, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 696 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 701 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 705 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 725 | 100000 | 100=99562.3, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 728 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 731 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 733 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 743 | 100000 | 100=99499.1, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 752 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 759 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 787 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 807 | 100000 | 100=99266.8, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 815 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 828 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 835 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 840 | 100000 | 100=99936.1, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 843 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 844 | 100000 | 100=99568.2, 101=99939.7, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 848 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 850 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 859 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 863 | 100000 | 100=99415.4, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 870 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 874 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 875 | 100000 | 100=99038.8, 101=99877, 102=100000, 103=100000, 104=100000 | 100=—, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 877 | 100000 | 100=98734.7, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 879 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 881 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 884 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 890 | 100000 | 100=99808.2, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 929 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 940 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 942 | 100000 | 100=99938.3, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 946 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 947 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 950 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 956 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 957 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 973 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 980 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 982 | 100000 | 100=99742.4, 101=100000, 102=100000, 103=100000, 104=100000 | 100=—, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 987 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 993 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |

- **joint**

| beam_id | n_eff_median | n_eff_per_seed | flag_per_seed |
|---|---|---|---|
| 2 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 13 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 21 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 24 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 25 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 41 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 42 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 46 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 54 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 55 | 100000 | 100=100000, 101=97614, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 56 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 67 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 70 | 100000 | 100=100000, 101=99518.9, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 88 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 96 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 105 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 113 | 100000 | 100=100000, 101=98988.2, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 114 | 100000 | 100=100000, 101=97636.9, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 134 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 157 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 184 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 186 | 100000 | 100=100000, 101=99574.1, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 190 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 193 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 197 | 100000 | 100=100000, 101=97978.5, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 208 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 213 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 219 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 220 | 100000 | 100=100000, 101=99889, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 223 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 224 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 235 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 241 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 242 | 100000 | 100=100000, 101=98454.7, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 244 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 245 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 247 | 100000 | 100=100000, 101=99320, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 253 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 264 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 283 | 100000 | 100=100000, 101=97378.2, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 285 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 287 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 292 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 296 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 299 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 303 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 324 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 328 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 343 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 360 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 370 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 379 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 387 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 396 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 404 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 406 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 411 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 416 | 100000 | 100=100000, 101=98892.9, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 424 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 425 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 427 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 428 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 429 | 100000 | 100=100000, 101=98109.8, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 430 | 100000 | 100=100000, 101=98581, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 446 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 454 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 459 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 460 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 467 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 492 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 493 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 500 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 502 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 503 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 506 | 100000 | 100=100000, 101=98375.7, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 510 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 511 | 100000 | 100=100000, 101=99145.8, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 522 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 530 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 533 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 534 | 100000 | 100=100000, 101=97241.5, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 540 | 100000 | 100=100000, 101=99333.7, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 559 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 564 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 567 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 569 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 578 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 582 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 590 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 597 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 601 | 100000 | 100=100000, 101=99887.1, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 611 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 615 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 616 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 629 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 630 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 632 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 636 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 645 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 649 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 650 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 651 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 666 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 670 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 680 | 100000 | 100=100000, 101=98768.9, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 682 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 691 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 696 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 701 | 100000 | 100=100000, 101=99660.9, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 705 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 725 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 728 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 731 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 733 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 743 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 752 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 759 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 787 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 807 | 100000 | 100=100000, 101=99016, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 815 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 828 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 835 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 840 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 843 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 844 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 848 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 850 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 859 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 863 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 870 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 874 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 875 | 100000 | 100=100000, 101=97975.6, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 877 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 879 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 881 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 884 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 890 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 929 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 940 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 942 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 946 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 947 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 950 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 956 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 957 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 973 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 980 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 982 | 100000 | 100=100000, 101=99740.7, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=—, 102=overflow, 103=overflow, 104=overflow |
| 987 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |
| 993 | 100000 | 100=100000, 101=100000, 102=100000, 103=100000, 104=100000 | 100=overflow, 101=overflow, 102=overflow, 103=overflow, 104=overflow |

## 6. 结论范围与后续

- 结论范围限于 `00` [S1] 的推广域声明：束群 = 锚点经预注册抖动线性传输，Phase 1 的方法可用域只在该邻域内成立；真实前端晶格的跨域泛化属 Phase 2。
- 放大比只在可验证域 [1e3, 1e5] 内有定义：N_eff 触顶（上溢）意味着"达到 1e4 工况的 10× 可验证上限"，不意味着更大的真实放大；触顶区间受参照噪声地板限制不可分辨（`20` [S2]）。
- 本报告只陈述门禁的机械判定结果，不预设研究方向（`00` GC6）；H1 不成立时按 `00` [S3] 升级用户重议 Phase 2 方向。
- 后续：Phase 2 入口为 L9 束流接口，进入条件 = Phase 1 交付完成 + 用户放行（`00` [S3]）。

## 7. 设计理由（Why）

### 7.1 为什么三类产出物由一个只读脚本生成，而不挂在门禁脚本里

- 选择：`map_report` 单独成模块，只消费 `gates.py`/`evaluate.py`/`e0.py` 的落盘产物。
- 备选：把图表与报告塞进 `evaluate_gates_from_disk`——被否，门禁的职责是判定（`40` [S3] C1 要求同输入逐位一致的可复算判定），产出物是判定的呈现层；混在一起会让"改报告格式"变成"改门禁"，也让 [S3] C2 的"只消费落盘 summary"边界模糊。
- 取舍依据：`40` Global Constraints 要求产出物脚本只读；分层后门禁可独立复算，报告可反复重生成而不触碰判定。

### 7.2 为什么报告不写生成时间戳与当次 git HEAD

- 选择：报告头只写输入产物的溯源（E0 时间戳、门禁产物里的 code_version）。
- 备选：沿 line1 stage_report 头部写"生成时间 + 当前 commit"——被否，[S4] C1 要求同输入逐位一致，墙钟时间与生成时的 HEAD 会让两次运行不可比对，也无法用文件哈希证明"报告未被手工改过"。
- 取舍依据：可复现性优先于"报告看起来很新"；需要生成时刻时看文件 mtime 即可。

### 7.3 为什么地图分两个面板，而不是共用一条 x 轴的双 y 轴

- 选择：面板 A 画 N_eff/N_low vs N_low（无量纲放大比），面板 B 画裸 M1 vs N（相对 RMSE）+ 参照噪声地板 + 网络 err_median 落点。
- 备选 1：单轴同画两条线——被否，y 量纲不同（比值 vs 相对 RMSE），同轴等于宣布二者可比。备选 2：共享 x 的 twinx 双 y 轴——被否，两线的视觉交点由轴缩放任意决定，读者会把交点读成"网络达到裸曲线水平"这类无依据结论。
- 取舍依据：[S4] 只要求地图含主曲线、E0 参考线与噪声地板线，未要求同轴；分面板后 [S4] C1 的三条要素都在，且噪声地板画在它被定义的 M1 面板内（`20` [S2] C3 的量纲）。面板 B 叠加网络 err_median 是为了让 E0 参考线真正起"参照"作用——否则它与本批 run 无视觉关联。

### 7.4 为什么主曲线画比值 N_eff/N_low，且上/下溢用域端点显式标注

- 选择：y = N_eff/N_low（对数），三档 N_low ∈ {3e3, 1e4, 1e5} 直接可比；带标记的行画在域端点（1e3 或 1e5 对应的比值）上并用空心叉号区分。
- 备选：y = N_eff 绝对值——被否，3e3 档的 N_eff = 1e5 与 1e4 档的 N_eff = 1e5语义强度不同（前者 33×、后者 10×），绝对值会把"可用域"读平。
- 取舍依据：`20` [S3] C2 规定 N_eff 要么落在 [1e3, 1e5] 内、要么携带显式标记，端点值是截断而非实测；不标注就会把域顶读成实测放大（`00` [S1]"可验证放大上限 = 10×"的语义边界）。

### 7.5 为什么 N_eff 表按 run 级聚合，束级明细留在门禁产物里

- 选择：CSV 一行 = 一个 (run, 站, N_low) 的 test 评估，含上/下溢标记。
- 备选：一行 = 一个 (run, 束)——被否，150 束 × 20 run 会让 [S4] 要求的"runs × 站 × N_low 明细表"淹没在束级行里；束级中位数与逐种子值已由门禁产物`beam_n_eff_snapshot` 落盘，报告第 5.5 节原样引用，不做第二份实现。
- 取舍依据：单一实现点（`20` Global Constraints 的"禁止旁路实现"同理适用于产物层）。

### 7.6 为什么顺序审计（`40` [S1] C5）在证据缺失时判"不可验证"而不判通过

- 选择：证据齐全且存在 run 的 started_at ≤ E0 completed_at → 抛错；证据缺失 → verifiable=False，第 4 章显式登记。
- 备选：证据缺失时默认通过——被否，那等于把"没记录"读成"守规矩"，与 `40` D8"顺序约束不编码就会在执行中被跳过"的立意相反。
- 取舍依据：与门禁的"不可判定"一等地位一致（`40` D4）：缺证据是一种结论，不是通过。违例则必须中断——尺子后行的批次其 N_eff 无定义基底，继续生成报告等于把无效矩阵包装成交付物。

### 7.7 为什么图内标签用英文而报告正文用中文

- 选择：`figure_avail_domain.png` 的标题、轴标签与图例用 ASCII 英文；报告正文、CSV 列名与代码 docstring 用中文。
- 备选：图内也用中文——被否，matplotlib 的中文字形依赖运行机上装了什么 CJK 字体，缺字体时字形降级为方框、且不同字体下渲染出的像素不同，直接破坏 [S4] C1 的"同输入逐位一致"（同一份产物在两台机器上哈希不同）。
- 取舍依据：与 line1 出图惯例一致（`src/evaluation/plots.py` 的图内标签同为英文），并让 PNG 只依赖 matplotlib 自带的 DejaVu Sans。

