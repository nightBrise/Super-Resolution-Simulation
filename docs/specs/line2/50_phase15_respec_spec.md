# Phase 1.5 协议修正与重训（测量修复 + v2 公共定标 + κ 重参数化 + 真迁移臂）Spec

> 研究线：line2_particle_sr；批次：Phase 1.5（Phase 1 图像接口的预注册修正重训）。
> 上游证据：`40` D21（T0 五实验）与 `40` D22（前置三实验）；证据链
> `studies/line2_particle_sr/reports/t0_diagnostics/{T0_SUMMARY,PREREQ_SUMMARY}.md`。
> 需求基线：`50_phase15_respec_spec.baseline.md`（用户目标逐字引用）。
> 版本：v0.2.3（Proposed，2026-10-13；v0.1 一审清零 → v0.2 二审清零 → v0.2.2 三审 approve + Minor 打磨 → v0.2.3 边界显式化（D14，用户确认阶梯式范围）；99 登记 v1.1）。
> 符号沿用 `00` [S1]；本 spec 新增符号在 [S1] 定义。

## [S1] 新符号、统计口径与 00 增补

**Review tier:** contract

Phase 1.5 新增/修订的术语与统计口径，及对 `00` 的四处最小增补（随本批次 99 v1.1 批准一并生效）。

- **B-const（零信息常数解，两站）**：站 s ∈ {0,1} 的 700 个 train 束、该站、全量（2×10⁵）目标图在 **v2 公共边**下逐投影归一的均值图；不含任何 test/束个体信息；`20` [S5] 冻结基线族新增成员（落实 `40` D18）。
- **公共域（common domain）**：某 (轴, 站) 的 129 个 bin 边界，由 **train 700 束**该站全量图的池化 min/max 外扩 5% 导出，同站同轴所有束共享；与 v1 的逐束自定标（min/max per beam）相对。每站 4 轴 × 129 边界，该站 6 个投影共享其轴边界。
- **参考尺度 s_i**：坐标 i 的公共域半宽（(max−min)/2），i ∈ {x, x′, y, y′}；κ 无量纲化的分母，可由 bin_edges 推得。
- **win_rate / advantage / CI**：配对逐束统计。对参照 X（B-const 或基线族成员）：逐束差 = err_X(束) − err_model(束)；win_rate = 差 > 0 的束占比（test 150 束）；advantage = 差的中位数；CI = 束级 paired bootstrap 的 95% 区间，**10,000 次重采样，SeedSequence(20260825)**。**聚合口径（G1/G2/G4/G5 统一适用）**：逐 (束, 子样本) 读数取中位数 → 逐束对 5 种子取中位数 → 对所得 150 个束级读数做束级统计；本条 CI 定义同样适用于 cos_true（G2），B-const 无种子维（其束级读数即逐束值）。
- **cos_al / drop_frac / disp_ratio**：前置③定义的增量对齐仪器。d_out = out − prior、d_tgt = target − prior（prior = 该站 B-const；M1 同款归一，六投影展平）；cos_true = med cos(d_out, d_tgt)，cos_scr = 输入按固定 derangement 打乱后的同款中位（SeedSequence(20260825)）；drop_frac = (cos_true − cos_scr)/cos_true；disp_ratio = med ‖out_scr − out_true‖ / med ‖out_true − prior‖。打乱所用输入对含该束的 κ（由打乱后输入派生）。
- **R_diff（难度归一残差比）**：`(err_s1 − B_s1)/(err_s0 − B_s0)`，B_s = 站 s 的 B-const 误差；式中 err_s·/B_s· 均取对应 150 个束级读数（[S1] 聚合口径，B-const 无种子维）的**中位数**后代入；原始比 `err_s1/err_s0` 一并披露、不进判定。
- **Phase 1.5 晶格边界**：本批全部数据恰用**一条冻结 FODO**（n_cells = 2、μ = 75°/cell、漂移 2.5 m——`10` [S3]；四极长 0.1 m——Phase 1 实现 `transport.py`）与两个观测站（站 0 入口 / 站 1 出口）；元件集 = 四极 + 漂移，**不含二极磁铁、六极磁铁及其他非线性元件，不含加速腔**。束间差异仅来自入射抖动（`10` [S3]：四极 ±5%、踢角 ±10 µrad、漂移 495–505 mm）——**晶格传递矩阵本身不随束变化（抖动 ≠ 换晶格）**。G4 的"迁移"限于该晶格的站 0 → 站 1；任何跨晶格泛化声明属 Phase 2 晶格随机化（见 Out of Scope）。

### Claims

- C1: `00` [S4] 的 spec 索引 SHALL 增加行：`50_phase15_respec_spec.md（Phase 1.5 协议修正与重训批次）`。验证 = 冻结时检查 `00` [S4] 文本。
- C2: `00` [S3] SHALL 增加一行：`Phase 1.5 出口 = 50 [S5] 的 G1–G5 判定（脚本化，枚举四值）`。验证 = 冻结时检查 `00` [S3] 文本。
- C3: `00` [S4] C8 的枚举 SHALL 由"00/10/20/30/40 五份"改为"00/10/20/30/40/50 六份"。验证 = 冻结时检查 `00` [S4] C8 文本。
- C4: `00` GC2 SHALL 增注：数据/训练随机性经 SeedSequence(20260913) 派生；**分析/仪器/打乱/合成随机性根种子 = 20260825**（SeedSequence 派生）。验证 = 冻结时检查 `00` GC2 文本。
- C5: 本 spec 的术语（B-const、公共域、s_i、win_rate/advantage/CI、cos_al/drop_frac/disp_ratio、R_diff、Phase 1.5 晶格边界）SHALL 为 line2 语料的唯一裁定来源；`scripts/check_spec_consistency.py` 重跑 SHALL exit 0 且 `spec_claim_index.md` 含 50 的 Claim 条目（v1.1 交付范围 = 50 文件本身与上述 00 四处增补；`10`/`20`/`30`/`40` 被取代条文的文字改写按 99 v1.1 取代清单在批准生效时逐条落地，见 99 登记表）。测试：`tests/unit/test_line2_spec_terms.py::test_spec_terms_registered`（对 check 脚本输出断言）。
- C6: `data/line2_ds_v2/manifest.json` SHALL 含 edge_rule 版本串、8 组轴边界数值（4 轴 × 2 站）及其 SHA256、各投影的轴配对元数据。测试：`test_line2_v2_dataset.py::test_manifest_edge_rule`。

## [S2] v2 数据集：公共物理域定标（解禁 θ 踢角通道）

**Review tier:** contract

v1 的缺陷（`40` D22/前置①）：bin_edges 逐束自定标（每束每站 min/max），纯平移类抖动（θ 踢角）被精确对齐删除（单旋钮 M1 = 0.0000）。v2 只改边界、不改粒子：粒子表由锚点 + 同一 SeedSequence(20260913) 种子链重生成，与 v1 逐位一致；划分（700/150/150）、五档 × k=20 无放回子样本（A ⊂ B）、FODO（n_cells=2、μ=75°）全部沿用 `10` [S2]–[S4] 不变。公共域**只由 train 700 束导出**（防泄露，见 Global Constraints）；对可能落在域外的 test 粒子实行域外夹取会计（C6）。

**Dependencies (optional):** 无

### Claims

- C1: WHEN 构建 `data/line2_ds_v2/` THEN 粒子内容 SHALL 与 v1 同种子推导逐位一致：全部 1000 束 × 2 站重传对账（pre1 同法），计数最大绝对差 = 0；唯一差异为边层规则（公共域定标及由此产生的域外夹取，见 C6）。测试：`tests/unit/test_line2_v2_dataset.py::test_v2_particles_match_v1_by_seed`。
- C2: 同站同轴的 bin_edges SHALL 跨全部 1000 束逐位相同，且由 [S1] 公共域规则（train 700 束池化 + 5% 外扩）确定性导出（重跑构建两次，edges SHA256 一致）。测试：`test_line2_v2_dataset.py::test_edges_common_across_beams`。
- C3: WHEN 对 v2 施加单旋钮 θ（踢角）扫掠（±10 µrad，其余旋钮为零）THEN 该束目标图组对基准束的 M1 SHALL ≥ 1×10⁻³。阈值依据：v1 逐束自定标下该旋钮 M1 = 0.0000；pre1 公共域口径实测 0.74–0.81——1×10⁻³ 为保守判"可见"下限（裕度 > 2 个数量级），非实测地板。测试：`test_line2_v2_dataset.py::test_theta_knob_visible_in_v2`（回归钉）。
- C4: v2 的 θ（踢角）通道外的另两旋钮 δ（四极）与 ℓ（漂移）、以及站间传输行为 SHALL 与 v1 逐位一致（同一抖动分布与传输代码路径）。测试：`test_line2_v2_dataset.py::test_v2_transport_matches_v1_non_theta`。
- C5: manifest.json SHALL 含 data_version="line2_ds_v2"、code_version（完整 40 位 git HEAD）、master_seed=20260913、锚点 MD5（= `10` C1 值）。测试：`test_line2_v2_dataset.py::test_dataset_v2_manifest`。
- C6: WHEN 任意束的任一粒子落在其 (轴, 站) 公共域之外 THEN 构建 SHALL 将其**夹取至最近边界格**（计数守恒：∑counts = 该子样本粒子数不变，沿用 `00` GC5；不做丢弃），逐束记录夹取计数，且全数据集夹取粒子占比 SHALL ≤ 0.1%（分母 = 全数据集粒子总数，即各束各站全量表行数之和），违例构建失败。阈值依据：train 池化 + 5% 外扩的域裕度大于实测束间窗口差（约 2 倍；窗口差实测见 T0-3 `exp3_report.md`（x/y 轴 <0.05%、xp/yp 轴 ±1–2.6%）与 `pre1_report.md` §4.1 同现象的另一统计表述）。测试：`test_line2_v2_dataset.py::test_clamp_accounting`。

## [S3] 指标层：准绳重排、容差线重造、基线族补全、个体信息仪器

**Review tier:** contract

**Dependencies (optional):** [S2]

主误差准绳不变（M1 group_rel_rmse，总计数归一口径沿用 `20` [S1]）。变更四件：判定输入白名单重排、容差线在 v2 重造（含 M6 去偏）、B-const（两站）入基线族、个体信息仪器函数化。**N_eff 保留计算与落盘，但降级为叙事尺度**：不得作为选型、路由或任何出口门禁的判定输入。

### Claims

- C1: 判定输入 SHALL 按类别白名单化：**选型与路由**只消费 val 段 val_nll（[S3] C2 定义）；**出口门禁 G1–G5** 只消费 test 段落盘读数（逐束 M1 err、13 维 obs、对齐仪器读数、B-const 逐束误差）；两段不交叉（沿用 `40` D6/D12 两段式）；N_eff 在两类判定中 SHALL 均不出现、仅作披露字段。测试：`tests/unit/test_line2_metrics.py::test_neff_not_in_gate_inputs`（对判定函数输入做静态断言，覆盖选型/路由/门禁三类）。
- C2: 选型读数 val_nll SHALL 定义为：逐 (束, 子样本) NLL → 每档对全束取中位数 → 三档等权算术平均；档集 {3×10³, 1×10⁴, 1×10⁵} × 150 val 束 × subs 0–4（站 0）；落盘 SHALL 附各档分项。测试：`test_line2_metrics.py::test_val_nll_three_tier_mean`。
- C3: v2 容差线表 SHALL 按 `20` [S4] 的"逐观测量 95 分位"口径在 v2 重造（13 项 × 3 训练档 × 站 0），其中 M6 行采用去偏口径：**以一半 train 子样本估计偏差中位数并去除，在另一半上验证**——验证判据 = 验证半区去偏后 |中位|/σ < 1 且 95 分位 / (1.96σ) ∈ [0.8, 1.2]（可证伪；v1 实测未去偏 |中位|/σ = 24.7）。测试：`tests/unit/test_line2_metric_reconciliation.py::test_m6_tolerance_debiased_is_noise_line`。
- C4: B-const（两站）SHALL 入冻结基线族：站 0 与站 1 各自以 v2 公共域重算（各站 700 train 束均值图），并落盘两站的逐束误差向量（各 150 test 束），作为 G1/G4/G5 与对照表的判定参照。测试：`test_line2_metrics.py::test_bconst_in_baseline_family`。
- C5: 个体信息仪器 SHALL 函数化（输入 = 预测图组集合 + prior + 束表，输出 = cos_true / cos_scrambled / drop_frac / disp_ratio，种子 20260825），合成用例上 SHALL 满足：偏差方向已知的构造数据 cos_true ≥ 0.999、打乱后 |cos_scr| ≤ 0.1。测试：`tests/unit/test_line2_individual_info.py::test_alignment_instrument_synthetic`。

## [S4] 模型与训练：κ 重参数化、混合档、训满预算

**Review tier:** contract

**Dependencies (optional):** [S2], [S3]

两臂拓扑沿用 `30` [S1]（joint FiLM U-Net + P_PSD；independent 独立六网），损失沿用 Poisson NLL（rate = p·2×10⁵）。变更三件：

1. **κ v2（无量纲）**：joint 臂 = (Σ_ij/(s_i·s_j) 的 10 个唯一分量——对角 4 + 上三角 6，沿用 `30` [S1] 顺序, log₁₀(6·N_low))；independent 臂 = (var_i/s_i², var_j/s_j², cov_ij/(s_i·s_j), **log₁₀(6·N_low)**)——log 维两臂同口径（沿用 v1 实现语义 `counts.sum = 6·N_low`，`30` D14/D20）。s_i 为 [S1] 参考尺度。
2. **混合档训练**：每个 run 的训练/验证集同时含 {3×10³, 1×10⁴, 1×10⁵} 三档（train 700 束，站 0；**每档每束 k_train = 5 个子样本（subs 0–4）**，本批预注册），每 step 按档均匀随机抽批（SeedSequence(20260913) 派生、与子样本流独立 spawn，禁裸随机）；log₁₀N 维随 batch 变化。
3. **训满预算 + 单一选型准绳**：max_steps = 50,000；早停与选型唯一准绳 = [S3] C2 的 val_nll（patience = 10）；`should_update_best` 用严格 `<`；站 1 的任何粒子不进训练/验证（真迁移前提）。

### Claims

- C1: WHEN 任意 v2 数据批次计算 κ THEN 全部分量 SHALL 为无量纲量且 |κ_i| ∈ [10⁻³, 10³]（跨全集断言；v1 实测前 10 维 5.6×10⁻¹¹…1.2×10⁻⁷，本断言即回归钉）。测试：`tests/unit/test_line2_model.py::test_kappa_dimensionless`。
- C2: WHEN 混合档训练集整体统计 THEN log₁₀N 维 SHALL 非常数（std > 0），且三档取值 4.255273/4.778151/5.778151（= log₁₀(6·N_low)）各出现。测试：`test_line2_model.py::test_kappa_logn_informative_mixed_tiers`。
- C3: run 配置断言 SHALL 自动校验：两臂种子集合逐元素相同 {100..104}、参数量差 ≤30%、训练档集 = {3×10³,1×10⁴,1×10⁵}、训练站集 = [0]、k_train = 5、每批含站 1 粒子数 = 0。测试：`tests/unit/test_line2_experiments.py::test_phase15_matrix_config_assertions`。
- C4: WHEN 训练结束 THEN summary SHALL 记录 stop_reason ∈ {val_nll_patience, budget_exhausted}、best_step（val_nll argmin）、best_val_nll、各档 val_nll 分项；选型分支 SHALL 不存在 N_eff 比较。测试：`test_line2_experiments.py::test_selection_val_nll_only`。

## [S5] 实验矩阵、评估与出口门禁 G1–G5

**Review tier:** contract

**Dependencies (optional):** [S2], [S3], [S4]

矩阵：**2 臂 × 5 种子 = 10 runs**（`e15_joint_s0..s4`、`e15_indep_s0..s4`，种子 {100..104}），混合档、stations=(0,)。E0' 先行：v2 裸误差–N 曲线（五档、口径沿用 `40` D11：300 束 val+test × 两站）+ 容差线表 + B-const（两站）全部在训练启动前落盘，编排器强制时间戳校验（沿用 `40` [S1] C5 纪律）。评估：每 run × test × {3×10³,1×10⁴,1×10⁵} × 站 {0,1} × subs 0–19（站 1 = 纯迁移评估，训练未见；test 束与 v1 相同）。路由：获胜臂 = 两臂按站 0 @1×10⁴ 的 val 段 val_nll 排序取优（**整体取代** `40` [S3] 的 5/5–4/5–3/5–≤2/5 计数分支，见 99 取代清单；test 评估调度自 Phase 1.5 起归本节）。暗桶 M2 单独报告保留（`40` [S3] 条款沿用）。**G2 的评估 ckpt = 选型 ckpt**（val_nll argmin 的 best_val.ckpt；全 spec 不再使用"last 口径"作为门禁对象）。**G1/G2/G4/G5 的聚合口径 = [S1] 统一定义**（逐 (束,子样本) 中位 → 逐束 5 种子中位 → 150 束统计；CI 同口径）。

### Claims

- C1: WHEN 编排器启动任一训练 THEN v2 E0' 三件产物（curve/tolerance/bconst 两站）的落盘时间戳 SHALL 早于该 run 启动时间戳，违例拒绝启动。测试：`test_line2_experiments.py::test_e0_prime_before_training`。
- C2: **G1 密度反超先验**（获胜臂 @1×10⁴ 站 0，参照 = B-const 站 0）：成立 = win_rate ≥ 0.90 且 advantage 的 95%CI 下界 > 0；方向成立 = CI 下界 > 0 且 win_rate < 0.90；不成立 = CI 上界 ≤ 0；不可判定 = CI 跨 0（CI_lo ≤ 0 < CI_hi）或评估缺失。测试：`test_line2_experiments.py::test_gate_g1`（含正/反例构造）。
- C3: **G2 个体信息**（获胜臂 5 种子，选型 ckpt，@1×10⁴ 站 0，prior = B-const 站 0）：记 k_a = 满足 [cos_true 的 CI > 0（正）] 的种子数、k_and = 满足 [（cos_true 的 CI > 0（正））**且**（drop_frac ≥ 0.80）] 的种子数。成立 = k_a ≥ 4 且 k_and ≥ 4；方向成立 = k_a ≥ 4 且 k_and = 3；不可判定 = (k_a = 3) 或 (k_a ≥ 4 且 k_and ≤ 2)；不成立 = k_a ≤ 2。四类互斥完备。测试：`test_line2_experiments.py::test_gate_g2`。
- C4: **G3 观测量达标**（获胜臂 @1×10⁴ 站 0，v2 容差线、M6 去偏，官方聚合 = 逐观测量先对 5 种子取中位数再判达标）：成立 = 达标率 ≥ 0.80；方向成立 = 0.50 < 达标率 < 0.80；不成立 = 达标率 ≤ 0.50；不可判定 = 容差线/评估缺失。测试：`test_line2_experiments.py::test_gate_g3`。
- C5: **G4 跨站真迁移**（获胜臂站 1 @1×10⁴；限于 [S1] 晶格边界的站 0 → 站 1，不含任何跨晶格声明）：成立 = R_diff ≤ 1.10；方向成立 = 1.10 < R_diff ≤ 1.25；不成立 = R_diff > 1.25；不可判定 = 站 1 评估或 B_s1 缺失。原始比 err_s1/err_s0 SHALL 并列落盘披露。测试：`test_line2_experiments.py::test_gate_g4`。
- C6: **G5 N 泛化**（获胜臂 @3×10³ 与 @1×10⁵ 站 0，参照 = B-const 站 0——同一均值图两档共用）：两档分别按 G1 同款四值判定；门禁组合规则 = ①任一档"不成立" → 不成立；②否则任一档"不可判定" → 不可判定；③两档均"成立" → 成立；④其余组合 → 方向成立。两档结论并列落盘。测试：`test_line2_experiments.py::test_gate_g5`。
- C7: 五条门禁 SHALL 由脚本消费落盘产物判定（无人工干预），`e15_gates.json` SHALL 含五门禁结论 + 每条输入快照 + routing（获胜臂、依据 val_nll）+ 暗桶 M2 分项。测试：`test_line2_experiments.py::test_e15_gates_payload`。
- C8: 全部 10 run × 6 评估（3 档 × 2 站）的 `eval_*.json` SHALL 落盘且字段与 v1 schema 相容（`per_beam[].{beam_id, err, obs, m2_buckets}`）并带站标记。测试：`test_line2_evaluate.py::test_phase15_eval_schema_station1`。
- C9: `map_report` SHALL 适配 v2 路径并渲染 `e15_gates.json`（G1–G5、B-const、暗桶）。测试：`test_line2_experiments.py::test_map_report_v2`。

## [S6] 交付：阶段报告与前后对照

**Review tier:** mechanical

**Dependencies (optional):** [S5]

交付 `studies/line2_particle_sr/reports/line2_phase15_stage_report.md`（结构沿用 Phase 1 阶段报告：数据/设置/结果/门禁/Why/局限/复现），`progress.md` 更新。

### Claims

- C1: 阶段报告 SHALL 含"前后对照表"：每行一个量（Phase 1 官方 H1/H3/H8 口径、G1–G5、三点 err、模型 vs B-const），三列 = Phase 1 官方（best_val）/ Phase 1 last 口径（pre2）/ Phase 1.5；数字 SHALL 由 `studies/line2_particle_sr/reports/tools/verify_report_numbers.py`（v2 适配版）溯源，exit 0。
- C2: 交付物存在性 SHALL 可由文件枚举脚本验证（阶段报告、e15_gates.json、10×6 eval JSON、对照表 CSV）。测试：`test_line2_experiments.py::test_phase15_deliverables_exist`。

## Global Constraints

- **冻结预注册**：本 spec 冻结后，门禁阈值（G1–G5）、统计口径（[S1] bootstrap 次数/种子）、种子、档位、矩阵、选型准绳 SHALL 不得修改；实现期歧义/偏差一律入 `50_phase15_respec_spec.decisions.md`，不改 spec 正文。
- **尺子先行**：v2 E0'（曲线 + 容差线 + B-const 两站）完成时间戳 SHALL 早于任一训练启动（编排器强制，违例拒绝启动）。
- **环境与确定性**：全部命令在 `sr-sim` 环境运行并先过 `scripts/check_env.py`；训练/评估启用 Phase 1 同款确定性口径（`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`PYTHONHASHSEED=0`，exec_mode 逐 run 留痕）。
- **种子纪律**：数据/训练 = SeedSequence(20260913)（训练种子 {100..104}）；分析/仪器/打乱/合成 = SeedSequence(20260825)；禁止裸随机调用（00 GC2 增注随本批生效）。
- **防泄露**：站 1 粒子与 test 束不得进入任何训练/验证/统计（含公共域导出——[S2] 已限定 train 700 束）；同一 (束, 站, 档) 的子样本不跨划分；κ/归一化的统计量只来自 train。
- **版本三元组**：每 run 的 config SHALL 含 code_version（完整 40 位 git HEAD）/data_version="line2_ds_v2"/spec_version（99 v1.1 批准号）。
- **代码复用与语义不变**：M1、暗桶 M2、六投影枚举、损失、两臂拓扑沿用 v1 语义；仅本 spec 声明的变更（边、κ、判定输入白名单、矩阵）允许触及既有行为。对冻结语料的取代以 99 v1.1 取代清单为准（逐条登记：`00` GC3、`00` [S3] C6、`30` [S3] 与 C2、`40` [S3] 路由、`20` [S4] 容差映射（含 `40` D12）、`20` [S5] 基线族、`10` [S4] **C1** 分箱范围/零截断/报错规则）；文字改写在批准生效时逐条落地，`50` 实现期冻结纪律即刻生效。
- **每 `[Sn]` 派 reviewer**：证据 = 测试名 / 命令输出 / file:line；prose 不是证据。

## Out of Scope

- 粒子生成与桥接评估（密度 → 粒子云的数值采样及其门禁）——Phase 2 桥接单独立项；
- 晶格随机化 / 域泛化（"任何晶格、任何位置"的第 4 级阶梯）——Phase 2 正题；具体排除：**任意数量与组合的元件（含二极磁铁、六极磁铁等非线性元件）与加速能量变化**。Phase 1.5 的全部训练恰在站 0、评估在站 0/1（[S1] 晶格边界：一条固定 FODO 的两站）上进行，不携带任何跨晶格泛化声明；
- 纵向 (z, δ) 与 6D/15 投影扩展；
- 分辨率阶梯（256² 及以上）与 N_low/N_high 扩容（第四级配 10⁸ 参照，另行预注册）；
- 生成式模型（扩散/流/点云网络）、DDP 双卡路径（`30` D12 遗留，不在本批）；
- 混合档 vs 单档专用模型的对照实验（部署形态决策，Phase 2 候选 deferred 项）；
- 包络重匹配（β≈23 m → 8.5 m）方案（前置①已判非必需，保留 FODO 失配构型）；
- 42–53× 外推作为宣传口径——仅限报告披露栏。
