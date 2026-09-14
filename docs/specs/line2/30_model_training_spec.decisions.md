# Decision Log: Phase 1 图像接口放大网络（模型与训练）

<!-- ACTIVE LOG — continuously appended throughout lifecycle; do not treat as stale -->

## D1: 骨干 = 残差 U-Net 起步
**Phase:** write-spec
**Chosen:** 承 line1 `src/models/unet.py` 层级结构改 6 进 6 出；transformer/diffusion 不入 Phase 1
**Rationale:** 任务是 128² image-to-image，U-Net 形状天然匹配；line1 有全套实现与训练设施可复用，风险最低。
**Rejected:** Transformer/attention 主干 ——128² 分辨率下收益不明且引入全新训练风险；diffusion ——属 H5 生成式选项，Phase 2 评估后再议。
**Spec section:** [S1]
**Status:** live

## D2: 损失 = Poisson NLL（micro 平均）
**Phase:** write-spec
**Chosen:** 期望计数语义 + lgamma 完整 NLL + micro 平均
**Rationale:** 目标是泊松计数，NLL 是其正确负对数似然；micro 平均对亮暗格等权（分桶诊断由 `20` M2 承担）。
**Rejected:** MSE/L1 ——忽略泊松异方差（暗格欠权、亮格过权），且无计数语义。
**Spec section:** [S2]
**Status:** live

## D3: 输出激活 = Softplus + 逐投影归一
**Phase:** write-spec
**Chosen:** Softplus 保证严格正，再逐投影除以自身和得到概率图
**Rationale:** 概率语义（v0.2，B2 联签）需要逐投影和为 1 的输出；Softplus 全域有梯度，无死区。
**Rejected:** exp ——数值溢出风险；ReLU ——负区零梯度且零概率使 NLL 爆炸；直接输出计数率不做归一 ——与 `20` 的形状比较语义脱钩（B2 审核发现的根源之一）。
**Spec section:** [S1] C1、[S2]
**Status:** live

## D4: 独立版 = 6 个互不共享权重的小模型，总参数量与联合版对齐
**Phase:** write-spec
**Chosen:** 独立版 6×(C_ind 基通道 U-Net)，C_ind = ceil(C/√6) 实测校准，参数总量差 ≤ 30%
**Rationale:** H3 的对照变量必须只有"跨投影信息是否流动"一项；共享骨干多头会让"联合"与"独立"的边界模糊。
**Rejected:** 共享骨干 + 6 头 ——这是弱联合而非独立，H3 对照失效。
**Spec section:** [S4] C1
**Status:** live

## D5: 种子执行计划 = 探索期 1 seed、终选 ≥5 seeds
**Phase:** write-spec
**Chosen:** 调参探索期每配置 1 seed（val N_eff 筛选），最终配置与主对比 ≥5 seeds 报中位数与极差（2026-09-14 用户升级自 ≥2）
**Rationale:** 用户要求种子 ≥5；探索期全程 5 seeds 成本 ×5 而信息增量集中在终选阶段。
**Rejected:** 全程每配置 5 seeds ——搜索空间约 36 配置 × 5 = 180 次训练，成本不可辩护。
**Spec section:** [S3]
**Status:** live

## D6: 早停与选型准绳 = val N_eff（非 val loss）
**Phase:** write-spec
**Chosen:** 早停 patience 以 val N_eff 无改善计
**Rationale:** 主指标是 N_eff（00 GC3）；val loss 收敛不等于放大等效性提升，两者可能背离。
**Rejected:** val loss 早停 ——与主指标脱钩，会选出"损失低但放大差"的 checkpoint。
**Spec section:** [S3]
**Status:** live

## D7: 输出语义 = 概率头 × N_high（审核 B2 训练侧联动，2026-09-14）
**Phase:** review
**Chosen:** 网络输出逐投影概率图（和为 1），率 = p·N_high；N_high = 2e5 为数据集常量
**Rationale:** 与 `20` M1 的形状比较语义闭合（对网络输出归一为恒等）；把"放大"编码为形状恢复，总计数由数据集常量承载，网络不需知道也不该学一个尺度因子。
**Rejected:** 输出 Σμ = N_low（v0.1 语义）——与 2e5 参照直比产生 B2 的量纲断裂，H1 不可达；输出 Σμ = 可学习尺度 ——尺度成为可调旋钮，违背放大语义。
**Spec section:** [S1] C1、[S2]、[S5] C3
**Status:** live

## D8: GPU 确定性条款（审核 M6 修复，2026-09-14）
**Phase:** review
**Chosen:** 强制 deterministic algorithms + cudnn deterministic；不可确定算子存在时放宽为 val N_eff 轨迹 < 0.5% 容差一致并记录执行模式
**Rationale:** 逐位重训的可检验性取决于算子确定性；无差别要求逐位在 GPU 上可能不可满足，容差 + 模式记录是诚实且可审计的退路。
**Rejected:** 只写种子派生不写执行模式 ——reviewer 指出这在 GPU 训练语境下是空承诺。
**Spec section:** [S3] C1
**Status:** live

## D9: max_steps = 50000 预注册（审核 M9 修复，2026-09-14）
**Phase:** review
**Chosen:** max_steps 作为 config 必填项固定 50000，GC4 的"相同步数预算"以此为准绳
**Rationale:** 审核指出 GC4 引用了未注册的预算值——公平性要件不能悬空。
**Rejected:** 各 run 自由设步数 ——公平性无从审计。
**Spec section:** [S3]
**Status:** live

## D10: F8 条件化落地 = 自归一输入 + 粗矩 FiLM（审核 M7 修复，2026-09-14）
**Phase:** review
**Chosen:** 输入自归一为形状图；条件向量 κ（10 维输入矩摘要 + log N_low）经 FiLM 注入；独立版条件缩为该投影 3 维摘要
**Rationale:** 大纲 F8（用户默认项）此前在契约中静默缺位，而 Phase 1 的跨站检验（H7）正依赖网络对站间统计差异的鲁棒性；显式条件化是 line1 方案 C 参数先验思想的直接复用。防泄露由 C5 测试钉死（κ 只由输入派生）。
**Rejected:** 声明 F8 缓期至 Phase 2 ——跨站泛化在 Phase 1 就被检验，条件化不能缺位；无条件化裸训 ——站间束宽/相关差异会退化为网络负担。
**Spec section:** [S1]、[S4]
**Status:** live

## D11: 独立臂参数量标定冻结 + E1 配置纠正
**Phase:** review
**Chosen:** 独立臂 = `IndependentEnsemble(base=16, num_levels=4, blocks=3)`（对应 E1 config 的 `ind_base=16`、`num_levels=4`、`ind_blocks=3`），实测 10,092,870 参数 vs 联合臂（`base=48, num_levels=4, blocks=2`）10,708,710 = **−5.8%**，满足 `30` [S4] C1 的 ±30%。属 `00` [S14.2]「标定采用值登记」预授权类。
**Rationale:** `30` [S4] 要求两臂参数量差 ≤ 30%（H3 的公平性前提：差异只能在"跨投影信息是否流动"）。`ceil(C/√6) = 20` 与冻结主干的 `GroupNorm(min(8, ch))` 的 8 整除约束冲突（构造即崩），故取 16，并在同深度、同 blocks 数下用 blocks 与深度对齐参数量。
**偏差与纠正:** E1 的 5 份 indep config 原先写 `num_levels=3`，实测独立臂 2,476,230（**−76.9%**）——破坏 H3 公平性前提。2026-09-14 已把 5 份 config 的 `num_levels` 改为 4（`ind_base`/`ind_blocks` 不变），回到标定值。此改动发生在 E1 首次有效训练启动之前，无已采信结果需要废弃。
**Rejected:** `base=24, num_levels=4, blocks=1`（−11.6%，深度贡献不足）与 `base=32, num_levels=3, blocks=3`（−7.9%，深度弱于联合臂）。
**Spec section:** [S4] C1
**Status:** live

## D12: Phase 1 训练为单卡路径，DDP 双卡冒烟缓期（偏差登记，待验收闸口裁决）
**Phase:** review
**Chosen:** `30` [S3] C4（DDP 双卡与单卡 fallback 两条路径都能完成训练）在 Phase 1 只满足单卡路径；双卡冒烟缓期。代码 `src/line2/training/train.py` 头部 docstring 已声明该偏差，本条目为其正式登记（此前只在 Progress 表以文字出现，日志缺条目）；该 docstring 与 Progress 表 [S3] 行同引本条目编号 D12。
**Rationale:** E1/E2 共 10–20 个 run 用 2 张卡各跑一条队列即可排完（单 run 实测 joint 0.255 s/step、independent 0.508 s/step，早停后约 1–2 h/run）；DDP 会改变浮点归约路径，引入与"同种子逐位可复现"（[S3] C1）的额外交互，收益低于风险。
**Rejected:** Phase 1 硬上 DDP ——跨模式浮点不一致需额外容差条款，且 2 卡吞吐已够。
**Spec section:** [S3] C4
**Status:** live（待验收闸口确认）

## D13: κ 未做尺度归一 ⇒ Σ 分量对 FiLM 数值惰性、软一致罚项在生产档实质失效（偏差登记，待验收闸口裁决）
**Phase:** review
**Chosen:** **不改代码，登记为偏差。** `30` [S1]/[S2] 的字面要求（κ = 10 维 Σ 摘要 + log₁₀N_low；P_PSD = max(0, −λ_min(Σ̂))²；FiLM 注入）已按 spec 实现；缺陷在量纲尺度，改动 κ 的参数化等于改预注册的处理，需 spec 级决策。
**Rationale:** 实测证据（真实数据集 val 束）：κ 前 10 维为**物理单位²**的 Σ 分量，量级 4.9e-11 … 1.2e-7；第 11 维 log₁₀N_low ≈ 6.08。缩配网络上：Σ 分量整体 ×100 → 输出 max|Δ| ≈ 1.9e-9（相对 3.2e-5）；log₁₀N_low +1 → max|Δ| ≈ 2.0e-4（相对 3.3）。即 FiLM 实际只对 N_low 敏感。`Σ̂` 元素 ~1e-8…1e-10 ⇒ 任何 λ_min 违反 ≤ ~5e-7 ⇒ `P_PSD ≤ ~2.5e-13`、`w_psd·P_PSD ≤ 2.5e-14`，而 NLL 量级 0.6–21 —— 罚项对联合版梯度无实质影响。（对照：同一非一致构造在归一化坐标下 P_PSD = 4.1e-3，差约 16 个数量级。）`50` line1 方案 C 的先验做过 log+z-score 标准化（`src/utils/h5data.py:71`），line2 的 κ 无标准化。
**影响面:** H3（联合 vs 独立）的对照变量"跨投影信息是否流动"**仍然成立**（拓扑差异仍在）；但"FiLM 矩条件化有效"与"软一致罚项起作用"两条机制在 Phase 1 实际未生效，H1 的"物理先验"解读须相应弱化，写进实验报告。
**Rejected:** 本轮临时给 κ 加归一化 —— 会改变已设计并启动的 E1 处理，且 spec 文本未规定归一化，属擅自扩大变更面。
**Spec section:** [S1]、[S2]
**Status:** live（待验收闸口确认）

## D14: κ 的 log₁₀N_low 分量口径 = 图组六投影总计数（偏差登记）
**Phase:** review
**Chosen:** κ 末位取 `counts.sum(dim=(1,2,3))` 的 log₁₀，即 **6·N_low**（`00` [S1] 定义 N_low 为输入端粒子数）。实测 tier_010000 → 60000，κ 末位 4.7782 而非 4.0。推理侧已与训练同口径（同源派生）。
**Rationale:** 对同一 N_low 档，偏差是常量 log₁₀6 ≈ 0.778，可被 FiLM 线性层吸收；E1 单档（1e4）与 E2 多档（3e3/1e4/1e5）中该偏移与档位无关，故不改变任何档间比较。风险仅在"把 κ 末位当 log₁₀N_low 解读/审计"之处。
**Rejected:** 改回 `N_low` —— 会改变已启动 run 的条件输入，属实质变更，收益为零。
**Spec section:** [S1] C5、`00` [S1]
**Status:** live

## D15: 图内轴向布局的消费者侧纠正（含对 `20` D13 归因的更正）
**Phase:** review
**Chosen:** 确认 `src/line2/data_pipeline/projections.py::six_projections` 存 `h.T`，图内布局为 **axis 0 = 后一坐标 a_j 的 bin、axis 1 = 前一坐标 a_i 的 bin**；全部消费方（`metrics.reconstruct_sigma`、`metrics.halo_fraction_gap`、`metrics.evaluate_group`、`baseline.moment_matching_baseline`、`unet.batch_sigma_from_counts`、`independent.proj_condition(_batch)`）在入口转置回 `[a_i, a_j]` 语义；生产端与已落盘数据集一字未动。
**实证:** 40 万粒子合成表（各轴 range/σ 互异；真值 = `np.cov(table, ddof=0)`）：修复前 `reconstruct_sigma` 对角相对误差 **−0.867 / +6.53 / +5.53 / −0.847**；修复后 **8.5e-5 / 5.8e-4 / 5.3e-4 / 5.7e-5**（= 分箱量化地板）。真实锚点对 `twiss.json`：对角相对误差由 **−7.0%/+7.7%/−13.2%/+15.3%** 降到 **7.8e-4/5.3e-4/6.9e-4/5.9e-4**，max|Δρ| 5.9e-4，ε₄D 4.8e-3。
**Rationale:** 旧实现的"各轴 range/σ 相同的合成高斯云"测试恰好让错配自抵消（对角误差 <5%），故长期未被抓住；新回归测试改用逐轴量级/形状互异的粒子表，修复前必失败。
**Rejected:** 改生产端 `six_projections` 去掉 `h.T` 并重新生成数据集 —— 5.2 GB 数据已落盘且 E1 正在消费，重生成会作废全部已启动 run，而消费方转置的代价仅一行且可被真值测试钉住。
**Spec section:** [S4] C2（`10`）、[S1] C3（`20`）
**Status:** live

## D16: 顺带修正 `metrics.evaluate_group` 的 M1 口径
**Phase:** review
**Chosen:** `evaluate_group` 的 M1 由整组一次归一的 `rel_rmse(group, group_ref)` 改为逐投影归一后各算再平均的 `group_rel_rmse`（= `20` [S1] M1 原文）。N_eff 路径（`n_eff.py`）与 `evaluate.py` 本就用 `group_rel_rmse`，主指标口径未受影响；风险只在谁拿 `evaluate_group["M1"]` 去插值。
**Rationale:** 一个度量一个函数、口径唯一（`20` Global Constraints）。实测差异：投影误差均匀时 +0.003%，投影误差极不均时 +100.6%。
**Rejected:** 保留整组一次归一 —— 与 `20` [S1] M1 原文及 N_eff 路径（`group_rel_rmse`）口径分叉，同一"主指标"出现两个数值来源。
**Spec section:** [S1] M1
**Status:** live

## Progress

| [Sn] | commit | review | status |
|---|---|---|---|
| [S1] | 807d949、fb04ef5、21aaedb | L3b/L4 批次复审（2026-09-14，结论见报告） | implemented（Phase1Net：FiLM U-Net + Softplus 概率头；参数 ≤6e7；条件化防泄露） |
| [S2] | 807d949、fb04ef5、21aaedb | L3b/L4 批次复审（2026-09-14，结论见报告） | implemented（Poisson NLL 手算 + PSD 判别 + 率语义） |
| [S3] | f3c45f5、fb04ef5、21aaedb | L3b/L4 批次复审（2026-09-14，结论见报告） | implemented（冒烟+确定性 30 步 ×2 轨迹一致；max_steps=50000 预注册；混合站台逐样本 edges PSD；D12 DDP 偏差登记） |
| [S4] | f3c45f5、fb04ef5、21aaedb | L3b/L4 批次复审（2026-09-14，结论见报告） | implemented（独立版 6×SingleProjNet，4 维物理 κ；H3 对照臂就绪） |
| [S5] | f3c45f5、fb04ef5、21aaedb | L3b/L4 批次复审（2026-09-14，结论见报告） | implemented（infer：确定性、率图逐投影和 = N_high、ckpt config 驱动重建） |
