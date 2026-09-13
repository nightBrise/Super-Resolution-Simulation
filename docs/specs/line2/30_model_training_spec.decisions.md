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

## Progress

| [Sn] | commit | review | status |
|---|---|---|---|
| [S1] | — | — | pending |
| [S2] | — | — | pending |
| [S3] | — | — | pending |
| [S4] | — | — | pending |
| [S5] | — | — | pending |
