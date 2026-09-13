# Decision Log: Phase 1 实验矩阵与判定

<!-- ACTIVE LOG — continuously appended throughout lifecycle; do not treat as stale -->

## D1: L4+L5+L7（图像级）合并为一份实验 spec
**Phase:** write-spec
**Chosen:** 核心对比、三档扫描、跨站泛化同入 `40`
**Rationale:** 三者共享同一批 runs、同一套 summary 与同一组门禁脚本，拆开会造成"同一 run 被三份 spec 引用"的归属混乱。
**Rejected:** 每叶一份 spec（L4/L5/L7 各一份）——轮次与判据强耦合，拆分只增加交叉引用无独立回滚价值。
**Spec section:** [S1]
**Status:** live

## D2: H3 判据用同种子成对计数三分类，不用符号检验 p 值
**Phase:** write-spec
**Chosen:** 5 对中"联合 ≥ 独立"的计数映射到 {成立, 方向成立, 不成立, 不可判定}
**Rationale:** 5 对样本的符号检验即使在全正时 p = 0.0625 也达不到 α = 0.05——形式化检验在这里必然"不可判定"，三分类计数更诚实且保留了中间态。
**Rejected:** Wilcoxon 符号秩检验 ——同样功效不足；二显著判"成立" ——制造虚假的统计强度。
**Spec section:** [S3] H3
**Status:** live

## D3: 跨站泛化边界 ρ ∈ {1.0, 0.5}
**Phase:** write-spec
**Chosen:** ρ ≥ 1 成立；0.5 ≤ ρ < 1 成立（弱）；ρ < 0.5 不成立（阈值冻结于本 spec 批准）
**Rationale:** 首轮检验的目标是"跨站能力存在性"；0.5 的衰减对诊断用途仍可能可用，而 0.8 的严格线更适合作 Phase 2 复检标准。
**Rejected:** ρ ≥ 0.8 才算成立 ——首轮过严易得假阴性，把本可用的方法误杀。
**Spec section:** [S2]
**Status:** live

## D4: 全部门禁三分类化
**Phase:** write-spec
**Chosen:** 每条门禁输出 {成立, 方向成立(仅 H3), 不成立, 不可判定}
**Rationale:** "不可判定"是 line1 的核心教训（G2 种子敏感判"不可判定"如实登记）——中间态必须有一等合法地位，防止把模糊结果硬拗成结论。
**Rejected:** 二分类成立/不成立 ——迫使边缘结果站队，制造虚假确定性。
**Spec section:** [S3]
**Status:** live

## D5: E2 只跑获胜变体
**Phase:** write-spec
**Chosen:** 三档曲线的 {3e3, 1e5} 两档只训 E1 获胜变体 × 5 种子
**Rationale:** E1 已裁决变体问题（H3）；全矩阵 2 变体 × 3 档 × 5 种子 = 30 次训练的增量信息只在变体×档位交互作用上，留待 L5 结果提示需要时再扩。
**Rejected:** 全矩阵 30 次训练 ——交互作用假设优先级低，成本翻三倍。
**Spec section:** [S1] E2
**Status:** live

## D6: 两段式判据消费——val 选型、test 一次（审核 M2 修复，2026-09-14）
**Phase:** review
**Chosen:** 变体路由与 E2 调度只消费 val N_eff；H1/H3/H8 出口门禁只消费 test 的一次性评估；脚本内数据路径断言强制边界
**Rationale:** 审核指出 E2 按"E1 门禁结论"路由但门禁切分未写明——若路由读了 test 即"用 test 选型"，违反 GC3 与 F9"test 只报一次"。
**Rejected:** 门禁同时报告 val 与 test 两套结论 ——给选择性引用留口子（line1 教训）。
**Spec section:** [S3]
**Status:** live

## D7: 获胜变体全分支路由（审核 M3 修复，2026-09-14）
**Phase:** review
**Chosen:** 5/5→联合；4/5→联合（弱）；3/5→val N_eff 中位数裁定并标注"H3 不可判定"；≤2/5→独立
**Rationale:** 审核指出 H3 非"成立"分支时获胜变体无定义，E2/E3/H1 全链悬空；val 裁定与两段式消费（D6）兼容。
**Rejected:** 3/5 时判定 H3 失败并默认独立 ——把不可判定硬拗成结论，违反 D4。
**Spec section:** [S3]
**Status:** live

## D8: 尺子先行 = E0 时间戳先于 E1（审核 m8 修复，2026-09-14）
**Phase:** review
**Chosen:** E0 完成时间戳 SHALL 早于 E1 任一训练启动时间戳（C5 可验证顺序约束）；主曲线命名"方法可用域地图"
**Rationale:** 大纲明文"尺子（raw 曲线）先于任何训练"——顺序约束不编码就会在执行中被跳过。
**Rejected:** 不编码顺序 ——靠纪律无靠审计。
**Spec section:** [S1] C5、[S4] C1
**Status:** live

## D9: H8 聚合维度预注册（审核 M4 修复，2026-09-14）
**Phase:** review
**Chosen:** 达标率聚合 = test 束 × 站 0 × N_low=1e4 × 5 种子中位数输出；容差线节点 = 曲线上与 N_eff 对数距离最近的档
**Rationale:** 审核指出达标率的聚合维度与容差线节点映射未定义——不预注册则达标率可被聚合方式操纵。
**Rejected:** 全维度（束×站×N）平均 ——掩盖暗桶行为，与分桶报告原则冲突。
**Spec section:** [S3] H8（联动 `20` [S4]）
**Status:** live

## Progress

| [Sn] | commit | review | status |
|---|---|---|---|
| [S1] | — | — | pending |
| [S2] | — | — | pending |
| [S3] | — | — | pending |
| [S4] | — | — | pending |
