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

## D10: E1 两臂种子统一为 {100,101,102,103,104}
**Phase:** review
**Chosen:** `e1_joint_s0..s4` 与 `e1_indep_s0..s4` 使用**同一组种子** 100–104（原先 joint = 0–4、independent = 100–104，两臂种子集合互不重叠）。
**Rationale:** `40` [S3] H3 判据是**同种子成对比较**（5 对），`30` [S4] C2 要求两版"相同种子派生链"。原配置下按值配对 = 0 对，H3 永远"不可判定"；这是配置缺陷而非判据缺陷。已加自动断言（两臂种子集合必须逐元素相同、同种子必须落在同序号 run_id 上）。
**Rejected:** 维持原配置（joint {0–4} / independent {100–104}）—— 按值配对为 0 对，H3 恒判"不可判定"，且两臂不在同一实现噪声下比较，与 `30` [S4] C2 的相同派生链要求冲突。
**Spec section:** [S1] C3、[S3] H3
**Status:** live

## D11: E0 门禁曲线口径 = 跨 val+test 束；训练期 val N_eff 用 val 束曲线（两段不交叉）
**Phase:** review
**Chosen:** E0（`40` [S1]）的裸误差–N 曲线按 `20` [S2]「跨 val/test 束中位数与 IQR」在 **300 束（val 150 + test 150）× 两站 × 五档 × k=20** 上计算，落盘 `results/e0_curve.json`，完成时间戳落 `results/e0_curve.meta.json`（`40` [S1] C5 的顺序证据）。训练期早停/选型用的 val N_eff 以 **val 束曲线**为插值基底（`30` [S3]）。
**Rationale:** 两段式判据消费（`40` [S3]）要求 val 段与 test 段各用其口径；E0 是纯数据基线（与任何模型输出无关），故其"完成早于 E1"实质保证"尺子先于读数"。
**Rejected:** 只用 val 束曲线进门禁 —— 门禁阶段要评 test 束，尺子口径与消费对象不一致会引入系统偏移。
**实测:** 300 束耗时 100.5 s；参照噪声地板 0.2856（1e5 档配对 M1 中位数 / √2）。
**Spec section:** [S1] C5、[S3]
**Status:** live

## D12: test 端评估与 H8 聚合口径
**Phase:** review
**Chosen:** run 的 test 端 N_eff = split `test` × station 0 × N_low 1e4 × subs 0–19，逐 (束, 子样本) 推理 → 每束的 err 与 13 维观测量对 sub 取中位数 → run 级 err = 全部束的中位数 → 在 E0 曲线上 log-N 插值得 N_eff（上/下溢带标记）。H8 的聚合按 `20` [S4]：**先对 5 个种子的逐观测量取中位数**，容差线按该束 N_eff（5 种子中位数）映射最近档，达标率 = 达标 (束 × 观测量) 对数 / (束数 × 13)；暗桶（B0/B1）的 M2 相对改进单独报告，不并入达标率。
**Rationale:** `20` [S4] 只规定"5 种子中位数输出 + 逐观测量先取中位数再判达标"，未规定子样本聚合与分子分母口径；此处把口径钉死以便复算（`40` [S3] C1）。
**Rejected:** 用 5 种子输出图逐像素中位数 —— 无法从已落的观测量恢复，且 spec 括号已指明按观测量取中位数。
**Spec section:** [S3] H8、[S4]
**Status:** live

## D13: E1 运行环境统一为 sr-sim（torch 2.4.0 / Python 3.11）
**Phase:** review
**Chosen:** 全部 E1 run 在 `sr-sim` 环境启动，`CUDA_VISIBLE_DEVICES` 逐 run 钉一张卡、`CUBLAS_WORKSPACE_CONFIG=:4096:8` 保证确定性模式无警告降级。首跑曾误入 `acc` 环境（torch 2.3.1 / Python 3.10），该 run 作废重跑。
**Rationale:** `00` GC4 公平性要求同一实验内除受控变量外全部一致；跨环境（不同 torch 版本）会引入与变体效应不可分离的混杂。
**Rejected:** 沿用首跑的 `acc` 环境 —— 与其余 run 的 torch 版本不同，构成与变体效应不可分离的混杂（GC4）。
**Spec section:** [S1] C3
**Status:** live

## Progress

| [Sn] | commit | review | status |
|---|---|---|---|
| [S1] | fb04ef5、21aaedb | L4 批次复审（2026-09-14） | implemented（train/eval/gates 编排） |
| [S2] | — | L4 批次复审（2026-09-14） | 待 E3 |
| [S3] | fb04ef5、21aaedb | L4 批次复审（2026-09-14） | implemented |
| [S4] | — | L4 批次复审（2026-09-14） | 待报告 |
