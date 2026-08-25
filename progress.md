# 项目进度（Agent 自动维护）

> 本文件由 Agent 按 [`00 [S13.4]`](docs/specs/00_master_spec.md) 自动维护，每阶段更新一次。

## 当前状态

| 项目 | 值 |
|---|---|
| **当前阶段** | M3 执行中（EXP-01a 完成 + 坍缩修复；σ_K 标定被 R_E 门阻塞）|
| **当前任务** | 待用户 P0 报批（σ_smooth,H 修订 + OQ-40-02/30-02/30-03 裁定 + 坍缩修复 spec 修订）；批准后 dev1 重生成 → 标定重跑 → G1(a) |
| **进度** | 2.5/7 里程碑（M1、M2 完成；M3 部分完成：EXP-01a 三方案健康）|
| **下一步** | P0 报批 → dev1 数据重生成（σ_smooth,H=0.125×w_fine）→ EXP-01 标定重跑（01b/c/d）→ G1(a) 判定 |
| **最近一次报告** | [`results/EXP-01_summary/stage_report.md`](results/EXP-01_summary/stage_report.md)（M3 EXP-01a + 标定，2026-08-26）；[`results/M2_dataset/stage_report.md`](results/M2_dataset/stage_report.md)（M2）|

## 阶段完成历史

按时间倒序记录。每次阶段报告生成时同步追加。

| 日期 | 阶段 | 结论 | 报告链接 |
|---|---|---|---|
| 2026-08-26 | M3 | 部分通过（EXP-01a 三方案健康完成：坍缩修复 + 哨兵全过；EXP-01b σ_K 标定被 R_E 门阻塞 OQ-30-03 → 共性根因 σ_smooth,H 待 P0 报批）| [`results/EXP-01_summary/stage_report.md`](results/EXP-01_summary/stage_report.md) |
| 2026-08-26 | M2 | 通过（dev1/v1 数据集 + manifest + G0 三判据 pass；L0+L1 111 通过、acceptance M2 全部通过；B 类登记：探针差分操作化 OQ-80-01、σ_n 判据互斥 OQ-30-02、hp 预算比主判据失效 OQ-40-03、EXP-06 读法 OQ-80-02）| [`results/M2_dataset/stage_report.md`](results/M2_dataset/stage_report.md) |
| 2026-08-26 | M1 | 通过（20/30/40 全部 AC；L0+L1 90 通过、acceptance 24 通过 + 1 xfail 为 B 类待裁定）| [`results/M1_generators/stage_report.md`](results/M1_generators/stage_report.md) |

## 待办（spec 层）

- [x] M0：Spec 集 v1.0 冻结（用户审批批次八补强后）
- [x] M1：基础模拟函数就绪（`20` `30` `40`，代码 + 测试 + 验收）
- [x] M2：数据集生成（`60 [S8]` + `60 [S14]` + G0 门禁）
- [ ] M3：方案 A baseline（EXP-01）
- [ ] M4：主实验 + 先验增益判定（EXP-02 + G2）
- [ ] M5：消融与归因（迷你 EXP-03/04 + EXP-07/08）
- [ ] M6：最终报告 + 出口决策

## Agent 工作守则（与 `00 [S13]` 同步）

- 每阶段开始前：完成 `[S13.2]` 7 条必检项
- 每阶段执行中：按各 spec 失败路径 + `80` [S11] R1–R9 监控
- 每阶段结束后：按 `[S13.3]` 6 章节生成 `results/<EXP>/stage_report.md` + 99 登记 + 更新本文件
- 失败路径决策：按 `80` 附录 A 优先级
- 跨文档一致性：grep 自动检查
- 进度追踪：本文件由 Agent 自动维护（`00 [S13.4]`）

## 未决问题

（与 `99` 未决问题清单同步）

| ID | 状态 | 摘要 |
|---|---|---|
| OQ-20-03 | Investigating（待 P0 报批）| **共性根因**：σ_smooth,H=0.5×w_fine 按构造抹杀精细结构（能量 5e-5）；推荐 0.125×w_fine（保留 54%）→ 数据重生成 + 高频判据复验 |
| OQ-30-03 | Investigating（待 P0 报批）| σ_K 标定 R_E 比率门不可操作（2.58>0.60，D2 反常大于 D1）；推荐先修 σ_smooth,H 再复评原门 + 后备分支 |
| OQ-40-02 | Open（待用户拍板）| AC14 SSIM 门无判别力（恒≈1.0）；Qwen 推荐方案 C（SSIM 降诊断量） |
| OQ-40-03 | Investigating（待 P0 报批）| σ_smooth,P 主判据：hp 预算比不可操作（同根 OQ-20-03）；根因修复后复验原判据，方案 A（全频 L1）为后备 |
| OQ-80-01 | Investigating（待 P0 报批）| G0 探针字面 ρ 不可操作（同根 OQ-20-03）；差分形式已实现通过；根因修复后复验字面量 |
| OQ-30-02 | Investigating（待用户拍板）| σ_n 判据互斥；Qwen 推荐 C4 增补"与 C8 不相容"分支 |
| OQ-80-02 | Investigating | EXP-06 放大读法：字面=仅上界放大；修正并入标定后重生成 |
| OQ-30-01 | Closed | 下采样"块平均"措辞与公式矛盾，实现按块求和 |
