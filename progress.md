# 项目进度（Agent 自动维护）

> 本文件由 Agent 按 [`00 [S13.4]`](docs/specs/00_master_spec.md) 自动维护，每阶段更新一次。

## 当前状态

| 项目 | 值 |
|---|---|
| **当前阶段** | M2 完成 → M3 待启动 |
| **当前任务** | M2 数据集生成已完成（`60 [S8]` 划分 + `60 [S14]` 工件契约 + G0 门禁通过）；4 项预注册判据修订待用户批准后启动 M3 |
| **进度** | 2/7 里程碑（M1、M2 完成；v1.0 Spec 冻结）|
| **下一步** | M3 方案 A baseline（EXP-01，G1(a)）——待 OQ-40-02/40-03/80-01/30-02 四项裁定批准 |
| **最近一次报告** | [`results/M2_dataset/stage_report.md`](results/M2_dataset/stage_report.md)（M2 数据集 + G0 门禁，2026-08-26）；[`results/M1_generators/stage_report.md`](results/M1_generators/stage_report.md)（M1 生成器）|

## 阶段完成历史

按时间倒序记录。每次阶段报告生成时同步追加。

| 日期 | 阶段 | 结论 | 报告链接 |
|---|---|---|---|
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
| OQ-40-02 | Open（待用户拍板）| AC14 SSIM 门在总强度归一化口径下无判别力（恒≈1.0）；Qwen 裁定推荐方案 C（SSIM 降为诊断量），已呈报用户 |
| OQ-40-03 | Investigating（待用户拍板）| σ_smooth,P 主判据升级：40 [S5] C5 hp 能量预算比不可操作（全档≈0.14 对旋钮无灵敏度）；Qwen 裁定方案 A（全频 L1 残差预算替代，锚点 0.55）|
| OQ-80-01 | Investigating（待用户拍板）| G0(b) 探针字面 ρ 不可操作（恒≈2.75 对 c_high 不敏感）；实现用差分形式通过；Qwen 裁定接受并建议修订 80 [S9] 文本 |
| OQ-30-02 | Investigating（待用户拍板）| 30 [S12] C4（σ_n 尾部 SNR 2–5）与 [S6] C8（SNR_hf<0.1）在 σ_K=2×median 下互斥；M2 采用 σ_K=11.0+σ_n=1.22e-4 通过；Qwen 裁定修订 C4 增补"与 C8 不相容时"分支 |
| OQ-80-02 | Investigating | EXP-06 放大读法：Qwen 裁定字面=仅上界放大；修正并入标定后 test_ood 重生成 |
| OQ-30-01 | Closed | 下采样"块平均"措辞与"保总强度"公式矛盾，实现按块求和；措辞修订列入下批变更 |
