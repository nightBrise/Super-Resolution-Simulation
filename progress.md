# 项目进度（Agent 自动维护）

> 本文件由 Agent 按 [`00 [S13.4]`](docs/specs/00_master_spec.md) 自动维护，每阶段更新一次。

## 当前状态

| 项目 | 值 |
|---|---|
| **当前阶段** | M6 完成（final_report 六章 + 五图 + 附录 A/B + N1–N8 自检通过）；出口决策呈报用户确认中 |
| **当前任务** | 出口决策定案（用户确认「不进入下一阶段」或提出后续提案）|
| **进度** | M1–M5 完成；**M6 完成**：final_report.md（536 行，六章 + 判据符号表 + 出口决策矩阵 + 四类外部效度证据 + 附录 A/B）+ 五图（三方案对比版 PNG 300dpi）+ N8 对账无 FAIL（99 行 186 登记）；出口决策矩阵——条件 1 不满足（等效/种子敏感，非显著负）→ 分支树 (a) 阴性/等效合法终点；条件 2 B 满足/C 触发；条件 3 证据缺失；**出口决定「不进入下一阶段」待用户确认** |
| **下一步** | 用户确认出口决策 → 99 定案 → progress.md 终态 → 项目完成 |
| **最近一次报告** | [`final_report.md`](docs/reports/line1_substitute_sr_final_report.md)（M6 最终报告，2026-08-30）；[`studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md)（M5）；[`studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md)（M4）|

## 阶段完成历史

按时间倒序记录。每次阶段报告生成时同步追加。

| 日期 | 阶段 | 结论 | 报告链接 |
|---|---|---|---|
| 2026-08-30 | M6 | 完成（final_report 六章 + 五图 + 附录 A/B + N1–N8 自检通过；出口决策矩阵：条件 1 不满足（等效/种子敏感）→ 分支 (a) 阴性/等效合法终点，出口决定「不进入下一阶段」呈报用户确认；99 行 186 登记 Implemented（出口决策待用户确认））| [`final_report.md`](docs/reports/line1_substitute_sr_final_report.md) |
| 2026-08-30 | M5 | 完成（40 次推理评估 + 4 种子聚合 + G3 判定：A−B 不反转 G3 通过；A−C/B−C 方向反转 E 类如实报告；EXP-07 c_mid 信息有效；EXP-08 C 离散度最低；99 行 185 登记 Implemented）| [`studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md) |
| 2026-08-29 | M4 | 通过（12 run 训练 + 评估 + 4 种子聚合 + G2 重判 R1–R4：A−B 等效（种子级，R2+R3 降级）/ A−C、B−C 种子敏感不可判定（R4 兜底）；次指标 C 4 种子一致减半；G1(b) 通过；99 行 184 登记 Implemented）| [`studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md) |
| 2026-08-28 | M4（过渡）| G2 跨种子不一致 → 失败路径已批准扩种子（seed2/3 训练启动中）；训练 6/6 + 评估 12/12 + G1(b) 通过 + 五图 + 聚合 + 99 登记完成；过渡版 stage_report 已落盘，待 seed2/3 完成按 R1-R4 重判 | [`studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md) |
| 2026-08-26 | M3 | 部分通过（EXP-01a 三方案健康完成：坍缩修复 + 哨兵全过；EXP-01b σ_K 标定被 R_E 门阻塞 OQ-30-03 → 共性根因 σ_smooth,H 待 P0 报批）| [`studies/line1_substitute_sr/results/summary/EXP-01_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-01_summary/stage_report.md) |
| 2026-08-26 | M2 | 通过（dev1/v1 数据集 + manifest + G0 三判据 pass；L0+L1 111 通过、acceptance M2 全部通过；B 类登记：探针差分操作化 OQ-80-01、σ_n 判据互斥 OQ-30-02、hp 预算比主判据失效 OQ-40-03、EXP-06 读法 OQ-80-02）| [`studies/line1_substitute_sr/reports/M2_dataset/stage_report.md`](studies/line1_substitute_sr/reports/M2_dataset/stage_report.md) |
| 2026-08-26 | M1 | 通过（20/30/40 全部 AC；L0+L1 90 通过、acceptance 24 通过 + 1 xfail 为 B 类待裁定）| [`studies/line1_substitute_sr/reports/M1_generators/stage_report.md`](studies/line1_substitute_sr/reports/M1_generators/stage_report.md) |

## 待办（spec 层）

- [x] M0：Spec 集 v1.0 冻结（用户审批批次八补强后）
- [x] M1：基础模拟函数就绪（`20` `30` `40`，代码 + 测试 + 验收）
- [x] M2：数据集生成（`60 [S8]` + `60 [S14]` + G0 门禁）
- [x] M3：方案 A baseline（EXP-01，G1(a) 通过）
- [x] M4：主实验 + 先验增益判定（EXP-02 + G2）—— 12 run 训练 + 评估 + 4 种子聚合 + G2 重判 R1–R4 完成（A−B 等效（种子级）/ A−C、B−C 种子敏感不可判定）；99 行 184 登记
- [x] M5：消融与归因（EXP-03/04/07/08 推理级 + G3 判定）—— A−B 不反转 G3 通过；A−C/B−C 方向反转 E 类如实报告；EXP-07 c_mid 有效；EXP-08 C 离散度最低；99 行 185 登记
- [/] M6：最终报告 + 出口决策（final_report 完成 + N8 通过；出口决策呈报用户确认中）

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

## 事故记录（2026-08-28，C 类环境失败，自主处置）

- **现象**：批次 4（A_seed2/A_seed3）训练进程于 ~16:04 同时静默死亡（log 止于 step=20200，无 traceback；GPU 归零；last.ckpt=step 20000）。批次 1/2/3 均正常完成 50000 步。
- **初诊四问**：(1) 真实问题（进程消失+GPU 空闲）；(2) 环境/进程级（非代码 bug、非研究结果）；(3) 系统性存疑（两进程同时死，但无 OOM/GPU 日志证据，journalctl/syslog 无权限）；(4) 不触红线。
- **分类**：C 类环境失败。
- **处置**：train.py 无 resume 支持；重启为确定性等价（同 seed → 同轨迹）。备份死日志为 `train.log.dead1`，重启批次 4（A_seed2 pid=258284 / A_seed3 pid=258285）。
- **监控**：若重启后再次在 ~20000 步死亡 → 升级为系统性问题，需查根因（可能需用户协助查系统日志）。

### 事故根因（补充，2026-08-28）

- **根因**：seed2/3 config 生成时从 seed0 复制，`gpu.device` 未改 —— A_seed2 与 A_seed3 均为 `cuda:0`，两进程同卡共享 24GB（12.2+11.6≈23.8GB，接近上限）→ 峰值 OOM → 同时静默死亡。B_seed2 亦误为 cuda:1。
- **修复**：A/B/C 的 seed3 config 改 `cuda:1`，B_seed2 改 `cuda:0`；kill 并重启 A_seed3（pid=263335）。现双卡各一进程，无共享 OOM 风险。
- **教训**：批次调度"每 batch 两 run 分置双卡"依赖 config device 正确；生成 config 时必须显式设置 device（后续 B/C 批次启动前需复核 device）。
