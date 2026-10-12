# 项目进度（Agent 自动维护）

> 本文件由 Agent 按 [`00 [S13.4]`](docs/specs/00_master_spec.md) 自动维护，每阶段更新一次。

## 当前状态

| 项目 | 值 |
|---|---|
| **当前阶段** | line1 收官（2026-09-12）；line2 Phase 1 交付 + 验收（2026-09-15）→ **T0 诊断反转头号结论（2026-10-12，`40` D21）→ 前置批次三项完成（2026-10-13，`40` D22）：站 1 同形=物理、last 口径 H8 0.71/0.95、逐束个体信息证实（cos 0.90–0.94）** |
| **当前任务** | line2 Phase 1 **交付完成**：L1/L2/L3b/L4 实现 + 两轮复审整改（needs fixes → 全部清零，默认套件 443 passed / 2 xfailed）；E0 曲线 + 容差线尺子先行（时间戳早于 E1）；**E1 十次（2026-09-14 训练完成）+ E2 十次（2026-09-14T23:22:22Z 训练完成、23:53:42Z 全链完成）**，站 0/站 1 test 端评估、门禁、三点放大主曲线、域地图与报告均已落盘。门禁结论：**H1 成立（饱和，5 种子 N_eff 全 1e5）/ H3 成立（5/5 对全为域顶平局，n_ties=5、n_strict_joint_wins=0）/ H7 不可判定（rho_median=1.0 披露但两站全 overflow）/ H8 不成立（达标率 0.1318，容差档 mode=1e5）**。核心定量结论：**放大比 N_eff/N_low 随 N_low 单调下降——3e3 档 ≥33.3×、1e4 档 ≥10×、1e5 档 1.0×（该档 5 个 run 中 1 个未触顶，N_eff=98,724）；3e3/1e4 两档的中位放大比恰等于 1e5/N_low，由域顶构造得出、分辨力为零，故仅 1e5 档含可测量的非饱和点**。执行期修掉六个实测缺陷（图内轴向布局、E1 配置两处、`run_e1` 相对路径、E2 链等待脚本、`map_report` 必填参数）并补两处规格合规（H8 容差档落盘、config 契约名）。**最终验收审查裁决（2026-09-15，审查对象 HEAD `1dfc197`）= needs fixes**：两条 blocking（报告 §3 的常数解论证被可复算对照否证；`40` [S3] H8 要求的暗桶 M2 单独报告缺失）+ Major M1–M9；**产物侧独立复算全部成立**（25 份 `test_eval*.json` 的 N_eff 逐位一致、`n_eff_table.csv` 字段级零失配、H1/H3/H8 逐位一致、默认套件 443 passed / 2 xfailed）。文档侧整改已落地：`line2_phase1_stage_report.md`（常数解对照重写、E2 时间窗、`tree_dirty`、版本核对命令、κ 影响面、选型饱和披露）+ 决策登记 `40` D15–D18、`10` D13、`30` D20/D21 + `00` 进度板；代码侧两项修法已落地（`cmd_eval` 默认跳过 + `--force` 覆盖 + `eval_history.jsonl` 留痕；`evaluate.py` 记 `per_beam[].m2_buckets`，`gates.h8_dark_bucket_report` 汇总暗桶），待提交后重生成 `phase1_report.md` 的 H8 小节暗桶段。**【2026-10-12 增补】T0 诊断批次（五项零训练实验，`reports/t0_diagnostics/T0_SUMMARY.md`）完成**：① 早停损失量化——3000/3000 束级配对差全部为负，test err 中位 0.272→0.088（−66%），20/20 run 反超常数解 0.194398；② κ 无信息消融实锤（信息臂=浮点地板，全零臂=承重常偏置）；③ T0-3 判据退化（机械零假设混淆），严格复读机 20/20 否证；④ H8——12/13 项实例内偏移主导、oracle 上界 0.9723、independent 臂 0.5174（不稳健）、M6 容差线非噪声线；⑤ 站 1 err 低 7.36%（A→B 不被证伪），但训练集默认含站 1、站 1 目标图近同形待查因。两份报告已就地标记勘误（`40` D21），冻结门禁判定输出不变 |
| **进度** | M0–M6 全部完成；出口决策「不进入下一阶段」定案——分支树 (a) 阴性/等效合法终点（条件 1 等效/种子敏感、非显著负），分支 (b) 不触发（读法裁决随 99 R187 登记）；99 [S4] 未决问题表 7 条全部闭环；核心结论——主指标上先验增益未获跨种子稳健确认（A−B 等效（种子级）/ A−C、B−C 种子敏感不可判定），次指标 ε_z 与预测离散度上方案 C 四种子一致占优（约为 A/B 的一半），G3：A−B 通过、A−C/B−C 极端退化档反转（E 类如实报告） |
| **下一步** | 起草 **Phase 1.5 预注册变更提案**（证据已齐：选型准绳改 val NLL/err + 训满预算；路由规则改次级准绳；迁移臂 `stations=(0,)` + target 定标改公共物理域（解禁 θ 踢角通道）；常数解补入基线族 `40` D18；N_eff 次级准绳升主 + 报告域扩展；H8 校准降级为收尾项；κ 重参数化与定标口径变更同批；"个体信息门禁"（增量对齐 cos + 打乱对照）正式化），走 99 待用户批准后重训。裁决候选：exp2 消融规则、训练含站 1 vs H7 语义（已由 D22 部分清账）、M6 容差线构造、pre1"伪影"分句记账 |
| **最近一次报告** | [`final_report.md`](docs/reports/line1_substitute_sr_final_report.md)（M6 最终报告，2026-08-30；2026-09-12 收官时修正 5 处图片相对链接）；[`studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md)（M5）；[`studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md)（M4）|

## 阶段完成历史

按时间倒序记录。每次阶段报告生成时同步追加。

| 日期 | 阶段 | 结论 | 报告链接 |
|---|---|---|---|
| 2026-09-12 | 收官 | 出口决策定案「不进入下一阶段」（用户确认；分支 (a) 成立、(b) 不触发裁决；99 [S4] 七条 OQ 闭环；final_report 图片链接与路径记账勘误——99 R187 登记） | [`99` R187](docs/specs/99_change_log.md) |
| 2026-08-30 | M6 | 完成（final_report 六章 + 五图 + 附录 A/B + N1–N8 自检通过；出口决策矩阵：条件 1 不满足（等效/种子敏感）→ 分支 (a) 阴性/等效合法终点，出口决定「不进入下一阶段」呈报用户确认；99 行 186 登记 Implemented（出口决策待用户确认））| [`final_report.md`](docs/reports/line1_substitute_sr_final_report.md) |
| 2026-08-30 | M5 | 完成（40 次推理评估 + 4 种子聚合 + G3 判定：A−B 不反转 G3 通过；A−C/B−C 方向反转 E 类如实报告；EXP-07 c_mid 信息有效；EXP-08 C 离散度最低；99 行 185 登记 Implemented）| [`studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-03_summary/stage_report.md) |
| 2026-08-29 | M4 | 通过（12 run 训练 + 评估 + 4 种子聚合 + G2 重判 R1–R4：A−B 等效（种子级，R2+R3 降级）/ A−C、B−C 种子敏感不可判定（R4 兜底）；次指标 C 4 种子一致减半；G1(b) 通过；99 行 184 登记 Implemented）| [`studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md) |
| 2026-08-28 | M4（过渡）| G2 跨种子不一致 → 失败路径已批准扩种子（seed2/3 训练启动中）；训练 6/6 + 评估 12/12 + G1(b) 通过 + 五图 + 聚合 + 99 登记完成；过渡版 stage_report 已落盘，待 seed2/3 完成按 R1-R4 重判 | [`studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-02_summary/stage_report.md) |
| 2026-08-26 | M3 | 通过（σ_smooth,H 根因修订（P0 批准）后 EXP-01 标定重跑完成：坍缩修复 + 三方案 dev2 重训哨兵全过；G1(a) 判定通过；EXP-01b σ_K 比率门复评仍失败判永久不可操作，启用预注册后备分支——「部分通过」为 P0 前中途快照，已于收官对账修正） | [`studies/line1_substitute_sr/results/summary/EXP-01_summary/stage_report.md`](studies/line1_substitute_sr/results/summary/EXP-01_summary/stage_report.md) |
| 2026-08-26 | M2 | 通过（dev1/v1 数据集 + manifest + G0 三判据 pass；L0+L1 111 通过、acceptance M2 全部通过；B 类登记：探针差分操作化 OQ-80-01、σ_n 判据互斥 OQ-30-02、hp 预算比主判据失效 OQ-40-03、EXP-06 读法 OQ-80-02——均已随 P0 批准闭环）| [`studies/line1_substitute_sr/reports/M2_dataset/stage_report.md`](studies/line1_substitute_sr/reports/M2_dataset/stage_report.md) |
| 2026-08-26 | M1 | 通过（20/30/40 全部 AC；L0+L1 90 通过、acceptance 24 通过 + 1 xfail 为 B 类待裁定——已随 P0 批准闭环）| [`studies/line1_substitute_sr/reports/M1_generators/stage_report.md`](studies/line1_substitute_sr/reports/M1_generators/stage_report.md) |

## 待办（spec 层）

- [x] M0：Spec 集 v1.0 冻结（用户审批批次八补强后）
- [x] M1：基础模拟函数就绪（`20` `30` `40`，代码 + 测试 + 验收）
- [x] M2：数据集生成（`60 [S8]` + `60 [S14]` + G0 门禁）
- [x] M3：方案 A baseline（EXP-01，G1(a) 通过）
- [x] M4：主实验 + 先验增益判定（EXP-02 + G2）—— 12 run 训练 + 评估 + 4 种子聚合 + G2 重判 R1–R4 完成（A−B 等效（种子级）/ A−C、B−C 种子敏感不可判定）；99 行 184 登记
- [x] M5：消融与归因（EXP-03/04/07/08 推理级 + G3 判定）—— A−B 不反转 G3 通过；A−C/B−C 方向反转 E 类如实报告；EXP-07 c_mid 有效；EXP-08 C 离散度最低；99 行 185 登记
- [x] M6：最终报告 + 出口决策（final_report 完成 + N8 通过；出口决策 2026-09-12 用户定案「不进入下一阶段」，line1_substitute_sr 收官）
- [x] line2 立项：4D 六投影一致超分（大纲 v0.8；五份 spec v1.0 冻结 2026-09-14，`docs/specs/line2/`）
- [x] line2 Phase 1 执行（2026-09-15 交付）：L1 数据管线 → L2 判据 → L3b 训练 → L4/L5 实验矩阵（E0/E1/E2/E3 全部完成、门禁与报告落盘）
  - [x] L1 `10` [S1]–[S5]：锚点入库（MD5 f39d905f…）+ 1000 束抖动群 700/150/150 + FODO + 五档 ×k=20 子采样落盘（`studies/line2_particle_sr/data/line2_ds_v1/`）
  - [x] L2 `20` [S1]–[S5]：度量族 M1–M6、裸误差–N 曲线（含参照噪声地板）、N_eff 恒等测试、容差线、基线族
  - [x] L3b `30` [S1]–[S5]：U-Net 概率头 + FiLM 条件化 + 独立/联合两版 + ★防泄露测试（**C4 未满足**：DDP 双卡缓期，已登记 D12）
  - [x] L4/L5 `40` [S1]–[S4]：E0 ✓ → E1 十次 ✓ → E2 十次 ✓ → E3 跨站 ✓ → H1/H3/H7/H8 门禁 + 三点放大域地图 + 双报告 ✓
  - [ ] line2 Phase 1 最终验收整改（2026-09-15 审查裁决 needs fixes）：文档侧已落地（`line2_phase1_stage_report.md` + `40` D15–D18、`10` D13、`30` D20/D21 + `00` 进度板）；代码侧修法已落地（派发端 `cmd_eval` 默认跳过 + `--force` + `eval_history.jsonl`；暗桶 `per_beam[].m2_buckets` + `gates.h8_dark_bucket_report`），待提交后重生成 `phase1_report.md` 的 H8 小节暗桶段

## Agent 工作守则（与 `00 [S13]` 同步）

- 每阶段开始前：完成 `[S13.2]` 7 条必检项
- 每阶段执行中：按各 spec 失败路径 + `80` [S11] R1–R9 监控
- 每阶段结束后：按 `[S13.3]` 6 章节生成 `results/<EXP>/stage_report.md` + 99 登记 + 更新本文件
- 失败路径决策：按 `80` 附录 A 优先级
- 跨文档一致性：grep 自动检查
- 进度追踪：本文件由 Agent 自动维护（`00 [S13.4]`）

## 未决问题

（与 `99` 未决问题清单同步；2026-09-12 收官对账后全部闭环）

| ID | 状态 | 摘要 |
|---|---|---|
| OQ-20-03 | Closed | σ_smooth,H=0.125×w_fine 根因修订——P0 第 1 项 2026-08-26 批准，dev2/v1 重生成 + G0 复评通过 + EXP-01 重跑完成 |
| OQ-30-03 | Closed | σ_K 比率门原门复评失败（1.38>0.60）判永久不可操作——P0 第 7 项批准，标定证据由后备分支承担（SNR_hf + G1(b)，M4 通过） |
| OQ-40-02 | Closed | SSIM 退出 AC14 门降为诊断量——P0 第 3 项批准 |
| OQ-40-03 | Closed | σ_smooth,P 主判据改全频 L1 残差预算（(0.2,0.9) 锚点 0.55）——P0 第 4 项批准；15× 采用值 EXP-01d 登记（0.5288） |
| OQ-80-01 | Closed | G0 探针差分形式入正文、字面量降诊断量——P0 第 5 项批准；dev2/v1 复评通过 |
| OQ-30-02 | Closed | 30 [S12] C4 增补「与 C8 不相容」分支——P0 第 6 项批准；D2 采用 σ_n=1.22e-4 |
| OQ-80-02 | Closed | EXP-06 读法裁定=仅上界放大；EXP-06 属 Phase 4 未执行，随 line1 收官闭环（后续消费 test_ood 前需核实采样读法：当前实现为双叶区间整体×1.5） |
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
