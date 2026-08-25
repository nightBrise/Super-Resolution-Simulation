# Specs 索引

本项目采用**多文档渐进披露**的 spec 集（十位留空编号），位于 `docs/specs/`。

> **当前状态：全部文档为 v0.1 草案，尚未冻结，待用户审查。**
> 审查通过后，将版本升级为 v1.0（冻结）并同步更新 `99_change_log.md` 的版本历史。`99_change_log.md` 本身为活跃维护的变更记录。

| 编号 | 文档 | 作用 | 状态 |
|---|---|---|---|
| `00` | [`00_master_spec.md`](00_master_spec.md) | 项目总纲、全局符号、文档索引、里程碑、变更规则 | 草案（待审查） |
| `05` | [`05_testing_spec.md`](05_testing_spec.md) | 测试策略（分层、目录、里程碑绑定、失败处理）| 草案（待审查） |
| `10` | [`10_research_plan.md`](10_research_plan.md) | 研究假设、实验逻辑、成功标准 | 草案（待审查） |
| `20` | [`20_physics_generator_spec.md`](20_physics_generator_spec.md) | 高分辨率真值 `H` 与物理标签生成 | 草案（待审查） |
| `30` | [`30_degradation_spec.md`](30_degradation_spec.md) | 低分辨率观测 `L` 退化生成 | 草案（待审查） |
| `40` | [`40_prior_spec.md`](40_prior_spec.md) | 物理先验 `P` 与先验等级 | 草案（待审查） |
| `50` | [`50_network_spec.md`](50_network_spec.md) | 方案 A / B / C 网络结构 | 草案（待审查） |
| `60` | [`60_training_spec.md`](60_training_spec.md) | 训练协议（损失、数据、优化） | 草案（待审查） |
| `70` | [`70_evaluation_spec.md`](70_evaluation_spec.md) | 图像、物理、精细结构评估指标 | 草案（待审查） |
| `80` | [`80_experiment_matrix.md`](80_experiment_matrix.md) | 实验阶段划分与实验矩阵 | 草案（待审查） |
| `90` | [`90_delivery_spec.md`](90_delivery_spec.md) | 交付物与最终报告规范 | 草案（待审查） |
| `99` | [`99_change_log.md`](99_change_log.md) | 版本历史、变更明细、未决问题 | 活跃（持续追加） |

**阅读顺序：** 先读 `00_master_spec.md`（总纲），再按当前执行模块精读对应文档。

**变更规则：** 修改任何 Spec 文档前，先在 `99_change_log.md` 记录 `Proposed` 条目；批准后改为 `Approved` 再修改。实现阶段的决策追加记录到 `99` 的未决问题/变更明细。

**来源归档：** 对话记录 `chat-超分辨率增强模拟1.txt`（spec 起草来源）已归档于 `docs/specs/archive/`，供审查与追溯时查阅。
