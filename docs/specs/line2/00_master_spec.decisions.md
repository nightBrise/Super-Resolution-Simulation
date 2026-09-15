# Decision Log: line2_particle_sr 总纲与全局约束

<!-- ACTIVE LOG — continuously appended throughout lifecycle; do not treat as stale -->

## D1: line2 spec 集目录与编号
**Phase:** write-spec
**Chosen:** 新建 `docs/specs/line2/` 子目录，编号沿用十位留空惯例（00/10/20/30/40），line2 自带 `99_change_log.md`
**Rationale:** line1 的 00–99 冻结集留在根级，混编会使两线变更日志与冻结边界互相污染；子目录保持 line1 历史完整、line2 自治。
**Rejected:** 根级续编 100+ ——两线 spec 混在同一命名空间，变更日志无法分线维护。
**Spec section:** [S4]
**Status:** live

## D2: N_eff 为唯一主指标
**Phase:** write-spec
**Chosen:** 等效放大因子 N_eff 作为唯一主指标，物理观测量为带容差的次级判据
**Rationale:** 多指标综合评分无法预注册且易被选择性引用（line1 Π_leak/主指标耦合教训）；单一可插值指标使"放大"可数值化。
**Rejected:** 多指标加权综合分 ——权重无先验依据，且综合分掩盖"亮核锐利、暗区保守"的分桶行为。
**Spec section:** [S2] GC3
**Status:** live

## D3: line2 master_seed = 20260913
**Phase:** write-spec
**Chosen:** 独立 master_seed（20260913），SeedSequence spawn 链与 line1 同构
**Rationale:** 独立研究线使用独立随机预算，避免与 line1 的 20260825 种子树任何潜在耦合。
**Rejected:** 沿用 line1 的 20260825 ——两线数据若共用种子树，可复现性审计边界模糊。
**Spec section:** [S2] GC2
**Status:** live

## D4: 测试策略嵌入各 spec Claims，不立独立测试 spec
**Phase:** write-spec
**Chosen:** 每条 Claim 自带可测验收（测试名/命令/断言），沿用仓库 pytest markers
**Rationale:** Phase 1 规模下独立测试 spec 是重复载体；line1 的 05 因无人值守全链治理需求而立，line2 暂无该需求。
**Rejected:** 独立 `05_testing_spec.md` ——若后续 L4/L5 需要统一测试治理，再按 99 流程补立。
**Spec section:** [S4]
**Status:** live

## Progress

| [Sn] | commit | review | status |
|---|---|---|---|
| [S1] | 74bfe84（冻结）、fb04ef5 | **Phase 1 最终验收审查（2026-09-15）实测**：C2 有断言（`tests/unit/test_line2_dataset.py::test_build_dataset_deterministic_and_traceable` 的 `axes == list(AXES)`）+ 生产件 128² int32、∑counts 逐档严格等于 N_low；C3 由 `beam_splits(1000)` 复算与生产 `manifest.json` 切分逐元素一致、抖动参数由 (beam_id, master_seed) 还原有测试（`test_beam_group_jitter_metadata_matches_params`）；**C1（术语单一来源的静态检查）未逐条审计**（仅抽读） | implemented（C1 未核实） |
| [S2] | 74bfe84、fb04ef5、ae9d4b8 | **C5 实测通过**（本次实跑 `python scripts/check_env.py` → `OK: 环境依赖版本一致`，ocelot/sdds 已入 `environment.yml`）；**C4（路径前缀断言）无任何测试**——三个 CLI 的默认输出确实指向 `studies/line2_particle_sr/`，但全仓无 `startswith`/`is_relative_to` 一类断言（审查实测） | implemented with gap（C4 无测试） |
| [S3] | 74bfe84、306d396、148604d | **C6 实测满足**（四条门禁各对应一个可执行表达式，`map_report._GATE_RULES` 与 `40` [S3] 逐条 1:1）；**C7 无专用存在性入口**——交付物实测齐备（`e1_gates.json` 四条结论 + `figure_avail_domain.png` + `n_eff_table.csv` + 两份报告，非空），但"交付物清单"由一个显式入口逐项断言的检查缺失 | implemented with gap（C7 无专用入口） |
| [S4] | 74bfe84、c00fe03、b758e57 | **C8 实测满足**（`docs/specs/line2/` 下 00/10/20/30/40 五份 spec + `99_change_log.md` 在册，[S4] 索引与文件一一对应）；**C9 未逐 commit 对账**（审计项，未核实） | implemented（C9 未核实） |

**范围与事实依据**：C5 由本次实跑确认，C2/C3/C6/C8 由 Phase 1 最终验收审查的实测记录与仓库现件确认，C1/C9 属抽读/审计项故记"未核实"；C4/C7 的覆盖缺口按本表登记（本批不新增测试）。
