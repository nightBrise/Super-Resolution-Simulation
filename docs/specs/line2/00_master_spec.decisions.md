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
| [S1] | — | — | pending |
| [S2] | — | — | pending |
| [S3] | — | — | pending |
| [S4] | — | — | pending |
