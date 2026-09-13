# 锚点数据管线（SDDS 导入 + 抖动束群 + 子采样配对 + FODO 传输 + 投影数据集）Spec

> 符号与全局约束以 `00_master_spec.md` [S1]/[S2] 为准；本 spec 产出 Phase 1 全部训练/验证/测试数据。
> 版本：v0.2.1（**Frozen v1.0**，2026-09-14 用户批准冻结）。上游：大纲 v0.7 叶 L1；修订史 = v0.2（审核 findings B1/B5/B6/M1/m5/m6 清零）+ v0.2.1（复审收口）。
>
> 锚点实测参考值（写入本 spec 供审计，来源 = 2026-09-13 实测）：N = 200,001；Q = 500 pC；σ_x = σ_y = 0.329 mm；corr(x,x′) = corr(y,y′) = 0.855；横向零耦合；ε_n ≈ 1.26 mm·mrad；(x,x′) 128²（±4σ）占用 35.8%、被占格均值 34、峰值 485。

## [S1] 锚点入库与元数据

将锚点文件及其元数据纳入项目管理（用户 2026-09-13 定案：复制入项目、SDDS git 入库；2026-09-14 定案：期望 MD5 硬编码于本 spec）。

**Dependencies (optional):** 无

### Claims

- C1: WHEN 执行入库脚本 THEN `studies/line2_particle_sr/data/anchor/injector.sdds` SHALL 存在、大小 = 9,600,397 字节、且实算 MD5 SHALL 等于 `f39d905f96abd5588091dad6dd3884b2`（该值即源文件实测值——副本与源逐位一致的可验证断言，不依赖仓库外路径）。
- C2: `anchor/twiss.json` SHALL 含以下字段且与独立重算一致：Σ(4×4，按 (x,x′,y,y′) 顺序)、α_x/β_x/α_y/β_y、ε_geom 与 ε_n（两平面）、σ_t、σ_dp/p、corr(t,p)、六列各自的偏度与峰度；数值校验 = 同公式独立实现重算，相对误差 < 1×10⁻¹²。该文件同时是 `30` [S1] 条件输入的备料。
- C3: `anchor/manifest.json` SHALL 为合法 JSON 且含：source_path、md5（= C1 值）、units_p = "beta_gamma"、extraction_script、extraction_date、master_seed。
- C4: `git check-ignore studies/line2_particle_sr/data/anchor/injector.sdds` SHALL 返回非忽略（白名单条目生效）。

## [S2] SDDS 导入与统一子采样

从锚点 SDDS 读取全量粒子表；按统一子采样方案构造配对数据（A ⊂ B；训练档与曲线档同机制，2026-09-14 定案）。

**Dependencies (optional):** [S1]

- 档位集合 T = {1×10³, 3×10³, 1×10⁴, 3×10⁴, 1×10⁵}（五档，含训练三档 {3×10³, 1×10⁴, 1×10⁵} 与曲线专属两档）；每档每站 k = 20 个无放回子样本；全量表（2×10⁵）本身即目标/参照，不抽样。随机性 = `SeedSequence(20260913).spawn(beam_idx).spawn(station_idx).spawn(tier_idx).spawn(sub_idx)`。

### Claims

- C1: WHEN 解析 `anchor/injector.sdds` THEN SHALL 得到 6 列 float64 数组（200,001×6，列序 x/xp/y/yp/t/p）与 Charge = 5.0×10⁻¹⁰ C；解析结果 SHALL 与 SDDS 页布局（行数前置于参数）的手工二进制解析逐位一致。
- C2: 每个子样本 SHALL 满足：(a) 大小恰为该档 N；(b) 无放回（索引去重后长度不变）；(c) 索引升序后逐行指向全量表唯一源行（A ⊂ B 可验证）。
- C3: WHEN 同 master_seed 重跑子采样 THEN 全部索引 SHALL 逐位一致（单进程与多进程一致，`00` GC2）。
- C4: 同档任意两个子样本的索引集合 SHALL 两两不等（互异性；2e5 全量表除外——它唯一且即参照）。

## [S3] 束群构造：抖动变换 + 共同 FODO（P1 线性）

**Dependencies (optional):** [S1]

- **束群（B1 定案，2026-09-14 用户批准）**：束 i（i = 0…999）= 锚点全量表先经束 i 的抖动线性变换、再过共同 FODO 的实现；束 0 无抖动。L = 1000，切分 train 700 / val 150 / test 150（切分在 [S5]）。
- **抖动分布（预注册，冻结于本 spec 批准）**：前置薄透镜焦距 f_j = f₀·(1+δ)，δ ~ U(−0.05, +0.05)；x/y 踢角 θ ~ U(−10, +10) µrad；前置漂移 ℓ ~ U(495, 505) mm；每束三个抖动量由 SeedSequence 派生、相互独立。
- **共同 FODO（定死，2026-09-14）**：n_cells = 2、每 cell 相位推进 μ = 75°、半 cell 漂移 L_d = 2.5 m；四极强度由 cos μ = 1 − L_d²/(2f²) 反解。守恒排除规则：n_cells·μ ∉ {180°, 360°, …}（2×75° = 150° ✓）。

### Claims

- C1: WHEN lattice 构建完成 THEN Ocelot 实算每 cell 相位推进与目标 μ = 75° 的偏差 SHALL < 1°。
- C2: 每束两站快照的粒子数 SHALL 均等于 200,001（无丢失或增补）。
- C3: WHEN 同 (beam_id, master_seed) 重跑该束全链（导入→抖动→FODO）THEN 两站粒子表 SHALL 逐位一致（确定性；`00` C3）。
- C4: 站 1 与站 0 的 corr(x,x′) 差的绝对值 SHALL ≥ 0.3（推导：150° 相位旋转下 corr′ = 0.855·cos 150° ≈ −0.74，|Δ| ≈ 1.6，阈值为该推导的保守下取整——传输数学性质，非研究结果断言）。
- C5: 束间多样性不变量：跨束的站 0 corr(x,x′) 样本标准差 SHALL > 0.01（抖动机制的可验证效果；基准束 0.855，抖动 δ=5% 量级下相关性变化远超此阈）。

## [S4] 投影分箱与归一化

对每站、每子样本及全量表，产出六张 128×128 投影计数图（图组，`00` [S1]）。

**Dependencies (optional):** [S2]、[S3]

### Claims

- C1: 每张投影的计数总和 SHALL 精确等于其来源粒子表行数——分箱范围取该站全量表各轴 [min, max] 外扩 1%，零截断；若发生截断 SHALL 报错而非静默丢弃。
- C2: 图组通道顺序 SHALL 为 `00` [S1] 六投影枚举，存储 dtype int32（值域上限 2³¹−1，本任务最大计数 ~10³ 远低于上限）、值 ≥ 0，元数据内嵌轴对与 bin 边界。
- C3: 坐标归一化 SHALL 以该站 bin 边界线性映射至 [−1, 1]，用存档 bin_edges 反归一化的往返误差 SHALL < 1×10⁻¹⁰。
- C4: 两站各自独立定标 bin 边界；同站同 (beam, N, sub_idx) 的重复生成 SHALL 逐位一致。

## [S5] 束级切分与数据集落盘

**Dependencies (optional):** [S4]

- 切分：束级 train 700 / val 150 / test 150（SeedSequence 派生，互斥、并集完备）。
- 落盘：每切分一个 HDF5（`data/<data_version>/`，gzip level 4，int32）；束为 group，group 内含两站的 {五档 × k=20 子样本图组, 全量目标图组} 与元数据（beam_id、station、N_low、sub_idx、bin_edges、jitter 参数）。总量估算：~75 MB/束 × 1000 束 ≈ **80 GB**（int32）。
- `manifest.json`：`00` GC7 三元组、束数与切分表（beam_id → split）、档位与 k、FODO/抖动分布参数、锚点 MD5。

### Claims

- C1: 三切分互斥且并集 = 全部 1000 个 `beam_id`；train/val/test 束数 SHALL 为 700/150/150。
- C2: 落盘结构 SHALL 为：group = beam_id → 子 group = station → datasets = {tier_N_low_sub_XX（k=20）, target_full}，全部 int32、形状 (6, 128, 128)，且元数据字段完整（可机读校验）。
- C3: manifest SHALL 含 `00` GC7 三元组、切分表、档位/k/FODO/抖动参数、锚点 MD5，且为合法 JSON。
- C4: WHEN 同 master_seed 全流程重跑（含束群重建）THEN 数据集文件 SHALL 逐位一致（端到端确定性，`00` C3）。
- C5: 数据集中任一样本 SHALL 能经 (beam_id, station, N_low, sub_idx) 四元组唯一定位（可追溯性）。

## Global Constraints

- 全部约束继承 `00_master_spec.md` [S2]（GC1–GC7，尤其 GC1 束级切分、GC2 种子纪律、GC5 计数语义）。
- 数据管线 SHALL 纯 CPU 可运行（line1 GPU 纪律延续：不得触碰 GPU）。
- 子采样 SHALL 无放回。
- 传输物理档固定 P1 线性（抖动变换亦为线性）；同束两站、同实验所有束之间物理语义一致（GC4）。

## Out of Scope

- 网络训练与推理（`30`）；指标与 N_eff 计算（`20`）；实验矩阵与门禁（`40`）。
- 空间电荷、CSR、wakefield、边缘场/二阶效应（P2/P3 物理档）。
- 束流生成（Twiss 或解析分布采样）——束群唯一为锚点粒子的抖动线性变换。
- 纵向 (t,p) 投影产出（列在管线中保留，主线不消费）。
