# line2_particle_sr 总纲与全局约束 Spec

> 研究线：粒子级相空间超分（line2_particle_sr）——4D 横向六投影图像接口放大（Phase 1）。
> 上游设计：`docs/research/line2_particle_sr.md` v0.7（设计冻结版，2026-09-13 用户逐项确认）；v0.2 修订 = 2026-09-14 审核 findings B1–B6/M1–M9/m1–m11 清零（用户批准：抖动传输束群 + 可验证放大上限 10×）。
> 版本：v0.2.1（**Frozen v1.0**，2026-09-14 用户批准冻结）。
> 本 spec 是 line2 全部 spec 的符号、布局、门禁与全局约束单一来源；其余 spec 引用本文件，不重复定义。

## [S1] 符号与定义

本节术语为 10/20/30/40 的唯一裁定来源。

- **锚点（anchor）**：`studies/line2_particle_sr/data/anchor/injector.sdds`——2×10⁵ 宏粒子表（实测 N = 200,001、Q = 500 pC、pc ≈ 130 MeV、ε_n ≈ 1.26 mm·mrad、σ_t = 2.41 ps、非高斯：xp/yp 峰度 ≈ +2.6）。单位契约：p = βγ（无量纲动量，elegant 惯例）；x, y [m]；xp, y′ [rad]；t [s]。锚点 MD5 = `f39d905f96abd5588091dad6dd3884b2`（9,600,397 字节）。
- **束（beam）**：锚点全量粒子表经**束级抖动线性传输**得到的实现——束 i 先经过一段从预注册抖动分布抽取的线性变换（FODO 前置：四极强度相对标称 ±5%、薄透镜踢角、漂移长度微扰；分布定义见 `10` [S3]），再过共同 FODO；束 0 为无抖动基准束。束群规模 L = 1000（train 700 / val 150 / test 150）。**推广域声明**：全部束共享同一底层 6D 分布（锚点），多样性来自确定性线性变换——Phase 1 方法的推广声明限于该邻域；真实前端晶格的跨域泛化属 Phase 2。
- **站（station）**：站 0 = 抖动变换后、FODO 入口快照；站 1 = FODO 出口快照。
- **子样本（subsample）**：从某站全量粒子表（2×10⁵ 颗）**无放回**随机抽取的 N_low 颗子集；子样本与其来源束构成配对（A ⊂ B）。
- **图组（image group）**：某站、某 N 的六张 128×128 投影计数图，通道顺序恒为约定枚举（见 C2）。
- **N_low**：输入端粒子数，训练档 ∈ {3×10³, 1×10⁴, 1×10⁵}（深挖工况 1×10⁴）；曲线另含 {1×10³, 3×10⁴} 档（`20` [S2]）。**N_high**：目标端粒子数，= 2×10⁵（锚点全量，亦为参照）。
- **收敛密度**：同一束流 N→∞ 时的期望直方图；操作化参照 = 该束 2×10⁵ 全量直方图（参照自身相对收敛密度的残余噪声构成 N_eff 高端的测量地板，见 `20` [S2]）。
- **等效放大因子 N_eff**：在预注册度量（`20` [S1] M1，含总计数归一）与裸误差–N 曲线（`20` [S2]，域 {1e3, 3e3, 1e4, 3e4, 1e5}）上，使独立 N′ 子样本图组的误差等于网络输出误差的 N′（log-N 线性插值；定义与协议见 `20` [S3]）。报告域 [1e3, 1e5]；N_eff ≥ 1e5 记上溢（语义 = 达到可验证域顶）。**可验证放大上限 = 10×**（1e4 → 1e5；>1e5 部分受参照噪声地板限制不可分辨，20× 为锚点规模给出的叙事上限）。
- **六投影枚举**（恒定顺序）：`(x,x′) (y,y′) (x,y) (x,y′) (x′,y) (x′,y′)`。

### Claims

- C1: 10/20/30/40 中出现的跨 spec 术语（锚点、束、站、子样本、图组、N_low、N_high、收敛密度、N_eff、六投影）SHALL 均以本 [S1] 定义为准，且各 spec 不再重复定义（静态检查）。
- C2: WHEN 任何数据文件或代码需要标注投影顺序 THEN SHALL 使用 [S1] 的六投影枚举顺序，且该顺序 SHALL 由测试断言（投影轴对与约定元数据逐项比对）。
- C3: 束群构造 SHALL 可由锚点 + 种子链端到端重现（`10` [S3]），且任一束的抖动参数 SHALL 可由 (beam_id, master_seed) 唯一还原。

## [S2] 研究线布局

- 数据：`studies/line2_particle_sr/data/`（锚点于 `data/anchor/`，数据集于 `data/<data_version>/`）。
- 产物：`studies/line2_particle_sr/results/`（run 目录、曲线、summary）；报告：`studies/line2_particle_sr/reports/`。
- 代码：`src/line2/` 包——`data_pipeline/`（锚点/导入/子采样/传输/分箱/落盘）、`metrics/`（判据）、`training/`（模型/损失/训练/推理）、`experiments/`（矩阵运行与门禁脚本）。
- CLI 约定：`python -m src.line2.<module>...`；所有命令运行前先 `python scripts/check_env.py`。
- 依赖：`ocelot`（仅传输，P1 线性）、`sdds`（锚点导入）入 `environment.yml` 并经 check_env 校验。

### Claims

- C4: WHEN 按 [S2] 布局执行数据管线 THEN 全部产物 SHALL 落于 `studies/line2_particle_sr/` 下对应目录（测试断言路径前缀）。
- C5: `python -c "import ocelot, sdds"` SHALL 在 `sr-sim` 环境成功（check_env 校验项）。

## [S3] Phase 门禁

- **Phase 1 出口**：`40` [S3] 的三条门禁各输出其预注册枚举结论（H1/H8 为三值，H3 为四值）；三份结论 + 主曲线（方法可用域地图）构成 Phase 1 交付。H8 为阴性不阻塞 Phase 2 启动评估（如实报告即可）；H1 不成立则 Phase 2 需重议方向并按根级 `docs/specs/00_master_spec.md` [S14] 三级咨询升级用户。
- **Phase 2 进入条件**：Phase 1 交付完成 + 用户放行；入口 spec 为 L9（束流接口）。
- **Phase 3 进入条件**：L4、L5 完成 + 用户放行。

### Claims

- C6: 每条 Phase 门禁 SHALL 恰好对应 `40` 中一个可执行判定表达式（脚本消费落盘 summary，无人工干预）。
- C7: Phase 1 交付物清单（三条门禁枚举结论 + 方法可用域地图主曲线 + 报告）SHALL 可由文件存在性检查验证。

## [S4] spec 索引与变更规则

- 索引：`10_data_pipeline_spec.md`（锚点数据管线，L1）；`20_metrics_spec.md`（判据与 N_eff，L2）；`30_model_training_spec.md`（Phase 1 主模型，L3b）；`40_experiment_spec.md`（实验矩阵与判定，L4+L5+L7 图像级）。
- 变更规则：line2 变更日志为 `99_change_log.md`（本目录）；先记录后修改；冻结 spec 在实现期不得修改（歧义记入对应 `.decisions.md`）；一次只修改一个模块。
- Phase 2/3 的 spec（束流接口 L9、分辨率扩展 L8）不在本批次，届时另行冻结。

### Claims

- C8: 本目录 SHALL 存在 00/10/20/30/40 五份 spec 与 `99_change_log.md`，且 [S4] 索引与实际文件一一对应。
- C9: WHEN 修改任何 line2 spec THEN `99_change_log.md` SHALL 先于该修改包含对应记录（审计项：人工/脚本核对，非自动化断言）。

## Global Constraints

- **GC1 束级切分**：同一束的任何派生数据（任意站的任意 N 子样本图组）SHALL NOT 跨 train/val/test。
- **GC2 种子纪律**：一切随机性 SHALL 经 `SeedSequence(master_seed=20260913)` 派生（`spawn` 链：束 → 抖动参数/子样本/切分/训练），禁止模块级全局随机源；同 master_seed 重跑 SHALL 逐位可复现。
- **GC3 主指标准绳**：N_eff（`20` [S3]）为唯一主指标；模型选型、变体路由与早停只以 **val** N_eff 为准绳；test 集在调参期零接触、全程只评一次（评估调度唯一归 `40`）。
- **GC4 公平性**：同一实验内全部变体（独立/联合、不同种子）SHALL 使用相同数据集、相同切分、相同传输物理档、相同 max_steps 预算（= 50000，预注册于 `30` [S3]）。
- **GC5 单位与计数语义**：p = βγ；投影图元素为非负整数计数且 ∑counts = 该子样本粒子数；坐标归一化 SHALL 可逆。网络输出的比较语义（总计数归一）定义于 `20` [S1] M1。
- **GC6 禁止断言研究结果**：测试与门禁脚本 SHALL NOT 断言研究方向（如"网络优于基线"），只断言协议、不变量与判定表达式的机械执行。
- **GC7 可复现三元组**：每个数据集与 run 的 manifest SHALL 含 `data_version`、`code_version`（git HEAD 完整 40 位 hash）、`spec_version`（`v0.x+<99 批次>`）。

## Out of Scope

- Phase 2 契约：束流接口（粒子云渲染 L9）、接口对比（H3b）、束流级传输闭环、真实 SXFEL 前端晶格。
- Phase 3 契约：分辨率与比值扩展（L8）。
- 纵向 (z,δ)/(t,p) 重建；空间电荷、CSR、wakefield、边缘场/二阶物理档。
- 束流生成（Twiss 或其他分布采样）——束源唯一为锚点 SDDS 粒子的抖动线性变换（本 [S1]）。
- H6（受控档植入实验）：**挂起**——受控档随"只用真实 SDDS 粒子"移除，需要时按 99 流程另立参数化受控束。
