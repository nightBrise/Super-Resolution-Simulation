# 05_testing_spec.md Spec

- 版本: v0.1（草案，待审查）
- 目录: `docs/specs/`
- 角色: 定义项目测试策略——分层、目录结构、关键用例、里程碑绑定、失败处理与时间预算
- 依赖: `00`（总纲）、`20`–`90`（各模块 AC）
- 来源: 主 agent 骨架 + Qwen 3.8 Max 详细稿综合（2026-08-25 用户批准）

## 文档目的

本文件定义测试策略，将 `20` [S10]、`30` [S11]、`40` [S12]、`50` [S14]、`60` [S13]/[S14]、`70` [S9]、`80` [S8]/[S9]、`90` [S5] 中已存在的验收标准（AC）机械化为可重复执行的测试套件，并补上 spec 未覆盖的过程性防护（划分泄露、配置公平、种子派生）。测试策略是执行层契约，不修改任何冻结 spec 的 Claim。

---

## [S1] 设计原则（三条铁律）

1. **只断言协议、契约与不变量，绝不断言研究结果。** "先验是否有效"是研究问题（`00` [S2]），由实验回答；测试验证的是"判定流程本身正确"——三分类、一票否决、CI 计算在任何一种结果下都给出正确标签。禁止写出 `assert gain > 0` 这类测试。
2. **防泄露优先。** 本项目最致命的 bug 是信息泄露（先验含 `c_high`、标准化用测试集统计量、同一 `c` 的噪声实现跨划分）。每类泄露都有专门的"扰动不变性"测试（[S3] 中标 ★ 的用例）。
3. **快反馈环。** 单元测试纯 CPU、≤2 分钟；任何代码变更触发；任何训练/生成任务启动前跑 smoke。目标是把 bug 拦截在分钟级。

### Claims

- C1: 测试 SHALL 只断言协议/契约/不变量，SHALL NOT 断言研究结果（如禁止 `assert gain > 0`）。
- C2: 每种信息泄露类型 SHALL 有对应的扰动不变性测试（★ 用例）。
- C3: 单元测试 SHALL 纯 CPU 且单次 ≤2 分钟；任何训练/生成任务启动前 SHALL 跑 smoke。

---

## [S2] 测试分层与时间预算

| 层 | 名称 | 运行条件 | 触发时机 | 时间预算 |
|---|---|---|---|---|
| **L0** | 单元测试 | 纯 CPU，禁 GPU，torch 线程限 8 | 每次代码变更（agent 每次改 `src/` 后）| **≤ 2 min** |
| **L1** | 集成测试 | 纯 CPU 为主（网络前向可 CPU）| 每次代码变更（与 L0 一起）；每个模块完成后全跑 | **≤ 8 min** |
| **L2** | Smoke 测试 | 单卡 `cuda:0`，显存峰值 < 6GB | 任何训练/数据生成任务启动前；EXP 启动前 | **≤ 10 min** |
| **L3** | 里程碑验收 | CPU + 读已有 `results/` 产物（不重训）| 每个里程碑边界（M1–M6 退出判据的组成部分）| M1 ≤30min；M2 ≤2h；M3–M6 各 ≤20min |

- **常规全量（L0+L1+L2）硬上限 20 min，红线 30 min**——超时即视为测试套件退化，必须先优化测试本身再继续研究推进。
- L3 验收测试读的是实验已产出的 `results/` 与 `data/`，不重复消耗训练算力；唯一例外是 M1/M2 的生成统计类检查（网格收敛、W8 覆盖率、G0 探针），这是纯 CPU 生成，可用 112 线程并行。
- 资源纪律：L0/L1 不得触碰 GPU，保证与在跑训练零冲突；L2 只占 `cuda:0` 且启动前检查该卡空闲（训练双卡占用时 smoke 排队而非抢占）。

### Claims

- C1: 测试 SHALL 分 L0/L1/L2/L3 四层，时间预算按本节表执行；全量（L0+L1+L2）超过 30 min 红线视为套件退化。
- C2: L0/L1 SHALL 纯 CPU；L2 SHALL 单卡且显存峰值 <6GB；L3 SHALL 读产物不重训。

---

## [S3] tests/ 目录结构与关键用例

### 3.1 目录结构

```
tests/
├── conftest.py                     # TEST_MASTER_SEED=20260825 + 自洽性示例参数组 + tmp + markers
├── unit/                           # L0：纯函数、数学不变量（纯 CPU）
│   ├── test_sampling.py            # 20 [S9] 参数范围、压缩三态、W1–W8 掩膜
│   ├── test_fbeam.py               # 20 [S10] AC1–AC12、[S7] 标签
│   ├── test_fdeg.py                # 30 [S11] D1–D12、SNR_hf、退化等级
│   ├── test_fprior.py              # 40 [S12] AC1–AC14、★c_high 泄露防护
│   ├── test_loss.py                # 60 [S2] L_space/L_spec、FFT ÷N²、五倍频程分带
│   ├── test_upsample_normalize.py  # 50 [S8]/[S13]、60 [S3] 上采样与总强度归一化
│   ├── test_metrics.py             # 70 [S3]–[S5] 图像/物理/精细结构指标
│   ├── test_statistics.py          # 70 [S4] 一票否决、[S7] bootstrap/Wilcoxon/Holm/三分类
│   └── test_seeds.py               # 60 [S14] C4/C8 SeedSequence 派生
├── integration/                    # L1：模块间接口与数据契约
│   ├── test_pipeline_hlp.py        # f_beam→f_deg→f_prior 全链、写入分工
│   ├── test_dataset_builder.py     # 60 [S8]/[S14] 划分、manifest、★划分不跨 c、γ 挖块洞
│   ├── test_models.py              # 50 [S14] N1–N12：A/B/C 前向、残差基准、主干一致
│   ├── test_fairness.py            # 00 [S6] 约束4、60 [S11] 三方案配置等价
│   ├── test_train_loop.py          # 60 [S10]/[S12] 早停、日志、checkpoint 字段
│   └── test_eval_pipeline.py       # 70 [S9]、80 [S8] metrics.csv/summary.json 契约
├── smoke/                          # L2：GPU 快速管线验证
│   ├── test_smoke_generate.py      # 32 样本全链生成跑通
│   ├── test_smoke_train.py         # A/B/C 各 100 步微训练：损失下降、无 NaN、非负
│   └── test_smoke_eval.py          # 评估管线端到端产出合法 metrics.csv/summary.json
└── acceptance/                     # L3：里程碑验收（读产物，不重训）
    ├── test_m1_generators.py       # 20/30/40 全部 AC + 网格收敛（slow）
    ├── test_m2_dataset.py          # G0(a)(b)(c)、20k/2k/2k、manifest 三元组
    ├── test_m3_baseline.py         # EXP-01 判据 + M3 出口（A 物理误差中位数 ≤ τ）
    ├── test_m4_main.py             # EXP-02 统计产出完整性 + 公平性复核
    ├── test_m5_ablation.py         # EXP-03/04/07/08 配对与冻结权重契约
    └── test_m6_delivery.py         # 90 [S5] N1–N8、预注册对账、5 图
```

配套文件：
- `pyproject.toml`（新增 `[tool.pytest.ini_options]`）：注册 markers `unit / integration / smoke / acceptance / slow / gpu / m1…m6`；默认 `addopts = -m "unit or integration" --strict-markers --timeout=120`。
- `environment.yml`：pip 段追加 `pytest>=8`、`pytest-xdist`、`pytest-timeout`；`scripts/check_env.py` 的 `EXPECTED` 增加 pytest 版本。
- `.gitignore`：追加 `studies/line1_substitute_sr/reports/test_reports/`。

### 3.2 ★ 防泄露关键用例（最高价值）

| 用例 | 断言 | 对应 |
|---|---|---|
| `test_c_high_invariance`（★★）| 固定 c_low+c_mid，分别改 a₃/γ/b₁ → P2 与 C 条件向量逐位不变；**全项目最重要的一条测试** | 40 [S9] C3、00 [S6] 约束3 |
| `test_c_high_not_used`（★★）| 修改批次数据中的 a₃/γ/b₁ → A/B/C 输出逐位不变 | 50 [S14] N8、[S11] C4 |
| `test_no_c_cross_split`（★★）| c 指纹（全参数圆整元组）在 train/val/test_id/test_pb 两两不相交 | 60 [S8] C2 |
| `test_gamma_block`（★）| test_pb 全部 \|γ\|∈[0.3,0.4]；train/val/test_id 无一样本落块内；用总体分位数常量 | 60 [S8] C4 |
| `test_fft_normalization`（★）| fft2÷N² 齐次性（L_spec(a·Ĥ,a·H)==a·L_spec）；未÷N² 时量级差 N² 倍 | 60 [S2] 实现约定 |
| `test_no_self_random_source`（★）| monkeypatch 禁用全局 np.random/torch 全局态 → 生成函数仍正常（随机源全来自 SeedSequence 分支）| 60 [S14] C4 |
| `test_normalize_no_leak`（★）| val/test 标准化复用训练集 μ/σ（传入不同划分返回同一统计量对象）| 60 [S5] C3 |

### 3.3 其他关键用例（摘要）

- **test_sampling**：固定种子抽 2000 组参数落入范围；α==(C−1)/a₁ 精确；三态占比各 1/3±5pp；8 组各仅违反一条掩膜的反例逐条被拒。
- **test_fbeam**：H 尺寸 256²、H≥0、ΣH≈1；c 单参数扰动 ε 连续（ε 减半变化量近似减半）；m 标签与 H 独立重算一致（h_eff==C_zδ/σ_z²、ε_z==√(σ_z²σ_δ²−C_zδ²)、I(z)==∫H dδ）；精细结构非随机（高频空间分布两次生成一致）。
- **test_fdeg**：L==max(0,L_clean+n) 且固定种子可复现；块平均保总强度；棋盘格抗混叠（★ 证明走 Blur→Down 而非抽稀）；SNR==mean(L_clean)/σ_n 精确；D1/D2/EXP-03/EXP-04 配置符合 [S7] 表。
- **test_loss**：4×4 fixture 手算 L_space 精确；Ĥ==H → L_space==0 且 L_spec<1e-12；5 带掩膜互不相交且并集==所有 f≤f_N 像素；λ 恒 1.0 且构造器拒绝修改；训练损失模块不导入 L_moment/L_marginal/L_forward。
- **test_metrics**：Ĥ==H → 全部指标理想值；输入 Ĥ'=3Ĥ 与 Ĥ 指标一致（指标前强制归一化）；解析高斯密度上 σ_z/σ_δ/h_eff/ε_z 与手算一致；DoG σ_outer 由 exp(−2π²σ²f_c²)=0.5 反算一致；R_E 四分支（真实恢复/纹理幻觉/过度平滑/如实报告）分类正确。
- **test_statistics**：bootstrap 固定种子可复现、重采样 10000 次；常数序列 d≡0.03 → CI=[0.03,0.03]；三分类（显著正/等效/显著负）标签正确且"等效"≠"无增益"；Wilcoxon 确为配对符号秩；Holm 手算一致；一票否决四分支（物理幻觉失效/噪声波动/部分失效/局部失效）标签正确；预注册常量（τ=0.05、触发率 0.20、主指标 ε_high^mask）从配置读取且与 config.yaml.template 一致。
- **test_models**：三方案输出 (B,1,256,256)、Ĥ.min()>0；残差基准（hook 置零末层 → Ĥ_A==Softplus(L_up) 等）；A/C 输入第二通道恒零；主干逐 named_modules 对比一致（48/96/192/384/384、每级 2 残差块）；FiLM 激活（两种 c_prior → 输出不同 + 梯度范数>0，防 G1"FiLM 未激活"）；无 bf16。
- **test_dataset_builder**：小规模 400 样本；manifest 含三元组/块信息/标定值；HDF5 字段 dtype float32 + gzip4 + sample_id 唯一；重生成逐位一致；EXP-03/04 sample_id 集 == test_id∪test_pb 且 H 逐位同；版本递增；并行（多进程 SeedSequence）与串行逐位一致。
- **test_fairness**：同 EXP 的 A/B/C config.yaml 逐字段比对，白名单仅 scheme/seed_index/run_tag；种子计数（代理 3 / 全量 2）。
- **test_train_loop**：日志含 train loss/output min/max/output sum/config hash/data version/spec version；checkpoint 随附配置/种子/数据版本；早停 mock val 序列连续 10 次无改善触发、不早于预算 50%。
- **test_eval_pipeline**：metrics.csv 逐样本含全部指标 + c_high + sample_id；summary.json 含三组增益；test_id/test_pb 分开计算不合并；评估强制 best_val.ckpt。

### Claims

- C1: tests/ SHALL 按本节目录结构组织（4 子目录 + 13 文件 + conftest + pyproject markers）。
- C2: 本节 ★ 防泄露用例 SHALL 全部实现并通过（c_high 不变性 / 划分不跨 c / 无自选随机源等）。
- C3: 测试断言 SHALL 与本节关键用例一致，覆盖各模块 AC。

---

## [S4] Smoke 测试定义（L2）

**目的**：在任何正式训练/生成启动前，10 分钟内回答"当前 commit 的管线端到端能跑、数值健康"。对应 `80` [S2] C3 的自动化兜底。

**test_smoke_generate.py**：全链生成 32 样本（Level 1、D2 初始值）——断言：全部过 W1–W8；SNR_hf 逐样本可计算；总耗时 ≤60s（生成性能回归哨兵）；落盘 8 张样图到 `debug/`（生命周期按 `60` [S15] 15.6）。

**test_smoke_train.py**：微训练——256 样本在线生成（或缓存），代理变体 C₀=24（smoke 非正式实验，允许代理配置，须在测试内显式声明，不违反 `50` [S7] C6），batch 4，A/B/C 各 100 步，单卡 cuda:0：
1. 三方案 forward/backward 可执行；
2. 损失有限（无 NaN/Inf，权重/损失/输出三处检查）；
3. mean(loss[后50步]) < mean(loss[前50步])（下降趋势，不要求收敛水平）；
4. Ĥ ≥ 0 且形状 256×256；
5. checkpoint 与 seeds.json 落盘成功。

**test_smoke_eval.py**：用 smoke 权重（或随机权重）对 16 样本跑评估——metrics.csv/summary.json 生成且通过 schema 校验；同时计算 L_up 零学习退化基线（G1(b) 依赖）。

**GPU 纪律**：启动前检查 cuda:0 空闲（有训练占用则排队）；显存峰值 <6GB；双卡训练期间禁止并发跑 smoke。

### Claims

- C1: 任何训练/数据生成任务启动前 SHALL 跑 L2 smoke；smoke 失败 SHALL 阻塞任务启动。
- C2: smoke 断言 SHALL 按本节（损失下降趋势、无 NaN、输出非负、落盘成功）。

---

## [S5] 里程碑绑定（测试全绿 = 退出判据组成部分）

| 里程碑 | 必须通过的测试 | 说明 |
|---|---|---|
| **M1 生成器** | 全部 unit/ + test_pipeline_hlp + acceptance/test_m1 | 20 [S10] AC1–12、30 [S11] D1–D12、40 [S12] AC1–AC14 全量；含网格收敛（slow：512 网格渲染与 256 结构一致）|
| **M2 数据集** | test_dataset_builder 全量 + acceptance/test_m2 | G0 三项测试化：(a) ≥2000 候选 W8 比例 ≥60%；(b) 探针法 3 参数 ×200 样本 min(s_x)<0.5（slow，CPU 并行）；(c) SNR_hf 批量中位数 <0.1；探针集与训练集 sample_id 不相交；20000/2000/1000/1000 计数与三元组 manifest |
| **M3 基线** | smoke/ 全部 + acceptance/test_m3 | 读 EXP-01 产物：A/B/C 损失下降、Ĥ≥0、形状正确（M3 退出）；**ε_z、I_peak 相对误差中位数作为诊断量记录（不断言阈值，批次二十一 Z2 与铁律 1 一致）**；σ_K/σ_n/σ_smooth 已写入 config 且 99 有登记；R_E(D2)/R_E(D1)≤60% 比率门可计算并被执行 |
| **M4 主实验** | 全部 unit/+integration/ 回归 + acceptance/test_m4 | 读 EXP-02 产物：三分类统计产出完整（Wilcoxon p、bootstrap CI、均值/中位数、标签）；主统计仅在 test_id、test_pb 分报；6 run 的 config 公平性复核；seeds.json 每方案恰 2 种子；评估用 best_val.ckpt。**任何三分类标签均合法**——测试只验协议不验结论 |
| **M5 消融** | acceptance/test_m5 | EXP-03/04 h5 配对（H 逐位同、L 变）；EXP-07 复用 EXP-02 B 权重且输入 P1；EXP-08 K=8 同 H 同退化参数仅噪声种子不同；报告字段含 R_E 联合判读分类、一票否决四分支标签、过冲/平滑型分类 |
| **M6 交付** | acceptance/test_m6 | 90 [S5] N1–N8：五类资产存在性与 schema；**N8 预注册对账**——从 final_report.md 与各 config.yaml 提取 λ、τ、触发率、主/次指标、ρ=0.1、CI 宽度 5%、MDE 5%、扩集上限与预注册常量一致；5 图 PNG+PDF 成对存在；metrics.csv 含 c_high 列；summary.json 含三组增益 |

**绑定规则**：里程碑退出 = 对应门禁判据（`80` [S9]）∧ 本表测试全绿。测试不全绿时 `00` [S7] C1 生效——不得进入下一里程碑。

### Claims

- C1: 里程碑退出 SHALL 要求本表对应测试全绿；测试不全绿 SHALL NOT 进入下一里程碑。

---

## [S6] 测试自身的可复现性

1. **固定种子**：conftest.py 定义 TEST_MASTER_SEED = 20260825；所有随机测试（生成、bootstrap、权重初始化）从该种子经 SeedSequence.spawn 派生，禁止测试内裸 np.random 调用。
2. **确定性纪律**：测试不得依赖时间、网络、字典迭代顺序；统计断言用确定性构造数据（首选）或固定种子 + 宽边界，容差推导写入测试 docstring。
3. **结果记录位置**：每次套件运行输出到 `studies/line1_substitute_sr/reports/test_reports/<UTC时间戳>_<commit8>/`（junit.xml + 运行摘要 + 逐用例耗时），目录入 .gitignore；progress.md 与阶段报告引用最近一次路径。
4. **测试代码同受版本纪律**：tests/ 入版本控制；测试变更的 commit footer 引用对应 spec 章节；冻结期测试断言的放松/删除须在 99 登记。
5. **环境**：测试只依赖 environment.yml 已锁依赖 + pytest 系；check_env.py 增加 pytest 版本核对。

### Claims

- C1: 测试 SHALL 使用固定 TEST_MASTER_SEED=20260825 派生，禁止裸随机调用。
- C2: 测试结果 SHALL 记录于 `studies/line1_substitute_sr/reports/test_reports/`（junit.xml + 摘要），目录入 .gitignore。
- C3: tests/ SHALL 入版本控制；测试断言变更 SHALL 在 99 登记。

---

## [S7] 失败处理：测试失败 → 风险登记 → 失败路径

### 7.1 分类与动作

| 类别 | 判据 | 动作 |
|---|---|---|
| **A. 实现缺陷** | L0/L1 断言失败，且对应 spec 条款明确 | 立即修复，修复后回归全层；未修复禁止启动生成/训练/里程碑推进。同一点修复 ≥3 次仍失败 → 按 `00` [S14] 升级 |
| **B. 契约/歧义失败** | 测试与 spec 表述无法对齐、或跨文档引用矛盾 | 先跑 check_spec_consistency.py；仍不清 → 第二级咨询并记录；冻结期不得改 spec，偏差记 99 |
| **C. 环境失败** | 依赖版本、GPU 不可用 | check_env.py（仅警告）；GPU 缺失时 gpu 标记用例跳过并显式标注，不得静默通过 |
| **D. 门禁验收失败** | L3 acceptance 失败 | 即对应门禁失败，进入 `80` [S9] 失败路径，按 `80` 附录 A 五级优先级选路，决策依据写入阶段报告第 4 节 |

### 7.2 与 R1–R9 衔接（测试即检测信号）

| 风险 | 测试充当检测信号 | 触发后流程 |
|---|---|---|
| R1 训练不收敛 | test_smoke_train（损失 NaN/不降）| G1 失败路径三选一 |
| R2 全量发散/NaN | test_smoke_train NaN 检查为预哨兵；修复后必须重跑全套 L0+L1+L2 | R2 预授权流程 + r2-fix 分支 |
| R3 任务太易 | test_snr_hf_function、M2 SNR_hf 中位数、M3 R_E 比率门 | G0/G1 + 99，上调 σ_K |
| R4 先验无效 | **不由测试触发**（研究结果）；test_statistics 保证判定流程正确 | G1/G2 失败路径 + 90 阴性模板 |
| R5 OOD 崩溃 | test_exp03_04_pairing、M2 OOD 子集检查 | G4 路径 |
| R6 一票否决触发 | test_veto_four_branches 保证标签与报告字段正确 | τ/20% 冻结，仅报告 |
| R7 数据质量不达标 | test_sampling（W8 快检）、M2 G0(a)(b)(c) | 回数据生成修正；未过不得进训练 |
| R8 GPU 利用率不足 | 不属于测试职责（训练期 gpu_utils.log）；测试仅保证不占训练卡 | 99 登记，不视为失败 |
| R9 标定失败 | M3 的 R_E 比率门、σ_n 尾部 SNR 2–5、σ_smooth 能量预算 [0.5,0.95] | 各标定出口路径 |

### 7.3 无人值守语义

- 类别 A/C：agent 自主修复/处置并记录，不暂停。
- 类别 B（第二级）：写咨询记录后继续，不暂停。
- 类别 D + `80` 附录 A.2 触发条件（路径走完仍失败 / 触不可变核心 / 研究方向）：按 `00` [S14] 第三级暂停并升级用户。**测试失败本身不是暂停条件；测试失败阻塞的是里程碑推进，不是报告。**

### Claims

- C1: 测试失败 SHALL 按 7.1 四分类处置；类别 D 进入对应门禁失败路径。
- C2: 类别 A/C 自主处置不暂停；类别 B 咨询后继续；类别 D + 触不可变核心 SHALL 暂停升级用户。

---

## [S8] 落地步骤（给实现 agent 的执行清单）

1. `99` 登记 Proposed：测试策略（本文件）+ environment.yml 增加 pytest 依赖 + `00` [S13.2] 增列第 7 项必检"对应层测试套件通过"。
2. 建 tests/ 骨架与 conftest.py、pyproject.toml markers、.gitignore 条目（studies/line1_substitute_sr/reports/test_reports/）。
3. 按里程碑顺序实现：M1 前完成 unit/（test_sampling/fbeam/fdeg/fprior/loss/upsample）+ test_pipeline_hlp；M2 前完成 test_dataset_builder + test_seeds；M3 前完成 test_models/test_train_loop/test_fairness + 全部 smoke/；M4 前完成 test_statistics/test_metrics/test_eval_pipeline；各 acceptance/ 文件在对应里程碑前一个迭代内完成。
4. 每个测试文件头部注明覆盖的 spec 章节与 Claim 编号（与 spec_claim_index.md 对齐）。
5. 首次全套运行后，把实测耗时写入 [S2] 时间预算表替换估计值，并在阶段报告第 7 章记录取舍（`00` [S13.3] C3）。

### Claims

- C1: 测试套件 SHALL 按本节顺序随里程碑实现；每个测试文件 SHALL 注明覆盖 spec 章节。

---

## Global Constraints

- 不改任何冻结条款：所有断言源自 `20` [S10]、`30` [S11]、`40` [S12]、`50` [S14]、`60` [S13]/[S14]、`70` [S9]、`80` [S8]/[S9]、`90` [S5] 的既有 AC 与实现约定。
- 预注册常量（λ=1.0、τ=0.05、20%、ρ=0.1、5%、主/次指标）在测试中只读不写；test_pre_registered_constants 与 M6 N8 对账双重守护（`00` [S10] 10.1 判据本体类）。
- 三方案公平、先验不泄露、总强度归一化、bilinear 上采样等不可变核心均有专属 ★ 用例。
- 测试策略变更走 `99` 普通变更流程；不触动 `00` [S10] 10.1 不可变核心。
