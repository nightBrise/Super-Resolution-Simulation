# AGENTS.md — 物理先验引导的纵向相空间超分辨率研究

> 本文件面向对此项目一无所知的 AI 编码代理，提供项目全貌、代码组织、运行命令与开发纪律。
> 研究契约以 `docs/specs/` 为准；git 流程约定以 `CONTRIBUTING.md` 为准；本文件只描述项目现状，不替代上述二者。

---

## 1. 项目概览

本项目是**替代验证（Substitute Verification）**范式下的配对监督超分辨率研究：用轻量级纵向相空间模拟函数（不依赖 elegant / Ocelot）生成严格配对的 `(H, L, P)` 数据，训练超分辨率神经网络，回答一个核心问题——**在低分辨率观测下，物理先验能否帮助网络更准确、更符合物理地恢复高分辨率纵向相空间？**

- 物理背景：直线加速器驱动的 FEL 束流诊断中，纵向相空间 `(z, δ)` 是判断束流品质的核心对象；TDS + 谱仪只能输出低分辨率观测，高分辨率下才有压缩折叠、细脊、电流尖峰等精细结构。
- 三方案公平对比（相同数据、相同损失、相同主干、相同训练配置、相同评估指标）：
  - **方案 A**（无先验 baseline）：`Ĥ = NonNeg(L_up + G0(L_up))`
  - **方案 B**（图像先验 + 残差，early fusion）：`Ĥ = NonNeg(P2 + G1(L_up, P2))`
  - **方案 C**（参数先验 + FiLM）：`Ĥ = NonNeg(L_up + G2(L_up | c_prior))`
- 训练目标：空域-频域混合损失 `L_total = L_space + λ·L_spec`，**λ 冻结为 1.0**（预注册常数，不可改）。
- 退化协议：256 → 64 下采样（r=4，块求和）+ 高斯模糊（σ_K）+ 高斯噪声（σ_n），非负截断 `L = max(0, L_clean + n)`。
- 坐标系：`z, δ ∈ [-1, 1]`，图像第 0 轴为 z、第 1 轴为 δ，像素中心采样（`src/generators/f_beam.py`）。

**当前进度（详见 `progress.md`，Agent 自动维护）**：Spec 集已冻结（v1.0+2026-08-26）；M1（基础模拟函数）、M2（数据集生成 + G0 门禁）完成；**M3 执行中**（EXP-01 标定重跑：01a 三方案重训完成 → 01b σ_K 复评 → 01c/01d → G1(a) 判定）；M4–M6 未开始。硬件为 2× Quadro RTX 6000（24GB，Turing CC 7.5，**不支持 bf16**）。

---

## 2. 技术栈与运行环境

- **语言/解释器**：Python 3.11（conda 环境 `sr-sim`，当前激活环境）。
- **核心依赖**（锁定于 `environment.yml`，版本被 `scripts/check_env.py` 校验）：
  - PyTorch 2.4.0、torchvision 0.19.0、numpy 1.26.4、scipy 1.13.0、h5py 3.11.0、scikit-image 0.24.0、matplotlib 3.9、pillow、scikit-learn、tensorboard、pytest≥8 + pytest-xdist + pytest-timeout。
- **无构建步骤**：纯 Python 包，所有命令以 `python -m src.<module>` 运行；测试以 `src.` 前缀导入生产代码（`tests/conftest.py` 将项目根插入 `sys.path`）。
- **环境管理**：
  - `conda env create -f environment.yml`（首次）或 `conda env update -f environment.yml`（增量）。
  - 改动依赖后必须同步 `environment.yml` 并跑 `python scripts/check_env.py` 验证（校验不通过仅警告，不阻塞，需登记 99）。

---

## 3. 仓库结构

```text
.
├── README.md / README.en.md        # 项目门面（中英双语）——注意：README 状态表已过期，以 progress.md 为准
├── CONTRIBUTING.md                 # 开发约定（GitHub Flow + Conventional Commits + R2 修复流程）
├── AGENTS.md                       # 本文件
├── config.yaml.template            # 实验配置模板（每个实验 config.yaml 的必填字段契约；顶层 study_root 指定研究线）
├── progress.md                     # Agent 维护的项目进度（每阶段更新一次，00 [S13.4]）
├── spec_claim_index.md             # scripts/check_spec_consistency.py 自动生成的全 spec Claim 索引
├── environment.yml                 # conda 环境锁定（sr-sim）
├── pyproject.toml                  # 项目元数据 + pytest 配置（markers、addopts）
├── .gitignore                      # data/、results/、*.pdf、*.h5、checkpoint、assets/ 等一律不入库
├── scripts/                        # 项目级工具：check_env.py / check_spec_consistency.py / remap_paths.py
├── src/                            # 生产代码（见 §4；跨研究线共享，路径经 study_root 解耦）
├── tests/                          # 四层测试（见 §7）
├── studies/                        # ★ 研究线：每条研究线自含 data/results/reports/drafts
│   ├── line1_substitute_sr/        # 当前研究线（替代验证超分，已迁移）
│   │   ├── data/<版本>/            # 数据集（train/val/test_id/test_pb/test_ood.h5 + manifest.json）
│   │   ├── results/
│   │   │   ├── summary/            # 判定证据（summary*.json / pooled / test_id_combined.csv / 预注册骨架）
│   │   │   ├── run/                # 各 run 证据（config/summary/metrics/logs；ckpt 不在内）
│   │   │   └── assets/             # 本线图（figure_*.png，PNG only 300dpi）
│   │   ├── reports/                # 阶段报告（M1/M2 stage_report、test_reports）
│   │   └── drafts/                 # 草稿区（spec/notes/scratch/figs，未定稿）
│   └── line2_<方向>/               # 未来研究线（空模板）
└── archive/                        # ★ 冷存储（只移动不删除）：可再生大文件 + 已完结研究线
    └── line1_substitute_sr/
        ├── checkpoints/  predictions/  data_dev/  misc_runs/  scripts_tmp/
└── docs/
    ├── specs/                      # Spec 集（00–99，十位留空编号；v1.0 冻结，99 为活跃变更日志）
    ├── reports/                    # 终版/跨线报告（line1_substitute_sr_final_report.md、M4_stage_report.html）
    └── wang2026_*.pdf              # 参考文献（本地保留，.gitignore 排除，不入库）
```

**研究线组织说明**：代码/脚本/spec 为跨研究线共享层（`src/`、`scripts/`、`tests/`、`docs/specs/`，保留根级）；数据与实验产物按研究线归置（`studies/<line>/`），每条线自含 `data/results/reports/drafts`。研究线根由 config 顶层 `study_root` 字段指定（非空走 `studies/<line>/`，空/缺省兜底项目根——向后兼容）。可再生的中间大文件（ckpt、预测 npz、开发版数据、一次性脚本）移入 `archive/<line>/` 冷存储（只移动不删除）。研究线之间互不干扰，换方向开新线即可。

---

## 4. 代码组织与模块职责

代码按 spec 模块组织，每个模块文件头部注明覆盖的 spec 章节与 Claim：

| 模块 | 文件 | 职责 |
|---|---|---|
| 生成器 | `src/generators/f_beam.py` | 20 规格：物理生成 `f_beam(c) → (H, m, c)`；基础密度→压缩折叠映射→光滑渲染→总强度归一化；`σ_smooth,H = 0.125×w_fine` 逐样本（2026-08-26 P0 修订） |
| | `src/generators/masks.py` | 20 [S9]：有效域掩膜 W1–W8 检查、`w_fine` 精细结构宽度 |
| | `src/generators/sampling.py` | 20 [S9]：压缩三态联合采样 + W1–W8 拒绝筛选；γ 块条件采样；OOD 极端参数采样 |
| | `src/generators/f_deg.py` | 30 规格：退化 `f_deg(H; d) → (L, L_clean, d, m_L)`；模糊→块求和下采样→噪声；SNR_hf |
| | `src/generators/f_prior.py` | 40 规格：图像先验 `f_prior(c, level) → (P, meta)`；P2 去掉 c_high，`σ_smooth,P > σ_smooth,H` |
| | `src/generators/image_ops.py` | 50/60：4 倍双线性上采样（scipy zoom, grid_mode）+ 总强度归一化 |
| | `src/generators/dataset_builder.py` | 60 [S8]/[S14]：划分协议、HDF5 落盘、manifest、种子派生、G3 掩膜复核、版本对账（code_version 一律取完整 40 位 git HEAD） |
| | `src/generators/build_dataset.py` | 数据集生成 CLI |
| | `src/generators/probe.py` | 80 [S9] G0：受控探针法（差分高频存活比 ρ）与 G0 三判据评估 |
| 模型 | `src/models/unet.py` | 50 [S7]：残差 U-Net 主干（5 级、通道 48/96/192/384/384、瓶颈封顶 384、FiLM 可选） |
| | `src/models/schemes.py` | 50：方案 A/B/C 组装；`Ĥ = Softplus(S·Base + R)`（工作尺度 S=N²=65536）；FiLM 注入瓶颈+解码器 |
| 训练 | `src/training/loss.py` | 60 [S2]：HybridLoss（空域 L1 + 五倍频程分带谱 L1，FFT÷N²，λ 冻结 1.0） |
| | `src/training/train.py` | 60：训练 CLI + 早停 + 日志 + 哨兵（Q 比/Pearson ρ/L_space 地板）+ DDP |
| 评估 | `src/evaluation/metrics.py` | 70：图像/物理/精细结构指标、统计判定（bootstrap/Wilcoxon/Holm/三分类）、一票否决 |
| | `src/evaluation/evaluate.py` | 70/80：评估 CLI，产出 metrics.csv（长表）+ summary.json |
| | `src/evaluation/infer.py` | 推理 CLI，产出 predictions_<split>.npz |
| | `src/evaluation/plots.py` | 可视化（2D 相空间/1D 剖面/残差，PNG+PDF 成对） |
| 工具 | `src/utils/config_utils.py` | config 解析、config_digest、输出目录命名、设备/精度解析（bf16 拒绝） |
| | `src/utils/h5data.py` | H5Dataset（训练/评估/推理共用读取）、方案 C 参数先验预处理（log+z-score） |
| | `src/utils/checkpoint.py` | checkpoint 与 seeds.json 读写 |

**注意**：`60_training_spec.md` §15.1 文件树中提到的 `src/generators/calibration.py`（EXP-01b/c/d 标定）与 `scripts/check_test_state_consistency.py`（99 [S5] 规则 7）**尚未实现**；EXP-01b 标定目前以临时脚本（`scripts_tmp/`）与既有模块完成。`config.yaml.template` 头部注释提到的 `src/utils/write_config.py` 不存在，config 写入由各 CLI 负责。

---

## 5. 数据管线（M2）

- 划分协议（60 [S8] C2/C4）：train/val/test_id 在 γ 块外采样（拒绝 `|γ| ∈ [0.3, 0.4]`），test_pb 在块内且与 test_id 1:1，test_ood 为 EXP-06 极端参数（β/γ 幅度×1.5、豁免掩膜、固定 500 样本）。γ 块边界由固定总体分位数导出（`GAMMA_BLOCK = (0.3, 0.4)`），**不得用经验分位数**。
- 种子派生（60 [S14] C4）：`seed_i = SeedSequence(master_seed).spawn(8)[split_branch].spawn(n)[i]`；并行/串行逐位一致；生成器不自选全局随机源。
- HDF5 字段（每划分一个文件）：图像 `H`(256²)/`H_neg_ch`(c_high 清零版，评估成分分解用)/`L`(64²)/`L_clean`/`L_up`/`P2`（float32、gzip level 4、按样本切分）+ 参数组 `c_low/c_mid/c_high` + 物理标签 `m` + 退化元数据 `m_L` + `masks` W1–W8 + `seed_i` + `sample_id`（`<划分>-<序号>`）。
- `manifest.json`：data_version、code_version（完整 40 位 hash）、spec_version（`v1.0+<99 最近批准批次>`）、master_seed、划分样本数、掩膜统计、G3 复核。
- 现有数据版本：`v1`（全量 20k/2k/1k/1k+500）、`dev1`（2k/500/250/250+500，σ_K=11.0）、`dev2`（σ_smooth,H=0.125× 修订后重生成）、`dev2_d1`（D1 档 σ_K=5.185，EXP-01b 标定用）。

---

## 6. 训练与评估

- **训练协议**（60）：AdamW（lr 3e-4、wd 1e-4）；损失在工作尺度空间计算（`target = H×S`，`Ĥ = Softplus(S·Base+R)`，评估时 ÷S 还原——**双空间契约**，`work_scale` 为 config 与 checkpoint 必填键，加载时断言一致）；早停 patience=10、不早于 50% 步数预算；方案 C 的标准化统计量只从训练集计算（★ 防测试集泄漏）；验证哨兵（80 [S4] C3）：Q_Ĥ/Q_H 中位数 ∈[0.1,10]、Pearson ρ 中位数 ≥0.1、val L_space < 1/N²，best_val.ckpt 与哨兵通过联动。
- **评估协议**（70）：所有指标在总强度归一化到 1 后计算；主指标 ε_high^mask（预注册）、次指标 ε_z 相对误差；三分类（显著正/等效/显著负）+ bootstrap 95% CI + Holm 校正（仅主+次指标）+ 一票否决四分支；R_E 守卫（>10 标注"锐化伪影档"，不否决）；掩膜成分分解（ch_in/b_in）与先验泄漏指数 Π_leak。
- **CLI 接口**（60 [S15] 15.7 统一约定；三方案由 config 内 `scheme` 字段指定；输出目录默认 `results/<EXP>_<arm>_<seed>_<run_tag>[_<config_tag>]`；评估默认读 `best_val.ckpt`；所有命令启动前先跑 check_env.py）：

```bash
python -m src.generators.build_dataset --config <path> [--split train|val|test_id|test_pb|test_ood] [--workers N]
python -m src.generators.probe --config <path>          # G0 受控探针法 → g0_report.json
python -m src.training.train --config <path> [--smoke] [--steps N] [--out <dir>]
python -m src.evaluation.evaluate --config <path> --split <test_id|test_pb|test_ood|exp03|exp04> [--checkpoint <path>]
python -m src.evaluation.infer --config <path> --split <...> --out <dir>
python -m src.evaluation.plots --config <path> --predictions <npz> --out <dir>
```

---

## 7. 构建与测试命令

```bash
# 环境校验（仅警告，不阻塞）
python scripts/check_env.py

# Spec 跨文档一致性检查（抽 Claim + 校验引用，重写 spec_claim_index.md；--strict 使 broken refs 非零退出）
python scripts/check_spec_consistency.py [--strict]

# 默认测试套件：L0 单元 + L1 集成（纯 CPU，05 [S2]：L0 ≤2min，L0+L1 全量 ≤8min）
python -m pytest

# 各层显式运行（markers 在 pyproject.toml 注册，--strict-markers 强制）
python -m pytest -m "unit or integration"     # L0+L1（默认）
python -m pytest -m smoke                     # L2 冒烟：单卡 cuda:0、显存<6GB；卡被占用时跳过而非抢占
python -m pytest -m acceptance                # L3 里程碑验收：读 results/、data/ 产物，不重训
python -m pytest -m "m1 or m2"                # 按里程碑绑定筛选
python -m pytest -m slow                      # 长耗时（如网格收敛）
```

**测试现状（2026-08-26 实测）**：共收集 219 个用例（unit 119 / integration 57 / smoke 2 / acceptance 41 / slow 2 / gpu 2），默认套件 43 个 deselected。`python -m pytest`（L0+L1）实测 **174 passed, 1 failed, 1 skipped**（124.6s）。

**已知失败（必须了解）**：`tests/integration/test_dataset_builder.py::test_manifest_triple_and_sections` 第 95 行断言 `manifest["code_version"] == "test"` ——该断言与 N4 修订（00 [S6] 约束 8 N4：`build_dataset` 一律以生成时 git HEAD 完整 40 位 hash 作为 code_version，忽略 config 中的旧值，commit 1fa3949/7344cf4）矛盾，属**过时测试未同步**，非实现缺陷。修复需按 05 [S6] C3 在 99 登记测试断言变更后改断言（如改为 `== git_head()` 且长度为 40）。修复前默认套件非全绿，推进里程碑前应先处置。

**1 skipped**：`tests/integration/test_eval_pipeline.py:118`（旧格式产物缺失时跳过新字段契约检查，M3 重生成后自动生效）。

**测试纪律**（05 [S6]）：固定 `TEST_MASTER_SEED = 20260825`（`tests/conftest.py`），所有随机测试经 SeedSequence 派生，禁止裸随机调用；测试结果记录到 `studies/line1_substitute_sr/reports/test_reports/`；测试只断言协议/契约/不变量，**禁止断言研究结果**（如 `assert gain > 0`）。

---

## 8. 开发约定（CONTRIBUTING.md 摘要）

- **Git 工作流**：GitHub Flow——仅 `main` 分支，无 develop/release 分支；分支命名 `<type>/<desc>`（`feat/ fix/ docs/ refactor/ test/ chore/ experiment/`），项目特殊前缀 `r2-fix/<YYYY-MM-DD>-<N>`（实现级重跑修复）；PR ≥1 approve、squash merge、禁 force-push、合并后删分支。
- **Commit**：Conventional Commits（`<type>[scope]: <description>`），scope 取模块名（`model-a/model-b/model-c/data-gen/degradation/prior/train/eval/docs/spec`）；footer 必须关联 issue/PR/spec 行号/99 变更日志（如 `Refs spec/50[S10]`、`Refs 99:R2-2026-09-15`）。
- **可复现性三元组**：每个实验 `results/<EXP>/config.yaml` 必含 `code_version`（完整 40 位 git hash）/`data_version`/`spec_version`，与 `final_report.md` 对账（90 [S5] N8）。
- **禁止**：未经许可直接 push main、force-push、直接修改 spec 不登记 99。

---

## 9. Spec 治理与 Agent 工作模式（00 [S9]/[S10]/[S13]/[S14]）

- **Spec 是研究契约**：`docs/specs/00_master_spec.md` 是唯一全局入口（总纲、符号、里程碑、变更规则）；子 spec 编号 `00/10/20/.../99`（十位留空，可插入子模块），`99_change_log.md` 是活跃变更日志 + 未决问题清单。
- **变更规则**：修改任何 spec 前必须先登记 `99` 为 `Proposed` → 批准 `Approved` → 修改 → `Implemented`；一次只改一个模块；**冻结 spec 在实现阶段不得修改**（触不可变核心 `00 [S10] 10.1` 属重大需求变更，走新 spec 流程）。
- **不可变核心**（冻结后不得动）：研究问题与三方案框架、公平性约束、数据契约、评估判据与预注册阈值（τ=5%、触发率 20%、λ=1.0、主/次指标）。可修改细节（通道宽度、训练超参、标定取值等）走 99 普通变更流程并保持三方案一致。
- **Agent 执行模式**（00 [S13]）：每阶段前完成 [S13.2] 必检（产物落盘、config 三元组、spec 通读、跨文档一致性、标定值、阶段报告、测试通过）；每阶段后按 [S13.3] 模板生成 `results/<EXP>/stage_report.md`（必含第 7 章"设计理由 Why"）并更新 `progress.md`；疑问按 [S14] 三级咨询（自主解决 → 更强模型/子 agent → 用户介入）。
- **预授权变更类**（[S14.2]，agent 可自批准并登记 99 注明"预授权类"）：标定采用值登记、引用/笔误修正、模板默认值填充、不改判据本体的阈值落地。仍强制升级用户的：触不可变核心、研究方向、R2 二次修复、M6 出口决策。
- **里程碑门禁**：M0（冻结，完成）→ M1（生成器 + AC，完成）→ M2（数据集 + G0，完成）→ M3（方案 A baseline + G1(a)，执行中）→ M4（主实验 + G2）→ M5（消融 + G3）→ M6（final_report + 出口决策）。前一里程碑未过不得进入下一里程碑。

---

## 10. 安全与注意事项

- **版本控制边界**：`data/`、`results/`、`*.pdf`（参考文献本地保留）、`*.h5`、`*.ckpt`、`debug/`、`final_report.md`、`assets/` 均被 `.gitignore` 排除，不得手工加入版本控制。
- **GPU 纪律**（05 [S2]）：L0/L1 测试不得触碰 GPU（与在跑训练零冲突）；L2 smoke 只占 `cuda:0` 且启动前检查该卡空闲（显存 >50% 空闲），双卡训练期间禁止并发 smoke；本机 GPU 为 Turing CC 7.5，**不支持 bf16**（配置 bf16 直接拒绝启动）。
- **防泄露**（全项目最高优先级）：先验不得含 `c_high`；方案 C 标准化统计量只来自训练集；同一 `c` 的噪声实现不得跨划分。相关 ★ 测试（`test_c_high_invariance`、`test_c_high_not_used`、`test_no_c_cross_split`、`test_no_self_random_source` 等）是全项目最重要的测试。
- **临时文件**：`scripts_tmp/` 为归档参考，不参与正式流程；调试脚本生命周期按 60 [S15]（任务完成即删，不得入库）。
