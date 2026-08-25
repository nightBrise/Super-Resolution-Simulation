# 物理先验引导的纵向相空间超分辨率研究

# Physics-Prior-Guided Longitudinal Phase Space Super-Resolution

[English version available →](README.en.md)

![版本](https://img.shields.io/badge/version-0.x.x-yellow)
![状态](https://img.shields.io/badge/status-spec%20draft-orange)
![Spec](https://img.shields.io/badge/spec-v0.1%20%E8%8D%89%E6%A1%88-lightgrey)

> 直线加速器驱动的 FEL 束流诊断中，从低分辨率纵向相空间观测恢复高分辨率精细结构的配对监督模拟研究。

---

## 中文简介

在直线加速器驱动的自由电子激光（FEL）束流诊断中，纵向相空间 $(z, \delta)$ 是判断束流品质的核心对象。TDS（横向偏转腔）+ 谱仪等诊断系统只能输出低分辨率观测图像，而真实高分辨率相空间中存在的精细结构（压缩折叠、细脊、电流尖峰、局部能散变化等）对 FEL 物理至关重要，却又无法从低分辨率图像清楚分辨。

本项目通过**不依赖 elegant / Ocelot 的轻量级纵向相空间模拟函数**生成严格配对的 $(H, L, P)$ 数据集，训练超分辨率神经网络，**回答一个核心问题**：在低分辨率观测下，物理先验能否帮助网络更准确、更符合物理地恢复高分辨率纵向相空间？

### 研究方法

- **三方案公平对比**（相同数据、相同损失、相同主干、相同训练配置、相同评估指标）：
  - 方案 A：无先验 (baseline) — $\hat{H} = \mathrm{NonNeg}(L_{\mathrm{up}} + G_0(L_{\mathrm{up}}))$
  - 方案 B：图像先验 + 残差 — $\hat{H} = \mathrm{NonNeg}(P_2 + G_1(L_{\mathrm{up}}, P_2))$
  - 方案 C：参数先验 + FiLM — $\hat{H} = \mathrm{NonNeg}(L_{\mathrm{up}} + G_2(L_{\mathrm{up}} \mid c_{\mathrm{prior}}))$
- **训练目标**：空域-频域混合损失 $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{space}} + \lambda \mathcal{L}_{\mathrm{spec}}$（$\lambda = 1.0$ 冻结预注册）
- **物理复杂度起点**：Level 1（非高斯剖面 + 不对称 + 局部厚度变化 + 三阶中心线 + 二/三阶压缩折叠）
- **退化协议**：256 → 64 下采样 + 高斯模糊 + 高斯噪声，标定采用值（$\sigma_K$ / $\sigma_n$ / $\sigma_{\mathrm{smooth}}$）登记于 `config.yaml` 与 `99_change_log.md`

### 理论支撑

本项目借鉴 ICLR 2026 论文 *Breaking Scale Anchoring*（wang2026_breaking-scale-anchoring）的两个核心洞察：

- **Scale Anchoring / 频谱偏差**：纯空间域 L1 损失会使网络优先拟合低频、抹平精细结构；引入频域损失 $\mathcal{L}_{\mathrm{spec}}$ 作为 $\mathcal{L}_{\mathrm{space}}$ 的补充，迫使网络恢复高频折叠。
- **信息论限制**：退化使高频成分落入噪声底以下；物理先验（$P_2$ 或 $c_{\mathrm{prior}}$）本质上就是"噪声底以下缺失信息的条件锚点"——这是先验形式必要性的理论背书。

---

## 仓库结构

```text
.
├── README.md                                              # 中文版（本文件）
├── README.en.md                                           # English version
├── LICENSE                                                # MIT
├── .gitignore
└── docs/
    ├── specs/                                             # Spec 集（v0.1 草案，待审查）
    │   ├── README.md                                      # Spec 索引
    │   ├── 00_master_spec.md                              # 总纲：项目目标、符号、里程碑
    │   ├── 10_research_plan.md
    │   ├── 20_physics_generator_spec.md
    │   ├── 30_degradation_spec.md
    │   ├── 40_prior_spec.md
    │   ├── 50_network_spec.md
    │   ├── 60_training_spec.md
    │   ├── 70_evaluation_spec.md
    │   ├── 80_experiment_matrix.md
    │   ├── 90_delivery_spec.md
    │   ├── 99_change_log.md
    │   └── archive/
    │       └── chat-超分辨率增强模拟1.txt                 # 原始对话归档
    # 参考论文 PDF 按 .gitignore 约定不上传仓库（本地保留），引用请走下方 arXiv URL

# 实现阶段将创建：
#   src/                    # 源代码（生成器 / 模型 / 训练 / 评估）
#   data/<版本>/            # 数据集
#   results/<EXP>.../       # 实验结果
#   final_report.md         # 最终研究报告
```

---

## 当前状态

| 项目 | 状态 |
|---|---|
| Spec 集 | **v0.1 草案**（待用户最终审查） |
| 版本号 | **0.x.x**（待全部流程跑通、报告输出、用户确认后升至 1.x.x） |
| 代码实现 | 尚未开始（M0 冻结后启动） |
| 数据集 | 未生成 |
| 最终报告 | 未产出 |

### 版本号约定

- **0.x.x（当前）**：Spec 集 v0.1 草案、代码未实现、流程未跑通；
- **1.x.x（目标）**：全部阶段流程（M0 → M6）跑通、`final_report.md` 产出、经用户审查通过后一次性升级到 1.0.0；后续按变更幅度递增 1.x.y。

### 里程碑

| 里程碑 | 名称 | 入口条件 | 退出判据 |
|---|---|---|---|
| M0 | Spec 冻结 | 用户审查（含跨文档一致性检查） | v1.0 冻结；版本历史登记 |
| M1 | 基础模拟函数 | M0 | `f(x) → (H, L, P)` 可用；通过 AC |
| M2 | 数据集生成 | G0 数据有效性门禁 | 规模/划分/可复现达标 |
| M3 | 无先验网络 (baseline) | G1(a) | 方案 A 收敛；基线指标 |
| M4 | 带先验网络 + 先验增益判定 | G2 | 三分类统计判定产出 |
| M5 | 物理评估与消融 | G3 | 增益分解 + 方向判定 + 幻觉判定 |
| M6 | 最终研究报告 + 出口决策 | `90` [S5] N1–N8 | `final_report.md` 验收通过 |

完整阶段—门禁—里程碑—实验对应关系见 [`docs/specs/80_experiment_matrix.md`](docs/specs/80_experiment_matrix.md) § [S10]。

---

## 阅读顺序

1. [`docs/specs/README.md`](docs/specs/README.md) — Spec 索引；
2. [`docs/specs/00_master_spec.md`](docs/specs/00_master_spec.md) — 总纲（必读）；
3. 按当前任务精读对应模块的子 Spec（10–90）；
4. 任何变更先登记 [`docs/specs/99_change_log.md`](docs/specs/99_change_log.md)，批准后修改，落地后置 Implemented。

---

## 引用文献

本项目借鉴：

- **Wang et al. (ICLR 2026)**. *Breaking Scale Anchoring: Frequency Representation Learning for Zero-Shot Super-Resolution*. arXiv:[2512.05132v2](https://arxiv.org/abs/2512.05132)。PDF 按 `.gitignore` 约定不上传仓库，本地保留作阅读。

---

## 许可证

[MIT](LICENSE)。

---

## 联系方式

待补充。