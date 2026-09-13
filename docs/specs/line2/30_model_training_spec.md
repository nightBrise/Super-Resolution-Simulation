# Phase 1 图像接口放大网络（模型与训练）Spec

> 符号与全局约束以 `00_master_spec.md` [S1]/[S2] 为准；数据与判据分别由 `10`/`20` 提供。实现归属 `src/line2/training/`。
> 版本：v0.2.1（**Frozen v1.0**，2026-09-14 用户批准冻结）。上游：大纲 v0.7 叶 L3b；修订史 = v0.2（审核 findings B2 训练侧联动/M5/M6/M7/M9 清零）+ v0.2.1（复审 N6/N7 残句清零）。

## [S1] U-Net 架构、输入预处理与条件化

6 通道输入 → 6 通道输出的残差 U-Net；实现承 line1 `src/models/unet.py` 的层级结构（5 级残差编码-解码、跳跃连接）。

**输入预处理（F8 (a)，每投影自归一）**：输入图组的每投影计数图 x_i SHALL 除以其计数总和得到形状图 s_i = x_i/Σx_i（Σx_i = 0 的退化输入 → 全零形状图 + 条件向量的 N_low 分量照常取 log₁₀(N_low)）；网络输入 = 6 张形状图（float32，逐投影和为 1）。

**条件输入（F8 (b)，矩条件化，承 line1 方案 C 的 FiLM 机制）**：条件向量 κ ∈ ℝ¹¹ = [输入图组逐投影二阶矩拼装的 10 维 Σ 摘要（对角 4 + 相关 6）, log₁₀(N_low)]；κ SHALL 仅由**输入图组与 N_low** 计算（`00` GC——防泄露）；κ 经 FiLM 注入 U-Net 每级（scale/shift，line1 方案 C 同构）。

**Dependencies (optional):** 无（可与 `10`/`20` 并行开发，集成依赖 [S5]）

### Claims

- C1: WHEN 前向输入 (B, 6, 128, 128) 形状图 + (B, 11) 条件向量 THEN 输出 SHALL 为 (B, 6, 128, 128) 的**逐投影概率图**（Softplus 后逐投影除以自身和，每张和 = 1，元素 > 0）。
- C2: 通道基 C ∈ {24, 48, 96} 与深度 d ∈ {3, 4, 5} SHALL 为 config 参数；默认 C=48、d=4；参数量 SHALL 记录于 config 且总参数量 ≤ 6×10⁷（公平上限）。
- C3: checkpoint 保存后重新加载，同输入的输出 SHALL 逐位一致；ckpt SHALL 校验 config 哈希与 master_seed。
- C4: 输入通道顺序 SHALL 为 `00` [S1] 六投影枚举（与 `10` [S4] 图组一致），并由测试断言。
- C5: **条件化防泄露**：κ SHALL 只由输入图组与 N_low 派生；WHEN 目标图组改变而输入图组不变 THEN κ 与网络输出 SHALL 均不变（测试：改目标不改输入，输出逐位不变——★ 防泄露级测试）。
- C6: 输入预处理可逆性：s_i 与 Σx_i（元数据）SHALL 以相对误差 < 1×10⁻⁶ 还原 x_i（float32 存储的容差断言）。

## [S2] 损失函数

**Dependencies (optional):** [S1]

- **输出语义（B2 训练侧联动，与 `20` [S1] M1 联签）**：网络输出 p（逐投影和 = 1）为收敛密度**形状**的概率估计；损失与度量使用的率为 **rate = p · N_high**（N_high = 2×10⁵，数据集级常量，取自 `10` manifest，非学习参数）。该语义下 M1 的总计数归一对网络输出为恒等（Σrate = Σr），对 raw 子样本输入为重标——两类被评对象进入同一比较语义。
- **Poisson NLL（主损失，独立版与联合版共用同一实现）**：L = (1/|Ω|) Σ_Ω [rate − c·ln(rate) + lgamma(c+1)]，其中 Ω = 全部 6×128×128 格的 micro 平均，c = 目标图组逐格计数（2e5 全量，非负整数），rate 下限 1×10⁻¹² 数值稳定。
- **软一致罚项（仅联合版）**：由输出 p 逐投影二阶矩拼装 4×4 矩阵 Σ̂，罚项 P_PSD = max(0, −λ_min(Σ̂))²，总损失 = NLL + w_psd·P_PSD（w_psd 默认 0.1，config 可调，进搜索空间）。

### Claims

- C1: GIVEN 固定样例（手选 c 矩阵与 p 矩阵）THEN 损失函数输出 SHALL 与按公式手算值一致（绝对误差 < 1×10⁻⁶），测试样例固化在单元测试中。
- C2: WHEN Σ̂ 半正定（λ_min ≥ 0）THEN P_PSD SHALL 恰为 0；WHEN 构造 λ_min < 0 的输出 THEN P_PSD SHALL > 0 且单调于 −λ_min。
- C3: 输出概率图 SHALL 逐投影 Σp = 1（Softplus+归一化后，测试断言到 1×10⁻⁶）。
- C4: 独立版与联合版 SHALL 调用同一 NLL 实现（单一函数，静态检查）。

## [S3] 训练循环

**Dependencies (optional):** [S1]、[S2]、`10` [S5]、`20` [S3]

- AdamW（lr ∈ {1×10⁻⁴, 3×10⁻⁴}，weight_decay 1×10⁻⁴）；batch_size（config，默认 16）；梯度裁剪（max_norm=1.0）；**max_steps = 50000（预注册，config 必填）**；早停 patience = 10 个评估周期、评估周期 = 1000 步；早停与选型准绳 = **val N_eff**（`20` [S3]，val 束上计算）；checkpoint = best_val + last（含 config 哈希、master_seed）。
- DDP 双卡 / 单卡 fallback（沿 line1 `train.py` 骨架）；数据加载顺序由种子派生（`00` GC2）。
- **确定性条款（审核 M6）**：CPU 单元/集成测试 SHALL 逐位确定；GPU 训练 SHALL 启用确定性模式（`torch.use_deterministic_algorithms(True)` + cudnn deterministic），同硬件同版本下重训轨迹 SHALL 逐位一致；WHEN 存在不可确定算子 THEN SHALL 记录执行模式并放宽为"val N_eff 轨迹相对偏差 < 0.5% 的容差一致"（决策日志 D8 记录实际取用）。
- 种子执行计划（F9，用户确认）：调参探索期每配置 1 seed；最终配置与主对比 ≥5 seeds。

### Claims

- C1: WHEN 同 master_seed 同 config 同执行模式重训 THEN val N_eff 轨迹 SHALL 逐位一致（GPU 确定性模式启用）；存在不可确定算子时 SHALL 满足容差一致并记录模式。
- C2: best_val.ckpt SHALL 对应 val N_eff 最优的评估周期（测试：构造人工退化序列验证选择逻辑）。
- C3: 训练日志 SHALL 逐评估周期记录：step、train NLL、val NLL、val N_eff、学习率、梯度范数（落盘可解析）。
- C4: DDP 双卡与单卡 fallback 两条路径 SHALL 均能完成训练（冒烟级；跨执行模式的浮点逐位一致不要求，须记录模式）。
- C5: 训练全程 SHALL NOT 读取 test 束（数据管线按束切分，训练代码物理上不接收 test 路径——GC3）。

## [S4] 独立版/联合版对照

**Dependencies (optional):** [S1]–[S3]

- **独立版**：6 个互不共享权重的 1 进 1 出 U-Net（每投影一个），基通道取 **C_ind = ceil(C/√6)**（设计期以实测参数量校准，满足 C1 后冻结实测值）；每模型只训自己那路投影，条件向量相应缩为该投影的 4 维摘要（σ_i, σ_j, ρ_ij, log₁₀N_low）。
- **联合版**：1 个 6 进 6 出 U-Net（[S1] 原样）+ 可选软一致罚项。

### Claims

- C1: 独立版 6 模型参数量之和与联合版参数量之差 SHALL ≤ 30%（设计期以实测值校准 C_ind 达成，实测值记录于 config 与决策日志）。
- C2: 两版 SHALL 消费完全相同的 train/val 束集合、相同步数预算（max_steps=50000）与相同种子派生链（差异仅在网络拓扑与损失罚项——GC4）。

## [S5] 推理输出

**Dependencies (optional):** [S1]、[S3]

- `python -m src.line2.training.infer --ckpt <path> --input <图组> --out <dir>`：推理命令 SHALL 产出组N 文件（npz：6×128×128，含概率图 p 与率图 rate = p·N_high 两套，float64；元数据 beam_id/station/N_low/sub_idx/ckpt 哈希）。

### Claims

- C1: 推理 SHALL 确定性：同 ckpt 同输入（eval 模式）多次运行逐位一致。
- C2: 输出 SHALL 形状 (6,128,128)、含概率与率双套数组、逐元素 > 0，元数据含输入来源四元组（beam_id, station, N_low, sub_idx）。
- C3: rate 图 SHALL 逐投影总和 = N_high（`20` M1 总计数归一对其为恒等——比较语义闭合）。

## Global Constraints

- 继承 `00` [S2] 全部（GC1–GC7）。
- 训练与推理 SHALL 只消费 `10` [S5] 数据集与其切分；物理上不接收 test 束路径。
- 条件向量 SHALL NOT 使用任何目标/参照信息（★ 防泄露，C5）。
- 模型与训练配置 SHALL 进 manifest 三元组（GC7）。
- 双卡训练期间 SHALL 遵守 line1 GPU 纪律（本 spec 训练可用 GPU；L0/L1 单元与集成测试保持纯 CPU）。

## Out of Scope

- 束流接口（粒子云渲染、可微 KDE）——Phase 2 L9。
- 实验矩阵、门禁判定、跨站泛化结论——`40`。
- test 集上的任何评估调度（`40` 唯一有权调度 test 评估）。
- 生成式输出（扩散/采样）——H5，Phase 2 评估后再议。
