# 60_training_spec.md Spec

- 版本: v0.1（草案，待审查）

## 文档目的

本文件定义方案 A / B / C 三个超分网络的统一训练协议：训练目标与损失、数据归一化与参数标准化、训练样本格式、输入构造、噪声处理、数据增强策略、数据集规模与划分、优化器与超参数、训练长度与早停、训练公平性、日志与 checkpoint、复现规则与验收标准。第一版训练原则为：不使用任何物理损失，三方案统一使用空域-频域混合重建损失（见 `[S2]`），使用相同数据与相同优化配置；物理一致性交由 `70_evaluation_spec.md` 在评估阶段检查。方案 A / B / C 的输入输出形式与网络结构由 `40` / `50` 定义，`60` 只定义训练流程本身。

---

## [S1] 范围

`60` 包含：统一训练目标、损失函数、数据归一化、参数标准化、训练样本格式、三方案输入构造、噪声处理、数据增强策略、数据集规模与划分、优化器、batch size、训练长度与早停、训练公平性、日志与 checkpoint、验收标准。

`60` 不包含：高分辨率真值 `H` 生成（属于 `20`）、低分辨率退化 `L`（属于 `30`）、先验生成（属于 `40`）、网络结构（属于 `50`）、评估指标（属于 `70`）。

### Claims

- C1: `60` SHALL 只定义训练协议；数据生成、先验生成、网络结构与评估指标 SHALL 分别由 `20` / `30` / `40` / `50` / `70` 定义。
- C2: 第一版训练 SHALL NOT 使用物理损失；物理一致性 SHALL 由 `70` 在评估阶段检查。

---

## [S2] 统一训练目标与损失

三个方案共享同一个训练目标：让网络预测逼近高分辨率真值。

$$
\hat{H} \approx H
$$

其中 `H` 为高分辨率真值，`Ĥ` 为网络预测。第一版训练损失为空域-频域混合损失：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{space}} + \lambda \, \mathcal{L}_{\text{spec}}
$$

其中 $\mathcal{L}_{\text{space}}$ 为对总强度归一化图像的逐像素 L1：

$$
\mathcal{L}_{\text{space}} = \frac{1}{N^2} \sum_{i,j} \left| \hat{H}_{ij} - H_{ij} \right|
$$

$\mathcal{L}_{\text{spec}}$ 为加权谱 L1：对归一化后的 $\hat{H}$、$H$ 取 2D FFT，按径向频率划分 5 个倍频程频带

$$
\left[0, \frac{f_c}{4}\right],\quad \left[\frac{f_c}{4}, \frac{f_c}{2}\right],\quad \left[\frac{f_c}{2}, f_c\right],\quad \left[f_c, 2f_c\right],\quad \left[2f_c, f_N\right]
$$

其中 $f_c = 1/8$ 为退化截止频率（归一化周期/像素，对应 $r = 4$ 下采样），$f_N = 0.5$ 为奈奎斯特频率。每频带内取谱误差均值、频带间等权：

$$
\mathcal{L}_{\text{spec}} = \frac{1}{5} \sum_{b=1}^{5} \; \underset{k \in \text{band}_b}{\mathrm{mean}} \, \left| \mathcal{F}(\hat{H})_k - \mathcal{F}(H)_k \right|
$$

等权分带的原因：避免低频分量主导谱损失。$H$ 已归一化至总强度 1，而 $\hat{H}$ 未归一化（$\hat{H} = \text{Softplus}(\text{Base} + R)$），两侧 DC 项并不相等，其差异保留于 $\mathcal{L}_{\text{spec}}$ 内，作为总强度匹配信号。

权重 $\lambda$ SHALL 冻结为 1.0（预注册常数，不做代理选择）；三方案使用完全相同的 $\mathcal{L}_{\text{total}}$。λ 恒为 1.0，无选择步骤。

不使用的损失：

$$
\mathcal{L}_{\text{moment}}, \quad \mathcal{L}_{\text{marginal}}, \quad \mathcal{L}_{\text{forward}}
$$

选择该混合损失的原因：$\mathcal{L}_{\text{space}}$ 保持逐像素保真，对噪声比 L2 更稳健；$\mathcal{L}_{\text{spec}}$ 显式约束频谱结构，抑制纯像素级 L1 的条件中值解倾向（平滑掉精细结构、频谱偏差使网络优先拟合低频）；两者均不引入额外物理先验（既不是物理损失，也不是感知损失），便于三方案公平比较。

风险说明：频域损失鼓励更锐利的高频输出，可能放大纹理幻觉；该物理幻觉风险不在训练阶段抑制，而由 `70` 的一票否决（`70` `[S4]`）与 $\mathcal{E}_{\text{high}}$/$R_E$ 联合读数在评估阶段兜底。

### Claims

- C1: 三方案总损失 SHALL 为空域-频域混合损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{space}} + \lambda \, \mathcal{L}_{\text{spec}}$，其中 $\mathcal{L}_{\text{space}}$ 为对总强度归一化图像的逐像素 L1，$\mathcal{L}_{\text{spec}}$ 为按径向频率五倍频程分带（$f_c = 1/8$、$f_N = 0.5$）的等权加权谱 L1。
- C2: 第一版训练 SHALL NOT 使用 $\mathcal{L}_{\text{moment}}$、$\mathcal{L}_{\text{marginal}}$、$\mathcal{L}_{\text{forward}}$ 等物理损失；$\mathcal{L}_{\text{spec}}$ SHALL 视为重建损失，SHALL NOT 视为物理损失。
- C3: $\lambda$ SHALL 冻结为 1.0（预注册常数，不做代理选择）；三方案 SHALL 使用完全相同的 $\mathcal{L}_{\text{total}}$。

---

## [S3] 数据归一化与参数先验

为让训练稳定并避免绝对强度主导损失，第一版对所有图像做总强度归一化（各图像分别归一化到总和为 1）：

$$
\sum_{i,j} H_{ij} = 1, \qquad \sum_{i,j} L_{\text{up},ij} = 1, \qquad \sum_{i,j} P_{2,ij} = 1
$$

其中 $L_{\text{up}}$ 为 `L` 上采样到 `H` 分辨率（$256 \times 256$）的结果。

在总强度归一化的前提下，总强度参数 `A` 不再提供有效信息，故方案 C 训练时将 `A` 从 $c_{\text{prior}}$ 中移除（或固定为常数），第一版方案 C 的参数先验为：

$$
c_{\text{prior}} = \{\sigma_z, n, \eta, b_0, a_1, \alpha, a_2, \beta\}
$$

不包含 `A`，也不包含：

$$
c_{\text{high}} = \{a_3, \gamma, b_1\}
$$

### Claims

- C1: 每个训练样本的 `H`、$L_{\text{up}}$、$P_2$ SHALL 分别归一化到总强度 1。
- C2: 方案 C 的 $c_{\text{prior}}$ SHALL 为 $\{\sigma_z, n, \eta, b_0, a_1, \alpha, a_2, \beta\}$，SHALL NOT 包含 `A` 与 $c_{\text{high}} = \{a_3, \gamma, b_1\}$。
- C3: 三方案的输入图像归一化方式 SHALL 完全一致。

---

## [S4] 训练样本格式与输入构造

每个训练样本包含以下字段：

| 字段 | 含义 |
|---|---|
| `H` | 高分辨率真值，$256 \times 256$ |
| `L` | 低分辨率观测，$64 \times 64$ |
| $L_{\text{clean}}$ | 无噪声低分辨率图像，可选 |
| $P_2$ | 中阶图像先验，$256 \times 256$ |
| $c_{\text{prior}}$ | 参数先验（方案 C 输入） |
| $c_{\text{high}}$ | 高分辨率精细参数，仅用于评估，不输入网络 |
| `d` | 退化参数 |
| `m` | 物理标签 |
| `seed` | 随机种子 |

### 方案 A：无先验

输入：

$$
\text{Input}_A = \text{concat}(L_{\text{up}}, 0)
$$

（补零通道，使输入通道数与方案 B 一致。）残差基准：

$$
\text{Base}_A = L_{\text{up}}
$$

前向：$R_A = G_A(\text{Input}_A)$；输出：

$$
\hat{H} = \text{Softplus}\left(L_{\text{up}} + R_A\right)
$$

损失：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{space}} + \lambda \, \mathcal{L}_{\text{spec}}
$$

（各分量定义见 `[S2]`。）

### 方案 B：图像先验 + 残差

输入：

$$
\text{Input}_B = \text{concat}(L_{\text{up}}, P_2)
$$

残差基准：

$$
\text{Base}_B = P_2
$$

前向：$R_B = G_B(\text{Input}_B)$；输出：

$$
\hat{H} = \text{Softplus}\left(P_2 + R_B\right)
$$

损失：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{space}} + \lambda \, \mathcal{L}_{\text{spec}}
$$

（各分量定义见 `[S2]`。）

### 方案 C：参数先验 + FiLM

输入图像 $L_{\text{up}}$，条件参数为标准化后的参数先验 $\tilde{c}_{\text{prior}}$：

$$
\tilde{c}_{\text{prior}} = \tilde{c}_{\text{prior}}(\sigma_z, n, \eta, b_0, a_1, \alpha, a_2, \beta)
$$

残差基准：

$$
\text{Base}_C = L_{\text{up}}
$$

前向：$R_C = G_C(L_{\text{up}} \mid \tilde{c}_{\text{prior}})$；输出：

$$
\hat{H} = \text{Softplus}\left(L_{\text{up}} + R_C\right)
$$

损失：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{space}} + \lambda \, \mathcal{L}_{\text{spec}}
$$

（各分量定义见 `[S2]`。）

三方案输出均为 $\hat{H} \ge 0$（由 Softplus 保证非负，属输出格式约束而非物理损失）。

### Claims

- C1: 每个训练样本 SHALL 包含 `H`、`L`、$P_2$、$c_{\text{prior}}$、`d`、`m` 与 `seed`；$L_{\text{clean}}$ 与 $c_{\text{high}}$ SHALL 可选，且 $c_{\text{high}}$ SHALL NOT 输入网络。
- C2: 方案 A SHALL 以 $\text{Input}_A = \text{concat}(L_{\text{up}}, 0)$ 为输入、$L_{\text{up}}$ 为残差基准，输出 $\hat{H} = \text{Softplus}(L_{\text{up}} + G_A(\text{Input}_A))$。
- C3: 方案 B SHALL 以 $\text{Input}_B = \text{concat}(L_{\text{up}}, P_2)$ 为输入、$P_2$ 为残差基准，输出 $\hat{H} = \text{Softplus}(P_2 + G_B(\text{Input}_B))$。
- C4: 方案 C SHALL 以 $L_{\text{up}}$ 为图像输入、标准化后的 $\tilde{c}_{\text{prior}}$ 为条件参数、$L_{\text{up}}$ 为残差基准，输出 $\hat{H} = \text{Softplus}(L_{\text{up}} + G_C(L_{\text{up}} \mid \tilde{c}_{\text{prior}}))$。
- C5: 三方案输出 SHALL 满足 $\hat{H} \ge 0$；该非负约束 SHALL 视为输出格式约束，SHALL NOT 视为物理损失。

---

## [S5] 参数标准化

方案 C 的参数先验必须标准化，规则如下：

1. 对正参数先取对数：$\sigma_z,\ b_0$；
2. 对每个参数计算训练集均值与标准差并做 z-score：

$$
\tilde{c} = \frac{c - \mu_c}{\sigma_c}
$$

3. 验证集与测试集使用训练集统计量，不得使用验证集或测试集自身统计量（防止测试集泄漏）。

### Claims

- C1: 参数标准化 SHALL 为 $\tilde{c} = (c - \mu_c)/\sigma_c$，其中 $\mu_c$、$\sigma_c$ SHALL 由训练集计算。
- C2: 正参数 $\sigma_z$、$b_0$ SHALL 在 z-score 前先取对数。
- C3: 验证集与测试集的参数标准化 SHALL 复用训练集统计量，SHALL NOT 使用验证集或测试集自身统计量。

---

## [S6] 噪声处理

第一版训练输入使用带噪声的低分辨率图像：

$$
L = \max(0, L_{\text{clean}} + n), \qquad n \sim \mathcal{N}(0, \sigma_n^2)
$$

训练目标仍然是干净的 `H`。

使用带噪声 `L` 的原因：更接近真实诊断信号；防止网络只学习去模糊；测试先验在噪声下的作用；避免任务过于简单。

$L_{\text{clean}}$ 不用于主训练输入，但可用于：调试；消融实验；区分模糊影响与噪声影响；后续 forward consistency 实验。

### Claims

- C1: 主训练输入 `L` SHALL 为带噪声图像 $L = \max(0, L_{\text{clean}} + n)$，其中 $n \sim \mathcal{N}(0, \sigma_n^2)$。
- C2: 训练目标 SHALL 为干净 `H`。
- C3: $L_{\text{clean}}$ SHALL NOT 作为主训练输入，SHALL 保留用于调试、消融与后续实验。

---

## [S7] 数据增强策略

图像是物理相空间（坐标 `z`、`δ` 均有物理意义），第一版不使用传统自然图像增强。

不建议使用的增强方式：

| 增强 | 原因 |
|---|---|
| 随机旋转 | `z` 与 `δ` 坐标方向有物理意义 |
| 随机翻转 | 可能改变 chirp 符号或头尾方向 |
| 随机裁剪 | 会破坏总强度和相空间范围 |
| 颜色扰动 | 图像是物理密度，不是 RGB 照片 |
| 强几何变形 | 可能产生非物理相空间 |

第一版推荐以生成参数空间采样代替图像增强，增强物理分布而非视觉纹理：

1. **物理参数采样**：采样不同 `c`（束团长度、剖面形状、不对称度、局部能散、线性 chirp、二阶弯曲、三阶 S 形、压缩折叠强度）；
2. **退化参数采样**：采样不同退化参数 `d`（模糊核大小、噪声强度、下采样倍数、增益、背景等）；
3. **噪声实现采样**：对同一 $(H, L_{\text{clean}})$ 使用不同噪声样本 $n$。

后续（第一版不加入）可考虑先验扰动增强：$c_{\text{prior}}' = c_{\text{prior}} + \Delta c$（参数噪声、参数偏差、图像先验过平滑、图像先验略微错误），用于测试网络在先验不完美时的鲁棒性。

增强必须满足的约束：

- **保持物理一致性**：密度不得为负；总强度守恒或归一化；折叠结构不得违反映射关系；噪声不得超过信号太多；模糊不得强到完全摧毁结构；
- **不泄露高分辨率答案**：不得用 `H` 直接生成默认先验；`L` 不得保留过多高频细节；先验不得包含 $c_{\text{high}}$；
- **不破坏训练目标**：增强应制造难度，但不得制造不可解任务（噪声过大、模糊过强、精细结构完全消失、先验完全错误）。

### Claims

- C1: 第一版训练 SHALL NOT 使用随机旋转、随机翻转、随机裁剪、颜色扰动与强几何变形等传统图像增强。
- C2: 第一版数据多样性 SHALL 通过采样不同物理参数 `c`、不同退化参数 `d` 与不同噪声实现 $n$ 获得。
- C3: 第一版 SHALL NOT 加入先验扰动增强（参数噪声、参数偏差、不完美先验）。
- C4: 增强后的样本 SHALL 保持物理一致性：密度非负、总强度守恒或归一化、折叠结构不违反映射。
- C5: 增强 SHALL NOT 泄露高分辨率答案：不得用 `H` 生成默认先验，`L` 不得保留过多高频细节，先验不得包含 $c_{\text{high}}$。

---

## [S8] 数据集规模与划分

数据集规模分三档：

| 规模 | Train | Val | Test | 用途 |
|---|---:|---:|---:|---|
| 调试规模 | 2,000 | 500 | 500 | 快速跑通、检查 shape、损失下降、输出非负 |
| 标准 demo 规模 | 20,000 | 2,000 | 2,000 | 主实验，比较 A / B / C，得到初步结论 |
| 更充分规模 | 50,000–100,000 | 5,000 | 5,000 | 更稳定结论、OOD 测试、精细结构分析 |

第一版流程：先用调试规模跑通，再用标准 demo 规模做主实验。

数据集至少划分三类：

- **In-distribution validation**：与训练集同分布，用于早停与调参；
- **In-distribution test**：与训练集同分布但训练不可见，用于最终比较；
- **Out-of-distribution test**：参数或退化条件超出训练范围（如更强的 $a_3,\gamma,\beta$、更高噪声、更强模糊、更明显的局部厚度变化、训练只用 Level 1 时的 Level 2 / Level 3 数据），用于检查泛化。

划分机制（参数空间划分协议）：划分 SHALL 先按内容参数 `c` 进行——同一 `c`（同一 `H`、同一先验）的所有噪声实现 SHALL 落在同一划分内，噪声重采样只允许发生在训练划分内部；划分完成后再在各划分内独立采样噪声。in-distribution test 由两个子集构成，标准 demo 规模下两者 1:1（test_id 1,000 / test_pb 1,000）：(i) 同分布留出子集（按 `c` 互不相交的随机划分，回答同分布增益）；(ii) 参数分块留出子集（回答插值）。参数分块留出细则：块维度为 |γ|（幅度坐标）；块区间为 $|\gamma| \in [0.3, 0.4]$，即 $|\gamma| \sim U[0.1, 0.6]$ 的分位秩 $[0.4, 0.6]$（幅度中央 20% 分位带），带符号写法为 $\gamma \in [-0.4, -0.3] \cup [0.3, 0.4]$（γ 按 `20` [S9] 的 $\gamma = -\mathrm{sign}(\beta)\cdot[0.1,0.6]$ 采样规则取值）；块区间按固定总体分位数确定，SHALL NOT 采用经验样本分位数。抽样规则：块外条件采样（拒绝块内候选）保证 train 20,000 / val 2,000 / test_id 1,000 全部落在块外；块内条件采样恰生成 1,000 个样本构成 test_pb。OOD test 数据的生成与保留为第一版必做，限定当前可定义子集：EXP-06 极端参数集与更强退化集（由 `80_experiment_matrix.md` 定义）；OOD 评估（EXP-05/06）第一版可选（Phase 4 Could）。

### Claims

- C1: 第一版主实验数据集规模 SHALL 为 20,000 train / 2,000 val / 2,000 test；调试阶段 SHALL 使用 2,000 / 500 / 500。
- C2: 数据集 SHALL 至少划分 in-distribution validation 与 in-distribution test；in-distribution test 数据 SHALL NOT 参与训练与早停；划分 SHALL 先按内容参数 `c` 进行，同一 `c` 的噪声实现 SHALL NOT 跨训练/验证/测试划分。
- C3: OOD 数据的生成与保留 SHALL 为第一版必做，限定当前可定义子集：EXP-06 极端参数集与更强退化集（见 `80`）；Level 2 数据生成义务与 `20` [S8] 参数范围预注册时点绑定。OOD 评估（EXP-05/06）第一版可选（Phase 4 Could）；不执行时，报告 SHALL 标注证据缺失（`90` [S2] C11）。
- C4: in-distribution test SHALL 由同分布留出子集（test_id）与参数分块留出子集（test_pb）构成，标准 demo 规模下两者 1:1（test_id 1,000 / test_pb 1,000）；test_pb 的块维度 SHALL 为 |γ|（幅度坐标），块区间 SHALL 为 $|\gamma| \in [0.3, 0.4]$（$|\gamma| \sim U[0.1, 0.6]$ 分位秩 $[0.4, 0.6]$，即幅度中央 20% 分位带；带符号 $\gamma \in [-0.4, -0.3] \cup [0.3, 0.4]$），按固定总体分位数确定，SHALL NOT 采用经验样本分位数；train / val / test_id 样本 SHALL 经块外条件采样（拒绝块内候选）全部落在块外，test_pb SHALL 由块内条件采样恰 1,000 个样本构成。

---

## [S9] 优化器与训练超参数

第一版优化器：

$$
\boxed{\text{AdamW}}
$$

默认超参数建议：

| 超参数 | 建议值 |
|---|---|
| optimizer | AdamW |
| learning rate | $1\times10^{-4}$ 或 $3\times10^{-4}$ |
| weight decay | $1\times10^{-4}$ 到 $1\times10^{-2}$ |
| beta1 | 0.9 |
| beta2 | 0.999 |
| gradient clipping | 可选，max norm 1.0 |

若训练不稳定，可先降低学习率。

batch size 按显存选择（图像为 $256 \times 256$）：

| 显存情况 | batch size |
|---|---:|
| 小显存 | 4–8 |
| 中等显存 | 16 |
| 大显存 | 32 |

三个方案必须使用相同 batch size。

### Claims

- C1: 三方案优化器 SHALL 均为 AdamW。
- C2: 学习率、weight decay、beta1、beta2 与 gradient clipping 设置 SHALL 在三方案间一致。
- C3: 三方案 SHALL 使用相同 batch size。

---

## [S10] 训练长度与早停

训练长度以 step 计，不用 epoch：

- 调试阶段：$5{,}000 \sim 10{,}000$ steps；
- 标准阶段：$50{,}000 \sim 100{,}000$ steps（取决于数据规模与收敛速度）。

早停基于验证集空域-频域混合损失 $\mathcal{L}_{\text{val}}$（即 $\mathcal{L}_{\text{total}}$ 在验证集上的均值，定义见 `[S2]`）：每 2,000 steps 验证一次；保存最优 checkpoint；若连续 $\text{patience} = 10$ 次验证无改善则停止训练。早停不得早于最大步数预算的 50%，以防止收敛速度差异造成的不公平截断。

### Claims

- C1: 训练长度 SHALL 以 step 计；调试阶段 SHALL 为 5,000–10,000 steps，标准阶段 SHALL 为 50,000–100,000 steps。
- C2: 早停 SHALL 基于验证集空域-频域混合损失，每 2,000 steps 验证一次并保存最优 checkpoint。
- C3: 训练 SHALL 在验证集损失连续 10 次验证无改善时停止（$\text{patience} = 10$）。
- C4: 早停 SHALL NOT 早于最大步数预算的 50%。

---

## [S11] 训练公平性

三个方案必须满足以下公平性要求：

| 项目 | 要求 |
|---|---|
| 数据 | 三方案使用同一份训练/验证/测试数据 |
| 数据顺序 | 建议使用相同 shuffle seed |
| 损失 | 均为空域-频域混合损失 |
| 优化器 | 均为 AdamW |
| 学习率 | 相同 |
| batch size | 相同 |
| 训练步数（最大预算） | 相同 |
| 随机种子 | 代理尺度 3 个种子、全量阶段每方案恰 2 个种子；扩充仅经 `80` [S9] G2 失败路径预注册后允许；全部记录且可控 |
| 归一化 | 相同 |
| 早停标准 | 均为验证集空域-频域混合损失 |
| 物理损失 | 均不加入 |
| 参数量 | 必须记录 |

### Claims

- C1: 三方案 SHALL 使用同一份训练/验证/测试数据，建议使用相同 shuffle seed。
- C2: 三方案 SHALL 使用相同损失（空域-频域混合损失 $\mathcal{L}_{\text{total}}$，含同一冻结 $\lambda$）、相同优化器（AdamW）、相同学习率、相同 batch size、相同最大训练步数预算（实际停止点由统一早停标准决定）与相同早停标准（验证集空域-频域混合损失）。
- C3: 三方案训练 SHALL 均不加入物理损失。
- C4: 代理尺度阶段 SHALL 使用 3 个种子；全量阶段 SHALL 为每方案恰 2 个种子；扩充种子数 SHALL 仅经 `80` [S9] G2 失败路径预注册后允许；全部种子 SHALL 记录且可控。
- C5: 每个方案的总参数量与可训练参数量 SHALL 被记录。

---

## [S12] 日志与 checkpoint

训练过程必须记录：

| 日志 | 说明 |
|---|---|
| train loss | 每若干 step 记录 |
| val loss | 定期验证 |
| learning rate | 若使用 schedule |
| gradient norm | 可选 |
| output min/max | 检查非负和数值稳定 |
| output sum | 检查总强度漂移 |
| checkpoint path | 保存最优模型 |
| config hash | 记录超参数版本 |
| data version | 记录数据集版本 |
| spec version | 记录 Spec 版本 |

checkpoint 必须保存：

1. 最新 checkpoint；
2. 最优 validation checkpoint；
3. 最终 checkpoint；
4. 配置文件；
5. 随机种子；
6. 数据版本；
7. 训练曲线。

最终评估必须使用 best validation checkpoint，而不是最后一个 checkpoint。

### Claims

- C1: 训练日志 SHALL 至少记录 train loss、val loss、output min/max、output sum、checkpoint path、config hash、data version 与 spec version。
- C2: 训练 SHALL 保存最新、最优 validation 与最终 checkpoint，并随 checkpoint 记录配置文件、随机种子与数据版本。
- C3: 最终评估 SHALL 使用 best validation checkpoint，SHALL NOT 使用最后一个 checkpoint。

---

## [S13] 验收标准

### Claims

- AC1: 三个方案 SHALL 使用相同训练数据。
- AC2: 三个方案 SHALL 使用相同空域-频域混合损失（含同一冻结 $\lambda$）。
- AC3: 训练 SHALL NOT 使用物理损失。
- AC4: 三方案的输入图像归一化方式 SHALL 一致。
- AC5: 参数标准化 SHALL 无测试集泄漏（验证集与测试集使用训练集统计量）。
- AC6: 训练输入 SHALL 使用带噪声 `L`，训练目标 SHALL 为干净 `H`。
- AC7: 三方案输出 SHALL 满足 $\hat{H} \ge 0$。
- AC8: 训练损失与验证损失 SHALL 正常下降，且验证集空域-频域混合损失用于早停与选最优 checkpoint。
- AC9: checkpoint 与日志 SHALL 完整（最优/最终 checkpoint、配置、随机种子、数据版本、训练曲线）。
- AC10: 三个方案的超参数 SHALL 一致。
- AC11: 三个方案的参数量 SHALL 被记录。

---

## [S14] 数据集工件契约

本节定义数据集的文件级契约，是 `20`/`30`/`40`（生产方）与 `60`/`70`/`80`（消费方）之间的唯一接口。数据集按划分为每划分一个 HDF5 文件，外加一个 manifest：

- 目录结构：`data/<版本>/`，包含 `train.h5`、`val.h5`、`test_id.h5`（同分布留出）、`test_pb.h5`（参数分块留出）、`test_ood.h5`（Level 2 部分待 `20` [S8] 参数范围预注册后补）、`test_exp03.h5`、`test_exp04.h5`（见下）、`manifest.json`。
- 每个样本记录包含：
  - 图像字段：`H`（256×256，float32，总强度 1）、`L`（64×64）、`L_up`（256×256，bilinear 插值后归一化，见 `50` [S8]/[S13]）、`P2`（256×256，见 `40` [S5]）；
  - 元数据字段：`sample_id`（字符串，格式 `<划分>-<序号>`）、全部内容参数 `c`（c_low/c_mid/c_high 全字段）与元数据 `m`（见 `00` [S4]）、导出物理量（$\varepsilon_z$、$I_{\text{peak}}$、能谱剖面 $S(\delta)$，供 `70` [S3]/[S4] 使用）、`seed_i`、掩膜标记（`20` [S9] W1–W8 通过/拒绝）、退化配置标记（D1 / D2 / EXP-03 / EXP-04）与退化元数据 `m_L`（`30` [S9]）。
- EXP-03/04 测试工件：`test_exp03.h5`、`test_exp04.h5` 由 EXP-02 测试 `H`（test_id ∪ test_pb）重退化生成，逐样本配对（见 `80` [S6]）。
- `manifest.json` 包含：主种子、数据版本号、各划分样本数与 `sample_id` 清单（即 [S8] 划分结果落盘）、参数分块留出信息（块维度 |γ|、块区间、分位派生与各子集样本数）、σ_K/σ_n/σ_smooth 标定采用值（标定后按版本更新）、生成时间戳与代码版本。
- 种子派生：样本生成种子由 `numpy.random.SeedSequence(master_seed).spawn(N)` 的第 i 个分支给出，N 为样本生成顺序；master_seed 登记于 manifest。各生成器 SHALL NOT 自选随机源。
- 写入分工：`20` 写入 `c`、`m`、`H` 与物理标签；`30` 追加 `L`、`L_up` 与 `m_L`；`40` 追加 `P2`；`60` [S8] 产生划分指派并写 manifest。
- 版本规则：任何生成配置变化（含标定采用值变更后的重生成）SHALL 递增版本号；不同版本数据 SHALL NOT 混用，除非在 manifest 中登记的跨版本对照实验。

### Claims

- C1: 数据集 SHALL 按本节的目录、文件与字段契约落盘；三方案训练与全部实验 SHALL 消费同一版本的数据文件；`manifest.json` SHALL 显式登记 `code_version`（生成该数据集时使用的 git commit hash）、`data_version`、`spec_version` 三元组（与 `00` [S6] 全局约束 8、`80` [S12] C2 配套）；任一字段缺失视为验收失败（参见 `90` [S5] N8 预注册对账）。
- C2: 每个样本 SHALL 具有唯一 `sample_id`；`80` [S8] 的逐样本结果记录 SHALL 以 `sample_id` 与数据集联接。
- C3: manifest SHALL 包含主种子、划分指派、标定采用值与版本信息，保证「相同数据、相同划分」可完整复现。
- C4: 样本种子 SHALL 按 SeedSequence 分支规则派生；任何生成器 SHALL NOT 自选随机源。
- C5: 生成配置变化 SHALL 递增数据版本号；不同版本数据 SHALL NOT 混用，除非在 manifest 中登记。
- C6: manifest SHALL 包含参数分块留出信息（块维度 |γ|、块区间、分位派生与各子集样本数）。

---

## Global Constraints

- 全部 spec 文档使用中文撰写；数学符号使用 LaTeX（行内 `$...$` 或行间 `$$...$$`）。
- 符号定义以 `00` 为唯一裁定来源：$c = (c_{\text{low}}, c_{\text{mid}}, c_{\text{high}})$，`H` 为 $256 \times 256$ 真值，`L` 为 $64 \times 64$ 观测，`P2` 为默认中阶图像先验。
- 第一版训练不使用物理损失（只空域-频域混合重建损失）；物理评估必须保留，且物理指标优先级高于图像指标。
- 三方案公平：相同数据、相同损失、相同主干、相同训练配置、相同评估指标。
- 数据归一化约定与 `40` 一致：总强度归一化下方案 C 的 $c_{\text{prior}}$ 不含 `A`。
- 可复现：随机种子、配置、数据版本、Spec 版本均须记录。
- 冻结后的 Spec 不得在实现阶段修改；需要变更时按 `00` `[S9]` 的规则执行。

## Out of Scope

- 高分辨率真值生成（`20`）、低分辨率退化（`30`）、先验生成（`40`）、网络结构（`50`）、评估指标（`70`）。
- 传统图像增强（随机旋转、翻转、裁剪、颜色扰动、强几何变形）。
- 先验扰动增强（参数噪声、参数偏差、不完美先验；第一版，预留后续实验）。
- OOD 评估（EXP-05/06）第一版可选（Phase 4 Could）；OOD 数据生成与保留为第一版必做（由 `80` 安排）。
- 物理损失参与训练（第一版；物理约束只在评估阶段检查）。
- 更充分规模（50,000–100,000）主实验（第一版以标准 demo 规模为主）。

## Decisions

### D1: 训练损失固定为空域-频域混合损失
**Chosen:** 三方案统一使用空域-频域混合损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{space}} + \lambda \, \mathcal{L}_{\text{spec}}$，其中 $\mathcal{L}_{\text{spec}}$ 为倍频程分带等权加权谱 L1（分带与 $f_c$、$f_N$ 定义见 `[S2]`）；$\lambda$ SHALL 冻结为 1.0（预注册常数，不做代理选择）。用户确认（2026-08-25）。
**Rationale:** 纯像素级 L1 的条件中值解会平滑掉精细结构，且频谱偏差使网络优先拟合低频；空域项保留逐像素保真与对噪声的稳健性，频域项以等权分带显式约束各频段，抑制低频主导；两项均不引入额外先验，保持三方案公平比较。依据：wang2026_breaking-scale-anchoring（ICLR 2026，归档于 `docs/wang2026_breaking-scale-anchoring_2512.05132v2.pdf`）。
**Rejected:**
- 纯像素级 L1 — 条件中值解平滑掉精细结构，频谱偏差使网络优先拟合低频（依据：wang2026_breaking-scale-anchoring，ICLR 2026，归档于 `docs/wang2026_breaking-scale-anchoring_2512.05132v2.pdf`）；
- L2 / MSE — 对噪声与尖峰更敏感，易被少数大误差主导；
- SSIM loss — 引入额外结构先验；
- LPIPS / perceptual loss — 依赖外部特征，不利于解释；
- GAN loss — 容易产生幻觉；
- 物理矩损失与 forward consistency — 第一版决定不加入（见 `40` D7）。

### D2: 图像总强度归一化并从 `c_prior` 移除 `A`
**Chosen:** 第一版对 `H`、$L_{\text{up}}$、$P_2$ 做总强度归一化（各图像分别归一化到总和 1），并将 `A` 从方案 C 的 $c_{\text{prior}}$ 中移除（或固定为常数），$c_{\text{prior}} = \{\sigma_z, n, \eta, b_0, a_1, \alpha, a_2, \beta\}$。用户确认。
**Rationale:** 训练稳定，避免绝对强度主导损失；总强度归一化后 `A` 不再提供有效信息，保留只会引入冗余；与 `40` 的归一化约定一致。
**Rejected:**
- 不归一化、保留绝对强度信息 — 与归一化方案互斥；第一版选择归一化换取训练稳定，此时 `A` 退化为常数。

### D3: 训练输入使用带噪声 `L`
**Chosen:** 主训练输入为 $L = L_{\text{clean}} + n$（$n \sim \mathcal{N}(0, \sigma_n^2)$），训练目标为干净 `H`。用户确认。
**Rationale:** 更接近真实诊断信号；防止网络只学习去模糊；能测试先验在噪声下的作用；避免任务过于简单。
**Rejected:**
- 使用 $L_{\text{clean}}$ 作为主训练输入 — 任务过于简单，网络只需学去模糊，无法检验先验在噪声下的价值；$L_{\text{clean}}$ 仅保留用于调试、消融与后续实验。

### D4: 第一版不做传统图像增强，改用参数采样增强
**Chosen:** 第一版不使用传统图像增强；通过采样不同物理参数 `c`、不同退化参数 `d` 与不同噪声实现 $n$ 获得数据多样性。先验扰动增强为后续实验，第一版不加入。用户确认。
**Rationale:** 图像是物理相空间密度，坐标 `z`、`δ` 方向有物理意义，旋转/翻转/裁剪会破坏物理含义；参数采样增强物理分布而非视觉纹理，既扩大训练分布又保持物理一致性。
**Rejected:**
- 随机旋转 — 混合 `z` 与 `δ` 坐标，破坏坐标方向的物理意义；
- 随机翻转 — 可能改变 chirp 符号或头尾方向；
- 随机裁剪 — 破坏总强度与相空间范围；
- 颜色扰动 — 图像是物理密度而非 RGB 照片；
- 强几何变形 — 可能产生非物理相空间；
- 第一版加入先验扰动增强 — 属于后续鲁棒性实验，会混淆「先验价值」与「先验误差」的评估。

### D5: 数据集规模采用 20k / 2k / 2k 标准，先跑调试规模
**Chosen:** 标准主实验规模为 20,000 train / 2,000 val / 2,000 test；先以 2,000 / 500 / 500 调试规模跑通管线（shape、损失下降、输出非负），再用标准规模做主实验。用户确认。
**Rationale:** 先低成本验证训练管线与输出约束，再以标准规模比较 A / B / C 得到初步结论；避免直接大规模训练后发现 pipeline 问题。
**Rejected:**
- 直接使用 50,000–100,000 更充分规模做主实验 — 训练成本高；第一版结论以标准 demo 规模即可获得，更充分规模留给后续稳定结论与 OOD 分析。
