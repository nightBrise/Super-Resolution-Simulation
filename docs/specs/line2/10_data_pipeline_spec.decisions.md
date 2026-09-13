# Decision Log: 锚点数据管线（SDDS 导入 + 抖动束群 + 子采样配对 + FODO 传输 + 投影数据集）

<!-- ACTIVE LOG — continuously appended throughout lifecycle; do not treat as stale -->

## D1: 子采样配对 A ⊂ B，不做独立束流生成
**Phase:** write-spec
**Chosen:** 只构造全量束 B，低 N 输入从同一粒子表无放回子采样
**Rationale:** 独立生成会使输入/目标差异混入束间实现差异，污染放大度量；子采样使差异为纯同源抽样噪声，且配对在晶格传输后自动保持（确定性传输可交换）。
**Rejected:** 独立生成 1e4 与 2e5 两束分别跟踪 ——配对断裂、计算翻倍、度量含非抽样噪声。
**Spec section:** [S2]、[S3]
**Status:** live

## D2: 跟踪一次、逐站子采样
**Phase:** write-spec
**Chosen:** 只跟踪全量束一次，站 1 的低 N 输入取站 1 全量表的子采样
**Rationale:** 传输逐粒子确定，"先子采样再传输"与"先传输再子采样"逐位等价，单独跟踪子样本是纯冗余计算。
**Rejected:** A 与 B 各自过 lattice ——结果等价但计算翻倍，且引入两次运行一致性的无谓审计负担。
**Spec section:** [S3]
**Status:** live

## D3: 四极强度由目标相位推进反解
**Phase:** write-spec
**Chosen:** config 给 n_cells、目标 μ、半 cell 漂移 L_d，四极强度按薄透镜公式 cos μ = 1 − L_d²/(2f²) 反解，构建后以 Ocelot 实算校验偏差 < 1°
**Rationale:** 手调 K1 不可复现也不可审计；参数化后 lattice 是 config 的确定函数。
**Rejected:** 直接给定 K1 手调 ——μ 落在哪个值无契约保证，跨 run 不可比。
**Spec section:** [S3] C1
**Status:** live

## D4: 训练目标 = 2e5 全量直方图
**Phase:** write-spec
**Chosen:** 以锚点全量（2e5）直方图为收敛密度的操作化参照与训练目标
**Rationale:** 用户定案只用真实 SDDS 粒子（受控档移除），解析真值不可得；2e5 是可用最高 N（放大上限已诚实声明）。
**Rejected:** 密度拟合采样造更高档真值 ——拟合密度成为真值，循环性风险；更大初始文件 ——当前不存在，挂起待议。
**Spec section:** [S4]、[S5]
**Status:** live

## D5: 子采样无放回
**Phase:** write-spec
**Chosen:** 子采样无放回
**Rationale:** 有放回采样产生重复粒子，扭曲低 N 端的计数统计（重复格计数翻倍），且不增加任何信息。
**Rejected:** 有放回 bootstrap ——与"放大 = 去抽样噪声"的任务语义冲突。
**Spec section:** [S2] C2
**Status:** live

## D6: 数据集落盘 HDF5（每切分一文件）
**Phase:** write-spec
**Chosen:** 每切分一个 HDF5（gzip level 4），束为 group，沿 line1 数据惯例
**Rationale:** 与 line1 工具链（H5Dataset 读取模式、manifest 惯例）一致，层级元数据（beam/station/N_low/sub_idx）表达自然。
**Rejected:** npz ——无层级 group 语义，多站多档组织需要文件名编码，易碎。
**Spec section:** [S5] C2
**Status:** live

## D7: 分箱范围 = 全量表 [min,max] 外扩 1%，截断即报错
**Phase:** write-spec
**Chosen:** 分箱范围按站定标且覆盖全部粒子，任何截断 SHALL 报错
**Rationale:** ∑counts = N 是计数语义的基石（GC5），静默截断会破坏它并引入未被建模的信息损失。
**Rejected:** 固定 ±4σ 窗 ——截断尾部粒子，∑counts < N，违背守恒 Claim。
**Spec section:** [S4] C1
**Status:** live

## D8: 束群 = 抖动线性传输的实现（审核 B1 修复，2026-09-14 用户批准）
**Phase:** review
**Chosen:** 束 i = 锚点全量表经束级抖动线性变换（前置薄透镜 δ∈±5%、踢角 ±10 µrad、漂移 495–505 mm）+ 共同 FODO；L = 1000（700/150/150）；推广域 = 锚点分布的线性变换邻域
**Rationale:** 审核指出单张锚点表撑不起束级切分、"跨束中位数"与"新束泛化"；独立生成与有放回 bootstrap 均已被既有决策排除。抖动传输是"对既有粒子做确定性变换"，不违反"不从 Twiss 生成束流"的定案，且物理上对应真实机器的注入抖动（sxfel.ele 即含 jitter 项）。
**Rejected:** (a) 有放回 bootstrap 实现束群 ——全部束同分布同实现均值，切分/泛化语义退化为"新噪声种子"，且与 D5 的无放回立场冲突；(b) 维持单束、切分改按子样本 ——GC1 失效，测试污染（line1 伪重复教训）；(c) 多 FODO 配置当多束 ——破坏 GC4 同物理公平底线。
**Spec section:** [S3]
**Status:** live

## D9: 统一子采样网格（审核 B5/B6 修复，2026-09-14）
**Phase:** review
**Chosen:** 每束每站对五档 T = {1e3, 3e3, 1e4, 3e4, 1e5} 各 k = 20 个子样本全部落盘（int32，~80 GB）；训练消费三档、曲线消费五档；2e5 全量为唯一目标/参照、不抽样；曲线域去 2e5 节点
**Rationale:** 审核发现曲线档（6 档 × k=20）与数据集档（3 档 × m=8）契约矛盾；统一网格后 20 的"一键重现/只消费数据集"两条 Claim 自洽。2e5 档无放回子样本只有全量表自身（err 恒 0），保留即为死代码，故移出曲线域、N_eff 域改 [1e3, 1e5]。
**Rejected:** (a) 曲线档即时重算不入盘 ——把判据计算耦合到管线重跑，破坏"判据只读数据集"的接口；(b) 保留 2e5 节点 ——上溢分支死代码，20× 认证路径虚假。
**Spec section:** [S2]、[S5]、（联动 `20` [S2]/[S3]）
**Status:** live

## D10: lattice 定死 n_cells = 2、μ = 75°（审核 M1 修复，2026-09-14）
**Phase:** review
**Chosen:** 参数从范围改为定值，并加守恒排除规则 n_cells·μ ∉ 180° 整数倍
**Rationale:** 大纲委托 L1"定死"lattice；且一阶光学下 corr′ = ρ·cos(n_cells·μ)，当 n_cells·μ ∈ 180°ℤ 时站间 corr 不变（|Δcorr| = 0），允许配置空间内 [S3] C4 可失效、跨站检验失去内容。n_cells=2、μ=75° 下 corr′ ≈ −0.74，|Δcorr| ≈ 1.6。
**Rejected:** 保留范围 + 排除区间 ——"定死"是大纲明文要求；n_cells=4、μ=60°（总 240°，corr′ ≈ −0.43 亦可用）——75° 单 cell 旋转更充分，且 cell 数更少成本更低。
**Spec section:** [S3] C1、C4
**Status:** live

## Progress

| [Sn] | commit | review | status |
|---|---|---|---|
| [S1] | — | — | pending |
| [S2] | — | — | pending |
| [S3] | — | — | pending |
| [S4] | — | — | pending |
| [S5] | — | — | pending |
