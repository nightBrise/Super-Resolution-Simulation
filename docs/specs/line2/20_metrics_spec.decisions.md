# Decision Log: 判据与等效放大因子（N_eff）

<!-- ACTIVE LOG — continuously appended throughout lifecycle; do not treat as stale -->

## D1: N_eff 基于 M1 相对 RMSE + log-N 线性插值
**Phase:** write-spec
**Chosen:** 主度量取总计数归一后全图相对 RMSE 的六投影平均，N_eff 在 (log N′, err) 裸曲线上线性插值反解
**Rationale:** 单一度量满足主指标准绳（00 GC3）且插值可确定性复算；log-N 插值在跨越两个数量级的曲线上分段线性近似足够。
**Rejected:** 多度量几何平均定 N_eff ——违背主指标唯一性，权重不可预注册；最近邻取档 ——分辨率只到离散档，且对曲线噪声敏感。
**Spec section:** [S3]
**Status:** live

## D2: 参照 = 2e5 全量直方图
**Phase:** write-spec
**Chosen:** 同束同站全量（2e5）直方图作为收敛密度的操作化参照与误差参照
**Rationale:** 用户定案只用真实 SDDS 粒子（受控档移除），解析密度不存在；2e5 是可得的最高 N。
**Rejected:** 解析密度参照 ——真实档无解析形式；更高 N 参照 ——锚点只有 2e5（放大上限已声明）。
**Spec section:** [S1]
**Status:** live

## D3: 晕占比以参照定义的椭圆计算
**Phase:** write-spec
**Chosen:** 椭圆仅由参照的矩定义（Phase 1 为逐投影 2D 边缘椭圆，取参照 Σ 对应 2×2 子块）
**Rationale:** 椭圆随束的各向异性自适应；固定参照定义防止被评图组反向定义检验区域（定义漂移）。
**Rejected:** 固定矩形窗 ——对椭圆状束（本锚点 corr(x,x′)=0.855）失真；被评图组自定义区域 ——被评方可通过改变区域让自己达标。
**Spec section:** [S1] C4、[S1] M5
**Status:** live

## D4: 观测量容差线 = 等效 N′ 裸子样本偏差的 95% 分位
**Phase:** write-spec
**Chosen:** 容差随 N_eff 动态导出（最近曲线档映射），不使用全局固定百分比
**Rationale:** "满足要求"的语义在放大定调下就是"达到等效裸模拟水平"——固定百分比（如 5%）无先验依据且不随桶/束变化。
**Rejected:** 固定 5% 容差 ——暗桶（r<0.5）天然达不到，亮桶又过于宽松；拍脑袋且不可辩护。
**Spec section:** [S4]
**Status:** live

## D5: M6 高频能量比的 σ_smooth 固定 1 bin 并预注册
**Phase:** write-spec
**Chosen:** 高通平滑尺度固定为 1 bin，写入 spec 禁调
**Rationale:** σ_smooth 是 line1 全链教训的根源（0.5×w_fine 按构造抹杀细结构，P0 才修正）；若留成可调旋钮，调参期会把它调成对自己有利的值。
**Rejected:** σ_smooth 可调并参与调参 ——先验/幻觉的裁判参数不允许被被评对象优化。
**Spec section:** [S1] M6
**Status:** live

## D6: M1 增总计数归一（审核 B2 修复，2026-09-14 用户批准）
**Phase:** review
**Chosen:** 被评图组逐投影重标至参照总计数（x̃ = x·Σr/Σx）后计算 M1/M3/M5，归一内置单一实现点
**Rationale:** 审核算术证明：无归一的直比使完美网络（形状正确、总量 N_low）的误差 ≈ 裸 1e4 误差，N_eff ≈ N_low，H1 阈值数学不可达——协议自证失败。放大定调的目标是收敛密度的形状，总量差异正是要去掉的量。
**Rejected:** 反向把 30 的输出改为 2e5 语义 ——与训练语义（输入 N_low、学 N_low 粒子的收敛形状）脱节，且放大倍数声明失去载体。
**Spec section:** [S1]、[S3]（联动 `30` [S2]）
**Status:** live

## D7: 曲线域去 2e5 节点 + 参照噪声地板（审核 B6 修复，2026-09-14 用户批准）
**Phase:** review
**Chosen:** 曲线节点 = {1e3, 3e3, 1e4, 3e4, 1e5}；N_eff 报告域 [1e3, 1e5]，上溢语义 = 达到可验证域顶（1e4 工况 10×）；以 1e5 档独立子样本两两配对的 M1 差中位数 / √2 报告参照噪声地板
**Rationale:** 无放回下 2e5 子样本唯一（= 参照自身），该档误差恒 0——插值域顶端是死点，"N_eff ≥ 2e5"为不可达分支；参照自身噪声使 >1e5 的差异不可分辨，如实声明地板比假装 20× 可测更诚实。
**Rejected:** 保留 2e5 节点 + 有放回子采样制造 2e5 档散点 ——重复粒子不产生新信息，散点是假的。
**Spec section:** [S2]、[S3] C2（联动 `10` D9）
**Status:** live

## D8: M4 近零分量改相关系数变化（审核 B4 修复，2026-09-14）
**Phase:** review
**Chosen:** 6 个非对角分量报相关系数变化 |Δρ|（有界、处处良定义），4 个对角分量保留相对误差
**Rationale:** 锚点横向零耦合 + P1 线性 FODO 不产生耦合 → 跨面 Σ 分量恒 ≈ 0，相对误差为 0/0（NaN），H8 达标率分母失效。相关系数在分量恒零时仍良定义。
**Rejected:** 相对误差加 floor（|Δ|/max(|ref|, ε)） ——floor 取值本身成为无依据旋钮；|Δρ| 是无量纲的标准统计量，无需旋钮。
**Spec section:** [S1] M4
**Status:** live

## D9: M5 改逐投影 2D 边缘椭圆（审核 B3 修复，2026-09-14）
**Phase:** review
**Chosen:** 每投影用参照 Σ 的 2×2 子块 + 投影均值定义 2D 椭圆，六投影平均
**Rationale:** 4D 椭圆需要 4D 粒子坐标，而 Phase 1 全部接口（数据集、基线、网络输出）均为 2D 图组——4D 版本在契约内不可计算。2D 边缘椭圆保留"参照定义防漂移"的全部语义。
**Rejected:** 显式引入 4D 直方图数据面 ——改动 10/20/30 三份契约，且 Phase 1 网络根本不输出 4D 信息。
**Spec section:** [S1] M5、[S1] C4
**Status:** live

## D10: 参照噪声语义声明（审核 M8 补录，2026-09-14）
**Phase:** review
**Chosen:** [S2] 显式声明参照（2e5 实现）相对收敛密度的残余噪声及其操作化地板（1e5 档独立子样本配对 M1 差/√2），并入曲线报告
**Rationale:** 大纲 L2 委托项"残余 label noise 可接受，写明"；该地板同时是 N_eff 高端解读的诚实边界。
**Rejected:** 不声明直接用 2e5 当真值 ——读者会把 N_eff 逼近 1e5 误读为"逼近绝对真值"。
**Spec section:** [S2]
**Status:** live

## D11: M2 桶界 {0.5, 2, 10} 的数值来源（审核 m7 补录，2026-09-14）
**Phase:** review
**Chosen:** 四桶边界取 {0.5, 2, 10}，与"暗桶（近不可见）/过渡/中等/亮核"四个机制区对齐，冻结于 spec 批准
**Rationale:** 锚点被占格均值 34（2e5）下，1e4 输入的暗桶期望计数恰落在 0–2 量级，桶界把"先验主导区"与"数据约束区"分开，使 H8 的分桶报告有机制含义。
**Rejected:** 等宽计数分桶 ——亮核一桶独大，暗桶信息被吞。
**Spec section:** [S1] M2
**Status:** live

## Progress

| [Sn] | commit | review | status |
|---|---|---|---|
| [S1] | — | — | pending |
| [S2] | — | — | pending |
| [S3] | — | — | pending |
| [S4] | — | — | pending |
| [S5] | — | — | pending |
