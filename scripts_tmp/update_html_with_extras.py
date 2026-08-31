"""增量注入 11 张新图到现有 HTML 报告（按 anchor 锚点）。"""
from __future__ import annotations
import base64, re
from pathlib import Path

ROOT = Path("/home/zhangny/Super-Resolution-Simulation")
HTML = ROOT / "M4_stage_report.html"
EXTRA = ROOT / "results/EXP-02_summary/figures_extra"

NEW_FIGURES = [
    ("fig01_m1_sample.png", "为何不用真实仿真器？",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 1 · M1 生成函数样本（H / 高频放大 / CSR 微聚束）</b>。左图为真值 H（256×256，log 色标显示精细结构：压缩折叠脊、CSR 微聚束波纹、峰值尖峰）；中图为 H 经过 DoG 高通滤波（σ=8）后取绝对值的 64×64 窗口放大——展示 c_high 贡献的"高频精细结构"在 H 内的具体形态（viridis 色标凸显结构纹理）；右图为沿 z 中心行的电流剖面（实线：原始 H(z)；虚线：σ=2 高斯光滑后的低频包络）—— 两线之差即 CSR 微聚束贡献的精细波纹。这张图直接展示生成函数"产出了什么"——是后续所有训练/评估的源头。</figcaption>\n  </figure>'),
    ("fig02_degradation.png", "退化协议：如何制造低分辨率观测？",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 2 · 退化协议四步示意</b>。从左到右：(a) 真值 H（256×256, ΣH=1）；(b) 加 σ_K=11.0 高斯模糊（模拟探测器点扩散函数）；(c) 4× 块求和下采样（256→64，物理含义：探测器像素积分）；(d) 加 σ_n=1.22e-4 高斯噪声（模拟探测器噪声，RdBu_r 色标显示正负偏差）；(e) 非负截断 max(0,·) + 双线性上采样到 256×256（送入网络的 L_up）。这张图直接展示 H 是如何一步步变成网络输入 L_up 的——也展示了"哪些信息被退化"：高频精细结构（CSR 微聚束、压缩折叠脊）主要在第 (c) 步（块求和平均）和第 (b) 步（高斯模糊）丢失，第 (d) 步只是叠加噪声。</figcaption>\n  </figure>'),
    ("fig11_sigma_smooth_revision.png", "退化协议：如何制造低分辨率观测？",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 11 · σ_smooth,H 修订前后 H 高频能量对比</b>。σ_smooth 是生成函数中"光滑渲染"步骤的高斯核尺度——决定 H 中精细结构（c_high 贡献）的保留程度。修订前 σ_smooth,H=0.5×w_fine 导致 H 高频能量仅 1.90%（精细结构被大量抹弱，H 看起来"太平滑"）；修订后 σ_smooth,H=0.125×w_fine 保留 7.16%，<span class="accent">高频能量提升 3.8 倍</span>。这是 2026-08-26 P0 报批包的关键根因修订——之前的 G0 探针"高频存活比"偏低、EXP-01 R_E 比率门失败等问题，最终都追溯到 σ_smooth,H 太大。</figcaption>\n  </figure>'),
    ("fig03_schemes.png", "三个方案可视化",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 3 · 三方案架构对比框图</b>。从顶到底：A 无先验 baseline（仅 L_up → U-Net → Ĥ）；B 图像先验 + 早期融合 + 残差（L_up 和 P₂ 在 2 通道早期拼接，U-Net 学残差 R，输出 Softplus(S·L_up+R)）；C 参数先验 + FiLM（L_up + 8 维 c_prior → z-score+MLP 编码 → FiLM 注入 U-Net 瓶颈+解码器）。<span class="accent">三方案共享同一 U-Net 主干、同一损失、同一训练配置</span>，唯一变量是先验信息的注入方式——这是"公平对照"的实现保障。颜色编码：绿色=输入端先验，蓝色=参数/参考端先验，青色=融合/编码，黄色=FiLM 注入（方案 C 独有），黑色=网络核心，红色=最终预测。</figcaption>\n  </figure>'),
    ("fig10_milestones.png", "整个研究分 6 个里程碑（M0–M6）",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 10 · M0–M6 里程碑时间线</b>。横向时间线展示研究执行的 7 个里程碑（M0–M6）：M0（Spec 冻结）→ M1（生成器）→ M2（数据集）→ M3（baseline）→ M4（主实验，当前进行中，红色高亮）→ M5（消融）→ M6（最终报告）。当前进度 4/7 里程碑完成，<span class="warn">M4 处于"G2 失败路径"状态</span>（跨种子三分类不一致，已批准扩种子至共 4 个，预计 ~24h 墙钟完成）。圆点颜色：绿色=已完成、红色=进行中、灰色=待启动。</figcaption>\n  </figure>'),
    ("fig04_training_curves.png", "本次六个 run 中三个哨兵全过。",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 4 · EXP-02 6 run 训练损失曲线</b>。横轴训练步数（0–50,000），纵轴验证损失 val_loss（工作尺度 N²=65536）。6 条曲线：实线 = seed0，虚线 = seed1；颜色编码：A 蓝、B 青、C 马鞍棕（与方案语义一致）。<span class="accent">所有 6 条曲线均收敛</span>，C 方案两种子最优（best 0.0180/0.0181），B 次之（0.0260/0.0264），A 最差（0.0308/0.0306）。注意：val_loss 是训练目标，<span class="warn">不等同于主指标 e_high_mask</span>——C 在 val_loss 最优但主指标跨种子翻转（见 §9）——这是研究方法论中"训练目标 vs 评估目标"差异的典型案例。</figcaption>\n  </figure>'),
    ("fig05_g1b_A_vs_Lup.png", "本次 A 相对 L_up 增益 +0.027（CI 全正），G1(b) 充分通过。",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 5 · G1(b) 零学习基线：A 网络预测 vs L_up 直出（test_id, n=1000/种子）</b>。直方图叠加显示：A 网络预测（马鞍棕）的 e_high_mask 均值（0.00126 / 0.00148）显著低于 L_up 直出（蓝色，均值 ~0.028）。配对差 d = L_up − A = +0.027（两种子均 positive，CI 全正，p≈0）——这证明<span class="accent">网络确实从训练中学到了有用的高频信息</span>，远超"零学习"基线。G1(b) 充分通过意味着：方案 B/C 与 A 的对比才有意义（A 不是 trivial 输出，可以作为"先验是否有用"的对照基准）。</figcaption>\n  </figure>'),
    ("fig06_g2_verdicts.png", "G2 三分类结果（test_id 主判定）",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 6 · G2 三分类结果热力图（3 对对比 × 2 种子）</b>。3 行（y−x 对：A−B, A−C, B−C）× 2 列（seed0, seed1），每个 cell 显示 verdict 文字与 CI 区间。颜色编码：<span class="warn">红色=显著正</span>（y 误差 > x 误差，即 x 更优）、白色=等效、绿色=显著负（y 更优）。<span class="warn">★ 三对对比全跨种子不一致</span>：A−B（seed0 等效、seed1 B 优）；A−C（seed0 C 优、seed1 A 优——符号翻转）；B−C（seed0 C 优、seed1 B 优——符号翻转）。按 70 [S7] C9 → G2「诊断不确定」（跨种子不一致）→ 80 [S9] G2 失败路径触发。</figcaption>\n  </figure>'),
    ("fig07_cross_seed_boxplot.png", "归因诊断",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 7 · e_high_mask 跨种子分布对比（log 尺度箱线图）</b>。6 个箱体：A_seed0/1（蓝）、B_seed0/1（青）、C_seed0/1（马鞍棕）。<span class="warn">★ C 方案的核心现象</span>：标准差跨种子几乎恒定（0.0006，均很窄），但<span class="warn">均值跨种子漂移 51%</span>（seed0=0.00106 → seed1=0.00160，标注的两条红色箭头指示）。A/B 的箱体跨种子接近，A 略有上漂（+18%），B 几乎不变（−6%）——说明 <span class="accent">C 方案（FiLM）对训练初始化高度敏感</span>，不同的 torch.manual_seed 让 FiLM 收敛到不同盆地，导致整体误差水平平移。</figcaption>\n  </figure>'),
    ("fig08_mask_composition.png", "掩膜成分分解披露（强制）",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 8 · 主指标掩膜成分分解 + 先验泄漏指数</b>。三柱：ch_in_mask=0.52（c_high 差分占比）、b_in_mask=1.14（β 差分占比）、Π_leak=2.40（先验泄漏指数）。阈值线：<span class="warn">b_in 触发阈值 = 1.5×ch_in = 0.78</span>（实测 1.14 超过 47%）；<span class="warn">Π_leak 触发阈值 = 0.5</span>（实测 2.40 超过 380%）。结论：<span class="warn">主指标掩膜内 β 能量是 c_high 差分的 2.2 倍，且先验 P₂ 与真值之差在掩膜内的能量比 = 2.40</span>—— 图像先验 P₂ 在主指标区域结构性占优。这意味着 C 方案的"主指标优势"包含两部分信号：(a) 更好地恢复 c_high 精细结构；(b) 更好地呈现 P₂ 的低阶结构。两者需要进一步分离（M5 增益分解）。</figcaption>\n  </figure>'),
    ("fig09_merged_ci_coverage.png", "合并 CI 的致命弱点 ★",
     '<figure class="figure-frame">\n__FIG__\n    <figcaption><b>图 9 · 合并 bootstrap CI 覆盖率警示</b>。Qwen 3.8 Max 通过 3000 次重复模拟证明：单 seed 的 bootstrap CI（n=1000，1000 次重复）覆盖率 95%——符合标称值；而合并 2 seed 的 CI（n=2000，3000 次重复，种子间不同分布）覆盖率仅 5%——远低于标称 95%。原因：合并时把两 seed 样本当 i.i.d.，但实际来自两个不同分布——配对差被错误平均，CI 变得过窄、过乐观。<span class="warn">结论：当前合并 CI 不可作判定依据</span>，仅作描述量；判定必须以<span class="accent">逐种子 verdict 为准</span>。这条警示直接影响 G2 三分类结论的解读——seed1 的 B−C 显著负（verdict）可信，seed0 的 B−C 显著正（verdict）也可信，但合并的"显著负"是伪信号。</figcaption>\n  </figure>'),
]


def b64(p):
    return base64.b64encode(p.read_bytes()).decode()


def img_tag(name):
    return f'<img src="data:image/png;base64,{b64(EXTRA / name)}" alt="{name}" style="display:block; max-width:900px; width:auto; height:auto; margin:32px auto; border-radius:16px; border:1px dashed #E8E4DC; box-shadow: 0 4px 24px rgba(0,0,0,0.04);" />'


def insert_after_anchor(t, anchor, snippet):
    """在 anchor 字符串后的第一个 </p> 后插入 snippet。"""
    idx = t.find(anchor)
    if idx < 0:
        return t, False
    # 找 anchor 后最近的 </p>
    end_p = t.find("</p>", idx)
    end_h3 = t.find("</h3>", idx)
    candidates = [x for x in [end_p, end_h3] if x > 0]
    if not candidates:
        return t, False
    pos = min(candidates) + len("</p>" if end_p < end_h3 else "</h3>")
    t = t[:pos] + "\n\n  " + snippet + t[pos:]
    return t, True


t = HTML.read_text()
inserted = 0
for fname, anchor, snippet_tmpl in NEW_FIGURES:
    snippet = snippet_tmpl.replace("__FIG__", img_tag(fname))
    t, ok = insert_after_anchor(t, anchor, snippet)
    if ok:
        print(f"✓ inserted {fname} after '{anchor[:40]}'")
        inserted += 1
    else:
        print(f"✗ anchor '{anchor[:40]}' NOT FOUND, skip {fname}")

HTML.write_text(t)
print(f"\n=== inserted {inserted}/11 figures; total HTML {len(t.encode('utf-8'))/1024:.0f} KB ===")