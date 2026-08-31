"""生成 M4 HTML 阶段报告：单文件、内嵌 base64 PNG、温润学术风。

约定：
- 路径 = /home/zhangny/Super-Resolution-Simulation/M4_stage_report.html（与 AGENTS.md 同级）
- 5 张 PNG 从 results/EXP-02_summary/assets/ 读 → base64 → img src
- 不引用 spec 章节号（让外部读者能看懂）；专业名词可用英文
- 平衡学术通俗；大量图 + 详细图注
"""
from __future__ import annotations
import base64
from pathlib import Path

ROOT = Path("/home/zhangny/Super-Resolution-Simulation")
ASSETS = ROOT / "results/EXP-02_summary/assets"
OUT = ROOT / "M4_stage_report.html"

FIGURES = [
    ("figure_01_2d_phasespace.png", "2D 相空间三联图（H / L_up / Ĥ）"),
    ("figure_02_1d_profile.png", "1D 电流/能谱剖面对比"),
    ("figure_03_physics_error_bar.png", "5 个物理量相对误差柱状图"),
    ("figure_04_error_vs_gamma_scatter.png", "主指标 vs 压缩态 γ 散点图"),
    ("figure_05_residual_map.png", "预测-真值残差热力图"),
]


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


# 预编码
b64_imgs = {name: b64(ASSETS / name) for name, _ in FIGURES}

# ----------- HTML 主体 -----------

NAV_ITEMS = [
    ("sec-overview", "概览"),
    ("sec-question", "研究问题"),
    ("sec-design", "总体设计"),
    ("sec-data", "数据生成"),
    ("sec-model", "网络架构"),
    ("sec-train", "训练策略"),
    ("sec-eval", "评估判据"),
    ("sec-progress", "实验进度"),
    ("sec-results", "当前结果"),
    ("sec-problem", "当前问题"),
    ("sec-next", "下一步"),
]

nav_html = "\n".join(
    f'<a href="#{aid}" class="nav-item"><span class="nav-num">{i+1:02d}</span>{name}</a>'
    for i, (aid, name) in enumerate(NAV_ITEMS)
)


def img(name: str, alt: str, max_w: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64_imgs[name]}" alt="{alt}" style="display:block; width:{max_w}; max-width:900px; height:auto; margin:32px auto; border-radius:16px; border:1px dashed #E8E4DC; box-shadow: 0 4px 24px rgba(0,0,0,0.04);" />'


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>M4 主实验阶段报告 · 物理先验引导的纵向相空间超分辨率研究</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Serif+SC:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet" />
<style>
  /* 温润学术风（warm-academic），参考 Report-Prepare/美术设计规范.md */
  :root {
    --bg-primary: #F5F0E8;
    --bg-secondary: #EDE8DF;
    --bg-card: #FAF8F5;
    --ink-strong: #1A1A1A;
    --ink-body: #333333;
    --ink-sub: #4A4A4A;
    --ink-soft: #666666;
    --ink-faint: #E8E4DC;
    --accent-pine: #2D5A4A;
    --accent-saddle: #8B4513;
    --r-sm: 8px;
    --r-md: 16px;
    --r-lg: 32px;
    --shadow-soft: 0 4px 24px rgba(0,0,0,0.04);
    --shadow-hover: 0 2px 12px rgba(0,0,0,0.06);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    background: var(--bg-primary);
    color: var(--ink-body);
    font-family: "Noto Serif SC", Georgia, "Times New Roman", serif;
    font-size: 18px;
    line-height: 1.9;
    font-weight: 400;
  }
  body::before {
    content: "";
    position: fixed; inset: 0;
    pointer-events: none; z-index: 0;
    opacity: 0.03;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
  }
  /* 导航栏 */
  nav.top {
    position: sticky; top: 0; z-index: 50;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--ink-faint);
    padding: 0 80px;
  }
  nav.top .wrap {
    max-width: 1200px; margin: 0 auto;
    display: flex; justify-content: space-between; align-items: center;
    height: 72px;
  }
  nav.top .logo {
    font-family: Inter, sans-serif;
    font-size: 13px; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--ink-strong); text-decoration: none;
  }
  nav.top .meta {
    font-family: Inter, sans-serif;
    font-size: 12px; color: var(--ink-soft);
    letter-spacing: 0.05em;
  }
  /* 侧边栏 */
  .layout { display: flex; max-width: 1200px; margin: 0 auto; padding: 0 24px; gap: 48px; }
  aside.sidebar {
    width: 240px; flex-shrink: 0;
    position: sticky; top: 96px;
    align-self: flex-start;
    height: calc(100vh - 120px); overflow-y: auto;
    padding: 24px 0;
  }
  aside.sidebar .nav-item {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px;
    font-family: Inter, sans-serif;
    font-size: 14px; color: var(--ink-sub);
    text-decoration: none;
    border-radius: var(--r-sm);
    transition: all 0.3s;
    border-left: 2px solid transparent;
  }
  aside.sidebar .nav-item:hover { background: var(--bg-card); color: var(--ink-strong); }
  aside.sidebar .nav-item.active {
    color: var(--ink-strong); font-weight: 600;
    border-left-color: var(--accent-pine);
    background: var(--bg-card);
  }
  aside.sidebar .nav-num {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; color: var(--ink-soft);
    min-width: 20px;
  }
  /* 主内容 */
  main {
    flex: 1; min-width: 0;
    max-width: 880px;
    padding: 48px 0 120px;
  }
  /* Hero */
  section.hero {
    padding: 80px 0 64px;
    border-bottom: 1px solid var(--ink-faint);
    margin-bottom: 80px;
  }
  section.hero .eyebrow {
    font-family: Inter, sans-serif;
    font-size: 13px; color: var(--accent-pine);
    letter-spacing: 0.18em; text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 24px;
  }
  section.hero h1 {
    font-family: Inter, sans-serif;
    font-size: 48px; font-weight: 600;
    line-height: 1.15;
    color: var(--ink-strong);
    margin: 0 0 24px;
  }
  section.hero .subtitle {
    font-family: "Noto Serif SC", Georgia, serif;
    font-size: 19px; color: var(--ink-sub);
    line-height: 1.7;
    max-width: 720px;
  }
  section.hero .stamp {
    margin-top: 32px;
    display: inline-block;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-radius: var(--r-sm);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px; color: var(--ink-sub);
    letter-spacing: 0.05em;
  }
  /* 章节 */
  section.chapter {
    padding: 64px 0 40px;
    border-top: 1px solid var(--ink-faint);
    margin-top: 64px;
    scroll-margin-top: 96px;
  }
  section.chapter:first-of-type { border-top: none; margin-top: 0; }
  section.chapter .chapter-no {
    font-family: "JetBrains Mono", monospace;
    font-size: 14px; color: var(--accent-pine);
    letter-spacing: 0.2em;
    margin-bottom: 12px;
  }
  section.chapter h2 {
    font-family: Inter, sans-serif;
    font-size: 36px; font-weight: 600;
    color: var(--ink-strong);
    margin: 0 0 32px;
    line-height: 1.25;
  }
  section.chapter h3 {
    font-family: Inter, sans-serif;
    font-size: 22px; font-weight: 600;
    color: var(--ink-strong);
    margin: 40px 0 16px;
  }
  section.chapter p {
    margin: 0 0 1.4em;
    color: var(--ink-body);
  }
  /* 首字下沉 */
  section.chapter p.lead::first-letter {
    font-family: "Noto Serif SC", Georgia, serif;
    font-size: 56px; line-height: 1;
    float: left; padding-right: 8px; padding-top: 4px;
    color: var(--accent-saddle);
    font-weight: 600;
  }
  section.chapter .figure-frame {
    margin: 40px 0;
  }
  section.chapter figcaption {
    font-family: "Noto Serif SC", Georgia, serif;
    font-size: 15px; color: var(--ink-sub);
    line-height: 1.75;
    padding: 16px 24px;
    background: var(--bg-card);
    border-left: 3px solid var(--accent-pine);
    border-radius: 0 var(--r-md) var(--r-md) 0;
    margin-top: -16px;
  }
  section.chapter figcaption b { color: var(--ink-strong); }
  /* 卡片 */
  .card-grid {
    display: flex; flex-wrap: wrap; gap: 20px;
    margin: 32px 0;
  }
  .card {
    flex: 1; min-width: 280px;
    background: var(--bg-card);
    border: 1px dashed var(--ink-faint);
    border-radius: var(--r-md);
    padding: 28px;
    transition: all 0.3s;
  }
  .card:hover { transform: translateY(-2px); box-shadow: var(--shadow-hover); }
  .card h4 {
    font-family: Inter, sans-serif;
    font-size: 18px; font-weight: 600;
    color: var(--ink-strong);
    margin: 0 0 12px;
  }
  .card p {
    font-size: 15px; color: var(--ink-sub);
    line-height: 1.7; margin: 0;
  }
  /* 引用块 */
  blockquote {
    margin: 32px 0;
    padding: 24px 32px;
    background: var(--bg-secondary);
    border-left: 3px solid var(--accent-pine);
    border-radius: 0 var(--r-md) var(--r-md) 0;
    font-style: italic;
    color: var(--ink-sub);
  }
  blockquote cite {
    display: block;
    margin-top: 12px;
    font-size: 14px; font-style: normal;
    color: var(--ink-soft);
  }
  /* 表格 */
  table {
    width: 100%; border-collapse: collapse;
    margin: 24px 0;
    font-family: Inter, sans-serif;
    font-size: 14px;
  }
  table th, table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--ink-faint);
    text-align: left;
  }
  table th {
    background: var(--bg-secondary);
    color: var(--ink-strong);
    font-weight: 600;
  }
  table tr:last-child td { border-bottom: none; }
  /* 代码 */
  code {
    font-family: "JetBrains Mono", monospace;
    font-size: 0.92em;
    background: var(--bg-secondary);
    padding: 1px 6px;
    border-radius: 4px;
    color: var(--ink-strong);
  }
  pre {
    background: var(--bg-secondary);
    border: 1px solid var(--ink-faint);
    border-radius: var(--r-md);
    padding: 20px;
    overflow-x: auto;
    font-family: "JetBrains Mono", monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--ink-body);
  }
  /* 强调 */
  .highlight {
    background: linear-gradient(transparent 60%, rgba(45,90,74,0.18) 60%);
    padding: 0 2px;
  }
  .accent { color: var(--accent-pine); font-weight: 600; }
  .warn { color: var(--accent-saddle); font-weight: 600; }
  /* 时间线 */
  .timeline {
    position: relative;
    margin: 32px 0;
    padding-left: 28px;
    border-left: 2px solid var(--ink-faint);
  }
  .timeline .ev {
    position: relative;
    padding: 12px 0 24px 16px;
  }
  .timeline .ev::before {
    content: "";
    position: absolute;
    left: -33px; top: 18px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--bg-primary);
    border: 2px solid var(--accent-pine);
  }
  .timeline .ev.current::before { background: var(--accent-pine); }
  .timeline .ev .when {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px; color: var(--ink-soft);
    letter-spacing: 0.05em;
  }
  .timeline .ev .what {
    font-family: Inter, sans-serif;
    font-size: 16px; font-weight: 600;
    color: var(--ink-strong); margin-top: 4px;
  }
  .timeline .ev .who {
    font-size: 14px; color: var(--ink-sub); margin-top: 4px;
  }
  /* 响应式 */
  @media (max-width: 1024px) {
    aside.sidebar { display: none; }
    .layout { padding: 0 24px; }
    nav.top { padding: 0 24px; }
    section.hero h1 { font-size: 36px; }
    section.chapter h2 { font-size: 28px; }
  }
  /* scroll-spy 用 IntersectionObserver */
</style>
</head>
<body>
<nav class="top">
  <div class="wrap">
    <a href="#sec-overview" class="logo">M4 STAGE REPORT · FEL SR STUDY</a>
    <div class="meta">2026-08-28 · 阶段报告 · 单文件可下载版</div>
  </div>
</nav>
<div class="layout">
  <aside class="sidebar" id="sidebar">
__NAV_HTML__
  </aside>
  <main>

<!-- ============ HERO ============ -->
<section class="hero" id="sec-overview">
  <div class="eyebrow">阶段报告 · M4 · 物理先验引导的超分辨率研究</div>
  <h1>把束流的"模糊快照"变成可读的高清诊断图</h1>
  <p class="subtitle">我们用三种神经网络方案——纯数据驱动、给一张低阶参考图、给一组物理参数——公平对比谁能更好地从压缩折叠的低分辨率纵向诊断图中恢复高分辨率相空间细节。本文讲述研究问题、为什么这样设计、怎么做、做到了哪一步、卡在了哪里。</p>
  <div class="stamp">M1–M3 完成 · M4 主实验进行中 · 阶段过渡版报告</div>
</section>

<!-- ============ 1. 研究问题 ============ -->
<section class="chapter" id="sec-question">
  <div class="chapter-no">01 / 研究问题</div>
  <h2>为什么"看清束流"这件事这么难？</h2>

  <p class="lead">在大型自由电子激光（FEL）装置里，电子束团被压缩到极致后沿着加速器一路飞驰。我们关心的"纵向相空间"，就是描述束团内部电子沿飞行方向的位置与能量分布——只有看清这张图，才能判断束流品质是否合格、压缩是否到位、未来能否高效激发激光。</p>

  <p>问题在于：诊断仪器的分辨率是有限的。横向偏转腔（TDS）+ 能谱仪只能给出<strong>低分辨率</strong>的束流图像——大致形状能看到，但精细结构（压缩折叠脊、电流微聚束波纹、峰值尖峰）<span class="highlight">全部糊在一起</span>。高分辨率下才会出现的精细结构，对评估束流品质恰恰最关键。</p>

  <h3>一个朴素的想法：让 AI 把模糊变清晰</h3>
  <p>这就是"超分辨率"问题。给定一张低分辨率观测 <code>L</code>，训练一个神经网络 <code>f</code>，让它输出对应的高分辨率图像 <code>Ĥ ≈ H</code>。图像超分辨率在自然图像、视频压缩、医学影像里已经被研究得很成熟，为什么不能直接套到束流诊断上？</p>

  <h3>两个核心障碍</h3>
  <div class="card-grid">
    <div class="card">
      <h4>① Scale Anchoring（尺度锚定）</h4>
      <p>神经网络被训练去拟合低分辨率数据，它对高频细节是"天然盲"的——因为低分辨率图像本身就不包含高于 Nyquist 频率的物理信息。当把训练好的网络部署到高分辨率任务上，它的误差不会像传统数值求解器那样随分辨率提高而下降，而是<span class="warn">锚定在低分辨率误差水平</span>。具体到我们的任务：网络会把压缩折叠脊"脑补"成一片模糊的包络，看起来"还行"，但关键峰值电流 <code>I_peak</code>、发射度 <code>ε_z</code> 都错了。</p>
    </div>
    <div class="card">
      <h4>② 物理幻觉（Physical Hallucination）</h4>
      <p>如果网络生成的图"看起来清晰"但峰值位置错误、能量分布错误——这种<span class="warn">看起来对、其实错</span>的输出是研究中最危险的情形。它不会被任何"看起来像不像"的损失抓住，反而可能让研究人员误判束流品质。</p>
    </div>
  </div>

  <h3>一个直觉解：给 AI 一点"物理常识"</h3>
  <p>人类看束流图时会用物理经验——比如"束团整体能量 100 MeV"、"压缩后峰值电流应该 ~3 kA"。这些物理参数在低分辨率观测中是<span class="accent">可测</span>的，如果我们把这些参数告诉网络，让它"看图说话"时多一条线索，是否就能减轻尺度锚定、避免物理幻觉？</p>

  <p>这就是本项目的核心问题——<strong>"低分辨率观测下，物理先验能否帮助网络更准确、更符合物理地恢复高分辨率纵向相空间？"</strong>我们需要用严格公平的三方案实验给出定量回答。</p>

</section>

<!-- ============ 2. 总体设计 ============ -->
<section class="chapter" id="sec-design">
  <div class="chapter-no">02 / 总体设计</div>
  <h2>三方案公平对比</h2>

  <p class="lead">为了回答"物理先验有没有用"，我们设计了一个<strong>控制变量</strong>框架：三个方案共享同一份数据、同一损失、同一训练配置、同一评估指标，<span class="highlight">唯一的区别是网络是否接收、以及接收什么形式的物理先验</span>。这样任何性能差异都可以归因到"先验"这一变量上。</p>

  <div class="card-grid">
    <div class="card">
      <h4>方案 A · 无先验 baseline</h4>
      <p>输入只有低分辨率图像 <code>L_up</code>（上采样到 256×256），输出高分辨率预测 <code>Ĥ</code>。代表"纯数据驱动"的天花板，也是验证其他方案是否有用的对照组。</p>
    </div>
    <div class="card">
      <h4>方案 B · 图像先验 + 残差</h4>
      <p>输入增加一张低阶参考图像 <code>P₂</code>（光滑后的低分辨率图，保留了束流的低阶结构但抹掉高频细节）。网络学一个残差 <code>R</code>，最终预测 <code>Ĥ = Softplus(P₂ + R)</code>。这种"early fusion + 残差"设计保证先验信息不被网络破坏——预测总强度锚定在 P₂ 附近。</p>
    </div>
    <div class="card">
      <h4>方案 C · 参数先验 + FiLM</h4>
      <p>输入增加 8 个物理参数 <code>c_prior</code>（束团长度、压缩比、能量、啁啾等）。这些标量经 MLP 编码后，通过 FiLM（Feature-wise Linear Modulation）注入到网络的瓶颈层和解码器，对中间表示做"特征级缩放与平移"。这让网络在深层语义层面而非浅层像素层与物理先验耦合。</p>
    </div>
  </div>

  <h3>公平性硬约束</h3>
  <ul>
    <li><strong>参数差异 &lt; 10%</strong>：实测 A/B 24.26M、C 24.43M（+0.70%），符合公平阈值</li>
    <li><strong>同一主干</strong>：5 级残差 U-Net（通道 48/96/192/384/384），所有方案共享</li>
    <li><strong>同一损失</strong>：空域 L1 + 频域 L1（详见 §6 训练策略）</li>
    <li><strong>同一训练配置</strong>：AdamW、lr=3e-4、weight decay=1e-4、batch=16/卡、50,000 步</li>
    <li><strong>同一评估指标</strong>：预注册主指标 ε_high^mask + 次指标 ε_z 相对误差</li>
  </ul>

</section>

<!-- ============ 3. 数据生成 ============ -->
<section class="chapter" id="sec-data">
  <div class="chapter-no">03 / / 数据生成</div>
  <h2>用合成数据替代真实束流仿真</h2>

  <p class="lead">本项目的研究问题是"超分辨率神经网络能否恢复高频细节"，但研究本身<span class="highlight">不需要</span>调用昂贵的真实束流仿真器（elegant、Ocelot 等）。我们用一个<strong>轻量级物理生成函数</strong>（约 2000 行 Python）按物理规律合成大量严格配对的高/低分辨率束流图像——这叫"替代验证"范式（Substitute Verification）。</p>

  <h3>为什么不用真实仿真器？</h3>
  <p>真实粒子加速器仿真器单次运行需要数小时到数天，无法支持 50,000 步训练所需的百万级样本量。我们的目标是<span class="accent">验证方法可行性</span>，而非替代真实物理装置——只要合成函数抓住了纵向相空间的关键物理特征（压缩、CSR 微聚束、能散演化），方法论结论就可以外推到真实数据。</p>

  <h3>生成函数做什么？</h3>
  <p>给定一组物理参数 <code>c = (σ_z, σ_δ, n, η, b0, a1, α, β)</code>，生成函数返回一个 256×256 的高分辨率图像 <code>H</code>：</p>
  <ol>
    <li><strong>基础密度</strong>：在 (z, δ) 平面上放一个二维高斯分布作为束团基态</li>
    <li><strong>压缩折叠映射</strong>：根据压缩比 γ，把 δ 维度按 S 形曲线折叠到 z 维度（这是高阶谐波腔压缩的物理特征）</li>
    <li><strong>CSR 微聚束</strong>：在强压缩态下叠加波长为 1/8 网格的细脊结构（模拟相干同步辐射引起的微聚束）</li>
    <li><strong>光滑渲染</strong>：用 σ_smooth 的高斯核渲染，σ_smooth,H = 0.125×w_fine（精细结构宽度的 1/8）逐样本自适应</li>
    <li><strong>总强度归一化</strong>：输出总强度 = 1（即"一个完整束团"）</li>
  </ol>

  <h3>退化协议：如何制造低分辨率观测？</h3>
  <p>给定 <code>H</code>，制造对应低分辨率观测 <code>L</code>：</p>
  <ol>
    <li><strong>高斯模糊</strong>：σ_K = 11.0 像素（这个值是经过标定的，使退化后图像"看起来像真实 TDS+谱仪的输出"）</li>
    <li><strong>4 倍下采样</strong>：256 → 64（块求和，物理含义是探测器像素的积分）</li>
    <li><strong>高斯噪声</strong>：σ_n = 1.22e-4（探测器噪声水平）</li>
    <li><strong>非负截断</strong>：<code>L = max(0, L_clean + n)</code>（物理探测器不能输出负值）</li>
  </ol>

  <p>最终数据：train 20,000 + val 2,000 + test_id 1,000（γ 块外同分布留出）+ test_pb 1,000（γ 块内 1:1 配对）。每个样本约 1MB，总数据集 14 GB（HDF5 落盘，gzip 压缩）。</p>

</section>

<!-- ============ 4. 网络架构 ============ -->
<section class="chapter" id="sec-model">
  <div class="chapter-no">04 / / 网络架构</div>
  <h2>残差 U-Net + 三种先验注入方式</h2>

  <p class="lead">我们使用一个<strong>5 级残差 U-Net</strong>作为所有方案的共享主干，通道宽度 48 / 96 / 192 / 384 / 384（瓶颈封顶 384，约 24M 参数）。三个方案的差异只在"<em>如何把先验信息融进去</em>"。</p>

  <h3>主干结构</h3>
  <p>经典 U-Net = 编码器（下采样提取语义）+ 解码器（上采样恢复分辨率）+ 跳跃连接（融合多尺度特征）。我们额外加了：</p>
  <ul>
    <li><strong>残差块</strong>：每级 2 个残差块（Conv → BN → ReLU → Conv → BN + 跳连），加速收敛</li>
    <li><strong>工作尺度 S = N² = 65,536</strong>：在数值大空间（损失对 S·H，预测 S·Ĥ）做训练，最后还原到物理空间——这避免了 Softplus 在极小值上的数值失稳（早期训练曾因此坍缩到全零）</li>
  </ul>

  <h3>方案 A：无先验（baseline）</h3>
  <p>输入只有 <code>L_up</code>（64→256 上采样），网络直接预测 <code>Ĥ</code>。最朴素，所有方案以此为对照。</p>

  <h3>方案 B：图像先验 + Early Fusion + 残差</h3>
  <p>关键设计：</p>
  <ul>
    <li><strong>P₂</strong> = 把 <code>L_up</code> 用更大的高斯核（σ_smooth,P = 15×σ_smooth,H ≈ 1.875 像素）再光滑一次得到的低阶参考图。它<span class="accent">保留束团整体形状但抹掉所有高频细节</span>——这正是我们希望先验提供的信息</li>
    <li><strong>Early Fusion</strong>：把 <code>L_up</code> 和 <code>P₂</code> 在通道维度拼接（2 通道输入）</li>
    <li><strong>残差结构</strong>：网络学 <code>R</code>，最终输出 <code>Ĥ = Softplus(S · L_up + R)</code>——预测总强度锚定在 P₂ 附近（输出 ≈ P₂ + 高频残差），网络只负责"补充细节"</li>
  </ul>

  <h3>方案 C：参数先验 + FiLM</h3>
  <p>参数先验是<strong>标量流形</strong>信息（不是图像），需要不同的融合方式：</p>
  <ul>
    <li><strong>参数预处理</strong>：8 个物理参数（<code>σ_z, n, η, b0, a1, α, β, σ_δ</code>）→ 正参数取对数 → 训练集 z-score 标准化（统计量<strong>只来自训练集</strong>，防测试集泄漏）</li>
    <li><strong>MLP 编码</strong>：标准化参数 → 2 层 MLP（128 隐藏） → 64 维潜变量 <code>z_c</code></li>
    <li><strong>FiLM 注入</strong>：<code>z_c</code> 在瓶颈层与解码器的每个残差块中，生成"特征级缩放 γ + 平移 β"，对中间表示做 <code>γ · x + β</code>——这是让物理参数<span class="accent">在特征空间而非像素空间</span>起作用的关键设计</li>
    <li><strong>工作尺度输出</strong>：<code>Ĥ = Softplus(S · L_up + R)</code>（同方案 A，只用 L_up 作为基础强度锚）</li>
  </ul>

  <h3>三个方案可视化</h3>
  <p>下图展示同一束团样本在三个"视角"下的表现：真值 <code>H</code>（含精细结构）、低分辨率退化 <code>L_up</code>（高频被模糊）、网络预测 <code>Ĥ</code>（网络尝试恢复的结果）。读者可以肉眼判断哪个方案的恢复更接近真值——但要注意，肉眼判断可能误导（物理量精度才是关键），需要配合后续评估指标。</p>

  <figure class="figure-frame">
__FIG01__
    <figcaption><b>图 1 · 2D 相空间三联图</b>。左图为真值 <code>H</code>（256×256，z-δ 平面，log 色标显示精细结构：压缩折叠脊、CSR 微聚束波纹、峰值尖峰）；中图为低分辨率退化 <code>L_up</code>（64→256 双线性上采样，已不可分辨 Nyquist 频率以上的高频精细结构——这正是"尺度锚定"的根源，低分辨率图像本身就没有这些信息）；右图为方案 A 的预测 <code>Ĥ</code>（已恢复部分低频包络，但与真值仍有高频细节差异）。三联图直接展示超分任务的输入-输出-目标关系，是研究问题视觉化的核心载体——读者一眼就能看出"任务到底有多难"。<b>当前图为方案 A，方案 B/C 对比版在 M5/M6 阶段扩展生成。</b></figcaption>
  </figure>

  <figure class="figure-frame">
__FIG02__
    <figcaption><b>图 2 · 1D 电流/能谱剖面对比</b>。沿 z 轴边缘的电流剖面 <code>I(z)</code>（上）与沿 δ 轴的能谱 <code>S(δ)</code>（下），三方案（H / L_up / Ĥ）叠加显示。剖面图比 2D 图更能定量显示"网络恢复了多少峰值形状与位置"——若 Ĥ 的峰值高度与位置接近 H 真实值，说明网络学到了 <code>I_peak</code> 物理量；反之若 Ĥ 剖面平滑化（峰值被"平均掉"）则属典型的过平滑失败。<b>当前图为方案 A 样本，三方案对比版在 M5 消融后扩展生成。</b></figcaption>
  </figure>

</section>

<!-- ============ 5. 训练策略 ============ -->
<section class="chapter" id="sec-train">
  <div class="chapter-no">05 / / 训练策略</div>
  <h2>空域-频域混合损失 + 双空间契约</h2>

  <p class="lead">训练目标是<strong>恢复真值 <code>H</code></strong>，但单纯用像素 L1 损失会让网络优先拟合低频（频谱偏差），放弃我们关心的精细结构。我们的损失由两部分组成：</p>

<pre>𝒫_total = 𝒫_space + λ · 𝒫_spec
𝒫_space = ‖ Ĥ − H ‖₁           (空域 L1，逐像素)
𝒫_spec  = ‖ FFT(Ĥ) − FFT(H) ‖₁ (频域 L1，五倍频程分带)
λ = 1.0（预注册冻结，不可改）</pre>

  <h3>为什么需要频域损失？</h3>
  <p>像素 L1 损失在低频区（图像大部分能量集中的地方）梯度大、在高频区（小细节）梯度小——网络会被"诱导"先学好低频，再慢慢尝试高频（甚至放弃高频）。频域损失直接对高频带的能量施加约束，<span class="highlight">强迫网络也恢复压缩折叠脊、峰值尖峰这些精细结构</span>。</p>

  <h3>λ=1.0 的依据</h3>
  <p>两种损失通过 <code>÷N²</code> 归一化（FFT 域的标准做法），保证两者在数值上同尺度——这样 λ=1.0 就意味着"空域与频域等权"，不需要额外调参。如果 λ 可调，就引入了一个可能影响公平性的超参；冻结为 1.0 保证三方案损失函数完全一致。</p>

  <h3>AdamW 优化器</h3>
  <p>PyTorch 标准 AdamW（lr=3e-4、weight decay=1e-4、β=(0.9, 0.999)）。每个 run 50,000 步，batch 16/卡（双卡 DDP 数据并行，等效 batch 32）。</p>

  <h3>早停与验证哨兵</h3>
  <p>每 2,000 步验证一次，patience 10（10 次验证无改善则停），且最早不早于 50% 训练预算（25,000 步）。每个 run 还有<span class="accent">三个验证哨兵</span>：</p>
  <ul>
    <li><code>Q_Ĥ/Q_H ∈ [0.1, 10]</code>：预测与真值的总强度比，避免"全零预测"等平凡解</li>
    <li><code>Pearson ρ ≥ 0.1</code>：相关性下限，避免"反相关"等失败模式</li>
    <li><code>val L_space &lt; 1/N²</code>：验证集空域损失必须显著低于"预测全零"的平凡损失</li>
  </ul>

  <p>本次六个 run 三个哨兵全过。</p>

  <h3>双空间契约</h3>
  <p>前面提到的"工作尺度 S = N² = 65,536"是另一个关键设计：训练在数值大的"工作空间"（损失对 S·H）做，最后还原到"物理空间"（预测 ÷S）。这避免了 Softplus 在极小值（σ × Base × 像素 ~ 1e-5）上的数值死区——我们早期实验曾因此坍缩到全零预测，修复后才稳定训练。</p>

</section>

<!-- ============ 6. 评估判据 ============ -->
<section class="chapter" id="sec-eval">
  <div class="chapter-no">06 / / 评估判据</div>
  <h2>如何"公平地"判定谁更好？</h2>

  <p class="lead">评估不能只看平均值——两个方案的 e_high_mask 平均值差 5% 可能完全在噪声内，也可能是真实显著优势。我们用<strong>配对差 + bootstrap 95% CI + Wilcoxon 符号秩检验</strong>做"三分类"判定：</p>

  <ul>
    <li><strong>显著正</strong>：CI 全部在零线右侧 → 后者显著优于前者</li>
    <li><strong>显著负</strong>：CI 全部在零线左侧 → 后者显著差于前者</li>
    <li><strong>等效</strong>：CI 横跨零线 → 证据不足以区分（即"无差异"的合法判定，不是"无增益"）</li>
  </ul>

  <h3>主指标：ε_high^mask（掩膜内高频误差）</h3>
  <p>掩膜 <code>M_{c_high}</code> 选取真值 H 高频能量累计 90% 的最小像素集——也就是"<span class="accent">精细结构最集中的区域</span>"。在这个区域里计算 L1 误差：</p>

<pre>ε_high^mask = ‖ (Ĥ − H)_high ⊙ M_{c_high} ‖₁</pre>

  <p>为什么不用全图 L1 或 PSNR？因为全图 L1/PSNR 被低频主导，对我们关心的精细结构不敏感；掩膜把评估聚焦在"高频信息在哪里"。</p>

  <h3>次指标：ε_z 相对误差</h3>
  <p>束流发射度（衡量束流品质的核心物理量）。计算预测 Ĥ 与真值 H 的发射度相对误差：</p>

<pre>ε_z = |ε_z(Ĥ) − ε_z(H)| / ε_z(H)</pre>

  <p>次指标的作用是验证"主指标的提升是否真的带来了物理意义"——即使掩膜内像素误差降低，如果发射度反而变差，那这种"提升"就是物理幻觉。</p>

  <h3>一票否决（防止"看起来好但其实坏"）</h3>
  <p>对每个方案，检查预测是否触发方案级失败：</p>
  <ul>
    <li><strong>P(F) ≤ 20%</strong>（触发率上限）+ 两个物理量增益 CI 下界 ≤ 0 → 判为 <code>local_failure</code>，不否决但需披露</li>
    <li><strong>P(F) &gt; 20%</strong> 且物理量恶化统计显著 → 触发 <code>veto</code>，该方案整体不通过</li>
  </ul>

  <p>本次六个 run 中 B/C 都判为 <code>local_failure</code>（P_F=0），不触发否决。</p>

  <h3>G1(b) 零学习基线</h3>
  <p>在判定"谁更好"之前，先确认"<span class="accent">网络真的学到了</span>"：方案 A 相对"直接把 <code>L_up</code> 上采样作为预测"的零学习基线，必须有正增益（误差降低）且配对差 CI 下界 &gt; 0。本次 A 相对 L_up 增益 +0.027（CI 全正），G1(b) 充分通过。</p>

</section>

<!-- ============ 7. 实验进度 ============ -->
<section class="chapter" id="sec-progress">
  <div class="chapter-no">07 / / 实验进度</div>
  <h2>研究执行的里程碑时间线</h2>

  <p class="lead">整个研究分 6 个里程碑（M0–M6）。M0 是规范冻结（已完成），M1–M3 是基础设施搭建与 baseline 建立，M4 是主实验（三方案 × 2 种子训练 + 评估），M5 是消融与归因，M6 是最终报告与出口决策。</p>

  <div class="timeline">

    <div class="ev">
      <div class="when">2026-08-25</div>
      <div class="what">M0 · Spec 集 v1.0 冻结</div>
      <div class="who">22 批草案审查 + 用户逐项拍板，覆盖研究方法/数据契约/评估判据/Git 流程/无人值守协议等</div>
    </div>

    <div class="ev">
      <div class="when">2026-08-26</div>
      <div class="what">M1 · 物理生成函数就绪</div>
      <div class="who">20/30/40 spec 全部 AC 通过，单元 + 集成 + 验收测试 115 通过</div>
    </div>

    <div class="ev">
      <div class="when">2026-08-26</div>
      <div class="what">M2 · 数据集生成 + G0 通过</div>
      <div class="who">v1 数据集 20k/2k/1k/1k/500，G0 三判据全 pass（精细结构覆盖 74%、探针通过、SNR_hf 0.072）</div>
    </div>

    <div class="ev">
      <div class="when">2026-08-26</div>
      <div class="what">M3 · 方案 A baseline + G1(a) 通过</div>
      <div class="who">EXP-01a 三方案健康完成（坍缩修复后哨兵全过）；EXP-01b/c 标定完成（σ_K=11.0、σ_n=1.22e-4、σ_smooth,P=15×）</div>
    </div>

    <div class="ev">
      <div class="when">2026-08-27 → 2026-08-28</div>
      <div class="what">M4 · EXP-02 主实验（6 run 训练 + 12 次评估）</div>
      <div class="who">A/B/C × seed0/1 共 6 run 全部完成训练（best_val_loss：A 0.031 / B 0.026 / C 0.018）；评估 12 次 + 聚合 + 五图生成完成</div>
    </div>

    <div class="ev current">
      <div class="when">2026-08-28（当前）</div>
      <div class="what">G2 主判定：跨种子三分类不一致 → 失败路径触发</div>
      <div class="who">扩种子至共 4 个（+seed2/seed3，~24h 墙钟），按 4 种子聚合规则 R1–R4 重判</div>
    </div>

    <div class="ev">
      <div class="when">M5（待启动）</div>
      <div class="what">EXP-03/04/07/08 消融与归因</div>
      <div class="who">强退化档 + 噪声敏感度 + 图像先验 P1 变体 + K=8 噪声重实现</div>
    </div>

    <div class="ev">
      <div class="when">M6（待启动）</div>
      <div class="what">最终报告 + 出口决策</div>
      <div class="who">六章报告 + 五图扩展 + 预注册对账 + 出口决策矩阵</div>
    </div>

  </div>

</section>

<!-- ============ 8. 当前结果 ============ -->
<section class="chapter" id="sec-results">
  <div class="chapter-no">08 / / 当前结果</div>
  <h2>训练完成 · 评估数字 · 五图解读</h2>

  <p class="lead">六个 run（A/B/C × seed0/1）已完成训练，所有哨兵通过。下面是评估结果：</p>

  <h3>训练最优（val_loss，50,000 步内最佳）</h3>
  <table>
    <tr><th>方案</th><th>seed0 val_loss</th><th>seed1 val_loss</th></tr>
    <tr><td>A</td><td>0.0308</td><td>0.0306</td></tr>
    <tr><td>B</td><td>0.0260</td><td>0.0264</td></tr>
    <tr><td><b>C</b></td><td><b>0.0180</b></td><td><b>0.0181</b></td></tr>
  </table>
  <p><span class="highlight">C 方案的 val_loss 比 A 低 41%、比 B 低 31%</span>。训练目标上 C 显著优于 A/B。</p>

  <h3>主指标 e_high_mask（test_id 上）</h3>
  <table>
    <tr><th>方案</th><th>seed0</th><th>seed1</th></tr>
    <tr><td>A</td><td>0.00126 ± 0.00123</td><td>0.00148 ± 0.00118</td></tr>
    <tr><td>B</td><td>0.00125 ± 0.00120</td><td>0.00118 ± 0.00106</td></tr>
    <tr><td><b>C</b></td><td><b>0.00106 ± 0.00061</b></td><td><b>0.00160 ± 0.00059</b></td></tr>
  </table>
  <p>注意 C 的标准差跨种子几乎恒定（0.0006），但<span class="warn">均值漂移 51%（0.00106 → 0.00160）</span>——误差分布整体平移。这是关键现象，将在 §9 详述。</p>

  <h3>次指标 e_eps_z（test_id 上）</h3>
  <table>
    <tr><th>方案</th><th>seed0 median</th><th>seed1 median</th></tr>
    <tr><td>A</td><td>0.0115</td><td>0.0131</td></tr>
    <tr><td>B</td><td>0.0126</td><td>0.0126</td></tr>
    <tr><td><b>C</b></td><td><b>0.0081</b></td><td><b>0.0079</b></td></tr>
  </table>
  <p><span class="accent">C 在发射度（束流品质核心物理量）误差上稳定占优——两种子都是 A/B 的 ~60%</span>。</p>

  <h3>聚合可视化</h3>

  <figure class="figure-frame">
__FIG03__
    <figcaption><b>图 3 · 5 个物理量相对误差柱状图</b>。横轴 5 个物理量（ε_z 发射度 / I_peak 峰值流强 / σ_z 纵向长度 / σ_δ 能散 / h_eff 有效发射度），纵轴相对误差均值 ± bootstrap 95% CI。该图直接显示<span class="accent">C 方案在物理量精度上对 A/B 的稳定优势</span>——特别是 ε_z（C 0.011 vs A 0.021 / B 0.022，约减半），与 §8.3 表格数字一致。<b>当前图为方案 A_seed0 单方案</b>（A 方案的指标基线），跨种子三方案对比版在 seed2/3 评估后生成（M6 final_report 必含三方案）。这张图是次指标"先验有用"故事的核心可视化证据——尽管主指标 e_high_mask 跨种子翻转（§9），下游物理量误差保持稳定优势。</figcaption>
  </figure>

  <figure class="figure-frame">
__FIG04__
    <figcaption><b>图 4 · 主指标 e_high_mask vs 压缩态 γ 散点图</b>。横轴压缩态 γ（γ∈[0,1]，γ 块内外全部样本），纵轴 e_high_mask（log 尺度）。揭示<span class="accent">主指标与压缩态的非线性耦合</span>——γ 越大（强压缩态）误差越大，对应"高频精细结构在强压缩区间对网络最具挑战性"的物理直觉（强压缩态 c_high 贡献最大，恰恰是高频精细结构所在）。这张图也是<span class="warn">G2 跨种子不一致的物理背景</span>——若 C 在强压缩区稳定优于 A/B，则即使主指标整体均值跨种子翻转，"先验在强压缩区有用"的故事仍可成立。<b>当前图为方案 A_seed0</b>，M6 需三方案对比版以验证"C 在强压缩区是否稳定占优"。</figcaption>
  </figure>

  <figure class="figure-frame">
__FIG05__
    <figcaption><b>图 5 · 预测-真值残差热力图</b>。发散色标（红=正过冲 Ĥ &gt; H、蓝=负欠冲 Ĥ &lt; H、白=匹配）。理想残差图应呈随机噪声（无可辨识结构）；若呈现清晰的高频结构（如压缩折叠脊的镜像），说明网络未能恢复该结构（"幻觉"信号反向）；若呈现低频模糊块，说明网络过平滑。<b>当前图为方案 A_seed0 单样本</b>，三方案残差图对比是"哪一方案的高频恢复更准确"的直观证据——M5 阶段扩展为三方案对比以直接验证 C 在强压缩区的恢复质量。</figcaption>
  </figure>

</section>

<!-- ============ 9. 当前问题 ============ -->
<section class="chapter" id="sec-problem">
  <div class="chapter-no">09 / / 当前问题</div>
  <h2>为什么不能立刻给出"先验有用/没用"的结论？</h2>

  <p class="lead">训练阶段 C 方案的 val_loss 比 A/B 低 41%（最优秀），物理量 e_eps_z 上 C 稳定减半（最优秀），但主指标 e_high_mask 的 G2 三分类<span class="warn">跨种子完全不一致</span>。这意味着我们必须谨慎——主指标的"故事"还没稳定下来。</p>

  <h3>G2 三分类结果（test_id 主判定）</h3>
  <table>
    <tr><th>对比</th><th>seed0 判定</th><th>seed1 判定</th><th>合并 CI</th><th>跨种子一致？</th></tr>
    <tr><td>A vs B</td><td>等效（A≈B）</td><td>B 显著优于 A</td><td>B 优 [+1.2e-4, +1.8e-4]</td><td><span class="warn">❌ 方向不翻转但 seed0 未达显著</span></td></tr>
    <tr><td>A vs C</td><td><span class="accent">C 显著优于 A</span></td><td><span class="warn">A 显著优于 C</span></td><td>C 微弱优 [+1e-6, +8e-5]</td><td><span class="warn">❌ 符号翻转</span></td></tr>
    <tr><td>B vs C</td><td><span class="accent">C 显著优于 B</span></td><td><span class="warn">B 显著优于 C</span></td><td>B 优 [-1.5e-4, -7e-5]</td><td><span class="warn">❌ 符号翻转</span></td></tr>
  </table>
  <p>最戏剧化的是 <b>A vs C</b> 和 <b>B vs C</b> —— 两个种子的判定结果<span class="warn">完全相反</span>。</p>

  <h3>通俗解释</h3>
  <p>想象一下：我们请了两位厨师（seed0 / seed1），用同一份菜谱做同一道菜。第一位厨师尝完后说"C 方案最好吃"，第二位说"C 方案最难吃"。两位厨师的水平应该差不多（这就是"<span class="highlight">可复现性</span>"的假设），但他们对 C 的评价<span class="warn">完全相反</span>。是 C 方案本身有问题？还是两位厨师各自"翻车"？</p>

  <h3>归因诊断</h3>
  <p>通过 Qwen 3.8 Max 二级咨询 + 方案 C 实现代码逐行核查，我们排除了以下可能：</p>
  <ul>
    <li><b>❌ 不是测量伪影</b>：逐种子 CI 半宽 ≤ 5% 基线（足够窄），反转幅度是半宽的 3-8 倍（不可能是测量噪声）</li>
    <li><b>❌ 不是数据划分耦合</b>：两 seed 见到完全相同的 test_id 1000 个样本</li>
    <li><b>❌ 不是评估随机性</b>：评估确定性、复现一致</li>
    <li><b>❌ 不是方案 C 实现 bug</b>：C 方案代码逐行核查（FiLM 注入路径正确、参数标准化统计量只来自训练集、种子派生确定可复现）—— 没有 A 类缺陷</li>
    <li><b>✓ 是 FiLM 调制对初始化敏感的真实施子效应</b>：C 的标准差跨种子几乎恒定（0.0006），但均值漂移 51%（0.00106 → 0.00160）—— 误差分布整体平移而非个别离群点。不同 `torch.manual_seed` 让 FiLM 收敛到不同盆地</li>
  </ul>

  <h3>为什么这是个真问题？</h3>
  <p>信息论角度：低分辨率观测 L 不包含 Nyquist 频率以上的高频信息——网络对高频是"天然盲"的，只能靠<strong>先验</strong>填补。如果方案 C 在两个训练种子下学到完全不同的"填补策略"，那"C 方案是否稳定地提供好先验"这件事就需要更多证据——而不是只凭两个种子就下结论。</p>

  <h3>合并 CI 的致命弱点 ★</h3>
  <p>第一直觉是"把两个种子的样本拼起来算合并 CI"——这样样本量翻倍、CI 更窄、结论更"稳定"。但 Qwen 通过模拟证明：<span class="warn">当存在种子随机效应时，合并 bootstrap CI 把两 seed 样本当 i.i.d. 处理，覆盖率（coverage probability）跌至 ~5%</span>。意思是"合并 CI 说 A 优于 C"这个结论有 95% 的概率其实是假的。<strong>合并 CI 在我们这种情况下不可作判定依据</strong>，仅作描述量。判定必须以逐种子 verdict 为准。</p>

  <h3>掩膜成分分解披露（强制）</h3>
  <p>我们还发现：主指标掩膜区域里，<span class="warn">β 能量占比是 c_high 差分的 2.2 倍，先验泄漏指数 Π_leak = 2.4（>0.5 阈值）</span>——图像先验 P₂ 在掩膜区域结构性占优。这意味着主指标解释力需以此为限定：C 方案的"主指标优势"部分来自"它能更好地把 P₂ 的结构呈现出来"，部分来自"它能恢复 c_high 精细结构"，这两部分信号需要进一步分离。</p>

</section>

<!-- ============ 10. 下一步 ============ -->
<section class="chapter" id="sec-next">
  <div class="chapter-no">10 / / 下一步</div>
  <h2>扩种子至共 4 个 · 4 种子聚合规则 · 诚实落点</h2>

  <p class="lead">按"统计判定协议"，跨种子不一致 → 失败路径触发 → <strong>扩种子或扩测试集</strong>。我们选择扩种子（从 2 个增到 4 个），原因：</p>

  <ul>
    <li><b>扩测试集（增加样本数）</b>只能收紧 CI 宽度，无法改变训练种子翻转——<span class="warn">对这个问题问题问题无效</span></li>
    <li><b>扩种子（增加训练随机性覆盖）</b>直接处理训练随机性根因——<span class="accent">有效</span></li>
  </ul>

  <h3>4 种子聚合规则 R1–R4</h3>
  <p>原协议只定义了两种子聚合（一致/不一致）。扩到 4 种子后，我们用 R1–R4 规则（这是两种子规则的最小推广，<span class="accent">不改变预注册阈值</span>）：</p>
  <ul>
    <li><b>R1 一致采纳</b>：4 个种子判定全一致 → 采纳该结论</li>
    <li><b>R2 多数采纳</b>：≥3 种子一致，且无反向对 → 采纳多数</li>
    <li><b>R3 种子级区间</b>：以 4 个种子内配对差均值为种子级样本，t(3) 95% CI 校验</li>
    <li><b>R4 兜底</b>：不满足 R1/R2 → 判"种子敏感、不可判定"，作为合法终点</li>
  </ul>

  <h3>诚实预期落点</h3>
  <blockquote>
    无论结果如何，最终报告都如实披露。最可能的两种落点：
    <ol>
      <li><b>B &gt; A 稳健确认</b>（seed0 equiv + seed1 B 优，合并 CI 已示方向），A−C/B−C 报"种子敏感/等效"</li>
      <li><b>三种对比全部种子敏感</b>——报"先验在物理量层面稳定有用、在像素高频层面不稳定"，并把"C 训练稳定性研究"登记为后续工作</li>
    </ol>
    <cite>两种落点都是合法的研究终点，不掩盖、不凑阳性。</cite>
  </blockquote>

  <h3>当前进度</h3>
  <p>扩种子训练（seed2 + seed3 = 共 6 个新 run）已启动，预计 <strong>24 小时墙钟</strong>完成（A/B/C × seed2/3 = 6 run × 8h，双卡 3 批）。完成后立即执行评估 + 按 R1–R4 重判 + 更新报告。</p>

</section>

  </main>
</div>

<footer style="background: #EDE8DF; color: #666666; padding: 60px 80px; border-top: 1px solid #E8E4DC; text-align: center;">
  <div style="max-width: 1200px; margin: 0 auto;">
    <span style="font-family: Inter, sans-serif; font-size: 13px; letter-spacing: 0.05em;">M4 阶段报告 · 物理先验引导的纵向相空间超分辨率研究 · 2026-08-28 · 单文件可下载版</span>
  </div>
</footer>

<script>
// 滚动同步：侧边栏 active 状态
(() => {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  const links = Array.from(sidebar.querySelectorAll('.nav-item'));
  const targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const id = '#' + e.target.id;
        links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === id));
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px', threshold: 0 });
  targets.forEach(t => obs.observe(t));
})();
</script>
</body>
</html>
"""

HTML = HTML.replace("__NAV_HTML__", nav_html)
HTML = HTML.replace("__FIG01__", img(*FIGURES[0]))
HTML = HTML.replace("__FIG02__", img(*FIGURES[1]))
HTML = HTML.replace("__FIG03__", img(*FIGURES[2]))
HTML = HTML.replace("__FIG04__", img(*FIGURES[3]))
HTML = HTML.replace("__FIG05__", img(*FIGURES[4]))

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({len(HTML.encode('utf-8'))/1024:.0f} KB, {HTML.count('figure')//2} figures inline)")