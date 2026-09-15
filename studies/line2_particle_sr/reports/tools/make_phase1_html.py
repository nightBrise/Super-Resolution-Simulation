"""生成 line2 Phase 1 单文件 HTML 阶段报告（图件 base64 内联）。

运行：python studies/line2_particle_sr/reports/tools/make_phase1_html.py
读：reports/tools/phase1_numbers.json、reports/figs/*.png、results/ 下的落盘产物、data 的 manifest。
写：studies/line2_particle_sr/reports/line2_phase1_report.html（唯一产物）。
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
LINE2 = ROOT / "studies" / "line2_particle_sr"
RESULTS = LINE2 / "results"
FIGS = LINE2 / "reports" / "figs"
OUT = LINE2 / "reports" / "line2_phase1_report.html"

N = json.loads((HERE / "phase1_numbers.json").read_text(encoding="utf-8"))
D = N["derived"]
GATES = N["gates"]
E0 = N["e0"]
E0M = N["e0_meta"]
TOL = N["tolerance"]
ROUTING = N["routing"]
DB = N["dark_buckets"]
TR = N["training"]
ZI = N["zero_information"]
MANIFEST = json.loads((LINE2 / "data" / "line2_ds_v1" / "manifest.json").read_text(encoding="utf-8"))
with (RESULTS / "n_eff_table.csv").open(encoding="utf-8") as fh:
    ROWS = list(csv.DictReader(fh))

SEEDS = [100, 101, 102, 103, 104]
JOINT_RUNS = [f"e1_joint_s{i}" for i in range(5)]
INDEP_RUNS = [f"e1_indep_s{i}" for i in range(5)]
E2_RUNS = [f"e2_joint_n{t}_s{i}" for t in (3000, 100000) for i in range(5)]
ALL_RUNS = JOINT_RUNS + INDEP_RUNS + E2_RUNS
TIER_LABEL = {1000: "1e3", 3000: "3e3", 10000: "1e4", 30000: "3e4", 100000: "1e5"}

FIG_INVENTORY = [
    ("fig01_pipeline.png", "任务与数据管线示意：粒子表 → 六投影 → 低计数输入 → 2×10⁵ 目标"),
    ("fig02_lattice.png", "同束在 FODO 站 0 / 站 1 的 (x,x′) 与 (y,y′) 二维直方图对照"),
    ("fig03_e0_curve.png", "E0 裸误差–N 曲线（中位数 + IQR 带）、参照噪声地板线与各档模型误差落点"),
    ("fig04_amplification.png", "三点放大主曲线 N_eff/N_low vs N_low（可验证域带、无放大线、域顶线、空心 = 上溢）"),
    ("fig05_training_dynamics.png", "20 个 run 的 val N_eff 轨迹（线型+标记分四组）+ e1_joint_s0 的 train/val NLL"),
    ("fig06_neff_and_err.png", "20 个 run 的 N_eff（标记形状 + 空心/实心）与 test 误差中位数（阴影线分档）"),
    ("fig07_h3_pairing.png", "H3 的 5 对同种子 joint vs independent 哑铃图（连线长度为零）"),
    ("fig08a_h8_passrate.png", "H8 十三项观测量达标率（阴影线分达标档 + 柱旁数值与通过束数）"),
    ("fig08b_dark_buckets.png", "暗桶 B0–B3 的 M2 裸输入（斜纹）vs 模型（实心）对照"),
    ("fig09a_zero_info_errs.png", "核心对照（上）：20 个 run 的逐束误差箱线图 vs 零信息常数解逐束散点"),
    ("fig09b_zero_info_wins.png", "核心对照（下）：各 run 相对常数解的胜出束数与中位误差比"),
    ("fig10_h7_cross_station.png", "H7 跨站：站 0 与站 1 逐种子配对的 N_eff（圆/菱形 + 站 1 上溢空心）"),
]


def rows_of(**kw) -> list[dict]:
    out = []
    for r in ROWS:
        if all(str(r[k]) == str(v) for k, v in kw.items()):
            out.append(r)
    return sorted(out, key=lambda r: (int(r["seed"]), r["run_id"]))


def run_row(run_id: str, station: str = "0") -> dict:
    for r in ROWS:
        if r["run_id"] == run_id and r["station"] == station:
            return r
    raise KeyError(run_id)


def img(name: str, style: str = "") -> str:
    data = base64.b64encode((FIGS / name).read_bytes()).decode("ascii")
    default = ("display:block; max-width:100%; width:auto; height:auto; margin:32px auto;"
               " border-radius:16px; border:1px dashed #E8E4DC;"
               " box-shadow: 0 4px 24px rgba(0,0,0,0.04);")
    return f'<img src="data:image/png;base64,{data}" alt="{name}" style="{style or default}" />'


def fig(name: str, caption: str, num: str, title: str) -> str:
    return (f'  <figure class="figure-frame">\n    {img(name)}\n'
            f'    <figcaption><b>图 {num} · {title}</b>。{caption}</figcaption>\n  </figure>')


def pct(x: float, n: int = 1) -> str:
    return f"{x * 100:.{n}f}%"


# ------------------------------------------------------------------ 图注
CAP = {}
CAP["fig01"] = (
    "<i>画的是什么：</i>从粒子表到训练用的“输入/目标”这一条数据流的全貌。左边是一名束团的六维粒子表，"
    "中间把它投影成六张计数直方图，右边分岔为两条：低计数输入（无放回抽 N_low 颗，五档、每档 20 个子样本）"
    "与同束同站的目标（2×10<sup>5</sup> 颗全量直方图，也就是参照 R）。"
    "<i>怎么读：</i>顺着箭头从左到右；下方琥珀色框是这张图最需要记住的一条约定。"
    "<i>结论：</i>输入与目标的<strong>像素网格完全相同</strong>，差别只在“画了多少颗粒子”，"
    "所以本阶段的“放大”指的是等效粒子数 N_eff/N_low，而不是提高图本身的像素分辨率。"
    "<i>限定：</i>像素分辨率的扩展（4096² 等）属 Phase 3，不在本阶段范围内；目标图也不是“另一台更好的机器拍的”，"
    "而是同一批粒子的全量统计——它自身仍带抽样噪声（见判据一章的噪声地板）。"
    '<br /><span style="color:#666">来源：data/line2_ds_v1/manifest.json、docs/specs/line2/00_master_spec.md [S1]</span>'
)
CAP["fig02"] = (
    "<i>画的是什么：</i>同一束粒子（val 划分的 beam_0006）在 FODO 入口（站 0）与出口（站 1）的相空间形状，"
    "左列是水平面 (x, x′)、右列是垂直面 (y, y′)，颜色是计数（log 色标；面板内的白字框给出该轴的物理 RMS 尺度，"
    "全文符号统一写作 x′ / y′，不再混用 xp / yp）。"
    "<i>怎么读：</i>上下对比同一列：形状与物理尺度都被传输改变了；每个面各自归一到 [-1, 1]，"
    "所以“填得多满”反映的是分布相对自身量程的形态，绝对尺度看白框里的数字。"
    f"<i>结论：</i>站 1 的输入图不是站 0 那张图的平移或缩放——例如 σ_x′ 由站 0 的 "
    f"{D['fig02_stats']['sigma_xp_s0']:.3e} 变成站 1 的 {D['fig02_stats']['sigma_xp_s1']:.3e}。"
    "因此“在站 0 训练的模型直接拿去站 1 用”必须单独检验（见 H7）。"
    "<i>限定：</i>这是一束的示例，不是统计结论；两站的 bin 边界也各自独立标定，跨站不能套用同一套坐标网格。"
    '<br /><span style="color:#666">来源：data/line2_ds_v1/val.h5（beam_0006/station0,1/target_full 及其 bin_edges 属性）</span>'
)
CAP["fig03"] = (
    "<i>画的是什么：</i>“尺子”——不经过任何网络，纯裸模拟的误差随粒子数 N 的变化（绿色折线为中位数，浅绿带为 IQR），"
    "外加一条水平参照（琥珀色虚线：参照噪声地板）与四个档位的模型误差落点（形状标记 + 深色描边，不依赖颜色）。"
    "<i>怎么读：</i>横轴 log 的粒子数、纵轴 log 的六投影平均相对 RMSE（M1）。"
    "先看曲线：粒子数每涨 10 倍，误差大约降到 1/3.2（1e3 档 4.030 → 1e4 档 1.246）。"
    "再看各形状的标记点：它们全部落在 1e5 档曲线的下方。"
    "<strong>两个 0.2856 的区别</strong>：参照噪声地板 0.285597 是 1e5 档子样本两两配对差 / √2，"
    "1e5 档裸误差 0.285636 是同档曲线节点的中位数——两者相差仅 "
    f"{D['floor_vs_tier_gap']:.1e}，是同一“可分辨性下界”的两种估计，不是两个互相矛盾的数。"
    f"<i>结论：</i>模型误差低于 1e5 档裸模拟的误差 → 按预注册定义一律触到报告域顶（上溢）；"
    f"同时曲线右端逼近地板 {E0['reference_noise_floor']:.6f}，意味着“再堆裸粒子也换不来可分辨的改善”。"
    "<i>限定：</i>地板是 1e5 档子样本两两互差的估计，它限制的是<strong>可分辨性</strong>——低于地板时，模型之间以及模型与裸模拟之间的差异在这个度量下已经无法区分；"
    "标记点落点用的是 test 束 × 站 0 的口径，与曲线的 val+test 口径不完全相同（1.2455 vs 1.2477 量级的差异即由此而来）。"
    '<br /><span style="color:#666">来源：results/e0_curve.json、e0_curve.meta.json、results/n_eff_table.csv</span>'
)
CAP["fig04"] = (
    "<i>画的是什么：</i>本阶段的交付主曲线——放大比 N_eff/N_low 对输入粒子数 N_low（log-log）。"
    "带色区是“可验证放大域”（≤10×，即 spec 声明的叙事上限），虚线是域顶线 10<sup>5</sup>/N_low，"
    "点线是无放大线（比值 =1）。"
    "<i>怎么读：</i>空心标记 = 该 run 的 N_eff 触到报告域顶（上溢），实心标记 = 域内真实插值值。"
    "三个 N_low 档位从左到右：3e3 档五个 run 全在 33.3×、1e4 档十个 run 全在 10×、1e5 档只有 s2 落在域内（98,724）。"
    "<i>结论：</i>“观测越稀疏、等效放大越大”这句话只能以<strong>下界</strong>形式成立——"
    "3e3 与 1e4 两档的中位放大比恰好等于 10<sup>5</sup>/N_low（33.3333 = 100000/3000、10.0000 = 100000/10000），"
    "说明它们是被域顶构造出来的，分辨力为零；33.3× 还超出 10× 的可验证边界。"
    "整条曲线只在最右端（1e5 档）有一个非饱和点，那才是可信的测量。"
    "<i>限定：</i>不要把三点连成的“下降趋势”当作物理规律——其中两点没有独立信息；"
    "域顶是报告域上限（1e5），不是目标粒子数（2e5）。"
    '<br /><span style="color:#666">来源：results/n_eff_table.csv、results/e0_curve.json、docs/specs/line2/00_master_spec.md [S1]</span>'
)
CAP["fig05"] = (
    "<i>画的是什么：</i>左格是全部 20 个 run 的验证选型准绳 val N_eff 随训练步数的轨迹；"
    "右格取其中一个 run（e1_joint_s0）的 train/val NLL 曲线，两条竖虚线标出“选型点”和“实际停机点”。"
    "<i>怎么读：</i>四组曲线除颜色外<strong>同时用线型与标记区分</strong>（joint@3e3 实线+圆、joint@1e4 虚线+方、"
    "indep@1e4 点划线+三角、joint@1e5 点线+菱形），黑白打印下同样可辨；"
    "左格里曲线一旦贴上 10<sup>5</sup> 就再也不下来——那是报告域顶；"
    "右格里 NLL 一直下降，说明训练本身没有变差，只是准绳不再更新 best。"
    f"<i>结论：</i>准绳在首个评估周期就会触顶，此后 patience 计数走到 10 就停机，"
    f"所以 20 个 run 都只用了 50,000 步预算的 22–26%（两臂均值 24.8%/22.4%），"
    f"被选中的 ckpt 并不是训练最充分的那一刻（e1_joint_s0 的 train NLL 从选型点的 "
    f"{TR['e1_joint_s0']['train_nll'][2]:.3f} 继续降到停机的 {TR['e1_joint_s0']['train_nll'][-1]:.3f}）。"
    "<i>限定：</i>这不代表“训练到 50,000 步一定更好”（没有做对照），只说明<strong>当前准绳在饱和后失去分辨力</strong>，"
    "选型点的语义因此空洞。"
    '<br /><span style="color:#666">来源：results/&lt;run&gt;/train_log.jsonl 与 summary.json（best_step/steps）</span>'
)
CAP["fig06"] = (
    "<i>画的是什么：</i>上格是 20 个 run 的 N_eff（按档位分组，log 纵轴），下格是同一批 run 的 test 形状误差中位数，"
    "并叠加两条水平参照（琥珀：1e5 档裸误差；紫：零信息常数解）。"
    "<i>怎么读：</i>上格里标记形状再编码档位（圆 = 3e3、方 = 1e4、三角 = 1e5），空心 = 上溢（贴在域顶线 "
    "10<sup>5</sup> 上）、实心 = 域内真实插值；下格柱形用<strong>阴影线</strong>区分档位（///、\\\\、xxx），"
    "不依赖颜色。下格里柱子越低越好，但两条水平线告诉你“低于哪条线”意味着什么。"
    f"<i>结论：</i>20 个 run 里 19 个 N_eff 都贴在域顶（= 达到可验证域顶），唯一例外是 1e5 档的 s2（98,724）；"
    f"误差中位数分布 0.2218–0.2898，横向跨越 30% 左右，比 N_eff 的差异更能分辨 run 之间的差别。"
    "<i>限定：</i>N_eff 在被截断的档位上不可排序（19 个点都写着 1e5），"
    "要比较 run 的强弱必须看下格的误差，且要用“相对基线”的读法（见诚实边界一章）。"
    '<br /><span style="color:#666">来源：results/n_eff_table.csv（N_eff/err 逐 run 明细）、results/e0_curve.json</span>'
)
CAP["fig07"] = (
    "<i>画的是什么：</i>H3 的 5 对同种子配对：每对里菱形是独立臂（6 个单投影子网）、圆是联合臂（1 个 6 进 6 出网），"
    "竖线连接同种子的两个值。"
    "<i>怎么读：</i>连线长度代表两臂在该种子上的差距；虚线是报告域顶。"
    "<i>结论：</i>五条连线长度都是零——两臂的 N_eff 全部触顶。按预注册规则（“联合 ≥ 独立”计入胜）判 joint_wins = 5/5，"
    "结论枚举为“成立”；但门禁同时披露 n_ties = 5、n_strict_joint_wins = 0、tied_at_domain_edge = true，"
    "说明这是<strong>字面成立、实测打平</strong>：1e4 档不具备区分两臂的能力。"
    "<i>限定：</i>“成立”不得被引用为“联合严格更优”；E2 只训练了获胜变体，"
    "因此 3e3 档也没有独立臂对照可用于事后复核这个选择。"
    '<br /><span style="color:#666">来源：results/e1_gates.json 的 H3 段、results/n_eff_table.csv</span>'
)
CAP["fig08a"] = (
    "<i>画的是什么：</i>H8 的 13 项物理观测量逐项达标率（横条），两条竖虚线是预注册的 50% 与 80% 阈值，"
    "每根柱右侧直接写出达标率与通过束数。"
    "<i>怎么读：</i>横条越长越好；阴影线给出达标档位（xx = 全通过、\\\\ = 部分通过、// = 全不通过），"
    "所以只看黑白也能分辨三档。"
    f"<i>结论：</i>13 项里只有 ε_4D 全过（150/150），σ_x′x′、|Δρ(x′,y)|、|Δρ(x′,y′)| 部分通过，其余 9 项一项未过；"
    f"整体达标率 {GATES['H8']['pass_rate']:.4f} ≤ 0.50 → <strong>H8 不成立</strong>。"
    "<i>限定：</i>容差线按每束 N_eff 映射到最近档，本批全部 150 束都落到最严的 1e5 档（见门禁一章），"
    "所以 H8 的“不成立”与“网络把 N_eff 推到域顶”是同一件事的两面；"
    "同一组观测量若改用 1e4 档容差，达标率为 0.9836（成立）。"
    '<br /><span style="color:#666">来源：results/e1_gates.json 的 H8.per_observable（达标率与逐项通过束数）</span>'
)
CAP["fig08b"] = (
    "<i>画的是什么：</i>M2 分桶（按参照计数分 B0–B3 四档）上“裸输入 vs 模型”的相对误差中位数对照。"
    "<i>怎么读：</i>每对柱子里斜纹是裸输入、实心是模型（13 项观测量之外的另一条独立信息）；"
    "柱顶给出绝对数值，柱下给出相对改进百分比。"
    f"<i>结论：</i>模型在分桶误差上确有实效：B1（暗而非常空）相对改进 {pct(DB['B1']['relative_improvement'])}、"
    f"B2 {pct(DB['B2']['relative_improvement'])}、B3 {pct(DB['B3']['relative_improvement'])}；"
    f"但 B0（参照计数 &lt; 0.5）的裸输入结构性恒零，相对改进无定义——而模型在该桶仍有 "
    f"{DB['B0']['model_median']:.6f} 的 M2，即<strong>在参照为空的格子里产生了计数</strong>。"
    "<i>限定：</i>暗桶分项由 40 [S3] 强制单独报告、不并入达标率；B2/B3 的相对改进与 B0/B1 同口径，"
    "均为“逐束先对 5 种子取中位、再对 150 束取中位”。"
    '<br /><span style="color:#666">来源：results/e1_gates.json 的 H8.dark_bucket_m2（B2/B3 由 results/&lt;run&gt;/test_eval.json 的 m2_buckets 按同口径聚合）</span>'
)
CAP["fig09a"] = (
    "<i>画的是什么：</i>核心对照实验的第一半：20 个 run 的逐束误差分布（水平箱线图），"
    "最下面一行紫色圆点是“零信息常数解”的逐束误差；四个档位分组有背景色带与行末组名，y 轴只标种子号。"
    "<i>怎么读：</i>常数解 = 700 个 train 束目标图逐投影归一后求平均得到的“均值束”，"
    "它不含任何 test 信息、也不随输入变化（纯零信息的退化解）；箱体与紫点越靠左越好，"
    "琥珀虚线是 1e5 档裸误差 0.285636（上溢判据线）。"
    f"<i>结论：</i>常数解的误差中位数只有 {ZI['const_median']:.6f}（逐束标准差 {ZI['const_std']:.4f}、"
    f"范围 {ZI['const_min']:.4f}–{ZI['const_max']:.4f}），落在全部 20 个 run 的箱体左侧；"
    f"同时 {ZI['const_median']:.6f} &lt; 0.285636，所以这个常数解同样会触发上溢。"
    "<i>限定：</i>这说的是“在冻结的 M1 聚合口径下，一个与束无关的常数解拿到更低的分”，"
    "否掉的是“模型恢复了束特异结构”这一读法；并不意味着网络输出等于常数"
    "（模型逐束误差标准差只有 0.0021–0.0063，常数解为 0.0870，相差 14–41 倍）。"
    '<br /><span style="color:#666">来源：本目录 tools/compute_phase1_numbers.py 复算（口径见诚实边界一章）、results/&lt;run&gt;/test_eval.json</span>'
)
CAP["fig09b"] = (
    "<i>画的是什么：</i>核心对照实验的第二半：每个 run 在 150 个 test 束上战胜常数解的束数，"
    "并在柱旁标注逐束误差比的中位数。"
    "<i>怎么读：</i>点线是 75/150（50% 平局线）——柱长全部在它左侧，说明多数束上模型并不比常数解更好；"
    "数值 &gt;1 表示模型误差更大。"
    f"<i>结论：</i>胜出束数 {min(v['n_wins_vs_const'] for v in ZI['per_run'].values())}–"
    f"{max(v['n_wins_vs_const'] for v in ZI['per_run'].values())}/150，逐束误差比中位数 "
    f"{min(v['median_ratio_model_over_const'] for v in ZI['per_run'].values()):.2f}–"
    f"{max(v['median_ratio_model_over_const'] for v in ZI['per_run'].values()):.2f}；"
    "最好的两个 run（joint@1e4 的 s3、indep@1e4 的 s3）也没到半数。"
    "<i>限定：</i>该对照不在 20 [S5] 冻结基线族内，属 spec 层缺口（已立条目）；"
    "它衡量的是“逐束胜负”，不替代主指标 N_eff 的聚合口径。"
    '<br /><span style="color:#666">来源：tools/phase1_numbers.json（复算）与 results/&lt;run&gt;/test_eval.json</span>'
)
CAP["fig10"] = (
    "<i>画的是什么：</i>H7 跨站检验：获胜变体（joint）的 5 个 ckpt <strong>不重训练</strong>，"
    "直接应用到站 1 的 test 束，并与站 0 的成绩逐种子配对（圆 = 站 0、空心菱形 = 站 1）。"
    "<i>怎么读：</i>ρ = N_eff(站1)/N_eff(站0) 是预注册判据（≥1 成立、0.5–1 弱成立、&lt;0.5 不成立）；"
    "虚线是报告域顶；每个种子旁边写着它在站 1 的误差中位数。"
    f"<i>结论：</i>ρ = {GATES['H7']['rho']:.1f}、种子中位数之比 rho_median = {GATES['H7']['rho_median']:.1f}，"
    "但站 0 与站 1 的 10 个 N_eff 全部携带 overflow 标记，按预注册规则“上/下溢使 ρ 无法计算 → 不可判定”，"
    f"故 H7 结论为<strong>不可判定</strong>（reason_code = {GATES['H7']['reason_code']}）。"
    "原始证据（站 1 误差中位数 0.2087–0.2739，与站 0 同量级甚至更低）指向没有明显跨站衰减。"
    "<i>限定：</i>域顶截断让“两侧相等”这个观测同时兼容“真的没有衰减”与“两个真值都被截到同一根线上”，"
    "因此不可辨识；H7 的不可判定是证据结构决定的，不是判定规则过严。"
    '<br /><span style="color:#666">来源：results/e1_gates.json 的 H7 段、results/n_eff_table.csv（station=0/1 两行组）</span>'
)

# ------------------------------------------------------------------ 组装
def build() -> str:
    n_runs = len(ALL_RUNS)
    e0_rows = "\n".join(
        f"      <tr><td>{p['n_low']:,}</td><td>{p['median']:.6f}</td><td>{p['iqr_lo']:.6f}</td>"
        f"<td>{p['iqr_hi']:.6f}</td><td>{p['n_samples']:,}</td>"
        f"<td>{E0['mutual_floor'][str(p['n_low'])]:.6f}</td></tr>"
        for p in E0["points"]
    )
    neff_rows = "\n".join(
        f"      <tr><td><code>{r['run_id']}</code></td><td>{r['variant']}</td><td>{r['seed']}</td>"
        f"<td>{int(r['n_low']):,}</td><td>{r['station']}</td><td>{float(r['err_median']):.6f}</td>"
        f"<td>{float(r['n_eff']):,.1f}</td><td>{r['n_eff_flag'] or '—'}</td></tr>"
        for r in sorted(ROWS, key=lambda r: (int(r["station"]), int(r["n_low"]), r["run_id"]))
    )
    arm_rows = []
    for label, runs in (("joint（联合）", JOINT_RUNS), ("independent（独立）", INDEP_RUNS)):
        best = ", ".join(str(TR[x]["best_step"]) for x in runs)
        steps = ", ".join("{:,}".format(TR[x]["steps_total"]) for x in runs)
        used = ", ".join(f"{TR[x]['steps_total'] / 50000 * 100:.1f}%" for x in runs)
        arm_rows.append(
            f"      <tr><td>{label}</td><td>{best}</td><td>{steps}</td><td>{used}</td></tr>"
        )
    tr_rows = "\n".join(arm_rows)
    ob_rows = "\n".join(
        f"      <tr><td>{o['index'] + 1}</td><td><code>{o['name']}</code></td><td>{o['n_pass']}/150</td>"
        f"<td>{o['pass_rate']:.4f}</td></tr>"
        for o in GATES["H8"]["per_observable"]
    )
    db_rows = "\n".join(
        f"      <tr><td>{b}（{'参照 &lt; 0.5' if b == 'B0' else '0.5 ≤ 参照 &lt; 2' if b == 'B1' else '2 ≤ 参照 &lt; 10' if b == 'B2' else '参照 ≥ 10'}）</td>"
        f"<td>{DB[b]['bare_median']:.6f}</td><td>{DB[b]['model_median']:.6f}</td>"
        f"<td>{'无定义（裸输入恒零）' if DB[b]['relative_improvement'] is None else pct(DB[b]['relative_improvement'])}</td></tr>"
        for b in ("B0", "B1", "B2", "B3")
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>line2 Phase 1 阶段报告 · 粒子级相空间超分辨率</title>
<style>
  /* 温润学术风（warm-academic），与 docs/reports/M4_stage_report.html 同一套样式，逐字复用 */
  :root {{
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
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--bg-primary);
    color: var(--ink-body);
    font-family: "Noto Serif SC", Georgia, "Times New Roman", serif;
    font-size: 18px;
    line-height: 1.9;
    font-weight: 400;
  }}
  body::before {{
    content: "";
    position: fixed; inset: 0;
    pointer-events: none; z-index: 0;
    opacity: 0.03;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
  }}
  /* 导航栏 */
  nav.top {{
    position: sticky; top: 0; z-index: 50;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--ink-faint);
    padding: 0 80px;
  }}
  nav.top .wrap {{
    max-width: 1200px; margin: 0 auto;
    display: flex; justify-content: space-between; align-items: center;
    height: 72px;
  }}
  nav.top .logo {{
    font-family: Inter, sans-serif;
    font-size: 13px; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--ink-strong); text-decoration: none;
  }}
  nav.top .meta {{
    font-family: Inter, sans-serif;
    font-size: 12px; color: var(--ink-soft);
    letter-spacing: 0.05em;
  }}
  /* 侧边栏 */
  .layout {{ display: flex; max-width: 1200px; margin: 0 auto; padding: 0 24px; gap: 48px; }}
  aside.sidebar {{
    width: 240px; flex-shrink: 0;
    position: sticky; top: 96px;
    align-self: flex-start;
    height: calc(100vh - 120px); overflow-y: auto;
    padding: 24px 0;
  }}
  aside.sidebar .nav-item {{
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px;
    font-family: Inter, sans-serif;
    font-size: 14px; color: var(--ink-sub);
    text-decoration: none;
    border-radius: var(--r-sm);
    transition: all 0.3s;
    border-left: 2px solid transparent;
  }}
  aside.sidebar .nav-item:hover {{ background: var(--bg-card); color: var(--ink-strong); }}
  aside.sidebar .nav-item.active {{
    color: var(--ink-strong); font-weight: 600;
    border-left-color: var(--accent-pine);
    background: var(--bg-card);
  }}
  aside.sidebar .nav-num {{
    font-family: "JetBrains Mono", monospace;
    font-size: 11px; color: var(--ink-soft);
    min-width: 20px;
  }}
  /* 主内容 */
  main {{
    flex: 1; min-width: 0;
    max-width: 880px;
    padding: 48px 0 120px;
  }}
  /* Hero */
  section.hero {{
    padding: 80px 0 64px;
    border-bottom: 1px solid var(--ink-faint);
    margin-bottom: 80px;
  }}
  section.hero .eyebrow {{
    font-family: Inter, sans-serif;
    font-size: 13px; color: var(--accent-pine);
    letter-spacing: 0.18em; text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 24px;
  }}
  section.hero h1 {{
    font-family: Inter, sans-serif;
    font-size: 48px; font-weight: 600;
    line-height: 1.15;
    color: var(--ink-strong);
    margin: 0 0 24px;
  }}
  section.hero .subtitle {{
    font-family: "Noto Serif SC", Georgia, serif;
    font-size: 19px; color: var(--ink-sub);
    line-height: 1.7;
    max-width: 720px;
  }}
  section.hero .stamp {{
    margin-top: 32px;
    display: inline-block;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-radius: var(--r-sm);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px; color: var(--ink-sub);
    letter-spacing: 0.05em;
  }}
  /* 章节 */
  section.chapter {{
    padding: 64px 0 40px;
    border-top: 1px solid var(--ink-faint);
    margin-top: 64px;
    scroll-margin-top: 96px;
  }}
  section.chapter:first-of-type {{ border-top: none; margin-top: 0; }}
  section.chapter .chapter-no {{
    font-family: "JetBrains Mono", monospace;
    font-size: 14px; color: var(--accent-pine);
    letter-spacing: 0.2em;
    margin-bottom: 12px;
  }}
  section.chapter h2 {{
    font-family: Inter, sans-serif;
    font-size: 36px; font-weight: 600;
    color: var(--ink-strong);
    margin: 0 0 32px;
    line-height: 1.25;
  }}
  section.chapter h3 {{
    font-family: Inter, sans-serif;
    font-size: 22px; font-weight: 600;
    color: var(--ink-strong);
    margin: 40px 0 16px;
  }}
  section.chapter p {{
    margin: 0 0 1.4em;
    color: var(--ink-body);
  }}
  /* 首字下沉 */
  section.chapter p.lead::first-letter {{
    font-family: "Noto Serif SC", Georgia, serif;
    font-size: 56px; line-height: 1;
    float: left; padding-right: 8px; padding-top: 4px;
    color: var(--accent-saddle);
    font-weight: 600;
  }}
  section.chapter .figure-frame {{
    margin: 40px 0;
  }}
  section.chapter figcaption {{
    font-family: "Noto Serif SC", Georgia, serif;
    font-size: 15px; color: var(--ink-sub);
    line-height: 1.75;
    padding: 16px 24px;
    background: var(--bg-card);
    border-left: 3px solid var(--accent-pine);
    border-radius: 0 var(--r-md) var(--r-md) 0;
    margin-top: -16px;
  }}
  section.chapter figcaption b {{ color: var(--ink-strong); }}
  /* 卡片 */
  .card-grid {{
    display: flex; flex-wrap: wrap; gap: 20px;
    margin: 32px 0;
  }}
  .card {{
    flex: 1; min-width: 280px;
    background: var(--bg-card);
    border: 1px dashed var(--ink-faint);
    border-radius: var(--r-md);
    padding: 28px;
    transition: all 0.3s;
  }}
  .card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-hover); }}
  .card h4 {{
    font-family: Inter, sans-serif;
    font-size: 18px; font-weight: 600;
    color: var(--ink-strong);
    margin: 0 0 12px;
  }}
  .card p {{
    font-size: 15px; color: var(--ink-sub);
    line-height: 1.7; margin: 0;
  }}
  /* 引用块 */
  blockquote {{
    margin: 32px 0;
    padding: 24px 32px;
    background: var(--bg-secondary);
    border-left: 3px solid var(--accent-pine);
    border-radius: 0 var(--r-md) var(--r-md) 0;
    font-style: italic;
    color: var(--ink-sub);
  }}
  blockquote cite {{
    display: block;
    margin-top: 12px;
    font-size: 14px; font-style: normal;
    color: var(--ink-soft);
  }}
  /* 表格 */
  table {{
    width: 100%; border-collapse: collapse;
    margin: 24px 0;
    font-family: Inter, sans-serif;
    font-size: 14px;
  }}
  table th, table td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--ink-faint);
    text-align: left;
  }}
  table th {{
    background: var(--bg-secondary);
    color: var(--ink-strong);
    font-weight: 600;
  }}
  table tr:last-child td {{ border-bottom: none; }}
  /* 代码 */
  code {{
    font-family: "JetBrains Mono", monospace;
    font-size: 0.92em;
    background: var(--bg-secondary);
    padding: 1px 6px;
    border-radius: 4px;
    color: var(--ink-strong);
  }}
  pre {{
    background: var(--bg-secondary);
    border: 1px solid var(--ink-faint);
    border-radius: var(--r-md);
    padding: 20px;
    overflow-x: auto;
    font-family: "JetBrains Mono", monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--ink-body);
  }}
  /* 强调 */
  .highlight {{
    background: linear-gradient(transparent 60%, rgba(45,90,74,0.18) 60%);
    padding: 0 2px;
  }}
  .accent {{ color: var(--accent-pine); font-weight: 600; }}
  .warn {{ color: var(--accent-saddle); font-weight: 600; }}
  /* 时间线 */
  .timeline {{
    position: relative;
    margin: 32px 0;
    padding-left: 28px;
    border-left: 2px solid var(--ink-faint);
  }}
  .timeline .ev {{
    position: relative;
    padding: 12px 0 24px 16px;
  }}
  .timeline .ev::before {{
    content: "";
    position: absolute;
    left: -33px; top: 18px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--bg-primary);
    border: 2px solid var(--accent-pine);
  }}
  .timeline .ev.current::before {{ background: var(--accent-pine); }}
  .timeline .ev .when {{
    font-family: "JetBrains Mono", monospace;
    font-size: 12px; color: var(--ink-soft);
    letter-spacing: 0.05em;
  }}
  .timeline .ev .what {{
    font-family: Inter, sans-serif;
    font-size: 16px; font-weight: 600;
    color: var(--ink-strong); margin-top: 4px;
  }}
  .timeline .ev .who {{
    font-size: 14px; color: var(--ink-sub); margin-top: 4px;
  }}
  /* 响应式 */
  @media (max-width: 1024px) {{
    aside.sidebar {{ display: none; }}
    .layout {{ padding: 0 24px; }}
    nav.top {{ padding: 0 24px; }}
    section.hero h1 {{ font-size: 36px; }}
    section.chapter h2 {{ font-size: 28px; }}
  }}
  /* 本机离线字体回退（不引用任何外部资源；上面的样式保持逐字不变） */
  body, section.chapter p, section.chapter figcaption {{
    font-family: "Noto Serif SC", "Noto Serif CJK SC", "Noto Serif CJK JP",
                 Georgia, "Times New Roman", serif;
  }}
  nav.top .logo, nav.top .meta, aside.sidebar .nav-item, section.hero h1,
  section.hero .eyebrow, section.chapter h2, section.chapter h3, .card h4,
  table, .timeline .ev .what {{
    font-family: Inter, "Noto Sans CJK SC", "Noto Sans CJK JP", system-ui, sans-serif;
  }}
  code, pre, aside.sidebar .nav-num, section.chapter .chapter-no,
  section.hero .stamp, .timeline .ev .when {{
    font-family: "JetBrains Mono", "DejaVu Sans Mono", "Noto Sans Mono CJK JP", monospace;
  }}
</style>
</head>
<body>
<nav class="top">
  <div class="wrap">
    <a href="#sec-question" class="logo">LINE2 PHASE 1 · PARTICLE SR</a>
    <div class="meta">2026-09-15 · 阶段报告 · 单文件离线版</div>
  </div>
</nav>
<div class="layout">
  <aside class="sidebar" id="sidebar">
<a href="#sec-question" class="nav-item"><span class="nav-num">01</span>问题是什么</a>
<a href="#sec-method" class="nav-item"><span class="nav-num">02</span>方法与公平性</a>
<a href="#sec-data" class="nav-item"><span class="nav-num">03</span>数据与几何</a>
<a href="#sec-criteria" class="nav-item"><span class="nav-num">04</span>判据</a>
<a href="#sec-train" class="nav-item"><span class="nav-num">05</span>训练</a>
<a href="#sec-results" class="nav-item"><span class="nav-num">06</span>结果</a>
<a href="#sec-gates" class="nav-item"><span class="nav-num">07</span>门禁结论</a>
<a href="#sec-limits" class="nav-item"><span class="nav-num">08</span>诚实边界</a>
<a href="#sec-timeline" class="nav-item"><span class="nav-num">09</span>时间线</a>
<a href="#sec-repro" class="nav-item"><span class="nav-num">10</span>可复现性</a>
  </aside>
  <main>

<section class="hero" id="sec-hero">
  <div class="eyebrow">阶段报告 · line2 Phase 1 · 粒子级相空间超分辨率</div>
  <h1>用一万颗粒子的预算，读十万颗才有的精度</h1>
  <p class="subtitle">加速器束流诊断里，图的清晰度取决于“画了多少颗粒子”。Phase 1 问一个直接问题：把只画了 1 万颗（1×10<sup>4</sup>）粒子的六投影直方图交给神经网络，它能不能给出相当于 10 万颗（1×10<sup>5</sup>）甚至更多颗粒子的读数？答案是<strong>在形状误差的意义上能</strong>——但一个与束无关的“均值束”常数解能做到更好，这条否定性对照划定了本阶段结论的边界。</p>
  <div class="stamp">Phase 1 全链完成 · E0 尺子 → E1 两臂 → E2 三点主曲线 → E3 跨站 · {n_runs} 个训练 run · 10 张图</div>
</section>

<section class="chapter" id="sec-question">
  <div class="chapter-no">01 / 问题是什么</div>
  <h2>为什么“用更少的粒子读出同样的细节”这么难？</h2>

  <p class="lead">在直线加速器驱动的自由电子激光装置里，束团的横向相空间决定了它能不能被聚焦、能不能稳定出光。我们能看到它，靠的是把束团里的粒子投影到几个平面上拍成图——每颗粒子贡献一个点，粒子画得越多，图里的结构越清楚。问题是：拍得越多，代价越大；而调束现场常常只来得及采到很少的粒子。</p>

  <p>这就是 Phase 1 要回答的事：<strong>给定一张“画得很少”的直方图，网络能不能把它补成“画得很多”的读数。</strong>本阶段的输入是 N_low 颗粒子投影出的六张 128×128 计数图（五档：1×10<sup>3</sup>、3×10<sup>3</sup>、1×10<sup>4</sup>、3×10<sup>4</sup>、1×10<sup>5</sup>），目标（参照）是<strong>同一束、同一站、全部 2×10<sup>5</sup> 颗粒子</strong>的同样六张图。</p>

  <div class="card-grid">
    <div class="card">
      <h4>先把“放大”这个词说清楚</h4>
      <p>输入和输出的<strong>像素网格完全相同</strong>，都是 128×128。差别只在图里画了多少颗粒子。所以本阶段的“超分”指的是<span class="highlight">等效粒子数</span>——输出的误差相当于用多少颗裸模拟粒子才能达到，记作 N_eff；图本身的像素网格并不改变。像素分辨率的扩展属 Phase 3，不在本阶段。</p>
    </div>
    <div class="card">
      <h4>难点一：稀疏统计的噪声</h4>
      <p>粒子少，图就“毛”。稀疏不是分辨率不够，而是<strong>统计量不够</strong>：同一束抽两次子样本，两次的图就长得不一样。任何“补图”动作都要先跨过这层随机性，而随机性的量级要先用尺子量出来才谈得上比较。</p>
    </div>
    <div class="card">
      <h4>难点二：看得清 ≠ 是对的</h4>
      <p>网络可以画出一张“看起来像相空间”的漂亮图，却在参照为空的格子里填上并不存在的计数。这类<span class="warn">幻觉</span>不会体现在整体形状误差上，必须用逐项物理量和分桶误差单独盯住。</p>
    </div>
  </div>

  <h3>本阶段要回答的四个问题</h3>
  <ul>
    <li><strong>H1 放大存在吗：</strong>在 1×10<sup>4</sup> 输入下，网络输出的等效粒子数 N_eff 能不能超过输入本身的粒子数、并达到 2 倍以上？</li>
    <li><strong>H3 联合建模比独立建模好吗：</strong>用一个 6 进 6 出的网络（六个投影一起处理）与六个各管一个投影的小网络，同种子逐对比较，联合版是否不劣？</li>
    <li><strong>H7 换个站还灵吗：</strong>在 FODO 入口（站 0）训练的模型，直接拿到出口（站 1）用，成绩会不会掉？</li>
    <li><strong>H8 物理量精度够吗：</strong>13 项二阶矩类物理观测量，有多少项能达到与 N_eff 对应档位裸模拟同等的容差？</li>
  </ul>
</section>

<section class="chapter" id="sec-method">
  <div class="chapter-no">02 / 方法与公平性</div>
  <h2>两臂对照 + 一把先量好的尺子</h2>

  <p class="lead">要让“联合建模有没有用”这个问题有一个可回答的形式，变量必须只剩一个。我们训练<strong>两臂</strong>：联合臂用一个 6 进 6 出的 FiLM U-Net 同时处理六个投影，独立臂用六个 1 进 1 出的小网络分别处理；除了拓扑，其余设置逐字相同。</p>

  <div class="card-grid">
    <div class="card">
      <h4>联合（joint）</h4>
      <p>1 个 6 进 6 出网络（base 48 / depth 4 / blocks 2），外加一条软一致罚项 P_PSD；参数量 <code>10,708,710</code>。条件向量 κ 为 Σ 的 10 个唯一分量 + log₁₀N。</p>
    </div>
    <div class="card">
      <h4>独立（independent）</h4>
      <p>6 个 1 进 1 出子网（base 16 / depth 4 / blocks 3），权重互不共享；参数量 <code>10,092,870</code>，比联合低 <strong>5.8%</strong>。条件向量 κ 为 (var_i, var_j, cov_ij, log₁₀N)。</p>
    </div>
  </div>

  <h3>公平性硬约束</h3>
  <ul>
    <li><strong>同数据、同切分</strong>：两臂消费同一个数据集（train 700 / val 150 / test 150 束），束级互斥（00 GC1）。</li>
    <li><strong>同种子</strong>：两臂各 5 个种子 {100, 101, 102, 103, 104}，逐对按<em>种子值</em>配对（不是按序号），共 5 对。</li>
    <li><strong>同步数预算</strong>：max_steps = 50,000（预注册）、AdamW（lr 3×10<sup>-4</sup>、wd 1×10<sup>-4</sup>）、batch 16、每 1,000 步评估、patience = 10。</li>
    <li><strong>同参数量级</strong>：两臂差 5.8%，在预注册的 30% 内（早前配置曾把独立臂参数量压到 −76.9%，已修正，见诚实边界一章）。</li>
    <li><strong>同一把尺子</strong>：所有误差都在同一个度量（M1，总计数归一后的相对 RMSE）与同一条裸误差–N 曲线上换算成 N_eff。</li>
  </ul>

  <h3>尺子先于读数</h3>
  <p>N_eff 的定义依赖一条裸误差–粒子数的曲线（E0）。如果这条曲线能在看到模型结果之后再定，任何阈值都能被事后调出来。所以 E0 先跑完、时间戳落盘，编排链在启动训练前强制校验这个顺序：<span class="accent">E0 完成于 {E0M['completed_at'][:19]}Z，E1 首个 run 启动于 {D['e1_first_start'][:19]}Z</span>，相差约 {D['e0_pre_e1_minutes']:.0f} 分钟。</p>

  <h3>两段式：val 选型，test 只评一次</h3>
  <p>调参、变体路由、早停选型全部只消费 <strong>val</strong> 段；<strong>test</strong> 段在出口门禁处只评一次。变体路由（决定 E2 用哪一臂）由脚本按 val 段结果自动决定，规则是「{ROUTING['branch_rule']}」，据此得到 <code>winner_variant = {ROUTING['winner_variant']}</code>、依据 <code>basis = {ROUTING['basis']}</code>。但注意：val 段两臂的中位数同为 1×10<sup>5</sup>（域顶平局），所以这个路由<strong>建立在平局之上</strong>；E2 只训练获胜变体，因此 3×10<sup>3</sup> 档没有独立臂对照可做事后复核。</p>
</section>

<section class="chapter" id="sec-data">
  <div class="chapter-no">03 / 数据与几何</div>
  <h2>粒子从哪来，两站为什么会不一样</h2>

  <p class="lead">本研究不调用昂贵的束流仿真器。束源是一份真实的粒子表：elegant 生成的 <code>injector.sdds</code>，200,001 行 × 6 列（x, x′, y, y′, t, p），电荷 500 pC。它被冻结进仓库并核验过指纹（MD5 = <code>{MANIFEST['anchor_md5']}</code>），此后所有数据都由它确定性地派生出来——这叫“替代验证”范式，好处是每一个数字都能端到端复现。</p>

  <h3>从一束到一千束，从入口到出口</h3>
  <p>把同一份粒子表做 1,000 次带扰动的直线传输，就得到 1,000 束“相似但不相同”的束流：1 束无扰动基准束 + 999 束抖动束（四极强度 ±5%、薄透镜踢角 1×10<sup>-5</sup> rad、漂移段 495–505 mm）。扰动参数由 <code>SeedSequence({MANIFEST['master_seed']})</code> 派生，与束号一一对应，所以任取一束都能还原出它的扰动。每束在 FODO 晶格的两个位置各留一张快照：<strong>站 0</strong> 是 FODO 入口、<strong>站 1</strong> 是出口（n_cells = {MANIFEST['fodo']['n_cells']}、μ = {MANIFEST['fodo']['mu_deg']:.0f}°/cell、漂移 {MANIFEST['fodo']['half_drift_m']} m、四极长 {MANIFEST['fodo']['quad_length_m']} m）。</p>

{fig('fig01_pipeline.png', CAP['fig01'], '1', '任务与数据管线：粒子表 → 六投影 → 五档低计数输入 → 2×10⁵ 目标')}

{fig('fig02_lattice.png', CAP['fig02'], '2', '同一束在 FODO 两站的相空间：跨站不是同一张图')}

  <h3>六投影、五档、二十个子样本</h3>
  <p>每个站把 4D 横向相空间 (x, x′, y, y′) 投影成<strong>六张</strong>二维计数图，通道顺序恒为 <code>(x,x′)</code>、<code>(y,y′)</code>、<code>(x,y)</code>、<code>(x,y′)</code>、<code>(x′,y)</code>、<code>(x′,y′)</code>（00 [S1] 的固定枚举）。输入档位是 <code>N_low ∈ {{1×10³, 3×10³, 1×10⁴, 3×10⁴, 1×10⁵}}</code>，每档对每束做 <strong>{MANIFEST['subsamples_per_tier']} 次无放回抽样</strong>（严格 A ⊂ B：小子样本一定包含在大子样本里），目标恒为同束同站 2×10<sup>5</sup> 的全量图。</p>

  <table>
    <thead><tr><th>项</th><th>值</th><th>来源</th></tr></thead>
    <tbody>
      <tr><td>束数 / 划分</td><td>1,000 束 = train 700 / val 150 / test 150</td><td>manifest.json</td></tr>
      <tr><td>像素网格 / 通道数</td><td>128 × 128 / 6 个投影</td><td>manifest.json</td></tr>
      <tr><td>输入档位 × 子样本 / 档</td><td>5 档 × 20</td><td>manifest.json</td></tr>
      <tr><td>目标粒子数</td><td>2×10<sup>5</sup>（参照 R）</td><td>manifest.json</td></tr>
      <tr><td>母种子 / 抖动尺度</td><td>{MANIFEST['master_seed']} / ±5%、1×10<sup>-5</sup> rad、495–505 mm</td><td>manifest.json</td></tr>
      <tr><td>落盘体积（train/val/test）</td><td>{D['dataset_gb']['train']:.2f} GB / {D['dataset_gb']['val']:.2f} GB / {D['dataset_gb']['test']:.2f} GB</td><td>data/line2_ds_v1/*.h5</td></tr>
    </tbody>
  </table>
</section>

<section class="chapter" id="sec-criteria">
  <div class="chapter-no">04 / 判据</div>
  <h2>先把尺子量好，再谈谁更好</h2>

  <p class="lead">“网络变好了”这句话必须有个单位。这里用的单位是<strong>等效粒子数</strong>：先量出不带网络的裸模拟“粒子数 → 误差”曲线，再把网络的误差放到曲线上反查——误差这么小，相当于要多少颗裸粒子才能做出来。这样结论就直接读作“1×10⁴ 颗输入的网络输出 ≈ 10⁵ 颗裸模拟的精度”，与实验成本同量纲。</p>

  <h3>度量：M1（六投影平均相对 RMSE）</h3>
  <p>每张被评图先按参照的总计数<strong>归一</strong>（比较的是形状，不是绝对计数），再逐格算相对 RMSE，六个投影取平均。这条规则内置在实现里，任何调用方都无法绕开——所以网络输出的大小不会成为得分来源。</p>

  <h3>尺子：E0 裸误差–粒子数曲线</h3>
  <table>
    <thead><tr><th>N_low</th><th>M1 中位数</th><th>IQR 下</th><th>IQR 上</th><th>样本数</th><th>互差地板</th></tr></thead>
    <tbody>
{e0_rows}
    </tbody>
  </table>
  <p>曲线上还有一个关键量：<strong>参照噪声地板 {E0['reference_noise_floor']:.6f}</strong>。它的含义是——即使你拿到 1×10<sup>5</sup> 颗粒子的图，它相对“真正的收敛密度”也还差这么多，因为 2×10<sup>5</sup> 的全量参照自身也是有限粒子的抽样实现。地板以下，再多的粒子数也换不来可分辨的改善；N_eff 的报告域因此上端封在 1×10<sup>5</sup>，越过即记上溢（语义 = 达到可验证域顶）。</p>

{fig('fig03_e0_curve.png', CAP['fig03'], '3', '“尺子”：裸模拟的误差–粒子数曲线与噪声地板')}

{fig('fig04_amplification.png', CAP['fig04'], '4', '三点放大主曲线：只有右端那一点具有分辨力')}
</section>

<section class="chapter" id="sec-train">
  <div class="chapter-no">05 / 训练</div>
  <h2>训练本身没问题，是“看成绩的尺子”先饱和了</h2>

  <p class="lead">E1 两臂 × 5 种子 = 10 个 run，加上 E2 获胜臂 × 两个档位 × 5 种子 = 10 个 run，一共 {n_runs} 个训练 run，全部在单卡上确定性执行（<code>CUBLAS_WORKSPACE_CONFIG=:4096:8</code>，执行模式逐 run 记录）。每个 run 都完整跑到了早停条件，没有中途异常。但早停发生得很早。</p>

  <table>
    <thead><tr><th>变体</th><th>选型点 best_step（5 个种子）</th><th>实际停机 step（5 个种子）</th><th>预算占用（逐种子）</th></tr></thead>
    <tbody>
{tr_rows}
    </tbody>
  </table>

  <p>原因是选型准绳 <strong>val N_eff</strong> 在第一个评估周期就会贴到报告域顶 1×10<sup>5</sup>，而更新 best 用的是严格大于号——触顶之后 best 永不刷新，patience = 10 一路走完就停机。<span class="warn">全部 {n_runs} 个 run 只用了 50,000 步预算的 22–26%</span>（两臂均值 24.8% / 22.4%）。这不是训练失败，而是准绳在饱和后失去了分辨力：被选中的 ckpt 是“并列最优里最早到达的那一个”，不是训练最充分的那一刻。</p>

{fig('fig05_training_dynamics.png', CAP['fig05'], '5', '训练动力学：选型准绳在首个评估周期即饱和')}
</section>

<section class="chapter" id="sec-results">
  <div class="chapter-no">06 / 结果</div>
  <h2>{n_runs} 个 run 的读数：19 个贴在域顶，1 个落在域内</h2>

  <p class="lead">评估口径：test 束 × 站 0 × 该 run 的档位 × 20 个子样本，逐 (束, 子样本) 推理 → 对子样本取误差中位数 → 全束中位数 → 在 E0 曲线上做 log-N 线性插值反解 N_eff。下面是 {n_runs} 个 run 的全部读数。</p>

{fig('fig06_neff_and_err.png', CAP['fig06'], '6', f'{n_runs} 个 run 的 N_eff 与形状误差中位数')}

  <table>
    <thead><tr><th>run</th><th>变体</th><th>种子</th><th>N_low</th><th>站</th><th>test err 中位数</th><th>N_eff</th><th>标记</th></tr></thead>
    <tbody>
{neff_rows}
    </tbody>
  </table>

  <h3>三点放大主曲线</h3>
  <p>把 E1 与 E2 合并成三个档位的放大比（joint 臂）：</p>
  <table>
    <thead><tr><th>N_low</th><th>变体</th><th>run 数</th><th>放大比中位数</th><th>范围</th><th>上溢</th><th>该档 err 中位数</th></tr></thead>
    <tbody>
      <tr><td>3×10³</td><td>joint</td><td>5</td><td><strong>33.3×</strong></td><td>[33.3, 33.3]</td><td>5/5</td><td>{D['curve_agg']['3e3']:.4f}</td></tr>
      <tr><td>1×10⁴</td><td>joint</td><td>5</td><td><strong>10×</strong></td><td>[10, 10]</td><td>5/5</td><td>{D['curve_agg']['1e4_joint']:.4f}</td></tr>
      <tr><td>1×10⁴</td><td>independent</td><td>5</td><td>10×</td><td>[10, 10]</td><td>5/5</td><td>{D['curve_agg']['1e4_indep']:.4f}</td></tr>
      <tr><td>1×10⁵</td><td>joint</td><td>5</td><td><strong>1.0×</strong></td><td>[0.987, 1.0]</td><td>4/5</td><td>{D['curve_agg']['1e5']:.4f}</td></tr>
    </tbody>
  </table>
  <p>读法：3×10³ 与 1×10⁴ 两档的中位放大比<strong>恰等于 10<sup>5</sup>/N_low</strong>（33.3333 = 100000/3000、10.0000 = 100000/10000），即由域顶与输入之比构造得出，分辨力为零；33.3× 还超出 00 [S1] 声明的“可验证放大上限 = 10×”叙事边界。只有 1×10⁵ 档出现了非饱和点（<code>e2_joint_n100000_s2</code>，err 0.2898 &gt; 该档裸误差 {E0['points'][-1]['median']:.6f}，解得 N_eff = {D['single_non_saturated']['n_eff']:,.1f}，无标记），那正是曲线右端可信度的直接证据。</p>
</section>

<section class="chapter" id="sec-gates">
  <div class="chapter-no">07 / 门禁结论</div>
  <h2>四条门禁的结论与它们的输入快照</h2>

  <p class="lead">出口门禁全部由脚本从落盘产物判定，没有人工裁量；每条结论下面都附上判定表达式实际吃到的数值。门禁的消费维度固定为 <strong>test 束 × 站 0 × N_low = 1×10⁴</strong>，所以 E2 加入后这四条结论不变。</p>

  <table>
    <thead><tr><th>门禁</th><th>结论</th><th>判定表达式的输入数值快照</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>H1 放大存在性</strong></td>
        <td><span class="accent">成立</span></td>
        <td>获胜变体 joint @1e4 的 5 种子 test N_eff = {{{', '.join(f'{k}: 1e5' for k in SEEDS)}}}；中位数 <code>{GATES['H1']['median_n_eff']:.0f}</code> ≥ 阈值 <code>{GATES['H1']['thresholds']['median_min']:.0f}</code>，且 <code>{GATES['H1']['n_above_n_low']}/{GATES['H1']['n_seeds']}</code> 个种子 &gt; N_low = <code>{GATES['H1']['thresholds']['n_low']:.0f}</code>。该“成立”是<strong>饱和成立</strong>（域顶）。</td>
      </tr>
      <tr>
        <td><strong>H3 联合 vs 独立</strong></td>
        <td><span class="accent">成立</span></td>
        <td>5 对同种子（按种子值配对 {GATES['H3']['pairing']['pairs']}）：<code>joint_wins = {GATES['H3']['joint_wins']}/{GATES['H3']['n_pairs']}</code>。但两臂 10 个 N_eff 全部 overflow ⇒ <code>n_ties = {GATES['H3']['n_ties']}</code>、<code>n_strict_joint_wins = {GATES['H3']['n_strict_joint_wins']}</code>、<code>tied_at_domain_edge = {str(GATES['H3']['tied_at_domain_edge']).lower()}</code>。<strong>不表示联合严格更优</strong>。</td>
      </tr>
      <tr>
        <td><strong>H7 跨站泛化</strong></td>
        <td><span class="warn">不可判定</span></td>
        <td>ρ = <code>{GATES['H7']['rho']:.1f}</code>、<code>rho_median = {GATES['H7']['rho_median']:.1f}</code>、<code>weak = {str(GATES['H7']['weak']).lower()}</code>、<code>decay = {GATES['H7']['decay']}</code>；<code>n_eff_station0_median = {GATES['H7']['n_eff_station0_median']:.0f}</code>、<code>n_eff_station1_median = {GATES['H7']['n_eff_station1_median']:.0f}</code>；5 个种子两侧全部 <code>flag = overflow</code> ⇒ <code>reason_code = {GATES['H7']['reason_code']}</code>。</td>
      </tr>
      <tr>
        <td><strong>H8 细结构可恢复性</strong></td>
        <td><span class="warn">不成立</span></td>
        <td>150 束 × 13 观测量的达标率 <code>pass_rate = {GATES['H8']['pass_rate']:.4f}</code> ≤ 0.50；容差线按每束 N_eff 映射到 <code>tolerance_tier_mode = {GATES['H8']['tolerance_tier_mode']}</code>（档位分布 <code>{{'100000': 150}}</code>，即全部 150 束都落到最严档）；聚合顺序 `逐观测量先对 5 种子取中位数再判达标`；<code>seeds = {GATES['H8']['seeds']}</code>。</td>
      </tr>
    </tbody>
  </table>

{fig('fig07_h3_pairing.png', CAP['fig07'], '7', 'H3 的 5 对同种子比较：逐对连线长度为零')}

{fig('fig08a_h8_passrate.png', CAP['fig08a'], '8a', 'H8 的 13 项观测量逐项达标率')}

{fig('fig08b_dark_buckets.png', CAP['fig08b'], '8b', '暗桶 M2：裸输入 vs 模型')}

  <h3>H8 逐项明细与暗桶分项（图 8a / 8b 的数值表）</h3>
  <table>
    <thead><tr><th>#</th><th>观测量</th><th>通过束数</th><th>达标率</th></tr></thead>
    <tbody>
{ob_rows}
    </tbody>
  </table>

  <table>
    <thead><tr><th>M2 分桶</th><th>裸输入 M2 中位</th><th>模型 M2 中位</th><th>相对改进</th></tr></thead>
    <tbody>
{db_rows}
    </tbody>
  </table>
  <p>暗桶的分项（图 8b）不并入达标率（40 [S3] 要求单独报告），而它给出的是一条独立信息：在<strong>计数稀少但真实存在结构</strong>的 B1 上，模型相对裸输入把 M2 相对误差压低 {pct(DB['B1']['relative_improvement'])}；但在<strong>参照恒为零</strong>的 B0 上，模型的 M2 仍有 {DB['B0']['model_median']:.6f}——那是它在空区产生的计数。</p>

{fig('fig09a_zero_info_errs.png', CAP['fig09a'], '9a', '核心对照（上）：逐束误差分布 vs 零信息常数解')}

{fig('fig09b_zero_info_wins.png', CAP['fig09b'], '9b', '核心对照（下）：胜出束数与逐束误差比')}

{fig('fig10_h7_cross_station.png', CAP['fig10'], '10', 'H7 跨站泛化：ρ = 1.0，但两侧上溢让结论不可辨识')}
</section>

<section class="chapter" id="sec-limits">
  <div class="chapter-no">08 / 诚实边界</div>
  <h2>这些结论不能被读成什么</h2>

  <p class="lead">Phase 1 的读数里有一组彼此牵制的限定条件。把它们集中写在这里，是因为结论的正确用法取决于它们。</p>

  <h3>① 一个零信息常数解比全部 {n_runs} 个 run 都好（见图 9a、图 9b）</h3>
  <p>把 700 个 <strong>train</strong> 束的目标图逐投影归一后求平均，得到一张“均值束”图（只看 train，不碰 test）。它与输入无关、与束无关，是彻底的零信息退化解。对 150 个 test 束：误差中位数 <strong>{ZI['const_median']:.6f}</strong>、逐束标准差 {ZI['const_std']:.4f}、范围 {ZI['const_min']:.4f}–{ZI['const_max']:.4f}，<span class="warn">优于全部 {n_runs} 个 run 的 test 误差中位数（0.2218–0.2898）</span>；而且 {ZI['const_median']:.6f} &lt; 1×10<sup>5</sup> 档裸误差 {E0['points'][-1]['median']:.6f}，所以它<strong>同样触发上溢、同样满足 H1 的判定表达式、同样在 H3 上构成域顶平局</strong>。模型只在 {min(v['n_wins_vs_const'] for v in ZI['per_run'].values())}–{max(v['n_wins_vs_const'] for v in ZI['per_run'].values())}/150 束上优于它（逐束误差比中位 {min(v['median_ratio_model_over_const'] for v in ZI['per_run'].values()):.2f}–{max(v['median_ratio_model_over_const'] for v in ZI['per_run'].values()):.2f}）。只取前 200 个 train 束构造该常数解，中位数为 {ZI['const_first200_median']:.4f}，结论不变。</p>
  <blockquote>
    H1 的“成立”只说明“网络在聚合形状误差上达到了 ≥10× 的等效精度”，<strong>不能</strong>读作“网络恢复了束特异结构”。
    <cite>口径：M1（六投影平均相对 RMSE，总计数归一）；该常数解不在 20 [S5] 冻结基线族内，属 spec 层缺口。</cite>
  </blockquote>
  <p>同时也要说清楚：模型的输出<strong>不是</strong>常数——它的逐束误差标准差只有 0.0021–0.0063，常数解为 {ZI['const_std']:.4f}，相差 14–41 倍；常数解的误差随束剧烈变化，模型的误差几乎与束无关，说明它确实在逐束产生不同的输出。它大幅领先另外两条冻结基线：对 B-raw @1e4（1.2477）约 4.3–5.6 倍、对 B-moment @1e4（1.5908）约 5.5–7.2 倍。</p>

  <h3>② 域顶饱和让“成立”变成下界陈述</h3>
  <p>报告域上端是 1×10<sup>5</sup>，而目标粒子数是 2×10<sup>5</sup>。模型在 1×10<sup>4</sup> 输入下很快就撞到域顶，于是：H1 是饱和成立；H3 的 5 对全是平局（判定表达式里的“联合 ≥ 独立”把平局计入胜，所以字面成立）；H7 因为两侧都溢出而不可辨识。三点主曲线里只有 1×10<sup>5</sup> 档有非饱和点。</p>

  <h3>③ 早停于预算的 22–26%（见图 5）</h3>
  <p><code>best_step</code>：joint {', '.join(str(TR[x]['best_step']) for x in JOINT_RUNS)}；independent {', '.join(str(TR[x]['best_step']) for x in INDEP_RUNS)}。<code>steps</code>：joint {min(TR[x]['steps_total'] for x in JOINT_RUNS):,}–{max(TR[x]['steps_total'] for x in JOINT_RUNS):,}、independent {min(TR[x]['steps_total'] for x in INDEP_RUNS):,}–{max(TR[x]['steps_total'] for x in INDEP_RUNS):,}。根因是准绳饱和，后果有三：选型点语义空洞（并列最优里最早到达的那个）；被评估的 ckpt 不是训练最充分的；<strong>两臂实际步数不等</strong>（joint 比 independent 多跑若干个百分点），而 H3 恰好是 0 严格胜的平局，两件事叠加使 H3 更加无从解读。</p>

  <h3>④ κ 条件向量在 Phase 1 没有任何信息量</h3>
  <p>κ 的前 10 维是物理单位² 的 Σ 分量，量级 5.6×10<sup>-11</sup>–1.2×10<sup>-7</sup>；第 11 维 = log₁₀(6·N_low) 在整个 run 内是常数（每个 run 只消费单一档位）。即 10 维数值惰性 + 1 维恒定，条件向量不携带可用信号，软一致罚项 P_PSD（≤ ~2.5×10<sup>-13</sup>）对 NLL（0.6–21）也无影响。<span class="warn">因此 FiLM 条件化在本阶段等价于一个偏置</span>：H3 的结论只支持“共享主干带来的跨投影信息流动”这一条，不得表述为“PSD 软约束有效”或“矩条件化有效”；H1 的“物理先验”解读相应弱化；跨站鲁棒性不能归因于条件化。修复需重新设计 κ 的参数化，会作废全部训练证据，故登记为偏差、本阶段不改实现。</p>

  <h3>⑤ 偏差与不可验证项清单</h3>
  <table>
    <thead><tr><th>条目</th><th>内容</th><th>登记</th></tr></thead>
    <tbody>
      <tr><td>κ 参数化（两臂）</td><td>前 10 维尺度惰性、第 11 维 run 内恒定 ⇒ 条件化等价于偏置</td><td>30 D13 / 30 D20</td></tr>
      <tr><td>独立臂 κ 第三分量口径</td><td>spec 文字写 ρ_ij，实现取 cov_ij（与尺度惰性叠加）</td><td>30 D17</td></tr>
      <tr><td>log₁₀N 分量口径</td><td>= 图组六投影总计数（6·N_low），同档位为常量偏移 log₁₀6 ≈ 0.778</td><td>30 D14</td></tr>
      <tr><td>DDP 双卡冒烟</td><td>本阶段训练为单卡路径，30 [S3] C4 未满足（非"缓期实现"）</td><td>30 D12</td></tr>
      <tr><td>数据集身份字段</td><td>station / N_low / sub_idx 由层级命名承载而非写成 attr；束 0 无 jitter_* 字段</td><td>10 D12</td></tr>
      <tr><td>标定路径</td><td>10 [S3] C1 字面薄透镜公式会给出 77.35°（超 1° 容差），实现改用二分实算 75.00°</td><td>10 D3</td></tr>
      <tr><td>两个 run 以 tree_dirty = true 执行</td><td><code>e2_joint_n3000_s0</code> / <code>s3</code>（code_version a7b3530）⇒ 执行代码不能由所记版本唯一确定；两者均参与 3×10³ 点</td><td>40 D17</td></tr>
      <tr><td>test 评估曾被重复派发</td><td>同一 (run, 站, 档) 被派发 2–3 次，产物由最后一次覆盖写；该重复派发未造成数值损害（数值路径 diff 为空、确定性测试钉住逐位可复现）</td><td>40 D16</td></tr>
      <tr><td>确定性证据强度</td><td>10 D12 的替代契约只覆盖一个示例路径；30 D18 的 GPU 侧轨迹容差分支从未被实测行使</td><td>10 D12 / 30 D18</td></tr>
      <tr><td>已接受的既有偏差</td><td>束 0（无抖动基准束）的 |Δcorr| = 0.1366 &lt; 0.3，测试以 xfail 留痕（束 1–5 全部 ≥ 0.3）</td><td>10 D11</td></tr>
    </tbody>
  </table>
  <p>另外，H1 与 H8 可以同时给出“成立”与“不成立”而不矛盾：前者度量聚合形状误差（网络误差已低于 1×10<sup>5</sup> 颗裸模拟），后者度量 13 项物理量的逐项容差达标率（容差线被映射到最严的 1×10<sup>5</sup> 档）。按档位敏感性实测，同一组观测量若改用 1×10<sup>4</sup> 档容差，达标率为 0.9836（成立）——即 H8 的“不成立”直接由“N_eff 已被推到域顶、容差随之落到最严档”决定。</p>
</section>

<section class="chapter" id="sec-timeline">
  <div class="chapter-no">09 / 时间线</div>
  <h2>从尺子到交付的执行顺序</h2>

  <div class="timeline">
    <div class="ev">
      <div class="when">{E0M['started_at'][:19]}Z → {E0M['completed_at'][:19]}Z（耗时 {E0M['elapsed_s']:.1f} s）</div>
      <div class="what">E0 · 尺子先行：裸误差–N 曲线</div>
      <div class="who">{E0M['n_beams']} 束（val+test）× 2 站 × 5 档 × k = {E0M['k']}，产 <code>e0_curve.json</code>、互差地板与参照噪声地板 {E0['reference_noise_floor']:.6f}</div>
    </div>
    <div class="ev">
      <div class="when">{D['e1_first_start'][:19]}Z 启动</div>
      <div class="what">E1 · 两臂 × 5 种子 @ N_low = 1×10<sup>4</sup>、站 0（10 runs）</div>
      <div class="who">两卡各一条队列串行；同种子成对；每 run 落 <code>run_meta.json</code>（GC7 三元组 + 时间戳 + 执行模式）</div>
    </div>
    <div class="ev">
      <div class="when">{TOL['started_at'][:19]}Z → {TOL['completed_at'][:19]}Z（耗时 {TOL['elapsed_s']:.1f} s，与 E1 训练并行）</div>
      <div class="what">E0 续 · 容差线表（13 项观测量 × 5 档的 95% 分位）</div>
      <div class="who">只读数据集、不依赖模型产物，因此可与训练并行；quantile = {TOL['quantile']}、scope = {TOL['scope']}，是 H8 的输入</div>
    </div>
    <div class="ev">
      <div class="when">{D['e1_last_end'][:19]}Z 训练结束</div>
      <div class="what">E1 十次训练全部完成</div>
      <div class="who">20 份训练记录的 returncode 均为 0；随后进入 test 端评估 → 路由 → E3</div>
    </div>
    <div class="ev">
      <div class="when">{D['e2_start'][:19]}Z → {D['e2_train_end'][:19]}Z（训练）</div>
      <div class="what">E2 · 获胜变体 joint × {{3×10³, 1×10⁵}} × 5 种子（10 runs）</div>
      <div class="who">档位由 <code>e2_routing.json</code> 脚本化决定；与 E1 合并成三点放大主曲线</div>
    </div>
    <div class="ev">
      <div class="when">{D['chain_end'][:19]}Z</div>
      <div class="what">E3 + 门禁 + 报告：无人值守链完成</div>
      <div class="who">跨站应用（站 0 的 5 个 ckpt 直接用于站 1）→ <code>e1_gates.json</code> 四条门禁 → <code>n_eff_table.csv</code> 与主曲线图</div>
    </div>
    <div class="ev current">
      <div class="when">2026-09-15</div>
      <div class="what">阶段交付：本报告（10 张图，单文件离线可看）</div>
      <div class="who">Phase 1 三条门禁枚举结论 + 方法可用域地图主曲线 + 阶段报告，构成 00 [S3] 的 Phase 1 出口交付</div>
    </div>
  </div>
</section>

<section class="chapter" id="sec-repro">
  <div class="chapter-no">10 / 可复现性</div>
  <h2>每个数字都能从磁盘上重算</h2>

  <p class="lead">本报告的全部数字都来自落盘产物。下面前四步重建数据与尺子，后四步重建模型读数与门禁结论；最后一步跑全量测试。</p>

<pre># 0) 环境（ocelot + sdds 已锁入 sr-sim）
source "$(conda info --base)/etc/profile.d/conda.sh" &amp;&amp; conda activate sr-sim
python scripts/check_env.py

# 1) 数据集（现成产物在 studies/line2_particle_sr/data/line2_ds_v1/）
python -m src.line2.data_pipeline.build_dataset \\
    --out studies/line2_particle_sr/data/line2_ds_v1 --n-beams 1000 \\
    --data-version line2_ds_v1

# 2) 尺子先行（E0 曲线 + 容差线表）
python -m src.line2.experiments.e0 --data-dir studies/line2_particle_sr/data/line2_ds_v1
python -m src.line2.experiments.tolerance_table

# 3) E1 十次训练（两卡各一条队列；E0 未完成则拒绝启动）
python -m src.line2.experiments.run_e1 train --runs e1_joint_s0 e1_joint_s1 e1_joint_s2 e1_joint_s3 e1_joint_s4 --gpus 0
python -m src.line2.experiments.run_e1 train --runs e1_indep_s0 e1_indep_s1 e1_indep_s2 e1_indep_s3 e1_indep_s4 --gpus 1

# 4) test 端评估 → 路由 → E3 → 门禁 → 产出物
python -m src.line2.experiments.run_e1 eval --station 0 --subs 0-19 --device cuda:0
python -m src.line2.experiments.make_e2_configs --results-dir studies/line2_particle_sr/results
python -m src.line2.experiments.run_e1 gates
python -m src.line2.experiments.map_report

# 5) 全量测试
python -m pytest -q</pre>

  <h3>本报告自身的复现链（只读产物，不训练）</h3>
<pre># 派生量与图件（读 results/ 与 data/，写 reports/tools 与 reports/figs）
python studies/line2_particle_sr/reports/tools/compute_phase1_numbers.py
python studies/line2_particle_sr/reports/tools/make_phase1_figures.py
python studies/line2_particle_sr/reports/tools/make_phase1_html.py

# 数字溯源自检（HTML 中每个数字 → 磁盘来源）
python studies/line2_particle_sr/reports/tools/verify_report_numbers.py</pre>

  <h3>产物路径</h3>
  <table>
    <thead><tr><th>类别</th><th>路径</th></tr></thead>
    <tbody>
      <tr><td>尺子</td><td><code>results/e0_curve.json</code>、<code>e0_curve.meta.json</code>、<code>tolerance.json</code></td></tr>
      <tr><td>逐 run 证据</td><td><code>results/&lt;run&gt;/{{run_meta.json, summary.json, train_log.jsonl, test_eval.json, test_eval_station1.json}}</code></td></tr>
      <tr><td>门禁与路由</td><td><code>results/e1_gates.json</code>、<code>results/e2_routing.json</code></td></tr>
      <tr><td>读数表与主曲线</td><td><code>results/n_eff_table.csv</code>、<code>reports/figure_avail_domain.png</code></td></tr>
      <tr><td>数据集</td><td><code>data/line2_ds_v1/{{train,val,test}}.h5</code>、<code>manifest.json</code>、<code>data/anchor/</code></td></tr>
      <tr><td>本报告图件</td><td><code>reports/figs/fig01…fig10</code>（300 dpi PNG，与 HTML 内联版本一致）</td></tr>
      <tr><td>契约</td><td><code>docs/specs/line2/{{00,10,20,30,40}}_*.md</code> 与各自 <code>.decisions.md</code></td></tr>
    </tbody>
  </table>

  <h3>图件清单</h3>
  <table>
    <thead><tr><th>#</th><th>文件</th><th>一句话说明</th><th>字节数</th></tr></thead>
    <tbody>
{chr(10).join(
  f'      <tr><td>{i+1}</td><td><code>{name}</code></td><td>{desc}</td><td>{(FIGS / name).stat().st_size:,}</td></tr>'
  for i, (name, desc) in enumerate(FIG_INVENTORY))}
    </tbody>
  </table>
</section>

  </main>
</div>

<footer style="background: #EDE8DF; color: #666666; padding: 60px 80px; border-top: 1px solid #E8E4DC; text-align: center;">
  <div style="max-width: 1200px; margin: 0 auto;">
    <span style="font-family: Inter, sans-serif; font-size: 13px; letter-spacing: 0.05em;">line2 Phase 1 阶段报告 · 粒子级相空间超分辨率（4D 横向六投影，图像接口放大） · 2026-09-15 · 单文件离线版</span>
  </div>
</footer>

<script>
// 滚动同步：侧边栏 active 状态
(() => {{
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  const links = Array.from(sidebar.querySelectorAll('.nav-item'));
  const targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const obs = new IntersectionObserver((entries) => {{
    entries.forEach(e => {{
      if (e.isIntersecting) {{
        const id = '#' + e.target.id;
        links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === id));
      }}
    }});
  }}, {{ rootMargin: '-30% 0px -60% 0px', threshold: 0 }});
  targets.forEach(t => obs.observe(t));
}})();
</script>
</body>
</html>
"""


def main() -> None:
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
