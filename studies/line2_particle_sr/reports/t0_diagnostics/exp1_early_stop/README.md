# T0-实验1：早停损失量化（line2 Phase 1，20 个训练 run）

**判定：证实 —— 早停损失真实存在，且量级很大。** 20/20 run 的 150/150 束全部变好，
中位 Δerr（last − best_val）在 −0.212 ~ −0.098 之间，全部 run 的 95% CI 不含 0 且为负。

## 预注册判定规则的核对

规则：joint 臂 5 个种子中 ≥4 个的中位 Δerr 的 95% CI 不含 0 且为负（last 更好）→ 判定"早停
损失真实存在且可通过换选型准绳挽回"。

| 分组（variant / N_low） | run 数 | CI 显著为负 | Δerr 中位（组内范围） |
|---|---|---|---|
| `e1_joint`（joint, 1e4） | 5/5 | 5/5 | −0.1602（−0.1824 ~ −0.1425） |
| `e1_indep`（independent, 1e4） | 5/5 | 5/5 | −0.2013（−0.2093 ~ −0.0982） |
| `e2_joint_n3000`（joint, 3e3） | 5/5 | 5/5 | −0.1789（−0.1958 ~ −0.1493） |
| `e2_joint_n100000`（joint, 1e5） | 5/5 | 5/5 | −0.1817（−0.2120 ~ −0.1674） |

三条 joint 组共 15/15 run 满足规则（规则只要求 e1 的 5 个种子里的 ≥4 个 ⇒ 4 倍余量）。

## 关键数字

- **err_median**：best_val 中位 **0.2724**（0.2218–0.2898）→ last 中位 **0.0884**（0.0707–0.1256）；
  相对降幅中位 **−66.2%**（−74.5% ~ −44.3%）。
- **3000/3000 束改善**：20 个 run × 150 束 = 3000 个配对差无一为正，逐束改好比例 1.00；
  束级 |Δerr| 落在 0.0829–0.2229。
- **val_nll**：从冻结选型点（step 1000–3000）到末步（11000–13000）中位降 **24.5%**（8.6%–26.7%）；
  argmin(val_nll) == 末步的 run 有 9/20，其余偏离 ≤1000 步（一例 2000 步）——轨迹接近单调下降，
  末步仍在下降（未收敛）。
- **步数**：全部 20 个 run 止于 11000–13000 步 = 预注册预算 50,000 的 **22–26%**；
  末步与 val N_eff 首个饱和步（1000–3000）相差 10 个评估周期（= patience）。
- **N_eff 不可分辨**：last 全部在报告域上溢（100000），best_val 19/20 也上溢（余一例 98724）
  ——现门禁口径（H1/H3/H8 消费 N_eff）看不见这部分 test 性能差异，只有 err 分辨得出。
- **确定性对照**：best_val.ckpt 以同一命令复算 vs 既有 `test_eval.json`，20/20 run 束级 err
  **逐位一致**（最大 |Δerr| = 0），本目录复算与 Phase 1 原始评估在同一数值口径上。

## 方法（逐参数复刻原始评估，只改两处）

原始评估由 `studies/line2_particle_sr/results/phase1_chain.sh` →
`run_e1 eval --station 0 --subs 0-19 --device cuda:0` 派发，其内部 argv（`run_e1.cmd_eval`）：

```bash
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
python -u -m src.line2.experiments.evaluate \
  --ckpt studies/line2_particle_sr/results/<run>/{best_val.ckpt|last.ckpt} \
  --data-dir studies/line2_particle_sr/data/line2_ds_v1 \
  --split test --station 0 --n-low <该 run 档位: e1 两臂 10000 / e2 3000 或 100000> \
  --subs 0-19 --curve studies/line2_particle_sr/results/e0_curve.json \
  --out studies/line2_particle_sr/reports/t0_diagnostics/exp1_early_stop/<run>_{bestval_eval.json|last_eval.json} \
  --device cuda:0
```

只改 `--ckpt`（best_val.ckpt → last.ckpt）与 `--out`（指向本目录）——`run_e1 eval` 会跳过已存在的
`test_eval.json`（test 只评一次），故不经过 `run_e1` 派发端而直接调用同一 `evaluate.main`。

配对与统计：同 run、同 beam_id、同 20 个子样本 → 逐束 Δ = err_last − err_best（负 = last 更好）；
配对 bootstrap 10,000 次束级重采样，中位 Δ 的 95% 百分位 CI，种子 `SeedSequence(20260825)`；
Wilcoxon 符号秩检验（scipy，双尾）。N_eff / err 一律来自 `src/line2/metrics` 的既有代码路径
（`evaluate_run` → `group_rel_rmse` / `n_eff_from_err` / `observable_vector`），未另写指标。

## 注意与边界

-  **last.ckpt 是"训练最充分的可用 ckpt"，不是"最优 ckpt"**：Δerr 给出早停损失的**量级与下界**；
  预算 50,000 只用了 22–26%，val_nll 在末步仍下降，继续训练可能不止此收益。
-  **策略级 A/B 未做**：实测的是"末步 ckpt vs 冻结选型点 ckpt"的差，不是"以 val_nll 选型并重训"
  与"以 N_eff 选型"的对照。val_nll 的 argmin 步 ≤ 末步，val_nll 选型会选到更早的 ckpt，
  其 test 表现本次未实测（需重训或另行重评）。
-  **CI 很窄（±0.0005）是配对结构的产物**：束级配对差的标准差仅 ~0.004，且逐束同向；
  跨种子一致性（每臂 5/5，run 级 Δ 全为负）才是主要证据。
-  **未检验 test 集以外**：本实验只评 `split test` 站 0，站 1（跨站）与 OOD 未覆盖。
-  **N_eff 主判据在此饱和**：若要复跑选型，需换用能在报告域内分辨的准绳（如 val_nll），
  否则门禁仍然看不见差异。
-  诊断脚本只 import 生产代码，未修改 `src/**`、`docs/**`、`tests/**`；`results/` 与 `data/` 全程只读，
  所有输出落在本目录。

## 文件清单

| 文件 | 内容 |
|---|---|
| `run_evals.py` | 评估派发（20 run × {last, bestval} = 40 次，4 并发），产 `eval_dispatch.json` |
| `analyze_trajectory.py` | 步骤 2a：train_log.jsonl 的 val_nll 轨迹 → `trajectory.csv` |
| `analyze_pairing.py` | 步骤 2c：逐束配对 + bootstrap + Wilcoxon → `pairing.csv` |
| `build_summary_table.py` | 步骤 3：合并 → `summary_table.csv`（含 last.ckpt 内嵌 step 核对） |
| `summary_table.csv` | 20 行汇总表（err best/last/Δ/CI/p/val_nll/steps/N_eff/对照一致性） |
| `trajectory.csv`、`pairing.csv` | 中间产物（轨迹 / 统计） |
| `<run>_last_eval.json`（20） | last.ckpt 的 test 评估（station 0、split test、20 子样本） |
| `<run>_bestval_eval.json`（20） | best_val.ckpt 的复算对照（用于确定性核对） |
| `eval_dispatch.json`、`run_evals_driver.log`、`<run>_{last,bestval}.log` | 派发记录与逐次 stdout |
