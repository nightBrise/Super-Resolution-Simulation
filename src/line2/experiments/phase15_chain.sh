#!/usr/bin/env bash
# line2 Phase 1.5 编排链（`50` [S5]）：E0' 尺子三件 → 10 训练（双卡，按臂整分）→ 60 评估
#   → 路由（val 段 val_nll）→ 获胜臂对齐仪器 → 门禁 G1–G5 → v2 门禁报告。
#
# 纪律（`50` Global Constraints）：
#   - 尺子先行：E0' 三件（曲线 / 容差线 / B-const 两站）全部先落盘；编排器的 train 子命令
#     在每个 run 启动前复验三件的时间戳，违例拒绝启动（本脚本不绕过该门禁）；
#   - 环境：sr-sim + scripts/check_env.py；确定性环境变量由编排器逐 run 注入；
#   - test 只评一次（`00` GC3）：eval 子命令对已存在的 eval_*.json 默认跳过。
#
# 执行：nohup bash src/line2/experiments/phase15_chain.sh（日志落 results/phase15_chain.log）；
# 每步失败即记 ERROR 并以非零退出，不在失败后继续后续步骤。
set -o pipefail
REPO=/home/zhangny/Super-Resolution-Simulation
cd "$REPO" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate sr-sim >>/dev/null 2>&1

R=studies/line2_particle_sr/results
D=studies/line2_particle_sr/data/line2_ds_v2
LOG=$R/phase15_chain.log

log() { echo "[e15chain $(date '+%F %T')] $*" >> "$LOG"; }
die() { log "ERROR: $*"; echo "[e15chain] ERROR: $*" >&2; exit 1; }

log "start; check_env"
python scripts/check_env.py >> "$LOG" 2>&1 || die "check_env 未通过"

# ---------------------------------------------------------------- 1) E0' 尺子三件
# 三件全部先于任何训练落盘（`50` [S5] C1）；顺序 = 曲线 → 容差线 → B-const。
python -u -m src.line2.experiments.e0 --data-dir "$D" --out "$R/e0_curve_v2.json" \
  >> "$LOG" 2>&1 || die "E0' 裸误差–N 曲线失败"
log "E0' curve done"
python -u -m src.line2.experiments.e0_v2 --data-dir "$D" --results-dir "$R" tolerance \
  >> "$LOG" 2>&1 || die "E0' 容差线失败"
log "E0' tolerance done"
python -u -m src.line2.experiments.e0_v2 --data-dir "$D" --results-dir "$R" bconst \
  >> "$LOG" 2>&1 || die "E0' B-const 失败"
log "E0' bconst done"

# ------------------------------------------------- 2) 10 份 config（过 [S4] C3 矩阵断言）
python -u -m src.line2.experiments.phase15_orchestrator configs \
  --results-dir "$R" --data-dir "$D" >> "$LOG" 2>&1 || die "phase15 configs 失败"
log "configs done"

# ------------------------------------------------- 3) 10 训练（双卡队列，按臂整分）
python3 - "$R" <<'PY' >> "$LOG" 2>&1
import json, sys
d = json.load(open(f"{sys.argv[1]}/e15_matrix.json"))
print("[e15chain] 矩阵:", len(d["configs"]), "runs；params:", d["validation"]["params_total"],
      "；station_check:", d["validation"]["station_check"])
PY
python -u -m src.line2.experiments.phase15_orchestrator train \
  --results-dir "$R" --data-dir "$D" --gpus 0 1 >> "$LOG" 2>&1 || die "phase15 train 失败"
log "10 runs 完成；等待 summary 齐备"
for _ in $(seq 1 60); do
  n=$(ls "$R"/e15_*/summary.json 2>/dev/null | wc -l)
  [ "$n" -ge 10 ] && break
  sleep 60
done
[ "$(ls "$R"/e15_*/summary.json 2>/dev/null | wc -l)" -ge 10 ] || die "summary 不足 10 份"

# --------------------------------------------- 4) 60 评估（10 run × 3 档 × 2 站，`50` [S5] C8）
python -u -m src.line2.experiments.phase15_orchestrator eval \
  --results-dir "$R" --data-dir "$D" --device cuda:0 >> "$LOG" 2>&1 || die "phase15 eval 失败"
log "60 evals done"

# ---------------------------------------------------------- 5) 路由（val 段 val_nll）
python -u -m src.line2.experiments.phase15_orchestrator route \
  --results-dir "$R" >> "$LOG" 2>&1 || die "phase15 route 失败"
log "routing done: $R/e15_routing.json"

# ------------------------------------------- 6) 获胜臂对齐仪器（G2 的输入，`50` [S5] C3）
python -u -m src.line2.experiments.phase15_orchestrator align \
  --results-dir "$R" --data-dir "$D" --device cuda:0 >> "$LOG" 2>&1 || die "phase15 align 失败"
log "alignment done"

# ---------------------------------------------------------- 7) 出口门禁 G1–G5
python -u -m src.line2.experiments.phase15_orchestrator gates \
  --results-dir "$R" >> "$LOG" 2>&1 || die "phase15 gates 失败"
log "gates done: $R/e15_gates.json"

# ---------------------------------------------------------- 8) v2 门禁报告（`50` [S5] C9）
python -u -m src.line2.experiments.phase15_orchestrator map \
  --results-dir "$R" >> "$LOG" 2>&1 || die "phase15 map 失败"
log "map done"
log "PHASE15 CHAIN COMPLETE"
