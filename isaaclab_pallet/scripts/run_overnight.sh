#!/usr/bin/env bash
# Overnight GAT fine-tune from the cjspec_v2 (92.3) baseline with Isaac physics.
#
# "Full" box randomization within Isaac's limits (PhysX bakes collider sizes at
# sim start, so per-episode rescale is impossible): a LARGE box pool (MAX_BOXES)
# so each episode draws a fresh random subset, AND the box-pool SEED is cycled
# every CYCLE_UPDATES so the policy keeps seeing brand-new sizes over the night.
# Resumes weights across cycles/crashes. Stop by `touch <run_dir>/STOP`.
#
# Launch from YOUR terminal (background Isaac under the agent hangs - no TTY):
#     nohup bash isaaclab_pallet/scripts/run_overnight.sh > /tmp/overnight.out 2>&1 &
# Watch:   bash isaaclab_pallet/scripts/watch_training.sh
# Stop:    touch isaaclab_pallet/runs/overnight_gat/STOP
set -u
cd "$(dirname "$0")/../.."   # -> project root

RUN_NAME="${RUN_NAME:-overnight_gat}"
RUN_DIR="isaaclab_pallet/runs/${RUN_NAME}"
BEST="${BEST:-Online-3D-BPP-PCT/logs/experiment/cjspec_v2-2026.06.24-23-29-47/PCT-best.pt}"
RESUME="${RUN_DIR}/PCT-resume.pt"
LOG="${RUN_DIR}/train.log"
STOP="${RUN_DIR}/STOP"

NUM_ENVS="${NUM_ENVS:-32}"
MAX_BOXES="${MAX_BOXES:-256}"      # box POOL size; each episode uses a random subset (~pallet capacity)
CYCLE_UPDATES="${CYCLE_UPDATES:-5000}"   # updates per pool before cycling to a fresh random pool
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
LR="${LR:-1e-6}"                   # gentle fine-tune; raise (e.g. 1e-5) for faster adaptation
BOX_SEED="${BOX_SEED:-0}"          # starting pool seed (incremented each cycle)
MAX_FAILS="${MAX_FAILS:-20}"

mkdir -p "$RUN_DIR"
fails=0
echo "[overnight] start $(date) run_dir=$RUN_DIR num_envs=$NUM_ENVS pool=$MAX_BOXES cycle=$CYCLE_UPDATES" | tee -a "$LOG"

while [ ! -f "$STOP" ]; do
  if [ -f "$RESUME" ]; then
    INIT=(--resume "$RESUME"); echo "[overnight] resume (box_seed=$BOX_SEED) $(date)" | tee -a "$LOG"
  elif [ "${SCRATCH:-0}" = "1" ]; then
    INIT=(); echo "[overnight] SCRATCH random-init (box_seed=$BOX_SEED) $(date)" | tee -a "$LOG"
  else
    INIT=(--load-model "$BEST"); echo "[overnight] warm-start (box_seed=$BOX_SEED) $(date)" | tee -a "$LOG"
  fi

  python3 isaaclab_pallet/scripts/train_pallet_gat.py \
    --run-name "$RUN_NAME" --num-envs "$NUM_ENVS" --max-boxes "$MAX_BOXES" \
    --box-seed "$BOX_SEED" --updates "$CYCLE_UPDATES" --save-interval "$SAVE_INTERVAL" \
    --learning-rate "$LR" --seed 0 --headless "${INIT[@]}" >> "$LOG" 2>&1
  code=$?

  echo "[overnight] cycle exited code=$code $(date)" | tee -a "$LOG"
  [ -f "$STOP" ] && break

  if [ "$code" -eq 0 ]; then
    # cycle finished cleanly -> rotate to a brand-new random box pool
    BOX_SEED=$((BOX_SEED + 1)); fails=0
    echo "[overnight] pool cycle complete -> new box_seed=$BOX_SEED" | tee -a "$LOG"
    continue
  fi

  # crash -> resume the SAME pool
  fails=$((fails + 1))
  if [ "$fails" -ge "$MAX_FAILS" ]; then
    echo "[overnight] $fails consecutive failures, aborting" | tee -a "$LOG"; break
  fi
  echo "[overnight] restart #$fails (same pool) in 15s" | tee -a "$LOG"; sleep 15
done
echo "[overnight] stopped $(date)" | tee -a "$LOG"
