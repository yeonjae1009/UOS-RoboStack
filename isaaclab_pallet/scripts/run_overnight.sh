#!/usr/bin/env bash
# Overnight GAT fine-tune from the cjspec_v2 (92.3) baseline with Isaac physics.
# Auto-resumes on crash (flaky Isaac startup). Stop by `touch <run_dir>/STOP`.
#
# Launch from YOUR terminal so it survives logout and has a TTY (background Isaac
# under the agent hangs):
#     nohup bash isaaclab_pallet/scripts/run_overnight.sh > /tmp/overnight.out 2>&1 &
# Watch:   tail -f isaaclab_pallet/runs/overnight_gat/train.log
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
MAX_BOXES="${MAX_BOXES:-64}"
UPDATES="${UPDATES:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
LR="${LR:-1e-6}"          # gentle fine-tune from the baseline; raise (e.g. 1e-5) for faster adaptation
MAX_FAILS="${MAX_FAILS:-20}"

mkdir -p "$RUN_DIR"
fails=0
echo "[overnight] start $(date) run_dir=$RUN_DIR num_envs=$NUM_ENVS max_boxes=$MAX_BOXES" | tee -a "$LOG"

while [ ! -f "$STOP" ]; do
  if [ -f "$RESUME" ]; then
    INIT=(--resume "$RESUME"); echo "[overnight] resume from $RESUME $(date)" | tee -a "$LOG"
  else
    INIT=(--load-model "$BEST"); echo "[overnight] warm-start from baseline $(date)" | tee -a "$LOG"
  fi

  python3 isaaclab_pallet/scripts/train_pallet_gat.py \
    --run-name "$RUN_NAME" --num-envs "$NUM_ENVS" --max-boxes "$MAX_BOXES" \
    --updates "$UPDATES" --save-interval "$SAVE_INTERVAL" --learning-rate "$LR" \
    --seed 0 --headless "${INIT[@]}" >> "$LOG" 2>&1
  code=$?

  echo "[overnight] train exited code=$code $(date)" | tee -a "$LOG"
  [ -f "$STOP" ] && break
  if [ "$code" -eq 0 ]; then echo "[overnight] reached --updates, done" | tee -a "$LOG"; break; fi

  fails=$((fails + 1))
  if [ "$fails" -ge "$MAX_FAILS" ]; then
    echo "[overnight] $fails consecutive failures, aborting" | tee -a "$LOG"; break
  fi
  echo "[overnight] restart #$fails in 15s" | tee -a "$LOG"; sleep 15
done
echo "[overnight] stopped $(date)" | tee -a "$LOG"
