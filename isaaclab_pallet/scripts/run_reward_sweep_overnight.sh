#!/usr/bin/env bash
# Run the existing Isaac Lab GAT overnight pipeline for several reward profiles.
#
# This is intentionally a thin wrapper around the README pipeline:
#   isaaclab_pallet/scripts/train_pallet_gat.py
#
# Launch:
#   nohup bash isaaclab_pallet/scripts/run_reward_sweep_overnight.sh > /tmp/reward_sweep.out 2>&1 &
#
# Watch:
#   tail -f /tmp/reward_sweep.out
#   bash isaaclab_pallet/scripts/watch_training.sh
#
# Stop the whole sweep:
#   touch isaaclab_pallet/runs/reward_sweep_STOP

set -u
cd "$(dirname "$0")/../.."   # -> project root

RUN_PREFIX="${RUN_PREFIX:-reward_sweep}"
BEST="${BEST:-isaaclab_pallet/runs/overnight_full/PCT-latest.pt}"
FALLBACK_BEST="Online-3D-BPP-PCT/logs/experiment/cjspec_v2-2026.06.24-23-29-47/PCT-best.pt"
STOP_ALL="isaaclab_pallet/runs/reward_sweep_STOP"

NUM_ENVS="${NUM_ENVS:-32}"
MAX_BOXES="${MAX_BOXES:-256}"
UPDATES_PER_PROFILE="${UPDATES_PER_PROFILE:-2500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
LR="${LR:-1e-6}"
BOX_SEED="${BOX_SEED:-0}"

if [ "$#" -gt 0 ]; then
  PROFILES=("$@")
else
  PROFILES=(floor_low smooth_low terminal_ratio finish_ratio)
fi

if [ ! -s "$BEST" ]; then
  echo "[reward-sweep] BEST not found: $BEST"
  echo "[reward-sweep] fallback: $FALLBACK_BEST"
  BEST="$FALLBACK_BEST"
fi

echo "[reward-sweep] start $(date)"
echo "[reward-sweep] profiles=${PROFILES[*]}"
echo "[reward-sweep] best=$BEST num_envs=$NUM_ENVS max_boxes=$MAX_BOXES updates/profile=$UPDATES_PER_PROFILE lr=$LR"

for i in "${!PROFILES[@]}"; do
  [ -f "$STOP_ALL" ] && break

  profile="${PROFILES[$i]}"
  run_name="${RUN_PREFIX}_${profile}"
  run_dir="isaaclab_pallet/runs/${run_name}"
  resume="${run_dir}/PCT-resume.pt"
  log="${run_dir}/train.log"
  seed="$((BOX_SEED + i))"

  mkdir -p "$run_dir"
  if [ -s "$resume" ]; then
    init=(--resume "$resume")
    echo "[reward-sweep] resume profile=$profile run=$run_name seed=$seed $(date)" | tee -a "$log"
  else
    init=(--load-model "$BEST")
    echo "[reward-sweep] warm-start profile=$profile run=$run_name seed=$seed $(date)" | tee -a "$log"
  fi

  python3 isaaclab_pallet/scripts/train_pallet_gat.py \
    --run-name "$run_name" \
    --reward-profile "$profile" \
    --num-envs "$NUM_ENVS" \
    --max-boxes "$MAX_BOXES" \
    --box-seed "$seed" \
    --updates "$UPDATES_PER_PROFILE" \
    --save-interval "$SAVE_INTERVAL" \
    --eval-interval "$EVAL_INTERVAL" \
    --learning-rate "$LR" \
    --seed 0 \
    --headless \
    "${init[@]}" >> "$log" 2>&1
  code=$?

  echo "[reward-sweep] profile=$profile exited code=$code $(date)" | tee -a "$log"
  if [ "$code" -ne 0 ]; then
    echo "[reward-sweep] stopping after failed profile=$profile; inspect $log"
    exit "$code"
  fi
done

echo "[reward-sweep] done $(date)"
echo "[reward-sweep] runs:"
for profile in "${PROFILES[@]}"; do
  echo "  isaaclab_pallet/runs/${RUN_PREFIX}_${profile}"
done
