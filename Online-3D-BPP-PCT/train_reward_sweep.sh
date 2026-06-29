#!/usr/bin/env bash
# Run three CJ reward-shaping experiments sequentially overnight.
#
# Default:
#   - warm-start from the current cjspec_v2 best checkpoint
#   - run each reward profile for 3 hours
#   - train in the fast PCT/Gym environment, not Isaac Sim/Isaac Lab
#
# Examples:
#   bash train_reward_sweep.sh
#   HOURS_PER_PROFILE=4 NUM_PROCESSES=16 bash train_reward_sweep.sh
#   LOAD_MODEL=0 HOURS_PER_PROFILE=2 bash train_reward_sweep.sh

set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_strip_ros() {
  printf '%s' "$1" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: -
}

export LD_LIBRARY_PATH="$(_strip_ros "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(_strip_ros "${PYTHONPATH:-}")"
export PYTHONUNBUFFERED=1

HOURS_PER_PROFILE="${HOURS_PER_PROFILE:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
PRINT_INTERVAL="${PRINT_INTERVAL:-20}"
MODEL_SAVE_INTERVAL="${MODEL_SAVE_INTERVAL:-200}"
MODEL_UPDATE_INTERVAL="${MODEL_UPDATE_INTERVAL:-20000}"
LOAD_MODEL="${LOAD_MODEL:-1}"
BASE_MODEL="${BASE_MODEL:-logs/experiment/cjspec_v2-2026.06.24-23-29-47/PCT-best.pt}"
RUN_PREFIX="${RUN_PREFIX:-reward_sweep}"
SEED_BASE="${SEED_BASE:-41}"

PROFILES=(
  "floor_low"
  "smooth_low"
  "terminal_ratio"
)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/reward_sweeps/${STAMP}"
mkdir -p "$LOG_DIR"

echo "[sweep] start: $(date)"
echo "[sweep] hours/profile=${HOURS_PER_PROFILE}"
echo "[sweep] num_processes=${NUM_PROCESSES}"
echo "[sweep] load_model=${LOAD_MODEL}"
echo "[sweep] base_model=${BASE_MODEL}"
echo "[sweep] log_dir=${LOG_DIR}"
echo "[sweep] note: this does NOT run Isaac Sim/Isaac Lab; use the simulator later for validation."

for i in "${!PROFILES[@]}"; do
  PROFILE="${PROFILES[$i]}"
  NAME="${RUN_PREFIX}_${PROFILE}"
  SEED="$((SEED_BASE + i))"
  LOG_FILE="${LOG_DIR}/${NAME}.log"

  CMD=(
    .venv-pct/bin/python main.py
    --continuous
    --setting 3
    --sample-from-distribution
    --sample-left-bound 0.13
    --sample-right-bound 0.33
    --internal-node-holder 200
    --leaf-node-holder 100
    --num-processes "${NUM_PROCESSES}"
    --print-log-interval "${PRINT_INTERVAL}"
    --model-save-interval "${MODEL_SAVE_INTERVAL}"
    --model-update-interval "${MODEL_UPDATE_INTERVAL}"
    --seed "${SEED}"
    --cj-reward-profile "${PROFILE}"
  )

  if [[ "${LOAD_MODEL}" == "1" ]]; then
    CMD+=(--load-model --model-path "${BASE_MODEL}")
  fi

  echo
  echo "[sweep] profile=${PROFILE} seed=${SEED} name=${NAME}"
  echo "[sweep] log=${LOG_FILE}"
  echo "[sweep] command: ${CMD[*]}"

  set +e
  printf '%s\n' "${NAME}" | timeout "${HOURS_PER_PROFILE}h" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
  STATUS=${PIPESTATUS[1]}
  set -e

  if [[ ${STATUS} -eq 124 ]]; then
    echo "[sweep] profile=${PROFILE} reached timeout after ${HOURS_PER_PROFILE}h; continuing."
  elif [[ ${STATUS} -ne 0 ]]; then
    echo "[sweep] profile=${PROFILE} exited with status ${STATUS}; continuing to next profile."
  else
    echo "[sweep] profile=${PROFILE} finished normally."
  fi
done

echo
echo "[sweep] done: $(date)"
echo "[sweep] inspect checkpoints under logs/experiment/${RUN_PREFIX}_*"
