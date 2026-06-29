#!/usr/bin/env bash
# Train reward-profile candidates and put Isaac validation inside the loop.
#
# Flow per round:
#   1. Train floor_low, smooth_low, terminal_ratio from the current base checkpoint.
#   2. Export each trained checkpoint to contest JSON for box_sequence_0/1.
#   3. Run Isaac headless validation.
#   4. Pick the best Isaac-scored checkpoint as the next round's base.
#
# Examples:
#   bash train_isaac_outer_loop.sh
#   ROUNDS=2 HOURS_PER_CHUNK=2 NUM_PROCESSES=16 bash train_isaac_outer_loop.sh
#   BASE_MODEL=logs/experiment/.../PCT-best.pt bash train_isaac_outer_loop.sh

set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_strip_ros() {
  printf '%s' "$1" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd: -
}

export LD_LIBRARY_PATH="$(_strip_ros "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(_strip_ros "${PYTHONPATH:-}")"
export PYTHONUNBUFFERED=1

PCT_PY="${PCT_PY:-.venv-pct/bin/python}"
ISAAC_PY="${ISAAC_PY:-/home/robotics/isaac-sim-5.1/python.sh}"
SIM_DIR="${SIM_DIR:-../palletizing_simulator}"
SEQ_DIR="${SEQ_DIR:-${SIM_DIR}/box_sequence}"
SIMULATOR="${SIMULATOR:-${SIM_DIR}/simulator.py}"

ROUNDS="${ROUNDS:-1}"
HOURS_PER_CHUNK="${HOURS_PER_CHUNK:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
PRINT_INTERVAL="${PRINT_INTERVAL:-20}"
MODEL_SAVE_INTERVAL="${MODEL_SAVE_INTERVAL:-50}"
MODEL_UPDATE_INTERVAL="${MODEL_UPDATE_INTERVAL:-20000}"
BASE_MODEL="${BASE_MODEL:-logs/experiment/cjspec_v2-2026.06.24-23-29-47/PCT-best.pt}"
RUN_PREFIX="${RUN_PREFIX:-reward_isaac}"
SEED_BASE="${SEED_BASE:-410}"

PROFILES=(
  "floor_low"
  "smooth_low"
  "terminal_ratio"
)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/reward_sweeps/${STAMP}_isaac_outer"
SUMMARY_TSV="${LOG_DIR}/isaac_summary.tsv"
mkdir -p "$LOG_DIR"

printf 'round\tprofile\tstatus\tavg_score\tfailures\tcollapse\tdrop\toob\tckpt\tresult_json\n' > "$SUMMARY_TSV"

latest_run_dir() {
  local name="$1"
  find logs/experiment -maxdepth 1 -type d -name "${name}-*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

choose_checkpoint() {
  local run_dir="$1"
  if [[ -s "${run_dir}/PCT-best.pt" ]]; then
    printf '%s\n' "${run_dir}/PCT-best.pt"
    return 0
  fi
  find "$run_dir" -maxdepth 1 -type f -name 'PCT-*.pt' ! -name 'PCT-resume.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

best_checkpoint_from_summary() {
  "$PCT_PY" - "$SUMMARY_TSV" <<'PY'
import csv
import sys

path = sys.argv[1]
best = None
with open(path, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["status"] != "ok":
            continue
        score = float(row["avg_score"])
        failures = int(row["failures"])
        # Higher score wins; tie-breaker prefers fewer physics failures.
        key = (score, -failures)
        if best is None or key > best[0]:
            best = (key, row["ckpt"])

print(best[1] if best else "")
PY
}

append_failed_summary() {
  local round="$1"
  local profile="$2"
  local ckpt="$3"
  local result_json="$4"
  printf '%s\t%s\tfailed\t0.00\t999999\t999999\t999999\t999999\t%s\t%s\n' \
    "$round" "$profile" "$ckpt" "$result_json" >> "$SUMMARY_TSV"
}

append_isaac_summary() {
  local round="$1"
  local profile="$2"
  local ckpt="$3"
  local result_json="$4"

  "$PCT_PY" - "$round" "$profile" "$ckpt" "$result_json" >> "$SUMMARY_TSV" <<'PY'
import json
import sys

round_id, profile, ckpt, result_json = sys.argv[1:5]
with open(result_json, encoding="utf-8") as f:
    data = json.load(f)

files = data.get("files", [])
results = data.get("results", {})
summary = data.get("summary", {})

avg_score = float(results.get("avg_score", 0.0))
collapse = int(summary.get("total_collapse", 0))
drop = int(summary.get("total_drop", 0))
oob = int(summary.get("total_out_of_bounds", 0))
failures = int(summary.get("total_failure_boxes", collapse + drop + oob))
height_fail = int(summary.get("height_overflow_episodes", 0))
failed_eps = int(summary.get("failure_episodes", 0))
status = "ok" if failures == 0 and height_fail == 0 and failed_eps == 0 else "physics_fail"

print(
    f"{round_id}\t{profile}\t{status}\t{avg_score:.2f}\t"
    f"{failures}\t{collapse}\t{drop}\t{oob}\t{ckpt}\t{result_json}"
)
PY
}

validate_checkpoint() {
  local round="$1"
  local profile="$2"
  local ckpt="$3"
  local tag="r${round}_${profile}"
  local val_dir="${LOG_DIR}/isaac/${tag}"
  local input_dir="${val_dir}/algorithm_results"
  local output_dir="${val_dir}/sim_results"
  local input_abs
  local output_abs
  local sim_abs

  mkdir -p "$input_dir" "$output_dir"
  input_abs="$(readlink -f "$input_dir")"
  output_abs="$(readlink -f "$output_dir")"
  sim_abs="$(readlink -f "$SIMULATOR")"

  echo "[isaac] export profile=${profile} round=${round} ckpt=${ckpt}"
  for name in box_sequence_0 box_sequence_1; do
    "$PCT_PY" export_to_contest.py \
      --model-path "$ckpt" \
      --box-sequence "${SEQ_DIR}/${name}.json" \
      --out "${input_dir}/${name}.json" \
      --internal-node-holder 200 \
      --leaf-node-holder 100 \
      --setting 3 \
      > "${val_dir}/export_${name}.log" 2>&1
  done

  echo "[isaac] run simulator profile=${profile} round=${round}"
  set +e
  "$ISAAC_PY" "$sim_abs" \
    --input-dir "$input_abs" \
    --output "$output_abs" \
    > "${val_dir}/isaac.log" 2>&1
  local status=$?
  set +e

  local result_json="${output_dir}/result.json"
  if [[ $status -ne 0 || ! -s "$result_json" ]]; then
    echo "[isaac] validation failed profile=${profile} status=${status}; see ${val_dir}/isaac.log"
    append_failed_summary "$round" "$profile" "$ckpt" "$result_json"
    return 0
  fi

  append_isaac_summary "$round" "$profile" "$ckpt" "$result_json"
  "$PCT_PY" - "$result_json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
print("[isaac] avg_score={:.2f} failures={} collapse={} drop={} oob={}".format(
    float(data.get("results", {}).get("avg_score", 0.0)),
    int(data.get("summary", {}).get("total_failure_boxes", 0)),
    int(data.get("summary", {}).get("total_collapse", 0)),
    int(data.get("summary", {}).get("total_drop", 0)),
    int(data.get("summary", {}).get("total_out_of_bounds", 0)),
))
PY
}

echo "[outer] start: $(date)"
echo "[outer] rounds=${ROUNDS}"
echo "[outer] hours/chunk=${HOURS_PER_CHUNK}"
echo "[outer] num_processes=${NUM_PROCESSES}"
echo "[outer] initial_base=${BASE_MODEL}"
echo "[outer] log_dir=${LOG_DIR}"
echo "[outer] summary=${SUMMARY_TSV}"

CURRENT_BASE="$BASE_MODEL"

for round in $(seq 1 "$ROUNDS"); do
  echo
  echo "[outer] ===== round ${round}/${ROUNDS} base=${CURRENT_BASE} ====="

  for i in "${!PROFILES[@]}"; do
    PROFILE="${PROFILES[$i]}"
    NAME="${RUN_PREFIX}_${PROFILE}_r${round}"
    SEED="$((SEED_BASE + round * 10 + i))"
    LOG_FILE="${LOG_DIR}/${NAME}.log"

    CMD=(
      "$PCT_PY" main.py
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

    if [[ -s "$CURRENT_BASE" ]]; then
      CMD+=(--load-model --model-path "$CURRENT_BASE")
    else
      echo "[outer] warning: base checkpoint not found, training from scratch: ${CURRENT_BASE}"
    fi

    echo
    echo "[outer] train profile=${PROFILE} round=${round} seed=${SEED}"
    echo "[outer] log=${LOG_FILE}"
    echo "[outer] command: ${CMD[*]}"

    set +e
    printf '%s\n' "${NAME}" | timeout "${HOURS_PER_CHUNK}h" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    STATUS=${PIPESTATUS[1]}
    set +e

    if [[ ${STATUS} -eq 124 ]]; then
      echo "[outer] profile=${PROFILE} hit ${HOURS_PER_CHUNK}h timeout; validating saved checkpoint."
    elif [[ ${STATUS} -ne 0 ]]; then
      echo "[outer] profile=${PROFILE} exited with status ${STATUS}; trying to validate any saved checkpoint."
    else
      echo "[outer] profile=${PROFILE} finished normally; validating saved checkpoint."
    fi

    RUN_DIR="$(latest_run_dir "$NAME")"
    if [[ -z "$RUN_DIR" ]]; then
      echo "[outer] no run dir found for ${NAME}; skipping Isaac validation."
      append_failed_summary "$round" "$PROFILE" "" ""
      continue
    fi

    CKPT="$(choose_checkpoint "$RUN_DIR")"
    if [[ -z "$CKPT" || ! -s "$CKPT" ]]; then
      echo "[outer] no checkpoint found in ${RUN_DIR}; skipping Isaac validation."
      append_failed_summary "$round" "$PROFILE" "" ""
      continue
    fi

    validate_checkpoint "$round" "$PROFILE" "$CKPT"
  done

  NEXT_BASE="$(best_checkpoint_from_summary)"
  if [[ -n "$NEXT_BASE" && -s "$NEXT_BASE" ]]; then
    CURRENT_BASE="$NEXT_BASE"
    printf '%s\n' "$CURRENT_BASE" > "${LOG_DIR}/best_isaac_checkpoint.txt"
    echo "[outer] selected next base from Isaac score: ${CURRENT_BASE}"
  else
    echo "[outer] no physics-clean checkpoint found yet; keeping current base: ${CURRENT_BASE}"
  fi
done

echo
echo "[outer] done: $(date)"
echo "[outer] summary: ${SUMMARY_TSV}"
echo "[outer] best checkpoint file: ${LOG_DIR}/best_isaac_checkpoint.txt"
if [[ -s "${LOG_DIR}/best_isaac_checkpoint.txt" ]]; then
  echo "[outer] best checkpoint: $(cat "${LOG_DIR}/best_isaac_checkpoint.txt")"
fi
