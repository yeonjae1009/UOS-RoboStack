#!/usr/bin/env bash
set -euo pipefail

cd /home/user/Documents/assignment2_project

RUN_DIR=${RUN_DIR:-isaaclab_pallet/runs/sun_c1400_env128_workers16_rand52official_gap_fill_v2_lr5e7_cont}
BASE_WORK=${BASE_WORK:-artifacts/isaac_best_selection/sun_c1400_env128_workers16_rand52official_gap_fill_v2_lr5e7_cont_parallel_extra}
EVAL52_DIR=${EVAL52_DIR:-artifacts/random_spec_eval_50_plus_official/box_sequence}
EVAL52_SEQS=${EVAL52_SEQS:-"$(printf 'random_spec_%03d ' $(seq 0 49))box_sequence_0 box_sequence_1"}
DEVICE=${DEVICE:-cuda:0}
PARALLEL_JOBS=${PARALLEL_JOBS:-3}

mkdir -p "$BASE_WORK/logs"

candidates=(
  candidate-000800
  candidate-000900
  candidate-000950
  candidate-001100
  candidate-001150
  candidate-001250
  candidate-001300
)

run_one() {
  local name=$1
  local work_dir="$BASE_WORK/$name"
  local log="$BASE_WORK/logs/$name.out"
  echo "[parallel-select] start $name $(date)" | tee "$log"
  python3 isaaclab_pallet/scripts/select_best_by_isaac.py \
    --run-dir "$RUN_DIR" \
    --patterns "deploy_eval/$name.pt" \
    --work-dir "$work_dir" \
    --box-seq-dir "$EVAL52_DIR" \
    --sequences $EVAL52_SEQS \
    --device "$DEVICE" \
    --out-name "PCT-best-isaac-rand52-$name.pt" \
    >> "$log" 2>&1
  echo "[parallel-select] done $name $(date)" >> "$log"
}

status=0
pids=()
names=()

for name in "${candidates[@]}"; do
  run_one "$name" &
  pids+=("$!")
  names+=("$name")

  if (( ${#pids[@]} >= PARALLEL_JOBS )); then
    if ! wait "${pids[0]}"; then
      echo "[parallel-select] failed ${names[0]}" >&2
      status=1
    fi
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
  fi
done

for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[parallel-select] failed ${names[$i]}" >&2
    status=1
  fi
done

summary="$BASE_WORK/summary.tsv"
{
  echo -e "checkpoint\tavg_score\tsuccess\tresult_json"
  for name in "${candidates[@]}"; do
    part="$BASE_WORK/$name/summary.tsv"
    if [[ -f "$part" ]]; then
      tail -n +2 "$part"
    else
      echo -e "$name.pt\tNA\tFalse\tmissing"
    fi
  done
} > "$summary"

echo "[parallel-select] merged summary -> $summary"
exit "$status"
