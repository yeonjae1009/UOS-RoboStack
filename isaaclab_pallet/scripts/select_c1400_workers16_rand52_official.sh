#!/usr/bin/env bash
set -euo pipefail

cd /home/user/Documents/assignment2_project

EVAL52_DIR=${EVAL52_DIR:-artifacts/random_spec_eval_50_plus_official/box_sequence}
EVAL52_SEQS=${EVAL52_SEQS:-"$(printf 'random_spec_%03d ' $(seq 0 49))box_sequence_0 box_sequence_1"}
DEVICE=${DEVICE:-cuda:0}

for run_name in \
  sun_c1400_env128_workers16_rand52official_gap_guard_v1_lr5e7_cont \
  sun_c1400_env128_workers16_rand52official_gap_fill_v2_lr5e7_cont
do
  run_dir=isaaclab_pallet/runs/$run_name
  work_dir=artifacts/isaac_best_selection/$run_name
  echo "[c1400-official-select] run=$run_name eval_dir=$EVAL52_DIR eval=$EVAL52_SEQS $(date)"

  python3 isaaclab_pallet/scripts/select_best_by_isaac.py \
    --run-dir "$run_dir" \
    --patterns "deploy_eval/candidate-*.pt" "PCT-update-*.pt" "PCT-latest.pt" "PCT-best.pt" \
    --work-dir "$work_dir" \
    --box-seq-dir "$EVAL52_DIR" \
    --sequences $EVAL52_SEQS \
    --device "$DEVICE" \
    --out-name PCT-best-isaac-rand52.pt
done
