#!/usr/bin/env bash
set -euo pipefail

cd /home/user/Documents/assignment2_project

NV_LIB_DIRS=$(
  python3 -c 'import glob, os, site
paths = []
for sp in site.getsitepackages() if hasattr(site, "getsitepackages") else []:
    paths.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(os.pathsep.join(paths))' 2>/dev/null
)
if [ -n "$NV_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="$NV_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
START_MODEL=${START_MODEL:-isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001400.pt}
EVAL52_DIR=${EVAL52_DIR:-artifacts/random_spec_eval_50_plus_official/box_sequence}
EVAL52_SEQS=${EVAL52_SEQS:-"$(printf 'random_spec_%03d ' $(seq 0 49))box_sequence_0 box_sequence_1"}

if [ "${SMOKE:-0}" = "1" ]; then
  NUM_ENVS=${NUM_ENVS:-2}
  NUM_PACKER_WORKERS=${NUM_PACKER_WORKERS:-0}
  UPDATES=${UPDATES:-2}
  SAVE_INTERVAL=${SAVE_INTERVAL:-2}
  EVAL_INTERVAL=${EVAL_INTERVAL:-2}
  RUN_SUFFIX=${RUN_SUFFIX:-smoke}
  ENABLE_WANDB=${ENABLE_WANDB:-0}
else
  NUM_ENVS=${NUM_ENVS:-128}
  NUM_PACKER_WORKERS=${NUM_PACKER_WORKERS:-16}
  UPDATES=${UPDATES:-1500}
  SAVE_INTERVAL=${SAVE_INTERVAL:-250}
  EVAL_INTERVAL=${EVAL_INTERVAL:-50}
  RUN_SUFFIX=${RUN_SUFFIX:-}
  ENABLE_WANDB=${ENABLE_WANDB:-1}
fi

MAX_BOXES=${MAX_BOXES:-256}
LEARNING_RATE=${LEARNING_RATE:-5e-7}
DEPLOY_EVAL_TIMEOUT=${DEPLOY_EVAL_TIMEOUT:-1800}
WANDB_PROJECT=${WANDB_PROJECT:-assignment2-pallet}
WANDB_GROUP=${WANDB_GROUP:-sun_c1400_workers16_rand52_deploy}

WANDB_ARGS=()
if [ "$ENABLE_WANDB" = "1" ]; then
  WANDB_ARGS=(--wandb --wandb-project "$WANDB_PROJECT")
fi

for item in \
  sun_gap_guard_v1:sun_c1400_env128_workers16_rand52official_gap_guard_v1_lr5e7_cont \
  sun_gap_fill_v2:sun_c1400_env128_workers16_rand52official_gap_fill_v2_lr5e7_cont
do
  profile=${item%%:*}
  run_name=${item#*:}
  if [ -n "$RUN_SUFFIX" ]; then
    run_name="${run_name}_${RUN_SUFFIX}"
  fi
  run_dir=isaaclab_pallet/runs/$run_name
  log=$run_dir/train.log

  mkdir -p "$run_dir"
  echo "[c1400-workers16] start profile=$profile run=$run_name start=$START_MODEL envs=$NUM_ENVS workers=$NUM_PACKER_WORKERS updates=$UPDATES eval_dir=$EVAL52_DIR eval=$EVAL52_SEQS $(date)" | tee -a "$log"

  "$PYTHON_BIN" isaaclab_pallet/scripts/train_pallet_gat.py \
    --run-name "$run_name" \
    --reward-profile "$profile" \
    --num-envs "$NUM_ENVS" \
    --num-packer-workers "$NUM_PACKER_WORKERS" \
    --max-boxes "$MAX_BOXES" \
    --box-seed 1 \
    --updates "$UPDATES" \
    --save-interval "$SAVE_INTERVAL" \
    --eval-interval "$EVAL_INTERVAL" \
    --learning-rate "$LEARNING_RATE" \
    --candidate-rerank-k 0 \
    --candidate-diversity-center-m 0.05 \
    --box-seq-dir "$EVAL52_DIR" \
    --eval-sequences $EVAL52_SEQS \
    --seed 0 \
    --headless \
    "${WANDB_ARGS[@]}" \
    --wandb-name "$run_name" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-tags sun c1400 env128 workers16 rand52official deploybest \
    --load-model "$START_MODEL" \
    --deploy-eval-best \
    --deploy-eval-buffer-size 0 \
    --deploy-eval-timeout "$DEPLOY_EVAL_TIMEOUT" >> "$log" 2>&1

  code=$?
  echo "[c1400-workers16] profile=$profile exited code=$code $(date)" | tee -a "$log"
done
