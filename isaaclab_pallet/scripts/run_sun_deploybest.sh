#!/usr/bin/env bash
set -u

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

BEST=isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt
EVAL_DIR=artifacts/random_spec_eval_10/box_sequence
EVAL_SEQS="random_spec_000 random_spec_001 random_spec_002 random_spec_003 random_spec_004 random_spec_005 random_spec_006 random_spec_007 random_spec_008 random_spec_009"

for item in sun_v1:0 sun_v2:1 sun_v3:1; do
  profile=${item%%:*}
  box_seed=${item##*:}
  run_name=sun_terminal18_env128_workers4_rand10eval_${profile}_deploybest
  run_dir=isaaclab_pallet/runs/$run_name
  log=$run_dir/train.log

  mkdir -p "$run_dir"
  echo "[sun-deploybest] start profile=$profile run=$run_name box_seed=$box_seed best=$BEST eval_dir=$EVAL_DIR eval=$EVAL_SEQS $(date)" | tee -a "$log"

  python3 isaaclab_pallet/scripts/train_pallet_gat.py \
    --run-name "$run_name" \
    --reward-profile "$profile" \
    --num-envs 128 \
    --num-packer-workers 4 \
    --max-boxes 256 \
    --box-seed "$box_seed" \
    --updates 2500 \
    --save-interval 250 \
    --eval-interval 50 \
    --learning-rate 1e-6 \
    --candidate-rerank-k 0 \
    --candidate-diversity-center-m 0.05 \
    --box-seq-dir "$EVAL_DIR" \
    --eval-sequences $EVAL_SEQS \
    --seed 0 \
    --headless \
    --wandb \
    --wandb-project assignment2-pallet \
    --wandb-name "$run_name" \
    --wandb-group sun_terminal18_env128_workers4_rand10eval_deploybest \
    --wandb-tags sun terminal18 env128 workers4 rand10eval deploybest \
    --load-model "$BEST" \
    --deploy-eval-best \
    --deploy-eval-buffer-size 0 >> "$log" 2>&1

  code=$?
  echo "[sun-deploybest] profile=$profile exited code=$code $(date)" | tee -a "$log"
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
done
