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
SUN_V2_RESUME=isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand10eval_sun_v2_deploybest/PCT-resume.pt
EVAL12_DIR=artifacts/random_spec_eval_10_plus_official/box_sequence
EVAL12_SEQS="$(printf 'random_spec_%03d ' $(seq 0 9))box_sequence_0 box_sequence_1"
EVAL52_DIR=artifacts/random_spec_eval_50_plus_official/box_sequence
EVAL52_SEQS="$(printf 'random_spec_%03d ' $(seq 0 49))box_sequence_0 box_sequence_1"

for item in \
  sun_v2:1:sun_terminal18_env128_workers4_rand12to52official_sun_v2_resume_deploybest:1000:resume \
  sun_gap_guard_v1:1:sun_terminal18_env128_workers4_rand12to52official_sun_gap_guard_v1_deploybest:2500:warm \
  sun_anchor_plateau_v1:1:sun_terminal18_env128_workers4_rand12to52official_sun_anchor_plateau_v1_deploybest:2500:warm
do
  profile=${item%%:*}
  rest=${item#*:}
  box_seed=${rest%%:*}
  rest=${rest#*:}
  run_name=${rest%%:*}
  rest=${rest#*:}
  updates=${rest%%:*}
  mode=${rest##*:}
  run_dir=isaaclab_pallet/runs/$run_name
  log=$run_dir/train.log

  mkdir -p "$run_dir"
  model_args=(--load-model "$BEST")
  if [ "$mode" = "resume" ]; then
    model_args=(--resume "$SUN_V2_RESUME")
  fi
  echo "[sun-deploybest] start profile=$profile run=$run_name box_seed=$box_seed updates=$updates mode=$mode best=$BEST resume=$SUN_V2_RESUME eval12_dir=$EVAL12_DIR eval12=$EVAL12_SEQS confirm_eval52_dir=$EVAL52_DIR confirm_eval52=$EVAL52_SEQS $(date)" | tee -a "$log"

  python3 isaaclab_pallet/scripts/train_pallet_gat.py \
    --run-name "$run_name" \
    --reward-profile "$profile" \
    --num-envs 128 \
    --num-packer-workers 4 \
    --max-boxes 256 \
    --box-seed "$box_seed" \
    --updates "$updates" \
    --save-interval 250 \
    --eval-interval 50 \
    --learning-rate 1e-6 \
    --candidate-rerank-k 0 \
    --candidate-diversity-center-m 0.05 \
    --box-seq-dir "$EVAL12_DIR" \
    --eval-sequences $EVAL12_SEQS \
    --confirm-box-seq-dir "$EVAL52_DIR" \
    --confirm-eval-sequences $EVAL52_SEQS \
    --seed 0 \
    --headless \
    --wandb \
    --wandb-project assignment2-pallet \
    --wandb-name "$run_name" \
    --wandb-group sun_terminal18_env128_workers4_rand12to52official_deploybest \
    --wandb-tags sun terminal18 env128 workers4 rand12to52official deploybest \
    "${model_args[@]}" \
    --deploy-eval-best \
    --deploy-eval-buffer-size 0 \
    --deploy-eval-timeout 1800 >> "$log" 2>&1

  code=$?
  echo "[sun-deploybest] profile=$profile exited code=$code $(date)" | tee -a "$log"
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
done
