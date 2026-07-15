#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

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

BEST="${BEST:-isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt}"
EVAL12_DIR="${EVAL12_DIR:-artifacts/random_spec_eval_10_plus_official/box_sequence}"
EVAL52_DIR="${EVAL52_DIR:-artifacts/random_spec_eval_50_plus_official/box_sequence}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_PACKER_WORKERS="${NUM_PACKER_WORKERS:-16}"
NUM_STEPS="${NUM_STEPS:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
SETTLE_MAX_STEPS="${SETTLE_MAX_STEPS:-12}"
LEAF_SCORE_TRIAL_LIMIT="${LEAF_SCORE_TRIAL_LIMIT:-32}"
DEPLOY_EVAL_TIMEOUT="${DEPLOY_EVAL_TIMEOUT:-7200}"
DEPLOY_EVAL_WORKERS="${DEPLOY_EVAL_WORKERS:-16}"
WANDB_PROJECT="${WANDB_PROJECT:-assignment2-pallet}"
WANDB_GROUP="${WANDB_GROUP:-sun_terminal18_env64_workers16_fast_leafscore32_rand12to52official_deploybest}"

mapfile -t EVAL12_SEQS < <(printf 'random_spec_%03d\n' $(seq 0 9); printf '%s\n' box_sequence_0 box_sequence_1)
mapfile -t EVAL52_SEQS < <(printf 'random_spec_%03d\n' $(seq 0 49); printf '%s\n' box_sequence_0 box_sequence_1)

if [ "${#EVAL12_SEQS[@]}" -ne 12 ]; then
  echo "[workers8-deploybest] expected 12 primary eval sequences, got ${#EVAL12_SEQS[@]}" >&2
  exit 1
fi
if [ "${#EVAL52_SEQS[@]}" -ne 52 ]; then
  echo "[workers8-deploybest] expected 52 confirm eval sequences, got ${#EVAL52_SEQS[@]}" >&2
  exit 1
fi
for name in "${EVAL12_SEQS[@]}"; do
  test -f "$EVAL12_DIR/$name.json" || { echo "[workers8-deploybest] missing primary eval: $EVAL12_DIR/$name.json" >&2; exit 1; }
done
for name in "${EVAL52_SEQS[@]}"; do
  test -f "$EVAL52_DIR/$name.json" || { echo "[workers8-deploybest] missing confirm eval: $EVAL52_DIR/$name.json" >&2; exit 1; }
done

for item in \
  sun_v2:1:sun_terminal18_env64_workers16_rand12to52official_sun_v2_fast_s12_steps10_leafscore32_deployworkers16:2500 \
  terminal_ratio_t18:1:sun_terminal18_env64_workers16_rand12to52official_terminal_ratio_t18_fast_s12_steps10_leafscore32_deployworkers16:2500
do
  profile=${item%%:*}
  rest=${item#*:}
  box_seed=${rest%%:*}
  rest=${rest#*:}
  run_name=${rest%%:*}
  rest=${rest#*:}
  updates=${rest}
  run_dir=isaaclab_pallet/runs/$run_name
  log=$run_dir/train.log

  mkdir -p "$run_dir"
  model_args=(--load-model "$BEST")
  echo "[workers8-deploybest] start profile=$profile run=$run_name box_seed=$box_seed updates=$updates mode=load_model num_envs=$NUM_ENVS workers=$NUM_PACKER_WORKERS num_steps=$NUM_STEPS settle_max_steps=$SETTLE_MAX_STEPS leaf_score_trial_limit=$LEAF_SCORE_TRIAL_LIMIT deploy_eval_workers=$DEPLOY_EVAL_WORKERS best=$BEST eval12_dir=$EVAL12_DIR eval12=${EVAL12_SEQS[*]} confirm_eval52_dir=$EVAL52_DIR confirm_eval52=${EVAL52_SEQS[*]} $(date)" | tee -a "$log"

  python3 isaaclab_pallet/scripts/train_pallet_gat.py \
    --run-name "$run_name" \
    --reward-profile "$profile" \
    --num-envs "$NUM_ENVS" \
    --num-packer-workers "$NUM_PACKER_WORKERS" \
    --max-boxes 256 \
    --box-seed "$box_seed" \
    --updates "$updates" \
    --num-steps "$NUM_STEPS" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    --eval-interval 50 \
    --learning-rate 1e-6 \
    --settle-max-steps "$SETTLE_MAX_STEPS" \
    --leaf-score-trial-limit "$LEAF_SCORE_TRIAL_LIMIT" \
    --candidate-rerank-k 0 \
    --candidate-diversity-center-m 0.05 \
    --box-seq-dir "$EVAL12_DIR" \
    --eval-sequences "${EVAL12_SEQS[@]}" \
    --confirm-box-seq-dir "$EVAL52_DIR" \
    --confirm-eval-sequences "${EVAL52_SEQS[@]}" \
    --seed 0 \
    --headless \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "$run_name" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-tags sun terminal18 env64 workers16 rand12to52official deploybest fast_s12_steps10 leafscore32 "$profile" \
    "${model_args[@]}" \
    --deploy-eval-best \
    --skip-torch-eval-with-deploy-best \
    --deploy-eval-buffer-size 0 \
    --deploy-eval-workers "$DEPLOY_EVAL_WORKERS" \
    --deploy-eval-timeout "$DEPLOY_EVAL_TIMEOUT" >> "$log" 2>&1

  code=$?
  echo "[workers8-deploybest] profile=$profile exited code=$code $(date)" | tee -a "$log"
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
done
