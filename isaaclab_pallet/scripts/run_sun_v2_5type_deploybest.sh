#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

NV_LIB_DIRS="$(python3 -c 'import glob, os, site
paths = []
for sp in site.getsitepackages() if hasattr(site, "getsitepackages") else []:
    paths.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(os.pathsep.join(paths))' 2>/dev/null)"
if [ -n "$NV_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="$NV_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUT_DIR="${OUT_DIR:-isaaclab_pallet/runs/sun_v2_5type_uniform}"
RUN_NAME="${RUN_NAME:-sun_v2_env128_workers8_5type_uniform_ems_deploybest}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"
LOG="${RUN_DIR}/train.log"

WARM_START="${WARM_START:-isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_v2_resume_deploybest/PCT-best.pt}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_PACKER_WORKERS="${NUM_PACKER_WORKERS:-8}"
MAX_BOXES="${MAX_BOXES:-256}"
DEVICE="${DEVICE:-cuda:0}"
UPDATES="${UPDATES:-2500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BOX_SEED="${BOX_SEED:-0}"
CANDIDATE_RERANK_K="${CANDIDATE_RERANK_K:-0}"
CANDIDATE_DIVERSITY_CENTER_M="${CANDIDATE_DIVERSITY_CENTER_M:-0.05}"
DEPLOY_EVAL_BUFFER_SIZE="${DEPLOY_EVAL_BUFFER_SIZE:-0}"
DEPLOY_EVAL_TIMEOUT="${DEPLOY_EVAL_TIMEOUT:-1800}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-assignment2-pallet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-sun_v2_5type_uniform_deploybest}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_TAGS="${WANDB_TAGS:-sun sun_v2 5type uniform ems deploybest}"

TRAIN_BOX_SEQ_DIR="${TRAIN_BOX_SEQ_DIR:-submit_buffer3_search/box_sequence}"
TRAIN_TYPE_SEQUENCES="${TRAIN_TYPE_SEQUENCES:-box_sequence_0 box_sequence_1}"
EVAL_BOX_SEQ_DIR="${EVAL_BOX_SEQ_DIR:-submit_buffer3_search/box_sequence}"
EVAL_SEQUENCE_NAMES="${EVAL_SEQUENCE_NAMES:-box_sequence_0 box_sequence_1}"

mkdir -p "$RUN_DIR"
if [ ! -s "$WARM_START" ]; then
  echo "[sun-v2-5type] missing warm_start=$WARM_START" | tee -a "$LOG"
  exit 1
fi

# shellcheck disable=SC2206
train_type_sequences_array=($TRAIN_TYPE_SEQUENCES)
# shellcheck disable=SC2206
eval_sequences_array=($EVAL_SEQUENCE_NAMES)

wandb_args=()
if [ "$WANDB_ENABLE" = "1" ]; then
  wandb_args=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-name "$RUN_NAME" --wandb-group "$WANDB_GROUP")
  if [ -n "$WANDB_ENTITY" ]; then
    wandb_args+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [ -n "$WANDB_MODE" ]; then
    wandb_args+=(--wandb-mode "$WANDB_MODE")
  fi
  if [ -n "$WANDB_DIR" ]; then
    wandb_args+=(--wandb-dir "$WANDB_DIR")
  fi
  if [ -n "$WANDB_TAGS" ]; then
    # shellcheck disable=SC2206
    wandb_tags_array=($WANDB_TAGS)
    wandb_args+=(--wandb-tags "${wandb_tags_array[@]}")
  fi
fi

echo "[sun-v2-5type] start run=$RUN_NAME device=$DEVICE num_envs=$NUM_ENVS workers=$NUM_PACKER_WORKERS max_boxes=$MAX_BOXES updates=$UPDATES box_seed=$BOX_SEED warm_start=$WARM_START $(date)" | tee -a "$LOG"
echo "[sun-v2-5type] train_box_seq_dir=$TRAIN_BOX_SEQ_DIR train_type_sequences=${train_type_sequences_array[*]} eval_box_seq_dir=$EVAL_BOX_SEQ_DIR eval_sequences=${eval_sequences_array[*]} deploy_buffer=$DEPLOY_EVAL_BUFFER_SIZE" | tee -a "$LOG"

python3 isaaclab_pallet/scripts/train_pallet_gat.py \
  --output-dir "$OUT_DIR" \
  --run-name "$RUN_NAME" \
  --reward-profile sun_v2 \
  --box-source fixed_type_random \
  --train-box-seq-dir "$TRAIN_BOX_SEQ_DIR" \
  --train-type-sequences "${train_type_sequences_array[@]}" \
  --candidate-generator ems \
  --num-envs "$NUM_ENVS" \
  --num-packer-workers "$NUM_PACKER_WORKERS" \
  --max-boxes "$MAX_BOXES" \
  --device "$DEVICE" \
  --box-seed "$BOX_SEED" \
  --updates "$UPDATES" \
  --save-interval "$SAVE_INTERVAL" \
  --eval-interval "$EVAL_INTERVAL" \
  --learning-rate "$LEARNING_RATE" \
  --candidate-rerank-k "$CANDIDATE_RERANK_K" \
  --candidate-diversity-center-m "$CANDIDATE_DIVERSITY_CENTER_M" \
  --box-seq-dir "$EVAL_BOX_SEQ_DIR" \
  --eval-sequences "${eval_sequences_array[@]}" \
  --seed 0 \
  --headless \
  "${wandb_args[@]}" \
  --load-model "$WARM_START" \
  --deploy-eval-best \
  --deploy-eval-buffer-size "$DEPLOY_EVAL_BUFFER_SIZE" \
  --deploy-eval-timeout "$DEPLOY_EVAL_TIMEOUT" >> "$LOG" 2>&1
code=$?

echo "[sun-v2-5type] exited code=$code $(date)" | tee -a "$LOG"
exit "$code"
