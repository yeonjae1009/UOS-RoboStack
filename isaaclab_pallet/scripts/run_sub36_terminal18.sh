#!/usr/bin/env bash
# Fine-tune terminal_ratio_t18 with the sub36 support-aware extreme-point candidate generator.
#
# Launch:
#   nohup bash isaaclab_pallet/scripts/run_sub36_terminal18.sh > /tmp/sub36_terminal18.out 2>&1 &
#
# Watch:
#   tail -f /tmp/sub36_terminal18.out
#   tail -f isaaclab_pallet/runs/sub36_terminal18_random50/terminal_ratio_t18_sub36_from_terminal18_best/train.log

set -u
cd "$(dirname "$0")/../.."

NV_LIB_DIRS="$(python3 -c 'import site,glob,os
paths=[]
for sp in (site.getsitepackages() if hasattr(site,"getsitepackages") else []):
    paths += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
print(os.pathsep.join(paths))' 2>/dev/null)"
if [ -n "$NV_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

OUT_DIR="${OUT_DIR:-isaaclab_pallet/runs/sub36_terminal18_random50}"
RUN_NAME="${RUN_NAME:-terminal_ratio_t18_sub36_from_terminal18_best}"
PCT_CONFIG_PATH="${PCT_CONFIG_PATH:-templete code/config/pct_config_sub36.yaml}"
WARM_START="${WARM_START:-isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt}"
RESUME="${RESUME:-}"

NUM_ENVS="${NUM_ENVS:-32}"
NUM_PACKER_WORKERS="${NUM_PACKER_WORKERS:-8}"
MAX_BOXES="${MAX_BOXES:-256}"
UPDATES="${UPDATES:-2500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
SAVE_UPDATE_CHECKPOINTS="${SAVE_UPDATE_CHECKPOINTS:-1}"
LR="${LR:-5e-7}"
BOX_SEED="${BOX_SEED:-0}"
CANDIDATE_RERANK_K="${CANDIDATE_RERANK_K:-0}"
CANDIDATE_DIVERSITY_CENTER_M="${CANDIDATE_DIVERSITY_CENTER_M:-0.05}"
EVAL_BOX_SEQ_DIR="${EVAL_BOX_SEQ_DIR:-artifacts/random_spec_eval_50/box_sequence}"
EVAL_SEQUENCE_NAMES="${EVAL_SEQUENCE_NAMES:-$(printf 'random_spec_%03d ' $(seq 0 49))}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-assignment2-pallet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-sub36_terminal18_random50}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_TAGS="${WANDB_TAGS:-sub36 random50 terminal_ratio_t18}"

mkdir -p "$OUT_DIR/$RUN_NAME"
log="${OUT_DIR}/${RUN_NAME}/train.log"
eval_sequences_array=($EVAL_SEQUENCE_NAMES)

if [ ! -s "$WARM_START" ]; then
  echo "[sub36-terminal18] missing WARM_START=$WARM_START" | tee -a "$log"
  exit 1
fi
if [ ! -s "$PCT_CONFIG_PATH" ]; then
  echo "[sub36-terminal18] missing PCT_CONFIG_PATH=$PCT_CONFIG_PATH" | tee -a "$log"
  exit 1
fi

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

checkpoint_args=()
if [ "$SAVE_UPDATE_CHECKPOINTS" = "0" ]; then
  checkpoint_args=(--no-save-update-checkpoints)
fi

resume_args=()
if [ -n "$RESUME" ]; then
  resume_args=(--resume "$RESUME")
fi

echo "[sub36-terminal18] start $(date)" | tee -a "$log"
echo "[sub36-terminal18] out_dir=$OUT_DIR run_name=$RUN_NAME pct_config=$PCT_CONFIG_PATH warm_start=$WARM_START" | tee -a "$log"
echo "[sub36-terminal18] num_envs=$NUM_ENVS packer_workers=$NUM_PACKER_WORKERS max_boxes=$MAX_BOXES updates=$UPDATES lr=$LR candidate_k=$CANDIDATE_RERANK_K" | tee -a "$log"
echo "[sub36-terminal18] eval_box_seq_dir=$EVAL_BOX_SEQ_DIR eval_count=${#eval_sequences_array[@]} eval_sequences=${eval_sequences_array[*]}" | tee -a "$log"

start_line="$(wc -l < "$log" 2>/dev/null || echo 0)"
python3 isaaclab_pallet/scripts/train_pallet_gat.py \
  --pct-config-path "$PCT_CONFIG_PATH" \
  --output-dir "$OUT_DIR" \
  --run-name "$RUN_NAME" \
  --reward-profile terminal_ratio_t18 \
  --num-envs "$NUM_ENVS" \
  --num-packer-workers "$NUM_PACKER_WORKERS" \
  --max-boxes "$MAX_BOXES" \
  --box-seed "$BOX_SEED" \
  --updates "$UPDATES" \
  --save-interval "$SAVE_INTERVAL" \
  --eval-interval "$EVAL_INTERVAL" \
  --box-seq-dir "$EVAL_BOX_SEQ_DIR" \
  --eval-sequences "${eval_sequences_array[@]}" \
  --learning-rate "$LR" \
  --candidate-rerank-k "$CANDIDATE_RERANK_K" \
  --candidate-diversity-center-m "$CANDIDATE_DIVERSITY_CENTER_M" \
  --seed 0 \
  --headless \
  "${checkpoint_args[@]}" \
  "${resume_args[@]}" \
  "${wandb_args[@]}" \
  --load-model "$WARM_START" >> "$log" 2>&1
code=$?

recent_log="$(mktemp)"
tail -n +"$((start_line + 1))" "$log" > "$recent_log"
if grep -Eq "Traceback \\(most recent call last\\)|ModuleNotFoundError|ImportError|Failed to startup python extension" "$recent_log"; then
  code=1
fi
rm -f "$recent_log"

echo "[sub36-terminal18] exited code=$code $(date)" | tee -a "$log"
exit "$code"
