#!/usr/bin/env bash
# Fine-tune the two swapped terminal-ratio reward variants in one isolated folder.
#
# Launch:
#   nohup bash isaaclab_pallet/scripts/run_terminal18_finish15.sh > /tmp/terminal18_finish15.out 2>&1 &
#
# Watch:
#   tail -f /tmp/terminal18_finish15.out
#   bash isaaclab_pallet/scripts/watch_training.sh

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

OUT_DIR="${OUT_DIR:-isaaclab_pallet/runs/reward_terminal18_finish15}"
STOP_ALL="${OUT_DIR}/STOP"

NUM_ENVS="${NUM_ENVS:-32}"
NUM_PACKER_WORKERS="${NUM_PACKER_WORKERS:-8}"
MAX_BOXES="${MAX_BOXES:-256}"
UPDATES_PER_PROFILE="${UPDATES_PER_PROFILE:-2500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
SAVE_UPDATE_CHECKPOINTS="${SAVE_UPDATE_CHECKPOINTS:-1}"
LR="${LR:-1e-6}"
BOX_SEED="${BOX_SEED:-0}"
CANDIDATE_RERANK_K="${CANDIDATE_RERANK_K:-0}"
CANDIDATE_DIVERSITY_CENTER_M="${CANDIDATE_DIVERSITY_CENTER_M:-0.05}"
EVAL_BOX_SEQ_DIR="${EVAL_BOX_SEQ_DIR:-artifacts/random_spec_eval_50/box_sequence}"
EVAL_SEQUENCE_NAMES="${EVAL_SEQUENCE_NAMES:-$(printf 'random_spec_%03d ' $(seq 0 49))}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-assignment2-pallet}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-reward_terminal18_finish15}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

mkdir -p "$OUT_DIR"
eval_sequences_array=($EVAL_SEQUENCE_NAMES)

profiles=(terminal_ratio_t18 finish_ratio_t15)
run_names=(terminal_ratio_t18_from_terminal_best finish_ratio_t15_from_finish_best)
warm_starts=(
  "${OUT_DIR}/previous/reward_sweep_small_terminal_ratio/PCT-best.pt"
  "${OUT_DIR}/previous/reward_sweep_small_finish_ratio/PCT-best.pt"
)

echo "[terminal18-finish15] start $(date)"
echo "[terminal18-finish15] out_dir=$OUT_DIR num_envs=$NUM_ENVS packer_workers=$NUM_PACKER_WORKERS max_boxes=$MAX_BOXES updates/profile=$UPDATES_PER_PROFILE lr=$LR candidate_k=$CANDIDATE_RERANK_K"
echo "[terminal18-finish15] eval_box_seq_dir=$EVAL_BOX_SEQ_DIR eval_sequences=${eval_sequences_array[*]}"

for i in "${!profiles[@]}"; do
  [ -f "$STOP_ALL" ] && break

  profile="${profiles[$i]}"
  run_name="${run_names[$i]}"
  warm_start="${warm_starts[$i]}"
  run_dir="${OUT_DIR}/${run_name}"
  log="${run_dir}/train.log"
  seed="$((BOX_SEED + i))"

  mkdir -p "$run_dir"
  if [ ! -s "$warm_start" ]; then
    echo "[terminal18-finish15] missing warm_start=$warm_start" | tee -a "$log"
    exit 1
  fi

  wandb_args=()
  if [ "$WANDB_ENABLE" = "1" ]; then
    wandb_args=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-name "$run_name" --wandb-group "$WANDB_GROUP")
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

  echo "[terminal18-finish15] warm-start profile=$profile run=$run_name seed=$seed warm_start=$warm_start $(date)" | tee -a "$log"
  start_line="$(wc -l < "$log" 2>/dev/null || echo 0)"
  python3 isaaclab_pallet/scripts/train_pallet_gat.py \
    --output-dir "$OUT_DIR" \
    --run-name "$run_name" \
    --reward-profile "$profile" \
    --num-envs "$NUM_ENVS" \
    --num-packer-workers "$NUM_PACKER_WORKERS" \
    --max-boxes "$MAX_BOXES" \
    --box-seed "$seed" \
    --updates "$UPDATES_PER_PROFILE" \
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
    "${wandb_args[@]}" \
    --load-model "$warm_start" >> "$log" 2>&1
  code=$?

  recent_log="$(mktemp)"
  tail -n +"$((start_line + 1))" "$log" > "$recent_log"
  if grep -Eq "Traceback \\(most recent call last\\)|ModuleNotFoundError|ImportError|Failed to startup python extension" "$recent_log"; then
    code=1
  fi
  rm -f "$recent_log"

  echo "[terminal18-finish15] profile=$profile exited code=$code $(date)" | tee -a "$log"
  if [ "$code" -ne 0 ]; then
    echo "[terminal18-finish15] stopping after failed profile=$profile; inspect $log"
    exit "$code"
  fi
done

echo "[terminal18-finish15] done $(date)"
