#!/usr/bin/env bash
# Option-3 push: start from run_warm4's PHYSICS-best checkpoint (update-3000 = 92.35,
# both seqs stable) and fine-tune EVEN gentler to squeeze past 92.35 before the
# policy re-diverges (warm4 diverged ~400 updates after its best at lr 5e-7).
#   - load update-3000 weights (bare state_dict -> --load-model, fresh optimizer)
#   - lr 2.5e-7  (half of warm4 -> slower drift, longer stable window)
#   - entropy-coef 0  (no exploration noise)
#   - save-interval 100  (fine granularity to catch the micro-optimum)
# The in-loop comp_score is NOT physics-faithful -> after the run, physics-score the
# PCT-update-*.pt checkpoints with score_warm4.sh (point RUN= at run_warm5).
#
# Launch from YOUR terminal (background Isaac under the agent hangs - no TTY):
#     nohup bash isaaclab_pallet/scripts/resume_warm5.sh > /tmp/warm5.out 2>&1 &
# Stop:  pkill -f train_pallet_gat.py   (no STOP-file hook when run directly)
cd "$(dirname "$0")/../.."   # -> project root

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

LOAD="isaaclab_pallet/runs/run_warm4/PCT-update-003000.pt"

python3 isaaclab_pallet/scripts/train_pallet_gat.py \
  --run-name run_warm5 \
  --load-model "$LOAD" \
  --num-envs 32 \
  --max-boxes 100 \
  --box-seed 0 \
  --updates 1200 \
  --save-interval 100 \
  --eval-interval 25 \
  --learning-rate 2.5e-7 \
  --entropy-coef 0 \
  --seed 0 \
  --headless
