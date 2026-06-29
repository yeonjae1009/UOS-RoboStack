#!/usr/bin/env bash
# Option-1 stability retrain: resume from run_warm2's best (seq_1 95.6 fill behaviour,
# but seq_0 goes out-of-bounds) and very gently fine-tune so the PHYSICS reward
# (out-of-bounds/drop/collapse zero the episode) pulls seq_0 back into stability
# WITHOUT wandering off the high-fill behaviour.
#   - lr 5e-7  (half of warm2/3's 1e-6 -> gentler, less drift)
#   - entropy-coef 0  (no exploration noise -> stay near current policy)
# NOTE: the in-loop comp_score is NOT physics-faithful (it ignores out-of-bounds),
# so run_warm4/PCT-best.pt is unreliable. After the run, physics-score the periodic
# PCT-update-*.pt checkpoints with score_warm4 to pick the real best.
#
# Launch from YOUR terminal (background Isaac under the agent hangs - no TTY):
#     nohup bash isaaclab_pallet/scripts/resume_warm4.sh > /tmp/warm4.out 2>&1 &
# Watch:  tail -f /tmp/warm4.out | grep --line-buffered 'gat-train'
# Stop:   touch isaaclab_pallet/runs/run_warm4/STOP   (or kill the process)
cd "$(dirname "$0")/../.."   # -> project root

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

RESUME="isaaclab_pallet/runs/run_warm2/PCT-best-resume.pt"

python3 isaaclab_pallet/scripts/train_pallet_gat.py \
  --run-name run_warm4 \
  --resume "$RESUME" \
  --num-envs 32 \
  --max-boxes 100 \
  --box-seed 0 \
  --updates 3000 \
  --save-interval 250 \
  --eval-interval 50 \
  --learning-rate 5e-7 \
  --entropy-coef 0 \
  --seed 0 \
  --headless
