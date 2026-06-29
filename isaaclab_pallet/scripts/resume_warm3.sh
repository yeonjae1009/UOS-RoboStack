#!/usr/bin/env bash
# Resume physics (Isaac) GAT training from the best checkpoint (95.89, update 1500)
# of run_warm2 into a FRESH dir run_warm3, so run_warm2/PCT-best.pt (95.89) stays safe.
#   - resume source : run_warm2/PCT-best-resume.pt  (weights + optimizer @ update 1500)
#   - eval-in-loop (★1) keeps run_warm3/PCT-best.pt = best physics score ever seen
# Launch from YOUR terminal (background Isaac under the agent hangs - no TTY):
#     nohup bash isaaclab_pallet/scripts/resume_warm3.sh > /tmp/warm3.out 2>&1 &
# Watch:  tail -f /tmp/warm3.out | grep --line-buffered 'gat-train'
# Stop:   touch isaaclab_pallet/runs/run_warm3/STOP   (or just kill the process)
set -u
cd "$(dirname "$0")/../.."   # -> project root

# Isaac Lab lives in the conda env `env_isaaclab`; nohup shells don't auto-activate it.
# Disable nounset around activation: IsaacLab's setup_conda_env.sh reads $ZSH_VERSION
# (unbound under `set -u`), which would otherwise abort the script.
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

RESUME="isaaclab_pallet/runs/run_warm2/PCT-best-resume.pt"

python3 isaaclab_pallet/scripts/train_pallet_gat.py \
  --run-name run_warm3 \
  --resume "$RESUME" \
  --num-envs 32 \
  --max-boxes 100 \
  --box-seed 0 \
  --updates 5000 \
  --save-interval 250 \
  --eval-interval 50 \
  --learning-rate 1e-6 \
  --seed 0 \
  --headless
