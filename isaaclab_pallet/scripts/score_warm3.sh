#!/usr/bin/env bash
# Independent PHYSICS scoring of run_warm3/PCT-best.pt (95.89 in-loop) on the
# competition sequences box_sequence_0 / box_sequence_1, using the OFFICIAL
# palletizing_simulator (Isaac physics) + evaluator.
#
# Inputs (already generated, no Isaac):  /tmp/warm3_results/box_sequence_{0,1}.json
# Outputs (kept separate from the 92.3 baseline in sim_results/):
#   /tmp/warm3_sim_out/result.json
#   /tmp/warm3_sim_out/box_sequence_0_result.png
#   /tmp/warm3_sim_out/box_sequence_1_result.png
#
# Launch from YOUR terminal (background Isaac under the agent hangs - no TTY):
#     bash isaaclab_pallet/scripts/score_warm3.sh 2>&1 | tee /tmp/warm3_score.out
cd "$(dirname "$0")/../.."   # -> project root

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
set -u

python3 palletizing_simulator/simulator.py \
  --config palletizing_simulator/config/sim_config.yaml \
  --input-dir /tmp/warm3_results \
  -o /tmp/warm3_sim_out

echo "=== result.json ==="
cat /tmp/warm3_sim_out/result.json
