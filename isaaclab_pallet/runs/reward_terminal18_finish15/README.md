# reward_terminal18_finish15

This folder is for the swapped terminal-ratio reward experiment.

## New profiles

- `terminal_ratio_t18`
  - `floor_coverage_reward_scale=1.2`
  - `boundary_floor_reward_scale=0.8`
  - `corner_floor_reward_scale=0.6`
  - `height_smoothness_reward_scale=0.8`
  - `support_reward_scale=0.08`
  - `weak_support_penalty_scale=0.08`
  - `elevation_penalty_scale=0.2`
  - `terminal_ratio_reward_scale=18.0`
  - `learn_finish_action=False`

- `finish_ratio_t15`
  - `floor_coverage_reward_scale=1.2`
  - `boundary_floor_reward_scale=0.8`
  - `corner_floor_reward_scale=0.6`
  - `height_smoothness_reward_scale=0.8`
  - `support_reward_scale=0.08`
  - `weak_support_penalty_scale=0.08`
  - `elevation_penalty_scale=0.2`
  - `terminal_ratio_reward_scale=15.0`
  - `learn_finish_action=True`

## Warm starts

Use model-only warm starts with a fresh optimizer:

- `terminal_ratio_t18`: `previous/reward_sweep_small_terminal_ratio/PCT-best.pt`
- `finish_ratio_t15`: `previous/reward_sweep_small_finish_ratio/PCT-best.pt`

This is intentional. The reward scales changed, so carrying over the previous optimizer state via `--resume` would mix in optimizer momentum from the old objective.

## Launch

```bash
nohup bash isaaclab_pallet/scripts/run_terminal18_finish15.sh > /tmp/terminal18_finish15.out 2>&1 &
```

Default training settings in the script:

- `NUM_ENVS=32`
- `MAX_BOXES=256`
- `UPDATES_PER_PROFILE=2500`
- `LR=1e-6`
- `CANDIDATE_RERANK_K=0`

Best-checkpoint evaluation uses the 10 fixed random spec datasets in:

- `artifacts/random_spec_eval_10/box_sequence/random_spec_000.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_001.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_002.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_003.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_004.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_005.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_006.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_007.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_008.json`
- `artifacts/random_spec_eval_10/box_sequence/random_spec_009.json`
