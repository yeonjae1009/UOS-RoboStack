"""Issue ⑥: standard-RL training via skrl PPO with in-policy action masking.

Replaces the hand-rolled A2C loop (train_pallet_mlp.py) with skrl's PPO. The only
PCT-specific piece is the masked Categorical policy (skrl_models.py), which derives
the invalid-leaf mask from the observation itself.

IMPORTANT: state_preprocessor is left OFF. The observation carries the per-leaf
valid flags (0/1); a RunningStandardScaler would normalize those away and break
mask reconstruction. The PCT observation is already roughly normalized, so raw
states are fine here.

Run (on the Isaac machine):
  python3 isaaclab_pallet/scripts/train_pallet_skrl.py --num-envs 64 --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="skrl PPO training for the Isaac pallet env.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--max-boxes", type=int, default=64)
parser.add_argument("--timesteps", type=int, default=100_000)
parser.add_argument("--rollouts", type=int, default=24)
parser.add_argument("--learning-epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=4)
parser.add_argument("--hidden-dim", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lambda-gae", type=float, default=0.95)
parser.add_argument("--entropy-coef", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--experiment-dir", type=str, default="isaaclab_pallet/runs")
parser.add_argument("--experiment-name", type=str, default="pallet_skrl_ppo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from skrl.agents.torch.ppo import PPO, PPO_CFG  # noqa: E402
from skrl.agents.torch.base import ExperimentCfg  # noqa: E402
from skrl.envs.wrappers.torch import wrap_env  # noqa: E402
from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.trainers.torch.sequential import SequentialTrainerCfg  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

from isaaclab_pallet import PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402
from isaaclab_pallet.skrl_models import MaskedCategoricalPolicy, ValueModel  # noqa: E402


def main() -> None:
    set_seed(args_cli.seed)

    cfg = PalletPackingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.max_boxes = args_cli.max_boxes
    cfg.sim.device = args_cli.device

    env = PalletPackingEnv(cfg)
    internal_nodes = env.internal_node_holder
    leaf_nodes = env.leaf_node_holder
    env = wrap_env(env)  # auto-detects the Isaac Lab wrapper
    device = env.device

    memory = RandomMemory(memory_size=args_cli.rollouts, num_envs=env.num_envs, device=device)

    models = {
        "policy": MaskedCategoricalPolicy(
            env.observation_space, env.action_space, device,
            internal_nodes=internal_nodes, leaf_nodes=leaf_nodes, hidden_dim=args_cli.hidden_dim,
        ),
        "value": ValueModel(env.observation_space, env.action_space, device, hidden_dim=args_cli.hidden_dim),
    }

    agent_cfg = PPO_CFG()
    agent_cfg.rollouts = args_cli.rollouts
    agent_cfg.learning_epochs = args_cli.learning_epochs
    agent_cfg.mini_batches = args_cli.mini_batches
    agent_cfg.discount_factor = args_cli.gamma
    agent_cfg.gae_lambda = args_cli.lambda_gae
    agent_cfg.learning_rate = args_cli.learning_rate
    agent_cfg.entropy_loss_scale = args_cli.entropy_coef
    agent_cfg.grad_norm_clip = 0.5
    # observation/state preprocessors stay None (skrl default) so the raw obs —
    # including the per-leaf valid flags — reaches the policy and the in-policy
    # mask survives. Only the scalar value target is normalized.
    agent_cfg.value_preprocessor = RunningStandardScaler
    agent_cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
    agent_cfg.experiment = ExperimentCfg(
        directory=args_cli.experiment_dir, experiment_name=args_cli.experiment_name
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    trainer_cfg = SequentialTrainerCfg()
    trainer_cfg.timesteps = args_cli.timesteps
    trainer_cfg.headless = True
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    print(
        f"[skrl] num_envs={env.num_envs} obs={env.observation_space} act={env.action_space} "
        f"rollouts={args_cli.rollouts} timesteps={args_cli.timesteps} device={device}",
        flush=True,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
