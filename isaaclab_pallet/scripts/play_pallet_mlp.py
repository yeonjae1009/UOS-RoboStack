from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="GUI/headless rollout for a trained pallet MLP checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--max-boxes", type=int, default=64)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--step-delay", type=float, default=0.5)
parser.add_argument("--hold-seconds", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from isaaclab_pallet import MaskedPalletMLP, PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402


def render_pause(env: PalletPackingEnv, seconds: float) -> None:
    if args_cli.headless or seconds <= 0.0:
        return
    end_time = time.time() + seconds
    while time.time() < end_time:
        env.sim.render()
        time.sleep(1.0 / 30.0)


def main() -> None:
    cfg = PalletPackingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.max_boxes = args_cli.max_boxes
    cfg.sim.device = args_cli.device

    env = PalletPackingEnv(cfg)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    action_mask = env.action_mask

    ckpt = torch.load(args_cli.checkpoint, map_location=env.device)
    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    if obs.shape[-1] != obs_dim:
        raise RuntimeError(f"Checkpoint obs_dim={obs_dim}, but env obs_dim={obs.shape[-1]}")

    policy = MaskedPalletMLP(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(env.device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    print(
        "[play] "
        f"checkpoint={args_cli.checkpoint} obs_shape={tuple(obs.shape)} "
        f"num_envs={env.num_envs} steps={args_cli.steps}",
        flush=True,
    )

    for step in range(args_cli.steps):
        with torch.no_grad():
            action, _, _, _ = policy.act(obs, action_mask, deterministic=True)
        obs_dict, reward, terminated, truncated, extras = env.step(action)
        obs = obs_dict["policy"]
        action_mask = extras["action_mask"]
        valid_counts = action_mask.sum(dim=1).detach().cpu().tolist()
        height_ratio = extras["physics_features"][:, 4].detach().cpu().tolist()
        print(
            "[play] "
            f"step={step} action={action.flatten().detach().cpu().tolist()} "
            f"reward={reward.detach().cpu().tolist()} "
            f"terminated={terminated.detach().cpu().tolist()} "
            f"truncated={truncated.detach().cpu().tolist()} "
            f"drift={env.last_drift.detach().cpu().tolist()} "
            f"tilt={env.last_tilt.detach().cpu().tolist()} "
            f"height_ratio={height_ratio} "
            f"done_reason={env.last_done_reason.detach().cpu().tolist()} "
            f"valid_leafs={valid_counts} "
            f"box_idx={env.current_box_idx}",
            flush=True,
        )
        render_pause(env, args_cli.step_delay)

    if not args_cli.headless and args_cli.hold_seconds > 0.0:
        print(f"[play] holding GUI for {args_cli.hold_seconds:.1f}s", flush=True)
        render_pause(env, args_cli.hold_seconds)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
