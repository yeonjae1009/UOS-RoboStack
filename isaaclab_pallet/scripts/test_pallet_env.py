from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke test for the Stage 2 pallet packing DirectRLEnv.")
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--max-boxes", type=int, default=8)
parser.add_argument("--action", type=int, default=0)
parser.add_argument("--policy", choices=["fixed", "first", "random"], default="fixed")
parser.add_argument("--step-delay", type=float, default=0.5)
parser.add_argument("--hold-seconds", type=float, default=8.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from isaaclab_pallet import PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402


def render_pause(env: PalletPackingEnv, seconds: float) -> None:
    if args_cli.headless or seconds <= 0.0:
        return
    end_time = time.time() + seconds
    while time.time() < end_time:
        env.sim.render()
        time.sleep(1.0 / 30.0)


def select_actions(env: PalletPackingEnv) -> torch.Tensor:
    if args_cli.policy == "fixed":
        return torch.full((env.num_envs, 1), args_cli.action, dtype=torch.long, device=env.device)

    mask = env.action_mask
    actions = torch.zeros((env.num_envs, 1), dtype=torch.long, device=env.device)
    for env_id in range(env.num_envs):
        valid = torch.nonzero(mask[env_id], as_tuple=False).flatten()
        if len(valid) == 0:
            continue
        if args_cli.policy == "random":
            choice = int(torch.randint(len(valid), (1,), device=env.device).item())
            actions[env_id, 0] = valid[choice]
        else:
            actions[env_id, 0] = valid[0]
    return actions


def main() -> None:
    if not args_cli.headless and args_cli.steps >= args_cli.max_boxes:
        print(
            "[stage2] note: steps >= max-boxes will trigger Isaac Lab auto-reset on the last placement. "
            "Use --max-boxes larger than --steps when you want to inspect the placed boxes.",
            flush=True,
        )

    cfg = PalletPackingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.max_boxes = args_cli.max_boxes
    cfg.sim.device = args_cli.device

    print("[stage2] creating env", flush=True)
    env = PalletPackingEnv(cfg)
    print("[stage2] env created", flush=True)
    obs, _ = env.reset()
    print(f"[stage2] reset obs shape: {tuple(obs['policy'].shape)}", flush=True)

    for step in range(args_cli.steps):
        actions = select_actions(env)
        obs, rew, terminated, truncated, extras = env.step(actions)
        valid_counts = extras["action_mask"].sum(dim=1).detach().cpu().tolist()
        height_ratio = extras["physics_features"][:, 4].detach().cpu().tolist()
        terminal_reason = extras["terminal_done_reason"].detach().cpu().tolist()
        terminal_drift = extras["terminal_drift"].detach().cpu().tolist()
        print(
            "[stage2] "
            f"step={step} action={actions.flatten().detach().cpu().tolist()} "
            f"reward={rew.detach().cpu().tolist()} "
            f"terminated={terminated.detach().cpu().tolist()} "
            f"truncated={truncated.detach().cpu().tolist()} "
            f"drift={env.last_drift.detach().cpu().tolist()} "
            f"tilt={env.last_tilt.detach().cpu().tolist()} "
            f"height_ratio={height_ratio} "
            f"done_reason={env.last_done_reason.detach().cpu().tolist()} "
            f"terminal_reason={terminal_reason} "
            f"terminal_drift={terminal_drift} "
            f"valid_leafs={valid_counts} "
            f"box_idx={env.current_box_idx}",
            flush=True,
        )
        render_pause(env, args_cli.step_delay)
        if bool(torch.any(terminated | truncated)):
            break

    if not args_cli.headless and args_cli.hold_seconds > 0.0:
        print(f"[stage2] holding GUI for {args_cli.hold_seconds:.1f}s", flush=True)
        render_pause(env, args_cli.hold_seconds)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
