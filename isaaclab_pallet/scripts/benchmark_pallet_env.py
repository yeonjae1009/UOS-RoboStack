from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Stage 3 throughput benchmark for the pallet packing DirectRLEnv.")
parser.add_argument("--num-envs-list", type=str, default="1,2,4,8")
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--warmup-steps", type=int, default=2)
parser.add_argument("--max-boxes", type=int, default=64)
parser.add_argument("--num-packer-workers", type=int, default=0,
                    help="CPU packer worker processes (0=serial).")
parser.add_argument("--policy", choices=["first", "random"], default="first")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from isaaclab_pallet import PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402


def parse_num_envs(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def select_actions(env: PalletPackingEnv) -> torch.Tensor:
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


def run_once(num_envs: int) -> None:
    cfg = PalletPackingEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.max_boxes = args_cli.max_boxes
    cfg.num_packer_workers = args_cli.num_packer_workers
    cfg.sim.device = args_cli.device

    env = PalletPackingEnv(cfg)
    env.reset()

    for _ in range(args_cli.warmup_steps):
        env.step(select_actions(env))

    if "cuda" in env.device:
        torch.cuda.synchronize()
    start = time.perf_counter()

    total_resets = 0
    for _ in range(args_cli.steps):
        _, _, terminated, truncated, extras = env.step(select_actions(env))
        total_resets += int(torch.count_nonzero(terminated | truncated).item())

    if "cuda" in env.device:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    env_steps_per_sec = args_cli.steps / elapsed
    placements_per_sec = (args_cli.steps * num_envs) / elapsed
    valid_mean = float(extras["action_mask"].sum(dim=1).float().mean().item())
    print(
        "[stage3] "
        f"num_envs={num_envs} "
        f"elapsed={elapsed:.3f}s "
        f"env_steps/s={env_steps_per_sec:.2f} "
        f"placements/s={placements_per_sec:.2f} "
        f"resets={total_resets} "
        f"mean_valid_leafs={valid_mean:.1f}",
        flush=True,
    )
    env.close()


def main() -> None:
    print(
        "[stage3] "
        f"policy={args_cli.policy} steps={args_cli.steps} warmup={args_cli.warmup_steps} "
        f"max_boxes={args_cli.max_boxes}",
        flush=True,
    )
    for num_envs in parse_num_envs(args_cli.num_envs_list):
        run_once(num_envs)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
