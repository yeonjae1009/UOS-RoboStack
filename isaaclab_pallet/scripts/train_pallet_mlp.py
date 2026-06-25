from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Headless PyTorch actor-critic training for the Isaac pallet env.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--max-boxes", type=int, default=64)
parser.add_argument("--updates", type=int, default=100)
parser.add_argument("--num-steps", type=int, default=8)
parser.add_argument("--hidden-dim", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--entropy-coef", type=float, default=0.01)
parser.add_argument("--value-loss-coef", type=float, default=0.5)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--log-interval", type=int, default=1)
parser.add_argument("--save-interval", type=int, default=10)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--output-dir", type=str, default="isaaclab_pallet/runs")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--seed", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from isaaclab_pallet import MaskedPalletMLP, PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402


def make_run_dir() -> Path:
    run_name = args_cli.run_name
    if not run_name:
        run_name = "mlp-" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_dir = Path(args_cli.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    path: Path,
    policy: MaskedPalletMLP,
    optimizer: torch.optim.Optimizer,
    update: int,
    obs_dim: int,
    action_dim: int,
) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": policy.hidden_dim,
            "policy_type": "MaskedPalletMLP",
        },
        path,
    )


def load_checkpoint(
    path: str,
    policy: MaskedPalletMLP,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("update", 0))


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(args_cli.seed)
    if "cuda" in args_cli.device:
        torch.cuda.manual_seed_all(args_cli.seed)

    cfg = PalletPackingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.max_boxes = args_cli.max_boxes
    cfg.sim.device = args_cli.device

    env = PalletPackingEnv(cfg)
    obs_dict, _ = env.reset(seed=args_cli.seed)
    obs = obs_dict["policy"]
    action_mask = env.action_mask

    obs_dim = obs.shape[-1]
    action_dim = env.leaf_node_holder
    policy = MaskedPalletMLP(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=args_cli.hidden_dim).to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)

    start_update = 0
    if args_cli.resume:
        start_update = load_checkpoint(args_cli.resume, policy, optimizer, env.device)
        print(f"[train] resumed from {args_cli.resume} at update={start_update}", flush=True)

    run_dir = make_run_dir()
    print(
        "[train] "
        f"run_dir={run_dir} device={env.device} num_envs={env.num_envs} "
        f"obs_dim={obs_dim} action_dim={action_dim} num_steps={args_cli.num_steps}",
        flush=True,
    )

    episode_returns = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    recent_returns: deque[float] = deque(maxlen=50)
    recent_lengths: deque[float] = deque(maxlen=50)
    train_start = time.perf_counter()

    for update in range(start_update + 1, start_update + args_cli.updates + 1):
        obs_buf = []
        action_buf = []
        log_prob_buf = []
        reward_buf = []
        mask_buf = []
        value_buf = []
        entropy_buf = []

        for _ in range(args_cli.num_steps):
            action, log_prob, entropy, value = policy.act(obs, action_mask)
            next_obs_dict, reward, terminated, truncated, extras = env.step(action)
            done = terminated | truncated

            obs_buf.append(obs)
            action_buf.append(action)
            log_prob_buf.append(log_prob)
            reward_buf.append(reward.unsqueeze(-1))
            mask_buf.append((~done).float().unsqueeze(-1))
            value_buf.append(value)
            entropy_buf.append(entropy)

            episode_returns += reward.detach()
            episode_lengths += 1
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            for env_id in done_ids.detach().cpu().tolist():
                recent_returns.append(float(episode_returns[env_id].item()))
                recent_lengths.append(float(episode_lengths[env_id].item()))
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0.0

            obs = next_obs_dict["policy"]
            action_mask = extras["action_mask"]

        with torch.no_grad():
            _, next_value = policy(obs, action_mask)

        returns = []
        running_return = next_value.detach()
        for step in reversed(range(args_cli.num_steps)):
            running_return = reward_buf[step] + args_cli.gamma * running_return * mask_buf[step]
            returns.insert(0, running_return)

        returns_t = torch.stack(returns)
        values_t = torch.stack(value_buf)
        log_probs_t = torch.stack(log_prob_buf)
        entropy_t = torch.stack(entropy_buf)

        advantages = returns_t - values_t
        actor_loss = -(advantages.detach() * log_probs_t).mean()
        value_loss = advantages.pow(2).mean()
        entropy = entropy_t.mean()
        loss = actor_loss + args_cli.value_loss_coef * value_loss - args_cli.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args_cli.max_grad_norm)
        optimizer.step()

        if update % args_cli.log_interval == 0:
            elapsed = max(time.perf_counter() - train_start, 1e-6)
            samples = (update - start_update) * args_cli.num_steps * env.num_envs
            fps = samples / elapsed
            mean_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
            mean_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
            print(
                "[train] "
                f"update={update} samples={samples} fps={fps:.1f} "
                f"loss={loss.item():.4f} actor={actor_loss.item():.4f} "
                f"value={value_loss.item():.4f} entropy={entropy.item():.4f} "
                f"mean_return={mean_return:.3f} mean_len={mean_length:.1f}",
                flush=True,
            )

        if update % args_cli.save_interval == 0 or update == start_update + args_cli.updates:
            save_checkpoint(run_dir / "pallet_mlp_latest.pt", policy, optimizer, update, obs_dim, action_dim)
            save_checkpoint(
                run_dir / f"pallet_mlp_update_{update:06d}.pt", policy, optimizer, update, obs_dim, action_dim
            )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
