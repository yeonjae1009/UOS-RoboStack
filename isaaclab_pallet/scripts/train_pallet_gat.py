from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Headless Isaac Lab training using the original Online-3D-BPP-PCT GAT.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--max-boxes", type=int, default=64)
parser.add_argument("--updates", type=int, default=100)
parser.add_argument("--num-steps", type=int, default=5)
parser.add_argument("--learning-rate", type=float, default=1e-6)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--actor-loss-coef", type=float, default=1.0)
parser.add_argument("--critic-loss-coef", type=float, default=1.0)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--embedding-size", type=int, default=64)
parser.add_argument("--hidden-size", type=int, default=128)
parser.add_argument("--gat-layer-num", type=int, default=1)
parser.add_argument("--log-interval", type=int, default=1)
parser.add_argument("--save-interval", type=int, default=10)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--output-dir", type=str, default="isaaclab_pallet/runs")
parser.add_argument("--load-model", type=str, default="")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--drift-fail-threshold", type=float, default=0.40)
parser.add_argument("--tilt-fail-threshold", type=float, default=0.35)
parser.add_argument("--out-of-bounds-margin", type=float, default=0.02)
parser.add_argument("--height-fail-margin", type=float, default=0.005)
parser.add_argument("--drop-fail-threshold", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_pallet import PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONLINE_PCT_DIR = PROJECT_ROOT / "Online-3D-BPP-PCT"
sys.path.insert(0, str(ONLINE_PCT_DIR))

import tools as pct_tools  # noqa: E402
from model import DRL_GAT  # noqa: E402
from storage import PCTRolloutStorage  # noqa: E402


def make_run_dir() -> Path:
    run_name = args_cli.run_name
    if not run_name:
        run_name = "gat-" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_dir = Path(args_cli.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_pct_args(env: PalletPackingEnv) -> SimpleNamespace:
    setting = int(env.pct_setting)
    internal_node_length = 7 if setting == 3 else 6
    norm_factor = float(env.pct_cfg.get("norm_factor", 1.0 / max(env.cfg.pallet_size)))
    return SimpleNamespace(
        setting=setting,
        internal_node_holder=env.internal_node_holder,
        internal_node_length=internal_node_length,
        leaf_node_holder=env.leaf_node_holder,
        embedding_size=args_cli.embedding_size,
        hidden_size=args_cli.hidden_size,
        gat_layer_num=args_cli.gat_layer_num,
        normFactor=norm_factor,
    )


def save_checkpoint(
    path: Path,
    policy: DRL_GAT,
    optimizer: torch.optim.Optimizer,
    update: int,
    pct_args: SimpleNamespace,
) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "policy_type": "Online3DBPP_DRL_GAT",
            "internal_node_holder": pct_args.internal_node_holder,
            "internal_node_length": pct_args.internal_node_length,
            "leaf_node_holder": pct_args.leaf_node_holder,
            "embedding_size": pct_args.embedding_size,
            "hidden_size": pct_args.hidden_size,
            "gat_layer_num": pct_args.gat_layer_num,
            "normFactor": pct_args.normFactor,
        },
        path,
    )


def load_resume(path: str, policy: DRL_GAT, optimizer: torch.optim.Optimizer, device: str) -> int:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    policy.load_state_dict(state_dict)
    if isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("update", 0)) if isinstance(ckpt, dict) else 0


def main() -> None:
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    if "cuda" in args_cli.device:
        torch.cuda.manual_seed_all(args_cli.seed)

    cfg = PalletPackingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.max_boxes = args_cli.max_boxes
    cfg.sim.device = args_cli.device
    cfg.drift_fail_threshold = args_cli.drift_fail_threshold
    cfg.tilt_fail_threshold = args_cli.tilt_fail_threshold
    cfg.out_of_bounds_margin = args_cli.out_of_bounds_margin
    cfg.height_fail_margin = args_cli.height_fail_margin
    cfg.drop_fail_threshold = args_cli.drop_fail_threshold

    env = PalletPackingEnv(cfg)
    obs_dict, _ = env.reset(seed=args_cli.seed)
    pct_obs = env.extras["pct_obs"]
    all_nodes, leaf_nodes = pct_tools.get_leaf_nodes(pct_obs, env.internal_node_holder, env.leaf_node_holder)
    all_nodes = all_nodes.to(env.device)
    leaf_nodes = leaf_nodes.to(env.device)

    pct_args = make_pct_args(env)
    policy = DRL_GAT(pct_args).to(env.device)
    if args_cli.load_model:
        policy = pct_tools.load_policy(args_cli.load_model, policy).to(env.device)
        print(f"[gat-train] loaded original PCT weights: {args_cli.load_model}", flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)

    start_update = 0
    if args_cli.resume:
        start_update = load_resume(args_cli.resume, policy, optimizer, env.device)
        print(f"[gat-train] resumed from {args_cli.resume} at update={start_update}", flush=True)

    run_dir = make_run_dir()
    storage = PCTRolloutStorage(
        args_cli.num_steps,
        env.num_envs,
        obs_shape=all_nodes.shape[1:],
        gamma=args_cli.gamma,
    )
    storage.to(env.device)
    storage.obs[0].copy_(all_nodes)

    batch_indices = torch.arange(env.num_envs, device=env.device)
    episode_returns = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    recent_returns: deque[float] = deque(maxlen=50)
    recent_lengths: deque[float] = deque(maxlen=50)
    train_start = time.perf_counter()

    print(
        "[gat-train] "
        f"run_dir={run_dir} device={env.device} num_envs={env.num_envs} "
        f"obs_shape={tuple(all_nodes.shape)} leaf_shape={tuple(leaf_nodes.shape)} "
        f"num_steps={args_cli.num_steps} normFactor={pct_args.normFactor} "
        f"drift_fail_threshold={cfg.drift_fail_threshold}",
        flush=True,
    )

    for update in range(start_update + 1, start_update + args_cli.updates + 1):
        policy.train()
        storage.step = 0

        for _ in range(args_cli.num_steps):
            with torch.no_grad():
                selected_log_prob, selected_idx, dist_entropy, _ = policy(all_nodes, normFactor=pct_args.normFactor)

            # Original Online-3D-BPP selected the leaf node here:
            # selected_leaf_node = leaf_nodes[batch_indices, selected_idx.squeeze()]
            # Isaac Lab env keeps the same index semantics, so env.step receives selected_idx directly.
            _selected_leaf_node = leaf_nodes[batch_indices, selected_idx.squeeze()]
            obs_dict, reward, terminated, truncated, extras = env.step(selected_idx)
            done = terminated | truncated

            pct_obs = extras["pct_obs"]
            all_nodes, leaf_nodes = pct_tools.get_leaf_nodes(pct_obs, env.internal_node_holder, env.leaf_node_holder)
            all_nodes = all_nodes.to(env.device)
            leaf_nodes = leaf_nodes.to(env.device)

            storage.insert(
                all_nodes,
                selected_idx,
                selected_log_prob,
                reward.unsqueeze(-1),
                (~done).float().unsqueeze(-1),
            )

            episode_returns += reward.detach()
            episode_lengths += 1
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            for env_id in done_ids.detach().cpu().tolist():
                recent_returns.append(float(episode_returns[env_id].item()))
                recent_lengths.append(float(episode_lengths[env_id].item()))
                episode_returns[env_id] = 0.0
                episode_lengths[env_id] = 0.0

        with torch.no_grad():
            _, _, _, next_value = policy(storage.obs[-1], normFactor=pct_args.normFactor)
        storage.compute_returns(next_value)

        obs_shape = storage.obs.size()[2:]
        action_shape = storage.actions.size()[-1]
        values, selected_log_prob, dist_entropy = policy.evaluate_actions(
            storage.obs[:-1].view(-1, *obs_shape),
            storage.actions.view(-1, action_shape),
            normFactor=pct_args.normFactor,
        )
        values = values.view(args_cli.num_steps, env.num_envs, 1)
        selected_log_prob = selected_log_prob.view(args_cli.num_steps, env.num_envs, 1)

        advantages = storage.returns[:-1] - values
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * selected_log_prob).mean()
        loss = args_cli.actor_loss_coef * actor_loss + args_cli.critic_loss_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args_cli.max_grad_norm)
        optimizer.step()
        storage.after_update()

        if update % args_cli.log_interval == 0:
            elapsed = max(time.perf_counter() - train_start, 1e-6)
            samples = (update - start_update) * args_cli.num_steps * env.num_envs
            fps = samples / elapsed
            mean_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
            mean_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
            print(
                "[gat-train] "
                f"update={update} samples={samples} fps={fps:.1f} "
                f"loss={loss.item():.4f} actor={actor_loss.item():.4f} "
                f"value={critic_loss.item():.4f} entropy={dist_entropy.mean().item():.4f} "
                f"mean_return={mean_return:.3f} mean_len={mean_length:.1f}",
                flush=True,
            )

        if update % args_cli.save_interval == 0 or update == start_update + args_cli.updates:
            save_checkpoint(run_dir / "PCT-resume.pt", policy, optimizer, update, pct_args)
            torch.save(policy.state_dict(), run_dir / "PCT-latest.pt")
            torch.save(policy.state_dict(), run_dir / f"PCT-update-{update:06d}.pt")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
