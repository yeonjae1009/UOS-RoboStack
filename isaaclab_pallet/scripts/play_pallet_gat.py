from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="GUI/headless rollout for an Online-3D-BPP-PCT GAT checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--max-boxes", type=int, default=64)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--step-delay", type=float, default=0.5)
parser.add_argument("--hold-seconds", type=float, default=20.0)
parser.add_argument("--sample-action", action="store_true", help="Sample from the policy instead of deterministic argmax.")
parser.add_argument("--box-seed", type=int, default=0, help="Seed for the spec-random box set; change it to see a different dataset.")
parser.add_argument("--drift-fail-threshold", type=float, default=0.40)
parser.add_argument("--tilt-fail-threshold", type=float, default=0.35)
parser.add_argument("--out-of-bounds-margin", type=float, default=0.02)
parser.add_argument("--height-fail-margin", type=float, default=0.005)
parser.add_argument("--drop-fail-threshold", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

from isaaclab_pallet import PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONLINE_PCT_DIR = PROJECT_ROOT / "Online-3D-BPP-PCT"
sys.path.insert(0, str(ONLINE_PCT_DIR))

import tools as pct_tools  # noqa: E402
from model import DRL_GAT  # noqa: E402


def render_pause(env: PalletPackingEnv, seconds: float) -> None:
    if args_cli.headless or seconds <= 0.0:
        return
    end_time = time.time() + seconds
    while time.time() < end_time:
        env.sim.render()
        time.sleep(1.0 / 30.0)


def make_pct_args(env: PalletPackingEnv, ckpt: dict | None = None) -> SimpleNamespace:
    setting = int(env.pct_setting)
    default_internal_node_length = 7 if setting == 3 else 6
    default_norm_factor = float(env.pct_cfg.get("norm_factor", 1.0 / max(env.cfg.pallet_size)))
    internal_node_length = (
        int(ckpt.get("internal_node_length", default_internal_node_length)) if ckpt else default_internal_node_length
    )
    norm_factor = float(ckpt.get("normFactor", default_norm_factor)) if ckpt else default_norm_factor
    return SimpleNamespace(
        setting=setting,
        internal_node_holder=int(ckpt.get("internal_node_holder", env.internal_node_holder)) if ckpt else env.internal_node_holder,
        internal_node_length=internal_node_length,
        leaf_node_holder=int(ckpt.get("leaf_node_holder", env.leaf_node_holder)) if ckpt else env.leaf_node_holder,
        embedding_size=int(ckpt.get("embedding_size", 64)) if ckpt else 64,
        hidden_size=int(ckpt.get("hidden_size", 128)) if ckpt else 128,
        gat_layer_num=int(ckpt.get("gat_layer_num", 1)) if ckpt else 1,
        normFactor=norm_factor,
    )


def load_checkpoint(path: str, device: str) -> tuple[dict, dict | None]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt
    return ckpt, None


def load_policy_weights(path: str, policy: DRL_GAT, device: str) -> None:
    state_dict, ckpt = load_checkpoint(path, device)
    if ckpt is not None:
        policy.load_state_dict(state_dict)
        return

    try:
        policy.load_state_dict(state_dict)
    except RuntimeError:
        # Raw Online-3D-BPP pretrained files use AddBias/DataParallel-style keys.
        # Reuse the original conversion logic from Online-3D-BPP-PCT/tools.py.
        pct_tools.load_policy(path, policy)


def main() -> None:
    cfg = PalletPackingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.max_boxes = args_cli.max_boxes
    cfg.box_seed = args_cli.box_seed
    cfg.sim.device = args_cli.device
    cfg.drift_fail_threshold = args_cli.drift_fail_threshold
    cfg.tilt_fail_threshold = args_cli.tilt_fail_threshold
    cfg.out_of_bounds_margin = args_cli.out_of_bounds_margin
    cfg.height_fail_margin = args_cli.height_fail_margin
    cfg.drop_fail_threshold = args_cli.drop_fail_threshold

    env = PalletPackingEnv(cfg)
    obs_dict, _ = env.reset()
    pct_obs = env.extras["pct_obs"]
    all_nodes, _ = pct_tools.get_leaf_nodes(pct_obs, env.internal_node_holder, env.leaf_node_holder)
    all_nodes = all_nodes.to(env.device)

    _, ckpt = load_checkpoint(args_cli.checkpoint, env.device)
    pct_args = make_pct_args(env, ckpt)
    policy = DRL_GAT(pct_args).to(env.device)
    load_policy_weights(args_cli.checkpoint, policy, env.device)
    policy.eval()

    print(
        "[gat-play] "
        f"checkpoint={args_cli.checkpoint} obs_shape={tuple(all_nodes.shape)} "
        f"num_envs={env.num_envs} steps={args_cli.steps} normFactor={pct_args.normFactor}",
        f"action_mode={'sample' if args_cli.sample_action else 'deterministic'} "
        f"drift_fail_threshold={cfg.drift_fail_threshold}",
        flush=True,
    )

    for step in range(args_cli.steps):
        with torch.no_grad():
            _, selected_idx, _, _ = policy(
                all_nodes,
                deterministic=not args_cli.sample_action,
                normFactor=pct_args.normFactor,
            )
        obs_dict, reward, terminated, truncated, extras = env.step(selected_idx)
        pct_obs = extras["pct_obs"]
        all_nodes, _ = pct_tools.get_leaf_nodes(pct_obs, env.internal_node_holder, env.leaf_node_holder)
        all_nodes = all_nodes.to(env.device)
        action_mask = extras["action_mask"]
        valid_counts = action_mask.sum(dim=1).detach().cpu().tolist()
        height_ratio = extras["physics_features"][:, 4].detach().cpu().tolist()
        terminal_reason = extras["terminal_done_reason"].detach().cpu().tolist()
        terminal_drift = extras["terminal_drift"].detach().cpu().tolist()
        terminal_tilt = extras["terminal_tilt"].detach().cpu().tolist()
        terminal_height = extras["terminal_height_ratio"].detach().cpu().tolist()
        print(
            "[gat-play] "
            f"step={step} action={selected_idx.flatten().detach().cpu().tolist()} "
            f"reward={reward.detach().cpu().tolist()} "
            f"terminated={terminated.detach().cpu().tolist()} "
            f"truncated={truncated.detach().cpu().tolist()} "
            f"drift={env.last_drift.detach().cpu().tolist()} "
            f"tilt={env.last_tilt.detach().cpu().tolist()} "
            f"height_ratio={height_ratio} "
            f"done_reason={env.last_done_reason.detach().cpu().tolist()} "
            f"terminal_reason={terminal_reason} "
            f"terminal_drift={terminal_drift} "
            f"terminal_tilt={terminal_tilt} "
            f"terminal_height={terminal_height} "
            f"valid_leafs={valid_counts} "
            f"box_idx={env.current_box_idx}",
            flush=True,
        )
        render_pause(env, args_cli.step_delay)

    if not args_cli.headless and args_cli.hold_seconds > 0.0:
        print(f"[gat-play] holding GUI for {args_cli.hold_seconds:.1f}s", flush=True)
        render_pause(env, args_cli.hold_seconds)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
