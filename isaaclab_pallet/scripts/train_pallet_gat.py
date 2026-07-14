from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "templete code"
ONLINE_PCT_DIR = PROJECT_ROOT / "Online-3D-BPP-PCT"
sys.path.insert(0, str(TEMPLATE_DIR))
sys.path.insert(0, str(ONLINE_PCT_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Headless Isaac Lab training using the original Online-3D-BPP-PCT GAT.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--num-packer-workers", type=int, default=0, help="CPU packer worker processes for observe/place/reward.")
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
parser.add_argument("--box-seed", type=int, default=0, help="Seed for the spec-random box pool; cycle it across runs for full size coverage.")
parser.add_argument(
    "--box-source",
    choices=["random", "sequence", "fixed_type_random"],
    default="random",
    help="Training box source: continuous random, one fixed sequence, or uniform random draw from fixed official types.",
)
parser.add_argument("--train-box-seq-dir", type=str, default="submit_buffer3_search/box_sequence")
parser.add_argument("--train-type-sequences", nargs="+", default=["box_sequence_0", "box_sequence_1"])
parser.add_argument("--candidate-generator", choices=["ems"], default="ems")
parser.add_argument("--drift-fail-threshold", type=float, default=0.40)
parser.add_argument("--tilt-fail-threshold", type=float, default=0.35)
parser.add_argument("--out-of-bounds-margin", type=float, default=0.02)
parser.add_argument("--height-fail-margin", type=float, default=0.005)
parser.add_argument("--drop-fail-threshold", type=float, default=0.08)
parser.add_argument("--candidate-rerank-k", type=int, default=0, help=">1: test diverse policy top-k leaves with Isaac before committing.")
parser.add_argument("--candidate-diversity-center-m", type=float, default=0.05)
parser.add_argument(
    "--reward-profile",
    choices=[
        "base",
        "floor_low",
        "smooth_low",
        "terminal_ratio",
        "finish_ratio",
        "terminal_ratio_t18",
        "finish_ratio_t15",
        "geometry_v1",
        "sun_v1",
        "sun_v2",
        "sun_v3",
        "sun_gap_guard_v1",
        "sun_anchor_plateau_v1",
    ],
    default="base",
    help="Reward-scale preset. Keeps the Isaac Lab GAT pipeline unchanged.",
)
# ★2 entropy bonus (escape plateau). dist_entropy is added to the loss to encourage exploration.
parser.add_argument("--entropy-coef", type=float, default=0.01)
# ★1 eval-in-loop + best-checkpoint: every N updates, score the policy on the real
# competition sequences and only save PCT-best.pt when the score improves.
parser.add_argument("--eval-interval", type=int, default=50, help="Updates between competition-score evals (0=off).")
parser.add_argument("--box-seq-dir", type=str, default="palletizing_simulator/box_sequence")
parser.add_argument("--eval-sequences", nargs="+", default=["box_sequence_0", "box_sequence_1"])
parser.add_argument("--confirm-box-seq-dir", type=str, default="", help="Optional second-stage deploy eval directory.")
parser.add_argument("--confirm-eval-sequences", nargs="+", default=[], help="Optional second-stage deploy eval sequence names.")
parser.add_argument("--deploy-eval-best", action="store_true", default=True, help="Confirm candidate best checkpoints through exported ONNX + submission main.py.")
parser.add_argument("--no-deploy-eval-best", dest="deploy_eval_best", action="store_false")
parser.add_argument("--deploy-eval-submit-dir", type=str, default="submit_buffer3_search")
parser.add_argument("--deploy-eval-buffer-size", type=int, default=0, help="buffer.size used in the temporary submission config for deployment best selection.")
parser.add_argument("--deploy-eval-timeout", type=float, default=300.0)
parser.add_argument("--deploy-eval-regression-guard", type=float, default=5.0, help="Reject deploy best candidates that regress any shared sequence by more than this many score points.")
parser.add_argument("--save-update-checkpoints", action="store_true", default=True)
parser.add_argument("--no-save-update-checkpoints", dest="save_update_checkpoints", action="store_false")
parser.add_argument("--wandb", action="store_true", help="Log training/eval metrics to Weights & Biases.")
parser.add_argument("--wandb-project", type=str, default="", help="W&B project. Falls back to WANDB_PROJECT or assignment2-pallet.")
parser.add_argument("--wandb-entity", type=str, default="", help="W&B entity/user/team. Falls back to WANDB_ENTITY.")
parser.add_argument("--wandb-name", type=str, default="", help="W&B run display name. Defaults to --run-name.")
parser.add_argument("--wandb-group", type=str, default="", help="W&B group for related runs.")
parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="", help="W&B mode. Falls back to WANDB_MODE.")
parser.add_argument("--wandb-dir", type=str, default="", help="Directory for W&B local files. Falls back to WANDB_DIR or run_dir/wandb.")
parser.add_argument("--wandb-tags", nargs="*", default=[], help="Space-separated W&B tags.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_pallet import PalletPackingEnv, PalletPackingEnvCfg  # noqa: E402

import tools as pct_tools  # noqa: E402
from model import DRL_GAT  # noqa: E402
from storage import PCTRolloutStorage  # noqa: E402

# Reuse the (tested) competition rollout to score checkpoints mid-training.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_competition_generate as _cg  # noqa: E402

_PALLET_VOLUME = 1.2 * 1.0 * 1.25


@dataclass(frozen=True)
class DeployEvalMetrics:
    mean_score: float
    min_score: float
    bottom10_score: float
    per_sequence: dict[str, float]


@dataclass(frozen=True)
class DeployBestDecision:
    accepted: bool
    mean_improved: bool
    max_regression: float
    regression_count: int
    guard_rejected: bool


def _summarize_sequence_scores(per_sequence: dict[str, float]) -> DeployEvalMetrics:
    scores = list(per_sequence.values())
    if not scores:
        return DeployEvalMetrics(0.0, 0.0, 0.0, per_sequence)
    sorted_scores = sorted(scores)
    bottom_n = max(1, ceil(len(sorted_scores) * 0.10))
    return DeployEvalMetrics(
        mean_score=sum(scores) / len(scores),
        min_score=sorted_scores[0],
        bottom10_score=sum(sorted_scores[:bottom_n]) / bottom_n,
        per_sequence=per_sequence,
    )


def _decide_deploy_best(
    candidate: DeployEvalMetrics,
    best: DeployEvalMetrics | None,
    regression_guard: float,
) -> DeployBestDecision:
    if best is None:
        return DeployBestDecision(
            accepted=True,
            mean_improved=True,
            max_regression=0.0,
            regression_count=0,
            guard_rejected=False,
        )

    mean_improved = candidate.mean_score > best.mean_score
    shared = sorted(set(best.per_sequence) & set(candidate.per_sequence))
    regressions = [best.per_sequence[name] - candidate.per_sequence[name] for name in shared]
    max_regression = max(regressions, default=0.0)
    regression_count = sum(1 for value in regressions if value > regression_guard)
    guard_rejected = regression_count > 0
    return DeployBestDecision(
        accepted=mean_improved and not guard_rejected,
        mean_improved=mean_improved,
        max_regression=max_regression,
        regression_count=regression_count,
        guard_rejected=guard_rejected,
    )


def apply_reward_profile(cfg: PalletPackingEnvCfg, profile: str) -> None:
    """Apply reward-scale presets to the existing Isaac Lab environment config."""
    profiles = {
        "base": {},
        "floor_low": {
            "floor_coverage_reward_scale": 2.5,
            "boundary_floor_reward_scale": 0.8,
            "corner_floor_reward_scale": 0.5,
            "height_smoothness_reward_scale": 0.8,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.08,
            "elevation_penalty_scale": 0.4,
        },
        "smooth_low": {
            "floor_coverage_reward_scale": 1.8,
            "boundary_floor_reward_scale": 0.8,
            "corner_floor_reward_scale": 0.5,
            "height_smoothness_reward_scale": 1.8,
            "support_reward_scale": 0.12,
            "weak_support_penalty_scale": 0.12,
            "elevation_penalty_scale": 0.3,
        },
        "terminal_ratio": {
            "floor_coverage_reward_scale": 1.2,
            "boundary_floor_reward_scale": 0.8,
            "corner_floor_reward_scale": 0.6,
            "height_smoothness_reward_scale": 0.8,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.08,
            "elevation_penalty_scale": 0.2,
            "terminal_ratio_reward_scale": 15.0,
        },
        "finish_ratio": {
            "floor_coverage_reward_scale": 1.2,
            "boundary_floor_reward_scale": 0.8,
            "corner_floor_reward_scale": 0.6,
            "height_smoothness_reward_scale": 0.8,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.08,
            "elevation_penalty_scale": 0.2,
            "terminal_ratio_reward_scale": 18.0,
            "learn_finish_action": True,
        },
        "terminal_ratio_t18": {
            "floor_coverage_reward_scale": 1.2,
            "boundary_floor_reward_scale": 0.8,
            "corner_floor_reward_scale": 0.6,
            "height_smoothness_reward_scale": 0.8,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.08,
            "elevation_penalty_scale": 0.2,
            "terminal_ratio_reward_scale": 18.0,
        },
        "finish_ratio_t15": {
            "floor_coverage_reward_scale": 1.2,
            "boundary_floor_reward_scale": 0.8,
            "corner_floor_reward_scale": 0.6,
            "height_smoothness_reward_scale": 0.8,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.08,
            "elevation_penalty_scale": 0.2,
            "terminal_ratio_reward_scale": 15.0,
            "learn_finish_action": True,
        },
        "geometry_v1": {
            "volume_reward_scale": 6.0,
            "floor_coverage_reward_scale": 1.2,
            "boundary_floor_reward_scale": 0.8,
            "corner_floor_reward_scale": 0.6,
            "height_smoothness_reward_scale": 1.0,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.10,
            "elevation_penalty_scale": 0.20,
            "corner_large_reward_scale": 0.55,
            "wall_anchor_reward_scale": 0.25,
            "tight_fit_reward_scale": 0.35,
            "void_reduction_reward_scale": 0.25,
            "support_margin_reward_scale": 0.25,
        },
        "sun_v1": {
            "volume_reward_scale": 0.0,
            "floor_coverage_reward_scale": 0.0,
            "boundary_floor_reward_scale": 0.0,
            "corner_floor_reward_scale": 0.0,
            "height_smoothness_reward_scale": 1.0,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.10,
            "elevation_penalty_scale": 0.20,
            "corner_large_reward_scale": 0.55,
            "wall_anchor_reward_scale": 0.0,
            "tight_fit_reward_scale": 0.7,
            "void_reduction_reward_scale": 0.25,
            "support_margin_reward_scale": 0.25,
            "active_layer_coverage_reward_scale": 1.2,
            "center_of_mass_z_penalty_scale": 0.25,
            "terminal_ratio_reward_scale": 15.0,
        },
        "sun_v2": {
            "volume_reward_scale": 0.0,
            "floor_coverage_reward_scale": 0.0,
            "boundary_floor_reward_scale": 0.0,
            "corner_floor_reward_scale": 0.0,
            "height_smoothness_reward_scale": 1.0,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.10,
            "elevation_penalty_scale": 0.20,
            "corner_large_reward_scale": 0.55,
            "wall_anchor_reward_scale": 0.0,
            "tight_fit_reward_scale": 0.7,
            "void_reduction_reward_scale": 0.25,
            "support_margin_reward_scale": 0.25,
            "active_layer_coverage_reward_scale": 1.2,
            "center_of_mass_z_penalty_scale": 0.25,
            "terminal_ratio_reward_scale": 18.0,
        },
        "sun_v3": {
            "volume_reward_scale": 0.0,
            "floor_coverage_reward_scale": 0.0,
            "boundary_floor_reward_scale": 0.0,
            "corner_floor_reward_scale": 0.0,
            "height_smoothness_reward_scale": 1.0,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.10,
            "elevation_penalty_scale": 0.20,
            "corner_large_reward_scale": 0.55,
            "wall_anchor_reward_scale": 0.0,
            "tight_fit_reward_scale": 0.7,
            "void_reduction_reward_scale": 0.25,
            "support_margin_reward_scale": 0.25,
            "active_layer_coverage_reward_scale": 1.2,
            "center_of_mass_z_penalty_scale": 0.25,
            "terminal_ratio_reward_scale": 21.0,
        },
        "sun_gap_guard_v1": {
            "volume_reward_scale": 0.0,
            "floor_coverage_reward_scale": 0.0,
            "boundary_floor_reward_scale": 0.0,
            "corner_floor_reward_scale": 0.0,
            "height_smoothness_reward_scale": 0.6,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.10,
            "elevation_penalty_scale": 0.20,
            "corner_large_reward_scale": 0.55,
            "wall_anchor_reward_scale": 0.0,
            "tight_fit_reward_scale": 0.65,
            "void_reduction_reward_scale": 0.25,
            "support_margin_reward_scale": 0.35,
            "active_layer_coverage_reward_scale": 0.8,
            "center_of_mass_z_penalty_scale": 0.25,
            "sliver_void_penalty_scale": 1.0,
            "blocked_void_penalty_scale": 0.8,
            "terminal_ratio_reward_scale": 18.0,
        },
        "sun_anchor_plateau_v1": {
            "volume_reward_scale": 0.0,
            "floor_coverage_reward_scale": 0.0,
            "boundary_floor_reward_scale": 0.0,
            "corner_floor_reward_scale": 0.0,
            "height_smoothness_reward_scale": 0.6,
            "support_reward_scale": 0.08,
            "weak_support_penalty_scale": 0.10,
            "elevation_penalty_scale": 0.20,
            "corner_large_reward_scale": 0.90,
            "wall_anchor_reward_scale": 0.45,
            "tight_fit_reward_scale": 0.90,
            "void_reduction_reward_scale": 0.25,
            "support_margin_reward_scale": 0.25,
            "active_layer_coverage_reward_scale": 0.8,
            "center_of_mass_z_penalty_scale": 0.25,
            "sliver_void_penalty_scale": 0.55,
            "blocked_void_penalty_scale": 0.45,
            "terminal_ratio_reward_scale": 18.0,
        },
    }
    for name, value in profiles[profile].items():
        setattr(cfg, name, value)


def evaluate_competition_score(policy, pct_args, pct_cfg, seq_paths, device) -> float:
    """Geometric competition score (fill% + buffer_bonus 20) on fixed sequences,
    averaged. Matches the official PHYSICS score because PCT placements settle stably
    (verified this session: geometric 89.3 == physics 89.25). Used to pick PCT-best.pt."""
    was_training = policy.training
    policy.eval()
    scores = []
    for path in seq_paths:
        boxes = _cg.load_boxes(Path(path))
        result = _cg.run_sequence(boxes, policy, pct_args, pct_cfg, device, sample=False)
        placed = result["sequence"]
        vol = sum(b["size"][0] * b["size"][1] * b["size"][2] for b in placed)
        fill = vol / _PALLET_VOLUME
        max_top = max((b["position"][2] + b["size"][2] / 2.0 for b in placed), default=0.0)
        # competition: episode 0 if it exceeds pallet height (1.25 m); else fill*100 + buffer_bonus(20)
        ep_score = (fill * 100.0 + 20.0) if max_top <= 1.25 + 1e-6 else 0.0
        scores.append(ep_score)
    if was_training:
        policy.train()
    return (sum(scores) / len(scores)) if scores else 0.0


def _copy_submission_for_deploy_eval(src: Path, dst: Path) -> None:
    if dst.exists():
        return

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache"}
        ignored.update(name for name in names if name.startswith("algorithm_results"))
        return ignored

    shutil.copytree(src, dst, ignore=ignore)


def _write_deploy_eval_config(submit_dir: Path, input_dir: Path, output_dir: Path, buffer_size: int) -> None:
    cfg_path = submit_dir / "config" / "algorithm_config.yaml"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg["input_path"] = str(input_dir.resolve())
    cfg["output_dir"] = str(output_dir.resolve())
    cfg.setdefault("buffer", {})["size"] = int(buffer_size)
    if int(buffer_size) == 0:
        cfg.setdefault("selector", {})["enabled"] = False
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _prepare_deploy_eval_inputs(seq_paths: list[str], dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for seq_path in seq_paths:
        src = Path(seq_path)
        shutil.copy2(src, dst / src.name)


def _score_submission_results(output_dir: Path, seq_paths: list[str], buffer_size: int) -> DeployEvalMetrics:
    expected = [Path(path).stem for path in seq_paths]
    missing = [name for name in expected if not (output_dir / f"{name}.json").is_file()]
    if missing:
        raise RuntimeError(
            "deploy eval missing output JSONs: "
            + ", ".join(missing)
            + f" in {output_dir}"
        )

    per_sequence: dict[str, float] = {}
    for name in expected:
        path = output_dir / f"{name}.json"
        data = json.loads(path.read_text())
        placed = data.get("sequence", [])
        vol = sum(float(b["size"][0]) * float(b["size"][1]) * float(b["size"][2]) for b in placed)
        fill = vol / _PALLET_VOLUME
        max_top = max(
            (float(b["position"][2]) + float(b["size"][2]) / 2.0 for b in placed),
            default=0.0,
        )
        bonus = max(0.0, 20.0 - float(buffer_size))
        per_sequence[name] = fill * 100.0 + bonus if max_top <= 1.25 + 1e-6 else 0.0
    return _summarize_sequence_scores(per_sequence)


def evaluate_deployment_score(
    policy,
    run_dir: Path,
    pct_args,
    seq_paths: list[str],
    update: int,
) -> DeployEvalMetrics:
    deploy_root = run_dir / "deploy_eval"
    submit_src = PROJECT_ROOT / args_cli.deploy_eval_submit_dir
    submit_dir = deploy_root / "submission"
    input_dir = deploy_root / "box_sequence"
    output_dir = deploy_root / "algorithm_results"
    candidate_pt = deploy_root / f"candidate-{update:06d}.pt"

    deploy_root.mkdir(parents=True, exist_ok=True)
    _copy_submission_for_deploy_eval(submit_src, submit_dir)
    _prepare_deploy_eval_inputs(seq_paths, input_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _write_deploy_eval_config(submit_dir, input_dir, output_dir, args_cli.deploy_eval_buffer_size)

    torch.save(policy.state_dict(), candidate_pt)
    export_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "export_pct_no_gym.py"),
        "--model-path",
        str(candidate_pt),
        "--out",
        str(submit_dir / "src" / "models" / "pct_model.onnx"),
        "--internal-node-holder",
        str(pct_args.internal_node_holder),
        "--leaf-node-holder",
        str(pct_args.leaf_node_holder),
        "--setting",
        str(pct_args.setting),
    ]
    subprocess.run(
        export_cmd,
        cwd=str(PROJECT_ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args_cli.deploy_eval_timeout,
    )
    subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(submit_dir),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args_cli.deploy_eval_timeout,
    )
    return _score_submission_results(output_dir, seq_paths, args_cli.deploy_eval_buffer_size)


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
        learn_finish_action=bool(env.cfg.learn_finish_action),
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
            "learn_finish_action": pct_args.learn_finish_action,
            "action_space": pct_args.leaf_node_holder + (1 if pct_args.learn_finish_action else 0),
            "candidate_rerank_k": args_cli.candidate_rerank_k,
            "candidate_diversity_center_m": args_cli.candidate_diversity_center_m,
        },
        path,
    )


def init_wandb_run(run_dir: Path, pct_args: SimpleNamespace, start_update: int):
    if not args_cli.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested with --wandb, but the `wandb` package is not installed. "
            "Install it in env_isaaclab or run without --wandb."
        ) from exc

    wandb_dir = Path(args_cli.wandb_dir or os.environ.get("WANDB_DIR") or (run_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run_id_path = run_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id + "\n")

    project = args_cli.wandb_project or os.environ.get("WANDB_PROJECT") or "assignment2-pallet"
    entity = args_cli.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    mode = args_cli.wandb_mode or os.environ.get("WANDB_MODE") or None
    group = args_cli.wandb_group or os.environ.get("WANDB_RUN_GROUP") or None
    name = args_cli.wandb_name or args_cli.run_name or run_dir.name
    tags = list(args_cli.wandb_tags)
    if args_cli.reward_profile not in tags:
        tags.append(args_cli.reward_profile)
    if f"k{args_cli.candidate_rerank_k}" not in tags:
        tags.append(f"k{args_cli.candidate_rerank_k}")

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        id=run_id,
        resume="allow",
        mode=mode,
        dir=str(wandb_dir),
        tags=tags,
        job_type="train",
        config={
            "run_name": args_cli.run_name,
            "reward_profile": args_cli.reward_profile,
            "num_envs": args_cli.num_envs,
            "max_boxes": args_cli.max_boxes,
            "updates": args_cli.updates,
            "start_update": start_update,
            "num_steps": args_cli.num_steps,
            "save_interval": args_cli.save_interval,
            "eval_interval": args_cli.eval_interval,
            "learning_rate": args_cli.learning_rate,
            "gamma": args_cli.gamma,
            "actor_loss_coef": args_cli.actor_loss_coef,
            "critic_loss_coef": args_cli.critic_loss_coef,
            "entropy_coef": args_cli.entropy_coef,
            "max_grad_norm": args_cli.max_grad_norm,
            "candidate_rerank_k": args_cli.candidate_rerank_k,
            "candidate_diversity_center_m": args_cli.candidate_diversity_center_m,
            "save_update_checkpoints": args_cli.save_update_checkpoints,
            "seed": args_cli.seed,
            "box_seed": args_cli.box_seed,
            "box_source": args_cli.box_source,
            "train_box_seq_dir": args_cli.train_box_seq_dir,
            "train_type_sequences": args_cli.train_type_sequences,
            "candidate_generator": args_cli.candidate_generator,
            "device": args_cli.device,
            "load_model": args_cli.load_model,
            "resume_checkpoint": args_cli.resume,
            "internal_node_holder": pct_args.internal_node_holder,
            "leaf_node_holder": pct_args.leaf_node_holder,
            "embedding_size": pct_args.embedding_size,
            "hidden_size": pct_args.hidden_size,
            "gat_layer_num": pct_args.gat_layer_num,
            "normFactor": pct_args.normFactor,
            "learn_finish_action": pct_args.learn_finish_action,
        },
    )
    run.define_metric("update")
    run.define_metric("train/*", step_metric="update")
    run.define_metric("eval/*", step_metric="update")
    run.define_metric("checkpoint/*", step_metric="update")
    print(f"[gat-train] wandb run url: {run.url}", flush=True)
    return run


def load_resume(path: str, policy: DRL_GAT, optimizer: torch.optim.Optimizer, device: str) -> int:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    loaded_strict = True
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError:
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        allowed_missing = all(name.startswith("actor.finish_head.") for name in missing)
        if unexpected or not allowed_missing:
            raise
        loaded_strict = False
        print(
            f"[gat-train] resumed compatible weights from {path}; initialized learned finish head",
            flush=True,
        )
    if loaded_strict and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    elif isinstance(ckpt, dict) and "optimizer" in ckpt:
        print("[gat-train] skipped optimizer state because model architecture changed", flush=True)
    return int(ckpt.get("update", 0)) if isinstance(ckpt, dict) else 0


def load_warm_start(path: str, policy: DRL_GAT) -> DRL_GAT:
    # Existing checkpoints do not have the learned finish head. For finish_ratio,
    # keep all compatible PCT weights and leave only actor.finish_head initialized.
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, (list, tuple)) and len(sd) == 2:
        sd = sd[0]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    try:
        policy.load_state_dict(sd, strict=True)
        print(f"[gat-train] loaded GAT checkpoint: {path}", flush=True)
        return policy
    except RuntimeError:
        try:
            missing, unexpected = policy.load_state_dict(sd, strict=False)
            allowed_missing = all(name.startswith("actor.finish_head.") for name in missing)
            if unexpected or not allowed_missing:
                raise RuntimeError(f"missing={missing} unexpected={unexpected}")
            print(
                f"[gat-train] loaded GAT checkpoint with learned finish head initialized: {path}",
                flush=True,
            )
            return policy
        except RuntimeError:
            if getattr(policy.actor, "learn_finish_action", False):
                load_dict = {}
                raw = torch.load(path, map_location="cpu")
                if isinstance(raw, (list, tuple)) and len(raw) == 2:
                    raw = raw[0]
                if isinstance(raw, dict) and "model" in raw:
                    raw = raw["model"]
                for k, v in raw.items():
                    if "actor.embedder.layers" in k:
                        load_dict[k.replace("module.weight", "weight")] = v
                    else:
                        load_dict[k.replace("module.", "")] = v
                load_dict = {k.replace("add_bias.", ""): v for k, v in load_dict.items()}
                load_dict = {k.replace("_bias", "bias"): v for k, v in load_dict.items()}
                for k, v in list(load_dict.items()):
                    if hasattr(v, "size") and len(v.size()) <= 3:
                        load_dict[k] = v.squeeze(dim=-1)
                missing, unexpected = policy.load_state_dict(load_dict, strict=False)
                allowed_missing = all(name.startswith("actor.finish_head.") for name in missing)
                if unexpected or not allowed_missing:
                    raise RuntimeError(f"missing={missing} unexpected={unexpected}")
                print(
                    f"[gat-train] loaded original PCT weights with learned finish head initialized: {path}",
                    flush=True,
                )
            else:
                policy = pct_tools.load_policy(path, policy)
                print(f"[gat-train] loaded original PCT weights: {path}", flush=True)
            return policy


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
    cfg.num_packer_workers = args_cli.num_packer_workers
    cfg.max_boxes = args_cli.max_boxes
    cfg.box_seed = args_cli.box_seed
    cfg.box_source = args_cli.box_source
    cfg.random_boxes = args_cli.box_source == "random"
    cfg.train_box_seq_dir = str(PROJECT_ROOT / args_cli.train_box_seq_dir)
    cfg.train_type_sequences = tuple(args_cli.train_type_sequences)
    cfg.sim.device = args_cli.device
    cfg.drift_fail_threshold = args_cli.drift_fail_threshold
    cfg.tilt_fail_threshold = args_cli.tilt_fail_threshold
    cfg.out_of_bounds_margin = args_cli.out_of_bounds_margin
    cfg.height_fail_margin = args_cli.height_fail_margin
    cfg.drop_fail_threshold = args_cli.drop_fail_threshold
    cfg.candidate_rerank_k = args_cli.candidate_rerank_k
    cfg.candidate_diversity_center_m = args_cli.candidate_diversity_center_m
    apply_reward_profile(cfg, args_cli.reward_profile)

    env = PalletPackingEnv(cfg)
    obs_dict, _ = env.reset(seed=args_cli.seed)
    pct_obs = env.extras["pct_obs"]
    all_nodes, leaf_nodes = pct_tools.get_leaf_nodes(pct_obs, env.internal_node_holder, env.leaf_node_holder)
    all_nodes = all_nodes.to(env.device)
    leaf_nodes = leaf_nodes.to(env.device)

    pct_args = make_pct_args(env)
    policy = DRL_GAT(pct_args).to(env.device)
    if args_cli.load_model:
        policy = load_warm_start(args_cli.load_model, policy)
        policy = policy.to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)

    start_update = 0
    if args_cli.resume:
        start_update = load_resume(args_cli.resume, policy, optimizer, env.device)
        print(f"[gat-train] resumed from {args_cli.resume} at update={start_update}", flush=True)

    run_dir = make_run_dir()
    wandb_run = init_wandb_run(run_dir, pct_args, start_update)
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
        f"num_packer_workers={cfg.num_packer_workers} "
        f"obs_shape={tuple(all_nodes.shape)} leaf_shape={tuple(leaf_nodes.shape)} "
        f"num_steps={args_cli.num_steps} normFactor={pct_args.normFactor} "
        f"drift_fail_threshold={cfg.drift_fail_threshold} "
        f"box_source={cfg.box_source} box_seed={cfg.box_seed} "
        f"train_box_seq_dir={args_cli.train_box_seq_dir} "
        f"train_type_sequences={','.join(args_cli.train_type_sequences)} "
        f"reward_profile={args_cli.reward_profile} "
        f"learn_finish_action={pct_args.learn_finish_action} action_space={cfg.action_space} "
        f"candidate_generator={args_cli.candidate_generator} "
        f"candidate_rerank_k={cfg.candidate_rerank_k} "
        f"candidate_diversity_center_m={cfg.candidate_diversity_center_m}",
        flush=True,
    )

    # ★1 eval-in-loop setup: resolve the real competition sequences and seed the best
    # score with the (warm-started) model so we can NEVER end up worse than the start.
    eval_seq_paths = [str(PROJECT_ROOT / args_cli.box_seq_dir / f"{n}.json") for n in args_cli.eval_sequences]
    confirm_eval_enabled = bool(args_cli.confirm_box_seq_dir and args_cli.confirm_eval_sequences)
    confirm_eval_seq_paths = [
        str(PROJECT_ROOT / args_cli.confirm_box_seq_dir / f"{n}.json")
        for n in args_cli.confirm_eval_sequences
    ]
    print(
        "[gat-train] "
        f"eval_seq_count={len(eval_seq_paths)} "
        f"box_seq_dir={args_cli.box_seq_dir} "
        f"eval_sequences={','.join(args_cli.eval_sequences)}",
        flush=True,
    )
    if confirm_eval_enabled:
        print(
            "[gat-train] "
            f"confirm_eval_seq_count={len(confirm_eval_seq_paths)} "
            f"confirm_box_seq_dir={args_cli.confirm_box_seq_dir} "
            f"confirm_eval_sequences={','.join(args_cli.confirm_eval_sequences)}",
            flush=True,
        )
    best_score = -1.0
    best_torch_score = -1.0
    best_deploy_metrics: DeployEvalMetrics | None = None
    best_confirm_deploy_metrics: DeployEvalMetrics | None = None
    if args_cli.eval_interval > 0:
        torch_score = evaluate_competition_score(policy, pct_args, env.pct_cfg, eval_seq_paths, env.device)
        best_torch_score = torch_score
        if args_cli.deploy_eval_best:
            best_deploy_metrics = evaluate_deployment_score(policy, run_dir, pct_args, eval_seq_paths, start_update)
            if confirm_eval_enabled:
                best_confirm_deploy_metrics = evaluate_deployment_score(
                    policy,
                    run_dir,
                    pct_args,
                    confirm_eval_seq_paths,
                    start_update,
                )
                best_score = best_confirm_deploy_metrics.mean_score
            else:
                best_score = best_deploy_metrics.mean_score
            confirm_text = (
                f"confirm_score={best_confirm_deploy_metrics.mean_score:.2f} "
                f"confirm_min={best_confirm_deploy_metrics.min_score:.2f} "
                f"confirm_bottom10={best_confirm_deploy_metrics.bottom10_score:.2f} "
                if best_confirm_deploy_metrics is not None
                else ""
            )
            print(
                f"[gat-train] initial deployment score = {best_deploy_metrics.mean_score:.2f} "
                f"min={best_deploy_metrics.min_score:.2f} "
                f"bottom10={best_deploy_metrics.bottom10_score:.2f} "
                f"{confirm_text}"
                f"(torch_eval={torch_score:.2f})  -> PCT-best.pt",
                flush=True,
            )
        else:
            best_score = torch_score
            print(f"[gat-train] initial competition score = {best_score:.2f}  -> PCT-best.pt", flush=True)
        torch.save(policy.state_dict(), run_dir / "PCT-best.pt")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "update": start_update,
                    "eval/competition_score": torch_score,
                    "eval/deployment_score": best_score,
                    "eval/deployment_primary_score": (
                        best_deploy_metrics.mean_score if best_deploy_metrics is not None else -1.0
                    ),
                    "eval/deployment_confirm_score": (
                        best_confirm_deploy_metrics.mean_score if best_confirm_deploy_metrics is not None else -1.0
                    ),
                    "eval/best_score": best_score,
                    "eval/deployment_min_score": (
                        best_deploy_metrics.min_score if best_deploy_metrics is not None else -1.0
                    ),
                    "eval/deployment_bottom10_score": (
                        best_deploy_metrics.bottom10_score if best_deploy_metrics is not None else -1.0
                    ),
                    "eval/deployment_max_regression": 0.0,
                    "eval/deployment_regression_count": 0,
                    "eval/deployment_guard_rejected": 0,
                    "eval/deployment_confirm_min_score": (
                        best_confirm_deploy_metrics.min_score if best_confirm_deploy_metrics is not None else -1.0
                    ),
                    "eval/deployment_confirm_bottom10_score": (
                        best_confirm_deploy_metrics.bottom10_score if best_confirm_deploy_metrics is not None else -1.0
                    ),
                    "eval/improved": 1,
                },
                step=start_update,
            )

    for update in range(start_update + 1, start_update + args_cli.updates + 1):
        policy.train()
        storage.step = 0

        for _ in range(args_cli.num_steps):
            with torch.no_grad():
                selected_log_prob, selected_idx, dist_entropy, hidden, dist = policy.actor(
                    all_nodes, normFactor=pct_args.normFactor
                )
                _value = policy.critic(hidden)

            # Original Online-3D-BPP selected the leaf node here:
            # selected_leaf_node = leaf_nodes[batch_indices, selected_idx.squeeze()]
            # Isaac Lab env keeps the same index semantics, so env.step receives selected_idx directly.
            _selected_leaf_node = None
            leaf_actions = selected_idx.squeeze(-1) < env.leaf_node_holder
            if bool(leaf_actions.any().item()):
                _selected_leaf_node = leaf_nodes[batch_indices[leaf_actions], selected_idx.squeeze(-1)[leaf_actions]]
            if args_cli.candidate_rerank_k > 1:
                obs_dict, reward, terminated, truncated, extras = env.step(dist.probs.detach())
                chosen_action = extras["chosen_action"].to(env.device)
                chosen_action = torch.where(chosen_action < 0, selected_idx, chosen_action)
                selected_log_prob = dist.log_probs(chosen_action)
                selected_idx = chosen_action
            else:
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
        entropy = dist_entropy.mean()
        # ★2: subtract entropy (maximize it) to keep exploring and escape plateaus.
        loss = (
            args_cli.actor_loss_coef * actor_loss
            + args_cli.critic_loss_coef * critic_loss
            - args_cli.entropy_coef * entropy
        )

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
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "update": update,
                        "train/samples": samples,
                        "train/fps": fps,
                        "train/loss": float(loss.item()),
                        "train/actor_loss": float(actor_loss.item()),
                        "train/critic_loss": float(critic_loss.item()),
                        "train/entropy": float(dist_entropy.mean().item()),
                        "train/mean_return": mean_return,
                        "train/mean_length": mean_length,
                        "train/learning_rate": args_cli.learning_rate,
                    },
                    step=update,
                )
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
            if args_cli.save_update_checkpoints:
                torch.save(policy.state_dict(), run_dir / f"PCT-update-{update:06d}.pt")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "update": update,
                        "checkpoint/saved": 1,
                        "checkpoint/latest_update": update,
                    },
                    step=update,
                )

        # ★1: score on the real competition sequences; keep PCT-best.pt only when it improves.
        if args_cli.eval_interval > 0 and update % args_cli.eval_interval == 0:
            torch_score = evaluate_competition_score(policy, pct_args, env.pct_cfg, eval_seq_paths, env.device)
            deploy_metrics = None
            confirm_deploy_metrics = None
            torch_candidate = torch_score > best_torch_score
            if torch_candidate:
                best_torch_score = torch_score
            if args_cli.deploy_eval_best:
                deploy_metrics = evaluate_deployment_score(policy, run_dir, pct_args, eval_seq_paths, update)
                primary_decision = _decide_deploy_best(
                    deploy_metrics,
                    best_deploy_metrics,
                    args_cli.deploy_eval_regression_guard,
                )
                deploy_decision = primary_decision
                run_confirm_eval = confirm_eval_enabled and primary_decision.mean_improved
                if run_confirm_eval:
                    confirm_deploy_metrics = evaluate_deployment_score(
                        policy,
                        run_dir,
                        pct_args,
                        confirm_eval_seq_paths,
                        update,
                    )
                    deploy_decision = _decide_deploy_best(
                        confirm_deploy_metrics,
                        best_confirm_deploy_metrics,
                        args_cli.deploy_eval_regression_guard,
                    )
                    score = confirm_deploy_metrics.mean_score
                    improved = primary_decision.accepted and deploy_decision.accepted
                elif confirm_eval_enabled:
                    score = best_score
                    improved = False
                else:
                    score = deploy_metrics.mean_score
                    improved = deploy_decision.accepted
            else:
                score = torch_score
                deploy_decision = DeployBestDecision(
                    accepted=score > best_score,
                    mean_improved=score > best_score,
                    max_regression=0.0,
                    regression_count=0,
                    guard_rejected=False,
                )
                improved = deploy_decision.accepted
            if improved:
                best_score = score
                if deploy_metrics is not None:
                    best_deploy_metrics = deploy_metrics
                if confirm_deploy_metrics is not None:
                    best_confirm_deploy_metrics = confirm_deploy_metrics
                torch.save(policy.state_dict(), run_dir / "PCT-best.pt")
                save_checkpoint(run_dir / "PCT-best-resume.pt", policy, optimizer, update, pct_args)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "update": update,
                        "eval/competition_score": torch_score,
                        "eval/deployment_score": (
                            confirm_deploy_metrics.mean_score
                            if confirm_deploy_metrics is not None
                            else (deploy_metrics.mean_score if deploy_metrics is not None else -1.0)
                        ),
                        "eval/deployment_primary_score": deploy_metrics.mean_score if deploy_metrics is not None else -1.0,
                        "eval/deployment_confirm_score": (
                            confirm_deploy_metrics.mean_score if confirm_deploy_metrics is not None else -1.0
                        ),
                        "eval/deployment_min_score": deploy_metrics.min_score if deploy_metrics is not None else -1.0,
                        "eval/deployment_bottom10_score": (
                            deploy_metrics.bottom10_score if deploy_metrics is not None else -1.0
                        ),
                        "eval/deployment_confirm_min_score": (
                            confirm_deploy_metrics.min_score if confirm_deploy_metrics is not None else -1.0
                        ),
                        "eval/deployment_confirm_bottom10_score": (
                            confirm_deploy_metrics.bottom10_score if confirm_deploy_metrics is not None else -1.0
                        ),
                        "eval/deployment_max_regression": deploy_decision.max_regression,
                        "eval/deployment_regression_count": deploy_decision.regression_count,
                        "eval/deployment_guard_rejected": int(deploy_decision.guard_rejected),
                        "eval/deployment_mean_improved": int(deploy_decision.mean_improved),
                        "eval/best_score": best_score,
                        "eval/improved": int(improved),
                        "eval/torch_candidate": int(torch_candidate),
                    },
                    step=update,
                )
            confirm_update_text = (
                f" confirm_score={confirm_deploy_metrics.mean_score:.2f}"
                f" confirm_min={confirm_deploy_metrics.min_score:.2f}"
                f" confirm_bottom10={confirm_deploy_metrics.bottom10_score:.2f}"
                if confirm_deploy_metrics is not None
                else ""
            )
            deploy_text = (
                ""
                if deploy_metrics is None
                else (
                    f" deploy_score={deploy_metrics.mean_score:.2f}"
                    f" min={deploy_metrics.min_score:.2f}"
                    f" bottom10={deploy_metrics.bottom10_score:.2f}"
                    f"{confirm_update_text}"
                    f" max_regression={deploy_decision.max_regression:.2f}"
                    f" regressions={deploy_decision.regression_count}"
                    f" guard_rejected={int(deploy_decision.guard_rejected)}"
                )
            )
            print(
                f"[gat-train] eval update={update} torch_score={torch_score:.2f}"
                f"{deploy_text} best={best_score:.2f}"
                f"{'  *NEW BEST -> PCT-best.pt*' if improved else ''}",
                flush=True,
            )

    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        # Print the REAL error first: on the error path Isaac's
        # simulation_app.close() shutdown callback can busy-loop forever, which
        # otherwise swallows the propagating traceback and looks like a silent
        # 100%-CPU hang. Hard-exit non-zero so the failure is fast and visible
        # (and the sweep wrapper's `code=$?` stops the pipeline instead of hanging).
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
