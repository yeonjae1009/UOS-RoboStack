from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from . import pct_reward
from . import packer_pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "templete code"
DEFAULT_PCT_CONFIG = TEMPLATE_DIR / "config" / "pct_config.yaml"
DEFAULT_BOX_SEQUENCE = PROJECT_ROOT / "palletizing_simulator" / "box_sequence" / "box_sequence_0.json"
PCT_INTERNAL_NODE_LENGTH = 7


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_boxes(path: str | Path, limit: int) -> list[dict]:
    boxes: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            boxes.append(json.loads(line))
            if len(boxes) >= limit:
                break
    if not boxes:
        raise ValueError(f"No boxes found in {path}")
    return boxes


def _generate_random_boxes(
    n: int,
    seed: int,
    wl_range: tuple[float, float],
    h_range: tuple[float, float],
    mass_range: tuple[float, float],
) -> list[dict]:
    """Spec-compliant continuous-random boxes (CJ palletizing rules).

    W,L ~ U(0.17, 0.32) m, H ~ U(0.13, 0.26) m, mass ~ U(0.5, 6.0) kg — sampled
    continuously, NOT drawn from a fixed type set. Mirrors Online-3D-BPP-PCT
    bin3D.gen_next_box (sample_from_distribution) + givenData bounds. Sizes are
    rounded to 3 decimals (mm) as in the original; z is always the vertical height
    so the competition's {0,90} rotation is an x<->y swap.
    """
    rng = np.random.default_rng(seed)
    boxes: list[dict] = []
    for i in range(n):
        w = round(float(rng.uniform(*wl_range)), 3)
        l = round(float(rng.uniform(*wl_range)), 3)
        h = round(float(rng.uniform(*h_range)), 3)
        mass = round(float(rng.uniform(*mass_range)), 3)
        boxes.append({"id": i, "size": [w, l, h], "mass": mass})
    return boxes


def _yaw_quat_wxyz(degrees: float) -> tuple[float, float, float, float]:
    rad = math.radians(degrees)
    return (math.cos(rad / 2.0), 0.0, 0.0, math.sin(rad / 2.0))


# The deterministic CPU layer lives in pct_reward.py (shared with the equivalence
# test) and is driven per-env by packer_pool.py. The env keeps only the torch-side
# observation decode below.
def _pct_decode_observation(observation: torch.Tensor, internal_node_holder: int, leaf_node_holder: int):
    """Mirror Online-3D-BPP-PCT/tools.py::observation_decode_leaf_node."""
    internal_nodes = observation[:, 0:internal_node_holder, 0:PCT_INTERNAL_NODE_LENGTH]
    leaf_nodes = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, 0:8]
    current_box = observation[:, internal_node_holder + leaf_node_holder:, 0:6]
    valid_flag = observation[:, internal_node_holder:internal_node_holder + leaf_node_holder, 8]
    full_mask = observation[:, :, -1]
    return internal_nodes, leaf_nodes, current_box, valid_flag, full_mask


def _quat_wxyz_to_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(pitch_arg)
    return roll, pitch


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y))),
            torch.stack((2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x))),
            torch.stack((2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))),
        )
    )


@configclass
class PalletPackingEnvCfg(DirectRLEnvCfg):
    decimation = 20
    episode_length_s = 20.0
    action_space = 100
    physics_feature_dim = 5
    observation_space = 301 * 9 + physics_feature_dim
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)

    # ① Stage-3 throughput: CPU packer worker processes. 0 = serial (default, in
    # main process). >0 routes per-env observe/place/reward through a sharded
    # process pool (packer_pool.py). Tune to ~physical cores on the Isaac machine.
    num_packer_workers: int = 0

    box_sequence_path: str = str(DEFAULT_BOX_SEQUENCE)
    pct_config_path: str = str(DEFAULT_PCT_CONFIG)
    max_boxes: int = 8

    # Box source. random_boxes=True (default) generates spec-compliant continuous
    # random boxes; the committed box_sequence_*.json files are a FIXED 5-type set
    # (item_size_set) and are kept only for reproducible CPU tests, not training.
    # Ranges mirror givenData / the CJ rules: W,L 0.17-0.32, H 0.13-0.26, mass 0.5-6.0.
    random_boxes: bool = True
    box_seed: int = 0
    box_wl_range: tuple[float, float] = (0.17, 0.32)
    box_h_range: tuple[float, float] = (0.13, 0.26)
    box_mass_range: tuple[float, float] = (0.5, 6.0)

    # Per-episode diversity. The box POOL geometry is fixed at scene setup (Isaac
    # clones share collision geometry across envs, so per-env/per-episode resizing
    # is not batch-friendly). Instead each episode draws a fresh random ORDER over
    # the pool — which is exactly the competition's "random order" supply and needs
    # no prim rescaling. Disable for a fully deterministic single sequence.
    shuffle_each_episode: bool = True
    pallet_size: tuple[float, float, float] = (1.2, 1.0, 1.25)
    pallet_thickness: float = 0.15
    hidden_x: float = -100.0
    hidden_y: float = -100.0
    hidden_z: float = 0.5
    hidden_spacing: float = 0.75
    drift_fail_threshold: float = 0.40
    tilt_fail_threshold: float = 0.35
    out_of_bounds_margin: float = 0.02
    height_fail_margin: float = 0.005
    drop_fail_threshold: float = 0.08

    # --- A1: explicit settle (velocity-gated) ---------------------------------
    # After a box is placed, step physics until the fastest active box is below
    # settle_vel_threshold (or settle_max_steps is reached) BEFORE reading drift.
    # Early-exits when already at rest, so a settled stack costs ~1 extra substep.
    # Set settle_max_steps = 0 to disable (falls back to the decimation window).
    # NOTE: tune on the Isaac machine — defaults are conservative starting points.
    settle_max_steps: int = 24
    settle_vel_threshold: float = 0.05  # m/s and rad/s (linear & angular share it)

    # --- A2: cumulative stack stability --------------------------------------
    # Every step we also measure how far ALL previously-placed boxes drifted from
    # their intended (packer-resolved) pose. If any exceeds this, the new box
    # toppled the stack -> collapse termination (done reason 8).
    stack_drift_fail_threshold: float = 0.12

    # Mirrors Online-3D-BPP-PCT/PctContinuous0 reward terms, with Isaac fail checks added after placement.
    floor_coverage_reward_scale: float = 1.0
    boundary_floor_reward_scale: float = 0.8
    corner_floor_reward_scale: float = 0.6
    height_smoothness_reward_scale: float = 0.5
    support_reward_scale: float = 0.05
    weak_support_penalty_scale: float = 0.05
    weak_support_threshold: float = 0.85
    elevation_penalty_scale: float = 0.0  # #4 density knob (fill-bottom-first); 0=off, tune on Isaac
    physics_fail_penalty: float = -10.0
    invalid_action_penalty: float = -10.0
    no_feasible_leaf_reward: float = 0.0
    # Competition scoring: a FAILED pallet (out-of-bounds / height / drop / collapse,
    # i.e. done reasons 2,3,4,6,7,8) is worth 0 — all its stacked volume is lost. So
    # on a physical fail we zero the WHOLE episode's accumulated reward instead of a
    # flat -10. (physics_fail_penalty is then unused; set this False to restore -10.)
    fail_zeroes_pallet: bool = True


class PalletPackingEnv(DirectRLEnv):
    cfg: PalletPackingEnvCfg

    def __init__(self, cfg: PalletPackingEnvCfg, render_mode: str | None = None, **kwargs):
        self.pct_cfg = _load_yaml(cfg.pct_config_path)
        if cfg.random_boxes:
            self.boxes = _generate_random_boxes(
                cfg.max_boxes, cfg.box_seed, cfg.box_wl_range, cfg.box_h_range, cfg.box_mass_range
            )
        else:
            self.boxes = _load_boxes(cfg.box_sequence_path, cfg.max_boxes)
        self.pct_setting = int(self.pct_cfg["setting"])
        self.density_max = float(self.pct_cfg.get("density_max", 1.0))
        self.internal_node_holder = int(self.pct_cfg["internal_node_holder"])
        self.leaf_node_holder = int(self.pct_cfg["leaf_node_holder"])
        self.obs_node_count = self.internal_node_holder + self.leaf_node_holder + 1
        self.pct_obs_dim = self.obs_node_count * 9
        self.physics_feature_dim = int(cfg.physics_feature_dim)
        cfg.observation_space = self.pct_obs_dim + self.physics_feature_dim
        cfg.action_space = self.leaf_node_holder

        # Ensure the packer driver is importable here and (via spawn-inherited
        # sys.path) in worker processes.
        sys.path.insert(0, str(TEMPLATE_DIR))

        self.current_box_idx = [0 for _ in range(cfg.scene.num_envs)]
        # Per-env supply order over the fixed box pool (identity until shuffled on
        # reset). logical step s -> pool/asset index box_order[env][s].
        self.box_order: list[np.ndarray] = [
            np.arange(len(self.boxes), dtype=np.int64) for _ in range(cfg.scene.num_envs)
        ]
        self._episode_count = [0 for _ in range(cfg.scene.num_envs)]
        # Main-side caches mirroring what the (possibly remote) packers hold, so
        # _update_physics_metrics never has to reach into a worker's packer.
        self.packed_records: list[list[list[float]]] = [[] for _ in range(cfg.scene.num_envs)]
        self.last_ratio: list[float] = [0.0 for _ in range(cfg.scene.num_envs)]
        self.last_obs_np = [np.zeros((self.obs_node_count, 9), dtype=np.float32) for _ in range(cfg.scene.num_envs)]
        self.pending_actions: torch.Tensor | None = None
        self.action_mask: torch.Tensor | None = None
        self.physics_features: torch.Tensor | None = None
        self.last_reward: torch.Tensor | None = None
        self.episode_reward_sum: torch.Tensor | None = None
        self.last_terminated: torch.Tensor | None = None
        self.last_drift: torch.Tensor | None = None
        self.last_tilt: torch.Tensor | None = None
        self.last_out_of_bounds: torch.Tensor | None = None
        self.last_invalid: torch.Tensor | None = None
        self.last_done_reason: torch.Tensor | None = None
        self.last_stack_drift: torch.Tensor | None = None
        self.terminal_done_reason: torch.Tensor | None = None
        self.terminal_drift: torch.Tensor | None = None
        self.terminal_tilt: torch.Tensor | None = None
        self.terminal_height_ratio: torch.Tensor | None = None
        self.terminal_stack_drift: torch.Tensor | None = None
        self.box_assets: list[RigidObject] = []
        self._reward_scales = pct_reward.RewardScales(
            floor_coverage=cfg.floor_coverage_reward_scale,
            boundary_floor=cfg.boundary_floor_reward_scale,
            corner_floor=cfg.corner_floor_reward_scale,
            height_smoothness=cfg.height_smoothness_reward_scale,
            support=cfg.support_reward_scale,
            weak_support=cfg.weak_support_penalty_scale,
            weak_support_threshold=cfg.weak_support_threshold,
            elevation_penalty=cfg.elevation_penalty_scale,
        )
        self._packer_config = packer_pool.PackerConfig(
            pallet_size=tuple(cfg.pallet_size),
            size_minimum=float(self.pct_cfg["size_minimum"]),
            internal_node_holder=self.internal_node_holder,
            leaf_node_holder=self.leaf_node_holder,
            setting=self.pct_setting,
            density_max=self.density_max,
            scales=self._reward_scales,
        )
        self.packer_pool = packer_pool.make_packer_pool(
            cfg.scene.num_envs, self._packer_config, cfg.num_packer_workers
        )

        super().__init__(cfg, render_mode, **kwargs)

        self.action_mask = torch.zeros(self.num_envs, self.leaf_node_holder, dtype=torch.bool, device=self.device)
        self.physics_features = torch.zeros(
            self.num_envs, self.physics_feature_dim, dtype=torch.float32, device=self.device
        )
        self.last_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_drift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_tilt = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_done_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_stack_drift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.terminal_done_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terminal_drift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.terminal_tilt = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.terminal_height_ratio = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.terminal_stack_drift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _hidden_local_position(self, box_idx: int) -> tuple[float, float, float]:
        return (
            self.cfg.hidden_x - self.cfg.hidden_spacing * float(box_idx),
            self.cfg.hidden_y,
            self.cfg.hidden_z,
        )

    def _asset_index(self, env_id: int, logical_step: int) -> int:
        """Map an env's logical placement step to a pool/box-asset index via its
        per-episode supply order."""
        return int(self.box_order[env_id][logical_step])

    def close(self):
        pool = getattr(self, "packer_pool", None)
        if pool is not None:
            pool.close()
        return super().close()

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        pallet_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.pallet_size[0], self.cfg.pallet_size[1], self.cfg.pallet_thickness),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
        )
        pallet_cfg.func(
            "/World/envs/env_0/pallet",
            pallet_cfg,
            translation=(
                self.cfg.pallet_size[0] / 2.0,
                self.cfg.pallet_size[1] / 2.0,
                self.cfg.pallet_thickness / 2.0,
            ),
        )

        for idx, box in enumerate(self.boxes):
            size = tuple(float(v) for v in box["size"])
            mass = float(box["mass"])
            box_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/box_{idx:03d}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=3.0,
                        angular_damping=6.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6,
                        dynamic_friction=0.5,
                        restitution=0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.80, 0.20)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=self._hidden_local_position(idx)),
            )
            box_asset = RigidObject(cfg=box_cfg)
            self.box_assets.append(box_asset)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        for idx, box_asset in enumerate(self.box_assets):
            self.scene.rigid_objects[f"box_{idx:03d}"] = box_asset

    def _pre_physics_step(self, actions: torch.Tensor):
        self.pending_actions = actions.clone().reshape(self.num_envs, -1)[:, 0].to(torch.long)

    def _apply_action(self):
        if self.pending_actions is None:
            return

        actions = self.pending_actions
        self.pending_actions = None
        self.last_reward.zero_()
        self.last_terminated.zero_()
        self.last_drift.zero_()
        self.last_tilt.zero_()
        self.last_out_of_bounds.zero_()
        self.last_invalid.zero_()
        self.last_done_reason.zero_()

        # Build per-env CPU requests; envs past the sequence are terminal.
        requests: dict[int, tuple] = {}
        for env_id in range(self.num_envs):
            if self.current_box_idx[env_id] >= len(self.boxes):
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 5
                continue
            box = self.boxes[self._asset_index(env_id, self.current_box_idx[env_id])]
            requests[env_id] = (box, int(actions[env_id].item()))

        # All per-env CPU work (observe -> select -> place -> reward) happens here,
        # serially or across worker processes depending on num_packer_workers.
        results = self.packer_pool.step(requests)

        for env_id, result in results.items():
            box_idx = self._asset_index(env_id, self.current_box_idx[env_id])
            status = result["status"]

            if status == "invalid":
                self.last_invalid[env_id] = True
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 1
                no_feasible_leaf = int(result["valid_count"]) <= 0
                self.last_reward[env_id] = (
                    self.cfg.no_feasible_leaf_reward if no_feasible_leaf else self.cfg.invalid_action_penalty
                )
                continue
            if status == "place_failed":
                self.last_invalid[env_id] = True
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 1
                self.last_reward[env_id] = self.cfg.invalid_action_penalty
                continue

            # ok: spawn at the packer-RESOLVED resting pose (packed lx,ly,lz), not
            # the raw EMS leaf z (issue ④). The worker returns the same packer
            # record the drift reference uses, so spawn intent == drift reference.
            self.packed_records[env_id] = result["packed_all"]
            self.last_ratio[env_id] = float(result["ratio"])
            intended_world = self._intended_world(result["packed"], env_id)
            quat = torch.tensor(_yaw_quat_wxyz(result["rotation"]), dtype=torch.float32, device=self.device)

            root_pose = torch.cat((intended_world, quat), dim=0).reshape(1, 7)
            root_vel = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
            self.box_assets[box_idx].write_root_pose_to_sim(root_pose, env_ids=torch.tensor([env_id], device=self.device))
            self.box_assets[box_idx].write_root_velocity_to_sim(root_vel, env_ids=torch.tensor([env_id], device=self.device))

            self.current_box_idx[env_id] += 1
            self.last_reward[env_id] = float(result["reward"])

    def _get_observations(self) -> dict:
        pct_obs = torch.zeros((self.num_envs, self.pct_obs_dim), dtype=torch.float32, device=self.device)
        obs_req = {
            env_id: self.boxes[self._asset_index(env_id, self.current_box_idx[env_id])]
            for env_id in range(self.num_envs)
            if self.current_box_idx[env_id] < len(self.boxes)
        }
        obs_out = self.packer_pool.observe(obs_req) if obs_req else {}
        for env_id in range(self.num_envs):
            obs_np = obs_out.get(env_id)
            if obs_np is None:
                obs_np = np.zeros((self.obs_node_count, 9), dtype=np.float32)
            self.last_obs_np[env_id] = obs_np.astype(np.float32, copy=True)
            pct_obs[env_id] = torch.from_numpy(self.last_obs_np[env_id].reshape(-1)).to(self.device)

        obs_nodes = pct_obs.reshape(self.num_envs, self.obs_node_count, 9)
        _, _, _, valid_flag, full_mask = _pct_decode_observation(
            obs_nodes, self.internal_node_holder, self.leaf_node_holder
        )
        action_mask = valid_flag.bool()
        leaf_node_mask = 1 - valid_flag
        self.action_mask = action_mask
        self.extras["action_mask"] = action_mask
        self.extras["pct_valid_flag"] = valid_flag
        self.extras["pct_leaf_node_mask"] = leaf_node_mask
        self.extras["pct_full_mask"] = full_mask
        physics_features = self.physics_features
        if physics_features is None:
            physics_features = torch.zeros(
                self.num_envs, self.physics_feature_dim, dtype=torch.float32, device=self.device
            )
        self.extras["physics_features"] = physics_features
        self.extras["pct_obs"] = pct_obs
        return {"policy": torch.cat((pct_obs, physics_features), dim=1)}

    # done reasons that count as a competition FAIL (pallet -> 0 points).
    _FAIL_REASONS = (2, 3, 4, 6, 7, 8)

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.fail_zeroes_pallet:
            fail = torch.zeros_like(self.last_terminated)
            for reason in self._FAIL_REASONS:
                fail |= self.last_done_reason == reason
            # Cancel everything earned this episode so the failed pallet nets 0
            # (the failing step's shaped reward is dropped too).
            self.last_reward = torch.where(fail, -self.episode_reward_sum, self.last_reward)
        self.episode_reward_sum = self.episode_reward_sum + self.last_reward
        return self.last_reward.clone()

    def _settle_boxes(self) -> None:
        """A1: step physics until the fastest active box is at rest (or max_steps).

        Runs AFTER the placement substeps so drift is measured on a settled stack
        rather than mid-motion. Velocity-gated, so an already-stable stack exits
        after a single substep. No-op when settle_max_steps <= 0.
        """
        max_steps = int(self.cfg.settle_max_steps)
        if max_steps <= 0 or not self.box_assets:
            return
        dt = self.sim.get_physics_dt()
        vel_thr = float(self.cfg.settle_vel_threshold)
        for _ in range(max_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt)
            max_speed = 0.0
            for box_asset in self.box_assets:
                lin = float(torch.linalg.norm(box_asset.data.root_lin_vel_w, dim=-1).max().item())
                ang = float(torch.linalg.norm(box_asset.data.root_ang_vel_w, dim=-1).max().item())
                max_speed = max(max_speed, lin, ang)
            if max_speed < vel_thr:
                break

    def _intended_world(self, packed, env_id: int) -> torch.Tensor:
        """Packer-resolved center of a placed box, in world coords for this env."""
        x, y, z, lx, ly, lz, _ = [float(v) for v in packed]
        return torch.tensor(
            [lx + x / 2.0, ly + y / 2.0, lz + z / 2.0 + self.cfg.pallet_thickness],
            dtype=torch.float32,
            device=self.device,
        ) + self.scene.env_origins[env_id]

    def _update_physics_metrics(self) -> None:
        self._settle_boxes()
        self.last_stack_drift.zero_()
        for env_id in range(self.num_envs):
            placed_idx = self.current_box_idx[env_id] - 1
            if placed_idx < 0 or placed_idx >= len(self.box_assets) or self.last_invalid[env_id]:
                continue
            box_asset = self.box_assets[self._asset_index(env_id, placed_idx)]
            final_pos = box_asset.data.root_pos_w[env_id]
            final_quat = box_asset.data.root_quat_w[env_id]
            packed = self.packed_records[env_id][-1]
            x, y, z, lx, ly, lz, _ = [float(v) for v in packed]
            intended = torch.tensor(
                [lx + x / 2.0, ly + y / 2.0, lz + z / 2.0 + self.cfg.pallet_thickness],
                dtype=torch.float32,
                device=self.device,
            ) + self.scene.env_origins[env_id]
            drift = torch.linalg.norm(final_pos - intended)
            roll, pitch = _quat_wxyz_to_roll_pitch(final_quat)
            tilt = torch.sqrt(roll.square() + pitch.square())
            rotation_matrix = _quat_wxyz_to_matrix(final_quat)
            half_size = torch.tensor([x / 2.0, y / 2.0, z / 2.0], dtype=torch.float32, device=self.device)
            world_half_extent = torch.abs(rotation_matrix) @ half_size

            local_pos = final_pos - self.scene.env_origins[env_id]
            oob = (
                (local_pos[0] - world_half_extent[0] < -self.cfg.out_of_bounds_margin)
                | (local_pos[1] - world_half_extent[1] < -self.cfg.out_of_bounds_margin)
                | (local_pos[0] + world_half_extent[0] > self.cfg.pallet_size[0] + self.cfg.out_of_bounds_margin)
                | (local_pos[1] + world_half_extent[1] > self.cfg.pallet_size[1] + self.cfg.out_of_bounds_margin)
            )
            top_height = local_pos[2] + world_half_extent[2] - self.cfg.pallet_thickness
            height_oob = top_height > self.cfg.pallet_size[2] + self.cfg.height_fail_margin
            dropped = intended[2] - final_pos[2] > self.cfg.drop_fail_threshold

            self.last_drift[env_id] = drift
            self.last_tilt[env_id] = tilt
            self.last_out_of_bounds[env_id] = oob
            physics_failed = (
                drift > self.cfg.drift_fail_threshold
                or tilt > self.cfg.tilt_fail_threshold
                or bool(oob.item())
                or bool(height_oob.item())
                or bool(dropped.item())
            )
            if physics_failed and not self.cfg.fail_zeroes_pallet:
                self.last_reward[env_id] += self.cfg.physics_fail_penalty
            if drift > self.cfg.drift_fail_threshold:
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 2
            if tilt > self.cfg.tilt_fail_threshold:
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 3
            if bool(oob.item()):
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 4
            if bool(height_oob.item()):
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 6
            if bool(dropped.item()):
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 7

            # A2: cumulative stack stability — did placing this box disturb any
            # EARLIER box? The current box's own drift is already covered above
            # (reason 2), so here we scan only the boxes settled before it.
            max_stack_drift = 0.0
            for k in range(placed_idx):
                if k >= len(self.box_assets):
                    break
                intended_k = self._intended_world(self.packed_records[env_id][k], env_id)
                pos_k = self.box_assets[self._asset_index(env_id, k)].data.root_pos_w[env_id]
                drift_k = float(torch.linalg.norm(pos_k - intended_k).item())
                max_stack_drift = max(max_stack_drift, drift_k)
            self.last_stack_drift[env_id] = max_stack_drift
            if max_stack_drift > self.cfg.stack_drift_fail_threshold and not self.last_terminated[env_id]:
                if not self.cfg.fail_zeroes_pallet:
                    self.last_reward[env_id] += self.cfg.physics_fail_penalty
                self.last_terminated[env_id] = True
                self.last_done_reason[env_id] = 8

            ratio = float(self.last_ratio[env_id])
            height_ratio = min(max(float(top_height.item()), 0.0) / max(self.cfg.pallet_size[2], 1e-6), 1.0)
            self.physics_features[env_id] = torch.tensor(
                [
                    min(float(drift.item()) / max(self.cfg.drift_fail_threshold, 1e-6), 10.0),
                    min(abs(float(roll.item())) / math.pi, 1.0),
                    min(abs(float(pitch.item())) / math.pi, 1.0),
                    ratio,
                    height_ratio,
                ],
                dtype=torch.float32,
                device=self.device,
            )

    # done_reason codes: 1=invalid/no-feasible-leaf, 2=drift, 3=tilt, 4=out-of-bounds,
    # 5=sequence completed, 6=height exceeded, 7=dropped, 8=stack collapse (A2).
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_physics_metrics()
        completed = torch.tensor(
            [idx >= len(self.boxes) for idx in self.current_box_idx],
            dtype=torch.bool,
            device=self.device,
        )
        if self.last_done_reason is not None:
            self.last_done_reason[completed] = 5
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.last_terminated | completed
        reset_mask = terminated | time_out
        height_ratio = self.physics_features[:, 4] if self.physics_features is not None else self.last_drift * 0.0
        self.terminal_done_reason[:] = 0
        self.terminal_drift[:] = 0.0
        self.terminal_tilt[:] = 0.0
        self.terminal_height_ratio[:] = 0.0
        self.terminal_stack_drift[:] = 0.0
        self.terminal_done_reason[reset_mask] = self.last_done_reason[reset_mask]
        self.terminal_drift[reset_mask] = self.last_drift[reset_mask]
        self.terminal_tilt[reset_mask] = self.last_tilt[reset_mask]
        self.terminal_height_ratio[reset_mask] = height_ratio[reset_mask]
        self.terminal_stack_drift[reset_mask] = self.last_stack_drift[reset_mask]
        self.extras["terminal_done_reason"] = self.terminal_done_reason.clone()
        self.extras["terminal_drift"] = self.terminal_drift.clone()
        self.extras["terminal_tilt"] = self.terminal_tilt.clone()
        self.extras["terminal_height_ratio"] = self.terminal_height_ratio.clone()
        self.extras["terminal_stack_drift"] = self.terminal_stack_drift.clone()
        self.extras["last_stack_drift"] = self.last_stack_drift.clone()
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        env_ids = list(int(i) for i in env_ids)
        super()._reset_idx(env_ids)

        self.packer_pool.reset(env_ids)
        for env_id in env_ids:
            self.current_box_idx[env_id] = 0
            self.last_obs_np[env_id][:] = 0.0
            self.packed_records[env_id] = []
            self.last_ratio[env_id] = 0.0
            # Fresh random supply order for this episode (competition feeds boxes in
            # random order). Seeded by (box_seed, env_id, episode) for reproducibility.
            if self.cfg.shuffle_each_episode:
                self._episode_count[env_id] += 1
                rng = np.random.default_rng(
                    (self.cfg.box_seed, env_id, self._episode_count[env_id])
                )
                self.box_order[env_id] = rng.permutation(len(self.boxes))

        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        hidden_vel = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device)
        for box_idx, box_asset in enumerate(self.box_assets):
            hidden_pose = torch.zeros((len(env_ids), 7), dtype=torch.float32, device=self.device)
            hidden_pose[:, :3] = self.scene.env_origins[env_ids_tensor]
            hidden_pose[:, 0] += self.cfg.hidden_x - self.cfg.hidden_spacing * float(box_idx)
            hidden_pose[:, 1] += self.cfg.hidden_y
            hidden_pose[:, 2] += self.cfg.hidden_z
            hidden_pose[:, 3] = 1.0
            box_asset.write_root_pose_to_sim(hidden_pose, env_ids=env_ids_tensor)
            box_asset.write_root_velocity_to_sim(hidden_vel, env_ids=env_ids_tensor)
            box_asset.reset(env_ids_tensor)

        if self.last_reward is not None:
            self.last_reward[env_ids_tensor] = 0.0
            self.episode_reward_sum[env_ids_tensor] = 0.0
            self.last_terminated[env_ids_tensor] = False
            self.last_drift[env_ids_tensor] = 0.0
            self.last_tilt[env_ids_tensor] = 0.0
            self.last_out_of_bounds[env_ids_tensor] = False
            self.physics_features[env_ids_tensor] = 0.0
            self.last_invalid[env_ids_tensor] = False
            self.last_done_reason[env_ids_tensor] = 0
            self.last_stack_drift[env_ids_tensor] = 0.0
