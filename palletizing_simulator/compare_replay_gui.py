from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import yaml


HERE = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(HERE, "config/sim_config_submit_eval10_slow_gui.yaml")
DEFAULT_LEFT_DIR = os.path.join(HERE, "../submit_buffer3_search/algorithm_results_eval10/1400")
DEFAULT_RIGHT_DIR = os.path.join(HERE, "../submit_buffer3_search/algorithm_results_eval10/1450")
DEFAULT_BOX_SEQUENCE_DIR = os.path.join(HERE, "../submit_buffer3_search/random_spec_eval_10/box_sequence")


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(HERE, path))


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_boxes(path: str) -> dict[int, dict]:
    boxes: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for key in ("id", "size", "mass"):
                if key not in item:
                    raise KeyError(f"Missing key '{key}' in {path}:{line_no}")
            bid = int(item["id"])
            if bid in boxes:
                raise ValueError(f"Duplicated box id={bid} in {path}")
            boxes[bid] = item
    return boxes


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--config", default=DEFAULT_CONFIG)
pre_args, _ = pre.parse_known_args()
cfg = load_config(pre_args.config)

parser = argparse.ArgumentParser(description="Side-by-side Isaac Sim replay for two result folders")
parser.add_argument("--config", default=DEFAULT_CONFIG)
parser.add_argument("--left-dir", default=DEFAULT_LEFT_DIR)
parser.add_argument("--right-dir", default=DEFAULT_RIGHT_DIR)
parser.add_argument("--box-sequence-dir", default=DEFAULT_BOX_SEQUENCE_DIR)
parser.add_argument("--left-label", default="1400")
parser.add_argument("--right-label", default="1450")
parser.add_argument("--limit", type=int, default=10)
parser.add_argument("--episode-delay", type=float, default=2.0)
parser.add_argument("--step-delay", type=float, default=cfg.get("runtime", {}).get("step_delay_sec", 0.4))
parser.add_argument("--settle-steps", type=int, default=cfg["settling"]["max_steps"])
parser.add_argument("--settle-vel", type=float, default=cfg["settling"]["velocity_threshold"])
parser.add_argument("--final-steps", type=int, default=cfg["settling"]["final_steps"])
parser.add_argument("--drop-offset", type=float, default=cfg["settling"]["drop_offset"])
parser.add_argument("--hold-final", type=float, default=0.0)
args, _ = parser.parse_known_args()


from isaacsim import SimulationApp  # noqa: E402

app_cfg = cfg["app"]
sim_config = {
    "experience": app_cfg["experience"],
    "width": app_cfg["width"],
    "height": app_cfg["height"],
    "window_width": app_cfg["width"],
    "window_height": app_cfg["height"],
    "headless": False,
    "hide_ui": app_cfg.get("hide_ui", False),
    "renderer": app_cfg.get("renderer", "RayTracedLighting"),
    "display_options": app_cfg.get("display_options", 3286),
}

simulation_app = SimulationApp(sim_config)

import carb  # noqa: E402
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics  # noqa: E402

for ext in app_cfg.get("extensions", []):
    try:
        enable_extension(ext)
    except Exception as exc:
        print(f"[compare] Warning: could not enable {ext}: {exc}", flush=True)

simulation_app.update()


@dataclass
class SideState:
    label: str
    prim_prefix: str
    offset: tuple[float, float, float]
    color: tuple[float, float, float]
    boxes_by_id: dict[int, dict]
    sequence: list[dict]
    placed_pairs: list[tuple[DynamicCuboid, float]]


class CompareMonitor:
    def __init__(self) -> None:
        self.window = ui.Window("Compare Replay", width=430, height=190)
        with self.window.frame:
            with ui.VStack(spacing=7):
                ui.Label("Side-by-side replay")
                self.episode = ui.Label("")
                self.step = ui.Label("")
                self.left = ui.Label("")
                self.right = ui.Label("")
                self.note = ui.Label("")

    def update(self, episode: str, step: int, total: int, left_id: int | None, right_id: int | None) -> None:
        self.episode.text = f"episode: {episode}"
        self.step.text = f"step: {step}/{total}"
        self.left.text = f"left  ({args.left_label}): id={left_id if left_id is not None else '-'}"
        self.right.text = f"right ({args.right_label}): id={right_id if right_id is not None else '-'}"
        self.note.text = "Close Isaac Sim window to stop."


monitor = CompareMonitor()


def rotation_quat(degrees: float) -> np.ndarray:
    rad = math.radians(degrees)
    return np.array([math.cos(rad / 2.0), 0.0, 0.0, math.sin(rad / 2.0)])


def rotated_half(size: list[float] | tuple[float, float, float], rot_deg: float) -> list[float]:
    hx, hy, hz = float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0
    return [hy, hx, hz] if abs(rot_deg % 180.0 - 90.0) < 1.0 else [hx, hy, hz]


def make_physics_material(path: str, section: str) -> PhysicsMaterial:
    mat = cfg["physics"][section]
    return PhysicsMaterial(
        prim_path=path,
        static_friction=mat["static_friction"],
        dynamic_friction=mat["dynamic_friction"],
        restitution=mat["restitution"],
    )


def setup_lighting() -> None:
    stage = omni.usd.get_context().get_stage()
    stage.RemovePrim("/World/Lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(500.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(650.0)
    distant.CreateAngleAttr(0.53)
    distant.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 5.0))
    distant.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))


def build_scene(world: World, pallet_size: list[float], left_offset: tuple[float, float, float], right_offset: tuple[float, float, float]) -> float:
    setup_lighting()
    scene = world.scene
    pallet_lx, pallet_ly = pallet_size[0], pallet_size[1]
    pallet_cfg = cfg["physics"]["pallet"]
    pallet_thickness = float(pallet_cfg["thickness"])

    ground_mat = make_physics_material("/World/Physics/ground_mat", "ground")
    scene.add(GroundPlane(
        prim_path="/World/GroundPlane",
        name="ground_plane",
        z_position=0.0,
        physics_material=ground_mat,
    ))

    pallet_mat_left = make_physics_material("/World/Physics/pallet_mat_left", "pallet")
    pallet_mat_right = make_physics_material("/World/Physics/pallet_mat_right", "pallet")
    for prim_prefix, label, offset, color, mat in (
        ("left", args.left_label, left_offset, (0.40, 0.55, 0.25), pallet_mat_left),
        ("right", args.right_label, right_offset, (0.25, 0.42, 0.70), pallet_mat_right),
    ):
        ox, oy, _ = offset
        pallet = FixedCuboid(
            prim_path=f"/World/{prim_prefix}_pallet",
            name=f"{prim_prefix}_pallet_{label}",
            position=np.array([ox + pallet_lx / 2.0, oy + pallet_ly / 2.0, pallet_thickness / 2.0]),
            scale=np.array([pallet_lx, pallet_ly, pallet_thickness]),
            size=1.0,
            color=np.array(color),
            physics_material=mat,
        )
        pallet.set_contact_offset(pallet_cfg["contact_offset"])
        pallet.set_rest_offset(pallet_cfg["rest_offset"])
        scene.add(pallet)

    return pallet_thickness


def create_cube(world: World, side: SideState, step_idx: int, bid: int, size: list[float], mass: float, rot: float, target_xyz: list[float]) -> DynamicCuboid:
    box_cfg = cfg["physics"]["box"]
    mat = make_physics_material(f"/World/Physics/box_mat_{side.prim_prefix}_{step_idx:03d}_{bid:03d}", "box")
    cube = DynamicCuboid(
        prim_path=f"/World/{side.prim_prefix}_box_{step_idx:03d}_{bid:03d}",
        name=f"{side.prim_prefix}_box_{step_idx:03d}_{bid:03d}",
        position=np.array(target_xyz, dtype=float),
        orientation=rotation_quat(rot),
        scale=np.array(size, dtype=float),
        size=1.0,
        mass=float(mass),
        color=np.array(side.color),
        physics_material=mat,
    )
    cube.set_contact_offset(box_cfg["contact_offset"])
    cube.set_rest_offset(box_cfg["rest_offset"])
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.prim)
    physx_rb.CreateLinearDampingAttr().Set(box_cfg["linear_damping"])
    physx_rb.CreateAngularDampingAttr().Set(box_cfg["angular_damping"])
    world.scene.add(cube)
    return cube


def xy_overlap_area(center_a, half_a, center_b, half_b) -> float:
    dx = min(center_a[0] + half_a[0], center_b[0] + half_b[0]) - max(center_a[0] - half_a[0], center_b[0] - half_b[0])
    dy = min(center_a[1] + half_a[1], center_b[1] + half_b[1]) - max(center_a[1] - half_a[1], center_b[1] - half_b[1])
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(dx * dy)


def has_min_support(target_xyz: list[float], size: list[float], rot_deg: float, placed_pairs: list[tuple[DynamicCuboid, float]], floor_z: float) -> bool:
    new_half = rotated_half(size, rot_deg)
    bottom_z = float(target_xyz[2] - new_half[2])
    base_area = max(float((new_half[0] * 2.0) * (new_half[1] * 2.0)), 1e-6)

    if abs(bottom_z - floor_z) <= 0.01:
        return True

    support_area = 0.0
    for cube, c_rot in placed_pairs:
        c_pos, _ = cube.get_world_pose()
        c_scale = cube.get_local_scale()
        c_half = rotated_half(list(c_scale), c_rot)
        top_z = float(c_pos[2] + c_half[2])
        if abs(bottom_z - top_z) <= 0.01:
            support_area += xy_overlap_area(
                [float(target_xyz[0]), float(target_xyz[1])],
                new_half,
                [float(c_pos[0]), float(c_pos[1])],
                c_half,
            )
    return support_area / base_area >= 0.30


def safe_spawn_z(target_xy: list[float], size: list[float], rot_deg: float, placed_pairs: list[tuple[DynamicCuboid, float]], floor_z: float) -> float:
    new_half = rotated_half(size, rot_deg)
    max_top_z = floor_z
    for cube, c_rot in placed_pairs:
        c_pos, _ = cube.get_world_pose()
        c_scale = cube.get_local_scale()
        c_half = rotated_half(list(c_scale), c_rot)
        if abs(c_pos[0] - target_xy[0]) < new_half[0] + c_half[0] and abs(c_pos[1] - target_xy[1]) < new_half[1] + c_half[1]:
            max_top_z = max(max_top_z, float(c_pos[2]) + c_half[2])
    return max_top_z + new_half[2]


def set_kinematic(cube: DynamicCuboid, enabled: bool) -> None:
    rb_api = UsdPhysics.RigidBodyAPI.Apply(cube.prim)
    rb_api.CreateKinematicEnabledAttr().Set(bool(enabled))


def settle_all(world: World, cubes: list[DynamicCuboid]) -> None:
    min_frames = cfg["settling"]["min_frames"]
    for step in range(args.settle_steps):
        if not simulation_app.is_running():
            break
        world.step(render=True)
        if step < min_frames:
            continue
        settled = True
        for cube in cubes:
            lin_vel = np.linalg.norm(cube.get_linear_velocity())
            ang_vel = np.linalg.norm(cube.get_angular_velocity())
            if lin_vel >= args.settle_vel or ang_vel >= args.settle_vel:
                settled = False
                break
        if settled:
            break


def hold_render(world: World, seconds: float) -> None:
    if seconds <= 0.0:
        return
    end_time = time.monotonic() + seconds
    while simulation_app.is_running() and time.monotonic() < end_time:
        world.step(render=True)
        time.sleep(min(1.0 / 60.0, max(0.0, end_time - time.monotonic())))


def collect_episode_pairs(left_dir: str, right_dir: str) -> list[tuple[str, str]]:
    left_files = {os.path.basename(p): p for p in glob.glob(os.path.join(left_dir, "*.json"))}
    right_files = {os.path.basename(p): p for p in glob.glob(os.path.join(right_dir, "*.json"))}
    names = sorted(set(left_files).intersection(right_files))
    if not names:
        raise FileNotFoundError(f"No matching JSON files between {left_dir} and {right_dir}")
    if args.limit > 0:
        names = names[:args.limit]
    return [(left_files[name], right_files[name]) for name in names]


def prepare_side(label: str, prim_prefix: str, offset: tuple[float, float, float], color: tuple[float, float, float], result_path: str, boxes_by_id: dict[int, dict]) -> SideState:
    result = load_json(result_path)
    if "sequence" not in result or not isinstance(result["sequence"], list):
        raise ValueError(f"Result JSON must contain list 'sequence': {result_path}")
    return SideState(
        label=label,
        prim_prefix=prim_prefix,
        offset=offset,
        color=color,
        boxes_by_id=boxes_by_id,
        sequence=result["sequence"],
        placed_pairs=[],
    )


def target_for_step(side: SideState, step: dict, pallet_thickness: float) -> tuple[int, list[float], list[float], float, float]:
    if "id" not in step or "position" not in step:
        raise KeyError(f"Step missing id/position: {step}")
    bid = int(step["id"])
    pos = step["position"]
    if bid not in side.boxes_by_id:
        raise KeyError(f"id={bid} not found in matching box_sequence for {side.label}")
    if not isinstance(pos, list) or len(pos) != 3:
        raise ValueError(f"position must be [x, y, z]: {step}")

    box = side.boxes_by_id[bid]
    size = [float(x) for x in box["size"]]
    mass = float(box["mass"])
    rot = float(step.get("rotation", 0.0))
    ox, oy, _ = side.offset
    target = [ox + float(pos[0]), oy + float(pos[1]), float(pos[2]) + pallet_thickness]
    if not has_min_support(target, size, rot, side.placed_pairs, pallet_thickness):
        target = [target[0], target[1], safe_spawn_z(target[:2], size, rot, side.placed_pairs, pallet_thickness)]
    return bid, size, target, mass, rot


def replay_episode(left_path: str, right_path: str, box_sequence_dir: str) -> None:
    episode_name = os.path.basename(left_path)
    print(f"\n[compare] Episode {episode_name}", flush=True)

    usd_ctx = omni.usd.get_context()
    usd_ctx.new_stage()
    simulation_app.update()
    simulation_app.update()

    pallet_size = [float(x) for x in cfg["pallet"]["size"]]
    pallet_lx = pallet_size[0]
    gap = 1.1
    left_offset = (0.0, 0.0, 0.0)
    right_offset = (pallet_lx + gap, 0.0, 0.0)

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    pallet_thickness = build_scene(world, pallet_size, left_offset, right_offset)
    world.reset()

    box_sequence_path = os.path.join(box_sequence_dir, episode_name)
    boxes_by_id = load_jsonl_boxes(box_sequence_path)
    left = prepare_side(args.left_label, "left", left_offset, (0.18, 0.78, 0.30), left_path, boxes_by_id)
    right = prepare_side(args.right_label, "right", right_offset, (0.25, 0.55, 0.95), right_path, boxes_by_id)

    total_steps = max(len(left.sequence), len(right.sequence))
    print(
        f"[compare] {args.left_label}: {len(left.sequence)} boxes, "
        f"{args.right_label}: {len(right.sequence)} boxes",
        flush=True,
    )

    for idx in range(total_steps):
        if not simulation_app.is_running():
            break

        pending: list[tuple[SideState, DynamicCuboid, float]] = []
        ids: dict[str, int | None] = {args.left_label: None, args.right_label: None}

        for side in (left, right):
            if idx >= len(side.sequence):
                continue
            bid, size, target, mass, rot = target_for_step(side, side.sequence[idx], pallet_thickness)
            cube = create_cube(world, side, idx + 1, bid, size, mass, rot, target)
            set_kinematic(cube, True)
            cube.set_world_pose(position=np.array(target, dtype=float), orientation=rotation_quat(rot))
            pending.append((side, cube, rot))
            ids[side.label] = bid

        for _ in range(3):
            if simulation_app.is_running():
                world.step(render=True)

        for _, cube, _ in pending:
            set_kinematic(cube, False)

        if simulation_app.is_running():
            world.step(render=True)

        cubes = [cube for _, cube, _ in pending]
        for cube in cubes:
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        settle_all(world, cubes)

        for side, cube, rot in pending:
            side.placed_pairs.append((cube, rot))

        monitor.update(episode_name, idx + 1, total_steps, ids[args.left_label], ids[args.right_label])
        print(
            f"[compare] step {idx + 1:03d}/{total_steps:03d}  "
            f"{args.left_label}=id {ids[args.left_label]}  "
            f"{args.right_label}=id {ids[args.right_label]}",
            flush=True,
        )
        hold_render(world, args.step_delay)

    print(f"[compare] Final settle {args.final_steps} steps", flush=True)
    for _ in range(args.final_steps):
        if not simulation_app.is_running():
            break
        world.step(render=True)

    hold_render(world, args.episode_delay)

    world.clear()
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World").IsValid():
        stage.RemovePrim(Sdf.Path("/World"))
    simulation_app.update()
    simulation_app.update()


def main() -> None:
    left_dir = resolve_path(args.left_dir)
    right_dir = resolve_path(args.right_dir)
    box_sequence_dir = resolve_path(args.box_sequence_dir)

    print(f"[compare] Config           : {args.config}", flush=True)
    print(f"[compare] Left results     : {left_dir}", flush=True)
    print(f"[compare] Right results    : {right_dir}", flush=True)
    print(f"[compare] Box sequence dir : {box_sequence_dir}", flush=True)
    print(f"[compare] step_delay={args.step_delay}s episode_delay={args.episode_delay}s", flush=True)

    pairs = collect_episode_pairs(left_dir, right_dir)
    print(f"[compare] Matched episodes : {len(pairs)}", flush=True)
    for left_path, _ in pairs:
        print(f"  - {os.path.basename(left_path)}", flush=True)

    for left_path, right_path in pairs:
        if not simulation_app.is_running():
            break
        replay_episode(left_path, right_path, box_sequence_dir)

    if args.hold_final > 0:
        end_time = time.monotonic() + args.hold_final
        while simulation_app.is_running() and time.monotonic() < end_time:
            simulation_app.update()
            time.sleep(1.0 / 30.0)
    elif args.hold_final == 0:
        print("[compare] Replay complete. Keeping Isaac Sim open until the window is closed.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(1.0 / 30.0)

    simulation_app.close()


if __name__ == "__main__":
    main()
