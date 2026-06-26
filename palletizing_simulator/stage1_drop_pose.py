from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "templete code"


def _resolve_path(path: str | os.PathLike[str], base: Path = SIM_DIR) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return base / p


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_experience(cfg: dict) -> None:
    exp = Path(cfg["app"]["experience"])
    if exp.exists():
        return

    fallback = Path("/home/user/isaacsim/apps/isaacsim.exp.base.python.kit")
    if fallback.exists():
        cfg["app"]["experience"] = str(fallback)


def _load_first_box(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No boxes found in {path}")


def _load_pct_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _density_for_box(box: dict, pct_cfg: dict) -> float:
    if int(pct_cfg.get("setting", 1)) < 3:
        return 1.0

    sx, sy, sz = [float(v) for v in box["size"]]
    vol = max(sx * sy * sz, 1e-9)
    density_max = float(pct_cfg.get("density_max", 1.0))
    return (float(box["mass"]) / vol) / density_max


def _select_leaf(leaf_nodes: np.ndarray, index: int | None) -> np.ndarray:
    valid = leaf_nodes[leaf_nodes[:, 8] > 0.5]
    if len(valid) == 0:
        raise RuntimeError("Packer returned no feasible leaf nodes for the first box.")

    order = np.lexsort((valid[:, 1], valid[:, 0], valid[:, 2]))
    valid = valid[order]

    if index is None:
        return valid[0]
    if index < 0 or index >= len(valid):
        raise IndexError(f"leaf index {index} out of range for {len(valid)} valid leaves")
    return valid[index]


def _leaf_center_and_rotation(leaf: np.ndarray, box_size: list[float]) -> tuple[list[float], list[float], int]:
    lx, ly, lz, hx, hy, _ = [float(v) for v in leaf[:6]]
    placed_size = [round(hx - lx, 6), round(hy - ly, 6), 0.0]

    remaining = [0, 1, 2]
    for axis in list(remaining):
        if abs(placed_size[0] - float(box_size[axis])) < 1e-6:
            remaining.remove(axis)
            break
    for axis in list(remaining):
        if abs(placed_size[1] - float(box_size[axis])) < 1e-6:
            remaining.remove(axis)
            break
    placed_size[2] = float(box_size[remaining[0]])

    center_local = [
        lx + placed_size[0] / 2.0,
        ly + placed_size[1] / 2.0,
        lz + placed_size[2] / 2.0,
    ]
    rotation = 0 if (
        abs(placed_size[0] - float(box_size[0])) < 1e-6
        and abs(placed_size[1] - float(box_size[1])) < 1e-6
    ) else 90
    return center_local, placed_size, rotation


def _build_app_config(cfg: dict, *, headless: bool | None) -> dict:
    app_cfg = cfg["app"]
    app_headless = bool(app_cfg["headless"] if headless is None else headless)
    return {
        "experience": app_cfg["experience"],
        "width": app_cfg["width"],
        "height": app_cfg["height"],
        "window_width": app_cfg["width"],
        "window_height": app_cfg["height"],
        "headless": app_headless,
        "hide_ui": app_cfg["hide_ui"],
        "renderer": app_cfg["renderer"],
        "display_options": app_cfg["display_options"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Isaac Sim drop/pose proof for pallet packing.")
    parser.add_argument("--config", default=str(SIM_DIR / "config" / "sim_config.yaml"))
    parser.add_argument("--pct-config", default=str(TEMPLATE_DIR / "config" / "pct_config.yaml"))
    parser.add_argument("--box-sequence", default=str(SIM_DIR / "box_sequence" / "box_sequence_0.json"))
    parser.add_argument("--out", default=str(SIM_DIR / "stage1_results" / "stage1_drop_pose.json"))
    parser.add_argument("--leaf-index", type=int, default=None)
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--settle-vel", type=float, default=None)
    parser.add_argument("--drop-offset", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = _resolve_path(args.config)
    cfg = _load_yaml(cfg_path)
    cfg = copy.deepcopy(cfg)
    _resolve_experience(cfg)

    pct_cfg_path = _resolve_path(args.pct_config, ROOT)
    pct_cfg = _load_pct_config(pct_cfg_path)
    box = _load_first_box(_resolve_path(args.box_sequence))

    sys.path.insert(0, str(TEMPLATE_DIR))
    from src.pct.packer import Packer  # noqa: PLC0415

    packer = Packer(
        container_size=cfg["pallet"]["size"],
        size_minimum=float(pct_cfg["size_minimum"]),
        internal_node_holder=int(pct_cfg["internal_node_holder"]),
        leaf_node_holder=int(pct_cfg["leaf_node_holder"]),
        setting=int(pct_cfg["setting"]),
    )
    packer.reset()
    density = _density_for_box(box, pct_cfg)
    obs = packer.observe(box["size"], density=density)
    obs_arr = obs.reshape(-1, 9)
    leaf_nodes = obs_arr[
        int(pct_cfg["internal_node_holder"]): int(pct_cfg["internal_node_holder"]) + int(pct_cfg["leaf_node_holder"])
    ]
    leaf = _select_leaf(leaf_nodes, args.leaf_index)
    intended_local, placed_size, rotation = _leaf_center_and_rotation(leaf, box["size"])

    os.chdir(SIM_DIR)

    from isaacsim import SimulationApp  # noqa: PLC0415

    simulation_app = SimulationApp(_build_app_config(cfg, headless=args.headless))

    try:
        from isaacsim.core.api import World  # noqa: PLC0415

        import scene  # noqa: PLC0415

        scene.init(cfg)
        scene.reset_shared_box_mat()

        world = World(stage_units_in_meters=1.0)
        pallet_thickness, _ = scene.build_base_scene(
            world,
            cfg["pallet"]["size"],
            n_buffer_slots=0,
        )
        world.reset()

        settle_steps = int(args.settle_steps if args.settle_steps is not None else cfg["settling"]["max_steps"])
        settle_vel = float(args.settle_vel if args.settle_vel is not None else cfg["settling"]["velocity_threshold"])
        drop_offset = float(args.drop_offset if args.drop_offset is not None else cfg["settling"]["drop_offset"])

        intended_world = [
            intended_local[0],
            intended_local[1],
            intended_local[2] + pallet_thickness,
        ]

        cube = scene.spawn_on_pallet(
            world=world,
            bid=int(box["id"]),
            size=placed_size,
            mass=float(box["mass"]),
            rotation_deg=float(rotation),
            target_xyz_world=intended_world,
            placed_pairs=[],
            floor_z=float(pallet_thickness),
            simulation_app=simulation_app,
            settle_steps=settle_steps,
            settle_vel=settle_vel,
            drop_offset=drop_offset,
        )

        final_pos, final_quat = cube.get_world_pose()
        final_pos = [float(v) for v in final_pos]
        final_local = [
            final_pos[0],
            final_pos[1],
            final_pos[2] - pallet_thickness,
        ]
        drift = float(np.linalg.norm(np.array(final_pos) - np.array(intended_world)))
        drift_xy = float(np.linalg.norm(np.array(final_pos[:2]) - np.array(intended_world[:2])))
        drift_z = float(final_pos[2] - intended_world[2])

        result = {
            "box": box,
            "pct": {
                "setting": int(pct_cfg["setting"]),
                "density": float(density),
                "valid_leaf_count": int(np.sum(leaf_nodes[:, 8] > 0.5)),
                "selected_leaf": [float(v) for v in leaf[:9]],
            },
            "placement": {
                "placed_size": [float(v) for v in placed_size],
                "rotation": int(rotation),
                "intended_local": [float(v) for v in intended_local],
                "intended_world": [float(v) for v in intended_world],
                "final_local": final_local,
                "final_world": final_pos,
                "final_quat_wxyz": [float(v) for v in final_quat],
                "drift_m": drift,
                "drift_xy_m": drift_xy,
                "drift_z_m": drift_z,
            },
            "settling": {
                "settle_steps": settle_steps,
                "settle_vel": settle_vel,
                "drop_offset": drop_offset,
            },
        }

        out_path = _resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(json.dumps(result, indent=2), flush=True)
        print(f"[stage1] wrote {out_path}", flush=True)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
