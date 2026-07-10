"""
Create an Isaac Sim palletizing scene with continuous-random boxes.

This is a visual/manual-policy sandbox, not the competition evaluator.  It uses
the same pallet, buffer platform, physics materials, and box spawn helpers as
``palletizing_simulator/simulator.py``.  Boxes are sampled from continuous
uniform ranges and laid out on the buffer platform slots.

Example:
    python3 palletizing_simulator/random_pallet_scene.py --gui --num-boxes 40 --seed 7
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "sim_config.yaml"
DEFAULT_OUTPUT_DIR = HERE / "random_scene_outputs"
sys.path.insert(0, str(HERE))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def generate_random_boxes(
    num_boxes: int,
    seed: int,
    wl_range: tuple[float, float],
    h_range: tuple[float, float],
    mass_range: tuple[float, float],
) -> list[dict[str, Any]]:
    """Generate CJ-style continuous random boxes.

    Width/length and height are sampled independently from continuous uniform
    ranges.  This intentionally avoids a fixed discrete box-type set.
    """
    rng = np.random.default_rng(seed)
    boxes: list[dict[str, Any]] = []
    for i in range(num_boxes):
        width = round(float(rng.uniform(*wl_range)), 3)
        length = round(float(rng.uniform(*wl_range)), 3)
        height = round(float(rng.uniform(*h_range)), 3)
        mass = round(float(rng.uniform(*mass_range)), 3)
        boxes.append(
            {
                "step": i,
                "id": i,
                "type": -1,
                "type_name": "continuous_random",
                "size": [width, length, height],
                "mass": mass,
            }
        )
    return boxes


def write_jsonl(path: Path, boxes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for box in boxes:
            f.write(json.dumps(box, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spawn a random-box palletizing sandbox scene in Isaac Sim."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-boxes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wl-min", type=float, default=0.17)
    parser.add_argument("--wl-max", type=float, default=0.32)
    parser.add_argument("--h-min", type=float, default=0.13)
    parser.add_argument("--h-max", type=float, default=0.26)
    parser.add_argument("--mass-min", type=float, default=0.5)
    parser.add_argument("--mass-max", type=float, default=6.0)
    parser.add_argument("--settle-steps", type=int, default=24)
    parser.add_argument("--settle-vel", type=float, default=0.05)
    parser.add_argument("--drop-offset", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-usd", type=Path, default=None)
    parser.add_argument("--gui", action="store_true", help="Open Isaac Sim GUI instead of headless mode.")
    parser.add_argument(
        "--full-gui",
        action="store_true",
        help="Keep the config's full Isaac editor experience. By default --gui uses the lighter base.python experience.",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep the Isaac app running after spawning so the scene can be inspected.",
    )
    return parser.parse_args()


def _find_python_near_experience(experience: str) -> Path | None:
    """Find the Python executable that belongs to an Isaac Sim install/venv."""
    exp_path = Path(experience).expanduser()
    for parent in (exp_path.parent, *exp_path.parents):
        candidate = parent / "bin" / "python"
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _installed_isaacsim_apps_dir() -> Path | None:
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or spec.origin is None:
        return None
    apps_dir = Path(spec.origin).resolve().parent / "apps"
    return apps_dir if apps_dir.exists() else None


def _resolve_experience(experience: str) -> str:
    """Resolve stale config paths against the installed Isaac Sim package."""
    exp_path = Path(experience).expanduser()
    if exp_path.exists():
        return str(exp_path)

    apps_dir = _installed_isaacsim_apps_dir()
    if apps_dir is not None:
        same_name = apps_dir / exp_path.name
        if same_name.exists():
            print(
                "[random-scene] configured Isaac experience does not exist; "
                f"using installed experience {same_name}",
                flush=True,
            )
            return str(same_name)

        fallback = apps_dir / "isaacsim.exp.base.python.kit"
        if fallback.exists():
            print(
                "[random-scene] configured Isaac experience does not exist; "
                f"using fallback experience {fallback}",
                flush=True,
            )
            return str(fallback)

    return experience


def _base_python_experience(experience: str) -> str:
    """Return the lightweight Isaac Sim python experience near the configured app."""
    exp_path = Path(_resolve_experience(experience)).expanduser()
    for parent in (exp_path.parent, *exp_path.parents):
        candidate = parent / "isaacsim.exp.base.python.kit"
        if candidate.exists():
            return str(candidate)
    return experience


def ensure_isaacsim_python(cfg: dict[str, Any]) -> None:
    """Re-exec with Isaac Sim's Python if the current interpreter cannot import it."""
    if importlib.util.find_spec("isaacsim") is not None:
        return

    if os.environ.get("RANDOM_PALLET_SCENE_REEXEC") == "1":
        experience = str(cfg.get("app", {}).get("experience", ""))
        raise ModuleNotFoundError(
            "Could not import 'isaacsim' even after trying the Isaac Sim Python. "
            f"Check app.experience in the config: {experience}"
        )

    experience = str(cfg.get("app", {}).get("experience", ""))
    isaac_python = _find_python_near_experience(experience)
    if isaac_python is None:
        raise ModuleNotFoundError(
            "Could not import 'isaacsim' from the current Python, and no "
            f"Isaac Sim Python was found near app.experience={experience!r}. "
            "Run with the Isaac Sim Python directly, for example: "
            "/home/robotics/isaac-sim-5.1/bin/python "
            "palletizing_simulator/random_pallet_scene.py ..."
        )

    if Path(sys.executable).resolve() == isaac_python.resolve():
        raise

    env = dict(os.environ)
    env["RANDOM_PALLET_SCENE_REEXEC"] = "1"
    print(
        "[random-scene] current Python cannot import isaacsim; "
        f"restarting with {isaac_python}",
        flush=True,
    )
    os.execve(str(isaac_python), [str(isaac_python), *sys.argv], env)


def main() -> None:
    args = parse_args()
    if args.num_boxes < 1:
        raise ValueError("--num-boxes must be >= 1")
    if args.wl_min <= 0.0 or args.h_min <= 0.0 or args.mass_min <= 0.0:
        raise ValueError("box size and mass lower bounds must be positive")
    if args.wl_min > args.wl_max or args.h_min > args.h_max or args.mass_min > args.mass_max:
        raise ValueError("range minimum must be <= range maximum")

    cfg = load_config(args.config)
    ensure_isaacsim_python(cfg)
    app_cfg = dict(cfg["app"])
    app_cfg["experience"] = _resolve_experience(str(app_cfg["experience"]))
    if args.gui:
        app_cfg["headless"] = False
        app_cfg["hide_ui"] = False
        if not args.full_gui:
            original_experience = str(app_cfg["experience"])
            app_cfg["experience"] = _base_python_experience(original_experience)
            if app_cfg["experience"] != original_experience:
                print(
                    "[random-scene] --gui: using lightweight Isaac experience "
                    f"{app_cfg['experience']} instead of {original_experience}. "
                    "Pass --full-gui to force the full editor experience.",
                    flush=True,
                )

    sim_config = {
        "experience": app_cfg["experience"],
        "width": app_cfg["width"],
        "height": app_cfg["height"],
        "window_width": app_cfg["width"],
        "window_height": app_cfg["height"],
        "headless": app_cfg["headless"],
        "hide_ui": app_cfg["hide_ui"],
        "renderer": app_cfg["renderer"],
        "display_options": app_cfg["display_options"],
    }

    # Heavy Isaac imports must happen after SimulationApp starts.
    from isaacsim import SimulationApp  # noqa: WPS433

    simulation_app = SimulationApp(sim_config)

    from isaacsim.core.api import World  # noqa: WPS433
    import omni.usd  # noqa: WPS433
    import scene  # noqa: WPS433

    scene.init(cfg)
    usd_ctx = omni.usd.get_context()
    usd_ctx.new_stage()
    simulation_app.update()

    boxes = generate_random_boxes(
        num_boxes=args.num_boxes,
        seed=args.seed,
        wl_range=(args.wl_min, args.wl_max),
        h_range=(args.h_min, args.h_max),
        mass_range=(args.mass_min, args.mass_max),
    )

    pallet_size = [float(v) for v in cfg["pallet"]["size"]]
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    _, buffer_thickness = scene.build_base_scene(world, pallet_size, args.num_boxes)
    world.reset()

    for slot_idx, box in enumerate(boxes):
        slot = scene.buffer_slot_world_pos(slot_idx, pallet_size[0], buffer_thickness)
        scene.spawn_in_buffer(
            world=world,
            bid=int(box["id"]),
            size=box["size"],
            mass=float(box["mass"]),
            rotation_deg=0.0,
            slot_pos=slot,
            label_z=0.0,
            simulation_app=simulation_app,
            settle_steps=args.settle_steps,
            settle_vel=args.settle_vel,
            drop_offset=args.drop_offset,
        )

    for _ in range(12):
        if simulation_app.is_running():
            world.step(render=not app_cfg["headless"])

    out_dir = args.output_dir
    write_jsonl(out_dir / f"random_boxes_seed_{args.seed}.jsonl", boxes)

    if args.save_usd is not None:
        args.save_usd.parent.mkdir(parents=True, exist_ok=True)
        usd_ctx.save_as_stage(str(args.save_usd))
        print(f"[random-scene] saved USD: {args.save_usd}", flush=True)

    print(
        "[random-scene] spawned "
        f"{len(boxes)} boxes on palletizing_simulator scene "
        f"(pallet={pallet_size}, seed={args.seed})",
        flush=True,
    )
    print(
        "[random-scene] random ranges: "
        f"W/L=[{args.wl_min}, {args.wl_max}], "
        f"H=[{args.h_min}, {args.h_max}], "
        f"mass=[{args.mass_min}, {args.mass_max}]",
        flush=True,
    )
    print(f"[random-scene] box JSONL: {out_dir / f'random_boxes_seed_{args.seed}.jsonl'}", flush=True)

    if args.hold or args.gui:
        print("[random-scene] holding Isaac app; close the window or stop the process to exit.", flush=True)
        while simulation_app.is_running():
            world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
