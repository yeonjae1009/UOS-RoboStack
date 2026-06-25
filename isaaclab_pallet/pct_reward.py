"""Deterministic (Isaac-free) CPU layer shared by the env and the equivalence test.

These functions are the *deterministic* layer of the port — leaf selection,
placement geometry decoding, layout metrics and the Online-3D-BPP-PCT shaped
reward. They are a 1:1 mirror of
``Online-3D-BPP-PCT/pct_envs/PctContinuous0/bin3D.py`` (``PackingContinuous``)
with **no torch / Isaac dependency** so they can be imported and tested without
booting Isaac Sim.

NOTE: this is the same pure-numpy math the original PCT already ran on the CPU.
Nothing is being moved off the GPU — the GPU PhysX physics (drift / tilt /
collapse) is the *replacement* for the original convex-hull stability heuristic
and lives in the env, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


PCT_INTERNAL_NODE_LENGTH = 7


@dataclass(frozen=True)
class RewardScales:
    """Mirrors the hard-coded constants in bin3D.py::_compute_shaped_reward."""

    floor_coverage: float = 1.0
    boundary_floor: float = 0.8
    corner_floor: float = 0.6
    height_smoothness: float = 0.5
    support: float = 0.05
    weak_support: float = 0.05
    weak_support_threshold: float = 0.85
    # Density-shaping knob (#4). Penalizes placing a box high up (large lz), which
    # nudges the policy to fill the bottom first -> denser, less empty space.
    # Default 0.0 = OFF (no behaviour change vs original PCT; equivalence preserved).
    # Tune on the Isaac machine where utilization can actually be measured.
    elevation_penalty: float = 0.0


def density_for_box(box: dict, setting: int, density_max: float) -> float:
    """next_den used by EMS pruning. Mirror bin3D cur_observation (setting 3)."""
    if setting < 3:
        return 1.0
    sx, sy, sz = [float(v) for v in box["size"]]
    vol = max(sx * sy * sz, 1e-9)
    return (float(box["mass"]) / vol) / density_max


def select_leaf(leaf_nodes: np.ndarray, action_idx: int) -> np.ndarray | None:
    action_idx = int(np.clip(action_idx, 0, leaf_nodes.shape[0] - 1))
    leaf = leaf_nodes[action_idx]
    if float(leaf[8]) <= 0.5:
        return None
    return leaf


def leaf_to_center_size_rotation(
    leaf: np.ndarray, box_size: Sequence[float]
) -> tuple[list[float], list[float], int]:
    """leaf(0:6) + raw box size -> (local center, placed size, rotation deg).

    Mirrors bin3D.LeafNode2Action axis-matching so the spawned box footprint
    equals what the packer drops.
    """
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

    center = [
        lx + placed_size[0] / 2.0,
        ly + placed_size[1] / 2.0,
        lz + placed_size[2] / 2.0,
    ]
    rotation = 0 if (
        abs(placed_size[0] - float(box_size[0])) < 1e-6
        and abs(placed_size[1] - float(box_size[1])) < 1e-6
    ) else 90
    return center, placed_size, rotation


def box_grid_slice(box, grid: float, nx: int, ny: int) -> tuple[int, int, int, int]:
    """Mirror bin3D._box_grid_slice."""
    ix0 = int(np.floor((box.lx + 1e-9) / grid))
    iy0 = int(np.floor((box.ly + 1e-9) / grid))
    ix1 = int(np.ceil((box.lx + box.x - 1e-9) / grid))
    iy1 = int(np.ceil((box.ly + box.y - 1e-9) / grid))

    ix0 = max(0, min(nx, ix0))
    iy0 = max(0, min(ny, iy0))
    ix1 = max(ix0, min(nx, ix1))
    iy1 = max(iy0, min(ny, iy1))
    return ix0, ix1, iy0, iy1


def build_height_map(boxes, pallet_size, grid: float = 0.025) -> np.ndarray:
    """Mirror bin3D._build_height_map."""
    pallet_x, pallet_y, _ = pallet_size
    nx = max(1, int(np.ceil(pallet_x / grid)))
    ny = max(1, int(np.ceil(pallet_y / grid)))
    height_map = np.zeros((nx, ny), dtype=np.float32)

    for box in boxes:
        ix0, ix1, iy0, iy1 = box_grid_slice(box, grid, nx, ny)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        height_map[ix0:ix1, iy0:iy1] = np.maximum(height_map[ix0:ix1, iy0:iy1], float(box.lz + box.z))
    return height_map


def layout_metrics(boxes, pallet_size, grid: float = 0.025) -> dict[str, float]:
    """Mirror bin3D._layout_metrics. ``boxes`` is ``space.boxes``."""
    pallet_x, pallet_y, pallet_z = pallet_size
    nx = max(1, int(np.ceil(pallet_x / grid)))
    ny = max(1, int(np.ceil(pallet_y / grid)))

    floor_map = np.zeros((nx, ny), dtype=bool)
    for box in boxes:
        if abs(box.lz) > 1e-6:
            continue
        ix0, ix1, iy0, iy1 = box_grid_slice(box, grid, nx, ny)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        floor_map[ix0:ix1, iy0:iy1] = True

    height_map = build_height_map(boxes, pallet_size, grid)
    occupied = height_map > 1e-6
    height_roughness = float(np.std(height_map[occupied]) / max(float(pallet_z), 1e-9)) if np.any(occupied) else 0.0

    x_centers = (np.arange(nx) + 0.5) * grid
    y_centers = (np.arange(ny) + 0.5) * grid
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")

    band = 0.13
    boundary_mask = (xx <= band) | (xx >= float(pallet_x) - band) | (yy <= band) | (yy >= float(pallet_y) - band)
    corner = 0.22
    corner_mask = ((xx <= corner) | (xx >= float(pallet_x) - corner)) & (
        (yy <= corner) | (yy >= float(pallet_y) - corner)
    )

    return {
        "floor_coverage": float(np.mean(floor_map)),
        "boundary_floor_coverage": float(np.sum(floor_map & boundary_mask) / max(np.sum(boundary_mask), 1)),
        "corner_floor_coverage": float(np.sum(floor_map & corner_mask) / max(np.sum(corner_mask), 1)),
        "height_roughness": height_roughness,
    }


def support_ratio(packed_box) -> float:
    """Mirror bin3D._support_ratio."""
    if abs(packed_box.lz) <= 1e-6:
        return 1.0

    support_area = 0.0
    for edge in packed_box.bottom_edges:
        if edge.area is None:
            continue
        x1, y1, x2, y2 = edge.area
        support_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)

    base_area = max(float(packed_box.x * packed_box.y), 1e-9)
    return float(np.clip(support_area / base_area, 0.0, 1.0))


def compute_online3dbpp_reward(
    box_ratio: float,
    packed_box,
    before: dict[str, float],
    after: dict[str, float],
    scales: RewardScales | None = None,
) -> tuple[float, dict[str, float]]:
    """Mirror bin3D._compute_shaped_reward. Returns (reward, term breakdown)."""
    scales = scales or RewardScales()

    volume_reward = box_ratio * 10.0
    floor_coverage_reward = scales.floor_coverage * (after["floor_coverage"] - before["floor_coverage"])
    boundary_floor_reward = scales.boundary_floor * (
        after["boundary_floor_coverage"] - before["boundary_floor_coverage"]
    )
    corner_floor_reward = scales.corner_floor * (after["corner_floor_coverage"] - before["corner_floor_coverage"])
    height_delta = before["height_roughness"] - after["height_roughness"]
    height_smoothness_reward = scales.height_smoothness * float(np.clip(height_delta, -0.05, 0.05))

    ratio = support_ratio(packed_box)
    support_reward = scales.support * ratio
    weak_support_penalty = (
        scales.weak_support * max(0.0, scales.weak_support_threshold - ratio) if packed_box.lz > 1e-6 else 0.0
    )
    # #4 density knob: penalize elevation (off by default -> reward unchanged).
    elevation_penalty = scales.elevation_penalty * max(0.0, float(packed_box.lz))

    reward = float(
        volume_reward
        + floor_coverage_reward
        + boundary_floor_reward
        + corner_floor_reward
        + height_smoothness_reward
        + support_reward
        - weak_support_penalty
        - elevation_penalty
    )
    terms = {
        "volume_reward": float(volume_reward),
        "floor_coverage_reward": float(floor_coverage_reward),
        "boundary_floor_reward": float(boundary_floor_reward),
        "corner_floor_reward": float(corner_floor_reward),
        "height_smoothness_reward": float(height_smoothness_reward),
        "support_reward": float(support_reward),
        "weak_support_penalty": float(weak_support_penalty),
        "elevation_penalty": float(elevation_penalty),
        "support_ratio": float(ratio),
        "reward": reward,
    }
    return reward, terms
