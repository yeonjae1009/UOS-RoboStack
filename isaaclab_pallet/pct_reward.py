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

    volume: float = 10.0
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
    corner_large: float = 0.0
    wall_anchor: float = 0.0
    tight_fit: float = 0.0
    void_reduction: float = 0.0
    support_margin: float = 0.0
    active_layer_coverage: float = 0.0
    center_of_mass_z_penalty: float = 0.0


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


def _box_bounds(box) -> tuple[float, float, float, float, float, float]:
    return (
        float(box.lx),
        float(box.ly),
        float(box.lz),
        float(box.lx + box.x),
        float(box.ly + box.y),
        float(box.lz + box.z),
    )


def _overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _required_support(mass: float, bottom_z: float) -> float:
    req = 0.20
    if bottom_z >= 0.25:
        req = 0.28
    if bottom_z >= 0.55:
        req = 0.42
    if bottom_z >= 0.85:
        req = 0.55
    if mass >= 4.0 and bottom_z >= 0.55:
        req = max(req, 0.62)
    if mass >= 5.5 and bottom_z >= 0.75:
        req = max(req, 0.72)
    if bottom_z >= 1.00 and mass >= 4.0:
        req = 0.80
    return req


def usable_void_volume(ems: np.ndarray, min_sorted: Sequence[float] = (0.13, 0.17, 0.17)) -> float:
    arr = np.asarray(ems, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = arr.reshape(-1, 6)
    sizes = np.maximum(arr[:, 3:6] - arr[:, 0:3], 0.0)
    sorted_sizes = np.sort(sizes, axis=1)
    min_sorted_arr = np.asarray(min_sorted, dtype=np.float64)
    volumes = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]
    usable = (sorted_sizes >= min_sorted_arr).all(axis=1) & (volumes >= 0.0035)
    return float(volumes[usable].sum())


def _layer_coverage(boxes, pallet_size, layer_z: float, grid: float = 0.025, tol: float = 0.006) -> float:
    pallet_x, pallet_y, _ = pallet_size
    nx = max(1, int(np.ceil(float(pallet_x) / grid)))
    ny = max(1, int(np.ceil(float(pallet_y) / grid)))
    layer_map = np.zeros((nx, ny), dtype=bool)
    for box in boxes:
        if abs(float(box.lz) - layer_z) > tol:
            continue
        ix0, ix1, iy0, iy1 = box_grid_slice(box, grid, nx, ny)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        layer_map[ix0:ix1, iy0:iy1] = True
    return float(np.mean(layer_map))


def _stack_center_of_mass_z(boxes, pallet_z: float) -> float:
    total_mass = 0.0
    weighted_z = 0.0
    for box in boxes:
        mass = max(0.0, float(getattr(box, "mass", 1.0)))
        center_z = float(box.lz + box.z / 2.0)
        total_mass += mass
        weighted_z += mass * center_z
    if total_mass <= 1e-9:
        return 0.0
    return float(np.clip((weighted_z / total_mass) / max(float(pallet_z), 1e-9), 0.0, 1.0))


def _axis_tight_fit(
    lo: float,
    hi: float,
    axis: int,
    other_lo: float,
    other_hi: float,
    bottom_z: float,
    boxes,
    pallet_extent: float,
    tol: float = 0.025,
) -> float:
    lower_gap = lo
    upper_gap = pallet_extent - hi
    for prev in boxes:
        if prev is None:
            continue
        px0, py0, pz0, px1, py1, pz1 = _box_bounds(prev)
        if abs(pz1 - bottom_z) > 0.006 and abs(pz0 - bottom_z) > 0.006:
            continue
        if axis == 0:
            overlap = _overlap_len(other_lo, other_hi, py0, py1)
            prev_lo, prev_hi = px0, px1
        else:
            overlap = _overlap_len(other_lo, other_hi, px0, px1)
            prev_lo, prev_hi = py0, py1
        if overlap <= 1e-6:
            continue
        if prev_hi <= lo + 1e-6:
            lower_gap = min(lower_gap, lo - prev_hi)
        if prev_lo >= hi - 1e-6:
            upper_gap = min(upper_gap, prev_lo - hi)
    if lower_gap > tol or upper_gap > tol:
        return 0.0
    return float(np.clip(1.0 - (lower_gap + upper_gap) / (2.0 * tol), 0.0, 1.0))


def placement_geometry_terms(
    packed_box,
    boxes_before,
    pallet_size,
    before_void_volume: float | None = None,
    after_void_volume: float | None = None,
) -> dict[str, float]:
    pallet_x, pallet_y, pallet_z = [float(v) for v in pallet_size]
    lx, ly, lz, hx, hy, hz = _box_bounds(packed_box)
    area_ratio = float(packed_box.x * packed_box.y) / max(pallet_x * pallet_y, 1e-9)
    large_factor = float(np.clip((area_ratio - 0.045) / 0.04, 0.0, 1.0))
    eps = 0.005
    x_wall = lx <= eps or hx >= pallet_x - eps
    y_wall = ly <= eps or hy >= pallet_y - eps
    corner_large_anchor = large_factor if x_wall and y_wall else 0.0

    flush_contacts = 0
    for prev in boxes_before:
        px0, py0, pz0, px1, py1, pz1 = _box_bounds(prev)
        z_overlap = _overlap_len(lz, hz, pz0, pz1)
        if z_overlap <= 1e-6:
            continue
        if (abs(lx - px1) <= eps or abs(hx - px0) <= eps) and _overlap_len(ly, hy, py0, py1) > 1e-6:
            flush_contacts += 1
        if (abs(ly - py1) <= eps or abs(hy - py0) <= eps) and _overlap_len(lx, hx, px0, px1) > 1e-6:
            flush_contacts += 1
    wall_anchor = float(np.clip((float(x_wall) + float(y_wall) + 0.5 * flush_contacts) / 2.0, 0.0, 1.0))

    tight_x = _axis_tight_fit(lx, hx, 0, ly, hy, lz, boxes_before, pallet_x)
    tight_y = _axis_tight_fit(ly, hy, 1, lx, hx, lz, boxes_before, pallet_y)
    tight_fit = max(tight_x, tight_y)

    ratio = support_ratio(packed_box)
    required = 1.0 if lz <= 1e-6 else _required_support(float(getattr(packed_box, "mass", 0.0)), lz)
    support_quality = 1.0 if lz <= 1e-6 else float(np.clip(ratio / max(required, 1e-9), 0.0, 1.0))
    support_margin = float(np.clip((ratio - required) / max(1.0 - required, 1e-9), 0.0, 1.0))
    unsupported_risk = 0.0 if lz <= 1e-6 else float(np.clip((required - ratio) / max(required, 1e-9), 0.0, 1.0))

    layer_before = _layer_coverage(boxes_before, pallet_size, lz)
    layer_after = _layer_coverage([*boxes_before, packed_box], pallet_size, lz)
    active_layer_coverage = float(np.clip(layer_after - layer_before, -1.0, 1.0))

    com_before = _stack_center_of_mass_z(boxes_before, pallet_z)
    com_after = _stack_center_of_mass_z([*boxes_before, packed_box], pallet_z)
    center_of_mass_z_increase = float(np.clip(com_after - com_before, 0.0, 1.0))

    void_reduction = 0.0
    if before_void_volume is not None and after_void_volume is not None:
        void_reduction = float(np.clip((before_void_volume - after_void_volume) / 0.05, 0.0, 1.0))

    return {
        "corner_large_anchor": float(corner_large_anchor),
        "wall_anchor": float(wall_anchor),
        "tight_fit": float(tight_fit),
        "support_quality": float(support_quality),
        "support_margin": float(support_margin),
        "unsupported_risk": float(unsupported_risk),
        "void_reduction": float(void_reduction),
        "active_layer_coverage": float(active_layer_coverage),
        "center_of_mass_z": float(com_after),
        "center_of_mass_z_increase": float(center_of_mass_z_increase),
    }


def compute_online3dbpp_reward(
    box_ratio: float,
    packed_box,
    before: dict[str, float],
    after: dict[str, float],
    scales: RewardScales | None = None,
    geometry: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Mirror bin3D._compute_shaped_reward. Returns (reward, term breakdown)."""
    scales = scales or RewardScales()
    geometry = geometry or {}

    volume_reward = box_ratio * scales.volume
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
    corner_large_reward = scales.corner_large * float(geometry.get("corner_large_anchor", 0.0))
    wall_anchor_reward = scales.wall_anchor * float(geometry.get("wall_anchor", 0.0))
    tight_fit_reward = scales.tight_fit * float(geometry.get("tight_fit", 0.0))
    void_reduction_reward = scales.void_reduction * float(geometry.get("void_reduction", 0.0))
    support_margin_reward = scales.support_margin * float(geometry.get("support_margin", 0.0))
    active_layer_coverage_reward = scales.active_layer_coverage * float(
        geometry.get("active_layer_coverage", 0.0)
    )
    center_of_mass_z_penalty = scales.center_of_mass_z_penalty * float(
        geometry.get("center_of_mass_z_increase", 0.0)
    )

    reward = float(
        volume_reward
        + floor_coverage_reward
        + boundary_floor_reward
        + corner_floor_reward
        + height_smoothness_reward
        + support_reward
        + corner_large_reward
        + wall_anchor_reward
        + tight_fit_reward
        + void_reduction_reward
        + support_margin_reward
        + active_layer_coverage_reward
        - weak_support_penalty
        - elevation_penalty
        - center_of_mass_z_penalty
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
        "corner_large_reward": float(corner_large_reward),
        "wall_anchor_reward": float(wall_anchor_reward),
        "tight_fit_reward": float(tight_fit_reward),
        "void_reduction_reward": float(void_reduction_reward),
        "support_margin_reward": float(support_margin_reward),
        "active_layer_coverage_reward": float(active_layer_coverage_reward),
        "center_of_mass_z_penalty": float(center_of_mass_z_penalty),
        "corner_large_anchor": float(geometry.get("corner_large_anchor", 0.0)),
        "wall_anchor": float(geometry.get("wall_anchor", 0.0)),
        "tight_fit": float(geometry.get("tight_fit", 0.0)),
        "void_reduction": float(geometry.get("void_reduction", 0.0)),
        "support_margin": float(geometry.get("support_margin", 0.0)),
        "active_layer_coverage": float(geometry.get("active_layer_coverage", 0.0)),
        "center_of_mass_z": float(geometry.get("center_of_mass_z", 0.0)),
        "center_of_mass_z_increase": float(geometry.get("center_of_mass_z_increase", 0.0)),
        "support_ratio": float(ratio),
        "reward": reward,
    }
    return reward, terms
