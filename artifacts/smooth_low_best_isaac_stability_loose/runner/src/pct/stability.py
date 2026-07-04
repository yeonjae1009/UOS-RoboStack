from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StabilityInfo:
    support_ratio: float
    required_support: float
    support_count: int
    reason: str


def _box_record(box) -> tuple[float, float, float, float, float, float]:
    if hasattr(box, "x"):
        return (
            float(box.x),
            float(box.y),
            float(box.z),
            float(box.lx),
            float(box.ly),
            float(box.lz),
        )
    return (
        float(box[0]),
        float(box[1]),
        float(box[2]),
        float(box[3]),
        float(box[4]),
        float(box[5]),
    )


def _leaf_to_box(leaf: Sequence[float], box_size: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    lx, ly, lz, hx, hy, _ = [float(v) for v in leaf[:6]]
    x = round(hx - lx, 6)
    y = round(hy - ly, 6)

    remaining = [0, 1, 2]
    for axis in list(remaining):
        if abs(x - float(box_size[axis])) < 1e-6:
            remaining.remove(axis)
            break
    for axis in list(remaining):
        if abs(y - float(box_size[axis])) < 1e-6:
            remaining.remove(axis)
            break
    z = float(box_size[remaining[0]])
    return x, y, z, lx, ly, lz


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float]:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return 0.0, 0.0
    return lo, hi


def _required_support(mass: float, bottom_z: float) -> float:
    req = 0.35
    if bottom_z >= 0.25:
        req = 0.45
    if bottom_z >= 0.55:
        req = 0.58
    if bottom_z >= 0.85:
        req = 0.68
    if mass >= 4.0 and bottom_z >= 0.55:
        req = max(req, 0.72)
    if mass >= 5.5 and bottom_z >= 0.75:
        req = max(req, 0.80)
    if bottom_z >= 1.00 and mass >= 4.0:
        req = 0.86
    return req


def evaluate_leaf_stability(
    leaf: Sequence[float],
    placed_boxes: Iterable,
    box_size: Sequence[float],
    mass: float,
    pallet_size: Sequence[float],
    z_tolerance: float = 0.006,
) -> StabilityInfo:
    if float(leaf[8]) <= 0.5 or float(np.sum(leaf[:6])) == 0.0:
        return StabilityInfo(0.0, 1.0, 0, "invalid_leaf")

    x, y, z, lx, ly, lz = _leaf_to_box(leaf, box_size)
    top = lz + z
    if top > float(pallet_size[2]) + 1e-6:
        return StabilityInfo(0.0, 1.0, 0, "height")

    if lz <= 1e-6:
        return StabilityInfo(1.0, 1.0, 1, "floor")

    base_area = max(x * y, 1e-9)
    support_area = 0.0
    support_count = 0
    sx0 = sy0 = float("inf")
    sx1 = sy1 = float("-inf")

    cand_x0, cand_x1 = lx, lx + x
    cand_y0, cand_y1 = ly, ly + y
    for prev in placed_boxes:
        px, py, pz, plx, ply, plz = _box_record(prev)
        if abs((plz + pz) - lz) > z_tolerance:
            continue
        ox0, ox1 = _overlap_1d(cand_x0, cand_x1, plx, plx + px)
        oy0, oy1 = _overlap_1d(cand_y0, cand_y1, ply, ply + py)
        area = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
        if area <= 1e-9:
            continue
        support_area += area
        support_count += 1
        sx0, sx1 = min(sx0, ox0), max(sx1, ox1)
        sy0, sy1 = min(sy0, oy0), max(sy1, oy1)

    ratio = float(np.clip(support_area / base_area, 0.0, 1.0))
    required = _required_support(float(mass), lz)
    if ratio + 1e-9 < required:
        return StabilityInfo(ratio, required, support_count, "weak_support")

    if support_count <= 0:
        return StabilityInfo(ratio, required, support_count, "no_support")

    cx = lx + x / 2.0
    cy = ly + y / 2.0
    if lz >= 0.75 or mass >= 5.5:
        margin = 0.005
        if not (sx0 + margin <= cx <= sx1 - margin and sy0 + margin <= cy <= sy1 - margin):
            return StabilityInfo(ratio, required, support_count, "center_margin")

    return StabilityInfo(ratio, required, support_count, "ok")


def apply_stability_mask(
    leaf_nodes: np.ndarray,
    placed_boxes: Iterable,
    box_size: Sequence[float],
    mass: float,
    pallet_size: Sequence[float],
) -> tuple[np.ndarray, list[StabilityInfo]]:
    masked = leaf_nodes.copy()
    infos: list[StabilityInfo] = []
    for i in range(masked.shape[0]):
        info = evaluate_leaf_stability(masked[i], placed_boxes, box_size, mass, pallet_size)
        infos.append(info)
        if info.reason not in ("ok", "floor"):
            masked[i, 8] = 0.0
    return masked, infos
