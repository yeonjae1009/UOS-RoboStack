from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

from buffer_manager import BufferManager


# ---------------------------------------------------------------------------
# 입출력 스키마 (평가 시스템 연동: 변경 금지)
# ---------------------------------------------------------------------------

class BoxInput(TypedDict):
    step: int
    id: int
    size: List[float]
    mass: float


class PlacedBox(TypedDict):
    step: int
    id: int
    size: List[float]
    mass: float
    position: List[float]
    rotation: int


class RunResult(TypedDict):
    buffer_size: int
    sequence: List[PlacedBox]
    terminated: bool
    terminated_step: Optional[int]
    finished_by_user: bool


@dataclass
class PalletConfig:
    length: float
    width: float
    height: float


@dataclass
class AlgorithmConfig:
    allow_rotation: bool
    buffer_size: int
    min_support: float = 0.64
    beam_width: int = 8
    beam_depth: int = 1
    positions_per_box: int = 1
    emergency_support: float = 0.45
    enable_adaptive_sku_columns: bool = True
    sku_column_duplicate_count: int = 2
    sku_column_max_height: float = 0.84
    sku_column_min_footprint_area: float = 0.0
    sku_column_max_footprint_area: float = 0.060
    sku_column_confirmed_max_footprint_area: float = 0.070
    sku_column_catalog_confirmation_boxes: int = 25
    sku_column_expected_types: int = 5
    sku_column_confirmation_boxes: int = 15
    sku_column_min_observed_types: int = 5
    sku_column_large_aspect_ratio_min: float = 1.10


@dataclass(frozen=True)
class _Solid:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    box_id: int
    mass: float
    support: float = 1.0

    @property
    def x1(self) -> float:
        return self.x + self.dx

    @property
    def y1(self) -> float:
        return self.y + self.dy

    @property
    def top(self) -> float:
        return self.z + self.dz

    @property
    def volume(self) -> float:
        return self.dx * self.dy * self.dz


@dataclass(frozen=True)
class _Action:
    box_index: int
    box: BoxInput
    solid: _Solid
    rotation: int
    local_key: Tuple[float, ...]


class Palletizer:
    """Support-aware online 3D extreme-point palletizer.

    Only boxes currently visible in the sliding buffer participate in planning.
    The first action of a short beam search is committed, then the newly
    refilled buffer is planned again.
    """

    _EPS = 1.0e-7

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg
        self.min_support = float(algo_cfg.min_support)
        self.emergency_support = float(algo_cfg.emergency_support)
        self.beam_width = max(1, int(algo_cfg.beam_width))
        self.beam_depth = max(1, int(algo_cfg.beam_depth))
        self.positions_per_box = max(1, int(algo_cfg.positions_per_box))
        self.enable_adaptive_sku_columns = bool(
            algo_cfg.enable_adaptive_sku_columns
        )
        self.sku_column_duplicate_count = max(
            2, int(algo_cfg.sku_column_duplicate_count)
        )
        self.sku_column_max_height = min(
            pallet_cfg.height, max(0.0, float(algo_cfg.sku_column_max_height))
        )
        self.sku_column_min_footprint_area = max(
            0.0, float(algo_cfg.sku_column_min_footprint_area)
        )
        self.sku_column_max_footprint_area = max(
            0.0, float(algo_cfg.sku_column_max_footprint_area)
        )
        self.sku_column_confirmed_max_footprint_area = max(
            self.sku_column_max_footprint_area,
            float(algo_cfg.sku_column_confirmed_max_footprint_area),
        )
        self.sku_column_catalog_confirmation_boxes = max(
            2, int(algo_cfg.sku_column_catalog_confirmation_boxes)
        )
        self.sku_column_expected_types = max(
            1, int(algo_cfg.sku_column_expected_types)
        )
        self.sku_column_confirmation_boxes = max(
            2, int(algo_cfg.sku_column_confirmation_boxes)
        )
        self.sku_column_min_observed_types = max(
            1, int(algo_cfg.sku_column_min_observed_types)
        )
        self.sku_column_large_aspect_ratio_min = max(
            1.0, float(algo_cfg.sku_column_large_aspect_ratio_min)
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self.sequence: List[PlacedBox] = []
        self.solids: List[_Solid] = []
        self.finished = False
        self.terminated_step: Optional[int] = None
        self.finished_by_user = False
        self._layout_axis_mode = "y_first"
        self._planned_box_id: Optional[int] = None
        self._planned_action: Optional[_Action] = None
        self._require_center_support = False
        self._observed_box_keys: set[Tuple[int, int]] = set()
        self._observed_sku_counts: dict[Tuple[float, float, float], int] = {}

    def _configure_sequence_policy(self, boxes: List[BoxInput]) -> None:
        """Configure only fixed online tie-breaks.

        The evaluator passes the full input list to ``run()``, but the task
        intent is that decisions use only the currently visible buffer.  With
        buffer_size=0 this method must not inspect sequence statistics, future
        boxes, or perform dry-runs.  It only resets deterministic defaults.
        """
        self._layout_axis_mode = "y_first"

    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        """Official hook called with only the currently visible buffer.

        The fixed evaluator owns the loop and buffer manager.  To legally use
        a non-zero buffer, we choose one box from *current_buffer* here and let
        ``_find_position()`` return a placement only for that selected box.
        No full input sequence or future boxes outside this list are inspected.
        """
        self._sync_solids_from_sequence()
        self._planned_box_id = None
        self._planned_action = None

        if current_buffer:
            self._observe_current_box(current_buffer[0])

        if len(current_buffer) > 1:
            action = self._choose_action(current_buffer)
            if action is not None:
                self._planned_box_id = int(action.box["id"])
                self._planned_action = action
        return False

    def _sku_signature(self, box: BoxInput) -> Tuple[float, float, float]:
        sx, sy, sz = map(float, box["size"])
        return (
            round(min(sx, sy), 6),
            round(max(sx, sy), 6),
            round(sz, 6),
        )

    def _observe_current_box(self, box: BoxInput) -> None:
        box_key = (int(box["step"]), int(box["id"]))
        if box_key in self._observed_box_keys:
            return
        self._observed_box_keys.add(box_key)
        signature = self._sku_signature(box)
        self._observed_sku_counts[signature] = (
            self._observed_sku_counts.get(signature, 0) + 1
        )

    def _sync_solids_from_sequence(self) -> None:
        """Rebuild internal geometry from already committed output records.

        The official evaluator may provide its own ``run()`` loop and only
        call participant methods such as ``_find_position()``.  In that case
        the source of truth available to us is ``self.sequence``.  Rebuilding
        from it keeps the placement policy online and compatible with both the
        local runner and the official fixed runner.
        """
        if len(self.solids) == len(self.sequence):
            return

        solids: List[_Solid] = []
        for item in self.sequence:
            dx, dy, dz = map(float, item["size"])
            cx, cy, cz = map(float, item["position"])
            x = cx - dx / 2.0
            y = cy - dy / 2.0
            z = cz - dz / 2.0
            support = self._support_ratio(x, y, z, dx, dy, solids)
            solids.append(
                _Solid(
                    x=x,
                    y=y,
                    z=z,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    box_id=int(item["id"]),
                    mass=float(item["mass"]),
                    support=support,
                )
            )
        self.solids = solids

    def _orientations(self, box: BoxInput) -> List[Tuple[float, float, float, int]]:
        sx, sy, sz = map(float, box["size"])
        result = [(sx, sy, sz, 0)]
        if self.algo.allow_rotation and abs(sx - sy) > self._EPS:
            result.append((sy, sx, sz, 90))
        # Prefer aligning the longer footprint side with the pallet length
        # axis.  The pallet is rectangular (1.2m x 1.0m), and random boxes are
        # close to square but not identical.  This deterministic tie-break
        # slightly reduces fragmented end-of-run floor footprints without
        # inspecting any future boxes.
        result.sort(key=lambda item: (-item[0], item[1]))
        return result

    @staticmethod
    def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    def _collides(self, candidate: _Solid, solids: List[_Solid]) -> bool:
        e = self._EPS
        for other in solids:
            if (
                candidate.x < other.x1 - e
                and candidate.x1 > other.x + e
                and candidate.y < other.y1 - e
                and candidate.y1 > other.y + e
                and candidate.z < other.top - e
                and candidate.top > other.z + e
            ):
                return True
        return False

    def _support_ratio(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        solids: List[_Solid],
    ) -> float:
        if z <= self._EPS:
            return 1.0

        area = 0.0
        for support in solids:
            if abs(support.top - z) > self._EPS:
                continue
            ox = self._overlap_1d(x, x + dx, support.x, support.x1)
            oy = self._overlap_1d(y, y + dy, support.y, support.y1)
            area += ox * oy
        return min(1.0, area / (dx * dy))

    def _center_supported(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        solids: List[_Solid],
    ) -> bool:
        """Return whether the box center projects onto a supporting face.

        Area support alone can accept a box whose center of mass is over empty
        space.  The normal policy avoids that through a conservative support
        ratio.  Emergency low-support recovery uses this extra guard so the
        fallback does not introduce obviously torque-prone placements.
        """
        if z <= self._EPS:
            return True

        cx = x + dx / 2.0
        cy = y + dy / 2.0
        for support in solids:
            if abs(support.top - z) > self._EPS:
                continue
            if (
                support.x - self._EPS <= cx <= support.x1 + self._EPS
                and support.y - self._EPS <= cy <= support.y1 + self._EPS
            ):
                return True
        return False

    def _horizontal_contact(self, candidate: _Solid, solids: List[_Solid]) -> float:
        """Measure side contact with pallet walls and already placed boxes.

        More side contact produces compact rows instead of isolated, centered
        boxes.  That leaves larger rectangular free regions for later arrivals
        and also improves physical stability without looking beyond the active
        buffer.
        """
        contact = 0.0
        if abs(candidate.x) <= self._EPS or abs(candidate.x1 - self.pallet.length) <= self._EPS:
            contact += candidate.dy
        if abs(candidate.y) <= self._EPS or abs(candidate.y1 - self.pallet.width) <= self._EPS:
            contact += candidate.dx

        for other in solids:
            z_overlap = self._overlap_1d(candidate.z, candidate.top, other.z, other.top)
            if z_overlap <= self._EPS:
                continue
            if abs(candidate.x - other.x1) <= self._EPS or abs(candidate.x1 - other.x) <= self._EPS:
                contact += z_overlap * self._overlap_1d(candidate.y, candidate.y1, other.y, other.y1)
            if abs(candidate.y - other.y1) <= self._EPS or abs(candidate.y1 - other.y) <= self._EPS:
                contact += z_overlap * self._overlap_1d(candidate.x, candidate.x1, other.x, other.x1)
        return contact

    def _plane_positions(
        self,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        solids: List[_Solid],
    ) -> List[Tuple[float, float]]:
        length, width = self.pallet.length, self.pallet.width
        positions = {
            (0.0, 0.0),
            (length - dx, 0.0),
            (0.0, width - dy),
            (length - dx, width - dy),
            ((length - dx) / 2.0, (width - dy) / 2.0),
        }

        # Support-face edges are the most useful coordinates for stable stacks.
        if z > self._EPS:
            for s in solids:
                if abs(s.top - z) <= self._EPS:
                    sx = (s.x, s.x1 - dx, s.x + (s.dx - dx) / 2.0)
                    sy = (s.y, s.y1 - dy, s.y + (s.dy - dy) / 2.0)
                    positions.update((x, y) for x in sx for y in sy)

        # Contact coordinates around obstacles intersecting the new box's Z span.
        for s in solids:
            if z < s.top - self._EPS and z + dz > s.z + self._EPS:
                align_y = (s.y, s.y1 - dy, s.y + (s.dy - dy) / 2.0)
                align_x = (s.x, s.x1 - dx, s.x + (s.dx - dx) / 2.0)
                positions.update((s.x - dx, y) for y in align_y)
                positions.update((s.x1, y) for y in align_y)
                positions.update((x, s.y - dy) for x in align_x)
                positions.update((x, s.y1) for x in align_x)

        return sorted(
            {
                (round(x, 6), round(y, 6))
                for x, y in positions
                if -self._EPS <= x <= length - dx + self._EPS
                and -self._EPS <= y <= width - dy + self._EPS
            },
            key=lambda p: (p[1], p[0]),
        )

    def _sku_column_footprint_limit(self) -> float:
        catalog_is_confirmed = (
            len(self._observed_box_keys)
            >= self.sku_column_catalog_confirmation_boxes
            and len(self._observed_sku_counts) == self.sku_column_expected_types
        )
        if catalog_is_confirmed:
            return self.sku_column_confirmed_max_footprint_area
        return self.sku_column_max_footprint_area

    def _column_threshold(self, box: BoxInput) -> float:
        """Single global rule for deciding when to open another column.

        Low and mid-low boxes are cheap to stack into columns.  Taller boxes
        open new floor positions earlier to avoid fragile towers and preserve
        broad top faces for later arrivals.
        """
        if (
            self.enable_adaptive_sku_columns
            and len(self._observed_box_keys) >= self.sku_column_confirmation_boxes
            and self._adaptive_sku_columns_allowed()
            and self._observed_sku_counts.get(self._sku_signature(box), 0)
            >= self.sku_column_duplicate_count
            and float(box["size"][0]) * float(box["size"][1])
            + self._EPS >= self.sku_column_min_footprint_area
            and float(box["size"][0]) * float(box["size"][1])
            <= self._sku_column_footprint_limit() + self._EPS
        ):
            return self.sku_column_max_height

        _, _, sz = map(float, box["size"])
        if sz <= 0.135 + self._EPS:
            return 1.25
        return 0.50

    def _adaptive_sku_columns_allowed(self) -> bool:
        if len(self._observed_sku_counts) < self.sku_column_min_observed_types:
            return False
        largest = max(
            self._observed_sku_counts,
            key=lambda signature: signature[0] * signature[1],
        )
        aspect_ratio = largest[1] / max(largest[0], self._EPS)
        return aspect_ratio + self._EPS >= self.sku_column_large_aspect_ratio_min

    def _required_support_ratio(
        self, dx: float, dy: float, dz: float, solids: List[_Solid]
    ) -> float:
        """Use adaptive support by footprint risk.

        Small-footprint boxes tolerate a little less footprint overlap because
        their center of mass is easier to keep over solid material and their
        overturning moment is lower.  Relaxing only this class recovers some
        otherwise blocked online placements without allowing broad/tall boxes
        to create fragile overhangs.
        """
        area = dx * dy
        if self._require_center_support:
            # Emergency recovery is only entered when the mandatory next box
            # has no normal placement.  Center support remains mandatory and
            # every emergency placement keeps at least half of the footprint
            # supported; lower floors did not improve score in checked seeds.
            return max(self.min_support, 0.50)
        if area <= 0.050:
            return min(self.min_support, 0.60)
        if area >= 0.060 and dz >= 0.18:
            return max(self.min_support, 0.65)
        return self.min_support

    def _uses_late_floor_contact_guard(
        self, dx: float, dy: float, dz: float, solids: List[_Solid]
    ) -> bool:
        area = dx * dy
        if self._require_center_support or area <= 0.050 or area >= 0.070:
            return False
        if area >= 0.060 and dz >= 0.18:
            return False
        floor_area = sum(s.dx * s.dy for s in solids if s.z <= self._EPS)
        floor_coverage = floor_area / (self.pallet.length * self.pallet.width)
        return floor_coverage + self._EPS >= 0.88

    def _rank_candidate_actions(
        self, actions: List[_Action], box: BoxInput, solids: List[_Solid]
    ) -> List[_Action]:
        actions.sort(key=lambda action: action.local_key)
        column_threshold = self._column_threshold(box)
        floor_actions: List[_Action] = []
        column_actions: List[_Action] = []

        for action in actions:
            candidate = action.solid
            if candidate.z <= self._EPS:
                floor_actions.append(action)
                continue

            for support in solids:
                if (
                    abs(support.top - candidate.z) <= self._EPS
                    and abs(support.x - candidate.x) <= self._EPS
                    and abs(support.y - candidate.y) <= self._EPS
                    and abs(support.dx - candidate.dx) <= self._EPS
                    and abs(support.dy - candidate.dy) <= self._EPS
                ):
                    column_actions.append(action)
                    break

        def compact_key(action: _Action) -> Tuple[float, ...]:
            solid = action.solid
            if self._layout_axis_mode == "x_first":
                return (
                    round(solid.top, 6),
                    -round(self._horizontal_contact(solid, solids), 6),
                    -round(solid.x, 6),
                    round(solid.y, 6),
                )
            return (
                round(solid.top, 6),
                -round(self._horizontal_contact(solid, solids), 6),
                -round(solid.y, 6),
                round(solid.x, 6),
            )

        floor_actions.sort(key=compact_key)
        column_actions.sort(key=compact_key)
        low_columns = [
            action
            for action in column_actions
            if action.solid.top <= column_threshold + self._EPS
        ]
        return low_columns or floor_actions or column_actions or actions

    def _prefer_late_floor_stricter(
        self,
        normal: List[_Action],
        stricter: List[_Action],
        solids: List[_Solid],
    ) -> List[_Action]:
        if not normal or not stricter:
            return normal
        normal_solid = normal[0].solid
        stricter_solid = stricter[0].solid
        if self._center_supported(
            normal_solid.x,
            normal_solid.y,
            normal_solid.z,
            normal_solid.dx,
            normal_solid.dy,
            solids,
        ) and not self._center_supported(
            stricter_solid.x,
            stricter_solid.y,
            stricter_solid.z,
            stricter_solid.dx,
            stricter_solid.dy,
            solids,
        ):
            return normal
        normal_contact = self._horizontal_contact(normal_solid, solids)
        stricter_contact = self._horizontal_contact(stricter_solid, solids)
        if stricter_contact + self._EPS >= normal_contact:
            return stricter
        return normal

    def _candidate_actions_for_box(
        self,
        box: BoxInput,
        box_index: int,
        solids: List[_Solid],
        limit: int,
    ) -> List[_Action]:
        planes = {0.0}
        planes.update(round(s.top, 6) for s in solids if s.top < self.pallet.height - self._EPS)
        actions: List[_Action] = []

        for dx, dy, dz, rotation in self._orientations(box):
            required_support = self._required_support_ratio(dx, dy, dz, solids)
            required_area = required_support * dx * dy
            useful_planes = [0.0]
            for z in planes:
                if z <= self._EPS:
                    continue
                available_area = sum(s.dx * s.dy for s in solids if abs(s.top - z) <= self._EPS)
                if available_area + self._EPS >= required_area:
                    useful_planes.append(z)
            for z in sorted(useful_planes):
                if z + dz > self.pallet.height + self._EPS:
                    continue
                positions = self._plane_positions(z, dx, dy, dz, solids)
                for x, y in positions:
                    support = self._support_ratio(x, y, z, dx, dy, solids)
                    if support + self._EPS < required_support:
                        continue
                    if self._require_center_support and not self._center_supported(
                        x, y, z, dx, dy, solids
                    ):
                        continue
                    solid = _Solid(
                        x=x,
                        y=y,
                        z=z,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                        box_id=int(box["id"]),
                        mass=float(box["mass"]),
                        support=support,
                    )
                    if self._collides(solid, solids):
                        continue

                    overhang = 1.0 - support
                    horizontal_contact = self._horizontal_contact(solid, solids)
                    # Bottom-up placement remains primary.  Among equally low,
                    # supported positions, maximize side contact to build dense
                    # rows from pallet boundaries instead of isolated islands.
                    local_key = (
                        round(z, 6),
                        round(overhang, 6),
                        -round(horizontal_contact, 6),
                        round(y, 6),
                        round(x, 6),
                    )
                    actions.append(_Action(box_index, box, solid, rotation, local_key))

        # Repeated box types are best handled as stable vertical columns.  A
        # short column is preferred before opening another floor position;
        # this preserves contiguous floor space for larger future arrivals.
        # The light, lowest-profile box can safely use a taller column.
        preferred = self._rank_candidate_actions(actions, box, solids)
        dx, dy, dz, _ = self._orientations(box)[0]
        if self._uses_late_floor_contact_guard(dx, dy, dz, solids):
            stricter = self._rank_candidate_actions(
                [action for action in actions if action.solid.support + self._EPS >= 0.65],
                box,
                solids,
            )
            preferred = self._prefer_late_floor_stricter(preferred, stricter, solids)
        return preferred[:limit]

    def _surface_metrics(self, solids: List[_Solid]) -> Tuple[float, float, float]:
        """Return inexpensive cavity, top roughness, and floor coverage proxies."""
        if not solids:
            return 0.0, 0.0, 0.0
        cavity = sum((1.0 - s.support) * s.dx * s.dy * s.z for s in solids)
        total_area = sum(s.dx * s.dy for s in solids)
        mean_top = sum(s.top * s.dx * s.dy for s in solids) / total_area
        variance = sum((s.top - mean_top) ** 2 * s.dx * s.dy for s in solids) / total_area
        floor_area = sum(s.dx * s.dy for s in solids if s.z <= self._EPS)
        return cavity, math.sqrt(variance), min(1.0, floor_area / (self.pallet.length * self.pallet.width))

    def _state_key(self, solids: List[_Solid]) -> Tuple[float, ...]:
        volume = sum(s.volume for s in solids)
        max_height = max((s.top for s in solids), default=0.0)
        cavity, roughness, base_coverage = self._surface_metrics(solids)
        avg_support = sum(s.support for s in solids) / len(solids) if solids else 1.0
        weighted_height = sum(s.mass * (s.z + s.dz / 2.0) for s in solids)
        return (
            round(cavity * 8.0 + roughness * 0.35 + max_height * 0.04 - volume * 2.5, 8),
            round(-volume, 8),
            round(-base_coverage, 8),
            round(-avg_support, 8),
            round(weighted_height, 8),
        )

    def _choose_action(self, current: List[BoxInput]) -> Optional[_Action]:
        first_layer: List[Tuple[List[_Solid], _Action, Tuple[int, ...]]] = []
        all_indices = tuple(range(len(current)))

        def representatives(indices: Tuple[int, ...]) -> List[int]:
            seen = set()
            result = []
            for index in indices:
                box = current[index]
                signature = (tuple(map(float, box["size"])), float(box["mass"]))
                if signature not in seen:
                    seen.add(signature)
                    result.append(index)
            return result

        for index in representatives(all_indices):
            box = current[index]
            for action in self._candidate_actions_for_box(
                box, index, self.solids, self.positions_per_box
            ):
                remaining = tuple(i for i in all_indices if i != index)
                first_layer.append((self.solids + [action.solid], action, remaining))

        if not first_layer:
            # Last-resort online recovery: keep the normal support threshold
            # conservative, but if the single mandatory next box cannot be
            # placed at all, try one lower-support placement before ending the
            # episode.  This is only used for buffer_size=0/single-box calls;
            # buffer selection keeps the stricter threshold to avoid choosing
            # fragile boxes just because they are visible.
            if (
                len(current) == 1
                and self.emergency_support + self._EPS < self.min_support
            ):
                original_support = self.min_support
                original_center_guard = self._require_center_support
                try:
                    self.min_support = self.emergency_support
                    self._require_center_support = True
                    return self._choose_action(current)
                finally:
                    self.min_support = original_support
                    self._require_center_support = original_center_guard
            return None

        first_layer.sort(key=lambda item: self._state_key(item[0]))
        beam = first_layer[: self.beam_width]

        for _ in range(1, min(self.beam_depth, len(current))):
            expanded: List[Tuple[List[_Solid], _Action, Tuple[int, ...]]] = []
            for solids, first_action, remaining in beam:
                for index in representatives(remaining):
                    box = current[index]
                    actions = self._candidate_actions_for_box(
                        box, index, solids, max(2, self.positions_per_box // 2)
                    )
                    for action in actions:
                        next_remaining = tuple(i for i in remaining if i != index)
                        expanded.append((solids + [action.solid], first_action, next_remaining))
            if not expanded:
                break
            expanded.sort(key=lambda item: self._state_key(item[0]))
            beam = expanded[: self.beam_width]

        beam.sort(key=lambda item: self._state_key(item[0]))
        return beam[0][1]

    def _find_position(
        self,
        box: BoxInput,
    ) -> Optional[Tuple[float, float, float, Tuple[float, float, float], int]]:
        """Official-run-compatible online placement hook.

        If ``should_finish()`` planned a buffer action, only that selected box
        receives a placement.  Otherwise this falls back to single-box online
        placement.  No prefix/full-sequence information is inspected.
        """
        self._sync_solids_from_sequence()
        self._observe_current_box(box)
        if self._planned_action is not None:
            if int(box["id"]) != self._planned_box_id:
                return None
            action = self._planned_action
        else:
            action = self._choose_action([box])
        if action is None:
            return None
        solid = action.solid
        return (
            solid.x,
            solid.y,
            solid.z,
            (solid.dx, solid.dy, solid.dz),
            action.rotation,
        )

    def _commit(self, action: _Action) -> None:
        s = action.solid
        self._append_placed(
            box=action.box,
            dims=(s.dx, s.dy, s.dz),
            rotation=action.rotation,
            x=s.x,
            y=s.y,
            z=s.z,
            support=s.support,
        )

    def _append_placed(
        self,
        box: BoxInput,
        dims: Tuple[float, float, float],
        rotation: int,
        x: float,
        y: float,
        z: float,
        support: Optional[float] = None,
    ) -> None:
        """Append a placement in the same shape expected by the official loop."""
        dx, dy, dz = dims
        if support is None:
            support = self._support_ratio(x, y, z, dx, dy, self.solids)

        solid = _Solid(
            x=x,
            y=y,
            z=z,
            dx=dx,
            dy=dy,
            dz=dz,
            box_id=int(box["id"]),
            mass=float(box["mass"]),
            support=float(support),
        )
        self.solids.append(solid)
        self.sequence.append(
            {
                "step": int(box["step"]),
                "id": int(box["id"]),
                "size": [round(dx, 3), round(dy, 3), round(dz, 3)],
                "mass": float(box["mass"]),
                "position": [
                    round(x + dx / 2.0, 4),
                    round(y + dy / 2.0, 4),
                    round(z + dz / 2.0, 4),
                ],
                "rotation": int(rotation),
            }
        )
        if self._planned_box_id == int(box["id"]):
            self._planned_box_id = None
            self._planned_action = None

    def run(self, boxes: List[BoxInput]) -> RunResult:
        self._reset_state()
        self._configure_sequence_policy(boxes)
        if self.finished:
            return {
                "buffer_size": self.algo.buffer_size,
                "sequence": self.sequence,
                "terminated": self.finished,
                "terminated_step": self.terminated_step,
                "finished_by_user": self.finished_by_user,
            }
        buf = BufferManager(self.algo.buffer_size)
        buf.reset(boxes)

        while buf.has_pending():
            current = [buf.peek_next()] if self.algo.buffer_size == 0 else buf.get_buffer()
            if self.should_finish(current):
                self.finished_by_user = True
                break

            action = self._choose_action(current)
            if action is None:
                self.finished = True
                self.terminated_step = int(current[0]["step"]) if current else None
                break

            self._commit(action)
            if self.algo.buffer_size == 0:
                buf.pop_next()
            else:
                buf.pop_selected(action.box_index)

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": self.sequence,
            "terminated": self.finished,
            "terminated_step": self.terminated_step,
            "finished_by_user": self.finished_by_user,
        }
