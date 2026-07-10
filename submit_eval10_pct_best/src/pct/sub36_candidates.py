from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class _Solid:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
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
class _Candidate:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    rotation: int
    support: float
    local_key: tuple[float, ...]

    @property
    def top(self) -> float:
        return self.z + self.dz


class Sub36CandidateGenerator:
    EPS = 1.0e-7

    def __init__(
        self,
        pallet_size: Iterable[float],
        *,
        min_support: float = 0.64,
        emergency_support: float = 0.45,
        allow_rotation: bool = True,
        column_low_threshold: float = 1.25,
        column_default_threshold: float = 0.50,
    ) -> None:
        px, py, pz = pallet_size
        self.pallet = (float(px), float(py), float(pz))
        self.min_support = float(min_support)
        self.emergency_support = float(emergency_support)
        self.allow_rotation = bool(allow_rotation)
        self.column_low_threshold = float(column_low_threshold)
        self.column_default_threshold = float(column_default_threshold)

    def generate(self, space, next_box, density: float, setting: int, limit: int) -> list[tuple[float, ...]]:
        solids = self._solids_from_space(space)
        normal = self._candidate_actions(next_box, solids, self.min_support, require_center=False)
        candidates = normal
        if not candidates and self.emergency_support + self.EPS < self.min_support:
            candidates = self._candidate_actions(
                next_box,
                solids,
                self.emergency_support,
                require_center=True,
            )

        out: list[tuple[float, ...]] = []
        seen: set[tuple[float, ...]] = set()
        for candidate in candidates:
            ok, resolved_z = space.drop_box_virtual(
                [candidate.dx, candidate.dy, candidate.dz],
                (candidate.x, candidate.y),
                False,
                density,
                setting,
                returnH=True,
            )
            if not ok or abs(float(resolved_z) - candidate.z) > 1.0e-5:
                continue
            key = (
                round(candidate.x, 6),
                round(candidate.y, 6),
                round(float(resolved_z), 6),
                round(candidate.dx, 6),
                round(candidate.dy, 6),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(
                (
                    key[0],
                    key[1],
                    key[2],
                    round(key[0] + key[3], 6),
                    round(key[1] + key[4], 6),
                    round(key[2] + candidate.dz, 6),
                )
            )
            if len(out) >= int(limit):
                break
        return out

    def _solids_from_space(self, space) -> list[_Solid]:
        solids: list[_Solid] = []
        for box in space.boxes:
            solids.append(
                _Solid(
                    x=float(box.lx),
                    y=float(box.ly),
                    z=float(box.lz),
                    dx=float(box.x),
                    dy=float(box.y),
                    dz=float(box.z),
                    mass=float(box.mass),
                    support=self._support_ratio(
                        float(box.lx),
                        float(box.ly),
                        float(box.lz),
                        float(box.x),
                        float(box.y),
                        solids,
                    ),
                )
            )
        return solids

    def _orientations(self, box) -> list[tuple[float, float, float, int]]:
        sx, sy, sz = map(float, box)
        result = [(sx, sy, sz, 0)]
        if self.allow_rotation and abs(sx - sy) > self.EPS:
            result.append((sy, sx, sz, 90))
        result.sort(key=lambda item: (-item[0], item[1]))
        return result

    @staticmethod
    def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    def _collides(self, candidate: _Solid, solids: list[_Solid]) -> bool:
        e = self.EPS
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
        solids: list[_Solid],
    ) -> float:
        if z <= self.EPS:
            return 1.0

        area = 0.0
        for support in solids:
            if abs(support.top - z) > self.EPS:
                continue
            area += self._overlap_1d(x, x + dx, support.x, support.x1) * self._overlap_1d(
                y, y + dy, support.y, support.y1
            )
        return min(1.0, area / max(dx * dy, 1.0e-12))

    def _center_supported(
        self,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        solids: list[_Solid],
    ) -> bool:
        if z <= self.EPS:
            return True

        cx = x + dx / 2.0
        cy = y + dy / 2.0
        for support in solids:
            if abs(support.top - z) > self.EPS:
                continue
            if (
                support.x - self.EPS <= cx <= support.x1 + self.EPS
                and support.y - self.EPS <= cy <= support.y1 + self.EPS
            ):
                return True
        return False

    def _horizontal_contact(self, candidate: _Solid, solids: list[_Solid]) -> float:
        contact = 0.0
        length, width, _ = self.pallet
        if abs(candidate.x) <= self.EPS or abs(candidate.x1 - length) <= self.EPS:
            contact += candidate.dy
        if abs(candidate.y) <= self.EPS or abs(candidate.y1 - width) <= self.EPS:
            contact += candidate.dx

        for other in solids:
            z_overlap = self._overlap_1d(candidate.z, candidate.top, other.z, other.top)
            if z_overlap <= self.EPS:
                continue
            if abs(candidate.x - other.x1) <= self.EPS or abs(candidate.x1 - other.x) <= self.EPS:
                contact += z_overlap * self._overlap_1d(candidate.y, candidate.y1, other.y, other.y1)
            if abs(candidate.y - other.y1) <= self.EPS or abs(candidate.y1 - other.y) <= self.EPS:
                contact += z_overlap * self._overlap_1d(candidate.x, candidate.x1, other.x, other.x1)
        return contact

    def _plane_positions(
        self,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        solids: list[_Solid],
    ) -> list[tuple[float, float]]:
        length, width, _ = self.pallet
        positions = {
            (0.0, 0.0),
            (length - dx, 0.0),
            (0.0, width - dy),
            (length - dx, width - dy),
            ((length - dx) / 2.0, (width - dy) / 2.0),
        }

        if z > self.EPS:
            for s in solids:
                if abs(s.top - z) <= self.EPS:
                    sx = (s.x, s.x1 - dx, s.x + (s.dx - dx) / 2.0)
                    sy = (s.y, s.y1 - dy, s.y + (s.dy - dy) / 2.0)
                    positions.update((x, y) for x in sx for y in sy)

        for s in solids:
            if z < s.top - self.EPS and z + dz > s.z + self.EPS:
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
                if -self.EPS <= x <= length - dx + self.EPS
                and -self.EPS <= y <= width - dy + self.EPS
            },
            key=lambda p: (p[1], p[0]),
        )

    def _required_support_ratio(
        self,
        dx: float,
        dy: float,
        dz: float,
        base_support: float,
        require_center: bool,
    ) -> float:
        if require_center:
            return max(base_support, 0.50)
        area = dx * dy
        if area <= 0.050:
            return min(base_support, 0.60)
        if area >= 0.060 and dz >= 0.18:
            return max(base_support, 0.65)
        return base_support

    def _column_threshold(self, box) -> float:
        _, _, sz = map(float, box)
        if sz <= 0.135 + self.EPS:
            return self.column_low_threshold
        return self.column_default_threshold

    def _candidate_actions(
        self,
        box,
        solids: list[_Solid],
        base_support: float,
        require_center: bool,
    ) -> list[_Candidate]:
        planes = {0.0}
        planes.update(round(s.top, 6) for s in solids if s.top < self.pallet[2] - self.EPS)
        actions: list[_Candidate] = []

        for dx, dy, dz, rotation in self._orientations(box):
            required_support = self._required_support_ratio(dx, dy, dz, base_support, require_center)
            required_area = required_support * dx * dy
            useful_planes = [0.0]
            for z in planes:
                if z <= self.EPS:
                    continue
                available_area = sum(s.dx * s.dy for s in solids if abs(s.top - z) <= self.EPS)
                if available_area + self.EPS >= required_area:
                    useful_planes.append(z)

            for z in sorted(useful_planes):
                if z + dz > self.pallet[2] + self.EPS:
                    continue
                for x, y in self._plane_positions(z, dx, dy, dz, solids):
                    support = self._support_ratio(x, y, z, dx, dy, solids)
                    if support + self.EPS < required_support:
                        continue
                    if require_center and not self._center_supported(x, y, z, dx, dy, solids):
                        continue
                    solid = _Solid(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, mass=0.0, support=support)
                    if self._collides(solid, solids):
                        continue
                    overhang = 1.0 - support
                    horizontal_contact = self._horizontal_contact(solid, solids)
                    local_key = (
                        round(z, 6),
                        round(overhang, 6),
                        -round(horizontal_contact, 6),
                        round(y, 6),
                        round(x, 6),
                    )
                    actions.append(_Candidate(x, y, z, dx, dy, dz, rotation, support, local_key))

        actions.sort(key=lambda a: a.local_key)
        return self._prioritize_actions(actions, solids, box)

    def _prioritize_actions(
        self,
        actions: list[_Candidate],
        solids: list[_Solid],
        box,
    ) -> list[_Candidate]:
        floor_actions: list[_Candidate] = []
        column_actions: list[_Candidate] = []
        other_actions: list[_Candidate] = []

        for action in actions:
            if action.z <= self.EPS:
                floor_actions.append(action)
                continue
            matched_column = False
            for support in solids:
                if (
                    abs(support.top - action.z) <= self.EPS
                    and abs(support.x - action.x) <= self.EPS
                    and abs(support.y - action.y) <= self.EPS
                    and abs(support.dx - action.dx) <= self.EPS
                    and abs(support.dy - action.dy) <= self.EPS
                ):
                    column_actions.append(action)
                    matched_column = True
                    break
            if not matched_column:
                other_actions.append(action)

        def compact_key(action: _Candidate) -> tuple[float, ...]:
            solid = _Solid(action.x, action.y, action.z, action.dx, action.dy, action.dz, 0.0, action.support)
            return (
                round(solid.top, 6),
                -round(self._horizontal_contact(solid, solids), 6),
                -round(solid.y, 6),
                round(solid.x, 6),
            )

        floor_actions.sort(key=compact_key)
        column_actions.sort(key=compact_key)
        other_actions.sort(key=compact_key)
        threshold = self._column_threshold(box)
        low_columns = [action for action in column_actions if action.top <= threshold + self.EPS]
        high_columns = [action for action in column_actions if action.top > threshold + self.EPS]

        ordered: list[_Candidate] = []
        seen: set[tuple[float, ...]] = set()
        for group in (low_columns, floor_actions, other_actions, high_columns, actions):
            for action in group:
                key = (
                    round(action.x, 6),
                    round(action.y, 6),
                    round(action.z, 6),
                    round(action.dx, 6),
                    round(action.dy, 6),
                )
                if key not in seen:
                    ordered.append(action)
                    seen.add(key)
        return ordered
