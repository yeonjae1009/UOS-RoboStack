from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from .sub36_candidates import Sub36CandidateGenerator, _Candidate, _Solid


class Sub36SunEnsembleCandidateGenerator(Sub36CandidateGenerator):
    """Diverse strategy-family candidate generator for PCT leaf slots.

    The generator deliberately does not emit one globally sorted heuristic list.
    Instead, several simple placement families compete for fixed leaf quotas so
    the policy can learn which family is useful in the current state.
    """

    FAMILY_NAMES = (
        "support_anchor",
        "low_smooth_layer",
        "floor_expansion",
        "wall_corner_anchor",
        "tight_fit_void_closure",
        "active_layer_coverage",
        "compact_center_of_mass",
        "random_feasible",
    )
    DEFAULT_QUOTAS = (14, 14, 14, 12, 12, 12, 10, 12)

    def __init__(self, pallet_size: Iterable[float], **kwargs) -> None:
        super().__init__(pallet_size, **kwargs)
        self._family_count = len(self.FAMILY_NAMES)

    def generate(self, space, next_box, density: float, setting: int, limit: int) -> list[tuple[float, ...]]:
        solids = self._solids_from_space(space)
        candidates = self._candidate_actions(next_box, solids, self.min_support, require_center=False)
        if not candidates and self.emergency_support + self.EPS < self.min_support:
            candidates = self._candidate_actions(next_box, solids, self.emergency_support, require_center=True)

        feasible = self._verified_candidates(space, candidates, density, setting)
        if not feasible:
            return []

        limit = int(limit)
        quotas = self._scaled_quotas(limit)
        scored = self._score_by_family(feasible, solids)

        out: list[tuple[float, ...]] = []
        seen: set[tuple[float, ...]] = set()

        taken_by_family = [0 for _ in quotas]
        cursor_by_family = [0 for _ in quotas]
        progress = True
        while progress and len(out) < limit:
            progress = False
            for family_id, quota in enumerate(quotas):
                if taken_by_family[family_id] >= quota or len(out) >= limit:
                    continue
                entries = scored[family_id]
                while cursor_by_family[family_id] < len(entries):
                    score, candidate = entries[cursor_by_family[family_id]]
                    cursor_by_family[family_id] += 1
                    key = self._candidate_key(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(self._to_leaf_tuple(candidate, family_id, score))
                    taken_by_family[family_id] += 1
                    progress = True
                    break

        if len(out) < limit:
            leftovers: list[tuple[float, int, _Candidate]] = []
            for family_id, entries in scored.items():
                leftovers.extend((score, family_id, candidate) for score, candidate in entries)
            leftovers.sort(key=lambda item: (-item[0], item[1], self._candidate_key(item[2])))
            for score, family_id, candidate in leftovers:
                if len(out) >= limit:
                    break
                key = self._candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                out.append(self._to_leaf_tuple(candidate, family_id, score))

        return out

    def _verified_candidates(
        self,
        space,
        candidates: list[_Candidate],
        density: float,
        setting: int,
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
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
            key = self._candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out

    def _scaled_quotas(self, limit: int) -> list[int]:
        if limit <= 0:
            return [0 for _ in self.DEFAULT_QUOTAS]
        base = list(self.DEFAULT_QUOTAS)
        total = sum(base)
        quotas = [max(1, int(round(limit * q / total))) for q in base]
        while sum(quotas) > limit:
            idx = max(range(len(quotas)), key=lambda i: quotas[i])
            quotas[idx] -= 1
        while sum(quotas) < limit:
            idx = min(range(len(quotas)), key=lambda i: quotas[i] / max(base[i], 1))
            quotas[idx] += 1
        return quotas

    def _score_by_family(
        self,
        candidates: list[_Candidate],
        solids: list[_Solid],
    ) -> dict[int, list[tuple[float, _Candidate]]]:
        scored: dict[int, list[tuple[float, _Candidate]]] = defaultdict(list)
        active_z = self._active_layer(solids)
        current_com = self._current_com(solids)

        for candidate in candidates:
            solid = _Solid(
                candidate.x,
                candidate.y,
                candidate.z,
                candidate.dx,
                candidate.dy,
                candidate.dz,
                0.0,
                candidate.support,
            )
            wall_contact = self._wall_contact_score(solid)
            side_contact = self._horizontal_contact(solid, solids)
            tight = self._tight_gap_score(solid, solids)
            height_score = self._low_height_score(solid)
            random_score = self._stable_random_score(candidate, solids)
            center_score = self._compact_score(solid, current_com)
            active_score = 1.0 / (1.0 + abs(candidate.z - active_z) / max(self.pallet[2], 1.0e-6))
            floor_bonus = 1.0 if candidate.z <= self.EPS else 0.0

            scored[0].append((0.70 * candidate.support + 0.20 * height_score + 0.10 * min(side_contact, 1.0), candidate))
            scored[1].append((0.75 * height_score + 0.25 * (1.0 - min(candidate.top / self.pallet[2], 1.0)), candidate))
            scored[2].append((floor_bonus * (0.60 + 0.25 * wall_contact + 0.15 * center_score), candidate))
            scored[3].append((0.70 * wall_contact + 0.30 * min(side_contact, 1.0), candidate))
            scored[4].append((0.55 * tight + 0.25 * min(side_contact, 1.0) + 0.20 * candidate.support, candidate))
            scored[5].append((0.60 * active_score + 0.25 * candidate.support + 0.15 * height_score, candidate))
            scored[6].append((0.70 * center_score + 0.20 * height_score + 0.10 * candidate.support, candidate))
            scored[7].append((random_score, candidate))

        for family_id, entries in scored.items():
            entries.sort(
                key=lambda item: (
                    -item[0],
                    self._stable_random_score(item[1], solids) if family_id == 7 else 0.0,
                    self._candidate_key(item[1]),
                )
            )
            scored[family_id] = [(self._normalize_score(score), candidate) for score, candidate in entries]
        return scored

    def _active_layer(self, solids: list[_Solid]) -> float:
        if not solids:
            return 0.0
        area_by_top: dict[float, float] = defaultdict(float)
        for solid in solids:
            area_by_top[round(solid.top, 6)] += solid.dx * solid.dy
        return max(area_by_top.items(), key=lambda item: (item[1], -item[0]))[0]

    def _current_com(self, solids: list[_Solid]) -> tuple[float, float]:
        if not solids:
            return (self.pallet[0] * 0.5, self.pallet[1] * 0.5)
        total = sum(max(s.volume, 1.0e-9) for s in solids)
        cx = sum((s.x + s.dx * 0.5) * max(s.volume, 1.0e-9) for s in solids) / total
        cy = sum((s.y + s.dy * 0.5) * max(s.volume, 1.0e-9) for s in solids) / total
        return (cx, cy)

    def _wall_contact_score(self, solid: _Solid) -> float:
        length, width, _ = self.pallet
        score = 0.0
        if abs(solid.x) <= self.EPS or abs(solid.x1 - length) <= self.EPS:
            score += 0.5
        if abs(solid.y) <= self.EPS or abs(solid.y1 - width) <= self.EPS:
            score += 0.5
        return min(score, 1.0)

    def _tight_gap_score(self, solid: _Solid, solids: list[_Solid]) -> float:
        length, width, _ = self.pallet
        gaps = [solid.x, solid.y, length - solid.x1, width - solid.y1]
        for other in solids:
            if self._overlap_1d(solid.z, solid.top, other.z, other.top) <= self.EPS:
                continue
            if self._overlap_1d(solid.y, solid.y1, other.y, other.y1) > self.EPS:
                if other.x1 <= solid.x + self.EPS:
                    gaps.append(solid.x - other.x1)
                if solid.x1 <= other.x + self.EPS:
                    gaps.append(other.x - solid.x1)
            if self._overlap_1d(solid.x, solid.x1, other.x, other.x1) > self.EPS:
                if other.y1 <= solid.y + self.EPS:
                    gaps.append(solid.y - other.y1)
                if solid.y1 <= other.y + self.EPS:
                    gaps.append(other.y - solid.y1)
        small_gap = min((g for g in gaps if g >= -self.EPS), default=max(length, width))
        return 1.0 / (1.0 + small_gap / max(min(solid.dx, solid.dy), 1.0e-6))

    def _low_height_score(self, solid: _Solid) -> float:
        return max(0.0, 1.0 - solid.top / max(self.pallet[2], 1.0e-6))

    def _compact_score(self, solid: _Solid, current_com: tuple[float, float]) -> float:
        target_x, target_y = current_com
        cx = solid.x + solid.dx * 0.5
        cy = solid.y + solid.dy * 0.5
        dist = ((cx - target_x) ** 2 + (cy - target_y) ** 2) ** 0.5
        diag = (self.pallet[0] ** 2 + self.pallet[1] ** 2) ** 0.5
        return max(0.0, 1.0 - dist / max(diag * 0.5, 1.0e-6))

    def _stable_random_score(self, candidate: _Candidate, solids: list[_Solid]) -> float:
        state = (len(solids),) + self._candidate_key(candidate)
        digest = hashlib.blake2b(repr(state).encode("ascii"), digest_size=8).digest()
        return int.from_bytes(digest, "big") / float(2**64 - 1)

    def _candidate_key(self, candidate: _Candidate) -> tuple[float, ...]:
        return (
            round(candidate.x, 6),
            round(candidate.y, 6),
            round(candidate.z, 6),
            round(candidate.dx, 6),
            round(candidate.dy, 6),
        )

    def _normalize_score(self, score: float) -> float:
        return max(0.0, min(1.0, float(score)))

    def _to_leaf_tuple(self, candidate: _Candidate, family_id: int, score: float) -> tuple[float, ...]:
        family_norm = float(family_id) / float(max(self._family_count - 1, 1))
        return (
            round(candidate.x, 6),
            round(candidate.y, 6),
            round(candidate.z, 6),
            round(candidate.x + candidate.dx, 6),
            round(candidate.y + candidate.dy, 6),
            round(candidate.z + candidate.dz, 6),
            family_norm,
            self._normalize_score(score),
        )
