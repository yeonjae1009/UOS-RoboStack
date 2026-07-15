"""
PCT 온라인 패킹 드라이버 (torch/gym 없이 순수 numpy).

원본 pct_envs/PctContinuous0/bin3D.py 의 PackingContinuous 에서
관찰값 생성(cur_observation/get_possible_position)과 배치(step) 로직만 추출하고,
torch.seed / gym / box_creator 결합을 제거했다. 박스는 외부에서 한 개씩 주입한다.

흐름:
  reset()                  -> 빈 팔레트로 초기화
  observe(next_box)        -> 현재 박스에 대한 관찰값(flat) 반환  (신경망 입력)
  place(leaf_node[0:6])    -> 선택된 잎 노드 위치에 배치, 성공 여부 반환
  packed                   -> [[x,y,z, lx,ly,lz, bin], ...] 배치 기록(적재 순서)
"""
import numpy as np
import copy
from .space import Space


class Packer:
    def __init__(self, container_size, size_minimum,
                 internal_node_holder=200, leaf_node_holder=100, setting=1,
                 leaf_score_trial_limit=200):
        self.bin_size = list(container_size)
        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = 1
        self.setting = setting
        self.leaf_score_trial_limit = int(leaf_score_trial_limit)
        self.size_minimum = size_minimum
        self.space = Space(*self.bin_size, size_minimum, internal_node_holder)
        self.next_box_vec = np.zeros((self.next_holder, 9))
        self.next_box = None
        self.next_den = 1
        self.packed = []

    def reset(self):
        self.space.reset()
        self.packed = []

    # 현재 박스에 대한 잎 노드(배치 후보) 생성
    def get_possible_position(self):
        try:
            positions = self._collect_candidate_positions()
            feasible = self._feasible_leaf_candidates(positions)
            if feasible:
                leaves = self._select_scored_leaves(feasible)
                if leaves:
                    return self._leaf_array(leaves)
        except Exception:
            pass
        return self._fallback_possible_position()

    def _fallback_possible_position(self):
        candidates = self._feasible_leaf_candidates(
            self.space.EMSPoint(self.next_box, self.setting),
            limit=self.leaf_node_holder,
        )
        return self._leaf_array([candidate["leaf"] for candidate in candidates])

    def _leaf_array(self, leaves):
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        if len(leaves) != 0:
            leaf_node_vec[0:len(leaves)] = np.array(leaves[:self.leaf_node_holder])
        return leaf_node_vec

    def _collect_candidate_positions(self):
        positions = []
        ems_positions = self.space.EMSPoint(self.next_box, self.setting)
        positions.extend(self._normalise_positions(ems_positions))
        try:
            event_positions = self.space.EventPoint(self.next_box, self.setting)
            positions.extend(self._normalise_positions(event_positions))
        except Exception:
            pass
        return self._deduplicate_positions(positions)

    def _normalise_positions(self, positions):
        normalised = []
        for position in positions:
            if len(position) != 6:
                continue
            normalised.append(tuple(round(float(v), 6) for v in position))
        return normalised

    def _deduplicate_positions(self, positions):
        seen = set()
        exact = []
        for position in positions:
            if position in seen:
                continue
            seen.add(position)
            exact.append(position)

        exact.sort(key=self._position_sort_key)
        selected = []
        selected_by_size = {}
        for position in exact:
            key = self._size_key(position)
            near = False
            for previous in selected_by_size.get(key, []):
                if self._is_near_duplicate(position, previous):
                    near = True
                    break
            if near:
                continue
            selected.append(position)
            selected_by_size.setdefault(key, []).append(position)
        return selected

    def _feasible_leaf_candidates(self, positions, limit=None):
        candidates = []
        seen = set()
        for position in positions:
            xs, ys, zs, xe, ye, ze = position
            x = round(xe - xs, 6)
            y = round(ye - ys, 6)
            z = round(ze - zs, 6)
            ok, height = self.space.drop_box_virtual(
                [x, y, z], (xs, ys), False, self.next_den, self.setting, returnH=True
            )
            if not ok:
                continue
            bottom_z = round(float(height), 6)
            leaf = [
                round(xs, 6),
                round(ys, 6),
                bottom_z,
                round(xs + x, 6),
                round(ys + y, 6),
                self.bin_size[2],
                0,
                0,
                1,
            ]
            key = tuple(round(float(v), 6) for v in leaf[0:5])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "leaf": leaf,
                "position_key": self._position_sort_key(position),
                "size_key": (round(x, 6), round(y, 6), round(z, 6)),
                "bottom_z": bottom_z,
            })
            if limit is not None and len(candidates) >= limit:
                break
        return candidates

    def _select_scored_leaves(self, candidates):
        trial_limit = int(getattr(self, "leaf_score_trial_limit", 200))
        candidates = sorted(
            candidates,
            key=lambda c: (-self._cheap_candidate_score(c["leaf"]), c["position_key"]),
        )
        if trial_limit <= 0:
            return [candidate["leaf"] for candidate in candidates[:self.leaf_node_holder]]

        base_metrics = self._ems_metrics(self.space)
        scored = []
        for idx, candidate in enumerate(candidates[:trial_limit]):
            score = self._score_candidate(candidate["leaf"], base_metrics)
            if score is None:
                continue
            scored.append((score, candidate["position_key"], idx, candidate["leaf"]))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        leaves = []
        seen = set()
        for item in scored:
            leaf = item[3]
            key = tuple(round(float(v), 6) for v in leaf[:6])
            if key in seen:
                continue
            leaves.append(leaf)
            seen.add(key)
            if len(leaves) >= self.leaf_node_holder:
                return leaves

        for candidate in candidates:
            leaf = candidate["leaf"]
            key = tuple(round(float(v), 6) for v in leaf[:6])
            if key in seen:
                continue
            leaves.append(leaf)
            seen.add(key)
            if len(leaves) >= self.leaf_node_holder:
                break
        return leaves

    def _cheap_candidate_score(self, leaf):
        bottom_z = float(leaf[2])
        return self._anchor_quality(leaf) - bottom_z / max(float(self.bin_size[2]), 1e-6)

    def _score_candidate(self, leaf, base_metrics):
        trial = copy.deepcopy(self)
        if not trial.place(leaf):
            return None

        trial_metrics = self._ems_metrics(trial.space)
        pallet_height = max(float(self.bin_size[2]), 1e-6)
        low_void_gain = base_metrics["low"] - trial_metrics["low"]
        usable_void_gain = base_metrics["usable"] - trial_metrics["usable"]
        well_delta = trial_metrics["well"] - base_metrics["well"]
        sliver_delta = trial_metrics["sliver"] - base_metrics["sliver"]
        bottom_z = float(leaf[2])

        return (
            2.0 * low_void_gain
            + 0.6 * usable_void_gain
            - 1.6 * max(well_delta, 0.0)
            - 0.8 * max(sliver_delta, 0.0)
            + 0.25 * self._anchor_quality(leaf)
            - 0.30 * bottom_z / pallet_height
            - 0.20 * self._rough_top(trial.space)
        )

    def _ems_metrics(self, space):
        ems = np.array(space.EMS[0:space.NOEMS], dtype=float)
        if len(ems) == 0:
            return {"usable": 0.0, "low": 0.0, "well": 0.0, "sliver": 0.0}

        size = np.maximum(ems[:, 3:6] - ems[:, 0:3], 0.0)
        volume = size[:, 0] * size[:, 1] * size[:, 2]
        sorted_size = np.sort(size, axis=1)
        usable_mask = (
            (sorted_size[:, 0] + 1e-9 >= 0.13)
            & (sorted_size[:, 1] + 1e-9 >= 0.17)
            & (sorted_size[:, 2] + 1e-9 >= 0.17)
            & (volume + 1e-12 >= 0.0035)
        )
        current_max_top = self._current_max_top(space)
        low_mask = usable_mask & (ems[:, 2] < current_max_top - 0.006)
        well_mask = low_mask & (size[:, 2] >= 0.35) & (np.minimum(size[:, 0], size[:, 1]) <= 0.22)
        sliver_mask = (
            (np.min(size, axis=1) <= 0.08)
            & (np.max(size, axis=1) >= 0.35)
            & (volume >= 0.001)
        )
        return {
            "usable": float(np.sum(volume[usable_mask])),
            "low": float(np.sum(volume[low_mask])),
            "well": float(np.sum(volume[well_mask])),
            "sliver": float(np.sum(volume[sliver_mask])),
        }

    def _anchor_quality(self, leaf):
        xs, ys, zs, xe, ye = [float(v) for v in leaf[0:5]]
        eps = 1e-6
        wall_hits = int(abs(xs) <= eps) + int(abs(ys) <= eps)
        wall_hits += int(abs(xe - self.bin_size[0]) <= eps) + int(abs(ye - self.bin_size[1]) <= eps)
        score = min(0.4, 0.2 * wall_hits)
        if (abs(xs) <= eps or abs(xe - self.bin_size[0]) <= eps) and (
            abs(ys) <= eps or abs(ye - self.bin_size[1]) <= eps
        ):
            score += 0.25

        flush_hits = 0
        for box in self.space.boxes:
            if abs(float(box.lz) - zs) > 1e-6:
                continue
            bx0, by0 = float(box.lx), float(box.ly)
            bx1, by1 = bx0 + float(box.x), by0 + float(box.y)
            y_overlap = min(ye, by1) - max(ys, by0)
            x_overlap = min(xe, bx1) - max(xs, bx0)
            if y_overlap > 1e-6 and (abs(xs - bx1) <= eps or abs(xe - bx0) <= eps):
                flush_hits += 1
            if x_overlap > 1e-6 and (abs(ys - by1) <= eps or abs(ye - by0) <= eps):
                flush_hits += 1
        score += min(0.35, 0.15 * flush_hits)
        return min(1.0, score)

    def _rough_top(self, space):
        if len(space.boxes) == 0:
            return 0.0
        tops = np.array([float(box.lz + box.z) for box in space.boxes], dtype=float)
        return float(np.std(tops) / max(float(self.bin_size[2]), 1e-6))

    def _current_max_top(self, space):
        if len(space.boxes) == 0:
            return 0.0
        return max(float(box.lz + box.z) for box in space.boxes)

    def _position_sort_key(self, position):
        return tuple(round(float(v), 6) for v in position)

    def _size_key(self, position):
        xs, ys, zs, xe, ye, ze = position
        return (round(xe - xs, 6), round(ye - ys, 6), round(ze - zs, 6))

    def _is_near_duplicate(self, a, b):
        ax = (a[0] + a[3]) / 2.0
        ay = (a[1] + a[4]) / 2.0
        bx = (b[0] + b[3]) / 2.0
        by = (b[1] + b[4]) / 2.0
        center_dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return center_dist < 0.01 and abs(a[2] - b[2]) < 0.006

    # 관찰값(flat vector) 생성: [내부노드 | 잎노드 | 다음박스]
    #   density: setting 3 학습 시 next_box_vec[:,0] 에 들어간 정규화 밀도(=mass/부피/DENSITY_MAX).
    #            잎 노드 생성(drop_box_virtual)이 next_den 으로 안정성을 보므로 관찰 전에 설정해야 함.
    def observe(self, next_box, density=1.0):
        self.next_box = [float(next_box[0]), float(next_box[1]), float(next_box[2])]
        self.next_den = float(density)
        boxes = [self.space.box_vec]
        leaf_nodes = [self.get_possible_position()]
        nb_sorted = sorted(list(self.next_box))
        self.next_box_vec[:, 3:6] = nb_sorted
        self.next_box_vec[:, 0] = self.next_den
        self.next_box_vec[:, -1] = 1
        return np.reshape(np.concatenate((*boxes, *leaf_nodes, self.next_box_vec)), (-1))

    # 잎 노드(앞 6개 값: xs,ys,zs,xe,ye,ze)를 실제 박스 배치 동작으로 변환
    def _leaf_to_action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0:
            return (0, 0, 0), tuple(self.next_box)
        x = round(leaf_node[3] - leaf_node[0], 6)
        y = round(leaf_node[4] - leaf_node[1], 6)
        record = [0, 1, 2]
        for r in record:
            if abs(x - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        for r in record:
            if abs(y - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        z = self.next_box[record[0]]
        action = (0, leaf_node[0], leaf_node[1])
        next_box = (x, y, z)
        return action, next_box

    # 선택된 잎 노드에 박스 배치. 성공하면 True 와 packed 기록 추가.
    def place(self, leaf_node):
        action, next_box = self._leaf_to_action(leaf_node)
        idx = [round(action[1], 6), round(action[2], 6)]
        rotation_flag = action[0]
        ok = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting)
        if not ok:
            return False
        pb = self.space.boxes[-1]
        self.space.GENEMS([pb.lx, pb.ly, pb.lz,
                           round(pb.lx + pb.x, 6),
                           round(pb.ly + pb.y, 6),
                           round(pb.lz + pb.z, 6)])
        self.packed.append([pb.x, pb.y, pb.z, pb.lx, pb.ly, pb.lz, 0])
        return True

    def get_ratio(self):
        return self.space.get_ratio()
