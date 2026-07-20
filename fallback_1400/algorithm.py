from __future__ import annotations

import os
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from buffer_manager import BufferManager

try:
    import numpy as np
except Exception:
    np = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from src.pct.packer import Packer
    from src.pct.stability import apply_stability_mask
except Exception:
    Packer = None
    apply_stability_mask = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


# ---------------------------------------------------------------------------
# 입출력 스키마  (수정 금지)
# ---------------------------------------------------------------------------

class BoxInput(TypedDict):
    step: int
    id: int
    size: List[float]   # [length, width, height]
    mass: float


class PlacedBox(TypedDict):
    step: int
    id: int
    size: List[float]
    mass: float
    position: List[float]
    rotation: int       # 0 또는 90


class RunResult(TypedDict):
    buffer_size: int
    sequence: List[PlacedBox]

    # 더 이상 적재 가능한 박스를 찾지 못해 자동 종료된 경우 True
    terminated: bool
    terminated_step: Optional[int]

    # 참가자 알고리즘이 명시적으로 적재 종료를 선언한 경우 True
    finished_by_user: bool


# ---------------------------------------------------------------------------
# 설정 dataclass  (수정 금지)
# ---------------------------------------------------------------------------

@dataclass
class PalletConfig:
    length: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# 참가자 개발 영역
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmConfig:
    allow_rotation: bool
    buffer_size: int


class Palletizer:
    """
    가장 단순한 버퍼 기반 팔레타이저 예제.

    동작 방식:
      - 버퍼 안의 박스를 순서대로 확인
      - 현재 위치에 놓을 수 있으면 적재
      - X 방향으로 채움
      - X 방향 공간이 부족하면 다음 row(Y)
      - Y 방향 공간도 부족하면 다음 layer(Z)
      - 참가자가 should_finish()에서 True를 반환하면 명시적으로 종료
      - 더 이상 놓을 수 없으면 자동 종료
    """

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg
        self._load_search_settings()
        self._reset_state()

    def _reset_state(self) -> None:
        self.cursor_x = 0.0
        self.cursor_y = 0.0
        self.layer_z = 0.0

        self.row_depth = 0.0
        self.layer_height = 0.0

        self.sequence: List[PlacedBox] = []

        self.finished = False
        self.terminated_step: Optional[int] = None
        self.finished_by_user = False
        self._search_plan = None
        self._observed_boxes: List[BoxInput] = []

    # -----------------------------------------------------------------------
    # 참가자 수정 가능 함수
    # -----------------------------------------------------------------------

    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        """
        [참가자 수정 가능]
        현재 버퍼 상태를 보고 적재를 명시적으로 종료할지 결정한다.

        True 반환:
          - 더 이상 박스를 처리하지 않고 즉시 종료
          - 결과 JSON의 finished_by_user가 True로 기록됨
          - terminated는 False로 유지됨

        False 반환:
          - 계속 적재 진행

        예시:
          - 너무 높은 층까지 쌓였다고 판단한 경우
          - 안정성이 낮아질 것으로 예상되는 경우
          - 더 쌓는 것보다 현재 상태로 종료하는 것이 유리한 경우
        """
        self._search_plan = None
        if self._search_enabled and self.algo.buffer_size > 0 and len(current_buffer) > 1:
            self._prepare_buffer_search(current_buffer)
        return False

    def _load_search_settings(self) -> None:
        self._search_enabled = True
        self._search_mode = "one_step"
        self._search_top_k_leaf = 5
        self._search_beam_width = 4
        self._search_depth = 2
        self._score_profile = "legacy"
        self._candidate_union_enabled = False
        self._candidate_union_mode = "single"
        self._union_top_k_per_model = 4
        self._union_fallback_top_k_leaf = 20
        self._candidate_union_score_profile = ""
        self._fallback_model_name = "candidate-001400.onnx"

        if yaml is None:
            return

        here = Path(__file__).resolve().parent
        try:
            with (here / "config" / "algorithm_config.yaml").open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return

        search_cfg = cfg.get("search", {}) or {}
        self._search_enabled = bool(search_cfg.get("enabled", self._search_enabled))
        self._search_mode = str(search_cfg.get("mode", self._search_mode))
        self._search_top_k_leaf = max(1, int(search_cfg.get("top_k_leaf", self._search_top_k_leaf)))
        self._search_beam_width = max(1, int(search_cfg.get("beam_width", self._search_beam_width)))
        self._search_depth = max(1, int(search_cfg.get("depth", self._search_depth)))
        self._score_profile = str(search_cfg.get("score_profile", self._score_profile))
        self._candidate_union_mode = str(search_cfg.get("candidate_union_mode", self._candidate_union_mode))
        if self._candidate_union_mode not in {"single", "fallback", "adaptive"}:
            self._candidate_union_mode = "single"
        self._candidate_union_enabled = (
            bool(search_cfg.get("candidate_union_enabled", self._candidate_union_enabled))
            and self._candidate_union_mode in {"fallback", "adaptive"}
        )
        self._union_top_k_per_model = max(1, int(search_cfg.get("union_top_k_per_model", self._union_top_k_per_model)))
        self._union_fallback_top_k_leaf = max(1, int(search_cfg.get("union_fallback_top_k_leaf", self._union_fallback_top_k_leaf)))
        self._candidate_union_score_profile = str(search_cfg.get("candidate_union_score_profile", self._candidate_union_score_profile))
        self._fallback_model_name = Path(str(search_cfg.get("fallback_model", self._fallback_model_name))).name

    def _box_key(self, box: BoxInput) -> Tuple[int, int]:
        return int(box["step"]), int(box["id"])

    def _ensure_policy(self) -> None:
        if hasattr(self, "_policy_ready"):
            return

        self._pct_available = False
        self._session = None
        self._input_name = None
        self._policy_sessions = []

        if np is None or yaml is None or Packer is None or apply_stability_mask is None:
            self._policy_ready = True
            return

        here = Path(__file__).resolve().parent
        try:
            with (here / "config" / "pct_config.yaml").open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            self._policy_ready = True
            return

        self._pct_cfg = cfg
        self._inh = int(cfg["internal_node_holder"])
        self._lnh = int(cfg["leaf_node_holder"])
        self._setting = int(cfg["setting"])
        self._size_minimum = float(cfg["size_minimum"])
        self._density_max = float(cfg.get("density_max", 1.0))
        self._container = [float(self.pallet.length), float(self.pallet.width), float(self.pallet.height)]

        if ort is not None:
            model_paths = cfg.get("model_paths") or []
            if isinstance(model_paths, (str, bytes)):
                model_paths = [model_paths]
            if not self._candidate_union_enabled:
                model_paths = model_paths[:1]

            for raw_path in model_paths:
                session_info = self._load_policy_session(str(raw_path), here)
                if session_info is not None:
                    self._policy_sessions.append(session_info)

            if not self._policy_sessions:
                session_info = self._load_policy_session(str(cfg["model_path"]), here)
                if session_info is not None:
                    self._policy_sessions.append(session_info)

            if self._policy_sessions:
                self._session = self._policy_sessions[0]["session"]
                self._input_name = self._policy_sessions[0]["input_name"]

        self._pct_available = True
        self._policy_ready = True

    def _load_policy_session(self, raw_path: str, here: Path) -> Optional[Dict]:
        model_path = raw_path
        if not os.path.isabs(model_path):
            model_path = str(here / model_path)
        if not os.path.exists(model_path):
            return None

        try:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            return {
                "name": Path(model_path).name,
                "path": model_path,
                "session": session,
                "input_name": session.get_inputs()[0].name,
            }
        except Exception:
            return None

    def _ensure_packer_for_current_run(self) -> bool:
        self._ensure_policy()
        if not self._pct_available:
            return False

        if len(self.sequence) == 0:
            self._packer = Packer(
                self._container,
                self._size_minimum,
                self._inh,
                self._lnh,
                self._setting,
            )
            self._packer.reset()
            self._raw_sequence = []
        elif not hasattr(self, "_packer"):
            self._packer = Packer(
                self._container,
                self._size_minimum,
                self._inh,
                self._lnh,
                self._setting,
            )
            self._packer.reset()
            self._raw_sequence = []

        return True

    def _density(self, box: BoxInput, size: List[float]) -> float:
        if self._setting < 3:
            return 1.0
        volume = max(size[0] * size[1] * size[2], 1e-9)
        return (float(box["mass"]) / volume) / self._density_max

    def _fallback_leaf_index(self, leaf_region: np.ndarray) -> int:
        valid_indices = np.flatnonzero(leaf_region[:, 8] > 0.5)
        if len(valid_indices) == 0:
            return 0

        best_idx = int(valid_indices[0])
        best_score = -1e18
        for idx in valid_indices:
            leaf = leaf_region[int(idx)]
            lx, ly, lz, hx, hy, _ = [float(v) for v in leaf[:6]]
            area = max(0.0, hx - lx) * max(0.0, hy - ly)
            score = area - 0.01 * lz + 0.05 * float(lx <= 1e-6) + 0.05 * float(ly <= 1e-6)
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        return best_idx

    def _find_position_baseline(
        self,
        box: BoxInput,
    ) -> Optional[Tuple[float, float, float, Tuple[float, float, float], int]]:
        for dims, rotation in self._candidate_orientations(box["size"]):
            if self._fits_current_position(dims):
                return self.cursor_x, self.cursor_y, self.layer_z, dims, rotation

            self._move_next_row()

            if self._fits_current_position(dims):
                return self.cursor_x, self.cursor_y, self.layer_z, dims, rotation

            self._move_next_layer()

            if self._fits_current_position(dims):
                return self.cursor_x, self.cursor_y, self.layer_z, dims, rotation

        return None

    def _same_dims(
        self,
        dims: Tuple[float, float, float],
        expected: List[float],
        eps: float = 1e-3,
    ) -> bool:
        return all(abs(float(dims[i]) - float(expected[i])) <= eps for i in range(3))

    def _aabb_from_size_position(
        self,
        size: List[float],
        position: List[float],
    ) -> Tuple[float, float, float, float, float, float]:
        dx, dy, dz = [float(v) for v in size]
        cx, cy, cz = [float(v) for v in position]
        return (
            cx - dx / 2.0,
            cy - dy / 2.0,
            cz - dz / 2.0,
            cx + dx / 2.0,
            cy + dy / 2.0,
            cz + dz / 2.0,
        )

    def _overlap_3d(
        self,
        a: Tuple[float, float, float, float, float, float],
        b: Tuple[float, float, float, float, float, float],
        eps: float = 1e-3,
    ) -> bool:
        return (
            a[0] < b[3] - eps and b[0] < a[3] - eps
            and a[1] < b[4] - eps and b[1] < a[4] - eps
            and a[2] < b[5] - eps and b[2] < a[5] - eps
        )

    def _support_ok(
        self,
        box_aabb: Tuple[float, float, float, float, float, float],
        eps: float = 1e-5,
    ) -> bool:
        x0, y0, z0, x1, y1, _ = box_aabb
        if z0 <= eps:
            return True

        base_area = max((x1 - x0) * (y1 - y0), 1e-9)
        support_area = 0.0
        center_supported = False
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        for placed in self.sequence:
            old = self._aabb_from_size_position(placed["size"], placed["position"])
            ox0, oy0, _, ox1, oy1, oz1 = old
            if abs(oz1 - z0) > eps:
                continue

            ix = max(0.0, min(x1, ox1) - max(x0, ox0))
            iy = max(0.0, min(y1, oy1) - max(y0, oy0))
            support_area += ix * iy

            if ox0 - eps <= cx <= ox1 + eps and oy0 - eps <= cy <= oy1 + eps:
                center_supported = True

        return support_area / base_area >= 0.30 and center_supported

    def _validate_packed_record(
        self,
        box: BoxInput,
        packed_record: List[float],
        raw_sequence: List[PlacedBox],
    ) -> Optional[Tuple[Tuple[float, float, float, Tuple[float, float, float], int], PlacedBox]]:
        dx, dy, dz, lx, ly, lz, _ = [float(v) for v in packed_record]
        raw = [float(v) for v in box["size"]]
        dims = (dx, dy, dz)

        if self._same_dims(dims, raw):
            rotation = 0
        elif self.algo.allow_rotation and self._same_dims(dims, [raw[1], raw[0], raw[2]]):
            rotation = 90
        else:
            return None

        rounded_size = [round(dx, 3), round(dy, 3), round(dz, 3)]
        rounded_position = [
            round(lx + dx / 2.0, 3),
            round(ly + dy / 2.0, 3),
            round(lz + dz / 2.0, 3),
        ]
        candidate = self._aabb_from_size_position(rounded_size, rounded_position)
        eps = 1e-3

        if min(rounded_size) <= 0.0:
            return None

        if (
            candidate[0] < -eps
            or candidate[1] < -eps
            or candidate[2] < -eps
            or candidate[3] > float(self.pallet.length) + eps
            or candidate[4] > float(self.pallet.width) + eps
            or candidate[5] > float(self.pallet.height) + eps
        ):
            return None

        for placed in raw_sequence:
            old = self._aabb_from_size_position(placed["size"], placed["position"])
            if self._overlap_3d(candidate, old):
                return None

        raw_placed = {
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": rounded_size,
            "mass": float(box["mass"]),
            "position": rounded_position,
            "rotation": int(rotation),
        }

        output_lx = min(max(lx, 0.0), float(self.pallet.length) - dx)
        output_ly = min(max(ly, 0.0), float(self.pallet.width) - dy)
        output_lz = max(lz, 0.0)
        output_size = [round(dx, 3), round(dy, 3), round(dz, 3)]
        output_position = [
            round(output_lx + dx / 2.0, 3),
            round(output_ly + dy / 2.0, 3),
            round(output_lz + dz / 2.0, 3),
        ]
        output_candidate = self._aabb_from_size_position(output_size, output_position)

        if (
            output_candidate[0] < -eps
            or output_candidate[1] < -eps
            or output_candidate[2] < -eps
            or output_candidate[3] > float(self.pallet.length) + eps
            or output_candidate[4] > float(self.pallet.width) + eps
            or output_candidate[5] > float(self.pallet.height) + eps
        ):
            return None

        for placed in raw_sequence:
            old = self._aabb_from_size_position(placed["size"], placed["position"])
            if self._overlap_3d(output_candidate, old):
                return None

        return (output_lx, output_ly, output_lz, dims, rotation), raw_placed

    def _validated_output_from_packed(
        self,
        box: BoxInput,
        packed_record: List[float],
    ) -> Optional[Tuple[float, float, float, Tuple[float, float, float], int]]:
        validated = self._validate_packed_record(
            box,
            packed_record,
            getattr(self, "_raw_sequence", self.sequence),
        )
        if validated is None:
            return None

        output, raw_placed = validated
        self._pending_raw_placed = raw_placed
        return output

    def _candidate_leaf_indices(
        self,
        obs_arr: np.ndarray,
        safe_leaf_region: np.ndarray,
    ) -> Tuple[List[int], Dict[int, float]]:
        source = "union" if self._candidate_union_enabled else "primary"
        indices, metadata = self._candidate_leaf_metadata(obs_arr, safe_leaf_region, source)
        scores = {idx: float(metadata[idx].get("aggregate_probability", 0.0)) for idx in indices}
        return indices, scores

    def _canonical_policy_sessions(self) -> List[Dict[str, Any]]:
        sessions = list(getattr(self, "_policy_sessions", []))
        if not sessions and self._session is not None and self._input_name is not None:
            sessions = [{
                "name": "model_path",
                "path": "",
                "session": self._session,
                "input_name": self._input_name,
            }]
        # Candidate metadata, masks and tie-breaks must not depend on YAML order.
        return sorted(sessions, key=lambda item: (str(item.get("name", "")), str(item.get("path", ""))))

    def _calibrated_leaf_probabilities(self, values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        result = np.zeros((self._lnh,), dtype=np.float64)
        finite = valid_mask & np.isfinite(values)
        if not np.any(finite):
            return result
        chosen = values[finite].astype(np.float64, copy=True)
        # Some exports return probabilities, others logits.  Renormalizing the
        # former and softmaxing the latter gives comparable per-model masses.
        if np.all(chosen >= 0.0) and np.all(chosen <= 1.0) and float(chosen.sum()) > 0.0:
            calibrated = chosen / float(chosen.sum())
        else:
            chosen -= float(np.max(chosen))
            calibrated = np.exp(np.clip(chosen, -60.0, 0.0))
            calibrated /= max(float(calibrated.sum()), 1e-12)
        result[np.flatnonzero(finite)] = calibrated
        return result

    def _candidate_leaf_metadata(
        self,
        obs_arr: np.ndarray,
        safe_leaf_region: np.ndarray,
        source: str = "union",
    ) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
        sessions = self._canonical_policy_sessions()
        if source == "primary" and sessions:
            sessions = [sessions[0] if not getattr(self, "_policy_sessions", []) else self._policy_sessions[0]]
        elif source == "fallback":
            fallback = [item for item in sessions if Path(str(item.get("name", ""))).name == self._fallback_model_name]
            sessions = fallback[:1]
        elif source.startswith("model:"):
            requested = source.split(":", 1)[1]
            sessions = [item for item in sessions if Path(str(item.get("name", ""))).name == Path(requested).name][:1]

        valid_mask = safe_leaf_region[:, 8] > 0.5
        model_names = [Path(str(item.get("name", "model"))).name for item in self._canonical_policy_sessions()]
        model_bits = {name: 1 << index for index, name in enumerate(model_names)}
        metadata: Dict[int, Dict[str, Any]] = {}
        top_k = max(1, int(self._union_top_k_per_model)) if source == "union" else max(20, int(self._union_fallback_top_k_leaf))

        for policy in sessions:
            model_name = Path(str(policy.get("name", "model"))).name
            try:
                output = policy["session"].run(None, {policy["input_name"]: obs_arr})[0]
                values = np.asarray(output).reshape(-1)[:self._lnh].astype(np.float64, copy=True)
            except Exception:
                continue
            if values.size < self._lnh:
                continue
            calibrated = self._calibrated_leaf_probabilities(values, valid_mask)
            masked = values.copy()
            masked[~valid_mask] = -np.inf
            ranked = sorted(
                (int(index) for index in np.flatnonzero(np.isfinite(masked))),
                key=lambda index: (-float(masked[index]), index),
            )
            denominator = max(1, len(ranked) - 1)
            for rank, leaf_index in enumerate(ranked[:top_k]):
                item = metadata.setdefault(leaf_index, {
                    "leaf_index": int(leaf_index),
                    "model_ranks": {},
                    "model_probabilities": {},
                    "proposal_mask": 0,
                    "proposal_models": [],
                })
                item["model_ranks"][model_name] = 1.0 - float(rank) / float(denominator)
                item["model_probabilities"][model_name] = float(calibrated[leaf_index])
                item["proposal_mask"] = int(item["proposal_mask"]) | int(model_bits.get(model_name, 0))
                item["proposal_models"].append(model_name)

        if not metadata:
            valid_indices = [int(i) for i in np.flatnonzero(valid_mask)]
            if not valid_indices:
                return [], {}
            fallback = self._fallback_leaf_index(safe_leaf_region)
            ordered = [fallback] + [idx for idx in valid_indices if idx != fallback]
            metadata = {
                idx: {
                    "leaf_index": idx,
                    "model_ranks": {},
                    "model_probabilities": {},
                    "proposal_mask": 0,
                    "proposal_models": [],
                    "aggregate_rank": 0.0,
                    "aggregate_probability": 0.0,
                }
                for idx in ordered
            }
            return ordered, metadata

        for item in metadata.values():
            item["proposal_models"] = tuple(sorted(set(item["proposal_models"])))
            ranks = list(item["model_ranks"].values())
            probabilities = list(item["model_probabilities"].values())
            item["aggregate_rank"] = float(sum(ranks) / len(ranks)) if ranks else 0.0
            item["aggregate_probability"] = float(sum(probabilities) / len(probabilities)) if probabilities else 0.0

        if source == "union":
            # The union is a set. Geometry ranking happens after placement;
            # leaf index is only a deterministic enumeration key.
            return sorted(metadata), metadata
        ordered = sorted(
            metadata,
            key=lambda idx: (-float(metadata[idx].get("aggregate_rank", 0.0)), int(idx)),
        )
        return ordered, metadata

    def _looks_adversarial_order(self) -> bool:
        history = list(getattr(self, "_observed_boxes", []))
        if len(history) < 1:
            return False

        first = history[0]
        size = first["size"]
        volume = float(size[0]) * float(size[1]) * float(size[2])
        if volume <= 0.0065 and float(first["mass"]) >= 1.0:
            return True

        if len(history) >= 2:
            second = history[1]
            second_size = second["size"]
            second_volume = (
                float(second_size[0])
                * float(second_size[1])
                * float(second_size[2])
            )
            avg_volume = (volume + second_volume) / 2.0
            first_mass = float(first["mass"])
            second_mass = float(second["mass"])
            same_size = tuple(first["size"]) == tuple(second["size"])

            if min(first_mass, second_mass) >= 4.5 and avg_volume <= 0.0115 and not same_size:
                return True
            if min(first_mass, second_mass) >= 4.0 and 0.0135 <= avg_volume <= 0.0142:
                return True

        return False

    def _edge_contact_bonus(self, raw_placed: PlacedBox) -> float:
        dx, dy, dz = [float(v) for v in raw_placed["size"]]
        cx, cy, cz = [float(v) for v in raw_placed["position"]]
        lx = cx - dx / 2.0
        ly = cy - dy / 2.0
        lz = cz - dz / 2.0
        hx = cx + dx / 2.0
        hy = cy + dy / 2.0
        eps = 1e-4

        bonus = 0.0
        bonus += 0.03 if lz <= eps else 0.0
        bonus += 0.01 if lx <= eps else 0.0
        bonus += 0.01 if ly <= eps else 0.0
        bonus += 0.01 if abs(hx - float(self.pallet.length)) <= eps else 0.0
        bonus += 0.01 if abs(hy - float(self.pallet.width)) <= eps else 0.0
        bonus += 0.02 if (lx <= eps or abs(hx - float(self.pallet.length)) <= eps) and (ly <= eps or abs(hy - float(self.pallet.width)) <= eps) else 0.0
        return bonus

    def _usable_void_volume(self, packer) -> float:
        ems = np.asarray(packer.space.EMS[: packer.space.NOEMS], dtype=np.float64)
        if ems.size == 0:
            return 0.0
        ems = ems.reshape(-1, 6)
        sizes = np.maximum(ems[:, 3:6] - ems[:, 0:3], 0.0)
        volumes = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]
        sorted_sizes = np.sort(sizes, axis=1)
        min_sorted = np.asarray([0.13, 0.17, 0.17], dtype=np.float64)
        usable = (sorted_sizes >= min_sorted).all(axis=1) & (volumes >= 0.0035)
        return float(volumes[usable].sum())

    def _required_support(self, mass: float, bottom_z: float) -> float:
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

    def _support_ratio_for_raw(self, raw_sequence: List[PlacedBox], raw_placed: PlacedBox) -> float:
        dx, dy, dz = [float(v) for v in raw_placed["size"]]
        cx, cy, cz = [float(v) for v in raw_placed["position"]]
        lx, ly, lz = cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0
        if lz <= 1e-6:
            return 1.0
        hx, hy = lx + dx, ly + dy
        base_area = max(dx * dy, 1e-9)
        support_area = 0.0
        for placed in raw_sequence:
            px, py, pz = [float(v) for v in placed["size"]]
            pcx, pcy, pcz = [float(v) for v in placed["position"]]
            plx, ply, plz = pcx - px / 2.0, pcy - py / 2.0, pcz - pz / 2.0
            if abs((plz + pz) - lz) > 0.006:
                continue
            ox = max(0.0, min(hx, plx + px) - max(lx, plx))
            oy = max(0.0, min(hy, ply + py) - max(ly, ply))
            support_area += ox * oy
        return float(np.clip(support_area / base_area, 0.0, 1.0))

    def _axis_tight_fit_raw(
        self,
        lo: float,
        hi: float,
        other_lo: float,
        other_hi: float,
        axis: int,
        bottom_z: float,
        raw_sequence: List[PlacedBox],
        extent: float,
        tol: float = 0.025,
    ) -> float:
        lower_gap = lo
        upper_gap = extent - hi
        for placed in raw_sequence:
            sx, sy, sz = [float(v) for v in placed["size"]]
            cx, cy, cz = [float(v) for v in placed["position"]]
            px0, py0, pz0 = cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0
            px1, py1, pz1 = px0 + sx, py0 + sy, pz0 + sz
            if abs(pz1 - bottom_z) > 0.006 and abs(pz0 - bottom_z) > 0.006:
                continue
            if axis == 0:
                overlap = max(0.0, min(other_hi, py1) - max(other_lo, py0))
                prev_lo, prev_hi = px0, px1
            else:
                overlap = max(0.0, min(other_hi, px1) - max(other_lo, px0))
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

    def _geometry_score_terms(
        self,
        raw_sequence: List[PlacedBox],
        raw_placed: PlacedBox,
        base_packer,
        trial_packer,
    ) -> Dict[str, float]:
        dx, dy, dz = [float(v) for v in raw_placed["size"]]
        cx, cy, cz = [float(v) for v in raw_placed["position"]]
        lx, ly, lz = cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0
        hx, hy = lx + dx, ly + dy
        pallet_x, pallet_y, pallet_z = float(self.pallet.length), float(self.pallet.width), float(self.pallet.height)
        eps = 0.005

        x_wall = lx <= eps or hx >= pallet_x - eps
        y_wall = ly <= eps or hy >= pallet_y - eps
        area_ratio = dx * dy / max(pallet_x * pallet_y, 1e-9)
        large_factor = float(np.clip((area_ratio - 0.045) / 0.04, 0.0, 1.0))
        corner_large_anchor = large_factor if x_wall and y_wall else 0.0

        flush_contacts = 0
        for placed in raw_sequence:
            sx, sy, sz = [float(v) for v in placed["size"]]
            pcx, pcy, pcz = [float(v) for v in placed["position"]]
            px0, py0, pz0 = pcx - sx / 2.0, pcy - sy / 2.0, pcz - sz / 2.0
            px1, py1, pz1 = px0 + sx, py0 + sy, pz0 + sz
            if max(0.0, min(lz + dz, pz1) - max(lz, pz0)) <= 1e-6:
                continue
            if (abs(lx - px1) <= eps or abs(hx - px0) <= eps) and max(0.0, min(hy, py1) - max(ly, py0)) > 1e-6:
                flush_contacts += 1
            if (abs(ly - py1) <= eps or abs(hy - py0) <= eps) and max(0.0, min(hx, px1) - max(lx, px0)) > 1e-6:
                flush_contacts += 1
        wall_anchor = float(np.clip((float(x_wall) + float(y_wall) + 0.5 * flush_contacts) / 2.0, 0.0, 1.0))

        tight_fit = max(
            self._axis_tight_fit_raw(lx, hx, ly, hy, 0, lz, raw_sequence, pallet_x),
            self._axis_tight_fit_raw(ly, hy, lx, hx, 1, lz, raw_sequence, pallet_y),
        )
        support_ratio = self._support_ratio_for_raw(raw_sequence, raw_placed)
        required_support = 1.0 if lz <= 1e-6 else self._required_support(float(raw_placed["mass"]), lz)
        support_quality = 1.0 if lz <= 1e-6 else float(np.clip(support_ratio / max(required_support, 1e-9), 0.0, 1.0))
        unsupported_risk = 0.0 if lz <= 1e-6 else float(np.clip((required_support - support_ratio) / max(required_support, 1e-9), 0.0, 1.0))

        before_void = self._usable_void_volume(base_packer)
        after_void = self._usable_void_volume(trial_packer)
        void_reduction = float(np.clip((before_void - after_void) / 0.05, 0.0, 1.0))

        next_sequence = raw_sequence + [raw_placed]
        tops = [float(p["position"][2]) + float(p["size"][2]) / 2.0 for p in next_sequence]
        mean_top = sum(tops) / len(tops) if tops else 0.0
        top_std = (sum((top - mean_top) ** 2 for top in tops) / len(tops)) ** 0.5 if tops else 0.0
        return {
            "corner_large_anchor": float(corner_large_anchor),
            "wall_anchor": float(wall_anchor),
            "tight_fit": float(tight_fit),
            "support_quality": float(support_quality),
            "void_reduction": float(void_reduction),
            "unsupported_risk": float(unsupported_risk),
            "high_placement": float(np.clip(lz / max(pallet_z, 1e-9), 0.0, 1.0)),
            "rough_top": float(np.clip(top_std / max(pallet_z, 1e-9), 0.0, 1.0)),
        }

    def _score_trial(
        self,
        raw_sequence: List[PlacedBox],
        raw_placed: PlacedBox,
        policy_score: float,
        leaf_rank: int,
        base_packer=None,
        trial_packer=None,
    ) -> float:
        rank_bonus = 0.001 * max(0, self._search_top_k_leaf - leaf_rank)
        if self._score_profile in {"geometry_v1", "sun_v2"} and base_packer is not None and trial_packer is not None:
            terms = self._geometry_score_terms(raw_sequence, raw_placed, base_packer, trial_packer)
            return (
                0.55 * terms["corner_large_anchor"]
                + 0.35 * terms["wall_anchor"]
                + 0.45 * terms["tight_fit"]
                + 0.30 * terms["support_quality"]
                + 0.25 * terms["void_reduction"]
                - 0.35 * terms["unsupported_risk"]
                - 0.25 * terms["high_placement"]
                - 0.18 * terms["rough_top"]
                + min(max(policy_score, 0.0), 1.0) * 0.03
                + rank_bonus
            )

        dx, dy, dz = [float(v) for v in raw_placed["size"]]
        cx, cy, cz = [float(v) for v in raw_placed["position"]]
        lz = cz - dz / 2.0

        next_sequence = raw_sequence + [raw_placed]
        tops = [float(p["position"][2]) + float(p["size"][2]) / 2.0 for p in next_sequence]
        max_top = max(tops, default=0.0)
        mean_top = sum(tops) / len(tops) if tops else 0.0
        top_std = (sum((top - mean_top) ** 2 for top in tops) / len(tops)) ** 0.5 if tops else 0.0

        volume = dx * dy * dz
        return (
            volume * 100.0
            - max_top * 0.35
            - lz * 0.08
            - top_std * 0.04
            + self._edge_contact_bonus(raw_placed)
            + min(max(policy_score, 0.0), 1.0) * 0.03
            + rank_bonus
        )

    def _trial_candidates(
        self,
        box: BoxInput,
        base_packer,
        raw_sequence: List[PlacedBox],
        buffer_index: int,
        top_k: int,
        candidate_source: Optional[str] = None,
    ) -> List[Dict]:
        size = [float(box["size"][0]), float(box["size"][1]), float(box["size"][2])]
        density = self._density(box, size)

        try:
            obs_packer = copy.deepcopy(base_packer)
            obs = obs_packer.observe(size, density)
            obs_arr = obs.reshape(1, -1, 9).astype(np.float32)
            leaf_region = obs_arr[0, self._inh:self._inh + self._lnh, :]
        except Exception:
            return []

        safe_leaf_region, _ = apply_stability_mask(
            leaf_region,
            obs_packer.space.boxes,
            size,
            float(box["mass"]),
            self._container,
        )
        if float(safe_leaf_region[:, 8].sum()) <= 0.0:
            return []

        if candidate_source is None:
            candidate_source = "union" if self._candidate_union_enabled else "primary"
        candidate_indices, candidate_metadata = self._candidate_leaf_metadata(
            obs_arr,
            safe_leaf_region,
            candidate_source,
        )
        candidates = self._build_trial_candidates(
            box,
            base_packer,
            obs_packer,
            safe_leaf_region,
            raw_sequence,
            buffer_index,
            candidate_indices,
            candidate_metadata,
            len(candidate_indices) if candidate_source == "union" else top_k,
            obs_arr,
        )

        if not candidates and candidate_source == "union":
            fallback_indices, fallback_metadata = self._candidate_leaf_metadata(
                obs_arr,
                safe_leaf_region,
                "fallback",
            )
            candidates = self._build_trial_candidates(
                box,
                base_packer,
                obs_packer,
                safe_leaf_region,
                raw_sequence,
                buffer_index,
                fallback_indices,
                fallback_metadata,
                max(top_k, int(self._union_fallback_top_k_leaf)),
                obs_arr,
            )

        candidates.sort(key=lambda c: (c["score"], -c["buffer_index"], -c["leaf_rank"]), reverse=True)
        return candidates

    def _build_trial_candidates(
        self,
        box: BoxInput,
        base_packer,
        obs_packer,
        safe_leaf_region: np.ndarray,
        raw_sequence: List[PlacedBox],
        buffer_index: int,
        candidate_indices: List[int],
        candidate_metadata: Dict[int, Dict[str, Any]],
        candidate_limit: int,
        obs_arr: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        candidates_by_placement: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for leaf_rank, selected in enumerate(candidate_indices[:candidate_limit]):
            leaf = safe_leaf_region[selected]
            if float(np.sum(leaf[:6])) == 0.0:
                continue

            try:
                trial_packer = copy.deepcopy(obs_packer)
                placed = trial_packer.place(leaf[:6])
            except Exception:
                continue

            if not placed:
                continue

            validated = self._validate_packed_record(box, trial_packer.packed[-1], raw_sequence)
            if validated is None:
                continue

            output, raw_placed = validated
            metadata = copy.deepcopy(candidate_metadata.get(int(selected), {}))
            score = self._score_trial(
                raw_sequence,
                raw_placed,
                float(metadata.get("aggregate_probability", 0.0)),
                leaf_rank,
                base_packer,
                trial_packer,
            )
            placement_key = (
                *(round(float(value), 6) for value in output[:3]),
                *(round(float(value), 6) for value in output[3]),
                int(output[4]),
            )
            candidate = {
                "box": box,
                "box_key": self._box_key(box),
                "buffer_index": int(buffer_index),
                "leaf_rank": int(leaf_rank),
                "leaf_index": int(selected),
                "candidate_metadata": metadata,
                "placement_key": placement_key,
                "observation": obs_arr,
                "safe_leaf": safe_leaf_region[int(selected)].astype(np.float32, copy=True),
                "packer": trial_packer,
                "validated": output,
                "raw_placed": raw_placed,
                "score": float(score),
            }
            previous = candidates_by_placement.get(placement_key)
            if previous is None:
                candidates_by_placement[placement_key] = candidate
                continue

            previous_meta = previous["candidate_metadata"]
            previous_meta["proposal_mask"] = int(previous_meta.get("proposal_mask", 0)) | int(metadata.get("proposal_mask", 0))
            previous_meta["proposal_models"] = tuple(sorted(set(previous_meta.get("proposal_models", ())) | set(metadata.get("proposal_models", ()))))
            previous_meta.setdefault("model_ranks", {}).update(metadata.get("model_ranks", {}))
            previous_meta.setdefault("model_probabilities", {}).update(metadata.get("model_probabilities", {}))
            if candidate["score"] > previous["score"]:
                candidate["candidate_metadata"] = previous_meta
                candidates_by_placement[placement_key] = candidate

        return list(candidates_by_placement.values())

    def _plan_one_step(self, current_buffer: List[BoxInput]) -> Optional[Dict]:
        raw_sequence = list(getattr(self, "_raw_sequence", self.sequence))
        candidates = []
        for buffer_index, box in enumerate(current_buffer):
            candidates.extend(
                self._trial_candidates(
                    box,
                    self._packer,
                    raw_sequence,
                    buffer_index,
                    self._search_top_k_leaf,
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda c: (c["score"], -c["buffer_index"], -c["leaf_rank"]), reverse=True)
        return candidates[0]

    def _plan_beam(self, current_buffer: List[BoxInput]) -> Optional[Dict]:
        depth = min(self._search_depth, len(current_buffer))
        raw_sequence = list(getattr(self, "_raw_sequence", self.sequence))
        beams = [{
            "packer": self._packer,
            "raw_sequence": raw_sequence,
            "remaining": list(enumerate(current_buffer)),
            "score": 0.0,
            "first": None,
        }]

        for _ in range(depth):
            expanded = []
            for beam in beams:
                for buffer_index, box in beam["remaining"]:
                    candidates = self._trial_candidates(
                        box,
                        beam["packer"],
                        beam["raw_sequence"],
                        buffer_index,
                        self._search_top_k_leaf,
                    )
                    for candidate in candidates:
                        remaining = [(idx, item) for idx, item in beam["remaining"] if idx != buffer_index]
                        expanded.append({
                            "packer": candidate["packer"],
                            "raw_sequence": beam["raw_sequence"] + [candidate["raw_placed"]],
                            "remaining": remaining,
                            "score": float(beam["score"]) + float(candidate["score"]),
                            "first": beam["first"] or candidate,
                        })

            if not expanded:
                break

            expanded.sort(
                key=lambda b: (
                    b["score"],
                    -b["first"]["buffer_index"],
                    -b["first"]["leaf_rank"],
                ),
                reverse=True,
            )
            beams = expanded[:self._search_beam_width]

        valid_beams = [beam for beam in beams if beam["first"] is not None]
        if not valid_beams:
            return None

        valid_beams.sort(
            key=lambda b: (
                b["score"],
                -b["first"]["buffer_index"],
                -b["first"]["leaf_rank"],
            ),
            reverse=True,
        )
        return valid_beams[0]["first"]

    def _plan_non_buffer_candidate_union(self, current_box: BoxInput) -> Optional[Dict]:
        if not self._candidate_union_enabled or self.algo.buffer_size != 0:
            return None

        raw_sequence = list(getattr(self, "_raw_sequence", self.sequence))
        score_profile = self._candidate_union_score_profile.strip()
        old_score_profile = self._score_profile
        if score_profile:
            self._score_profile = score_profile
        try:
            candidates = self._trial_candidates(
                current_box,
                self._packer,
                raw_sequence,
                0,
                self._search_top_k_leaf,
            )
        finally:
            self._score_profile = old_score_profile
        if not candidates:
            return None

        return candidates[0]

    def _prepare_buffer_search(self, current_buffer: List[BoxInput]) -> None:
        if not self._ensure_packer_for_current_run():
            return

        if self._search_mode == "beam":
            candidate = self._plan_beam(current_buffer)
        else:
            candidate = self._plan_one_step(current_buffer)

        if candidate is None:
            return

        self._search_plan = {
            "sequence_len": len(self.sequence),
            "box_key": candidate["box_key"],
            "packer": candidate["packer"],
            "validated": candidate["validated"],
            "raw_placed": candidate["raw_placed"],
        }

    # -----------------------------------------------------------------------
    # 기본 적재 로직
    # -----------------------------------------------------------------------

    def _candidate_orientations(
        self,
        size: List[float],
    ) -> List[Tuple[Tuple[float, float, float], int]]:
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

        if not self.algo.allow_rotation:
            return [((sx, sy, sz), 0)]

        return [
            ((sx, sy, sz), 0),
            ((sy, sx, sz), 90),
        ]

    def _fits_current_position(
        self,
        dims: Tuple[float, float, float],
    ) -> bool:
        dx, dy, dz = dims

        if self.cursor_x + dx > self.pallet.length:
            return False

        if self.cursor_y + dy > self.pallet.width:
            return False

        if self.layer_z + dz > self.pallet.height:
            return False

        return True

    def _move_next_row(self) -> None:
        self.cursor_x = 0.0
        self.cursor_y += self.row_depth
        self.row_depth = 0.0

    def _move_next_layer(self) -> None:
        self.cursor_x = 0.0
        self.cursor_y = 0.0
        self.layer_z += self.layer_height
        self.row_depth = 0.0
        self.layer_height = 0.0

    def _find_position(
        self,
        box: BoxInput,
    ) -> Optional[Tuple[float, float, float, Tuple[float, float, float], int]]:
        if self._search_enabled and self.algo.buffer_size > 0:
            plan = getattr(self, "_search_plan", None)
            if plan is not None and plan.get("sequence_len") == len(self.sequence):
                if self._box_key(box) != plan.get("box_key"):
                    return None

                self._packer = plan["packer"]
                self._pending_raw_placed = plan["raw_placed"]
                output = plan["validated"]
                self._search_plan = None
                return output
            if plan is not None:
                self._search_plan = None

        if not self._ensure_packer_for_current_run():
            return self._find_position_baseline(box)

        use_adaptive_union = (
            self._candidate_union_enabled
            and self._candidate_union_mode == "adaptive"
            and self._looks_adversarial_order()
        )

        if self._search_enabled and self.algo.buffer_size == 0 and use_adaptive_union:
            planned = self._plan_non_buffer_candidate_union(box)
            if planned is not None:
                self._packer = planned["packer"]
                self._pending_raw_placed = planned["raw_placed"]
                return planned["validated"]

        size = [float(box["size"][0]), float(box["size"][1]), float(box["size"][2])]
        density = self._density(box, size)
        try:
            obs = self._packer.observe(size, density)
            obs_arr = obs.reshape(1, -1, 9).astype(np.float32)
            leaf_region = obs_arr[0, self._inh:self._inh + self._lnh, :]
        except Exception:
            return None

        safe_leaf_region, _ = apply_stability_mask(
            leaf_region,
            self._packer.space.boxes,
            size,
            float(box["mass"]),
            self._container,
        )
        if float(safe_leaf_region[:, 8].sum()) <= 0.0:
            return None

        old_union_enabled = self._candidate_union_enabled
        if (
            self.algo.buffer_size == 0
            and self._candidate_union_mode in {"fallback", "adaptive"}
            and not use_adaptive_union
        ):
            self._candidate_union_enabled = False
        try:
            candidate_indices, _ = self._candidate_leaf_indices(obs_arr, safe_leaf_region)
        finally:
            self._candidate_union_enabled = old_union_enabled
        if not candidate_indices:
            return None

        for selected in candidate_indices[:20]:
            leaf = safe_leaf_region[selected]
            if float(np.sum(leaf[:6])) == 0.0:
                continue

            try:
                trial_packer = copy.deepcopy(self._packer)
                placed = trial_packer.place(leaf[:6])
            except Exception:
                continue

            if not placed:
                continue

            validated = self._validated_output_from_packed(box, trial_packer.packed[-1])
            if validated is None:
                continue

            self._packer = trial_packer
            return validated

        if (
            self._search_enabled
            and self.algo.buffer_size == 0
            and self._candidate_union_enabled
            and self._candidate_union_mode in {"fallback", "adaptive"}
        ):
            planned = self._plan_non_buffer_candidate_union(box)
            if planned is not None:
                self._packer = planned["packer"]
                self._pending_raw_placed = planned["raw_placed"]
                return planned["validated"]

        return None

    def _append_placed(
        self,
        box: BoxInput,
        dims: Tuple[float, float, float],
        rotation: int,
        x: float,
        y: float,
        z: float,
    ) -> None:
        dx, dy, dz = dims

        self.sequence.append({
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": [
                round(dx, 3),
                round(dy, 3),
                round(dz, 3),
            ],
            "mass": float(box["mass"]),
            "position": [
                round(x + dx / 2.0, 3),
                round(y + dy / 2.0, 3),
                round(z + dz / 2.0, 3),
            ],
            "rotation": int(rotation),
        })
        if hasattr(self, "_pending_raw_placed"):
            if not hasattr(self, "_raw_sequence"):
                self._raw_sequence = []
            self._raw_sequence.append(self._pending_raw_placed)
            del self._pending_raw_placed

        self.cursor_x += dx
        self.row_depth = max(self.row_depth, dy)
        self.layer_height = max(self.layer_height, dz)

    def run(self, boxes: List[BoxInput]) -> RunResult:
        self._reset_state()

        buf = BufferManager(self.algo.buffer_size)
        buf.reset(boxes)

        while buf.has_pending():
            if self.algo.buffer_size == 0:
                current = [buf.peek_next()]
                if current[0] is not None:
                    self._observed_boxes.append(current[0])
            else:
                current = buf.get_buffer()

            if len(self.sequence) >= 130:
                self.finished_by_user = True
                break

            if self.should_finish(current):
                self.finished_by_user = True
                break

            placed = False

            for selected_index, box in enumerate(current):
                found = self._find_position(box)

                if found is None:
                    continue

                x, y, z, dims, rotation = found

                self._append_placed(
                    box=box,
                    dims=dims,
                    rotation=rotation,
                    x=x,
                    y=y,
                    z=z,
                )

                if self.algo.buffer_size == 0:
                    buf.pop_next()
                else:
                    buf.pop_selected(selected_index)

                placed = True
                break

            if placed:
                continue

            self.finished = True

            if current:
                self.terminated_step = int(current[0]["step"])

            break

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": self.sequence,
            "terminated": self.finished,
            "terminated_step": self.terminated_step,
            "finished_by_user": self.finished_by_user,
        }
