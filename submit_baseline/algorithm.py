from __future__ import annotations

import os
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

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
        return False

    def _ensure_policy(self) -> None:
        if hasattr(self, "_policy_ready"):
            return

        self._pct_available = False
        self._session = None
        self._input_name = None

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

        model_path = str(cfg["model_path"])
        if not os.path.isabs(model_path):
            model_path = str(here / model_path)

        if ort is not None:
            try:
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = 1
                self._session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
            except Exception:
                self._session = None
                self._input_name = None

        self._pct_available = True
        self._policy_ready = True

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

    def _validated_output_from_packed(
        self,
        box: BoxInput,
        packed_record: List[float],
    ) -> Optional[Tuple[float, float, float, Tuple[float, float, float], int]]:
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

        for placed in getattr(self, "_raw_sequence", self.sequence):
            old = self._aabb_from_size_position(placed["size"], placed["position"])
            if self._overlap_3d(candidate, old):
                return None

        self._pending_raw_placed = {
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": rounded_size,
            "mass": float(box["mass"]),
            "position": rounded_position,
            "rotation": int(rotation),
        }

        margin = 0.002
        output_lx = min(max(lx, margin), float(self.pallet.length) - dx - margin)
        output_ly = min(max(ly, margin), float(self.pallet.width) - dy - margin)
        output_lz = max(lz, 0.0)

        return output_lx, output_ly, output_lz, dims, rotation

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
        if not self._ensure_packer_for_current_run():
            return self._find_position_baseline(box)

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

        if self._session is not None and self._input_name is not None:
            try:
                probs = self._session.run(None, {self._input_name: obs_arr})[0][0].astype(np.float64, copy=True)
                probs[safe_leaf_region[:, 8] <= 0.5] = -np.inf
                candidate_indices = [int(i) for i in np.argsort(probs)[::-1] if np.isfinite(probs[int(i)])]
            except Exception:
                candidate_indices = [self._fallback_leaf_index(safe_leaf_region)]
        else:
            valid_indices = np.flatnonzero(safe_leaf_region[:, 8] > 0.5)
            candidate_indices = [int(i) for i in valid_indices]

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
