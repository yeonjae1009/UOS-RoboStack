from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import yaml

from buffer_manager import BufferManager
from src.pct.packer import Packer
from src.pct.stability import apply_stability_mask

try:
    import onnxruntime as ort
except ImportError:
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

        here = Path(__file__).resolve().parent
        with (here / "config" / "pct_config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._pct_cfg = cfg
        self._inh = int(cfg["internal_node_holder"])
        self._lnh = int(cfg["leaf_node_holder"])
        self._setting = int(cfg["setting"])
        self._size_minimum = float(cfg["size_minimum"])
        self._density_max = float(cfg.get("density_max", 1.0))
        self._container = [float(self.pallet.length), float(self.pallet.width), float(self.pallet.height)]
        self._session = None
        self._input_name = None

        model_path = str(cfg["model_path"])
        if not os.path.isabs(model_path):
            model_path = str(here / model_path)

        if ort is not None:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name

        self._policy_ready = True

    def _ensure_packer_for_current_run(self) -> None:
        self._ensure_policy()
        if len(self.sequence) == 0:
            self._packer = Packer(
                self._container,
                self._size_minimum,
                self._inh,
                self._lnh,
                self._setting,
            )
            self._packer.reset()
        elif not hasattr(self, "_packer"):
            self._packer = Packer(
                self._container,
                self._size_minimum,
                self._inh,
                self._lnh,
                self._setting,
            )
            self._packer.reset()

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
        self._ensure_packer_for_current_run()

        size = [float(box["size"][0]), float(box["size"][1]), float(box["size"][2])]
        density = self._density(box, size)
        obs = self._packer.observe(size, density)
        obs_arr = obs.reshape(1, -1, 9).astype(np.float32)
        leaf_region = obs_arr[0, self._inh:self._inh + self._lnh, :]

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
            probs = self._session.run(None, {self._input_name: obs_arr})[0][0].astype(np.float64, copy=True)
            probs[safe_leaf_region[:, 8] <= 0.5] = -np.inf
            selected = int(np.argmax(probs))
        else:
            selected = self._fallback_leaf_index(safe_leaf_region)

        leaf = safe_leaf_region[selected]
        if float(np.sum(leaf[:6])) == 0.0:
            return None

        if not self._packer.place(leaf[:6]):
            return None

        x, y, z, lx, ly, lz, _ = [float(v) for v in self._packer.packed[-1]]
        raw_size = [float(v) for v in box["size"]]
        rotation = 0 if (abs(x - raw_size[0]) < 1e-3 and abs(y - raw_size[1]) < 1e-3) else 90
        return lx, ly, lz, (x, y, z), rotation

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
