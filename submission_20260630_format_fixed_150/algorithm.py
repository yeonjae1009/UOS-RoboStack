from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict

from buffer_manager import BufferManager


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
        import os
        from pathlib import Path

        import numpy as np
        import onnxruntime as ort
        import yaml

        runtime = getattr(self, "_pct_runtime", None)

        if runtime is None:
            here = Path(__file__).resolve().parent

            with open(here / "config" / "pct_config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            model_path = str(cfg["model_path"])

            if not os.path.isabs(model_path):
                model_path = str(here / model_path)

            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

            runtime = {
                "internal_node_holder": int(cfg["internal_node_holder"]),
                "leaf_node_holder": int(cfg["leaf_node_holder"]),
                "setting": int(cfg["setting"]),
                "size_minimum": float(cfg["size_minimum"]),
                "density_max": float(cfg.get("density_max", 1.0)),
                "session": session,
                "input_name": session.get_inputs()[0].name,
            }
            self._pct_runtime = runtime

        if getattr(self, "_pct_sequence_ref", None) is not self.sequence:
            from src.pct.packer import Packer

            container = [
                float(self.pallet.length),
                float(self.pallet.width),
                float(self.pallet.height),
            ]
            packer = Packer(
                container,
                runtime["size_minimum"],
                runtime["internal_node_holder"],
                runtime["leaf_node_holder"],
                runtime["setting"],
            )
            packer.reset()
            self._pct_packer = packer
            self._pct_sequence_ref = self.sequence

        packer = self._pct_packer

        size = [
            float(box["size"][0]),
            float(box["size"][1]),
            float(box["size"][2]),
        ]

        if runtime["setting"] >= 3:
            volume = max(size[0] * size[1] * size[2], 1e-9)
            density = (float(box["mass"]) / volume) / runtime["density_max"]
        else:
            density = 1.0

        obs = packer.observe(size, density=density)
        obs_arr = obs.reshape(1, -1, 9).astype(np.float32)

        leaf_start = runtime["internal_node_holder"]
        leaf_end = leaf_start + runtime["leaf_node_holder"]
        leaf_region = obs_arr[0, leaf_start:leaf_end, :]

        if float(leaf_region[:, 8].sum()) <= 0.0:
            return None

        probs = runtime["session"].run(None, {runtime["input_name"]: obs_arr})[0]
        selected = int(np.argmax(probs[0]))
        leaf = leaf_region[selected]

        if float(np.sum(leaf[0:6])) == 0.0:
            return None

        if not packer.place(leaf[0:6]):
            return None

        dx, dy, dz, x, y, z, _bin_id = [float(v) for v in packer.packed[-1]]
        rotation = 0 if abs(dx - size[0]) < 1e-3 and abs(dy - size[1]) < 1e-3 else 90

        return x, y, z, (dx, dy, dz), rotation

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

            if len(self.sequence) >= 150:
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
