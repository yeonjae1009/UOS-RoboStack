from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TypedDict

import numpy as np
import onnxruntime as ort
import yaml

from buffer_manager import BufferManager  # noqa: F401  (프레임워크 호환용)
from src.pct.packer import Packer


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
    terminated: bool
    terminated_step: Optional[int]
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


_HERE = Path(__file__).resolve().parent


def _load_pct_config() -> dict:
    """config/pct_config.yaml 로드 (모든 설정값은 YAML에서 읽는다)."""
    with open(_HERE / "config" / "pct_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Palletizer:
    """
    PCT(Packing Configuration Tree) DRL 정책 기반 팔레타이저.

    학습된 graph-attention 정책을 ONNX(onnxruntime)로 추론한다. 박스가 도착할 때마다
    (온라인, 버퍼 미사용) 현재 팔레트 상태로 EMS 후보(잎 노드)를 생성하고, 정책이
    최적 후보를 골라 배치한다. 더 놓을 자리가 없으면 종료한다.

    - 패킹 기하(EMS/안정성)는 src/pct (순수 numpy)로 수행
    - 정책 추론만 ONNX 로 수행 (requirements: numpy + onnxruntime)
    """

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg

        cfg = _load_pct_config()
        self.INH = int(cfg["internal_node_holder"])
        self.LNH = int(cfg["leaf_node_holder"])
        self.setting = int(cfg["setting"])
        self.size_minimum = float(cfg["size_minimum"])
        # setting 3: density(=mass/부피/density_max) 를 관찰 입력으로 사용. 학습과 동일 상수여야 함.
        self.density_max = float(cfg.get("density_max", 1.0))

        model_path = cfg["model_path"]
        if not os.path.isabs(model_path):
            model_path = str(_HERE / model_path)

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    # -----------------------------------------------------------------------
    # 참가자 수정 가능 함수
    # -----------------------------------------------------------------------

    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        """더 놓을 자리가 없을 때의 자연 종료에 맡기므로 항상 False."""
        return False

    # -----------------------------------------------------------------------
    # 메인 루프 (run 시그니처 수정 금지)
    # -----------------------------------------------------------------------

    def run(self, boxes: List[BoxInput]) -> RunResult:
        container = [float(self.pallet.length), float(self.pallet.width), float(self.pallet.height)]
        packer = Packer(container, self.size_minimum, self.INH, self.LNH, self.setting)
        packer.reset()

        sequence: List[PlacedBox] = []
        terminated = False
        terminated_step: Optional[int] = None

        for box in boxes:
            size = [float(box["size"][0]), float(box["size"][1]), float(box["size"][2])]

            # setting 3 density: 학습과 동일하게 mass/부피/density_max 로 정규화 (setting<3 이면 1)
            if self.setting >= 3:
                vol = max(size[0] * size[1] * size[2], 1e-9)
                density = (float(box["mass"]) / vol) / self.density_max
            else:
                density = 1.0

            # 1) 현재 상태 관찰값 생성
            obs = packer.observe(size, density)
            obs_arr = obs.reshape(1, -1, 9).astype(np.float32)

            # 2) 놓을 수 있는 후보(잎 노드)가 하나도 없으면 종료
            leaf_region = obs_arr[0, self.INH:self.INH + self.LNH, :]
            if float(leaf_region[:, 8].sum()) <= 0.0:
                terminated = True
                terminated_step = int(box["step"])
                break

            # 3) 정책(ONNX)으로 잎 노드 선택 (마스킹된 확률의 argmax)
            probs = self.session.run(None, {self.input_name: obs_arr})[0]
            sel = int(np.argmax(probs[0]))
            leaf = leaf_region[sel]
            if float(np.sum(leaf[0:6])) == 0.0:
                terminated = True
                terminated_step = int(box["step"])
                break

            # 4) 실제 배치
            if not packer.place(leaf[0:6]):
                terminated = True
                terminated_step = int(box["step"])
                break

            # 5) 대회 출력 형식으로 기록 (중심좌표 + 회전반영 크기)
            x, y, z, lx, ly, lz, _ = [float(v) for v in packer.packed[-1]]
            L, W, _H = size
            rotation = 0 if (abs(x - L) < 1e-3 and abs(y - W) < 1e-3) else 90
            sequence.append({
                "step": int(box["step"]),
                "id": int(box["id"]),
                "size": [round(x, 3), round(y, 3), round(z, 3)],
                "mass": float(box["mass"]),
                "position": [round(lx + x / 2.0, 3), round(ly + y / 2.0, 3), round(lz + z / 2.0, 3)],
                "rotation": int(rotation),
            })

        return {
            "buffer_size": self.algo.buffer_size,
            "sequence": sequence,
            "terminated": terminated,
            "terminated_step": terminated_step,
            "finished_by_user": False,
        }
