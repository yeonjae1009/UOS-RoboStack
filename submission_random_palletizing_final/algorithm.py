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
except ImportError:  # Evaluation still remains runnable if extra packages are not installed.
    ort = None


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


_HERE = Path(__file__).resolve().parent


def _load_pct_config() -> dict:
    with (_HERE / "config" / "pct_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Palletizer:
    """Online PCT/ONNX palletizer using only the current buffer state."""

    def __init__(self, pallet_cfg: PalletConfig, algo_cfg: AlgorithmConfig) -> None:
        self.pallet = pallet_cfg
        self.algo = algo_cfg

        cfg = _load_pct_config()
        self.inh = int(cfg["internal_node_holder"])
        self.lnh = int(cfg["leaf_node_holder"])
        self.setting = int(cfg["setting"])
        self.size_minimum = float(cfg["size_minimum"])
        self.density_max = float(cfg.get("density_max", 1.0))

        model_path = str(cfg["model_path"])
        if not os.path.isabs(model_path):
            model_path = str(_HERE / model_path)

        self.session = None
        self.input_name = None
        if ort is not None:
            try:
                session_options = ort.SessionOptions()
                session_options.intra_op_num_threads = 1
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                self.input_name = self.session.get_inputs()[0].name
            except Exception:
                self.session = None
                self.input_name = None

        self.container = [float(self.pallet.length), float(self.pallet.width), float(self.pallet.height)]
        self._reset_state()

    def _reset_state(self) -> None:
        self.packer = Packer(
            self.container,
            self.size_minimum,
            self.inh,
            self.lnh,
            self.setting,
        )
        self.packer.reset()
        self.sequence: List[PlacedBox] = []
        self.finished = False
        self.terminated_step: Optional[int] = None
        self.finished_by_user = False

    def should_finish(self, current_buffer: List[BoxInput]) -> bool:
        return False

    def _density(self, box: BoxInput, size: List[float]) -> float:
        if self.setting < 3:
            return 1.0
        volume = max(float(size[0]) * float(size[1]) * float(size[2]), 1e-9)
        return (float(box["mass"]) / volume) / self.density_max

    def _select_leaf(self, box: BoxInput, size: List[float]) -> Optional[np.ndarray]:
        density = self._density(box, size)
        obs = self.packer.observe(size, density)
        obs_arr = obs.reshape(1, -1, 9).astype(np.float32)
        leaf_region = obs_arr[0, self.inh:self.inh + self.lnh, :]

        safe_leaf_region, _ = apply_stability_mask(
            leaf_region,
            self.packer.space.boxes,
            size,
            float(box["mass"]),
            self.container,
        )
        if float(safe_leaf_region[:, 8].sum()) <= 0.0:
            return None

        if self.session is not None and self.input_name is not None:
            probs = self.session.run(None, {self.input_name: obs_arr})[0][0].astype(np.float64, copy=True)
            probs[safe_leaf_region[:, 8] <= 0.5] = -np.inf
            selected = int(np.argmax(probs))
        else:
            selected = self._fallback_leaf_index(safe_leaf_region)
        leaf = safe_leaf_region[selected]
        if float(np.sum(leaf[0:6])) == 0.0:
            return None
        return leaf

    def _fallback_leaf_index(self, leaf_region: np.ndarray) -> int:
        valid_indices = np.flatnonzero(leaf_region[:, 8] > 0.5)
        if len(valid_indices) == 0:
            return 0
        best_idx = int(valid_indices[0])
        best_score = -1e18
        for idx in valid_indices:
            leaf = leaf_region[int(idx)]
            lx, ly, lz, hx, hy, _hz = [float(v) for v in leaf[:6]]
            footprint = max(0.0, hx - lx) * max(0.0, hy - ly)
            edge_bonus = 0.05 * (float(lx <= 1e-6) + float(ly <= 1e-6))
            low_bonus = -0.01 * lz
            score = footprint + edge_bonus + low_bonus
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        return best_idx

    def _find_position(
        self,
        box: BoxInput,
    ) -> Optional[Tuple[float, float, float, Tuple[float, float, float], int]]:
        size = [float(box["size"][0]), float(box["size"][1]), float(box["size"][2])]
        leaf = self._select_leaf(box, size)
        if leaf is None:
            return None

        if not self.packer.place(leaf[0:6]):
            return None

        x, y, z, lx, ly, lz, _bin = [float(v) for v in self.packer.packed[-1]]
        rotation = 0 if (abs(x - size[0]) < 1e-3 and abs(y - size[1]) < 1e-3) else 90
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
            "size": [round(dx, 3), round(dy, 3), round(dz, 3)],
            "mass": float(box["mass"]),
            "position": [
                round(x + dx / 2.0, 3),
                round(y + dy / 2.0, 3),
                round(z + dz / 2.0, 3),
            ],
            "rotation": int(rotation),
        })

    def run(self, boxes: List[BoxInput]) -> RunResult:
        self._reset_state()
        buffer = BufferManager(self.algo.buffer_size)
        buffer.reset(boxes)

        while buffer.has_pending():
            if self.algo.buffer_size == 0:
                current = [buffer.peek_next()]
            else:
                current = buffer.get_buffer()

            if self.should_finish(current):
                self.finished_by_user = True
                break

            placed = False
            for selected_index, box in enumerate(current):
                found = self._find_position(box)
                if found is None:
                    continue

                x, y, z, dims, rotation = found
                self._append_placed(box, dims, rotation, x, y, z)

                if self.algo.buffer_size == 0:
                    buffer.pop_next()
                else:
                    buffer.pop_selected(selected_index)

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
