"""Deployment smoke test (no Isaac): run the real Palletizer.run() pipeline on
spec-compliant continuous-random boxes and report stacking utilization.

Validates: numpy 2.x safety (#2), the setting-driven density path (#1, packer +
algorithm), output integrity, and gives a baseline utilization number to tune
against. Compares setting 1 (mass-blind, current) vs setting 3 (mass-aware).

  python3 submission_UOS-robostack/smoke_test.py --n 250 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

import algorithm as algo_mod  # noqa: E402
from algorithm import Palletizer, PalletConfig, AlgorithmConfig  # noqa: E402

PALLET = (1.2, 1.0, 1.25)  # = config length/width/height
PALLET_VOL = PALLET[0] * PALLET[1] * PALLET[2]
SETTING1_MODEL = HERE / "models" / "pct_model.onnx"
SETTING3_MODEL = PROJECT / "Online-3D-BPP-PCT" / "pct_model_cjspec_v2.onnx"
DENSITY_MAX = 6.0 / (0.17 * 0.17 * 0.13)  # = givenData DENSITY_MAX ~ 1597


def gen_boxes(n: int, seed: int) -> list[dict]:
    """Spec-compliant continuous-random boxes in the competition input format."""
    rng = np.random.default_rng(seed)
    boxes = []
    for i in range(n):
        w = round(float(rng.uniform(0.17, 0.32)), 3)
        l = round(float(rng.uniform(0.17, 0.32)), 3)
        h = round(float(rng.uniform(0.13, 0.26)), 3)
        mass = round(float(rng.uniform(0.5, 6.0)), 3)
        boxes.append({"step": i, "id": i, "size": [w, l, h], "mass": mass})
    return boxes


def run_config(label: str, setting: int, model_path: Path, boxes: list[dict]) -> dict:
    cfg = {
        "internal_node_holder": 200,
        "leaf_node_holder": 100,
        "setting": setting,
        "size_minimum": 0.13 if setting >= 3 else 0.134,
        "density_max": DENSITY_MAX,
        "model_path": str(model_path),
    }
    # Drive the REAL Palletizer with this config (monkeypatch the file loader).
    algo_mod._load_pct_config = lambda: cfg
    pallet = PalletConfig(length=PALLET[0], width=PALLET[1], height=PALLET[2])
    algo_cfg = AlgorithmConfig(buffer_size=0, allow_rotation=True)
    palletizer = Palletizer(pallet, algo_cfg)
    result = palletizer.run(boxes)

    seq = result["sequence"]
    placed_vol = sum(s["size"][0] * s["size"][1] * s["size"][2] for s in seq)
    util = placed_vol / PALLET_VOL
    # Geometric bounds check. Output is rounded to 3 decimals, so allow a rounding
    # margin (~1.5e-3) — tighter would flag rounding noise as OOB.
    tol = 1.5e-3
    oob_xy = oob_z = 0
    max_top = 0.0
    for s in seq:
        cx, cy, cz = s["position"]
        hx, hy, hz = s["size"][0] / 2, s["size"][1] / 2, s["size"][2] / 2
        if (cx - hx < -tol or cy - hy < -tol or cx + hx > PALLET[0] + tol or cy + hy > PALLET[1] + tol):
            oob_xy += 1
        if cz + hz > PALLET[2] + tol:
            oob_z += 1
        max_top = max(max_top, cz + hz)
    return {
        "label": label, "setting": setting, "placed": len(seq),
        "util": util, "terminated": result["terminated"],
        "terminated_step": result["terminated_step"],
        "oob_xy": oob_xy, "oob_z": oob_z, "max_top": max_top,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    boxes = gen_boxes(args.n, args.seed)
    uniq_sizes = len({tuple(b["size"]) for b in boxes})
    print(f"boxes={len(boxes)} unique_sizes={uniq_sizes} (연속 랜덤 확인) seed={args.seed}")
    print(f"pallet={PALLET} vol={PALLET_VOL:.4f} m^3  density_max={DENSITY_MAX:.2f}\n")

    runs = []
    runs.append(run_config("setting1 (mass-blind, 현행)", 1, SETTING1_MODEL, boxes))
    if SETTING3_MODEL.exists():
        runs.append(run_config("setting3 (mass-aware, density #1)", 3, SETTING3_MODEL, boxes))
    else:
        print(f"[skip] setting3 model 없음: {SETTING3_MODEL}")

    print(f"{'config':36s} {'placed':>6s} {'util%':>7s} {'oobXY':>5s} {'oobZ':>4s} {'maxTop':>7s} {'term@':>6s}")
    for r in runs:
        ts = r["terminated_step"] if r["terminated_step"] is not None else "-"
        print(f"{r['label']:36s} {r['placed']:6d} {r['util']*100:6.2f}% {r['oob_xy']:5d} {r['oob_z']:4d} "
              f"{r['max_top']:6.3f}m {str(ts):>6s}")
    print(f"\n팔레트 높이 한계 = {PALLET[2]}m. oobZ>0 또는 maxTop>{PALLET[2]} 이면 높이초과 Fail 위험.")
    print("참고: 물리 붕괴/드랍 채점은 Isaac 필요(여기선 기하 OOB만).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
