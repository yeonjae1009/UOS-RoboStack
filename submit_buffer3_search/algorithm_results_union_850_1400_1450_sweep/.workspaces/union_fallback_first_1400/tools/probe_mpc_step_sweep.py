#!/usr/bin/env python3
"""Locate the earliest harmful MPC decision count on representative episodes."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from algorithm import AlgorithmConfig, PalletConfig, Palletizer  # noqa: E402


EPISODES = (
    "validation_continuous_iid_000",
    "validation_continuous_iid_001",
    "validation_edge_geometry_001",
    "validation_large_heavy_001",
)
STEPS = (1, 2, 4, 6, 8)


def _run(episode: str, steps: int, path: Path) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    boxes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    palletizer = Palletizer(PalletConfig(1.2, 1.0, 1.25), AlgorithmConfig(True, 0))
    palletizer._search_enabled = True
    palletizer._candidate_union_enabled = True
    palletizer._candidate_union_mode = "mpc_rule"
    palletizer._mpc_max_search_steps = steps
    started = time.monotonic()
    result = palletizer.run(boxes)
    return {
        "episode": episode,
        "max_search_steps": steps,
        "actual_mpc_steps": palletizer._mpc_decisions_used,
        "fill_pct": 100.0 * sum(math.prod(float(v) for v in item["size"]) for item in result["sequence"]) / 1.5,
        "placed_boxes": len(result["sequence"]),
        "runtime_sec": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts" / "hybrid_mpc_dev_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "mpc_step_sweep_v1")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(episode, steps, args.dataset / f"{episode}.json") for episode in EPISODES for steps in STEPS]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(_run, *task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{index}/{len(tasks)}] {json.dumps(row)}", flush=True)
    rows.sort(key=lambda row: (row["episode"], row["max_search_steps"]))
    (args.output / "report.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
