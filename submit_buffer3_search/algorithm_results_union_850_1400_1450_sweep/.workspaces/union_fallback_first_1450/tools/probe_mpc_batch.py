#!/usr/bin/env python3
"""Parallel development-only batch probe for single-1400 and rule MPC."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pstdev


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from algorithm import AlgorithmConfig, PalletConfig, Palletizer  # noqa: E402


COUNTS = {
    "continuous_iid": 5,
    "edge_geometry": 2,
    "small_dense": 2,
    "large_heavy": 2,
    "switch_5sku": 2,
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_one(mode: str, family: str, episode: str, path: Path) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    palletizer = Palletizer(PalletConfig(1.2, 1.0, 1.25), AlgorithmConfig(True, 0))
    if mode == "single_1400":
        palletizer._search_enabled = False
        palletizer._candidate_union_enabled = False
    else:
        palletizer._search_enabled = True
        palletizer._candidate_union_enabled = True
        palletizer._candidate_union_mode = "mpc_rule"
    started = time.monotonic()
    result = palletizer.run(_load(path))
    elapsed = time.monotonic() - started
    sequence = result["sequence"]
    return {
        "mode": mode,
        "family": family,
        "episode": episode,
        "fill_pct": 100.0 * sum(math.prod(float(v) for v in item["size"]) for item in sequence) / 1.5,
        "placed_boxes": len(sequence),
        "runtime_sec": elapsed,
        "mpc_steps": int(palletizer._mpc_decisions_used),
        "terminated": result["terminated"],
        "terminated_step": result["terminated_step"],
    }


def _summary(rows: list[dict], mode: str) -> dict:
    selected = [row for row in rows if row["mode"] == mode]
    iid = [row["fill_pct"] for row in selected if row["family"] == "continuous_iid"]
    stress = [row["fill_pct"] for row in selected if row["family"] in {"edge_geometry", "small_dense", "large_heavy"}]
    switch = [row["fill_pct"] for row in selected if row["family"] == "switch_5sku"]
    return {
        "iid_mean_fill_pct": mean(iid),
        "iid_stddev_fill_pct": pstdev(iid),
        "stress_mean_fill_pct": mean(stress),
        "weighted_fill_pct": .7 * mean(iid) + .3 * mean(stress),
        "switch_5sku_mean_fill_pct": mean(switch),
        "mean_placed_boxes": mean(row["placed_boxes"] for row in selected),
        "max_runtime_sec": max(row["runtime_sec"] for row in selected),
        "min_runtime_sec": min(row["runtime_sec"] for row in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts" / "hybrid_mpc_dev_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "mpc_rule_batch_probe_v1")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for family, count in COUNTS.items():
        for index in range(count):
            episode = f"validation_{family}_{index:03d}"
            for mode in ("single_1400", "mpc_rule"):
                tasks.append((mode, family, episode, args.dataset / f"{episode}.json"))
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(_run_one, *task): task for task in tasks}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{completed_index}/{len(tasks)}] {json.dumps(row)}", flush=True)
    rows.sort(key=lambda row: (row["mode"], row["episode"]))
    summaries = {mode: _summary(rows, mode) for mode in ("single_1400", "mpc_rule")}
    single, rule = summaries["single_1400"], summaries["mpc_rule"]
    report = {
        "scope": "development validation subset; final holdout not used",
        "counts": COUNTS,
        "rows": rows,
        "summary": summaries,
        "rule_minus_single_pct_point": {
            "iid": rule["iid_mean_fill_pct"] - single["iid_mean_fill_pct"],
            "stress": rule["stress_mean_fill_pct"] - single["stress_mean_fill_pct"],
            "weighted": rule["weighted_fill_pct"] - single["weighted_fill_pct"],
            "switch_5sku": rule["switch_5sku_mean_fill_pct"] - single["switch_5sku_mean_fill_pct"],
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
