#!/usr/bin/env python3
"""Run a resumable five-family development probe for single vs rule MPC."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from algorithm import AlgorithmConfig, PalletConfig, Palletizer  # noqa: E402


EPISODES = (
    "validation_continuous_iid_000",
    "validation_edge_geometry_000",
    "validation_small_dense_000",
    "validation_large_heavy_000",
    "validation_switch_5sku_000",
)


def _family(name: str) -> str:
    return next(family for family in ("continuous_iid", "edge_geometry", "small_dense", "large_heavy", "switch_5sku") if family in name)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(mode: str, boxes: list[dict]) -> tuple[dict, float]:
    palletizer = Palletizer(PalletConfig(1.2, 1.0, 1.25), AlgorithmConfig(True, 0))
    if mode == "single_1400":
        palletizer._search_enabled = False
        palletizer._candidate_union_enabled = False
    else:
        palletizer._search_enabled = True
        palletizer._candidate_union_enabled = True
        palletizer._candidate_union_mode = "mpc_rule"
    started = time.monotonic()
    result = palletizer.run(boxes)
    return result, time.monotonic() - started


def _row(mode: str, episode: str, result: dict, elapsed: float) -> dict:
    sequence = result["sequence"]
    return {
        "mode": mode,
        "episode": episode,
        "family": _family(episode),
        "fill_pct": 100.0 * sum(math.prod(float(value) for value in item["size"]) for item in sequence) / 1.5,
        "placed_boxes": len(sequence),
        "runtime_sec": elapsed,
        "terminated": result["terminated"],
        "terminated_step": result["terminated_step"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts" / "hybrid_mpc_dev_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "mpc_rule_probe_v1")
    parser.add_argument("--baseline-report", type=Path, help="reuse single_1400 rows from an earlier probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    modes = ("single_1400", "mpc_rule")
    if args.baseline_report:
        prior = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        rows.extend(row for row in prior["rows"] if row["mode"] == "single_1400")
        modes = ("mpc_rule",)
    for mode in modes:
        for episode in EPISODES:
            cache = args.output / f"{mode}__{episode}.json"
            if cache.is_file():
                row = json.loads(cache.read_text(encoding="utf-8"))
            else:
                result, elapsed = _run(mode, _load(args.dataset / f"{episode}.json"))
                row = _row(mode, episode, result, elapsed)
                cache.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
            print(json.dumps(row), flush=True)

    by_mode = {}
    for mode in ("single_1400", "mpc_rule"):
        selected = [row for row in rows if row["mode"] == mode]
        iid = [row["fill_pct"] for row in selected if row["family"] == "continuous_iid"]
        stress = [row["fill_pct"] for row in selected if row["family"] in {"edge_geometry", "small_dense", "large_heavy"}]
        switch = [row["fill_pct"] for row in selected if row["family"] == "switch_5sku"]
        by_mode[mode] = {
            "iid_fill_pct": mean(iid),
            "stress_fill_pct": mean(stress),
            "weighted_fill_pct": .7 * mean(iid) + .3 * mean(stress),
            "switch_5sku_fill_pct": mean(switch),
            "mean_placed_boxes": mean(row["placed_boxes"] for row in selected),
            "max_runtime_sec": max(row["runtime_sec"] for row in selected),
        }
    baseline, rule = by_mode["single_1400"], by_mode["mpc_rule"]
    report = {
        "scope": "five-family development probe; not final validation and not holdout",
        "episodes": list(EPISODES),
        "rows": rows,
        "summary": by_mode,
        "rule_minus_single_pct_point": {
            key: rule[key] - baseline[key]
            for key in ("iid_fill_pct", "stress_fill_pct", "weighted_fill_pct", "switch_5sku_fill_pct")
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
