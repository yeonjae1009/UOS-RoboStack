#!/usr/bin/env python3
"""Compare single-1400, rule MPC and value MPC on a locked data split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/robotics/Documents/assignment2/.venv/bin/python")
BASELINE_RECORDED = {"single_1400_raw_fill_pct": 65.7291, "sequence_oracle_upper_pct": 68.2814, "switch_5sku_pct": 62.2715}
EXPERIMENTS = ("single_1400", "mpc_rule", "mpc_value")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _prepare(workspace: Path, dataset: Path, output: Path, experiment: str) -> None:
    algorithm_path = workspace / "config" / "algorithm_config.yaml"
    pct_path = workspace / "config" / "pct_config.yaml"
    algorithm, pct = _load_yaml(algorithm_path), _load_yaml(pct_path)
    algorithm["input_path"] = str(dataset)
    algorithm["output_dir"] = str(output)
    algorithm["buffer"]["size"] = 0
    search = algorithm["search"]
    search["non_buffer_enabled"] = False
    search["enabled"] = experiment != "single_1400"
    search["candidate_union_enabled"] = experiment != "single_1400"
    search["candidate_union_mode"] = experiment if experiment.startswith("mpc_") else "single"
    model_paths = [
        "src/models/candidate-001400.onnx",
        "src/models/candidate-001450.onnx",
        "src/models/pct_model_bestonnx.onnx",
    ]
    pct["model_path"] = model_paths[0]
    pct["model_paths"] = model_paths if experiment != "single_1400" else model_paths[:1]
    _write_yaml(algorithm_path, algorithm)
    _write_yaml(pct_path, pct)


def _copy_workspace(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        HERE,
        destination,
        ignore=shutil.ignore_patterns("algorithm_results*", "final_model_selection_v1*", "__pycache__", "*.zip"),
    )


def _run(experiment: str, dataset: Path, root: Path) -> dict[str, Any]:
    workspace, output = root / ".workspaces" / experiment, root / experiment
    _copy_workspace(workspace)
    if output.exists():
        shutil.rmtree(output)
    _prepare(workspace, dataset, output, experiment)
    log_path = root / f"{experiment}.log"
    env = {**os.environ, "PALLET_SKIP_VISUALIZE": "1", "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run([str(PYTHON), "main.py"], cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    text = log_path.read_text(encoding="utf-8")
    runtimes = [float(value) for value in re.findall(r"file_processing_time\s*:\s*([0-9.]+)", text)]
    return {
        "returncode": completed.returncode,
        "elapsed_sec": time.monotonic() - started,
        "max_sequence_runtime_sec": max(runtimes, default=float("inf")),
        "output": str(output),
        "log": str(log_path),
    }


def _valid_sequence(sequence: list[dict[str, Any]]) -> bool:
    seen = set()
    boxes = []
    for item in sequence:
        key = (int(item["step"]), int(item["id"]))
        if key in seen or int(item.get("rotation", -1)) not in (0, 90):
            return False
        seen.add(key)
        size, position = [float(v) for v in item["size"]], [float(v) for v in item["position"]]
        if len(size) != 3 or len(position) != 3 or not all(math.isfinite(v) for v in size + position):
            return False
        low = [position[i] - size[i] / 2 for i in range(3)]
        high = [position[i] + size[i] / 2 for i in range(3)]
        if min(size) <= 0 or any(low[i] < -1e-3 or high[i] > (1.2, 1.0, 1.25)[i] + 1e-3 for i in range(3)):
            return False
        for old_low, old_high in boxes:
            if all(low[i] < old_high[i] - 1e-3 and old_low[i] < high[i] - 1e-3 for i in range(3)):
                return False
        boxes.append((low, high))
    return bool(sequence)


def _summarize(output: Path, runtime: float) -> dict[str, Any]:
    families: dict[str, list[tuple[float, int]]] = {}
    invalid = 0
    for path in sorted(output.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        family = next(name for name in ("continuous_iid", "edge_geometry", "small_dense", "large_heavy", "switch_5sku") if name in path.stem)
        sequence = result.get("sequence", [])
        invalid += int(not _valid_sequence(sequence))
        fill = 100.0 * sum(math.prod(float(v) for v in item["size"]) for item in sequence) / 1.5
        families.setdefault(family, []).append((fill, len(sequence)))
    iid = [value for value, _ in families.get("continuous_iid", [])]
    stress = [value for family in ("edge_geometry", "small_dense", "large_heavy") for value, _ in families.get(family, [])]
    switch = [value for value, _ in families.get("switch_5sku", [])]
    return {
        "iid_mean_raw_fill_pct": mean(iid),
        "iid_stddev_raw_fill_pct": pstdev(iid),
        "stress_mean_raw_fill_pct": mean(stress),
        "weighted_raw_fill_pct": .7 * mean(iid) + .3 * mean(stress),
        "switch_5sku_mean_raw_fill_pct": mean(switch),
        "mean_placed_boxes": mean(count for family in families.values() for _, count in family),
        "invalid_output_count": invalid,
        "max_sequence_runtime_sec": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-once", action="store_true", help="create a lock and refuse any later holdout rerun")
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    is_holdout = manifest.get("version") == "final_model_selection_v1"
    if is_holdout != args.holdout_once:
        raise ValueError("final holdout requires --holdout-once; development data must omit it")
    lock = args.output / "HOLDOUT_EVALUATED.lock"
    if is_holdout and lock.exists():
        raise RuntimeError(f"holdout was already evaluated: {lock}")
    args.output.mkdir(parents=True, exist_ok=True)
    if is_holdout:
        frozen = {
            "manifest_sha256": _sha256(manifest_path),
            "algorithm_sha256": _sha256(HERE / "algorithm.py"),
            "algorithm_config_sha256": _sha256(HERE / "config" / "algorithm_config.yaml"),
            "pct_config_sha256": _sha256(HERE / "config" / "pct_config.yaml"),
        }
        lock.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    dataset_for_run = args.dataset.resolve()
    if not is_holdout:
        input_view = args.output / "validation_inputs"
        if input_view.exists():
            shutil.rmtree(input_view)
        input_view.mkdir(parents=True)
        validation_entries = [entry for entry in manifest["sequences"] if entry.get("split") == "validation"]
        if len(validation_entries) != 78:
            raise ValueError(f"expected 78 development validation episodes, got {len(validation_entries)}")
        for entry in validation_entries:
            (input_view / entry["file"]).symlink_to((args.dataset / entry["file"]).resolve())
        dataset_for_run = input_view.resolve()

    rows = []
    for experiment in EXPERIMENTS:
        execution = _run(experiment, dataset_for_run, args.output.resolve())
        row = {"experiment": experiment, **execution}
        if execution["returncode"] == 0:
            row.update(_summarize(Path(execution["output"]), execution["max_sequence_runtime_sec"]))
        rows.append(row)
        print(json.dumps(row), flush=True)
    baseline = next(row for row in rows if row["experiment"] == "single_1400")
    for row in rows:
        if row["experiment"] == "single_1400" or row.get("returncode") != 0:
            row["passes"] = row.get("returncode") == 0
            continue
        row["passes"] = bool(
            row["weighted_raw_fill_pct"] >= baseline["weighted_raw_fill_pct"] + .75
            and row["iid_mean_raw_fill_pct"] >= baseline["iid_mean_raw_fill_pct"] - .5
            and row["stress_mean_raw_fill_pct"] >= baseline["stress_mean_raw_fill_pct"] - .5
            and row["switch_5sku_mean_raw_fill_pct"] >= BASELINE_RECORDED["switch_5sku_pct"]
            and row["invalid_output_count"] == 0
            and row["max_sequence_runtime_sec"] <= 80.0
        )
    eligible = [row for row in rows if row.get("passes")]
    ranking = sorted(eligible, key=lambda row: (-row["weighted_raw_fill_pct"], row["iid_stddev_raw_fill_pct"], -row["mean_placed_boxes"]))
    report = {"recorded_reference": BASELINE_RECORDED, "holdout": is_holdout, "experiments": rows, "ranking": ranking}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if len(rows) == len(EXPERIMENTS) and all(row.get("returncode") == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
