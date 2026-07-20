#!/usr/bin/env python3
"""Run the requested 3 single + 6 union ONNX experiments reproducibly.

Every condition runs exactly ``/home/robotics/Documents/assignment2/.venv/bin/
python main.py`` in its own disposable copy of the submission directory.  This
keeps the user's checked-in configs untouched even if a job is interrupted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/robotics/Documents/assignment2/.venv/bin/python")
MODELS = (("1400", "src/models/candidate-001400.onnx"), ("1450", "src/models/candidate-001450.onnx"), ("best", "src/models/pct_model_bestonnx.onnx"))
ORDERS = (("1400", (MODELS[0][1], MODELS[1][1], MODELS[2][1])), ("1450", (MODELS[1][1], MODELS[0][1], MODELS[2][1])), ("best", (MODELS[2][1], MODELS[0][1], MODELS[1][1])))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _family(name: str) -> str:
    for family in ("continuous_iid", "edge_geometry", "small_dense", "large_heavy", "switch_5sku"):
        if name.startswith(family + "_"):
            return family
    raise ValueError(f"unknown input filename: {name}")


def _prepare_input_view(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(source.glob("*.json")) if path.name != "manifest.json"]
    if len(files) != 78:
        raise ValueError(f"expected 78 sequence files in {source}, found {len(files)}")
    for path in files:
        target = destination / path.name
        if target.exists() or target.is_symlink():
            if target.resolve() == path.resolve():
                continue
            target.unlink()
        target.symlink_to(path.resolve())


def _configure(workspace: Path, input_dir: Path, output_dir: Path, paths: list[str], union: bool, mode: str) -> None:
    alg_path, pct_path = workspace / "config" / "algorithm_config.yaml", workspace / "config" / "pct_config.yaml"
    alg, pct = _load_yaml(alg_path), _load_yaml(pct_path)
    alg["input_path"] = str(input_dir)
    alg["output_dir"] = str(output_dir)
    alg.setdefault("buffer", {})["size"] = 0
    search = alg.setdefault("search", {})
    # Required for fallback/adaptive to reach the implementation's union path.
    search["enabled"] = True
    search["candidate_union_enabled"] = union
    search["candidate_union_mode"] = mode
    search["non_buffer_enabled"] = False
    pct["model_path"] = paths[0]
    pct["model_paths"] = paths
    _write_yaml(alg_path, alg)
    _write_yaml(pct_path, pct)


def _copy_workspace(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name.startswith("algorithm_results") or name in {"final_model_selection_v1", "__pycache__"} or name.endswith(".zip")}

    shutil.copytree(HERE, destination, ignore=ignore)


def _run_one(label: str, paths: list[str], union: bool, mode: str, results_root: Path, input_view: Path) -> dict[str, Any]:
    workspace = results_root / ".workspaces" / label
    output_dir = results_root / label
    _copy_workspace(workspace)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    _configure(workspace, input_view, output_dir, paths, union, mode)
    log_path = results_root / f"{label}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PALLET_SKIP_VISUALIZE"] = "1"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run([str(PYTHON), "main.py"], cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {"returncode": completed.returncode, "elapsed_sec": time.monotonic() - started, "log": str(log_path), "output_dir": str(output_dir)}


def _summarize(label: str, output_dir: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        sequence = result.get("sequence", [])
        rows.append({"experiment": label, "sequence": path.stem, "family": _family(path.stem), "raw_fill_pct": 100.0 * sum(math.prod(float(v) for v in box["size"]) for box in sequence) / 1.5, "placed_boxes": len(sequence), "terminated": bool(result.get("terminated")), "terminated_step": result.get("terminated_step"), "finished_by_user": bool(result.get("finished_by_user"))})
    if len(rows) != 78:
        raise ValueError(f"{label}: expected 78 output files, got {len(rows)}")
    iid = [row["raw_fill_pct"] for row in rows if row["family"] == "continuous_iid"]
    stress = [row["raw_fill_pct"] for row in rows if row["family"] in {"edge_geometry", "small_dense", "large_heavy"}]
    switch = [row["raw_fill_pct"] for row in rows if row["family"] == "switch_5sku"]
    return ({**spec, "experiment": label, "iid_mean_raw_fill_pct": mean(iid), "iid_stddev_raw_fill_pct": pstdev(iid), "stress_mean_raw_fill_pct": mean(stress), "selection_weighted_raw_fill_pct": 0.7 * mean(iid) + 0.3 * mean(stress), "switch_5sku_mean_raw_fill_pct": mean(switch), "selection_mean_placed_boxes": mean(row["placed_boxes"] for row in rows if row["family"] != "switch_5sku"), "selection_terminated_count": sum(1 for row in rows if row["family"] != "switch_5sku" and row["terminated"]), "empty_output_count": sum(1 for row in rows if row["placed_boxes"] == 0)}, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"required interpreter not found: {PYTHON}")
    user_config = _load_yaml(HERE / "config" / "algorithm_config.yaml")
    input_source = Path(user_config["input_path"])
    if not input_source.is_absolute():
        input_source = HERE / input_source
    results_root = HERE / "algorithm_results_final_model_selection_sweep"
    input_view = results_root / "input_sequences"
    _prepare_input_view(input_source, input_view)
    experiments: list[tuple[str, list[str], bool, str, dict[str, Any]]] = []
    for label, path in MODELS:
        experiments.append((f"single_{label}", [path], False, "single", {"kind": "single", "first_model": label, "model_paths": [path]}))
    for mode in ("fallback", "adaptive"):
        for first, paths in ORDERS:
            experiments.append((f"union_{mode}_first_{first}", list(paths), True, mode, {"kind": "union", "union_mode": mode, "first_model": first, "model_paths": list(paths)}))

    completed: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(_run_one, label, paths, union, mode, results_root, input_view): (label, spec) for label, paths, union, mode, spec in experiments}
        for future in as_completed(futures):
            label, spec = futures[future]
            try:
                completed[label] = {**spec, **future.result()}
                print(f"[sweep] done {label} rc={completed[label]['returncode']}", flush=True)
            except Exception as exc:
                completed[label] = {**spec, "returncode": -1, "error": f"{type(exc).__name__}: {exc}"}

    summaries, per_sequence = [], []
    for label, _paths, _union, _mode, spec in experiments:
        record = completed[label]
        if record.get("returncode") == 0:
            metrics, rows = _summarize(label, Path(record["output_dir"]), spec)
            record.update(metrics)
            per_sequence.extend(rows)
        summaries.append(record)
    with (results_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with (results_root / "per_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["experiment", "sequence", "family", "raw_fill_pct", "placed_boxes", "terminated", "terminated_step", "finished_by_user"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sequence)
    ranking = sorted([row for row in summaries if row.get("returncode") == 0], key=lambda row: (-row["selection_weighted_raw_fill_pct"], row["iid_stddev_raw_fill_pct"], -row["selection_mean_placed_boxes"]))
    report = {"interpreter": str(PYTHON), "jobs": args.jobs, "experiments": summaries, "ranking": ranking}
    (results_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[sweep] report={results_root / 'report.json'}", flush=True)
    return 0 if len(ranking) == len(experiments) else 1


if __name__ == "__main__":
    raise SystemExit(main())
