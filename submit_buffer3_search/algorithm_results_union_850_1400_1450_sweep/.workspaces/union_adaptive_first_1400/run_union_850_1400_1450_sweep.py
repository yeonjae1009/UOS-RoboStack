#!/usr/bin/env python3
"""Run the six requested 850/1400/1450 fallback/adaptive experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_final_model_selection_sweep as sweep


MODELS = (
    ("850", "src/models/pct_model.onnx"),
    ("1400", "src/models/candidate-001400.onnx"),
    ("1450", "src/models/candidate-001450.onnx"),
)
ORDERS = (
    ("850", (MODELS[0][1], MODELS[1][1], MODELS[2][1])),
    ("1400", (MODELS[1][1], MODELS[0][1], MODELS[2][1])),
    ("1450", (MODELS[2][1], MODELS[0][1], MODELS[1][1])),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if not sweep.PYTHON.is_file():
        raise FileNotFoundError(sweep.PYTHON)

    config = sweep._load_yaml(sweep.HERE / "config" / "algorithm_config.yaml")
    input_source = Path(config["input_path"])
    if not input_source.is_absolute():
        input_source = sweep.HERE / input_source
    results_root = sweep.HERE / "algorithm_results_union_850_1400_1450_sweep"
    input_view = results_root / "input_sequences"
    sweep._prepare_input_view(input_source, input_view)

    experiments = []
    for mode in ("fallback", "adaptive"):
        for first, paths in ORDERS:
            label = f"union_{mode}_first_{first}"
            spec = {
                "kind": "union",
                "union_mode": mode,
                "first_model": first,
                "model_paths": list(paths),
            }
            experiments.append((label, list(paths), True, mode, spec))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    completed = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                sweep._run_one,
                label,
                paths,
                union,
                mode,
                results_root,
                input_view,
            ): (label, spec)
            for label, paths, union, mode, spec in experiments
        }
        for future in as_completed(futures):
            label, spec = futures[future]
            try:
                completed[label] = {**spec, **future.result()}
                print(
                    f"[sweep] done {label} rc={completed[label]['returncode']}",
                    flush=True,
                )
            except Exception as exc:
                completed[label] = {
                    **spec,
                    "returncode": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    summaries, per_sequence = [], []
    for label, _paths, _union, _mode, spec in experiments:
        record = completed[label]
        if record.get("returncode") == 0:
            metrics, rows = sweep._summarize(label, Path(record["output_dir"]), spec)
            record.update(metrics)
            per_sequence.extend(rows)
        summaries.append(record)

    import csv

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    with (results_root / "per_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "experiment",
            "sequence",
            "family",
            "raw_fill_pct",
            "placed_boxes",
            "terminated",
            "terminated_step",
            "finished_by_user",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sequence)

    ranking = sorted(
        [row for row in summaries if row.get("returncode") == 0],
        key=lambda row: (
            -row["selection_weighted_raw_fill_pct"],
            row["iid_stddev_raw_fill_pct"],
            -row["selection_mean_placed_boxes"],
        ),
    )
    report = {
        "interpreter": str(sweep.PYTHON),
        "jobs": args.jobs,
        "input_source": str(input_source.resolve()),
        "experiments": summaries,
        "ranking": ranking,
    }
    report_path = results_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[sweep] report={report_path}", flush=True)
    return 0 if len(ranking) == len(experiments) else 1


if __name__ == "__main__":
    raise SystemExit(main())
