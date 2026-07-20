#!/usr/bin/env python3
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
ALG_CFG = HERE / "config" / "algorithm_config.yaml"
PCT_CFG = HERE / "config" / "pct_config.yaml"
SUMMARY_PATH = HERE / "algorithm_results_first_model_sweep" / "summary.json"

MODELS = [
    "src/models/candidate-001400.onnx",
    "src/models/candidate-001450.onnx",
    "src/models/pct_model_bestonnx.onnx",
]

MODES = ["adaptive", "fallback"]
ORDERS = [
    (
        "1400",
        (
            "src/models/candidate-001400.onnx",
            "src/models/candidate-001450.onnx",
            "src/models/pct_model_bestonnx.onnx",
        ),
    ),
    (
        "1450",
        (
            "src/models/candidate-001450.onnx",
            "src/models/candidate-001400.onnx",
            "src/models/pct_model_bestonnx.onnx",
        ),
    ),
    (
        "best",
        (
            "src/models/pct_model_bestonnx.onnx",
            "src/models/candidate-001400.onnx",
            "src/models/candidate-001450.onnx",
        ),
    ),
]


def model_label(path: str) -> str:
    name = Path(path).name
    if "001400" in name:
        return "1400"
    if "001450" in name:
        return "1450"
    if "best" in name:
        return "best"
    return name.replace(".onnx", "")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def configure(order: tuple[str, ...], mode: str, output_dir: str) -> None:
    alg = load_yaml(ALG_CFG)
    pct = load_yaml(PCT_CFG)

    alg["output_dir"] = output_dir
    alg.setdefault("buffer", {})["size"] = 0
    search = alg.setdefault("search", {})
    search["enabled"] = True
    search["candidate_union_enabled"] = True
    search["candidate_union_mode"] = mode

    pct["model_paths"] = list(order)

    write_yaml(ALG_CFG, alg)
    write_yaml(PCT_CFG, pct)


def summarize_result(output_dir: str) -> dict:
    result_dir = HERE / output_dir
    rows = []
    for path in sorted(result_dir.glob("random_spec_*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        volume = sum(math.prod(item["size"]) for item in data["sequence"])
        util = volume / 1.5
        rows.append(
            {
                "file": path.name,
                "utilization": util,
                "placed_count": len(data["sequence"]),
                "terminated": bool(data.get("terminated")),
                "terminated_step": data.get("terminated_step"),
                "buffer_size": data.get("buffer_size"),
            }
        )

    if not rows:
        return {"files": 0}

    min_row = min(rows, key=lambda item: item["utilization"])
    max_row = max(rows, key=lambda item: item["utilization"])
    avg_util = sum(item["utilization"] for item in rows) / len(rows)
    avg_count = sum(item["placed_count"] for item in rows) / len(rows)
    return {
        "files": len(rows),
        "avg_utilization": avg_util,
        "avg_percent": avg_util * 100.0,
        "avg_count": avg_count,
        "total_count": sum(item["placed_count"] for item in rows),
        "min_file": min_row["file"],
        "min_percent": min_row["utilization"] * 100.0,
        "min_count": min_row["placed_count"],
        "max_file": max_row["file"],
        "max_percent": max_row["utilization"] * 100.0,
        "max_count": max_row["placed_count"],
        "terminated_count": sum(1 for item in rows if item["terminated"]),
        "nonzero_buffer_files": sum(1 for item in rows if item["buffer_size"]),
    }


def main() -> int:
    original_alg = ALG_CFG.read_text(encoding="utf-8")
    original_pct = PCT_CFG.read_text(encoding="utf-8")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    total = len(MODES) * len(ORDERS)
    index = 0

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-codex"
    env["PYTHONUNBUFFERED"] = "1"

    try:
        for mode in MODES:
            for first_label, order in ORDERS:
                index += 1
                order_label = "-".join(model_label(path) for path in order)
                run_name = f"{mode}_first-{first_label}_{order_label}"
                output_dir = f"algorithm_results_first_model_sweep/{run_name}"
                log_path = HERE / "algorithm_results_first_model_sweep" / f"{run_name}.log"
                print(
                    f"[SWEEP] {index}/{total} start mode={mode} first={first_label} "
                    f"order={order_label}",
                    flush=True,
                )

                configure(order, mode, output_dir)
                start = time.time()
                with log_path.open("w", encoding="utf-8") as log_file:
                    proc = subprocess.run(
                        [sys.executable, "main.py"],
                        cwd=str(HERE),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                elapsed = time.time() - start
                result = summarize_result(output_dir)
                result.update(
                    {
                        "mode": mode,
                        "order": list(order),
                        "order_label": order_label,
                        "output_dir": output_dir,
                        "log": str(log_path.relative_to(HERE)),
                        "returncode": proc.returncode,
                        "elapsed_sec": elapsed,
                    }
                )
                summaries.append(result)
                SUMMARY_PATH.write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                avg = result.get("avg_percent")
                avg_text = f"{avg:.4f}%" if isinstance(avg, (int, float)) else "n/a"
                print(
                    f"[SWEEP] {index}/{total} done rc={proc.returncode} avg={avg_text} "
                    f"count={result.get('total_count')} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        ALG_CFG.write_text(original_alg, encoding="utf-8")
        PCT_CFG.write_text(original_pct, encoding="utf-8")
        print("[SWEEP] restored original config files", flush=True)

    failed = [item for item in summaries if item.get("returncode") != 0]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
