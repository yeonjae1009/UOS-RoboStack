#!/usr/bin/env python3
"""Run initial labels, pass-1 training, DAgger labels and final training."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
INFERENCE_PYTHON = Path("/home/robotics/Documents/assignment2/.venv/bin/python")
TRAIN_PYTHON = Path("/usr/bin/python3")


def _run(stage: str, command: list[str], status_path: Path) -> None:
    started = time.time()
    status = {"stage": stage, "state": "running", "started_unix": started, "command": command}
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"[pipeline] START {stage}", flush=True)
    completed = subprocess.run(command, cwd=HERE, check=False)
    status.update({
        "state": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_sec": time.time() - started,
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"[pipeline] END {stage} rc={completed.returncode} elapsed={status['elapsed_sec']:.1f}s", flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "candidate_value_training_v1")
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "pipeline_status.json"
    dataset = (ROOT / "artifacts" / "hybrid_mpc_dev_v1").resolve()
    initial = artifact_root / "labels_initial"
    dagger = artifact_root / "labels_dagger2"
    pass1 = HERE / "src" / "models" / "candidate_value_pass1.onnx"
    final = HERE / "src" / "models" / "candidate_value.onnx"

    if not (initial / "metadata.json").is_file():
        _run("initial_labels", [
            str(INFERENCE_PYTHON), "tools/collect_candidate_value_labels.py",
            "--dataset", str(dataset), "--output", str(initial),
            "--pass-name", "initial", "--jobs", str(args.jobs),
        ], status_path)
    else:
        print("[pipeline] SKIP initial_labels (metadata exists)", flush=True)

    if not pass1.is_file():
        _run("pass1_training", [
            str(TRAIN_PYTHON), "tools/train_candidate_value.py",
            "--labels", str(initial), "--output", str(pass1),
            "--epochs", str(args.epochs),
        ], status_path)
    else:
        print("[pipeline] SKIP pass1_training (model exists)", flush=True)

    if not (dagger / "metadata.json").is_file():
        _run("dagger2_labels", [
            str(INFERENCE_PYTHON), "tools/collect_candidate_value_labels.py",
            "--dataset", str(dataset), "--output", str(dagger),
            "--behavior-model", str(pass1), "--pass-name", "dagger2",
            "--jobs", str(args.jobs),
        ], status_path)
    else:
        print("[pipeline] SKIP dagger2_labels (metadata exists)", flush=True)

    _run("final_training", [
        str(TRAIN_PYTHON), "tools/train_candidate_value.py",
        "--labels", str(initial), str(dagger), "--output", str(final),
        "--epochs", str(args.epochs),
    ], status_path)
    summary = {
        "state": "complete",
        "final_model": str(final),
        "final_metadata": str(final.with_suffix(".metadata.json")),
        "initial_labels": str(initial),
        "dagger_labels": str(dagger),
    }
    status_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[pipeline] COMPLETE final={final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
