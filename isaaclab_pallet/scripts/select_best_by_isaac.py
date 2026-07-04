#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_nvidia_libs(env: dict[str, str]) -> dict[str, str]:
    lib_dirs: list[str] = []
    for sp in site.getsitepackages():
        lib_dirs.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    if lib_dirs:
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + ([old] if old else []))
    return env


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name == "PCT-best.pt":
        return (-3, name)
    if name == "PCT-latest.pt":
        return (-2, name)
    if name.startswith("PCT-update-"):
        try:
            return (int(name.removeprefix("PCT-update-").removesuffix(".pt")), name)
        except ValueError:
            pass
    return (-1, name)


def _collect_checkpoints(run_dir: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern), key=_checkpoint_sort_key)
        found.extend(p for p in matches if p.is_file())
    dedup: dict[Path, Path] = {}
    for p in found:
        dedup[p.resolve()] = p
    return sorted(dedup.values(), key=_checkpoint_sort_key)


def _run(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def _score_checkpoint(
    checkpoint: Path,
    work_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict:
    stem = checkpoint.stem
    ckpt_work = work_dir / stem
    alg_dir = ckpt_work / "algorithm_results"
    sim_dir = ckpt_work / "sim_results"
    log_path = ckpt_work / "eval.log"
    alg_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    gen_cmd = [
        sys.executable,
        "isaaclab_pallet/scripts/eval_competition_generate.py",
        "--checkpoint",
        str(checkpoint),
        "--box-seq-dir",
        args.box_seq_dir,
        "--out-dir",
        str(alg_dir),
        "--device",
        args.device,
        "--sequences",
        *args.sequences,
    ]
    _run(gen_cmd, env, log_path)

    sim_cmd = [
        sys.executable,
        "palletizing_simulator/simulator.py",
        "--config",
        args.sim_config,
        "--input-dir",
        str(alg_dir.resolve()),
        "-o",
        str(sim_dir.resolve()),
    ]
    _run(sim_cmd, env, log_path)

    result_path = sim_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    episodes = result["results"]["episodes"]
    success = all(ep["status"] == "success" for ep in episodes)
    return {
        "checkpoint": str(checkpoint),
        "avg_score": float(result["results"]["avg_score"]),
        "success": bool(success),
        "episodes": episodes,
        "result_json": str(result_path),
        "log": str(log_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick the best checkpoint by official Isaac simulator score.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--patterns", nargs="+", default=["PCT-best.pt", "PCT-latest.pt", "PCT-update-*.pt"])
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--box-seq-dir", default=str(PROJECT_ROOT / "palletizing_simulator" / "box_sequence"))
    ap.add_argument("--sim-config", default=str(PROJECT_ROOT / "palletizing_simulator" / "config" / "sim_config.yaml"))
    ap.add_argument("--sequences", nargs="+", default=["box_sequence_0", "box_sequence_1"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-name", default="PCT-best-isaac.pt")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    checkpoints = _collect_checkpoints(run_dir, args.patterns)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched in {run_dir}: {args.patterns}")

    work_dir = Path(args.work_dir) if args.work_dir else PROJECT_ROOT / "artifacts" / "isaac_best_selection" / run_dir.name
    if not work_dir.is_absolute():
        work_dir = PROJECT_ROOT / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    env = _ensure_nvidia_libs(os.environ.copy())
    records = []
    best = None
    for checkpoint in checkpoints:
        print(f"[isaac-best] scoring {checkpoint}", flush=True)
        record = _score_checkpoint(checkpoint, work_dir, args, env)
        records.append(record)
        print(
            f"[isaac-best] score={record['avg_score']:.2f} "
            f"success={record['success']} checkpoint={checkpoint.name}",
            flush=True,
        )
        if best is None:
            best = record
            continue
        if (record["success"], record["avg_score"]) > (best["success"], best["avg_score"]):
            best = record

    assert best is not None
    out_path = run_dir / args.out_name
    shutil.copy2(best["checkpoint"], out_path)

    summary = {
        "best": best,
        "copied_to": str(out_path),
        "records": records,
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tsv_path = work_dir / "summary.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("checkpoint\tavg_score\tsuccess\tresult_json\n")
        for rec in records:
            f.write(f"{Path(rec['checkpoint']).name}\t{rec['avg_score']:.2f}\t{rec['success']}\t{rec['result_json']}\n")

    print(f"[isaac-best] BEST {Path(best['checkpoint']).name} score={best['avg_score']:.2f} success={best['success']}")
    print(f"[isaac-best] copied -> {out_path}")
    print(f"[isaac-best] summary -> {summary_path}")


if __name__ == "__main__":
    main()
