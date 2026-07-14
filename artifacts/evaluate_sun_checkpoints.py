#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "isaaclab_pallet" / "scripts"))

from select_best_by_isaac import _ensure_nvidia_libs, _score_checkpoint  # noqa: E402


DATASETS = {
    "eval12": {
        "box_seq_dir": "artifacts/random_spec_eval_10_plus_official/box_sequence",
        "sequences": [*(f"random_spec_{i:03d}" for i in range(10)), "box_sequence_0", "box_sequence_1"],
    },
    "eval52": {
        "box_seq_dir": "artifacts/random_spec_eval_50_plus_official/box_sequence",
        "sequences": [*(f"random_spec_{i:03d}" for i in range(50)), "box_sequence_0", "box_sequence_1"],
    },
}


CHECKPOINTS = [
    ("base_terminal18", "isaaclab_pallet/runs/reward_terminal18_finish15/terminal_ratio_t18_from_terminal_best/PCT-best.pt"),
    ("gap52_best", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand52official_sun_gap_guard_v1_deploybest/PCT-best.pt"),
    ("gap52_latest", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand52official_sun_gap_guard_v1_deploybest/PCT-latest.pt"),
    ("gap52_c1000", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001000.pt"),
    ("gap52_c1400", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001400.pt"),
    ("gap52_c1450", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001450.pt"),
    ("gap52_c1750", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001750.pt"),
    ("gap12to52_best", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_gap_guard_v1_deploybest/PCT-best.pt"),
    ("gap12to52_latest", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_gap_guard_v1_deploybest/PCT-latest.pt"),
    ("gap12to52_c0900", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-000900.pt"),
    ("gap12to52_c1450", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001450.pt"),
    ("gap12to52_c1600", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001600.pt"),
    ("gap12to52_c1750", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_gap_guard_v1_deploybest/deploy_eval/candidate-001750.pt"),
    ("sunv2_12to52_best", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_v2_resume_deploybest/PCT-best.pt"),
    ("sunv2_12to52_latest", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_v2_resume_deploybest/PCT-latest.pt"),
    ("sunv2_12to52_c1800", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_v2_resume_deploybest/deploy_eval/candidate-001800.pt"),
    ("sunv2_12to52_c2250", "isaaclab_pallet/runs/sun_terminal18_env128_workers4_rand12to52official_sun_v2_resume_deploybest/deploy_eval/candidate-002250.pt"),
]


def _episode_scores(record: dict) -> list[float]:
    scores = []
    for episode in record["episodes"]:
        score = episode.get("score")
        if score is None:
            score = episode.get("result", {}).get("score")
        scores.append(float(score))
    return scores


def _write_sim_config(work_dir: Path, box_seq_dir: Path) -> Path:
    src = ROOT / "palletizing_simulator" / "config" / "sim_config.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    cfg.setdefault("paths", {})["box_sequence_dir"] = str(box_seq_dir.resolve())
    cfg["paths"]["input_dir"] = "algorithm_results"
    cfg["paths"]["output_dir"] = "sim_results"
    out = work_dir / "sim_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="artifacts/sun_checkpoint_eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    work_dir = ROOT / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    env = _ensure_nvidia_libs(os.environ.copy())
    records = []

    for label, rel_path in CHECKPOINTS:
        if args.only and label not in args.only:
            continue
        checkpoint = ROOT / rel_path
        if not checkpoint.exists():
            print(f"[skip] missing {label}: {rel_path}", flush=True)
            continue
        for ds_name, ds in DATASETS.items():
            eval_dir = work_dir / label / ds_name
            sim_config = _write_sim_config(eval_dir, ROOT / ds["box_seq_dir"])
            ns = argparse.Namespace(
                box_seq_dir=str(ROOT / ds["box_seq_dir"]),
                sim_config=str(sim_config),
                sequences=ds["sequences"],
                device=args.device,
            )
            print(f"[eval] {label} {ds_name} {checkpoint}", flush=True)
            record = _score_checkpoint(checkpoint, eval_dir, ns, env)
            scores = _episode_scores(record)
            out = {
                "label": label,
                "dataset": ds_name,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "count": len(scores),
                "avg": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "success": record["success"],
                "result_json": record["result_json"],
            }
            records.append(out)
            print(
                f"[result] {label} {ds_name} avg={out['avg']:.4f} min={out['min']:.4f} max={out['max']:.4f}",
                flush=True,
            )
            (work_dir / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            with (work_dir / "summary.tsv").open("w", encoding="utf-8") as f:
                f.write("label\tdataset\tcount\tavg\tmin\tmax\tsuccess\tcheckpoint\tresult_json\n")
                for rec in records:
                    f.write(
                        f"{rec['label']}\t{rec['dataset']}\t{rec['count']}\t"
                        f"{rec['avg']:.4f}\t{rec['min']:.4f}\t{rec['max']:.4f}\t"
                        f"{rec['success']}\t{rec['checkpoint']}\t{rec['result_json']}\n"
                    )


if __name__ == "__main__":
    main()
