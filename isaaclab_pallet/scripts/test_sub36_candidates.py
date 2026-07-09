#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "templete code"
PALLET_PKG_DIR = PROJECT_ROOT / "isaaclab_pallet"
DEFAULT_CONFIG = TEMPLATE_DIR / "config" / "pct_config.yaml"
DEFAULT_SEQUENCE = PROJECT_ROOT / "palletizing_simulator" / "box_sequence" / "box_sequence_0.json"
DEFAULT_PALLET = (1.2, 1.0, 1.25)

for p in (str(TEMPLATE_DIR), str(PALLET_PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pct_reward  # noqa: E402
from src.pct.packer import Packer  # noqa: E402


def load_boxes(path: Path, limit: int) -> list[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        boxes = data if isinstance(data, list) else data.get("sequence", [])
    except json.JSONDecodeError:
        boxes = [json.loads(line) for line in text.splitlines() if line.strip()]
    return boxes[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the sub36 PCT candidate generator.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--max-boxes", type=int, default=120)
    parser.add_argument("--policy", choices=["first", "last", "cycle"], default="cycle")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    internal = int(cfg["internal_node_holder"])
    leaf = int(cfg["leaf_node_holder"])
    setting = int(cfg["setting"])
    density_max = float(cfg.get("density_max", 1.0))
    boxes = load_boxes(args.sequence, args.max_boxes)

    packer = Packer(
        list(DEFAULT_PALLET),
        float(cfg["size_minimum"]),
        internal,
        leaf,
        setting,
        candidate_generator="sub36",
    )
    packer.reset()

    placed = 0
    for step, box in enumerate(boxes):
        density = pct_reward.density_for_box(box, setting, density_max)
        obs = packer.observe(box["size"], density=density).reshape(internal + leaf + 1, 9)
        leaves = obs[internal:internal + leaf]
        valid = [idx for idx in range(leaf) if float(leaves[idx, 8]) > 0.5]
        if not valid:
            print(f"[step {step}] no valid sub36 leaf; placed={placed}")
            break
        if args.policy == "first":
            action_idx = valid[0]
        elif args.policy == "last":
            action_idx = valid[-1]
        else:
            action_idx = valid[step % len(valid)]
        if not packer.place(leaves[action_idx, :6]):
            print(f"[step {step}] place failed action_idx={action_idx} valid={len(valid)}")
            return 1
        placed += 1

    ratio = packer.get_ratio()
    print(f"sub36 candidate smoke ok: placed={placed}/{len(boxes)} ratio={ratio:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
