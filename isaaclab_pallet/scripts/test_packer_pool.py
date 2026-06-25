"""Issue ①: prove the parallel packer pool == serial pool (bit-identical), and
benchmark the speedup. Runs WITHOUT Isaac (packer is pure numpy).

  python3 isaaclab_pallet/scripts/test_packer_pool.py --num-envs 16 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")

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
from packer_pool import PackerConfig, make_packer_pool  # noqa: E402


def load_boxes(path, limit):
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                boxes.append(json.loads(line))
            if len(boxes) >= limit:
                break
    return boxes


def make_cfg(yaml_cfg):
    return PackerConfig(
        pallet_size=DEFAULT_PALLET,
        size_minimum=float(yaml_cfg["size_minimum"]),
        internal_node_holder=int(yaml_cfg["internal_node_holder"]),
        leaf_node_holder=int(yaml_cfg["leaf_node_holder"]),
        setting=int(yaml_cfg["setting"]),
        density_max=float(yaml_cfg.get("density_max", 1.0)),
        scales=pct_reward.RewardScales(),
    )


def run_pool(pool, cfg, boxes, num_envs, max_steps):
    """Drive a pool; each env uses a different deterministic policy offset so the
    workers exercise independent packer states. Returns per-env trajectory."""
    internal, leaf = cfg.internal_node_holder, cfg.leaf_node_holder
    box_idx = {e: 0 for e in range(num_envs)}
    active = set(range(num_envs))
    traj = {e: [] for e in range(num_envs)}

    for step in range(max_steps):
        if not active:
            break
        # 1) observe active envs for their current box
        obs_req = {e: boxes[box_idx[e]] for e in active if box_idx[e] < len(boxes)}
        if not obs_req:
            break
        obs_out = pool.observe(obs_req)
        # 2) per-env deterministic action from its valid leaves
        step_req = {}
        for e, obs in obs_out.items():
            leaves = obs[internal:internal + leaf]
            valid = [i for i in range(leaf) if leaves[i, 8] > 0.5]
            if not valid:
                active.discard(e)
                continue
            aidx = valid[(step + e) % len(valid)]
            step_req[e] = (boxes[box_idx[e]], aidx)
        if not step_req:
            break
        # 3) place
        res = pool.step(step_req)
        for e, r in res.items():
            if r["status"] != "ok":
                active.discard(e)
                traj[e].append(("end", r["status"]))
                continue
            traj[e].append((
                round(r["reward"], 9), tuple(round(v, 9) for v in r["packed"]),
                round(r["ratio"], 9), r["valid_count"], r["rotation"],
            ))
            box_idx[e] += 1
            if box_idx[e] >= len(boxes):
                active.discard(e)
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-boxes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()

    yaml_cfg = yaml.safe_load(open(args.config))
    cfg = make_cfg(yaml_cfg)
    boxes = load_boxes(args.sequence, args.max_boxes)
    print(f"num_envs={args.num_envs} workers={args.workers} boxes={len(boxes)}")

    # Serial reference
    t0 = time.perf_counter()
    serial = make_packer_pool(args.num_envs, cfg, num_workers=0)
    serial_traj = run_pool(serial, cfg, boxes, args.num_envs, args.max_steps)
    serial.close()
    t_serial = time.perf_counter() - t0

    # Parallel
    t0 = time.perf_counter()
    parallel = make_packer_pool(args.num_envs, cfg, num_workers=args.workers)
    parallel_traj = run_pool(parallel, cfg, boxes, args.num_envs, args.max_steps)
    parallel.close()
    t_parallel = time.perf_counter() - t0

    # Compare
    mismatches = 0
    for e in range(args.num_envs):
        if serial_traj[e] != parallel_traj[e]:
            mismatches += 1
            if mismatches <= 3:
                n = min(len(serial_traj[e]), len(parallel_traj[e]))
                first = next((i for i in range(n) if serial_traj[e][i] != parallel_traj[e][i]), n)
                print(f"  env {e} differs at step {first}: "
                      f"serial={serial_traj[e][first] if first < len(serial_traj[e]) else 'END'} "
                      f"parallel={parallel_traj[e][first] if first < len(parallel_traj[e]) else 'END'}")
    total_placed = sum(len([t for t in serial_traj[e] if t[0] != 'end']) for e in range(args.num_envs))
    print("-" * 60)
    print(f"placements compared: {total_placed}   env mismatches: {mismatches}/{args.num_envs}")
    print(f"serial:   {t_serial:.3f}s")
    print(f"parallel: {t_parallel:.3f}s   speedup x{t_serial / max(t_parallel, 1e-9):.2f}")
    ok = mismatches == 0
    print(f"\n{'PASS - parallel is bit-identical to serial' if ok else 'FAIL - trajectories diverge'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
