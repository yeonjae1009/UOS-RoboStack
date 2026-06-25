"""Equivalence regression test (Stage B): original PCT math env  ==  Isaac port CPU layer.

Goal: prove that porting to Isaac Lab did NOT change the deterministic packing /
reward math — i.e. "수학 부분은 완전 동일, 물리만 교체" 의 '동일' 절반을 수치로 보장.

It drives, over the same box sequence with the same deterministic leaf policy:

  REFERENCE = Online-3D-BPP-PCT/pct_envs/PctContinuous0/bin3D.py  (PackingContinuous)
              — the original math-approximation gym env (pure numpy).
  PORT      = templete code/src/pct/packer.py  +  isaaclab_pallet/pct_reward.py
              — the exact CPU code the Isaac DirectRLEnv calls at runtime.

and asserts they agree on, every step:
  internal nodes (EMS), leaf-node candidates, placed box record, occupancy ratio,
  shaped reward and all 7 reward terms.

This does NOT boot Isaac Sim — it tests only the shared CPU layer, which is the
part that must stay identical. The physics (drift/tilt/collapse) is the new piece
and is validated separately.

Run:
  python3 isaaclab_pallet/scripts/test_equivalence.py
  python3 isaaclab_pallet/scripts/test_equivalence.py --max-boxes 250 --policy cycle --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")
# Original convex_hull.py uses np.float (removed in numpy >= 1.24). Shim it so the
# reference runs on modern numpy; it only affects an astype() and equals float64.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = PROJECT_ROOT / "Online-3D-BPP-PCT"
TEMPLATE_DIR = PROJECT_ROOT / "templete code"
PALLET_PKG_DIR = PROJECT_ROOT / "isaaclab_pallet"
DEFAULT_CONFIG = TEMPLATE_DIR / "config" / "pct_config.yaml"
DEFAULT_SEQUENCE = PROJECT_ROOT / "palletizing_simulator" / "box_sequence" / "box_sequence_0.json"
DEFAULT_PALLET = (1.2, 1.0, 1.25)

for p in (str(ORIGINAL_DIR), str(TEMPLATE_DIR), str(PALLET_PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pct_reward  # noqa: E402  (isaaclab_pallet/pct_reward.py — the shipped CPU layer)
from src.pct.packer import Packer  # noqa: E402  (the port's packing driver)
from pct_envs.PctContinuous0.bin3D import PackingContinuous  # noqa: E402  (reference)
from pct_envs.PctContinuous0.space import Space  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_boxes(path: Path, limit: int) -> list[dict]:
    boxes: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            boxes.append(json.loads(line))
            if len(boxes) >= limit:
                break
    return boxes


def pick_action(valid_idx: list[int], step: int, policy: str) -> int:
    """Deterministic leaf choice. Both sides must call this with identical inputs."""
    if policy == "first":
        return valid_idx[0]
    if policy == "last":
        return valid_idx[-1]
    if policy == "cycle":
        return valid_idx[step % len(valid_idx)]
    raise ValueError(f"unknown policy {policy}")


def make_reference(cfg: dict, pallet, internal, leaf):
    env = PackingContinuous(
        setting=int(cfg["setting"]),
        container_size=tuple(pallet),
        item_set=[(0.2, 0.2, 0.2)],
        internal_node_holder=internal,
        leaf_node_holder=leaf,
        sample_from_distribution=False,
    )
    # Force the geometry core to match the port exactly (size_minimum from config).
    env.space = Space(*pallet, float(cfg["size_minimum"]), internal)
    env.space.reset()
    env.size_minimum = float(cfg["size_minimum"])
    env.setting = int(cfg["setting"])
    return env


def make_port(cfg: dict, pallet, internal, leaf):
    packer = Packer(
        container_size=list(pallet),
        size_minimum=float(cfg["size_minimum"]),
        internal_node_holder=internal,
        leaf_node_holder=leaf,
        setting=int(cfg["setting"]),
    )
    packer.reset()
    return packer


def reference_probe(env, box, density):
    """Set the current box and build leaf candidates (no placement yet).

    Returns (internal_nodes_pre_place, leaf_candidates).
    """
    env.next_box = [float(v) for v in box["size"]]
    env.next_den = float(density)
    leaves = env.get_possible_position()
    internal = np.array(env.space.box_vec, dtype=np.float64)
    return internal, np.array(leaves, dtype=np.float64)


def reference_place(env, leaves, action_idx):
    before = env._layout_metrics()
    box_ratio = env.get_box_ratio()
    action, next_box = env.LeafNode2Action(leaves[action_idx])
    idx = [round(action[1], 6), round(action[2], 6)]
    if not env.space.drop_box(next_box, idx, action[0], env.next_den, env.setting):
        return None
    packed = env.space.boxes[-1]
    after = env._layout_metrics()
    env.space.GENEMS([
        packed.lx, packed.ly, packed.lz,
        round(packed.lx + packed.x, 6),
        round(packed.ly + packed.y, 6),
        round(packed.lz + packed.z, 6),
    ])
    reward = env._compute_shaped_reward(box_ratio, packed, before, after)
    return {
        "record": np.array([packed.x, packed.y, packed.z, packed.lx, packed.ly, packed.lz], dtype=np.float64),
        "ratio": float(env.space.get_ratio()),
        "reward": float(reward),
        "terms": {k: float(v) for k, v in env.last_reward_terms.items()},
    }


def port_probe(packer, box, density, internal, leaf_holder):
    """observe() builds leaf candidates without placing. Returns (internal, leaves)."""
    obs_arr = packer.observe(box["size"], density=density).reshape(internal + leaf_holder + 1, 9)
    return (np.array(obs_arr[:internal], dtype=np.float64),
            np.array(obs_arr[internal:internal + leaf_holder], dtype=np.float64))


def port_place(packer, leaves, box, pallet, action_idx):
    before = pct_reward.layout_metrics(packer.space.boxes, pallet)
    box_ratio = float(np.prod([float(v) for v in box["size"]]) / np.prod(pallet))
    leaf = pct_reward.select_leaf(leaves, action_idx)
    if leaf is None or not packer.place(leaf[:6]):
        return None
    packed = packer.space.boxes[-1]
    after = pct_reward.layout_metrics(packer.space.boxes, pallet)
    reward, terms = pct_reward.compute_online3dbpp_reward(box_ratio, packed, before, after)
    return {
        "record": np.array(packer.packed[-1][:6], dtype=np.float64),
        "ratio": float(packer.get_ratio()),
        "reward": float(reward),
        "terms": {k: float(v) for k, v in terms.items()},
    }


def valid_indices(leaves: np.ndarray) -> list[int]:
    return [i for i in range(leaves.shape[0]) if float(leaves[i, 8]) > 0.5]


def main() -> int:
    parser = argparse.ArgumentParser(description="Original PCT vs Isaac port CPU equivalence test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--max-boxes", type=int, default=250)
    parser.add_argument("--policy", choices=["first", "last", "cycle"], default="cycle")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    internal = int(cfg["internal_node_holder"])
    leaf_holder = int(cfg["leaf_node_holder"])
    setting = int(cfg["setting"])
    density_max = float(cfg.get("density_max", 1.0))
    pallet = DEFAULT_PALLET
    boxes = load_boxes(args.sequence, args.max_boxes)

    print(f"config={args.config.name} setting={setting} internal={internal} leaf={leaf_holder} "
          f"density_max={density_max:.4f}")
    print(f"sequence={args.sequence.name} boxes={len(boxes)} pallet={pallet} policy={args.policy} tol={args.tol:g}")

    env = make_reference(cfg, pallet, internal, leaf_holder)
    packer = make_port(cfg, pallet, internal, leaf_holder)

    keys = ["internal", "leaves", "record", "ratio", "reward"]
    worst = {k: 0.0 for k in keys}
    worst_term = 0.0
    placed = 0
    mismatches = 0

    for step, box in enumerate(boxes):
        density = pct_reward.density_for_box(box, setting, density_max)

        # 1) Probe: both sides build leaf candidates from identical state (no placement).
        ref_internal, ref_leaves = reference_probe(env, box, density)
        port_internal, port_leaves = port_probe(packer, box, density, internal, leaf_holder)

        ref_valid = valid_indices(ref_leaves)
        port_valid = valid_indices(port_leaves)
        if ref_valid != port_valid:
            print(f"[step {step}] FAIL: valid-leaf index sets differ "
                  f"(ref {len(ref_valid)} vs port {len(port_valid)})")
            mismatches += 1
            break
        if not ref_valid:
            print(f"[step {step}] both terminate: no feasible leaf. placed={placed}")
            break

        # 2) Same deterministic policy -> same leaf index on both sides.
        action_idx = pick_action(ref_valid, step, args.policy)

        # 3) Place and collect outcomes.
        ref = reference_place(env, ref_leaves, action_idx)
        port = port_place(packer, port_leaves, box, pallet, action_idx)
        if ref is None or port is None:
            print(f"[step {step}] placement failed ref={ref is None} port={port is None}")
            mismatches += 1
            break
        ref["internal"], ref["leaves"] = ref_internal, ref_leaves
        port["internal"], port["leaves"] = port_internal, port_leaves

        step_diffs = {}
        for k in keys:
            d = float(np.max(np.abs(np.asarray(ref[k]) - np.asarray(port[k]))))
            step_diffs[k] = d
            worst[k] = max(worst[k], d)
        term_diff = max(abs(ref["terms"][t] - port["terms"][t]) for t in ref["terms"])
        worst_term = max(worst_term, term_diff)
        placed += 1

        bad = max(step_diffs.values()) > args.tol or term_diff > args.tol
        if bad:
            mismatches += 1
            print(f"[step {step}] MISMATCH idx={action_idx} diffs={ {k: f'{v:.2e}' for k, v in step_diffs.items()} } "
                  f"term={term_diff:.2e}")
            print(f"           ref reward={ref['reward']:.8f} port reward={port['reward']:.8f}")
            if mismatches >= 5:
                break
        elif args.verbose:
            print(f"[step {step}] ok idx={action_idx} reward={ref['reward']:.6f} ratio={ref['ratio']:.4f} "
                  f"max_diff={max(step_diffs.values()):.2e}")

    print("-" * 70)
    print("worst abs diff per field:")
    for k in keys:
        print(f"  {k:10s}: {worst[k]:.3e}")
    print(f"  {'terms':10s}: {worst_term:.3e}")
    print(f"placed steps compared: {placed}   mismatches: {mismatches}")

    overall = max(list(worst.values()) + [worst_term])
    ok = mismatches == 0 and overall <= args.tol
    print(f"\n{'PASS ✅' if ok else 'FAIL ❌'}  (max abs diff {overall:.3e} <= tol {args.tol:g} = {overall <= args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
