#!/usr/bin/env python3
"""Generate competition algorithm_results JSON from a trained Isaac GAT checkpoint.

Mirrors the deployment `algorithm.py.Palletizer.run()` loop EXACTLY (same packer,
same observation, same output format) but swaps the ONNX session for our torch
DRL_GAT checkpoint (PCT-latest.pt). The produced JSON is then fed to the official
`palletizing_simulator/simulator.py` (Isaac physics) + `evaluator.py` for scoring.

No Isaac needed here — pure numpy packer + torch policy (CPU/GPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "templete code"
ONLINE_PCT_DIR = PROJECT_ROOT / "Online-3D-BPP-PCT"
sys.path.insert(0, str(TEMPLATE_DIR))          # for `src.pct.packer`
sys.path.insert(0, str(ONLINE_PCT_DIR))        # for `model`, `tools`

from src.pct.packer import Packer  # noqa: E402  (templete packer = training packer)
import tools as pct_tools  # noqa: E402
from model import DRL_GAT  # noqa: E402


def load_boxes(path: Path) -> list[dict]:
    """Read JSONL (one box per line) or a JSON array."""
    text = path.read_text().strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("sequence", [])
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_policy(ckpt_path: Path, cfg: dict, device: str) -> tuple[DRL_GAT, SimpleNamespace]:
    setting = int(cfg["setting"])
    pct_args = SimpleNamespace(
        setting=setting,
        internal_node_holder=int(cfg["internal_node_holder"]),
        internal_node_length=7 if setting == 3 else 6,
        leaf_node_holder=int(cfg["leaf_node_holder"]),
        embedding_size=64,
        hidden_size=128,
        gat_layer_num=1,
        normFactor=float(cfg.get("norm_factor", 0.8)),
    )
    policy = DRL_GAT(pct_args)
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    try:
        policy.load_state_dict(sd)
    except RuntimeError:
        pct_tools.load_policy(str(ckpt_path), policy)
    policy.to(device).eval()
    return policy, pct_args


def run_sequence(boxes: list[dict], policy: DRL_GAT, pct_args: SimpleNamespace, cfg: dict,
                 device: str, sample: bool = False) -> dict:
    container = [1.2, 1.0, 1.25]
    INH = pct_args.internal_node_holder
    LNH = pct_args.leaf_node_holder
    setting = pct_args.setting
    density_max = float(cfg.get("density_max", 1.0))
    node_count = INH + LNH + 1

    packer = Packer(container, float(cfg["size_minimum"]), INH, LNH, setting)
    packer.reset()

    sequence: list[dict] = []
    terminated = False
    terminated_step = None

    for box in boxes:
        size = [float(box["size"][0]), float(box["size"][1]), float(box["size"][2])]
        if setting >= 3:
            vol = max(size[0] * size[1] * size[2], 1e-9)
            density = (float(box["mass"]) / vol) / density_max
            obs = packer.observe(size, density=density)
        else:
            obs = packer.observe(size)
        obs_arr = obs.reshape(node_count, 9).astype(np.float32)

        leaf_region = obs_arr[INH:INH + LNH, :]
        if float(leaf_region[:, 8].sum()) <= 0.0:
            terminated, terminated_step = True, int(box["step"])
            break

        pct_obs = torch.from_numpy(obs_arr.reshape(1, node_count, 9)).to(device)
        all_nodes, _leaf_nodes = pct_tools.get_leaf_nodes(pct_obs, INH, LNH)
        all_nodes = all_nodes.to(device)
        with torch.no_grad():
            _, selected_idx, _, _ = policy(
                all_nodes, deterministic=not sample, normFactor=pct_args.normFactor
            )
        sel = int(selected_idx.flatten()[0].item())

        leaf = leaf_region[sel]
        if float(np.sum(leaf[0:6])) == 0.0:
            terminated, terminated_step = True, int(box["step"])
            break
        if not packer.place(leaf[0:6]):
            terminated, terminated_step = True, int(box["step"])
            break

        x, y, z, lx, ly, lz, _ = [float(v) for v in packer.packed[-1]]
        L, W, _H = size
        rotation = 0 if (abs(x - L) < 1e-3 and abs(y - W) < 1e-3) else 90
        sequence.append({
            "step": int(box["step"]),
            "id": int(box["id"]),
            "size": [round(x, 3), round(y, 3), round(z, 3)],
            "mass": float(box["mass"]),
            "position": [round(lx + x / 2.0, 3), round(ly + y / 2.0, 3), round(lz + z / 2.0, 3)],
            "rotation": int(rotation),
        })

    return {
        "buffer_size": 0,
        "sequence": sequence,
        "terminated": terminated,
        "terminated_step": terminated_step,
        "finished_by_user": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pct-config", default=str(TEMPLATE_DIR / "config" / "pct_config.yaml"))
    ap.add_argument("--box-seq-dir", default=str(PROJECT_ROOT / "palletizing_simulator" / "box_sequence"))
    ap.add_argument("--out-dir", required=True, help="Where to write algorithm_results JSON")
    ap.add_argument("--sequences", nargs="+", default=["box_sequence_0", "box_sequence_1"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.pct_config).read_text())
    policy, pct_args = build_policy(Path(args.checkpoint), cfg, args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pallet_vol = 1.2 * 1.0 * 1.25
    print(f"[gen] checkpoint={args.checkpoint} setting={pct_args.setting} "
          f"INH={pct_args.internal_node_holder} LNH={pct_args.leaf_node_holder} "
          f"density_max={cfg.get('density_max')}", flush=True)

    for name in args.sequences:
        boxes = load_boxes(Path(args.box_seq_dir) / f"{name}.json")
        result = run_sequence(boxes, policy, pct_args, cfg, args.device, args.sample)
        placed = result["sequence"]
        stacked_vol = sum(b["size"][0] * b["size"][1] * b["size"][2] for b in placed)
        max_top = max((b["position"][2] + b["size"][2] / 2.0 for b in placed), default=0.0)
        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[gen] {name}: placed={len(placed)}/{len(boxes)} "
              f"raw_fill={stacked_vol / pallet_vol * 100:.1f}% max_top={max_top:.3f}m "
              f"terminated_step={result['terminated_step']} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
