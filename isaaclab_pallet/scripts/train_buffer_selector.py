#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "isaaclab_pallet"))

from buffer_selector_model import (  # noqa: E402
    BUFFER_SELECTOR_BUFFER_SIZE,
    BUFFER_SELECTOR_INPUT_DIM,
    BufferSelectorMLP,
)


def generate_random_boxes(
    count: int,
    rng: np.random.Generator,
    wl_range: tuple[float, float] = (0.17, 0.32),
    h_range: tuple[float, float] = (0.13, 0.26),
    mass_range: tuple[float, float] = (0.5, 6.0),
) -> list[dict]:
    boxes = []
    for idx in range(count):
        boxes.append({
            "step": idx,
            "id": idx,
            "size": [
                round(float(rng.uniform(*wl_range)), 3),
                round(float(rng.uniform(*wl_range)), 3),
                round(float(rng.uniform(*h_range)), 3),
            ],
            "mass": round(float(rng.uniform(*mass_range)), 3),
        })
    return boxes


def load_submit_classes(submit_dir: Path):
    sys.path.insert(0, str(submit_dir.resolve()))
    from algorithm import AlgorithmConfig, PalletConfig, Palletizer  # noqa: PLC0415
    from buffer_manager import BufferManager  # noqa: PLC0415

    return AlgorithmConfig, PalletConfig, Palletizer, BufferManager


def collect_oracle_samples(
    submit_dir: Path,
    episodes: int,
    max_boxes: int,
    seed: int,
    oracle_mode: str,
    oracle_top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    AlgorithmConfig, PalletConfig, Palletizer, BufferManager = load_submit_classes(submit_dir)
    cfg = yaml.safe_load((submit_dir / "config" / "algorithm_config.yaml").read_text())
    pallet_cfg = PalletConfig(
        length=float(cfg["pallet"]["length"]),
        width=float(cfg["pallet"]["width"]),
        height=float(cfg["pallet"]["height"]),
    )
    algo_cfg = AlgorithmConfig(
        allow_rotation=bool(cfg["algorithm"]["allow_rotation"]),
        buffer_size=BUFFER_SELECTOR_BUFFER_SIZE,
    )

    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []

    for episode in range(episodes):
        boxes = generate_random_boxes(max_boxes, rng)
        palletizer = Palletizer(pallet_cfg, algo_cfg)
        palletizer._selector_enabled = False
        palletizer._search_enabled = True
        palletizer._search_mode = oracle_mode
        palletizer._search_top_k_leaf = int(oracle_top_k)
        palletizer._reset_state()

        buf = BufferManager(BUFFER_SELECTOR_BUFFER_SIZE)
        buf.reset(boxes)

        while buf.has_pending():
            current = buf.get_buffer()
            if len(current) == 0:
                break
            if len(palletizer.sequence) >= 130:
                break
            if not palletizer._ensure_packer_for_current_run():
                break

            feat, valid_mask = palletizer._selector_features(current)
            if oracle_mode == "beam":
                candidate = palletizer._plan_beam(current)
            else:
                candidate = palletizer._plan_one_step(current)
            if candidate is None:
                break

            label = int(candidate["buffer_index"])
            if label < 0 or label >= BUFFER_SELECTOR_BUFFER_SIZE:
                break

            features.append(feat)
            masks.append(valid_mask.astype(np.bool_))
            labels.append(label)

            x, y, z, dims, rotation = candidate["validated"]
            palletizer._packer = candidate["packer"]
            palletizer._pending_raw_placed = candidate["raw_placed"]
            palletizer._append_placed(
                box=candidate["box"],
                dims=dims,
                rotation=rotation,
                x=x,
                y=y,
                z=z,
            )
            buf.pop_selected(label)

        if (episode + 1) % max(1, episodes // 10) == 0:
            print(
                f"[selector-data] episode={episode + 1}/{episodes} samples={len(labels)}",
                flush=True,
            )

    if not labels:
        raise RuntimeError("No selector samples were collected")

    return (
        np.stack(features).astype(np.float32),
        np.stack(masks).astype(np.bool_),
        np.asarray(labels, dtype=np.int64),
    )


def train_selector(
    features: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> tuple[BufferSelectorMLP, dict]:
    rng = np.random.default_rng(0)
    order = rng.permutation(len(labels))
    split = max(1, int(len(labels) * 0.9))
    train_idx = order[:split]
    val_idx = order[split:] if split < len(labels) else order[:0]

    model = BufferSelectorMLP(input_dim=features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    x_all = torch.from_numpy(features).to(device)
    mask_all = torch.from_numpy(masks).to(device)
    y_all = torch.from_numpy(labels).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_perm = train_idx[rng.permutation(len(train_idx))]
        total_loss = 0.0
        total_correct = 0
        total = 0
        for start in range(0, len(train_perm), batch_size):
            idx = torch.from_numpy(train_perm[start:start + batch_size]).to(device)
            logits = model(x_all[idx])
            logits = logits.masked_fill(~mask_all[idx], -1.0e9)
            loss = F.cross_entropy(logits, y_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pred = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * int(idx.numel())
            total_correct += int((pred == y_all[idx]).sum().item())
            total += int(idx.numel())

        val_acc = 0.0
        if len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                idx = torch.from_numpy(val_idx).to(device)
                logits = model(x_all[idx]).masked_fill(~mask_all[idx], -1.0e9)
                val_acc = float((torch.argmax(logits, dim=1) == y_all[idx]).float().mean().item())
        print(
            f"[selector-train] epoch={epoch:03d} "
            f"loss={total_loss / max(total, 1):.5f} train_acc={total_correct / max(total, 1):.3f} "
            f"val_acc={val_acc:.3f}",
            flush=True,
        )

    stats = {
        "samples": int(len(labels)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "input_dim": int(features.shape[1]),
    }
    return model, stats


def export_onnx(model: BufferSelectorMLP, output_path: Path, input_dim: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a buffer-index selector from the buffer-search oracle.")
    parser.add_argument("--submit-dir", type=Path, default=PROJECT_ROOT / "submit_buffer3_search")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-boxes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle-mode", choices=["one_step", "beam"], default="one_step")
    parser.add_argument("--oracle-top-k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "isaaclab_pallet" / "runs" / "buffer_selector")
    parser.add_argument("--install-to-submit", action="store_true")
    args = parser.parse_args()

    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, masks, labels = collect_oracle_samples(
        args.submit_dir,
        args.episodes,
        args.max_boxes,
        args.seed,
        args.oracle_mode,
        args.oracle_top_k,
    )
    if features.shape[1] != BUFFER_SELECTOR_INPUT_DIM:
        raise RuntimeError(f"unexpected selector input dim {features.shape[1]} != {BUFFER_SELECTOR_INPUT_DIM}")

    np.savez_compressed(
        args.output_dir / "buffer_selector_dataset.npz",
        features=features,
        masks=masks,
        labels=labels,
    )

    model, stats = train_selector(
        features,
        masks,
        labels,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.device,
    )
    torch.save(
        {"model": model.state_dict(), "stats": stats},
        args.output_dir / "buffer_selector.pt",
    )
    onnx_path = args.output_dir / "buffer_selector.onnx"
    export_onnx(model, onnx_path, BUFFER_SELECTOR_INPUT_DIM)

    metadata = {
        **stats,
        "episodes": args.episodes,
        "max_boxes": args.max_boxes,
        "seed": args.seed,
        "oracle_mode": args.oracle_mode,
        "oracle_top_k": args.oracle_top_k,
        "elapsed_s": round(time.time() - start, 3),
    }
    (args.output_dir / "buffer_selector_metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.install_to_submit:
        dst = args.submit_dir / "src" / "models" / "buffer_selector.onnx"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, dst)
        print(f"[selector-train] installed {dst}", flush=True)

    print(f"[selector-train] wrote {onnx_path}", flush=True)


if __name__ == "__main__":
    main()
