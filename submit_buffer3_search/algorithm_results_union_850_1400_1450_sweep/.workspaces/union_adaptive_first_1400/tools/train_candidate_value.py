#!/usr/bin/env python3
"""Train and export the regret-ranked candidate value network.

PyTorch is a development-only dependency.  The exported submission continues
to require only NumPy and ONNX Runtime.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as functional


class CandidateValueNet(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _label_roots(paths: list[Path]) -> tuple[int, list[tuple[Path, dict]], list[tuple[Path, dict]], list[dict]]:
    train, validation, metadata_rows = [], [], []
    feature_dim = None
    for root in paths:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("future_boxes_in_features") is not False or metadata.get("episode_split") is not True:
            raise ValueError(f"unsafe label metadata: {root}")
        if feature_dim is None:
            feature_dim = int(metadata["feature_dim"])
        elif feature_dim != int(metadata["feature_dim"]):
            raise ValueError("feature dimension mismatch between DAgger passes")
        metadata_rows.append(metadata)
        for shard in metadata["shards"]:
            item = (root / shard["file"], shard)
            (train if shard["split"] == "train" else validation).append(item)
    if feature_dim is None or not train or not validation:
        raise ValueError("both episode-level train and validation shards are required")
    return feature_dim, train, validation, metadata_rows


def _groups(path: Path, shuffle: bool, rng: random.Random):
    payload = np.load(path)
    x = payload["features"]
    y = payload["targets"]
    groups = payload["groups"]
    ids = np.unique(groups).tolist()
    if shuffle:
        rng.shuffle(ids)
    for group_id in ids:
        selected = groups == group_id
        if int(selected.sum()) >= 2:
            yield x[selected].astype(np.float32), y[selected].astype(np.float32)


def _ranking_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    best_index = int(torch.argmax(targets).item())
    regret = targets[best_index] - targets
    keep = regret > 1e-8
    if not bool(torch.any(keep)):
        return scores.sum() * 0.0
    margins = scores[best_index] - scores[keep]
    weights = regret[keep] / torch.clamp(regret[keep].mean(), min=1e-8)
    return (functional.softplus(-margins) * weights).mean()


@torch.no_grad()
def _validate(model: nn.Module, shards: list[tuple[Path, dict]], device: torch.device) -> dict[str, float]:
    model.eval()
    regrets, accuracies = [], []
    for path, _ in shards:
        for features, targets in _groups(path, False, random.Random(0)):
            scores = model(torch.from_numpy(features).to(device)).cpu().numpy()
            predicted, target = int(np.argmax(scores)), int(np.argmax(targets))
            regrets.append(float(targets[target] - targets[predicted]))
            accuracies.append(float(predicted == target))
    return {
        "mean_regret": float(np.mean(regrets)) if regrets else float("inf"),
        "argmax_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="initial and DAgger-2 label roots")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--states-per-update", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    feature_dim, train_shards, validation_shards, source_metadata = _label_roots(args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateValueNet(feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    rng = random.Random(args.seed)
    history = []

    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train_shards)
        optimizer.zero_grad(set_to_none=True)
        pending, losses = 0, []
        for path, _ in train_shards:
            for features, targets in _groups(path, True, rng):
                scores = model(torch.from_numpy(features).to(device))
                loss = _ranking_loss(scores, torch.from_numpy(targets).to(device))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite ranking loss")
                (loss / args.states_per_update).backward()
                pending += 1
                losses.append(float(loss.detach().cpu()))
                if pending == args.states_per_update:
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
        if pending:
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = _validate(model, validation_shards, device)
        row = {"epoch": epoch + 1, "train_ranking_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    example = torch.zeros((2, feature_dim), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["candidate_features"],
        output_names=["candidate_utility"],
        dynamic_axes={"candidate_features": {0: "candidate_count"}, "candidate_utility": {0: "candidate_count"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Required parity gate: numeric output and candidate argmax must agree.
    sample_path = validation_shards[0][0]
    sample = np.load(sample_path)["features"][:32].astype(np.float32)
    with torch.no_grad():
        torch_scores = model(torch.from_numpy(sample)).numpy().reshape(-1)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_scores = session.run(None, {session.get_inputs()[0].name: sample})[0].reshape(-1)
    max_error = float(np.max(np.abs(torch_scores - onnx_scores)))
    argmax_match = int(np.argmax(torch_scores)) == int(np.argmax(onnx_scores))
    if max_error > 1e-4 or not argmax_match:
        raise RuntimeError(f"ONNX parity failed: max_error={max_error}, argmax_match={argmax_match}")

    metadata = {
        "version": "candidate_value_v1",
        "dagger_metadata_version": "dagger-v2",
        "feature_dim": feature_dim,
        "input_shape": ["candidate_count", feature_dim],
        "output_shape": ["candidate_count"],
        "loss": "regret_weighted_best_vs_candidate_ranking",
        "episode_train_validation_split": True,
        "runtime_dependencies": ["numpy", "onnxruntime"],
        "sources": [{"pass_name": row["pass_name"], "behavior_model": row["behavior_model"]} for row in source_metadata],
        "history": history,
        "onnx_parity": {"max_abs_error": max_error, "argmax_match": argmax_match},
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
