#!/usr/bin/env python3
"""Collect counterfactual candidate-value labels from development episodes.

True future boxes are passed only to :func:`_counterfactual_target`; feature
construction happens before that call and can see only the online state and
the current box.  One shard per episode keeps the large data set streamable.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from algorithm import AlgorithmConfig, PalletConfig, Palletizer  # noqa: E402
from algorithm import apply_stability_mask  # noqa: E402


PALLET_VOLUME = 1.2 * 1.0 * 1.25
LABEL_FUTURE_BOXES = 12


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _model_choice(candidates: list[dict[str, Any]], model_name: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            -float(candidate.get("candidate_metadata", {}).get("model_ranks", {}).get(model_name, -1.0)),
            candidate.get("placement_key", ()),
        ),
    )


def _first_valid_model_candidate(
    palletizer: Palletizer,
    box: dict[str, Any],
    base_packer,
    raw_sequence: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any] | None:
    """Mirror deployed single-policy order without scoring unused leaves."""
    size = [float(value) for value in box["size"]]
    try:
        observed_packer = copy.deepcopy(base_packer)
        observation = observed_packer.observe(size, palletizer._density(box, size))
        observation = observation.reshape(1, -1, 9).astype(np.float32)
        leaves = observation[0, palletizer._inh:palletizer._inh + palletizer._lnh]
        safe_leaves, _ = apply_stability_mask(
            leaves,
            observed_packer.space.boxes,
            size,
            float(box["mass"]),
            palletizer._container,
        )
        indices, metadata = palletizer._candidate_leaf_metadata(
            observation,
            safe_leaves,
            f"model:{model_name}",
        )
    except Exception:
        return None
    for leaf_rank, leaf_index in enumerate(indices[:max(20, palletizer._union_fallback_top_k_leaf)]):
        try:
            trial = copy.deepcopy(observed_packer)
            if not trial.place(safe_leaves[leaf_index, :6]):
                continue
            validated = palletizer._validate_packed_record(box, trial.packed[-1], raw_sequence)
        except Exception:
            continue
        if validated is None:
            continue
        output, raw_placed = validated
        return {
            "packer": trial,
            "raw_placed": raw_placed,
            "validated": output,
            "leaf_rank": leaf_rank,
            "leaf_index": leaf_index,
            "candidate_metadata": metadata[leaf_index],
        }
    return None


def _counterfactual_target(
    palletizer: Palletizer,
    root: dict[str, Any],
    raw_sequence: list[dict[str, Any]],
    future_boxes: list[dict[str, Any]],
    model_names: list[str],
) -> float:
    root_volume = float(np.prod(np.asarray(root["raw_placed"]["size"], dtype=np.float64)))
    best_volume = root_volume
    for model_name in model_names:
        packer = copy.deepcopy(root["packer"])
        branch_sequence = raw_sequence + [root["raw_placed"]]
        added_volume = root_volume
        for future_box in future_boxes[:LABEL_FUTURE_BOXES]:
            chosen = _first_valid_model_candidate(
                palletizer,
                future_box,
                packer,
                branch_sequence,
                model_name,
            )
            if chosen is None:
                break
            packer = chosen["packer"]
            branch_sequence.append(chosen["raw_placed"])
            added_volume += float(np.prod(np.asarray(chosen["raw_placed"]["size"], dtype=np.float64)))
        best_volume = max(best_volume, added_volume)
    return best_volume / PALLET_VOLUME


def _advance_state(palletizer: Palletizer, chosen: dict[str, Any]) -> None:
    palletizer._packer = chosen["packer"]
    palletizer._raw_sequence.append(chosen["raw_placed"])
    palletizer.sequence.append(copy.deepcopy(chosen["raw_placed"]))


def collect_episode(
    boxes: list[dict[str, Any]],
    behavior_model: str | None,
    state_stride: int,
    max_states: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    palletizer = Palletizer(PalletConfig(1.2, 1.0, 1.25), AlgorithmConfig(True, 0))
    palletizer._reset_state()
    if not palletizer._ensure_packer_for_current_run():
        raise RuntimeError("PCT policy could not be initialized")
    if behavior_model:
        palletizer._value_model_path = behavior_model
        # _ensure_policy may already have run; load the DAgger policy directly.
        import onnxruntime as ort
        path = Path(behavior_model)
        if not path.is_absolute():
            path = HERE / path
        palletizer._value_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        palletizer._value_input_name = palletizer._value_session.get_inputs()[0].name
        palletizer._value_output_name = palletizer._value_session.get_outputs()[0].name

    model_names = [Path(str(item["name"])).name for item in palletizer._canonical_policy_sessions()]
    fallback_name = palletizer._fallback_model_name
    features: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    collected_states = 0

    for index, box in enumerate(boxes[:130]):
        palletizer._observed_boxes.append(box)
        raw_sequence = list(palletizer._raw_sequence)
        candidates = palletizer._trial_candidates(
            box,
            palletizer._packer,
            raw_sequence,
            0,
            palletizer._union_top_k_per_model * max(1, len(model_names)),
            candidate_source="union",
        )
        if not candidates:
            break
        for candidate in candidates:
            candidate["base_packer"] = palletizer._packer

        should_collect = index % max(1, state_stride) == 0 and (max_states is None or collected_states < max_states)
        if should_collect:
            group = collected_states
            # Feature extraction is deliberately completed before actual future
            # boxes enter the label-only function below.
            state_features = [palletizer._candidate_value_features(candidate, raw_sequence) for candidate in candidates]
            future = boxes[index + 1:index + 1 + LABEL_FUTURE_BOXES]
            state_labels = [
                _counterfactual_target(palletizer, candidate, raw_sequence, future, model_names)
                for candidate in candidates
            ]
            features.extend(state_features)
            labels.extend(state_labels)
            groups.extend([group] * len(candidates))
            collected_states += 1

        chosen = None
        if behavior_model:
            scores = palletizer._score_candidates_with_value(candidates, raw_sequence)
            if scores is not None:
                chosen = candidates[int(np.argmax(scores))]
        if chosen is None:
            chosen = _first_valid_model_candidate(
                palletizer,
                box,
                palletizer._packer,
                raw_sequence,
                fallback_name,
            )
        if chosen is None:
            break
        _advance_state(palletizer, chosen)
        if max_states is not None and collected_states >= max_states:
            break

    if not features:
        feature_dim = (palletizer._inh + palletizer._lnh + 1) * 9 + 9 + 26
        return np.zeros((0, feature_dim), np.float16), np.zeros((0,), np.float32), np.zeros((0,), np.int32)
    return np.stack(features).astype(np.float16), np.asarray(labels, np.float32), np.asarray(groups, np.int32)


def _collect_shard(
    dataset: Path,
    output: Path,
    entry: dict[str, Any],
    behavior_model: str | None,
    state_stride: int,
    max_states: int | None,
) -> dict[str, Any]:
    shard = output / f"{entry['name']}.npz"
    if shard.is_file():
        try:
            payload = np.load(shard)
            x, y, group = payload["features"], payload["targets"], payload["groups"]
            if x.ndim == 2 and len(x) == len(y) == len(group):
                return {
                    "episode": entry["name"],
                    "split": entry["split"],
                    "family": entry["family"],
                    "file": shard.name,
                    "candidate_count": int(len(y)),
                    "state_count": int(len(np.unique(group))),
                    "feature_dim": int(x.shape[1]),
                    "resumed": True,
                }
        except Exception:
            pass
    x, y, group = collect_episode(
        _read_jsonl(dataset / entry["file"]),
        behavior_model,
        state_stride,
        max_states,
    )
    np.savez(shard, features=x, targets=y, groups=group)
    return {
        "episode": entry["name"],
        "split": entry["split"],
        "family": entry["family"],
        "file": shard.name,
        "candidate_count": int(len(y)),
        "state_count": int(len(np.unique(group))),
        "feature_dim": int(x.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--behavior-model", help="candidate_value.onnx for the second DAgger pass")
    parser.add_argument("--pass-name", default="initial")
    parser.add_argument("--state-stride", type=int, default=1)
    parser.add_argument("--max-states-per-episode", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--jobs", type=int, default=1, help="episode workers; each owns independent ONNX sessions")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("holdout_used") is not False:
        raise ValueError("collector accepts only a development manifest with holdout_used=false")
    entries = [entry for entry in manifest["sequences"] if args.split == "all" or entry["split"] == args.split]
    if args.max_episodes is not None:
        entries = entries[:args.max_episodes]
    args.output.mkdir(parents=True, exist_ok=True)
    shard_records = []
    feature_dim = None
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(
                _collect_shard,
                args.dataset.resolve(),
                args.output.resolve(),
                entry,
                args.behavior_model,
                args.state_stride,
                args.max_states_per_episode,
            ): entry
            for entry in entries
        }
        for completed_index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            feature_dim = record.pop("feature_dim")
            shard_records.append(record)
            print(f"[{completed_index}/{len(entries)}] {record['episode']} states={record['state_count']}", flush=True)
    shard_records.sort(key=lambda record: record["episode"])

    metadata = {
        "version": "candidate_value_labels_v1",
        "dagger_metadata_version": "dagger-v2",
        "pass_name": args.pass_name,
        "behavior_model": args.behavior_model,
        "feature_dim": feature_dim,
        "label": "max_three_policy_counterfactual_added_volume_fraction",
        "label_future_boxes": LABEL_FUTURE_BOXES,
        "future_boxes_in_features": False,
        "episode_split": True,
        "source_manifest_version": manifest["version"],
        "shards": shard_records,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
