#!/usr/bin/env python3
"""Generate the deterministic train/validation set for hybrid MPC.

This generator never reads or writes ``final_model_selection_v1``.  Sequence
files use the competition JSONL representation and every payload is protected
by a SHA-256 entry in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable


VERSION = "hybrid_mpc_dev_v1"
ROOT_SEED = 2026071602
BOX_COUNT = 250
COUNTS = {
    "train": {"continuous_iid": 200, "edge_geometry": 40, "small_dense": 24, "large_heavy": 24, "switch_5sku": 24},
    "validation": {"continuous_iid": 50, "edge_geometry": 10, "small_dense": 6, "large_heavy": 6, "switch_5sku": 6},
}
LIMITS = {
    "length_width_m": [0.170, 0.320],
    "height_m": [0.130, 0.260],
    "mass_kg": [0.500, 6.000],
    "rounding_decimals": 3,
}
SKUS = (
    (0.176, 0.284, 0.151, 1.150),
    (0.207, 0.259, 0.179, 2.275),
    (0.229, 0.198, 0.216, 3.450),
    (0.271, 0.241, 0.193, 4.700),
    (0.314, 0.302, 0.251, 5.800),
)


def _round(value: float) -> float:
    return round(value, 3)


def _seed(name: str) -> int:
    digest = hashlib.sha256(f"{VERSION}:{ROOT_SEED}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _uniform(rng: random.Random, low: float, high: float) -> float:
    return _round(rng.uniform(low, high))


def _iid(rng: random.Random) -> tuple[float, float, float, float]:
    return (_uniform(rng, .17, .32), _uniform(rng, .17, .32), _uniform(rng, .13, .26), _uniform(rng, .5, 6.0))


def _edge_value(rng: random.Random, low: float, high: float) -> float:
    return _uniform(rng, low, low + .02) if rng.random() < .5 else _uniform(rng, high - .02, high)


def _edge(rng: random.Random) -> tuple[float, float, float, float]:
    return (_edge_value(rng, .17, .32), _edge_value(rng, .17, .32), _edge_value(rng, .13, .26), _uniform(rng, .5, 6.0))


def _small(rng: random.Random) -> tuple[float, float, float, float]:
    return (_uniform(rng, .17, .23), _uniform(rng, .17, .23), _uniform(rng, .13, .19), _uniform(rng, 4.0, 6.0))


def _large(rng: random.Random) -> tuple[float, float, float, float]:
    return (_uniform(rng, .26, .32), _uniform(rng, .26, .32), _uniform(rng, .20, .26), _uniform(rng, 4.0, 6.0))


GENERATORS: dict[str, Callable[[random.Random], tuple[float, float, float, float]]] = {
    "continuous_iid": _iid,
    "edge_geometry": _edge,
    "small_dense": _small,
    "large_heavy": _large,
}


def _boxes(name: str, family: str) -> list[dict]:
    rng = random.Random(_seed(name))
    if family == "switch_5sku":
        values = [sku for sku in SKUS for _ in range(BOX_COUNT // len(SKUS))]
        rng.shuffle(values)
    else:
        values = [GENERATORS[family](rng) for _ in range(BOX_COUNT)]
    return [
        {"step": index, "id": index, "size": list(value[:3]), "mass": value[3]}
        for index, value in enumerate(values)
    ]


def _payload(rows: list[dict]) -> str:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(output: Path, overwrite: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for split, families in COUNTS.items():
        for family, count in families.items():
            for index in range(count):
                name = f"{split}_{family}_{index:03d}"
                path = output / f"{name}.json"
                path.write_text(_payload(_boxes(name, family)), encoding="utf-8")
                entries.append({
                    "name": name,
                    "file": path.name,
                    "split": split,
                    "family": family,
                    "seed": _seed(name),
                    "box_count": BOX_COUNT,
                    "sha256": _sha256(path),
                })
    manifest = {
        "version": VERSION,
        "root_seed": ROOT_SEED,
        "format": "competition_jsonl_one_object_per_line",
        "box_count_per_sequence": BOX_COUNT,
        "limits": LIMITS,
        "counts": COUNTS,
        "switch_5sku": [{"size": list(value[:3]), "mass": value[3]} for value in SKUS],
        "holdout_used": False,
        "sequences": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("holdout_used") is not False:
        raise ValueError("wrong or unsafe manifest")
    expected = {(split, family): count for split, families in COUNTS.items() for family, count in families.items()}
    actual = Counter((entry["split"], entry["family"]) for entry in manifest["sequences"])
    if actual != expected:
        raise ValueError(f"count mismatch: {actual}")
    for entry in manifest["sequences"]:
        path = output / entry["file"]
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != BOX_COUNT or any(row["step"] != index or row["id"] != index for index, row in enumerate(rows)):
            raise ValueError(f"invalid sequence: {path}")
    return {"sequence_count": len(manifest["sequences"]), "sha256_verified": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        generate(args.output, args.overwrite)
    print(json.dumps(validate(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
