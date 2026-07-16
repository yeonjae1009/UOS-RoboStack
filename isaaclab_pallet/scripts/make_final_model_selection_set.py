#!/usr/bin/env python3
"""Create and validate the locked ``final_model_selection_v1`` data set.

The data set is intentionally separate from every training/evaluation set.  It
contains 72 ranking sequences (50 IID + 22 stress) and six 5-SKU hybrid-gate
sequences.  Files use the competition's JSONL convention despite their ``.json``
suffix: one ``{step, id, size, mass}`` object per line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "final_model_selection_v1"
VERSION = "final_model_selection_v1"
ROOT_SEED = 20260716
BOX_COUNT = 250
LIMITS = {
    "length_width_m": [0.170, 0.320],
    "height_m": [0.130, 0.260],
    "mass_kg": [0.500, 6.000],
    "rounding_decimals": 3,
}

# These are deliberately different from both public five-SKU dimensions.
SWITCH_SKUS: tuple[tuple[float, float, float, float], ...] = (
    (0.181, 0.291, 0.146, 1.275),
    (0.214, 0.267, 0.171, 2.350),
    (0.236, 0.193, 0.221, 3.625),
    (0.278, 0.247, 0.186, 4.825),
    (0.307, 0.309, 0.244, 5.575),
)


def _round(value: float) -> float:
    return round(value, int(LIMITS["rounding_decimals"]))


def _seed_for(name: str) -> int:
    digest = hashlib.sha256(f"{VERSION}:{ROOT_SEED}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return _round(rng.uniform(lo, hi))


def _iid_box(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        _uniform(rng, 0.170, 0.320),
        _uniform(rng, 0.170, 0.320),
        _uniform(rng, 0.130, 0.260),
        _uniform(rng, 0.500, 6.000),
    )


def _edge_value(rng: random.Random, lo: float, hi: float) -> float:
    return _uniform(rng, lo, lo + 0.020) if rng.random() < 0.5 else _uniform(rng, hi - 0.020, hi)


def _edge_box(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        _edge_value(rng, 0.170, 0.320),
        _edge_value(rng, 0.170, 0.320),
        _edge_value(rng, 0.130, 0.260),
        _uniform(rng, 0.500, 6.000),
    )


def _small_dense_box(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        _uniform(rng, 0.170, 0.230),
        _uniform(rng, 0.170, 0.230),
        _uniform(rng, 0.130, 0.190),
        _uniform(rng, 4.000, 6.000),
    )


def _large_heavy_box(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        _uniform(rng, 0.260, 0.320),
        _uniform(rng, 0.260, 0.320),
        _uniform(rng, 0.200, 0.260),
        _uniform(rng, 4.000, 6.000),
    )


def _sequence_names() -> dict[str, list[str]]:
    return {
        "continuous_iid": [f"continuous_iid_{i:03d}" for i in range(50)],
        "edge_geometry": [f"edge_geometry_{i:03d}" for i in range(10)],
        "small_dense": [f"small_dense_{i:03d}" for i in range(6)],
        "large_heavy": [f"large_heavy_{i:03d}" for i in range(6)],
        "switch_5sku": [f"switch_5sku_{i:03d}" for i in range(6)],
    }


def _make_boxes(name: str, family: str) -> list[dict[str, Any]]:
    rng = random.Random(_seed_for(name))
    if family == "continuous_iid":
        source = lambda: _iid_box(rng)
        dims = [source() for _ in range(BOX_COUNT)]
    elif family == "edge_geometry":
        dims = [_edge_box(rng) for _ in range(BOX_COUNT)]
    elif family == "small_dense":
        dims = [_small_dense_box(rng) for _ in range(BOX_COUNT)]
    elif family == "large_heavy":
        dims = [_large_heavy_box(rng) for _ in range(BOX_COUNT)]
    elif family == "switch_5sku":
        # Each SKU occurs exactly 50 times.  Shuffling preserves a genuine
        # arrival sequence and prevents an accidental five-block heuristic.
        dims = [sku for sku in SWITCH_SKUS for _ in range(50)]
        rng.shuffle(dims)
    else:
        raise ValueError(f"unknown family: {family}")
    return [
        {"step": index, "id": index, "size": list(dim[:3]), "mass": dim[3]}
        for index, dim in enumerate(dims)
    ]


def _jsonl(boxes: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(box, separators=(",", ":"), ensure_ascii=False) + "\n" for box in boxes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_skus() -> set[tuple[float, float, float]]:
    public_dir = PROJECT_ROOT / "palletizing_simulator" / "box_sequence"
    skus: set[tuple[float, float, float]] = set()
    for path in public_dir.glob("box_sequence_*.json"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                skus.add(tuple(float(v) for v in row["size"]))
    return skus


def generate(out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists; use --overwrite only for deliberate regeneration")
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = _sequence_names()
    entries: list[dict[str, Any]] = []
    for family, names in groups.items():
        for name in names:
            path = out_dir / f"{name}.json"
            path.write_text(_jsonl(_make_boxes(name, family)), encoding="utf-8")
            entries.append({
                "name": name,
                "file": path.name,
                "family": family,
                "split": "hybrid_gate" if family == "switch_5sku" else "selection",
                "seed": _seed_for(name),
                "sha256": _sha256(path),
                "box_count": BOX_COUNT,
            })

    public_skus = _public_skus()
    switch_skus = [list(sku[:3]) for sku in SWITCH_SKUS]
    overlap = sorted(set(tuple(s) for s in switch_skus) & public_skus)
    if overlap:
        raise RuntimeError(f"switch SKU overlaps public dimensions: {overlap}")
    manifest = {
        "version": VERSION,
        "root_seed": ROOT_SEED,
        "format": "competition_jsonl_one_object_per_line",
        "box_count_per_sequence": BOX_COUNT,
        "limits": LIMITS,
        "ranking": {
            "selection_sequence_count": 72,
            "iid_family": "continuous_iid",
            "iid_weight": 0.7,
            "stress_families": ["edge_geometry", "small_dense", "large_heavy"],
            "stress_weight": 0.3,
        },
        "hybrid_gate": {
            "family": "switch_5sku",
            "sequence_count": 6,
            "skus": [{"size": list(sku[:3]), "mass": sku[3]} for sku in SWITCH_SKUS],
            "public_sku_overlap": False,
        },
        "sequences": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _is_rounded(value: float) -> bool:
    return abs(value * 1000.0 - round(value * 1000.0)) < 1e-7


def validate(out_dir: Path, check_regeneration: bool = False) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError(f"unexpected dataset version: {manifest.get('version')}")
    entries = manifest.get("sequences", [])
    if len(entries) != 78:
        raise ValueError(f"expected 78 files, got {len(entries)}")
    selection = [entry for entry in entries if entry["split"] == "selection"]
    gate = [entry for entry in entries if entry["split"] == "hybrid_gate"]
    if len(selection) != 72 or len(gate) != 6:
        raise ValueError(f"expected 72 selection + 6 hybrid gate, got {len(selection)} + {len(gate)}")

    expected_families = {"continuous_iid": 50, "edge_geometry": 10, "small_dense": 6, "large_heavy": 6, "switch_5sku": 6}
    if Counter(entry["family"] for entry in entries) != expected_families:
        raise ValueError("unexpected sequence family counts")

    for entry in entries:
        path = out_dir / entry["file"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != BOX_COUNT:
            raise ValueError(f"{path}: expected {BOX_COUNT} boxes")
        for index, box in enumerate(rows):
            if box.get("step") != index or box.get("id") != index:
                raise ValueError(f"{path}: non-contiguous step/id at row {index}")
            size = box.get("size", [])
            mass = box.get("mass")
            if len(size) != 3 or not all(isinstance(v, (int, float)) for v in [*size, mass]):
                raise ValueError(f"{path}: malformed box at row {index}")
            if not (0.170 <= size[0] <= 0.320 and 0.170 <= size[1] <= 0.320 and 0.130 <= size[2] <= 0.260 and 0.500 <= mass <= 6.000):
                raise ValueError(f"{path}: out-of-range box at row {index}")
            if not all(_is_rounded(float(v)) for v in [*size, mass]):
                raise ValueError(f"{path}: unrounded value at row {index}")
        if entry["family"] == "switch_5sku":
            counts = Counter((tuple(row["size"]), row["mass"]) for row in rows)
            if len(counts) != 5 or set(counts.values()) != {50}:
                raise ValueError(f"{path}: switch sequence must contain five SKUs x 50")

    if check_regeneration:
        with tempfile.TemporaryDirectory(prefix="final_model_selection_") as tmp:
            regenerated = generate(Path(tmp) / VERSION)
            generated_hashes = {entry["file"]: entry["sha256"] for entry in regenerated["sequences"]}
            locked_hashes = {entry["file"]: entry["sha256"] for entry in entries}
            if generated_hashes != locked_hashes:
                raise ValueError("manifest regeneration hash mismatch")
    return {"selection_sequences": len(selection), "hybrid_gate_sequences": len(gate), "regeneration_checked": check_regeneration}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or validate the locked final model selection data set.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--validate", action="store_true", help="Validate an existing manifest/data set instead of creating it.")
    parser.add_argument("--check-regeneration", action="store_true", help="Also regenerate in a temporary directory and compare hashes.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.validate:
        result = validate(args.out_dir, check_regeneration=args.check_regeneration)
        print(json.dumps({"valid": True, **result}, indent=2))
    else:
        manifest = generate(args.out_dir, overwrite=args.overwrite)
        result = validate(args.out_dir, check_regeneration=args.check_regeneration)
        print(f"created={args.out_dir} files={len(manifest['sequences'])} {json.dumps(result)}")


if __name__ == "__main__":
    main()
