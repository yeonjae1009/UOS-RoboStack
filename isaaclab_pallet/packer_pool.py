"""Parallel CPU packer pool (issue ①: Stage-3 throughput).

The porting guide's "real wall" is that EMS leaf generation + placement
(`packer.observe`/`packer.place`, pure numpy) runs once per env, serially. The
GPU physics is already batched and nearly free, so at 256-1024 envs the serial
CPU packing dominates and the GPU idles.

This module isolates ALL per-env CPU work behind a small interface with two
interchangeable backends:

  * SerialPackerPool   — packers live in the main process (current behaviour).
  * ParallelPackerPool — persistent worker processes, each owning a fixed shard
                         of envs' packers, doing observe+select+place+reward and
                         returning only picklable results. The main process keeps
                         the GPU physics.

Both expose the SAME interface, so the env can switch with one config flag. The
parallel backend is proven bit-identical to the serial one in
scripts/test_packer_pool.py (runs without Isaac).

A worker computes a full CPU step (`_packer_step`) so the main process never
needs to touch `packer.space.boxes` — the reward and the packer-resolved spawn
pose come back as plain data.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

try:  # works both as a package submodule (env) and as a top-level module (tests/workers)
    from . import pct_reward
except ImportError:  # pragma: no cover
    import pct_reward


@dataclass
class PackerConfig:
    pallet_size: tuple
    size_minimum: float
    internal_node_holder: int
    leaf_node_holder: int
    setting: int
    density_max: float
    scales: pct_reward.RewardScales


def _make_packer(cfg: PackerConfig):
    # Imported here so worker processes import the driver independently.
    from src.pct.packer import Packer

    packer = Packer(
        container_size=list(cfg.pallet_size),
        size_minimum=float(cfg.size_minimum),
        internal_node_holder=cfg.internal_node_holder,
        leaf_node_holder=cfg.leaf_node_holder,
        setting=cfg.setting,
    )
    packer.reset()
    return packer


def _observe(packer, box, cfg: PackerConfig) -> np.ndarray:
    density = pct_reward.density_for_box(box, cfg.setting, cfg.density_max)
    node_count = cfg.internal_node_holder + cfg.leaf_node_holder + 1
    return packer.observe(box["size"], density=density).reshape(node_count, 9).astype(np.float32)


def _packer_step(packer, box, action_idx: int, cfg: PackerConfig) -> dict:
    """Full per-env CPU step: observe -> select leaf -> place -> reward.

    Returns a picklable dict. ``status`` is one of {invalid, place_failed, ok}.
    On ok, ``packed`` is the packer-resolved [x,y,z,lx,ly,lz,bin] used by the
    main process to spawn the box at its true resting pose (issue ④ fix) and as
    the drift reference.
    """
    density = pct_reward.density_for_box(box, cfg.setting, cfg.density_max)
    node_count = cfg.internal_node_holder + cfg.leaf_node_holder + 1
    obs = packer.observe(box["size"], density=density).reshape(node_count, 9).astype(np.float32)
    leaf_nodes = obs[cfg.internal_node_holder:cfg.internal_node_holder + cfg.leaf_node_holder]
    valid_count = int((leaf_nodes[:, 8] > 0.5).sum())

    leaf = pct_reward.select_leaf(leaf_nodes, action_idx)
    if leaf is None:
        return {"status": "invalid", "obs": obs, "valid_count": valid_count}

    _, _, rotation = pct_reward.leaf_to_center_size_rotation(leaf, box["size"])
    before = pct_reward.layout_metrics(packer.space.boxes, cfg.pallet_size)
    box_ratio = float(np.prod([float(v) for v in box["size"]]) / np.prod(cfg.pallet_size))
    if not packer.place(leaf[:6]):
        return {"status": "place_failed", "obs": obs, "valid_count": valid_count}

    packed_box = packer.space.boxes[-1]
    after = pct_reward.layout_metrics(packer.space.boxes, cfg.pallet_size)
    reward, _terms = pct_reward.compute_online3dbpp_reward(box_ratio, packed_box, before, after, cfg.scales)
    return {
        "status": "ok",
        "obs": obs,
        "valid_count": valid_count,
        "reward": float(reward),
        "rotation": int(rotation),
        "packed": [float(v) for v in packer.packed[-1]],
        "packed_all": [[float(v) for v in rec] for rec in packer.packed],
        "ratio": float(packer.get_ratio()),
    }


# --------------------------------------------------------------------------- #
# Serial backend
# --------------------------------------------------------------------------- #
class SerialPackerPool:
    def __init__(self, num_envs: int, cfg: PackerConfig):
        self.num_envs = num_envs
        self.cfg = cfg
        self._packers = [_make_packer(cfg) for _ in range(num_envs)]

    def reset(self, env_ids):
        for env_id in env_ids:
            self._packers[env_id] = _make_packer(self.cfg)

    def observe(self, boxes: dict) -> dict:
        return {eid: _observe(self._packers[eid], box, self.cfg) for eid, box in boxes.items()}

    def step(self, requests: dict) -> dict:
        return {eid: _packer_step(self._packers[eid], box, aidx, self.cfg)
                for eid, (box, aidx) in requests.items()}

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Parallel backend (persistent sharded workers)
# --------------------------------------------------------------------------- #
def _worker_loop(conn, env_ids, cfg: PackerConfig):
    packers = {eid: _make_packer(cfg) for eid in env_ids}
    try:
        while True:
            cmd = conn.recv()
            op = cmd[0]
            if op == "close":
                break
            if op == "reset":
                for eid in cmd[1]:
                    if eid in packers:
                        packers[eid] = _make_packer(cfg)
                conn.send(True)
            elif op == "observe":
                boxes = cmd[1]
                conn.send({eid: _observe(packers[eid], box, cfg) for eid, box in boxes.items()})
            elif op == "step":
                reqs = cmd[1]
                conn.send({eid: _packer_step(packers[eid], box, aidx, cfg)
                           for eid, (box, aidx) in reqs.items()})
            else:
                conn.send(RuntimeError(f"unknown op {op}"))
    finally:
        conn.close()


class ParallelPackerPool:
    def __init__(self, num_envs: int, cfg: PackerConfig, num_workers: int):
        self.num_envs = num_envs
        self.cfg = cfg
        self.num_workers = max(1, min(num_workers, num_envs))
        ctx = mp.get_context("spawn")
        shards = [list(range(i, num_envs, self.num_workers)) for i in range(self.num_workers)]
        self._shards = [s for s in shards if s]
        self._env_to_worker = {}
        self._conns = []
        self._procs = []
        for w, shard in enumerate(self._shards):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_loop, args=(child, shard, cfg), daemon=True)
            proc.start()
            self._conns.append(parent)
            self._procs.append(proc)
            for eid in shard:
                self._env_to_worker[eid] = w

    def _dispatch(self, op: str, per_env: dict) -> dict:
        # Partition the request by owning worker, send, then gather.
        buckets = [dict() for _ in self._shards]
        for eid, payload in per_env.items():
            buckets[self._env_to_worker[eid]][eid] = payload
        for w, bucket in enumerate(buckets):
            self._conns[w].send((op, bucket))
        out = {}
        for w in range(len(self._shards)):
            out.update(self._conns[w].recv())
        return out

    def reset(self, env_ids):
        by_worker = [list() for _ in self._shards]
        for eid in env_ids:
            by_worker[self._env_to_worker[eid]].append(eid)
        for w, ids in enumerate(by_worker):
            self._conns[w].send(("reset", ids))
        for w in range(len(self._shards)):
            self._conns[w].recv()

    def observe(self, boxes: dict) -> dict:
        return self._dispatch("observe", boxes)

    def step(self, requests: dict) -> dict:
        return self._dispatch("step", requests)

    def close(self):
        for conn in self._conns:
            try:
                conn.send(("close",))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._procs:
            proc.join(timeout=2.0)


def make_packer_pool(num_envs: int, cfg: PackerConfig, num_workers: int = 0):
    """num_workers <= 0 -> serial (default). >0 -> parallel sharded workers."""
    if num_workers and num_workers > 0:
        return ParallelPackerPool(num_envs, cfg, num_workers)
    return SerialPackerPool(num_envs, cfg)
