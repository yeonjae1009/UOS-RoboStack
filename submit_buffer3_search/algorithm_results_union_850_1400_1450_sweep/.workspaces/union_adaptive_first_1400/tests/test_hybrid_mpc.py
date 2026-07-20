from __future__ import annotations

import json
import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import algorithm as algorithm_module
from algorithm import AlgorithmConfig, PalletConfig, Palletizer


PALLET = PalletConfig(1.2, 1.0, 1.25)


class FakePolicySession:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).reshape(1, -1)

    def run(self, _outputs, _inputs):
        return [self.values.copy()]


class HybridMpcTests(unittest.TestCase):
    def _metadata_palletizer(self, policies):
        palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
        palletizer._lnh = 5
        palletizer._policy_sessions = [
            {"name": name, "path": name, "session": FakePolicySession(values), "input_name": "obs"}
            for name, values in policies
        ]
        return palletizer

    def test_union_contains_every_model_and_is_order_independent(self):
        policies = [
            ("candidate-001400.onnx", [0.1, 0.9, 0.3, 0.2, 0.0]),
            ("candidate-001450.onnx", [0.8, 0.1, 0.7, 0.2, 0.0]),
            ("pct_model_bestonnx.onnx", [0.2, 0.3, 0.4, 0.9, 0.0]),
        ]
        observation = np.zeros((1, 45), dtype=np.float32)
        leaves = np.ones((5, 9), dtype=np.float32)
        first = self._metadata_palletizer(policies)
        second = self._metadata_palletizer(list(reversed(policies)))
        indices_a, metadata_a = first._candidate_leaf_metadata(observation, leaves, "union")
        indices_b, metadata_b = second._candidate_leaf_metadata(observation, leaves, "union")
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(metadata_a, metadata_b)
        proposed = {name for item in metadata_a.values() for name in item["proposal_models"]}
        self.assertEqual(proposed, {name for name, _ in policies})

    def test_fallback_uses_1400_ranking_not_yaml_order(self):
        policies = [
            ("candidate-001450.onnx", [0.9, 0.1, 0.2, 0.3, 0.0]),
            ("candidate-001400.onnx", [0.1, 0.2, 0.95, 0.3, 0.0]),
            ("pct_model_bestonnx.onnx", [0.2, 0.8, 0.1, 0.3, 0.0]),
        ]
        palletizer = self._metadata_palletizer(policies)
        leaves = np.ones((5, 9), dtype=np.float32)
        indices, metadata = palletizer._candidate_leaf_metadata(np.zeros((1, 45), np.float32), leaves, "fallback")
        self.assertEqual(indices[0], 2)
        self.assertEqual(metadata[2]["proposal_models"], ("candidate-001400.onnx",))

    def test_identical_placement_candidates_are_deduplicated(self):
        class FakePacker:
            def __init__(self):
                self.packed = []

            def place(self, _leaf):
                self.packed.append([.2, .2, .2, 0, 0, 0, 0])
                return True

        palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
        palletizer._score_trial = lambda *_args, **_kwargs: 1.0
        box = {"step": 0, "id": 0, "size": [.2, .2, .2], "mass": 1.0}
        leaves = np.zeros((2, 9), np.float32)
        leaves[:, :6] = [0, 0, 0, .2, .2, 1.25]
        leaves[:, 8] = 1
        metadata = {
            0: {"proposal_mask": 1, "proposal_models": ("a",), "model_ranks": {"a": 1.0}, "model_probabilities": {"a": .5}},
            1: {"proposal_mask": 2, "proposal_models": ("b",), "model_ranks": {"b": 1.0}, "model_probabilities": {"b": .5}},
        }
        candidates = palletizer._build_trial_candidates(
            box, FakePacker(), FakePacker(), leaves, [], 0, [0, 1], metadata, 2, 0.0,
            np.zeros((1, 18), np.float32),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_metadata"]["proposal_mask"], 3)
        self.assertEqual(candidates[0]["candidate_metadata"]["proposal_models"], ("a", "b"))

    def test_buffer_zero_never_requests_a_future_item(self):
        class SpyBuffer:
            instances = []

            def __init__(self, size):
                self.size, self.items, self.index, self.peeked = size, [], 0, []
                self.__class__.instances.append(self)

            def reset(self, boxes):
                self.items = boxes

            def has_pending(self):
                return self.index < len(self.items)

            def peek_next(self):
                self.peeked.append(self.index)
                return self.items[self.index]

            def get_buffer(self):
                raise AssertionError("buffer=0 inference requested future buffer contents")

            def pop_next(self):
                self.index += 1

        class OnlinePalletizer(Palletizer):
            def _find_position(self, box):
                return (0.2 * int(box["step"]), 0.0, 0.0, (0.1, 0.1, 0.1), 0)

        boxes = [{"step": index, "id": index, "size": [.1, .1, .1], "mass": 1.0} for index in range(3)]
        with mock.patch.object(algorithm_module, "BufferManager", SpyBuffer):
            result = OnlinePalletizer(PALLET, AlgorithmConfig(True, 0)).run(boxes)
        self.assertEqual([item["id"] for item in result["sequence"]], [0, 1, 2])
        self.assertEqual(SpyBuffer.instances[0].peeked, [0, 1, 2])

    def test_timeout_and_search_failure_commit_1400_fallback(self):
        class FallbackPalletizer(Palletizer):
            def _ensure_packer_for_current_run(self):
                self._packer = getattr(self, "_packer", object())
                return True

            def _plan_hybrid_mpc(self, _box):
                self.plan_called = True
                return None

            def _fallback_single_candidate(self, box, base_packer=None, raw_sequence=None):
                self.fallback_called = True
                raw = {"step": box["step"], "id": box["id"], "size": box["size"], "mass": box["mass"], "position": [.1, .1, .1], "rotation": 0}
                return {"packer": "fallback-1400", "raw_placed": raw, "validated": (0.0, 0.0, 0.0, tuple(box["size"]), 0)}

        box = {"step": 0, "id": 7, "size": [.2, .2, .2], "mass": 1.0}
        for deadline in (time.monotonic() - 1.0, time.monotonic() + 10.0):
            palletizer = FallbackPalletizer(PALLET, AlgorithmConfig(True, 0))
            palletizer._search_enabled = True
            palletizer._candidate_union_enabled = True
            palletizer._candidate_union_mode = "mpc_value"
            palletizer._search_deadline = deadline
            output = palletizer._find_position(box)
            self.assertIsNotNone(output)
            self.assertTrue(palletizer.fallback_called)
            self.assertEqual(palletizer._packer, "fallback-1400")

    def test_onnx_scorer_parity_and_exception_degrades_to_rule(self):
        weights = np.asarray([[1.0], [2.0], [-1.0]], dtype=np.float32)
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["candidate_features", "weights"], ["candidate_utility"])],
            "linear_value",
            [helper.make_tensor_value_info("candidate_features", TensorProto.FLOAT, [None, 3])],
            [helper.make_tensor_value_info("candidate_utility", TensorProto.FLOAT, [None, 1])],
            [numpy_helper.from_array(weights, "weights")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = min(model.ir_version, 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "value.onnx"
            onnx.save(model, path)
            palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
            palletizer._value_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            palletizer._value_input_name = palletizer._value_session.get_inputs()[0].name
            palletizer._value_output_name = palletizer._value_session.get_outputs()[0].name
            rows = [np.asarray([1, 0, 0], np.float32), np.asarray([0, 2, 0], np.float32)]
            palletizer._candidate_value_features = lambda candidate, _raw: rows[candidate["index"]]
            scores = palletizer._score_candidates_with_value([{"index": 0}, {"index": 1}], [])
            expected = np.stack(rows) @ weights
            np.testing.assert_allclose(scores, expected.reshape(-1), atol=1e-6)
            self.assertEqual(int(np.argmax(scores)), int(np.argmax(expected)))

            class BrokenSession:
                def run(self, *_args, **_kwargs):
                    raise RuntimeError("simulated ONNX failure")

            palletizer._value_session = BrokenSession()
            palletizer._value_input_name = "features"
            self.assertIsNone(palletizer._score_candidates_with_value([{"index": 0}], []))
            self.assertIsNone(palletizer._value_session)

    def test_same_seed_produces_identical_future_samples(self):
        box = {"step": 4, "id": 9, "size": [.2, .21, .15], "mass": 2.0}
        first = Palletizer(PALLET, AlgorithmConfig(True, 0))
        second = Palletizer(PALLET, AlgorithmConfig(True, 0))
        first._observed_boxes = second._observed_boxes = [box] * 8
        payload_a = first._sample_pseudo_future_boxes(box, 3, 4)
        payload_b = second._sample_pseudo_future_boxes(box, 3, 4)
        self.assertEqual(json.dumps(payload_a, sort_keys=True), json.dumps(payload_b, sort_keys=True))

    def test_same_seed_produces_identical_final_json(self):
        boxes = [
            {"step": 0, "id": 0, "size": [.21, .19, .15], "mass": 1.2},
            {"step": 1, "id": 1, "size": [.24, .18, .17], "mass": 2.1},
        ]
        results = []
        for _ in range(2):
            palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
            palletizer._search_enabled = True
            palletizer._candidate_union_enabled = True
            palletizer._candidate_union_mode = "mpc_rule"
            palletizer._mpc_root_candidates = 1
            palletizer._mpc_horizon = 1
            palletizer._mpc_samples = 1
            results.append(palletizer.run(copy.deepcopy(boxes)))
        self.assertEqual(json.dumps(results[0], sort_keys=True), json.dumps(results[1], sort_keys=True))


if __name__ == "__main__":
    unittest.main()
