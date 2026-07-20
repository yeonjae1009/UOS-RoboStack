from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import algorithm as algorithm_module
from algorithm import AlgorithmConfig, PalletConfig, Palletizer


PALLET = PalletConfig(1.2, 1.0, 1.25)


class FakePolicySession:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).reshape(1, -1)

    def run(self, _outputs, _inputs):
        return [self.values.copy()]


class CandidateUnionTests(unittest.TestCase):
    def _metadata_palletizer(self, policies):
        palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
        palletizer._lnh = 5
        palletizer._policy_sessions = [
            {
                "name": name,
                "path": name,
                "session": FakePolicySession(values),
                "input_name": "obs",
            }
            for name, values in policies
        ]
        return palletizer

    def test_union_contains_every_model_and_is_order_independent(self):
        policies = [
            ("candidate-001400.onnx", [0.1, 0.9, 0.3, 0.2, 0.0]),
            ("candidate-001450.onnx", [0.8, 0.1, 0.7, 0.2, 0.0]),
            ("pct_model.onnx", [0.2, 0.3, 0.4, 0.9, 0.0]),
        ]
        observation = np.zeros((1, 45), dtype=np.float32)
        leaves = np.ones((5, 9), dtype=np.float32)
        first = self._metadata_palletizer(policies)
        second = self._metadata_palletizer(list(reversed(policies)))
        indices_a, metadata_a = first._candidate_leaf_metadata(
            observation, leaves, "union"
        )
        indices_b, metadata_b = second._candidate_leaf_metadata(
            observation, leaves, "union"
        )
        self.assertEqual(indices_a, indices_b)
        self.assertEqual(metadata_a, metadata_b)
        proposed = {
            name
            for item in metadata_a.values()
            for name in item["proposal_models"]
        }
        self.assertEqual(proposed, {name for name, _ in policies})

    def test_config_accepts_only_single_fallback_and_adaptive(self):
        palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
        self.assertIn(
            palletizer._candidate_union_mode,
            {"single", "fallback", "adaptive"},
        )

    def test_single_loads_only_the_first_policy_model(self):
        palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
        palletizer._ensure_policy()
        self.assertEqual(len(palletizer._policy_sessions), 1)
        self.assertEqual(
            palletizer._policy_sessions[0]["name"],
            "pct_model.onnx",
        )

    def test_fallback_loads_all_configured_policy_models(self):
        palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
        palletizer._candidate_union_enabled = True
        palletizer._candidate_union_mode = "fallback"
        palletizer._ensure_policy()
        self.assertEqual(len(palletizer._policy_sessions), 3)

    def test_fallback_uses_configured_fallback_model(self):
        policies = [
            ("candidate-001450.onnx", [0.9, 0.1, 0.2, 0.3, 0.0]),
            ("candidate-001400.onnx", [0.1, 0.2, 0.95, 0.3, 0.0]),
            ("pct_model.onnx", [0.2, 0.8, 0.1, 0.3, 0.0]),
        ]
        palletizer = self._metadata_palletizer(policies)
        leaves = np.ones((5, 9), dtype=np.float32)
        indices, metadata = palletizer._candidate_leaf_metadata(
            np.zeros((1, 45), np.float32),
            leaves,
            "fallback",
        )
        self.assertEqual(indices[0], 2)
        self.assertEqual(
            metadata[2]["proposal_models"],
            ("candidate-001400.onnx",),
        )

    def test_identical_placement_candidates_are_deduplicated(self):
        class FakePacker:
            def __init__(self):
                self.packed = []

            def place(self, _leaf):
                self.packed.append([0.2, 0.2, 0.2, 0, 0, 0, 0])
                return True

        palletizer = Palletizer(PALLET, AlgorithmConfig(True, 0))
        palletizer._score_trial = lambda *_args, **_kwargs: 1.0
        box = {"step": 0, "id": 0, "size": [0.2, 0.2, 0.2], "mass": 1.0}
        leaves = np.zeros((2, 9), np.float32)
        leaves[:, :6] = [0, 0, 0, 0.2, 0.2, 1.25]
        leaves[:, 8] = 1
        metadata = {
            0: {
                "proposal_mask": 1,
                "proposal_models": ("a",),
                "model_ranks": {"a": 1.0},
                "model_probabilities": {"a": 0.5},
            },
            1: {
                "proposal_mask": 2,
                "proposal_models": ("b",),
                "model_ranks": {"b": 1.0},
                "model_probabilities": {"b": 0.5},
            },
        }
        candidates = palletizer._build_trial_candidates(
            box,
            FakePacker(),
            FakePacker(),
            leaves,
            [],
            0,
            [0, 1],
            metadata,
            2,
            np.zeros((1, 18), np.float32),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_metadata"]["proposal_mask"], 3)
        self.assertEqual(
            candidates[0]["candidate_metadata"]["proposal_models"],
            ("a", "b"),
        )

    def test_buffer_zero_never_requests_a_future_item(self):
        class SpyBuffer:
            instances = []

            def __init__(self, size):
                self.size = size
                self.items = []
                self.index = 0
                self.peeked = []
                self.__class__.instances.append(self)

            def reset(self, boxes):
                self.items = boxes

            def has_pending(self):
                return self.index < len(self.items)

            def peek_next(self):
                self.peeked.append(self.index)
                return self.items[self.index]

            def get_buffer(self):
                raise AssertionError(
                    "buffer=0 inference requested future buffer contents"
                )

            def pop_next(self):
                self.index += 1

        class OnlinePalletizer(Palletizer):
            def _find_position(self, box):
                return (
                    0.2 * int(box["step"]),
                    0.0,
                    0.0,
                    (0.1, 0.1, 0.1),
                    0,
                )

        boxes = [
            {
                "step": index,
                "id": index,
                "size": [0.1, 0.1, 0.1],
                "mass": 1.0,
            }
            for index in range(3)
        ]
        with mock.patch.object(algorithm_module, "BufferManager", SpyBuffer):
            result = OnlinePalletizer(
                PALLET, AlgorithmConfig(True, 0)
            ).run(boxes)
        self.assertEqual(
            [item["id"] for item in result["sequence"]],
            [0, 1, 2],
        )
        self.assertEqual(SpyBuffer.instances[0].peeked, [0, 1, 2])

if __name__ == "__main__":
    unittest.main()
