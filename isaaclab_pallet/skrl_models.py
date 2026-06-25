"""skrl policy/value models for the pallet env (issue ⑥: standard RL via skrl).

Key trick (porting guide §4): Isaac Lab's RL wrappers have NO action-masking
wiring, so masking is done *inside the policy*. The leaf validity flag is already
carried in the observation (each PCT leaf node's slot 8), so the policy
reconstructs the mask from the observation tensor itself — no side channel needed.

Layout of the policy observation (see PalletPackingEnv._get_observations):
    [ obs_node_count * 9  PCT nodes ] [ physics_feature_dim ]
with obs_node_count = internal_node_holder + leaf_node_holder + 1, and each node's
slot 8 = valid flag. Leaf nodes are nodes[internal : internal+leaf].
"""
from __future__ import annotations

import torch
from torch import nn

from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model


_NODE_DIM = 9
_VALID_SLOT = 8


def _leaf_valid_mask(states: torch.Tensor, internal_nodes: int, leaf_nodes: int) -> torch.Tensor:
    """Reconstruct the per-leaf valid mask from the flattened observation.

    Returns a (B, leaf_nodes) bool tensor. Rows with no valid leaf fall back to
    all-True so the Categorical distribution stays well-defined (the env will
    terminate those envs anyway).
    """
    node_count = internal_nodes + leaf_nodes + 1
    nodes = states[:, : node_count * _NODE_DIM].view(states.shape[0], node_count, _NODE_DIM)
    valid = nodes[:, internal_nodes:internal_nodes + leaf_nodes, _VALID_SLOT] > 0.5
    any_valid = valid.any(dim=-1, keepdim=True)
    return torch.where(any_valid, valid, torch.ones_like(valid))


class MaskedCategoricalPolicy(CategoricalMixin, Model):
    """Discrete leaf-index policy with observation-derived invalid-action masking."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        internal_nodes: int = 200,
        leaf_nodes: int = 100,
        hidden_dim: int = 256,
        unnormalized_log_prob: bool = True,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=unnormalized_log_prob)
        self.internal_nodes = int(internal_nodes)
        self.leaf_nodes = int(leaf_nodes)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logit_head = nn.Linear(hidden_dim, self.num_actions)

    def compute(self, inputs, role):
        states = inputs["states"]
        logits = self.logit_head(self.net(states))
        mask = _leaf_valid_mask(states, self.internal_nodes, self.leaf_nodes)
        logits = logits.masked_fill(~mask, -1.0e9)
        return logits, {}


class ValueModel(DeterministicMixin, Model):
    """State-value critic."""

    def __init__(self, observation_space, action_space, device, hidden_dim: int = 256, clip_actions: bool = False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
