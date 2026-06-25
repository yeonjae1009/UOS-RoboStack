from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class MaskedPalletMLP(nn.Module):
    """Small actor-critic policy for the first Isaac Lab training loop.

    This is intentionally simpler than Online-3D-BPP-PCT's GAT. It keeps the same
    discrete leaf-index action semantics and invalid-leaf masking, so it can be
    replaced by the original GAT-style policy once the Isaac physics loop is stable.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.shared = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, self.action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def _masked_logits(self, logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return logits
        action_mask = action_mask.bool()
        valid_any = action_mask.any(dim=-1, keepdim=True)
        safe_mask = torch.where(valid_any, action_mask, torch.ones_like(action_mask))
        return logits.masked_fill(~safe_mask, -1.0e9)

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(obs)
        logits = self._masked_logits(self.actor(hidden), action_mask)
        value = self.critic(hidden)
        return logits, value

    def act(
        self, obs: torch.Tensor, action_mask: torch.Tensor | None = None, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return action.unsqueeze(-1), log_prob, entropy, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        action = actions.squeeze(-1)
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return log_prob, entropy, value
