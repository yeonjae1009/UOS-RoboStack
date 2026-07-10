from __future__ import annotations

import torch
import torch.nn as nn


BUFFER_SELECTOR_BUFFER_SIZE = 3
BUFFER_SELECTOR_SLOT_FEATURES = 5
BUFFER_SELECTOR_INPUT_DIM = 200 * 9 + BUFFER_SELECTOR_BUFFER_SIZE * BUFFER_SELECTOR_SLOT_FEATURES


class BufferSelectorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = BUFFER_SELECTOR_INPUT_DIM,
        hidden_dim: int = 256,
        output_dim: int = BUFFER_SELECTOR_BUFFER_SIZE,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
