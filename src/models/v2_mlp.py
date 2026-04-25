"""
Shared V2 MLP model definition for training and realtime inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


V2_HIDDEN_SIZES = [128, 64, 32]
V2_DROPOUT = 0.2


class V2MLPPredictor(nn.Module):
    """MLP architecture used by the V2 lag-feature experiment."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = V2_DROPOUT,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or V2_HIDDEN_SIZES

        layers: list[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
