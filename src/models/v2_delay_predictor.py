"""
Shared V2 delay predictor definitions.

This module keeps the realtime inference stack aligned with the
V2 temporal-split experiment checkpoint.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

V2_EXPERIMENT_VERSION = "v2_lag_features_temporal"
V2_FEATURE_VERSION = "v2_causal_statistics"
V2_FEATURE_COLUMNS = [
    "is_weekend",
    "is_rush_hour",
    "route_encoded",
    "stop_encoded",
    "direction_encoded",
    "scheduled_headway",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "route_delay_mean",
    "route_delay_std",
    "stop_delay_mean",
    "stop_delay_std",
    "hour_delay_mean",
    "route_hour_delay_mean",
]
V2_MLP_HIDDEN_SIZES = (128, 64, 32)
V2_MLP_DROPOUT = 0.2
V2_CHECKPOINT_NAME = f"delay_predictor_mlp_{V2_EXPERIMENT_VERSION}.pt"
V2_REALTIME_BUNDLE_NAME = (
    f"delay_predictor_mlp_{V2_EXPERIMENT_VERSION}_realtime_bundle.pt"
)


class V2MLPPredictor(nn.Module):
    """MLP architecture used by the V2 delay predictor checkpoint."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = V2_MLP_HIDDEN_SIZES,
        dropout: float = V2_MLP_DROPOUT,
    ) -> None:
        super().__init__()

        layers = []
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


def build_v2_model_config(
    input_size: int = len(V2_FEATURE_COLUMNS),
    hidden_sizes: Sequence[int] = V2_MLP_HIDDEN_SIZES,
    dropout: float = V2_MLP_DROPOUT,
) -> dict:
    """Return a serializable config for the V2 MLP predictor."""
    return {
        "input_size": int(input_size),
        "hidden_sizes": [int(size) for size in hidden_sizes],
        "dropout": float(dropout),
    }


__all__ = [
    "V2_CHECKPOINT_NAME",
    "V2_EXPERIMENT_VERSION",
    "V2_FEATURE_COLUMNS",
    "V2_FEATURE_VERSION",
    "V2_MLP_DROPOUT",
    "V2_MLP_HIDDEN_SIZES",
    "V2_REALTIME_BUNDLE_NAME",
    "V2MLPPredictor",
    "build_v2_model_config",
]
