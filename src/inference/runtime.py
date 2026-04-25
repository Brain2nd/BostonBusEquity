"""
Realtime inference runtime for the V2 causal MLP model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from src.inference.bundle_utils import V2_FEATURE_COLUMNS
from src.models.v2_mlp import V2MLPPredictor

LOCAL_TZ = ZoneInfo("America/New_York")


class InferenceInputError(ValueError):
    """Raised when runtime input cannot be converted into model features."""


@dataclass
class PredictRequest:
    route_id: str
    stop_id: str
    scheduled_time: str
    scheduled_headway: float | None = None
    direction_id: str | None = None


class RealtimeDelayPredictor:
    """Loads the realtime bundle and runs single-record predictions."""

    def __init__(self, bundle: dict) -> None:
        self.bundle = bundle
        self.feature_columns = bundle["feature_columns"]
        self.stats = bundle["statistics"]
        self.mappings = bundle["mappings"]
        self.scaler_x = bundle["scaler_X"]
        self.scaler_y = bundle["scaler_y"]

        model_config = bundle["model_config"]
        self.model = V2MLPPredictor(
            input_size=model_config["input_size"],
            hidden_sizes=model_config["hidden_sizes"],
            dropout=model_config["dropout"],
        )
        self.model.load_state_dict(bundle["model_state_dict"])
        self.model.eval()

    @classmethod
    def from_path(cls, bundle_path: str | Path) -> "RealtimeDelayPredictor":
        bundle = torch.load(Path(bundle_path), map_location="cpu")
        return cls(bundle)

    def _parse_time(self, scheduled_time: str) -> pd.Timestamp:
        parsed = pd.Timestamp(scheduled_time)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(LOCAL_TZ)
        return parsed.tz_convert(LOCAL_TZ)

    def build_feature_vector(self, request: PredictRequest) -> tuple[np.ndarray, list[str]]:
        used_defaults: list[str] = []

        route_id = str(request.route_id)
        stop_id = str(request.stop_id)
        if request.direction_id is None:
            direction_id = "Unknown"
            used_defaults.append("direction_id")
        else:
            direction_id = str(request.direction_id)

        if route_id not in self.mappings["route_id"]:
            raise InferenceInputError(f"Unknown route_id: {route_id}")
        if stop_id not in self.mappings["stop_id"]:
            raise InferenceInputError(f"Unknown stop_id: {stop_id}")
        if direction_id not in self.mappings["direction_id"]:
            direction_id = "Unknown"
            used_defaults.append("direction_id")
        if direction_id not in self.mappings["direction_id"]:
            raise InferenceInputError("Bundle does not include an 'Unknown' direction mapping")

        scheduled_time = self._parse_time(request.scheduled_time)
        scheduled_headway = request.scheduled_headway
        if scheduled_headway is None:
            scheduled_headway = self.stats["scheduled_headway_median"]
            used_defaults.append("scheduled_headway")

        hour = scheduled_time.hour
        day_of_week = scheduled_time.dayofweek
        month = scheduled_time.month
        route_hour_key = f"{route_id}_{hour}"

        vector = {
            "is_weekend": float(day_of_week >= 5),
            "is_rush_hour": float((7 <= hour <= 9) or (16 <= hour <= 19)),
            "route_encoded": float(self.mappings["route_id"][route_id]),
            "stop_encoded": float(self.mappings["stop_id"][stop_id]),
            "direction_encoded": float(self.mappings["direction_id"][direction_id]),
            "scheduled_headway": float(scheduled_headway),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
            "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
            "month_sin": float(np.sin(2 * np.pi * month / 12)),
            "month_cos": float(np.cos(2 * np.pi * month / 12)),
            "route_delay_mean": float(
                self.stats["route_delay_mean"].get(route_id, self.stats["global_mean"])
            ),
            "route_delay_std": float(
                self.stats["route_delay_std"].get(route_id, self.stats["global_std"])
            ),
            "stop_delay_mean": float(
                self.stats["stop_delay_mean"].get(stop_id, self.stats["global_mean"])
            ),
            "stop_delay_std": float(
                self.stats["stop_delay_std"].get(stop_id, self.stats["global_std"])
            ),
            "hour_delay_mean": float(
                self.stats["hour_delay_mean"].get(str(hour), self.stats["global_mean"])
            ),
            "route_hour_delay_mean": float(
                self.stats["route_hour_delay_mean"].get(route_hour_key, self.stats["global_mean"])
            ),
        }

        feature_values = np.array(
            [[vector[column] for column in self.feature_columns]],
            dtype=np.float32,
        )
        scale = np.asarray(self.scaler_x["scale"], dtype=np.float32)
        mean = np.asarray(self.scaler_x["mean"], dtype=np.float32)
        scaled = (feature_values - mean) / scale
        return scaled, used_defaults

    def predict(self, request: PredictRequest) -> dict:
        feature_values, used_defaults = self.build_feature_vector(request)
        tensor = torch.tensor(feature_values, dtype=torch.float32)
        with torch.no_grad():
            scaled_prediction = self.model(tensor).cpu().numpy().reshape(-1)

        prediction = (
            scaled_prediction * np.asarray(self.scaler_y["scale"], dtype=np.float32)
            + np.asarray(self.scaler_y["mean"], dtype=np.float32)
        )[0]

        return {
            "predicted_delay_minutes": float(prediction),
            "model": self.bundle["model_name"],
            "experiment": self.bundle["experiment"],
            "used_defaults": used_defaults,
        }
