"""Runtime helpers for local realtime delay inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.v2_delay_predictor import V2_FEATURE_COLUMNS, V2MLPPredictor
from src.models.v4_delay_predictor import V4_FEATURE_COLUMNS

LOCAL_TIMEZONE = ZoneInfo("America/New_York")


class PredictionInputError(ValueError):
    """Raised when a prediction request cannot be satisfied safely."""


@dataclass
class PredictionResult:
    predicted_delay_minutes: float
    model: str
    experiment: str
    used_defaults: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_delay_minutes": self.predicted_delay_minutes,
            "model": self.model,
            "experiment": self.experiment,
            "used_defaults": list(self.used_defaults),
        }


def _ensure_required_keys(bundle: dict[str, Any], required_keys: list[str]) -> None:
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise ValueError(f"Bundle is missing required keys: {missing}")


def _as_string(value: Any) -> str:
    return str(value).strip()


def _parse_scheduled_time(value: datetime | str) -> datetime:
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        scheduled_time = datetime.fromisoformat(normalized)
    else:
        scheduled_time = value

    if scheduled_time.tzinfo is None:
        return scheduled_time.replace(tzinfo=LOCAL_TIMEZONE)

    return scheduled_time.astimezone(LOCAL_TIMEZONE)


def _scale_array(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scale == 0, 1.0, scale)
    return (values - mean) / safe_scale


def _inverse_scale_scalar(value: float, mean: float, scale: float) -> float:
    safe_scale = scale if scale != 0 else 1.0
    return float(value * safe_scale + mean)


def _coerce_headway_to_training_seconds(
    value: float | int | None,
    default_seconds: float,
    used_defaults: list[str],
) -> float:
    """Normalize public headway input to the seconds used by training data.

    Dashboard/API users enter minutes, while the historical MBTA processed
    feature is stored in seconds. Values above two hours are treated as already
    being seconds for backwards compatibility with scripts that pass raw data.
    """

    if value is None:
        used_defaults.append("scheduled_headway")
        return float(default_seconds)
    numeric = float(value)
    if numeric <= 120:
        return numeric * 60.0
    return numeric


class DelayPredictorRuntime:
    """Load a serialized realtime bundle and run V2 or V4 predictions."""

    def __init__(self, bundle: dict[str, Any], bundle_path: Path | None = None) -> None:
        self.bundle = bundle
        self.bundle_path = bundle_path
        self.model_family = str(bundle.get("model_family", "v2_mlp"))

        if self.model_family == "v4_tree":
            self._init_v4_bundle(bundle)
        else:
            self._init_v2_bundle(bundle)

    def _init_v2_bundle(self, bundle: dict[str, Any]) -> None:
        required_keys = [
            "bundle_version",
            "model_name",
            "experiment",
            "feature_version",
            "feature_columns",
            "model_config",
            "model_state_dict",
            "scalers",
            "encoders",
            "stats",
        ]
        _ensure_required_keys(bundle, required_keys)

        model_config = dict(bundle["model_config"])
        self.model = V2MLPPredictor(
            input_size=model_config["input_size"],
            hidden_sizes=model_config["hidden_sizes"],
            dropout=model_config["dropout"],
        )
        self.model.load_state_dict(bundle["model_state_dict"])
        self.model.eval()

        self.feature_columns = list(bundle["feature_columns"])
        self.scalers = bundle["scalers"]
        self.encoders = bundle["encoders"]
        self.stats = bundle["stats"]
        self.model_kind = "torch_mlp"

    def _init_v4_bundle(self, bundle: dict[str, Any]) -> None:
        required_keys = [
            "bundle_version",
            "model_name",
            "experiment",
            "feature_version",
            "feature_columns",
            "model",
            "encoders",
            "stats",
        ]
        _ensure_required_keys(bundle, required_keys)

        self.model = bundle["model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.scalers = {}
        self.encoders = bundle["encoders"]
        self.stats = bundle["stats"]
        self.model_kind = str(bundle.get("model_kind", "tree"))

    @classmethod
    def from_bundle_path(cls, bundle_path: str | Path) -> "DelayPredictorRuntime":
        resolved = Path(bundle_path).resolve()
        try:
            bundle = torch.load(resolved, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(resolved, map_location="cpu")
        except Exception:
            bundle = joblib.load(resolved)
        return cls(bundle=bundle, bundle_path=resolved)

    def health(self) -> dict[str, Any]:
        return {
            "bundle_loaded": True,
            "model": self.bundle["model_name"],
            "experiment": self.bundle["experiment"],
            "feature_version": self.bundle["feature_version"],
            "model_family": self.model_family,
            "model_kind": self.bundle.get("model_kind"),
            "feature_profile": self.bundle.get("feature_profile"),
            "training_protocol": self.bundle.get("training_protocol"),
            "bundle_path": str(self.bundle_path) if self.bundle_path else None,
        }

    def _get_encoded_value(
        self,
        field_name: str,
        raw_value: str | None,
        used_defaults: list[str],
    ) -> int:
        mapping = self.encoders[field_name]

        if field_name in {"route_id", "stop_id"}:
            value = self._canonical_known_value(field_name, _as_string(raw_value))
            if value not in mapping:
                raise PredictionInputError(f"Unknown {field_name}: {value}")
            return int(mapping[value])

        if raw_value is None:
            used_defaults.append("direction_id")
            value = "Unknown"
        else:
            value = _as_string(raw_value)

        if value not in mapping:
            raise PredictionInputError(f"Unknown {field_name}: {value}")

        return int(mapping[value])

    def _canonical_known_value(self, field_name: str, value: str) -> str:
        """Resolve harmless MBTA/historical ID formatting differences.

        Historical processed bus route IDs may be zero-padded (`01`) while the
        live MBTA API uses unpadded IDs (`1`). We only return aliases that are
        already present in the trained encoder, so unknown IDs still fail.
        """
        mapping = self.encoders.get(field_name, {})
        if value in mapping:
            return value
        if field_name == "route_id" and value.isdigit():
            for width in [2, 3]:
                candidate = value.zfill(width)
                if candidate in mapping:
                    return candidate
            stripped_matches = [
                key for key in mapping if key.isdigit() and key.lstrip("0") == value
            ]
            if len(stripped_matches) == 1:
                return stripped_matches[0]
        return value

    def _build_feature_vector(
        self,
        route_id: str,
        stop_id: str,
        scheduled_time: datetime | str,
        scheduled_headway: float | None = None,
        direction_id: str | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        used_defaults: list[str] = []
        route_key = self._canonical_known_value("route_id", _as_string(route_id))
        stop_key = self._canonical_known_value("stop_id", _as_string(stop_id))
        scheduled_local = _parse_scheduled_time(scheduled_time)

        route_encoded = self._get_encoded_value("route_id", route_key, used_defaults)
        stop_encoded = self._get_encoded_value("stop_id", stop_key, used_defaults)
        direction_encoded = self._get_encoded_value(
            "direction_id",
            direction_id,
            used_defaults,
        )

        scheduled_headway_value = _coerce_headway_to_training_seconds(
            scheduled_headway,
            float(self.stats["scheduled_headway_median"]),
            used_defaults,
        )

        hour = scheduled_local.hour
        day_of_week = scheduled_local.weekday()
        month = scheduled_local.month
        route_hour_key = f"{route_key}_{hour}"

        route_stats = self.stats["route"].get(route_key, {})
        stop_stats = self.stats["stop"].get(stop_key, {})
        hour_delay_mean = self.stats["hour"].get(str(hour), self.stats["global_mean"])
        route_hour_delay_mean = self.stats["route_hour"].get(
            route_hour_key,
            self.stats["global_mean"],
        )

        feature_values = np.array(
            [
                int(day_of_week >= 5),
                int((7 <= hour <= 9) or (16 <= hour <= 19)),
                route_encoded,
                stop_encoded,
                direction_encoded,
                scheduled_headway_value,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7),
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                route_stats.get("mean", self.stats["global_mean"]),
                route_stats.get("std", self.stats["global_std"]),
                stop_stats.get("mean", self.stats["global_mean"]),
                stop_stats.get("std", self.stats["global_std"]),
                hour_delay_mean,
                route_hour_delay_mean,
            ],
            dtype=np.float32,
        )

        if self.feature_columns != V2_FEATURE_COLUMNS:
            raise ValueError("Bundle feature columns do not match the V2 runtime layout")

        return feature_values, used_defaults

    @staticmethod
    def _append_default(used_defaults: list[str], field_name: str) -> None:
        if field_name not in used_defaults:
            used_defaults.append(field_name)

    def _get_v4_encoded_value(
        self,
        field_name: str,
        raw_value: str | None,
        used_defaults: list[str],
    ) -> int:
        mapping = self.encoders[field_name]

        if field_name in {"route_id", "stop_id"}:
            value = self._canonical_known_value(field_name, _as_string(raw_value))
            if value not in mapping:
                raise PredictionInputError(f"Unknown {field_name}: {value}")
            return int(mapping[value])

        if raw_value is None:
            self._append_default(used_defaults, field_name)
            value = "Unknown"
        else:
            value = _as_string(raw_value)

        if value not in mapping:
            if "Unknown" in mapping:
                self._append_default(used_defaults, f"{field_name}_unknown")
                value = "Unknown"
            else:
                raise PredictionInputError(f"Unknown {field_name}: {value}")

        return int(mapping[value])

    def _v4_stat_pair(self, map_name: str, key: str) -> tuple[float, float]:
        value = self.stats.get(map_name, {}).get(str(key), {})
        return (
            float(value.get("mean", self.stats["global_mean"])),
            float(value.get("std", self.stats["global_std"])),
        )

    def historical_baseline_delay(
        self,
        route_id: str,
        stop_id: str,
        scheduled_time: datetime | str,
        direction_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the strongest available historical mean baseline.

        This is intentionally simple and interpretable: prefer route-stop-hour
        history, then route-stop, route-hour, route, stop, hour, and global
        training means. It is a baseline, not a learned realtime model.
        """

        route_key = self._canonical_known_value("route_id", _as_string(route_id))
        stop_key = self._canonical_known_value("stop_id", _as_string(stop_id))
        if route_key not in self.encoders.get("route_id", {}):
            raise PredictionInputError(f"Unknown route_id: {route_key}")
        if stop_key not in self.encoders.get("stop_id", {}):
            raise PredictionInputError(f"Unknown stop_id: {stop_key}")

        scheduled_local = _parse_scheduled_time(scheduled_time)
        hour = scheduled_local.hour
        candidates = [
            ("route_stop_hour", f"{route_key}_{stop_key}_{hour}"),
            ("route_stop", f"{route_key}_{stop_key}"),
            ("route_direction_hour", f"{route_key}_{direction_id or 'Unknown'}_{hour}"),
            ("route_hour", f"{route_key}_{hour}"),
            ("route", route_key),
            ("stop", stop_key),
            ("hour", str(hour)),
        ]
        for stat_name, key in candidates:
            value = self.stats.get(stat_name, {}).get(str(key))
            if isinstance(value, dict) and "mean" in value:
                return {
                    "predicted_delay_minutes": float(value["mean"]),
                    "source": stat_name,
                    "key": str(key),
                }
            if value is not None and not isinstance(value, dict):
                return {
                    "predicted_delay_minutes": float(value),
                    "source": stat_name,
                    "key": str(key),
                }

        return {
            "predicted_delay_minutes": float(self.stats["global_mean"]),
            "source": "global_mean",
            "key": "global_mean",
        }

    def _numeric_or_default(
        self,
        value: float | int | None,
        default: float,
        field_name: str,
        used_defaults: list[str],
    ) -> float:
        if value is None:
            self._append_default(used_defaults, field_name)
            return float(default)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            self._append_default(used_defaults, field_name)
            return float(default)
        if not np.isfinite(numeric):
            self._append_default(used_defaults, field_name)
            return float(default)
        return numeric

    def _build_v4_feature_frame(
        self,
        route_id: str,
        stop_id: str,
        scheduled_time: datetime | str,
        scheduled_headway: float | None = None,
        direction_id: str | None = None,
        trip_id: str | None = None,
        current_stop_sequence: float | None = None,
        vehicle_speed: float | None = None,
        official_prediction_age_seconds: float | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        used_defaults: list[str] = []
        route_key = self._canonical_known_value("route_id", _as_string(route_id))
        stop_key = self._canonical_known_value("stop_id", _as_string(stop_id))
        scheduled_local = _parse_scheduled_time(scheduled_time)

        route_encoded = self._get_v4_encoded_value("route_id", route_key, used_defaults)
        stop_encoded = self._get_v4_encoded_value("stop_id", stop_key, used_defaults)
        direction_encoded = self._get_v4_encoded_value(
            "direction_id",
            direction_id,
            used_defaults,
        )
        half_trip_encoded = self._get_v4_encoded_value(
            "half_trip_id",
            trip_id,
            used_defaults,
        )

        headway_missing = scheduled_headway is None
        scheduled_headway_value = _coerce_headway_to_training_seconds(
            scheduled_headway,
            float(self.stats["scheduled_headway_median"]),
            used_defaults,
        )

        current_sequence_value = self._numeric_or_default(
            current_stop_sequence,
            0.0,
            "current_stop_sequence",
            used_defaults,
        )
        vehicle_speed_value = self._numeric_or_default(
            vehicle_speed,
            0.0,
            "vehicle_speed",
            used_defaults,
        )
        prediction_age_value = self._numeric_or_default(
            official_prediction_age_seconds,
            0.0,
            "official_prediction_age_seconds",
            used_defaults,
        )

        hour = scheduled_local.hour
        minute = scheduled_local.minute
        minute_of_day = hour * 60 + minute
        day_of_week = scheduled_local.weekday()
        day_of_year = scheduled_local.timetuple().tm_yday
        month = scheduled_local.month
        route_hour_key = f"{route_key}_{hour}"
        route_stop_key = f"{route_key}_{stop_key}"
        route_stop_hour_key = f"{route_stop_key}_{hour}"
        order_key = str(int(round(current_sequence_value))) if current_sequence_value else "-1"
        route_stop_order_key = f"{route_stop_key}_{order_key}"
        route_direction_hour_key = f"{route_key}_{direction_id or 'Unknown'}_{hour}"

        route_mean, route_std = self._v4_stat_pair("route", route_key)
        stop_mean, stop_std = self._v4_stat_pair("stop", stop_key)
        hour_mean, _ = self._v4_stat_pair("hour", str(hour))
        route_hour_mean, _ = self._v4_stat_pair("route_hour", route_hour_key)
        route_stop_mean, route_stop_std = self._v4_stat_pair(
            "route_stop",
            route_stop_key,
        )
        route_stop_hour_mean, route_stop_hour_std = self._v4_stat_pair(
            "route_stop_hour",
            route_stop_hour_key,
        )
        route_stop_order_mean, route_stop_order_std = self._v4_stat_pair(
            "route_stop_order",
            route_stop_order_key,
        )
        route_direction_hour_mean, route_direction_hour_std = self._v4_stat_pair(
            "route_direction_hour",
            route_direction_hour_key,
        )
        route_max_sequence = float(
            self.stats.get("route_max_time_point_order", {}).get(
                route_key,
                self.stats.get("global_max_time_point_order", 1.0),
            )
        )
        if not np.isfinite(route_max_sequence) or route_max_sequence <= 0:
            route_max_sequence = float(self.stats.get("global_max_time_point_order", 1.0))
        stop_sequence_fraction = float(np.clip(current_sequence_value / route_max_sequence, 0.0, 1.0))

        self._append_default(used_defaults, "trip_history")
        self._append_default(used_defaults, "previous_headway_deviation")
        global_mean = float(self.stats["global_mean"])
        feature_values = {
            "is_weekend": int(day_of_week >= 5),
            "is_rush_hour": int((7 <= hour <= 9) or (16 <= hour <= 19)),
            "route_encoded": route_encoded,
            "stop_encoded": stop_encoded,
            "direction_encoded": direction_encoded,
            "scheduled_headway": scheduled_headway_value,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "minute_of_day": minute_of_day,
            "minute_of_day_sin": np.sin(2 * np.pi * minute_of_day / 1440),
            "minute_of_day_cos": np.cos(2 * np.pi * minute_of_day / 1440),
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 366),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 366),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "route_delay_mean": route_mean,
            "route_delay_std": route_std,
            "stop_delay_mean": stop_mean,
            "stop_delay_std": stop_std,
            "hour_delay_mean": hour_mean,
            "route_hour_delay_mean": route_hour_mean,
            "time_point_order": current_sequence_value,
            "half_trip_encoded": half_trip_encoded,
            "trip_delay_lag_1": global_mean,
            "trip_delay_lag_2": global_mean,
            "trip_delay_lag_3": global_mean,
            "trip_delay_rolling_mean": global_mean,
            "trip_delay_rolling_std": 0.0,
            "trip_delay_rolling_max": global_mean,
            "route_stop_delay_mean": route_stop_mean,
            "route_stop_delay_std": route_stop_std,
            "route_stop_hour_delay_mean": route_stop_hour_mean,
            "route_stop_hour_delay_std": route_stop_hour_std,
            "route_stop_order_delay_mean": route_stop_order_mean,
            "route_stop_order_delay_std": route_stop_order_std,
            "route_direction_hour_delay_mean": route_direction_hour_mean,
            "route_direction_hour_delay_std": route_direction_hour_std,
            "previous_headway_deviation": 0.0,
            "scheduled_headway_missing": int(headway_missing),
            "scheduled_headway_log": np.log1p(max(scheduled_headway_value, 0.0)),
            "stop_sequence_fraction": stop_sequence_fraction,
            "current_stop_sequence": current_sequence_value,
            "vehicle_speed": vehicle_speed_value,
            "official_prediction_age_seconds": prediction_age_value,
        }

        for column in self.feature_columns:
            feature_values.setdefault(column, 0.0)

        return pd.DataFrame([feature_values], columns=self.feature_columns), used_defaults

    def predict(
        self,
        route_id: str,
        stop_id: str,
        scheduled_time: datetime | str,
        scheduled_headway: float | None = None,
        direction_id: str | None = None,
        trip_id: str | None = None,
        vehicle_id: str | None = None,
        current_stop_sequence: float | None = None,
        vehicle_speed: float | None = None,
        vehicle_status: str | None = None,
        official_predicted_delay_minutes: float | None = None,
        official_prediction_age_seconds: float | None = None,
    ) -> dict[str, Any]:
        if self.model_family == "v4_tree":
            feature_frame, used_defaults = self._build_v4_feature_frame(
                route_id=route_id,
                stop_id=stop_id,
                scheduled_time=scheduled_time,
                scheduled_headway=scheduled_headway,
                direction_id=direction_id,
                trip_id=trip_id,
                current_stop_sequence=current_stop_sequence,
                vehicle_speed=vehicle_speed,
                official_prediction_age_seconds=official_prediction_age_seconds,
            )
            prediction = float(np.asarray(self.model.predict(feature_frame))[0])
            result = PredictionResult(
                predicted_delay_minutes=prediction,
                model=self.bundle["model_name"],
                experiment=self.bundle["experiment"],
                used_defaults=used_defaults,
            )
            return result.to_dict()

        feature_values, used_defaults = self._build_feature_vector(
            route_id=route_id,
            stop_id=stop_id,
            scheduled_time=scheduled_time,
            scheduled_headway=scheduled_headway,
            direction_id=direction_id,
        )

        scaler_x = self.scalers["x"]
        scaler_y = self.scalers["y"]
        feature_mean = np.asarray(scaler_x["mean"], dtype=np.float32)
        feature_scale = np.asarray(scaler_x["scale"], dtype=np.float32)
        scaled = _scale_array(feature_values, feature_mean, feature_scale)

        model_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction_scaled = float(self.model(model_input).item())

        prediction = _inverse_scale_scalar(
            prediction_scaled,
            float(scaler_y["mean"][0]),
            float(scaler_y["scale"][0]),
        )

        result = PredictionResult(
            predicted_delay_minutes=prediction,
            model=self.bundle["model_name"],
            experiment=self.bundle["experiment"],
            used_defaults=used_defaults,
        )
        return result.to_dict()
