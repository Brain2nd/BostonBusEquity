"""
Real-time inference utilities for existing MBTA delay predictor checkpoints.

This module keeps inference tied to the original project checkpoints. It does
not introduce any external model families; it rebuilds the exact architectures
already used by the training scripts and layers an online feature builder on
top so the models can consume live records.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import os
import time

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

from src.config import MBTA_API_BASE_URL, PROJECT_ROOT

try:
    import pywt

    HAS_PYWT = True
except ImportError:  # pragma: no cover - optional dependency
    pywt = None
    HAS_PYWT = False


BASELINE_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "route_encoded",
    "stop_encoded",
    "direction_encoded",
    "scheduled_headway",
]

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

V3_FEATURE_COLUMNS = [
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
    "delay_trend",
    "delay_detail",
    "delay_denoised",
    "delay_rolling_mean",
    "delay_rolling_std",
    "delay_rolling_min",
    "delay_rolling_max",
    "delay_lag_1",
    "delay_lag_2",
    "delay_lag_3",
    "delay_diff",
]

FEATURE_COLUMNS_BY_VERSION = {
    "v1": BASELINE_FEATURE_COLUMNS,
    "v2": V2_FEATURE_COLUMNS,
    "v3": V3_FEATURE_COLUMNS,
}


@dataclass
class ModelSpec:
    """Architecture metadata reconstructed from the existing checkpoints."""

    checkpoint_name: str
    model_family: str
    feature_version: str
    input_size: int
    hidden_sizes: Tuple[int, ...] = ()
    hidden_size: int = 0
    num_layers: int = 0


KNOWN_MODEL_SPECS = {
    "delay_predictor_mlp_v1_baseline_temporal.pt": ModelSpec(
        checkpoint_name="delay_predictor_mlp_v1_baseline_temporal.pt",
        model_family="mlp",
        feature_version="v1",
        input_size=9,
        hidden_sizes=(128, 64, 32),
    ),
    "delay_predictor_mlp_v2_lag_features_temporal.pt": ModelSpec(
        checkpoint_name="delay_predictor_mlp_v2_lag_features_temporal.pt",
        model_family="mlp",
        feature_version="v2",
        input_size=18,
        hidden_sizes=(128, 64, 32),
    ),
    "delay_predictor_lstm_v3_wavelet_temporal.pt": ModelSpec(
        checkpoint_name="delay_predictor_lstm_v3_wavelet_temporal.pt",
        model_family="lstm",
        feature_version="v3",
        input_size=28,
        hidden_size=128,
        num_layers=2,
    ),
    "delay_predictor_gru_v3_wavelet_temporal.pt": ModelSpec(
        checkpoint_name="delay_predictor_gru_v3_wavelet_temporal.pt",
        model_family="gru",
        feature_version="v3",
        input_size=28,
        hidden_size=128,
        num_layers=2,
    ),
}


@dataclass
class SimpleScaler:
    """Minimal StandardScaler replacement for inference-time normalization."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "SimpleScaler":
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return (values * self.scale + self.mean).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimpleScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )


@dataclass
class PreprocessingArtifacts:
    """
    Inference-time metadata reconstructed from historical data.

    The training checkpoints in this repository do not include scalers or
    categorical encoders, so these artifacts recreate the original feature
    pipeline in a form that can be saved and reused for live inference.
    """

    feature_version: str
    feature_columns: List[str]
    feature_scaler: SimpleScaler
    target_scaler: SimpleScaler
    route_encoder: Dict[str, int]
    stop_encoder: Dict[str, int]
    direction_encoder: Dict[str, int]
    headway_median: float
    global_delay_mean: float
    global_delay_std: float
    route_delay_mean: Dict[str, float] = field(default_factory=dict)
    route_delay_std: Dict[str, float] = field(default_factory=dict)
    stop_delay_mean: Dict[str, float] = field(default_factory=dict)
    stop_delay_std: Dict[str, float] = field(default_factory=dict)
    hour_delay_mean: Dict[str, float] = field(default_factory=dict)
    route_hour_delay_mean: Dict[str, float] = field(default_factory=dict)
    history_maxlen: int = 32
    history_window: int = 10
    training_cutoff_year: int = 2025

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "feature_columns": list(self.feature_columns),
            "feature_scaler": self.feature_scaler.to_dict(),
            "target_scaler": self.target_scaler.to_dict(),
            "route_encoder": dict(self.route_encoder),
            "stop_encoder": dict(self.stop_encoder),
            "direction_encoder": dict(self.direction_encoder),
            "headway_median": float(self.headway_median),
            "global_delay_mean": float(self.global_delay_mean),
            "global_delay_std": float(self.global_delay_std),
            "route_delay_mean": dict(self.route_delay_mean),
            "route_delay_std": dict(self.route_delay_std),
            "stop_delay_mean": dict(self.stop_delay_mean),
            "stop_delay_std": dict(self.stop_delay_std),
            "hour_delay_mean": dict(self.hour_delay_mean),
            "route_hour_delay_mean": dict(self.route_hour_delay_mean),
            "history_maxlen": int(self.history_maxlen),
            "history_window": int(self.history_window),
            "training_cutoff_year": int(self.training_cutoff_year),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), target)
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreprocessingArtifacts":
        return cls(
            feature_version=str(payload["feature_version"]),
            feature_columns=list(payload["feature_columns"]),
            feature_scaler=SimpleScaler.from_dict(payload["feature_scaler"]),
            target_scaler=SimpleScaler.from_dict(payload["target_scaler"]),
            route_encoder={str(k): int(v) for k, v in payload["route_encoder"].items()},
            stop_encoder={str(k): int(v) for k, v in payload["stop_encoder"].items()},
            direction_encoder={str(k): int(v) for k, v in payload["direction_encoder"].items()},
            headway_median=float(payload["headway_median"]),
            global_delay_mean=float(payload["global_delay_mean"]),
            global_delay_std=float(payload["global_delay_std"]),
            route_delay_mean={str(k): float(v) for k, v in payload.get("route_delay_mean", {}).items()},
            route_delay_std={str(k): float(v) for k, v in payload.get("route_delay_std", {}).items()},
            stop_delay_mean={str(k): float(v) for k, v in payload.get("stop_delay_mean", {}).items()},
            stop_delay_std={str(k): float(v) for k, v in payload.get("stop_delay_std", {}).items()},
            hour_delay_mean={str(k): float(v) for k, v in payload.get("hour_delay_mean", {}).items()},
            route_hour_delay_mean={str(k): float(v) for k, v in payload.get("route_hour_delay_mean", {}).items()},
            history_maxlen=int(payload.get("history_maxlen", 32)),
            history_window=int(payload.get("history_window", 10)),
            training_cutoff_year=int(payload.get("training_cutoff_year", 2025)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingArtifacts":
        payload = torch.load(Path(path), map_location="cpu")
        return cls.from_dict(payload)


@dataclass
class PredictionResult:
    """Structured response for a single online prediction."""

    predicted_delay_minutes: float
    model_latency_ms: float
    feature_version: str
    checkpoint_name: str
    route_id: str
    stop_id: str
    scheduled: str
    used_history: int
    feature_values: Dict[str, float]

    def to_dict(self, include_feature_values: bool = True) -> Dict[str, Any]:
        payload = {
            "predicted_delay_minutes": round(self.predicted_delay_minutes, 6),
            "model_latency_ms": round(self.model_latency_ms, 6),
            "feature_version": self.feature_version,
            "checkpoint_name": self.checkpoint_name,
            "route_id": self.route_id,
            "stop_id": self.stop_id,
            "scheduled": self.scheduled,
            "used_history": self.used_history,
        }
        if include_feature_values:
            payload["feature_values"] = {
                key: round(value, 6) for key, value in self.feature_values.items()
            }
        return payload


class MLPPredictor(nn.Module):
    """Matches the MLP checkpoint structure used in V1 and V2."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalLSTMPredictor(nn.Module):
    """Matches the V3 LSTM checkpoint structure."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TemporalGRUPredictor(nn.Module):
    """Matches the V3 GRU checkpoint structure."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


class MBTARealtimeAdapter:
    """
    Adapter for normalizing live MBTA prediction payloads.

    The adapter is intentionally conservative: it understands the project's
    internal normalized record format and can also parse JSON:API-like payloads
    if the live endpoint returns nested `attributes` and `relationships`.
    """

    def __init__(self, base_url: str = MBTA_API_BASE_URL, endpoint: str = "predictions", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.api_key = api_key or os.getenv("MBTA_API_KEY")

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.endpoint}"

    def build_query_params(
        self,
        route_ids: Optional[Sequence[str]] = None,
        stop_ids: Optional[Sequence[str]] = None,
        direction_id: Optional[str | int] = None,
    ) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if route_ids:
            params["filter[route]"] = ",".join(str(route_id) for route_id in route_ids)
        if stop_ids:
            params["filter[stop]"] = ",".join(str(stop_id) for stop_id in stop_ids)
        if direction_id is not None:
            params["filter[direction_id]"] = str(direction_id)
        return params

    def fetch_payload(
        self,
        route_ids: Optional[Sequence[str]] = None,
        stop_ids: Optional[Sequence[str]] = None,
        direction_id: Optional[str | int] = None,
        timeout: int = 15,
        session: Optional[requests.Session] = None,
    ) -> Dict[str, Any]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        request_session = session or requests.Session()
        response = request_session.get(
            self.url,
            params=self.build_query_params(route_ids=route_ids, stop_ids=stop_ids, direction_id=direction_id),
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def normalize_records(self, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(payload, Mapping) and "data" in payload:
            items = payload["data"]
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            items = payload
        else:
            items = [payload]  # type: ignore[list-item]

        normalized: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            attributes = item.get("attributes") if isinstance(item.get("attributes"), Mapping) else {}
            relationships = item.get("relationships") if isinstance(item.get("relationships"), Mapping) else {}
            record = {
                "route_id": item.get("route_id") or self._relationship_id(relationships, "route"),
                "stop_id": item.get("stop_id") or self._relationship_id(relationships, "stop"),
                "direction_id": item.get("direction_id", attributes.get("direction_id")),
                "service_date": item.get("service_date"),
                "scheduled": item.get("scheduled")
                or attributes.get("departure_time")
                or attributes.get("arrival_time"),
                "scheduled_headway": item.get("scheduled_headway", attributes.get("scheduled_headway")),
                "observed_delay_minutes": item.get("observed_delay_minutes"),
                "actual": item.get("actual") or attributes.get("actual_time"),
                "vehicle_id": item.get("vehicle_id") or self._relationship_id(relationships, "vehicle"),
                "status": item.get("status") or attributes.get("status"),
            }
            if record["route_id"] and record["stop_id"] and record["scheduled"]:
                normalized.append(record)
        return normalized

    @staticmethod
    def _relationship_id(relationships: Mapping[str, Any], key: str) -> Optional[str]:
        relation = relationships.get(key)
        if not isinstance(relation, Mapping):
            return None
        data = relation.get("data")
        if isinstance(data, Mapping):
            relation_id = data.get("id")
            if relation_id is not None:
                return str(relation_id)
        return None


class RealtimeFeatureBuilder:
    """Builds online feature vectors that match the original training schemas."""

    def __init__(self, artifacts: PreprocessingArtifacts, state: Optional[Dict[Tuple[str, str], Deque[float]]] = None):
        self.artifacts = artifacts
        self.state = state if state is not None else defaultdict(
            lambda: deque(maxlen=self.artifacts.history_maxlen)
        )

    def build_feature_vector(self, record: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], np.ndarray]:
        normalized = self._normalize_record(record)
        route_id = normalized["route_id"]
        stop_id = normalized["stop_id"]
        scheduled = normalized["scheduled"]
        service_date = normalized["service_date"]

        hour = int(scheduled.hour)
        day_of_week = int(service_date.dayofweek)
        month = int(service_date.month)
        is_weekend = float(day_of_week >= 5)
        is_rush_hour = float((7 <= hour <= 9) or (16 <= hour <= 19))
        route_encoded = float(self._encode(route_id, self.artifacts.route_encoder))
        stop_encoded = float(self._encode(stop_id, self.artifacts.stop_encoder))
        direction_encoded = float(self._encode(normalized["direction_id"], self.artifacts.direction_encoder))
        scheduled_headway = normalized["scheduled_headway"]

        values: Dict[str, float] = {
            "hour": float(hour),
            "day_of_week": float(day_of_week),
            "month": float(month),
            "is_weekend": is_weekend,
            "is_rush_hour": is_rush_hour,
            "route_encoded": route_encoded,
            "stop_encoded": stop_encoded,
            "direction_encoded": direction_encoded,
            "scheduled_headway": float(scheduled_headway),
        }

        if self.artifacts.feature_version in {"v2", "v3"}:
            route_delay_mean = self.artifacts.route_delay_mean.get(route_id, self.artifacts.global_delay_mean)
            route_delay_std = self.artifacts.route_delay_std.get(route_id, self.artifacts.global_delay_std)
            stop_delay_mean = self.artifacts.stop_delay_mean.get(stop_id, self.artifacts.global_delay_mean)
            stop_delay_std = self.artifacts.stop_delay_std.get(stop_id, self.artifacts.global_delay_std)
            hour_delay_mean = self.artifacts.hour_delay_mean.get(str(hour), self.artifacts.global_delay_mean)
            route_hour_key = f"{route_id}_{hour}"
            route_hour_delay_mean = self.artifacts.route_hour_delay_mean.get(route_hour_key, route_delay_mean)

            values.update(
                {
                    "hour_sin": math.sin(2 * math.pi * hour / 24.0),
                    "hour_cos": math.cos(2 * math.pi * hour / 24.0),
                    "dow_sin": math.sin(2 * math.pi * day_of_week / 7.0),
                    "dow_cos": math.cos(2 * math.pi * day_of_week / 7.0),
                    "month_sin": math.sin(2 * math.pi * month / 12.0),
                    "month_cos": math.cos(2 * math.pi * month / 12.0),
                    "route_delay_mean": route_delay_mean,
                    "route_delay_std": route_delay_std,
                    "stop_delay_mean": stop_delay_mean,
                    "stop_delay_std": stop_delay_std,
                    "hour_delay_mean": hour_delay_mean,
                    "route_hour_delay_mean": route_hour_delay_mean,
                }
            )

        if self.artifacts.feature_version == "v3":
            fallback_mean = values["route_hour_delay_mean"]
            values.update(self._build_history_features(route_id, stop_id, fallback_mean))

        feature_vector = np.asarray(
            [float(values[column]) for column in self.artifacts.feature_columns],
            dtype=np.float32,
        )
        return normalized, values, feature_vector

    def ingest_observation(self, record: Mapping[str, Any]) -> Optional[float]:
        normalized = self._normalize_record(record)
        observed_delay = normalized["observed_delay_minutes"]
        if observed_delay is None or not np.isfinite(observed_delay):
            return None
        key = (normalized["route_id"], normalized["stop_id"])
        self.state[key].append(float(observed_delay))
        return float(observed_delay)

    def _build_history_features(self, route_id: str, stop_id: str, fallback_mean: float) -> Dict[str, float]:
        history = np.asarray(list(self.state[(route_id, stop_id)]), dtype=np.float32)
        if history.size == 0:
            return {
                "delay_trend": fallback_mean,
                "delay_detail": 0.0,
                "delay_denoised": fallback_mean,
                "delay_rolling_mean": fallback_mean,
                "delay_rolling_std": self.artifacts.global_delay_std,
                "delay_rolling_min": fallback_mean,
                "delay_rolling_max": fallback_mean,
                "delay_lag_1": 0.0,
                "delay_lag_2": 0.0,
                "delay_lag_3": 0.0,
                "delay_diff": 0.0,
            }

        trend, detail = _wavelet_decompose(history)
        denoised = _wavelet_denoise(history)
        rolling_window = history[-self.artifacts.history_window :]
        lag_1 = float(history[-1])
        lag_2 = float(history[-2]) if history.size >= 2 else 0.0
        lag_3 = float(history[-3]) if history.size >= 3 else 0.0
        return {
            "delay_trend": float(trend[-1]),
            "delay_detail": float(detail[-1]),
            "delay_denoised": float(denoised[-1]),
            "delay_rolling_mean": float(rolling_window.mean()),
            "delay_rolling_std": float(rolling_window.std()),
            "delay_rolling_min": float(rolling_window.min()),
            "delay_rolling_max": float(rolling_window.max()),
            "delay_lag_1": lag_1,
            "delay_lag_2": lag_2,
            "delay_lag_3": lag_3,
            "delay_diff": lag_1 - lag_2 if history.size >= 2 else 0.0,
        }

    def _normalize_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        route_id = str(record.get("route_id", "")).strip()
        stop_id = str(record.get("stop_id", "")).strip()
        if not route_id or route_id == "None":
            raise ValueError("Live record is missing route_id.")
        if not stop_id or stop_id == "None":
            raise ValueError("Live record is missing stop_id.")

        scheduled = _coerce_timestamp(record.get("scheduled"))
        if scheduled is None:
            raise ValueError("Live record is missing a valid scheduled timestamp.")
        scheduled = _strip_timezone(scheduled)

        service_date = _coerce_timestamp(record.get("service_date"))
        service_date = _strip_timezone(service_date) if service_date is not None else scheduled.normalize()

        actual = _coerce_timestamp(record.get("actual"))
        actual = _strip_timezone(actual) if actual is not None else None

        observed_delay = record.get("observed_delay_minutes")
        if observed_delay is None and actual is not None:
            observed_delay = (actual - scheduled).total_seconds() / 60.0
        observed_delay = _safe_float(observed_delay)
        if np.isnan(observed_delay):
            observed_delay = None

        direction_id = record.get("direction_id")
        if direction_id is None or str(direction_id).strip() == "":
            direction_id = "Unknown"
        direction_id = str(direction_id)

        scheduled_headway = _safe_float(record.get("scheduled_headway"), fallback=self.artifacts.headway_median)
        if np.isnan(scheduled_headway):
            scheduled_headway = self.artifacts.headway_median

        return {
            "route_id": route_id,
            "stop_id": stop_id,
            "direction_id": direction_id,
            "service_date": service_date.normalize(),
            "scheduled": scheduled,
            "actual": actual,
            "scheduled_headway": float(scheduled_headway),
            "observed_delay_minutes": observed_delay,
        }

    @staticmethod
    def _encode(value: str, mapping: Mapping[str, int]) -> int:
        if value in mapping:
            return mapping[value]
        return len(mapping)


class RealtimeDelayPredictor:
    """Loads an existing checkpoint and exposes online prediction helpers."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_spec: ModelSpec,
        model: nn.Module,
        artifacts: PreprocessingArtifacts,
        device: torch.device,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_spec = model_spec
        self.model = model.to(device).eval()
        self.artifacts = artifacts
        self.device = device
        self.state: Dict[Tuple[str, str], Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.artifacts.history_maxlen)
        )
        self.feature_builder = RealtimeFeatureBuilder(self.artifacts, self.state)

    @classmethod
    def from_artifacts(
        cls,
        checkpoint_path: str | Path,
        artifacts: PreprocessingArtifacts,
        device: Optional[str] = None,
    ) -> "RealtimeDelayPredictor":
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_spec = infer_model_spec(checkpoint_path, checkpoint)
        if model_spec.feature_version != artifacts.feature_version:
            raise ValueError(
                f"Checkpoint feature version '{model_spec.feature_version}' does not match "
                f"artifacts '{artifacts.feature_version}'."
            )
        model = build_model_from_spec(model_spec)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            checkpoint_path=checkpoint_path,
            model_spec=model_spec,
            model=model,
            artifacts=artifacts,
            device=_resolve_device(device),
        )

    @classmethod
    def from_artifacts_file(
        cls,
        checkpoint_path: str | Path,
        artifacts_path: str | Path,
        device: Optional[str] = None,
    ) -> "RealtimeDelayPredictor":
        return cls.from_artifacts(
            checkpoint_path=checkpoint_path,
            artifacts=PreprocessingArtifacts.load(artifacts_path),
            device=device,
        )

    @classmethod
    def bootstrap(
        cls,
        checkpoint_path: str | Path,
        historical_data: pd.DataFrame | str | Path,
        artifacts_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        training_cutoff_year: int = 2025,
        sample_size: int = 50000,
    ) -> "RealtimeDelayPredictor":
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_spec = infer_model_spec(checkpoint_path, checkpoint)
        artifacts = build_preprocessing_artifacts(
            historical_data=historical_data,
            feature_version=model_spec.feature_version,
            training_cutoff_year=training_cutoff_year,
            sample_size=sample_size,
        )
        if artifacts_path:
            artifacts.save(artifacts_path)
        model = build_model_from_spec(model_spec)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictor = cls(
            checkpoint_path=checkpoint_path,
            model_spec=model_spec,
            model=model,
            artifacts=artifacts,
            device=_resolve_device(device),
        )
        predictor.warm_state_from_historical_data(historical_data)
        return predictor

    def reset_state(self) -> None:
        self.state = defaultdict(lambda: deque(maxlen=self.artifacts.history_maxlen))
        self.feature_builder.state = self.state

    def warm_state_from_historical_data(self, historical_data: pd.DataFrame | str | Path) -> None:
        history_df = prepare_historical_frame(
            load_historical_frame(historical_data),
            training_cutoff_year=None,
        )
        if history_df.empty:
            return

        seed_df = (
            history_df.sort_values(["scheduled", "route_id", "stop_id"])
            .groupby(["route_id", "stop_id"], group_keys=False)
            .tail(self.artifacts.history_maxlen)
            .sort_values(["scheduled", "route_id", "stop_id"])
        )

        self.reset_state()
        for row in seed_df.to_dict("records"):
            self.feature_builder.ingest_observation(
                {
                    "route_id": row["route_id"],
                    "stop_id": row["stop_id"],
                    "direction_id": row.get("direction_id", "Unknown"),
                    "service_date": row["service_date"],
                    "scheduled": row["scheduled"],
                    "scheduled_headway": row.get("scheduled_headway"),
                    "observed_delay_minutes": row["delay_minutes"],
                }
            )

    def ingest_observation(self, record: Mapping[str, Any]) -> Optional[float]:
        return self.feature_builder.ingest_observation(record)

    def predict_one(self, record: Mapping[str, Any]) -> PredictionResult:
        normalized, feature_values, feature_vector = self.feature_builder.build_feature_vector(record)
        scaled_features = self.artifacts.feature_scaler.transform(feature_vector).astype(np.float32)
        feature_tensor = torch.from_numpy(scaled_features).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            scaled_prediction = self.model(feature_tensor).detach().cpu().numpy()
        model_latency_ms = (time.perf_counter() - start) * 1000.0

        prediction = self.artifacts.target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0]
        used_history = len(self.state[(normalized["route_id"], normalized["stop_id"])])
        return PredictionResult(
            predicted_delay_minutes=float(prediction),
            model_latency_ms=float(model_latency_ms),
            feature_version=self.model_spec.feature_version,
            checkpoint_name=self.checkpoint_path.name,
            route_id=normalized["route_id"],
            stop_id=normalized["stop_id"],
            scheduled=normalized["scheduled"].isoformat(),
            used_history=used_history,
            feature_values=feature_values,
        )

    def predict_many(
        self,
        records: Iterable[Mapping[str, Any]],
        ingest_observations: bool = False,
        use_predictions_as_history: bool = False,
    ) -> List[PredictionResult]:
        results: List[PredictionResult] = []
        for record in records:
            result = self.predict_one(record)
            results.append(result)
            if ingest_observations:
                self.ingest_observation(record)
            elif use_predictions_as_history:
                synthetic_record = dict(record)
                synthetic_record["observed_delay_minutes"] = result.predicted_delay_minutes
                self.ingest_observation(synthetic_record)
        return results

    def predict_from_payload(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        adapter: Optional[MBTARealtimeAdapter] = None,
        ingest_observations: bool = False,
        use_predictions_as_history: bool = False,
    ) -> List[PredictionResult]:
        source_adapter = adapter or MBTARealtimeAdapter()
        records = source_adapter.normalize_records(payload)
        return self.predict_many(
            records,
            ingest_observations=ingest_observations,
            use_predictions_as_history=use_predictions_as_history,
        )


def infer_model_spec(checkpoint_path: str | Path, checkpoint: Optional[Mapping[str, Any]] = None) -> ModelSpec:
    path = Path(checkpoint_path)
    if path.name in KNOWN_MODEL_SPECS:
        return KNOWN_MODEL_SPECS[path.name]

    loaded_checkpoint = checkpoint or torch.load(path, map_location="cpu")
    state_dict = loaded_checkpoint["model_state_dict"]

    if "gru.weight_ih_l0" in state_dict:
        input_size = int(state_dict["input_proj.weight"].shape[1])
        hidden_size = int(state_dict["input_proj.weight"].shape[0])
        num_layers = _count_recurrent_layers(state_dict, "gru.weight_ih_l")
        return ModelSpec(path.name, "gru", _feature_version_from_input_size(input_size), input_size, hidden_size=hidden_size, num_layers=num_layers)

    if "lstm.weight_ih_l0" in state_dict:
        input_size = int(state_dict["input_proj.weight"].shape[1])
        hidden_size = int(state_dict["input_proj.weight"].shape[0])
        num_layers = _count_recurrent_layers(state_dict, "lstm.weight_ih_l")
        return ModelSpec(path.name, "lstm", _feature_version_from_input_size(input_size), input_size, hidden_size=hidden_size, num_layers=num_layers)

    linear_shapes = [
        tensor.shape for name, tensor in state_dict.items() if name.startswith("network.") and tensor.ndim == 2
    ]
    if linear_shapes:
        input_size = int(linear_shapes[0][1])
        hidden_sizes = tuple(int(shape[0]) for shape in linear_shapes[:-1])
        return ModelSpec(path.name, "mlp", _feature_version_from_input_size(input_size), input_size, hidden_sizes=hidden_sizes)

    raise ValueError(f"Unable to infer model spec from checkpoint: {path}")


def build_model_from_spec(model_spec: ModelSpec) -> nn.Module:
    if model_spec.model_family == "mlp":
        return MLPPredictor(input_size=model_spec.input_size, hidden_sizes=model_spec.hidden_sizes)
    if model_spec.model_family == "lstm":
        return TemporalLSTMPredictor(
            input_size=model_spec.input_size,
            hidden_size=model_spec.hidden_size,
            num_layers=model_spec.num_layers,
        )
    if model_spec.model_family == "gru":
        return TemporalGRUPredictor(
            input_size=model_spec.input_size,
            hidden_size=model_spec.hidden_size,
            num_layers=model_spec.num_layers,
        )
    raise ValueError(f"Unsupported model family: {model_spec.model_family}")


def build_preprocessing_artifacts(
    historical_data: pd.DataFrame | str | Path,
    feature_version: str,
    training_cutoff_year: int = 2025,
    sample_size: int = 50000,
) -> PreprocessingArtifacts:
    if feature_version not in FEATURE_COLUMNS_BY_VERSION:
        raise ValueError(f"Unsupported feature version: {feature_version}")

    history_df = prepare_historical_frame(
        load_historical_frame(historical_data),
        training_cutoff_year=training_cutoff_year,
    )
    if history_df.empty:
        raise ValueError("Historical data did not produce any rows after preprocessing.")

    if sample_size and len(history_df) > sample_size:
        history_df = history_df.sample(n=sample_size, random_state=42)

    history_df = history_df.sort_values(["scheduled", "route_id", "stop_id"]).reset_index(drop=True)
    headway_series = pd.to_numeric(history_df["scheduled_headway"], errors="coerce")
    headway_median = float(headway_series.median()) if not headway_series.dropna().empty else 0.0
    global_delay_mean = float(history_df["delay_minutes"].mean())
    global_delay_std = float(history_df["delay_minutes"].std(ddof=0))
    if not np.isfinite(global_delay_std) or global_delay_std <= 0:
        global_delay_std = 1.0

    route_delay_stats = history_df.groupby("route_id")["delay_minutes"].agg(["mean", "std"]).fillna(0.0)
    stop_delay_stats = history_df.groupby("stop_id")["delay_minutes"].agg(["mean", "std"]).fillna(0.0)
    history_df["hour"] = history_df["scheduled"].dt.hour
    hour_delay_mean = history_df.groupby("hour")["delay_minutes"].mean()
    history_df["route_hour_key"] = history_df["route_id"].astype(str) + "_" + history_df["hour"].astype(str)
    route_hour_delay_mean = history_df.groupby("route_hour_key")["delay_minutes"].mean()

    artifacts = PreprocessingArtifacts(
        feature_version=feature_version,
        feature_columns=list(FEATURE_COLUMNS_BY_VERSION[feature_version]),
        feature_scaler=SimpleScaler(
            mean=np.zeros(len(FEATURE_COLUMNS_BY_VERSION[feature_version]), dtype=np.float32),
            scale=np.ones(len(FEATURE_COLUMNS_BY_VERSION[feature_version]), dtype=np.float32),
        ),
        target_scaler=SimpleScaler(mean=np.zeros(1, dtype=np.float32), scale=np.ones(1, dtype=np.float32)),
        route_encoder=_build_category_mapping(history_df["route_id"]),
        stop_encoder=_build_category_mapping(history_df["stop_id"]),
        direction_encoder=_build_category_mapping(history_df["direction_id"].fillna("Unknown")),
        headway_median=headway_median,
        global_delay_mean=global_delay_mean,
        global_delay_std=global_delay_std,
        route_delay_mean={str(idx): float(row["mean"]) for idx, row in route_delay_stats.iterrows()},
        route_delay_std={
            str(idx): float(row["std"]) if float(row["std"]) > 0 else global_delay_std
            for idx, row in route_delay_stats.iterrows()
        },
        stop_delay_mean={str(idx): float(row["mean"]) for idx, row in stop_delay_stats.iterrows()},
        stop_delay_std={
            str(idx): float(row["std"]) if float(row["std"]) > 0 else global_delay_std
            for idx, row in stop_delay_stats.iterrows()
        },
        hour_delay_mean={str(idx): float(value) for idx, value in hour_delay_mean.items()},
        route_hour_delay_mean={str(idx): float(value) for idx, value in route_hour_delay_mean.items()},
        training_cutoff_year=training_cutoff_year,
    )

    feature_builder = RealtimeFeatureBuilder(artifacts)
    feature_rows: List[np.ndarray] = []
    target_rows: List[List[float]] = []
    for row in history_df.to_dict("records"):
        _, _, feature_vector = feature_builder.build_feature_vector(
            {
                "route_id": row["route_id"],
                "stop_id": row["stop_id"],
                "direction_id": row.get("direction_id", "Unknown"),
                "service_date": row["service_date"],
                "scheduled": row["scheduled"],
                "scheduled_headway": row.get("scheduled_headway"),
            }
        )
        feature_rows.append(feature_vector)
        target_rows.append([float(row["delay_minutes"])])
        feature_builder.ingest_observation(
            {
                "route_id": row["route_id"],
                "stop_id": row["stop_id"],
                "direction_id": row.get("direction_id", "Unknown"),
                "service_date": row["service_date"],
                "scheduled": row["scheduled"],
                "scheduled_headway": row.get("scheduled_headway"),
                "observed_delay_minutes": row["delay_minutes"],
            }
        )

    feature_matrix = np.vstack(feature_rows)
    target_matrix = np.asarray(target_rows, dtype=np.float32)
    artifacts.feature_scaler = SimpleScaler.fit(feature_matrix)
    artifacts.target_scaler = SimpleScaler.fit(target_matrix)
    return artifacts


def load_historical_frame(historical_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(historical_data, pd.DataFrame):
        return historical_data.copy()

    path = Path(historical_data)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported historical data file type: {path.suffix}")


def prepare_historical_frame(
    history_df: pd.DataFrame,
    training_cutoff_year: Optional[int] = 2025,
) -> pd.DataFrame:
    required_columns = {"route_id", "stop_id", "scheduled"}
    missing_columns = required_columns.difference(history_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Historical data is missing required columns: {missing_list}")

    df = history_df.copy()
    df["scheduled"] = pd.to_datetime(df["scheduled"], errors="coerce")
    if "service_date" in df.columns:
        df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    else:
        df["service_date"] = df["scheduled"]
    if "actual" in df.columns:
        df["actual"] = pd.to_datetime(df["actual"], errors="coerce")
    if "scheduled_headway" not in df.columns:
        df["scheduled_headway"] = np.nan
    if "direction_id" not in df.columns:
        df["direction_id"] = "Unknown"
    else:
        df["direction_id"] = df["direction_id"].fillna("Unknown").astype(str)

    if "delay_minutes" not in df.columns:
        if "actual" not in df.columns:
            raise ValueError("Historical data must include either delay_minutes or actual.")
        df["delay_minutes"] = (df["actual"] - df["scheduled"]).dt.total_seconds() / 60.0

    df = df.dropna(subset=["scheduled", "service_date", "delay_minutes"])
    df["scheduled"] = pd.to_datetime(df["scheduled"].apply(_strip_timezone), errors="coerce")
    df["service_date"] = pd.to_datetime(
        df["service_date"].apply(_strip_timezone),
        errors="coerce",
    ).dt.normalize()
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")
    df["scheduled_headway"] = pd.to_numeric(df["scheduled_headway"], errors="coerce")
    df = df.dropna(subset=["delay_minutes"])
    df = df[(df["delay_minutes"] >= -30) & (df["delay_minutes"] <= 60)]
    if training_cutoff_year is not None:
        df = df[df["service_date"].dt.year < training_cutoff_year]
    return df.reset_index(drop=True)


def benchmark_predictor(
    predictor: RealtimeDelayPredictor,
    records: Sequence[Mapping[str, Any]],
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    if not records:
        raise ValueError("At least one live record is required for benchmarking.")

    total_latencies: List[float] = []
    model_latencies: List[float] = []

    for _ in range(max(warmup, 0)):
        for record in records:
            predictor.predict_one(record)

    for _ in range(max(iterations, 1)):
        for record in records:
            start = time.perf_counter()
            result = predictor.predict_one(record)
            total_latencies.append((time.perf_counter() - start) * 1000.0)
            model_latencies.append(result.model_latency_ms)

    total_array = np.asarray(total_latencies, dtype=np.float64)
    model_array = np.asarray(model_latencies, dtype=np.float64)
    total_calls = float(total_array.size)
    elapsed_ms = float(total_array.sum())
    return {
        "calls": total_calls,
        "avg_latency_ms": float(total_array.mean()),
        "p50_latency_ms": float(np.percentile(total_array, 50)),
        "p95_latency_ms": float(np.percentile(total_array, 95)),
        "max_latency_ms": float(total_array.max()),
        "avg_model_latency_ms": float(model_array.mean()),
        "throughput_qps": float(total_calls / max(elapsed_ms / 1000.0, 1e-9)),
    }


def format_benchmark_summary(summary: Mapping[str, float]) -> str:
    return (
        "Baseline Inference Latency\n"
        f"calls: {int(summary['calls'])}\n"
        f"avg_latency_ms: {summary['avg_latency_ms']:.6f}\n"
        f"p50_latency_ms: {summary['p50_latency_ms']:.6f}\n"
        f"p95_latency_ms: {summary['p95_latency_ms']:.6f}\n"
        f"max_latency_ms: {summary['max_latency_ms']:.6f}\n"
        f"avg_model_latency_ms: {summary['avg_model_latency_ms']:.6f}\n"
        f"throughput_qps: {summary['throughput_qps']:.2f}"
    )


def build_demo_historical_frame(num_rows: int = 256) -> pd.DataFrame:
    """Small deterministic frame for CI and smoke tests."""

    base_time = pd.Timestamp("2024-01-02 06:00:00")
    routes = ["22", "28", "111"]
    stops = ["70091", "70092", "70093", "70094"]
    rows: List[Dict[str, Any]] = []
    for index in range(num_rows):
        route_id = routes[index % len(routes)]
        stop_id = stops[(index // len(routes)) % len(stops)]
        scheduled = base_time + pd.Timedelta(minutes=15 * index)
        delay_minutes = 4.0 + 2.25 * math.sin(index / 6.0) + ((index % 5) - 2) * 0.35
        actual = scheduled + pd.Timedelta(minutes=float(delay_minutes))
        rows.append(
            {
                "service_date": scheduled.normalize(),
                "route_id": route_id,
                "stop_id": stop_id,
                "direction_id": str(index % 2),
                "scheduled": scheduled,
                "actual": actual,
                "scheduled_headway": 10.0 + (index % 4),
                "delay_minutes": float(delay_minutes),
            }
        )
    return pd.DataFrame(rows)


def build_demo_live_records(count: int = 12) -> List[Dict[str, Any]]:
    """Sample online records for local testing and latency benchmarks."""

    base_time = pd.Timestamp("2026-01-06 07:00:00")
    routes = ["22", "28", "111"]
    stops = ["70091", "70092", "70093", "70094"]
    records: List[Dict[str, Any]] = []
    for index in range(count):
        scheduled = base_time + pd.Timedelta(minutes=12 * index)
        records.append(
            {
                "service_date": scheduled.normalize(),
                "route_id": routes[index % len(routes)],
                "stop_id": stops[(index // len(routes)) % len(stops)],
                "direction_id": str(index % 2),
                "scheduled": scheduled,
                "scheduled_headway": 9.0 + (index % 3),
            }
        )
    return records


def _build_category_mapping(values: pd.Series) -> Dict[str, int]:
    unique_values = pd.Series(values).fillna("Unknown").astype(str).unique()
    return {str(value): index for index, value in enumerate(unique_values)}


def _coerce_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def _safe_float(value: Any, fallback: float = np.nan) -> float:
    if value is None or value == "":
        return float(fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _strip_timezone(timestamp: pd.Timestamp) -> pd.Timestamp:
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def _wavelet_decompose(signal: np.ndarray, wavelet: str = "db4", level: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if signal.size < 8:
        return signal.astype(np.float32), np.zeros_like(signal, dtype=np.float32)

    if not HAS_PYWT:
        trend = pd.Series(signal).rolling(window=8, min_periods=1).mean().to_numpy(dtype=np.float32)
        detail = signal.astype(np.float32) - trend
        return trend, detail

    padded_len = 2 ** int(np.ceil(np.log2(signal.size)))
    padded = np.pad(signal, (0, padded_len - signal.size), mode="reflect")
    coeffs = pywt.wavedec(padded, wavelet, level=level)
    approx_coeffs = [coeffs[0]] + [np.zeros_like(component) for component in coeffs[1:]]
    trend = pywt.waverec(approx_coeffs, wavelet)[: signal.size]
    detail_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
    detail = pywt.waverec(detail_coeffs, wavelet)[: signal.size]
    return trend.astype(np.float32), detail.astype(np.float32)


def _wavelet_denoise(signal: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    if signal.size < 8:
        return signal.astype(np.float32)

    if not HAS_PYWT:
        return pd.Series(signal).rolling(window=5, min_periods=1).mean().to_numpy(dtype=np.float32)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]]
    for component in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(component, threshold, mode="soft"))
    return pywt.waverec(denoised_coeffs, wavelet)[: signal.size].astype(np.float32)


def _feature_version_from_input_size(input_size: int) -> str:
    if input_size == 9:
        return "v1"
    if input_size == 18:
        return "v2"
    if input_size == 28:
        return "v3"
    raise ValueError(f"Unsupported input size for known feature sets: {input_size}")


def _count_recurrent_layers(state_dict: Mapping[str, torch.Tensor], prefix: str) -> int:
    layer_indices = {
        int(name.split(prefix, 1)[1])
        for name in state_dict
        if name.startswith(prefix)
    }
    return max(layer_indices) + 1 if layer_indices else 1


def _resolve_device(device: Optional[str]) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_BASELINE_CHECKPOINT = DEFAULT_MODEL_DIR / "delay_predictor_mlp_v1_baseline_temporal.pt"
