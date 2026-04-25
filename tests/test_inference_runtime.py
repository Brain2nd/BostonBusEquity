import math

import pytest

torch = pytest.importorskip("torch")

from src.inference.runtime import PredictRequest, RealtimeDelayPredictor
from src.models.v2_mlp import V2MLPPredictor


def make_bundle():
    model = V2MLPPredictor(input_size=18, hidden_sizes=[1], dropout=0.0)
    return {
        "bundle_version": 1,
        "model_name": "V2MLPPredictor",
        "experiment": "test",
        "feature_version": "test",
        "feature_columns": [
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
        ],
        "model_config": {"input_size": 18, "hidden_sizes": [1], "dropout": 0.0},
        "model_state_dict": model.state_dict(),
        "scaler_X": {"mean": [0.0] * 18, "scale": [1.0] * 18},
        "scaler_y": {"mean": [0.0], "scale": [1.0]},
        "mappings": {
            "route_id": {"1": 0},
            "stop_id": {"64": 0},
            "direction_id": {"0": 0, "Unknown": 1},
        },
        "statistics": {
            "route_delay_mean": {"1": 1.0},
            "route_delay_std": {"1": 2.0},
            "stop_delay_mean": {"64": 1.5},
            "stop_delay_std": {"64": 2.5},
            "hour_delay_mean": {"12": 3.0},
            "route_hour_delay_mean": {"1_12": 4.0},
            "global_mean": 1.0,
            "global_std": 1.0,
            "scheduled_headway_median": 10.0,
        },
    }


def test_runtime_predicts_finite_value():
    predictor = RealtimeDelayPredictor(make_bundle())
    response = predictor.predict(
        PredictRequest(
            route_id="1",
            stop_id="64",
            direction_id="0",
            scheduled_time="2026-04-24T12:00:00-04:00",
            scheduled_headway=8.0,
        )
    )
    assert math.isfinite(response["predicted_delay_minutes"])


def test_runtime_uses_default_values():
    predictor = RealtimeDelayPredictor(make_bundle())
    response = predictor.predict(
        PredictRequest(
            route_id="1",
            stop_id="64",
            scheduled_time="2026-04-24T12:00:00",
        )
    )
    assert "scheduled_headway" in response["used_defaults"]
    assert "direction_id" in response["used_defaults"]
