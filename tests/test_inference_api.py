import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient

from src.inference.api import create_app


def test_health_endpoint():
    import torch
    from pathlib import Path
    from src.models.v2_mlp import V2MLPPredictor

    model = V2MLPPredictor(input_size=18, hidden_sizes=[128, 64, 32], dropout=0.2)
    bundle = {
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
        "model_config": {"input_size": 18, "hidden_sizes": [128, 64, 32], "dropout": 0.2},
        "model_state_dict": model.state_dict(),
        "scaler_X": {"mean": [0.0] * 18, "scale": [1.0] * 18},
        "scaler_y": {"mean": [0.0], "scale": [1.0]},
        "mappings": {"route_id": {"1": 0}, "stop_id": {"64": 0}, "direction_id": {"Unknown": 0}},
        "statistics": {
            "route_delay_mean": {"1": 1.0},
            "route_delay_std": {"1": 1.0},
            "stop_delay_mean": {"64": 1.0},
            "stop_delay_std": {"64": 1.0},
            "hour_delay_mean": {"12": 1.0},
            "route_hour_delay_mean": {"1_12": 1.0},
            "global_mean": 1.0,
            "global_std": 1.0,
            "scheduled_headway_median": 10.0,
        },
    }
    bundle_path = Path("models/test_api_bundle.pt")
    torch.save(bundle, bundle_path)

    try:
        client = TestClient(create_app(bundle_path))
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["loaded"] is True
    finally:
        if bundle_path.exists():
            bundle_path.unlink()
