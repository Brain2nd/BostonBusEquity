from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.build_v5_residual_dataset import build_residual_dataset
from src.inference.log_mbta_live_snapshots import merge_prediction_vehicle_fields
from src.inference.mbta_v3_client import normalize_prediction_payload, normalize_vehicle_payload
from src.inference.runtime import DelayPredictorRuntime, PredictionInputError
from src.models.v4_delay_predictor import (
    V4_FEATURE_COLUMNS,
    add_v4_time_features,
    build_v4_bundle,
    build_v4_dataset,
    build_v4_feature_frame_from_dataframe,
    normalize_v4_dataframe,
    save_v4_bundle,
)


def _make_v4_dataframe() -> pd.DataFrame:
    rows = []
    for year in [2024, 2025, 2026]:
        for trip_number in range(4):
            route_id = "1" if trip_number % 2 == 0 else "2"
            direction_id = str(trip_number % 2)
            base_time = pd.Timestamp(
                year=year,
                month=1,
                day=3 + trip_number,
                hour=8 + trip_number,
                minute=0,
                tz="UTC",
            )
            half_trip_id = f"{year}-trip-{trip_number}"
            for stop_order, stop_id in enumerate(["A", "B", "C"], start=1):
                scheduled = base_time + pd.Timedelta(minutes=10 * (stop_order - 1))
                delay = (
                    0.4 * trip_number
                    + 0.6 * stop_order
                    + (1.0 if route_id == "2" else 0.0)
                )
                rows.append(
                    {
                        "service_date": scheduled.date().isoformat(),
                        "route_id": route_id,
                        "stop_id": stop_id,
                        "direction_id": direction_id,
                        "half_trip_id": half_trip_id,
                        "time_point_order": stop_order,
                        "scheduled": scheduled.isoformat(),
                        "actual": (scheduled + pd.Timedelta(minutes=delay)).isoformat(),
                        "scheduled_headway": 12,
                        "year": year,
                    }
                )
    return pd.DataFrame(rows)


def _make_prediction_payload() -> dict:
    return {
        "data": [
            {
                "id": "prediction-1",
                "type": "prediction",
                "attributes": {
                    "arrival_time": "2026-01-03T08:04:00Z",
                    "departure_time": None,
                    "direction_id": 0,
                    "status": "4 min",
                    "schedule_relationship": None,
                },
                "relationships": {
                    "route": {"data": {"id": "1", "type": "route"}},
                    "stop": {"data": {"id": "A", "type": "stop"}},
                    "trip": {"data": {"id": "trip-1", "type": "trip"}},
                    "vehicle": {"data": {"id": "vehicle-1", "type": "vehicle"}},
                    "schedule": {"data": {"id": "schedule-1", "type": "schedule"}},
                },
            }
        ],
        "included": [
            {
                "id": "schedule-1",
                "type": "schedule",
                "attributes": {
                    "arrival_time": "2026-01-03T08:00:00Z",
                    "departure_time": None,
                },
            }
        ],
    }


def _make_vehicle_payload() -> dict:
    return {
        "data": [
            {
                "id": "vehicle-1",
                "type": "vehicle",
                "attributes": {
                    "current_status": "IN_TRANSIT_TO",
                    "current_stop_sequence": 3,
                    "latitude": 42.35,
                    "longitude": -71.06,
                    "bearing": 90,
                    "speed": 5.5,
                    "updated_at": "2026-01-03T07:59:30Z",
                },
                "relationships": {
                    "route": {"data": {"id": "1", "type": "route"}},
                    "stop": {"data": {"id": "A", "type": "stop"}},
                    "trip": {"data": {"id": "trip-1", "type": "trip"}},
                },
            }
        ],
        "included": [
            {
                "id": "trip-1",
                "type": "trip",
                "attributes": {"direction_id": 0},
            }
        ],
    }


@pytest.fixture()
def v4_bundle_path(tmp_path: Path) -> Path:
    dataset = build_v4_dataset(
        dataframe=_make_v4_dataframe(),
        random_state=7,
    )
    model = DummyRegressor(strategy="mean")
    model.fit(
        dataset.train[dataset.feature_columns],
        dataset.train[dataset.target_column],
    )
    metrics = {
        "model_kind": "dummy_test_regressor",
        "train": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
        "validation": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
        "test": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
    }
    bundle = build_v4_bundle(model=model, dataset=dataset, metrics=metrics)
    return save_v4_bundle(bundle, tmp_path / "delay_predictor_v4_tree_realtime_bundle.joblib")


def test_v4_trip_history_uses_only_prior_stops() -> None:
    normalized = normalize_v4_dataframe(_make_v4_dataframe())
    train = normalized[normalized["year"].eq(2024)].copy()
    train = add_v4_time_features(train)
    features, _, stats = build_v4_feature_frame_from_dataframe(train)

    trip = (
        features[features["half_trip_id"].eq("2024-trip-0")]
        .sort_values("time_point_order")
        .reset_index(drop=True)
    )
    assert math.isclose(
        float(trip.loc[0, "trip_delay_lag_1"]),
        float(stats["global_mean"]),
    )
    assert math.isclose(
        float(trip.loc[1, "trip_delay_lag_1"]),
        float(trip.loc[0, "delay_minutes"]),
    )
    assert math.isclose(
        float(trip.loc[2, "trip_delay_lag_2"]),
        float(trip.loc[0, "delay_minutes"]),
    )
    assert math.isclose(
        float(trip.loc[2, "trip_delay_rolling_mean"]),
        float(trip.loc[:1, "delay_minutes"].mean()),
    )


def test_v4_runtime_predicts_and_rejects_unknown_route(v4_bundle_path: Path) -> None:
    runtime = DelayPredictorRuntime.from_bundle_path(v4_bundle_path)
    health = runtime.health()

    assert health["model"] == "V4Tree"
    assert health["feature_version"] == "v4_causal_trip_history"

    prediction = runtime.predict(
        route_id="1",
        stop_id="A",
        scheduled_time="2026-01-03T08:00:00",
        scheduled_headway=None,
        direction_id=None,
        current_stop_sequence=3,
        vehicle_speed=5.5,
    )
    assert math.isfinite(prediction["predicted_delay_minutes"])
    assert "scheduled_headway" in prediction["used_defaults"]
    assert "direction_id" in prediction["used_defaults"]
    assert "trip_history" in prediction["used_defaults"]

    feature_frame, _ = runtime._build_v4_feature_frame(
        route_id="1",
        stop_id="A",
        scheduled_time="2026-01-03T08:00:00",
        scheduled_headway=10,
        direction_id="0",
    )
    assert math.isclose(float(feature_frame.iloc[0]["scheduled_headway"]), 600.0)

    baseline = runtime.historical_baseline_delay(
        route_id="1",
        stop_id="A",
        scheduled_time="2026-01-03T08:00:00",
    )
    assert math.isfinite(baseline["predicted_delay_minutes"])
    assert baseline["source"] in {"route_stop_hour", "route_stop", "route_hour"}

    with pytest.raises(PredictionInputError):
        runtime.predict(
            route_id="missing",
            stop_id="A",
            scheduled_time="2026-01-03T08:00:00",
        )


def test_v4_api_keeps_old_shape_and_accepts_live_fields(v4_bundle_path: Path) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    from src.inference.api import create_app

    async def _run_checks() -> None:
        transport = httpx.ASGITransport(app=create_app(v4_bundle_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["model"] == "V4Tree"

            legacy_response = await client.post(
                "/predict",
                json={
                    "route_id": "1",
                    "stop_id": "A",
                    "scheduled_time": "2026-01-03T08:00:00",
                },
            )
            assert legacy_response.status_code == 200

            live_response = await client.post(
                "/predict",
                json={
                    "route_id": "1",
                    "stop_id": "A",
                    "scheduled_time": "2026-01-03T08:00:00",
                    "scheduled_headway": 12,
                    "direction_id": "0",
                    "trip_id": "trip-1",
                    "vehicle_id": "vehicle-1",
                    "current_stop_sequence": 3,
                    "vehicle_speed": 5.5,
                    "vehicle_status": "IN_TRANSIT_TO",
                    "official_predicted_delay_minutes": 4.0,
                    "official_prediction_age_seconds": 15,
                },
            )
            assert live_response.status_code == 200
            assert math.isfinite(live_response.json()["predicted_delay_minutes"])

    asyncio.run(_run_checks())


def test_vehicle_payload_normalization_and_merge() -> None:
    observed_at = pd.Timestamp("2026-01-03T03:00:00-05:00")
    predictions = normalize_prediction_payload(_make_prediction_payload(), observed_at)
    vehicles = normalize_vehicle_payload(_make_vehicle_payload(), observed_at)
    merged = merge_prediction_vehicle_fields(predictions, vehicles)

    assert len(vehicles) == 1
    assert int(vehicles.iloc[0]["current_stop_sequence"]) == 3
    assert math.isclose(float(merged.iloc[0]["speed"]), 5.5)
    assert int(merged.iloc[0]["current_stop_sequence"]) == 3


def test_v5_residual_dataset_matches_live_snapshot_to_actual(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    predictions = normalize_prediction_payload(
        _make_prediction_payload(),
        observed_at=pd.Timestamp("2026-01-03T03:00:00-05:00"),
    )
    vehicles = normalize_vehicle_payload(
        _make_vehicle_payload(),
        observed_at=pd.Timestamp("2026-01-03T03:00:00-05:00"),
    )
    merged = merge_prediction_vehicle_fields(predictions, vehicles)
    merged.to_parquet(snapshot_dir / "merged_test.parquet", index=False)

    actuals = pd.DataFrame(
        [
            {
                "route_id": "1",
                "stop_id": "A",
                "scheduled": "2026-01-03T08:00:00Z",
                "actual": "2026-01-03T08:05:00Z",
            }
        ]
    )
    processed_path = tmp_path / "arrival_departure.parquet"
    actuals.to_parquet(processed_path, index=False)

    residual = build_residual_dataset(
        snapshot_dir=snapshot_dir,
        processed_path=processed_path,
    )

    assert len(residual) == 1
    assert math.isclose(float(residual.iloc[0]["actual_delay_minutes"]), 5.0)
    assert math.isclose(float(residual.iloc[0]["official_delay_minutes"]), 4.0)
    assert math.isclose(float(residual.iloc[0]["official_residual_label"]), 1.0)
    assert V4_FEATURE_COLUMNS
