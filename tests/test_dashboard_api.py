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

from src.models.v4_delay_predictor import build_v4_bundle, build_v4_dataset, save_v4_bundle


def _make_dashboard_dataframe() -> pd.DataFrame:
    rows = []
    for year in [2024, 2025, 2026]:
        for route_id in ["1", "2"]:
            for stop_order, stop_id in enumerate(["A", "B", "C"], start=1):
                scheduled = pd.Timestamp(
                    year=year,
                    month=1,
                    day=stop_order + 2,
                    hour=8,
                    minute=10 * stop_order,
                    tz="UTC",
                )
                delay = 1.0 + stop_order * 0.5 + (0.5 if route_id == "2" else 0.0)
                rows.append(
                    {
                        "service_date": scheduled.date().isoformat(),
                        "route_id": route_id,
                        "stop_id": stop_id,
                        "direction_id": "0",
                        "half_trip_id": f"{year}-{route_id}-trip",
                        "time_point_order": stop_order,
                        "scheduled": scheduled.isoformat(),
                        "actual": (scheduled + pd.Timedelta(minutes=delay)).isoformat(),
                        "scheduled_headway": 10,
                        "year": year,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def dashboard_bundle_path(tmp_path: Path) -> Path:
    dataset = build_v4_dataset(_make_dashboard_dataframe(), random_state=11)
    model = DummyRegressor(strategy="mean")
    model.fit(dataset.train[dataset.feature_columns], dataset.train[dataset.target_column])
    metrics = {
        "model_kind": "dummy_test_regressor",
        "train": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
        "validation": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
        "test": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
    }
    return save_v4_bundle(
        build_v4_bundle(model=model, dataset=dataset, metrics=metrics),
        tmp_path / "dashboard_bundle.joblib",
    )


def test_dashboard_html_and_json_endpoints(dashboard_bundle_path: Path) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    async def _run_checks() -> None:
        transport = httpx.ASGITransport(app=create_app(dashboard_bundle_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            page = await client.get("/")
            assert page.status_code == 200
            assert "Boston Bus Equity" in page.text
            assert 'select name="route_id"' in page.text
            assert 'select name="stop_id"' in page.text

            summary = await client.get("/api/project-summary")
            assert summary.status_code == 200
            summary_json = summary.json()
            assert "kpis" in summary_json
            assert "model" in summary_json

            visuals = await client.get("/api/visualizations")
            assert visuals.status_code == 200
            assert len(visuals.json()["items"]) >= 5

            metrics = await client.get("/api/model-metrics")
            assert metrics.status_code == 200
            metrics_json = metrics.json()
            assert "summary" in metrics_json
            assert "scoring" in metrics_json
            assert "score_rows" in metrics_json

            notes = await client.get("/api/data-model-notes")
            assert notes.status_code == 200
            assert "data_processing" in notes.json()

            options = await client.get("/api/options")
            assert options.status_code == 200
            options_json = options.json()
            route_entries_for_1 = [r for r in options_json["routes"] if r["bundle_value"] == "1"]
            assert route_entries_for_1, "expected route with bundle_value '1' in options"
            assert route_entries_for_1[0]["value"] == "1"
            assert route_entries_for_1[0]["label"].startswith("1")
            assert {"value": "A", "label": "A"} in options_json["route_stop_map"]["1"]

    asyncio.run(_run_checks())


def test_dashboard_predict_endpoint(dashboard_bundle_path: Path) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    async def _run_checks() -> None:
        transport = httpx.ASGITransport(app=create_app(dashboard_bundle_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/predict",
                json={
                    "route_id": "1",
                    "stop_id": "A",
                    "scheduled_time": "2026-01-03T08:00:00",
                    "scheduled_headway": 10,
                    "direction_id": "0",
                },
            )
            assert response.status_code == 200
            assert math.isfinite(response.json()["predicted_delay_minutes"])

            horizon = await client.post(
                "/api/predict-horizon",
                json={
                    "route_id": "1",
                    "stop_id": "A",
                    "scheduled_time": "2026-01-03T08:00:00",
                    "scheduled_headway": 10,
                    "direction_id": "0",
                    "horizon_hours": 2,
                    "interval_minutes": 30,
                },
            )
            assert horizon.status_code == 200
            horizon_json = horizon.json()
            assert horizon_json["horizon_hours"] == 2
            assert len(horizon_json["rows"]) == 5
            assert all(
                math.isfinite(row["predicted_delay_minutes"])
                for row in horizon_json["rows"]
            )
            assert all(
                math.isfinite(row["historical_baseline_delay_minutes"])
                for row in horizon_json["rows"]
            )

    asyncio.run(_run_checks())


def test_dashboard_live_compare_handles_no_predictions(
    dashboard_bundle_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    def _empty_snapshot(*args, **kwargs):
        empty = pd.DataFrame()
        return {"predictions": empty, "vehicles": empty, "merged": empty}

    monkeypatch.setattr("src.inference.dashboard.collect_live_snapshot", _empty_snapshot)

    async def _run_checks() -> None:
        transport = httpx.ASGITransport(app=create_app(dashboard_bundle_path))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/live-compare",
                json={"route_id": "1", "stop_id": "A", "prediction_limit": 3},
            )
            assert response.status_code == 200
            assert response.json()["mode"] == "no_predictions"

    asyncio.run(_run_checks())
