from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.build_bundle import build_realtime_bundle_from_dataframe
from src.inference.runtime import DelayPredictorRuntime, PredictionInputError
from src.models.v2_delay_predictor import V2_FEATURE_COLUMNS, V2MLPPredictor

RUNTIME_BASELINE_AVG_MS = float(
    os.environ.get("BOSTON_BUS_RUNTIME_BASELINE_AVG_MS", "50.0")
)
RUNTIME_BASELINE_P95_MS = float(
    os.environ.get("BOSTON_BUS_RUNTIME_BASELINE_P95_MS", "100.0")
)
API_BASELINE_AVG_MS = float(os.environ.get("BOSTON_BUS_API_BASELINE_AVG_MS", "100.0"))
API_BASELINE_P95_MS = float(os.environ.get("BOSTON_BUS_API_BASELINE_P95_MS", "200.0"))


def _make_synthetic_dataframe() -> pd.DataFrame:
    rows = []
    for year in [2024, 2025]:
        for route_id, route_bias in [("1", 0.5), ("2", 2.0)]:
            for stop_id, stop_bias in [("A", 0.2), ("B", 0.7)]:
                for day in [3, 4]:
                    for hour in [8, 17]:
                        scheduled = pd.Timestamp(
                            year=year,
                            month=1,
                            day=day,
                            hour=hour,
                            minute=0,
                            tz="UTC",
                        )
                        delay_minutes = route_bias + stop_bias + (0.5 if hour == 17 else 0.0)
                        actual = scheduled + pd.Timedelta(minutes=delay_minutes)
                        rows.append(
                            {
                                "service_date": scheduled.date().isoformat(),
                                "route_id": route_id,
                                "stop_id": stop_id,
                                "direction_id": "0",
                                "scheduled": scheduled.isoformat(),
                                "actual": actual.isoformat(),
                                "scheduled_headway": 12,
                                "year": year,
                            }
                        )

    return pd.DataFrame(rows)


def _make_prediction_kwargs() -> dict:
    return {
        "route_id": "1",
        "stop_id": "A",
        "scheduled_time": datetime(2026, 1, 5, 8, 0, 0),
    }


def _measure_call_latencies(callable_obj, iterations: int = 100, warmup: int = 10) -> tuple[float, float]:
    for _ in range(warmup):
        callable_obj()

    samples_ms: list[float] = []
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        callable_obj()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        samples_ms.append(elapsed_ms)

    samples_ms.sort()
    average_ms = sum(samples_ms) / len(samples_ms)
    p95_index = max(0, math.ceil(len(samples_ms) * 0.95) - 1)
    p95_ms = samples_ms[p95_index]
    return average_ms, p95_ms


@pytest.fixture()
def bundle_path(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "delay_predictor_mlp_v2_lag_features_temporal.pt"
    bundle_path = tmp_path / "delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt"

    model = V2MLPPredictor(input_size=len(V2_FEATURE_COLUMNS))
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    build_realtime_bundle_from_dataframe(
        dataframe=_make_synthetic_dataframe(),
        checkpoint_path=checkpoint_path,
        output_path=bundle_path,
    )

    return bundle_path


def test_build_bundle_contains_expected_fields(bundle_path: Path) -> None:
    bundle = torch.load(bundle_path, map_location="cpu")

    assert bundle["feature_columns"] == V2_FEATURE_COLUMNS
    assert bundle["feature_version"] == "v2_causal_statistics"
    assert bundle["model_config"]["input_size"] == 18
    assert "model_state_dict" in bundle
    assert "x" in bundle["scalers"]
    assert "y" in bundle["scalers"]
    assert "route_id" in bundle["encoders"]
    assert "stop_id" in bundle["encoders"]
    assert "direction_id" in bundle["encoders"]
    assert "scheduled_headway_median" in bundle["stats"]


def test_runtime_predict_returns_finite_value(bundle_path: Path) -> None:
    runtime = DelayPredictorRuntime.from_bundle_path(bundle_path)
    prediction = runtime.predict(
        **_make_prediction_kwargs(),
    )

    assert math.isfinite(prediction["predicted_delay_minutes"])
    assert prediction["model"] == "V2MLP"
    assert "direction_id" in prediction["used_defaults"]
    assert "scheduled_headway" in prediction["used_defaults"]


def test_runtime_predict_baseline_latency(bundle_path: Path) -> None:
    runtime = DelayPredictorRuntime.from_bundle_path(bundle_path)

    average_ms, p95_ms = _measure_call_latencies(
        lambda: runtime.predict(**_make_prediction_kwargs())
    )

    assert average_ms < RUNTIME_BASELINE_AVG_MS, (
        f"runtime average latency {average_ms:.3f} ms exceeded "
        f"baseline {RUNTIME_BASELINE_AVG_MS:.3f} ms"
    )
    assert p95_ms < RUNTIME_BASELINE_P95_MS, (
        f"runtime p95 latency {p95_ms:.3f} ms exceeded "
        f"baseline {RUNTIME_BASELINE_P95_MS:.3f} ms"
    )


def test_runtime_rejects_unknown_route(bundle_path: Path) -> None:
    runtime = DelayPredictorRuntime.from_bundle_path(bundle_path)

    with pytest.raises(PredictionInputError):
        runtime.predict(
            route_id="999",
            stop_id="A",
            scheduled_time=datetime(2026, 1, 5, 8, 0, 0),
            scheduled_headway=10,
        )


def test_api_endpoints(bundle_path: Path) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    from src.inference.api import create_app

    async def _run_checks():
        transport = httpx.ASGITransport(app=create_app(bundle_path))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["bundle_loaded"] is True

            response = await client.post(
                "/predict",
                json={
                    "route_id": "1",
                    "stop_id": "A",
                    "scheduled_time": "2026-01-05T08:00:00",
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert math.isfinite(body["predicted_delay_minutes"])
            assert "scheduled_headway" in body["used_defaults"]

            unknown = await client.post(
                "/predict",
                json={
                    "route_id": "999",
                    "stop_id": "A",
                    "scheduled_time": "2026-01-05T08:00:00-05:00",
                    "scheduled_headway": 12,
                },
            )
            assert unknown.status_code == 422

    asyncio.run(_run_checks())


def test_api_predict_baseline_latency(bundle_path: Path) -> None:
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    from src.inference.api import create_app

    async def _run_benchmark():
        payload = {
            "route_id": "1",
            "stop_id": "A",
            "scheduled_time": "2026-01-05T08:00:00",
        }
        transport = httpx.ASGITransport(app=create_app(bundle_path))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            for _ in range(5):
                response = await client.post("/predict", json=payload)
                assert response.status_code == 200

            samples_ms: list[float] = []
            for _ in range(50):
                start_ns = time.perf_counter_ns()
                response = await client.post("/predict", json=payload)
                assert response.status_code == 200
                samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

        samples_ms.sort()
        average_ms = sum(samples_ms) / len(samples_ms)
        p95_ms = samples_ms[math.ceil(len(samples_ms) * 0.95) - 1]
        return average_ms, p95_ms

    average_ms, p95_ms = asyncio.run(_run_benchmark())

    assert average_ms < API_BASELINE_AVG_MS, (
        f"api average latency {average_ms:.3f} ms exceeded "
        f"baseline {API_BASELINE_AVG_MS:.3f} ms"
    )
    assert p95_ms < API_BASELINE_P95_MS, (
        f"api p95 latency {p95_ms:.3f} ms exceeded "
        f"baseline {API_BASELINE_P95_MS:.3f} ms"
    )
