from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.build_bundle import build_realtime_bundle_from_dataframe
from src.inference.mbta_v3_client import normalize_prediction_payload
from src.inference.plot_mbta_realtime_comparison import apply_runtime_predictions
from src.inference.runtime import DelayPredictorRuntime
from src.models.v2_delay_predictor import V2_FEATURE_COLUMNS, V2MLPPredictor


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


def _make_sample_payload() -> dict:
    return {
        "data": [
            {
                "id": "prediction-1",
                "type": "prediction",
                "attributes": {
                    "arrival_time": "2026-04-24T12:04:00Z",
                    "departure_time": None,
                    "direction_id": 1,
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
            },
            {
                "id": "prediction-2",
                "type": "prediction",
                "attributes": {
                    "arrival_time": "2026-04-24T12:17:00Z",
                    "departure_time": None,
                    "direction_id": 1,
                    "status": "7 min",
                    "schedule_relationship": None,
                },
                "relationships": {
                    "route": {"data": {"id": "1", "type": "route"}},
                    "stop": {"data": {"id": "A", "type": "stop"}},
                    "trip": {"data": {"id": "trip-2", "type": "trip"}},
                    "vehicle": {"data": {"id": "vehicle-2", "type": "vehicle"}},
                    "schedule": {"data": {"id": "schedule-2", "type": "schedule"}},
                },
            },
        ],
        "included": [
            {
                "id": "schedule-1",
                "type": "schedule",
                "attributes": {
                    "arrival_time": "2026-04-24T12:00:00Z",
                    "departure_time": None,
                },
            },
            {
                "id": "schedule-2",
                "type": "schedule",
                "attributes": {
                    "arrival_time": "2026-04-24T12:10:00Z",
                    "departure_time": None,
                },
            },
        ],
    }


def test_normalize_prediction_payload_computes_delay_and_headway() -> None:
    dataframe = normalize_prediction_payload(
        _make_sample_payload(),
        observed_at=pd.Timestamp("2026-04-24T08:00:00-04:00"),
    )

    assert len(dataframe) == 2
    assert list(dataframe["prediction_rank"]) == [1, 2]
    assert math.isclose(float(dataframe.iloc[0]["official_delay_minutes"]), 4.0)
    assert math.isclose(float(dataframe.iloc[1]["official_delay_minutes"]), 7.0)
    assert math.isclose(float(dataframe.iloc[0]["scheduled_headway_minutes"]), 10.0)
    assert math.isclose(float(dataframe.iloc[1]["scheduled_headway_minutes"]), 10.0)


def test_apply_runtime_predictions_on_realtime_rows(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "delay_predictor_mlp_v2_lag_features_temporal.pt"
    bundle_path = tmp_path / "delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt"

    model = V2MLPPredictor(input_size=len(V2_FEATURE_COLUMNS))
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    build_realtime_bundle_from_dataframe(
        dataframe=_make_synthetic_dataframe(),
        checkpoint_path=checkpoint_path,
        output_path=bundle_path,
    )
    runtime = DelayPredictorRuntime.from_bundle_path(bundle_path)

    realtime_rows = normalize_prediction_payload(
        _make_sample_payload(),
        observed_at=pd.Timestamp("2026-04-24T08:00:00-04:00"),
    )
    enriched = apply_runtime_predictions(realtime_rows, runtime=runtime)

    assert enriched["model_predicted_delay_minutes"].notna().all()
    assert enriched["model_error"].eq("").all()
    assert enriched["model_used_defaults"].str.contains("direction_id").all()
    assert enriched["route_id"].eq("1").all()
    assert enriched["stop_id"].eq("A").all()
