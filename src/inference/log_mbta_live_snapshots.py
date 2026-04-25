"""Log MBTA live prediction and vehicle snapshots for V5 residual labels."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.inference.mbta_v3_client import (
    MBTAV3Client,
    normalize_prediction_payload,
    normalize_vehicle_payload,
)

LOCAL_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_OUTPUT_DIR = REPORTS_DIR / "live_prediction_snapshots"


def merge_prediction_vehicle_fields(
    predictions: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> pd.DataFrame:
    merged = predictions.copy()
    if merged.empty:
        return merged

    if vehicles.empty or "vehicle_id" not in merged.columns:
        for column in [
            "current_stop_sequence",
            "current_status",
            "speed",
            "latitude",
            "longitude",
            "bearing",
            "vehicle_updated_at",
        ]:
            merged[column] = pd.NA
        return merged

    vehicle_subset = vehicles[
        [
            "vehicle_id",
            "current_stop_sequence",
            "current_status",
            "speed",
            "latitude",
            "longitude",
            "bearing",
            "updated_at",
        ]
    ].drop_duplicates("vehicle_id", keep="last")
    vehicle_subset = vehicle_subset.rename(
        columns={
            "current_stop_sequence": "vehicle_current_stop_sequence",
            "updated_at": "vehicle_updated_at",
        }
    )
    merged = merged.merge(vehicle_subset, on="vehicle_id", how="left")
    if "current_stop_sequence" in merged.columns:
        prediction_sequence = pd.to_numeric(
            merged["current_stop_sequence"],
            errors="coerce",
        )
        vehicle_sequence = pd.to_numeric(
            merged["vehicle_current_stop_sequence"],
            errors="coerce",
        )
        merged["current_stop_sequence"] = np.where(
            prediction_sequence.notna(),
            prediction_sequence,
            vehicle_sequence,
        )
    else:
        merged["current_stop_sequence"] = merged["vehicle_current_stop_sequence"]
    return merged


def _write_snapshot(dataframe: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_parquet(path, index=False)
        return path
    except Exception:
        fallback_path = path.with_suffix(".csv")
        dataframe.to_csv(fallback_path, index=False)
        return fallback_path


def collect_live_snapshot(
    client: MBTAV3Client,
    route_id: str,
    stop_id: str,
    direction_id: str | int | None,
    prediction_limit: int,
    vehicle_limit: int,
    observed_at: datetime | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    observed_timestamp = pd.Timestamp(observed_at or datetime.now(tz=LOCAL_TIMEZONE))
    prediction_payload = client.fetch_predictions_payload(
        route_id=route_id,
        stop_id=stop_id,
        direction_id=direction_id,
        limit=prediction_limit,
    )
    vehicle_payload = client.fetch_vehicles_payload(
        route_id=route_id,
        limit=vehicle_limit,
    )

    predictions = normalize_prediction_payload(
        prediction_payload,
        observed_at=observed_timestamp,
    )
    vehicles = normalize_vehicle_payload(vehicle_payload, observed_at=observed_timestamp)
    if direction_id is not None and "direction_id" in vehicles.columns:
        vehicles = vehicles[
            vehicles["direction_id"].astype("string") == str(direction_id)
        ].reset_index(drop=True)

    merged = merge_prediction_vehicle_fields(predictions, vehicles)
    return {"predictions": predictions, "vehicles": vehicles, "merged": merged}


def log_live_snapshots(
    route_id: str,
    stop_id: str,
    direction_id: str | int | None = None,
    poll_count: int = 3,
    poll_interval_seconds: float = 20.0,
    prediction_limit: int = 10,
    vehicle_limit: int = 100,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    if poll_count <= 0:
        raise ValueError("poll_count must be positive")
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")

    client = MBTAV3Client()
    written: list[str] = []
    total_prediction_rows = 0
    total_vehicle_rows = 0

    for poll_index in range(poll_count):
        observed_at = pd.Timestamp(datetime.now(tz=LOCAL_TIMEZONE))
        snapshots = collect_live_snapshot(
            client=client,
            route_id=route_id,
            stop_id=stop_id,
            direction_id=direction_id,
            prediction_limit=prediction_limit,
            vehicle_limit=vehicle_limit,
            observed_at=observed_at,
        )
        stamp = observed_at.strftime("%Y%m%dT%H%M%S")
        for name, dataframe in snapshots.items():
            path = output_dir / f"{name}_{stamp}_poll{poll_index + 1:03d}.parquet"
            written_path = _write_snapshot(dataframe, path)
            written.append(str(written_path))

        total_prediction_rows += len(snapshots["predictions"])
        total_vehicle_rows += len(snapshots["vehicles"])
        if poll_index < poll_count - 1:
            time.sleep(poll_interval_seconds)

    return {
        "output_dir": str(output_dir),
        "files": written,
        "prediction_rows": total_prediction_rows,
        "vehicle_rows": total_vehicle_rows,
        "poll_count": poll_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log MBTA /predictions and /vehicles snapshots for V5 residual data",
    )
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--stop-id", required=True)
    parser.add_argument("--direction-id", default=None)
    parser.add_argument("--poll-count", type=int, default=3)
    parser.add_argument("--poll-interval-seconds", type=float, default=20.0)
    parser.add_argument("--prediction-limit", type=int, default=10)
    parser.add_argument("--vehicle-limit", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = log_live_snapshots(
        route_id=args.route_id,
        stop_id=args.stop_id,
        direction_id=args.direction_id,
        poll_count=args.poll_count,
        poll_interval_seconds=args.poll_interval_seconds,
        prediction_limit=args.prediction_limit,
        vehicle_limit=args.vehicle_limit,
        output_dir=args.output_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
