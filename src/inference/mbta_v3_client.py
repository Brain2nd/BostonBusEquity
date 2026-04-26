"""MBTA V3 API helpers for realtime prediction polling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.config import MBTA_API_BASE_URL

LOCAL_TIMEZONE = ZoneInfo("America/New_York")
MBTA_API_KEY_ENV = "MBTA_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 20
PREDICTION_COLUMNS = [
    "observed_at",
    "prediction_id",
    "route_id",
    "stop_id",
    "direction_id",
    "trip_id",
    "vehicle_id",
    "status",
    "schedule_relationship",
    "scheduled_time",
    "predicted_time",
    "official_delay_minutes",
    "prediction_rank",
    "scheduled_headway_minutes",
    "current_stop_sequence",
]
VEHICLE_COLUMNS = [
    "observed_at",
    "vehicle_id",
    "route_id",
    "stop_id",
    "trip_id",
    "direction_id",
    "current_status",
    "current_stop_sequence",
    "latitude",
    "longitude",
    "bearing",
    "speed",
    "updated_at",
]


def _parse_timestamp(value: str | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(LOCAL_TIMEZONE)


def _coalesce_time(primary_value: str | None, secondary_value: str | None) -> pd.Timestamp | pd.NaT:
    primary = _parse_timestamp(primary_value)
    if not pd.isna(primary):
        return primary
    return _parse_timestamp(secondary_value)


def _relationship_id(relationships: dict[str, Any], name: str) -> str | None:
    data = (relationships.get(name) or {}).get("data")
    if not data:
        return None
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]
    return str(data.get("id")) if data.get("id") is not None else None


def _build_included_lookup(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("included", []):
        item_type = item.get("type")
        item_id = item.get("id")
        if item_type and item_id:
            lookup[(str(item_type), str(item_id))] = item
    return lookup


def _empty_prediction_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _empty_vehicle_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=VEHICLE_COLUMNS)


def _add_prediction_rank_and_headway(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return _empty_prediction_dataframe()

    groups: list[pd.DataFrame] = []
    for _, group_df in dataframe.groupby("observed_at", sort=True):
        ordered = group_df.sort_values(
            by=["scheduled_time", "predicted_time", "prediction_id"],
            na_position="last",
        ).copy()
        ordered["prediction_rank"] = np.arange(1, len(ordered) + 1, dtype=int)

        prev_gap = ordered["scheduled_time"].diff().dt.total_seconds() / 60
        next_gap = (
            ordered["scheduled_time"].shift(-1) - ordered["scheduled_time"]
        ).dt.total_seconds() / 60
        candidate_gaps = pd.concat([prev_gap, next_gap], ignore_index=True)
        candidate_gaps = candidate_gaps[candidate_gaps > 0]
        median_gap = float(candidate_gaps.median()) if not candidate_gaps.empty else np.nan

        ordered["scheduled_headway_minutes"] = next_gap.where(next_gap > 0)
        ordered["scheduled_headway_minutes"] = ordered["scheduled_headway_minutes"].fillna(
            prev_gap.where(prev_gap > 0)
        )
        ordered["scheduled_headway_minutes"] = ordered["scheduled_headway_minutes"].fillna(
            median_gap
        )
        groups.append(ordered)

    return pd.concat(groups, ignore_index=True)


def normalize_prediction_payload(
    payload: dict[str, Any],
    observed_at: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not payload.get("data"):
        return _empty_prediction_dataframe()

    observed_timestamp = pd.Timestamp(observed_at or datetime.now(tz=LOCAL_TIMEZONE))
    if observed_timestamp.tzinfo is None:
        observed_timestamp = observed_timestamp.tz_localize(LOCAL_TIMEZONE)
    else:
        observed_timestamp = observed_timestamp.tz_convert(LOCAL_TIMEZONE)

    included_lookup = _build_included_lookup(payload)
    rows: list[dict[str, Any]] = []

    for item in payload.get("data", []):
        attributes = item.get("attributes", {})
        relationships = item.get("relationships", {})
        schedule_id = _relationship_id(relationships, "schedule")
        schedule_attributes = {}
        if schedule_id is not None:
            schedule_attributes = (
                included_lookup.get(("schedule", schedule_id), {}).get("attributes", {})
            )

        scheduled_time = _coalesce_time(
            schedule_attributes.get("departure_time"),
            schedule_attributes.get("arrival_time"),
        )
        predicted_time = _coalesce_time(
            attributes.get("departure_time"),
            attributes.get("arrival_time"),
        )

        official_delay_minutes = np.nan
        if not pd.isna(scheduled_time) and not pd.isna(predicted_time):
            official_delay_minutes = float(
                (predicted_time - scheduled_time).total_seconds() / 60
            )

        rows.append(
            {
                "observed_at": observed_timestamp,
                "prediction_id": str(item.get("id")),
                "route_id": _relationship_id(relationships, "route"),
                "stop_id": _relationship_id(relationships, "stop"),
                "direction_id": attributes.get("direction_id"),
                "trip_id": _relationship_id(relationships, "trip"),
                "vehicle_id": _relationship_id(relationships, "vehicle"),
                "status": attributes.get("status"),
                "schedule_relationship": attributes.get("schedule_relationship"),
                "scheduled_time": scheduled_time,
                "predicted_time": predicted_time,
                "official_delay_minutes": official_delay_minutes,
                "current_stop_sequence": attributes.get("stop_sequence"),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe["route_id"] = dataframe["route_id"].astype("string")
    dataframe["stop_id"] = dataframe["stop_id"].astype("string")
    dataframe["trip_id"] = dataframe["trip_id"].astype("string")
    dataframe["vehicle_id"] = dataframe["vehicle_id"].astype("string")
    return _add_prediction_rank_and_headway(dataframe)


def normalize_vehicle_payload(
    payload: dict[str, Any],
    observed_at: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not payload.get("data"):
        return _empty_vehicle_dataframe()

    observed_timestamp = pd.Timestamp(observed_at or datetime.now(tz=LOCAL_TIMEZONE))
    if observed_timestamp.tzinfo is None:
        observed_timestamp = observed_timestamp.tz_localize(LOCAL_TIMEZONE)
    else:
        observed_timestamp = observed_timestamp.tz_convert(LOCAL_TIMEZONE)

    included_lookup = _build_included_lookup(payload)
    rows: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        attributes = item.get("attributes", {})
        relationships = item.get("relationships", {})
        trip_id = _relationship_id(relationships, "trip")
        trip_attributes = {}
        if trip_id is not None:
            trip_attributes = (
                included_lookup.get(("trip", trip_id), {}).get("attributes", {})
            )

        rows.append(
            {
                "observed_at": observed_timestamp,
                "vehicle_id": str(item.get("id")),
                "route_id": _relationship_id(relationships, "route"),
                "stop_id": _relationship_id(relationships, "stop"),
                "trip_id": trip_id,
                "direction_id": trip_attributes.get("direction_id"),
                "current_status": attributes.get("current_status"),
                "current_stop_sequence": attributes.get("current_stop_sequence"),
                "latitude": attributes.get("latitude"),
                "longitude": attributes.get("longitude"),
                "bearing": attributes.get("bearing"),
                "speed": attributes.get("speed"),
                "updated_at": _parse_timestamp(attributes.get("updated_at")),
            }
        )

    dataframe = pd.DataFrame(rows)
    for column in ["vehicle_id", "route_id", "stop_id", "trip_id"]:
        dataframe[column] = dataframe[column].astype("string")
    return dataframe.reindex(columns=VEHICLE_COLUMNS)


@dataclass
class MBTAV3Client:
    """Thin MBTA V3 API client for polling realtime predictions."""

    base_url: str = MBTA_API_BASE_URL
    api_key: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get(MBTA_API_KEY_ENV)
        if self.session is None:
            self.session = requests.Session()

        self.session.headers.setdefault("Accept", "application/json")
        if self.api_key:
            self.session.headers.setdefault("x-api-key", self.api_key)

    def fetch_predictions_payload(
        self,
        route_id: str,
        stop_id: str,
        direction_id: str | int | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        params = {
            "filter[route]": str(route_id),
            "filter[stop]": str(stop_id),
            "include": "schedule,trip,vehicle,route,stop",
            "sort": "departure_time",
            "page[limit]": int(limit),
        }
        if direction_id is not None:
            params["filter[direction_id]"] = str(direction_id)

        response = self.session.get(
            f"{self.base_url}/predictions",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def fetch_predictions_dataframe(
        self,
        route_id: str,
        stop_id: str,
        direction_id: str | int | None = None,
        limit: int = 5,
    ) -> pd.DataFrame:
        payload = self.fetch_predictions_payload(
            route_id=route_id,
            stop_id=stop_id,
            direction_id=direction_id,
            limit=limit,
        )
        return normalize_prediction_payload(payload)

    def fetch_vehicles_payload(
        self,
        route_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "include": "route,trip,stop",
            "page[limit]": int(limit),
        }
        if route_id is not None:
            params["filter[route]"] = str(route_id)

        response = self.session.get(
            f"{self.base_url}/vehicles",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def fetch_vehicles_dataframe(
        self,
        route_id: str | None = None,
        direction_id: str | int | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        payload = self.fetch_vehicles_payload(route_id=route_id, limit=limit)
        dataframe = normalize_vehicle_payload(payload)
        if direction_id is not None and "direction_id" in dataframe.columns:
            dataframe = dataframe[
                dataframe["direction_id"].astype("string") == str(direction_id)
            ].reset_index(drop=True)
        return dataframe
