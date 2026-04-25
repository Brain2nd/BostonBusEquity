"""
Optional MBTA V3 API integration for live request construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.inference.bundle_utils import iter_future_schedule_times

API_BASE_URL = "https://api-v3.mbta.com"
LOCAL_TZ = ZoneInfo("America/New_York")


class MBTARealtimeError(RuntimeError):
    """Raised when MBTA V3 data cannot be fetched or interpreted."""


@dataclass
class MBTARealtimeRecord:
    route_id: str
    stop_id: str
    direction_id: str | None
    scheduled_time: str
    scheduled_headway: float | None
    schedule_id: str | None
    prediction_departure_time: str | None
    mbta_prediction_delay_minutes: float | None


class MBTARealtimeClient:
    def __init__(self, api_key: str | None = None, timeout: int = 15) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, path: str, params: dict) -> dict:
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        response = requests.get(
            f"{API_BASE_URL}{path}",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def fetch_live_record(
        self,
        route_id: str,
        stop_id: str,
        direction_id: str | None = None,
        now: datetime | None = None,
    ) -> MBTARealtimeRecord:
        now_ts = pd.Timestamp(now or datetime.now(tz=LOCAL_TZ))

        schedule_params = {
            "filter[route]": route_id,
            "filter[stop]": stop_id,
            "filter[min_time]": now_ts.strftime("%H:%M"),
            "sort": "arrival_time",
            "page[limit]": 5,
        }
        prediction_params = {
            "filter[route]": route_id,
            "filter[stop]": stop_id,
            "sort": "arrival_time",
            "page[limit]": 3,
        }
        if direction_id is not None:
            schedule_params["filter[direction_id]"] = direction_id
            prediction_params["filter[direction_id]"] = direction_id

        schedule_payload = self._request("/schedules", schedule_params)
        prediction_payload = self._request("/predictions", prediction_params)

        schedule_records = []
        for item in schedule_payload.get("data", []):
            attributes = item.get("attributes", {})
            departure_time = attributes.get("departure_time") or attributes.get("arrival_time")
            if not departure_time:
                continue
            departure_ts = pd.Timestamp(departure_time)
            if departure_ts.tzinfo is None:
                departure_ts = departure_ts.tz_localize(LOCAL_TZ)
            if departure_ts >= now_ts:
                schedule_records.append(
                    {
                        "id": item.get("id"),
                        "departure_time": departure_time,
                        "direction_id": attributes.get("direction_id"),
                    }
                )

        if not schedule_records:
            raise MBTARealtimeError("No upcoming MBTA schedule records found for the requested route/stop")

        schedule_times = iter_future_schedule_times(schedule_records, "departure_time")
        scheduled_headway = None
        if len(schedule_times) >= 2:
            scheduled_headway = float((schedule_times[1] - schedule_times[0]).total_seconds() / 60)

        schedule_record = schedule_records[0]

        prediction_delay = None
        prediction_departure_time = None
        for item in prediction_payload.get("data", []):
            attributes = item.get("attributes", {})
            departure_time = attributes.get("departure_time") or attributes.get("arrival_time")
            if not departure_time:
                continue
            prediction_departure_time = departure_time
            prediction_ts = pd.Timestamp(departure_time)
            if prediction_ts.tzinfo is None:
                prediction_ts = prediction_ts.tz_localize(LOCAL_TZ)
            if prediction_ts < now_ts:
                continue
            scheduled_ts = pd.Timestamp(schedule_record["departure_time"])
            if scheduled_ts.tzinfo is None:
                scheduled_ts = scheduled_ts.tz_localize(LOCAL_TZ)
            prediction_delay = float((prediction_ts - scheduled_ts).total_seconds() / 60)
            break

        return MBTARealtimeRecord(
            route_id=route_id,
            stop_id=stop_id,
            direction_id=(str(direction_id) if direction_id is not None else None),
            scheduled_time=schedule_record["departure_time"],
            scheduled_headway=scheduled_headway,
            schedule_id=schedule_record["id"],
            prediction_departure_time=prediction_departure_time,
            mbta_prediction_delay_minutes=prediction_delay,
        )
