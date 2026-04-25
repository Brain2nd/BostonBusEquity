"""
FastAPI application for realtime inference.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.mbta_realtime import MBTARealtimeClient, MBTARealtimeError
from src.inference.runtime import InferenceInputError, PredictRequest, RealtimeDelayPredictor


class PredictBody(BaseModel):
    route_id: str = Field(..., min_length=1)
    stop_id: str = Field(..., min_length=1)
    scheduled_time: str
    scheduled_headway: float | None = None
    direction_id: str | None = None


class MBTAPredictBody(BaseModel):
    route_id: str = Field(..., min_length=1)
    stop_id: str = Field(..., min_length=1)
    direction_id: str | None = None
    api_key: str | None = None


def create_app(bundle_path: str | Path) -> FastAPI:
    predictor = RealtimeDelayPredictor.from_path(bundle_path)
    app = FastAPI(title="Boston Bus Equity Realtime Inference API")

    @app.get("/health")
    def health() -> dict:
        return {
            "loaded": True,
            "model": predictor.bundle["model_name"],
            "experiment": predictor.bundle["experiment"],
            "feature_version": predictor.bundle["feature_version"],
            "bundle_version": predictor.bundle["bundle_version"],
        }

    @app.post("/predict")
    def predict(body: PredictBody) -> dict:
        try:
            payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
            return predictor.predict(PredictRequest(**payload))
        except InferenceInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict/mbta")
    def predict_from_mbta(body: MBTAPredictBody) -> dict:
        client = MBTARealtimeClient(api_key=body.api_key)
        try:
            live_record = client.fetch_live_record(
                route_id=body.route_id,
                stop_id=body.stop_id,
                direction_id=body.direction_id,
            )
        except MBTARealtimeError as exc:
            raise HTTPException(status_code=424, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"MBTA API request failed: {exc}") from exc

        try:
            prediction = predictor.predict(
                PredictRequest(
                    route_id=live_record.route_id,
                    stop_id=live_record.stop_id,
                    direction_id=live_record.direction_id,
                    scheduled_time=live_record.scheduled_time,
                    scheduled_headway=live_record.scheduled_headway,
                )
            )
        except InferenceInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            **prediction,
            "source": "mbta_v3_api",
            "schedule_id": live_record.schedule_id,
            "scheduled_time": live_record.scheduled_time,
            "scheduled_headway": live_record.scheduled_headway,
            "mbta_prediction_departure_time": live_record.prediction_departure_time,
            "mbta_prediction_delay_minutes": live_record.mbta_prediction_delay_minutes,
        }

    return app
