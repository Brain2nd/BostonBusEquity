"""FastAPI application factory for realtime delay prediction."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from fastapi.responses import FileResponse, RedirectResponse
except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
    raise RuntimeError(
        "FastAPI and its dependencies are required for the realtime API. "
        "Install `fastapi` before importing src.inference.api."
    ) from exc

from .dashboard import (
    DEFAULT_MODEL_ID,
    allowed_figure_path,
    asset_path,
    data_and_model_notes,
    defense_qa,
    list_available_models,
    live_compare,
    live_enriched_forecast,
    model_metrics,
    predict_with_registry_model,
    project_summary,
    selection_options,
    visualizations,
)
from .runtime import DelayPredictorRuntime, PredictionInputError


class PredictRequest(BaseModel):
    route_id: str
    stop_id: str
    scheduled_time: datetime
    scheduled_headway: float | None = None
    direction_id: str | None = None
    model_id: str | None = None
    trip_id: str | None = None
    vehicle_id: str | None = None
    current_stop_sequence: float | None = None
    vehicle_speed: float | None = None
    vehicle_status: str | None = None
    official_predicted_delay_minutes: float | None = None
    official_prediction_age_seconds: float | None = None


class PredictResponse(BaseModel):
    predicted_delay_minutes: float
    model: str
    experiment: str | None = None
    used_defaults: list[str] = Field(default_factory=list)
    # Multi-model registry fields (populated when model_id != default)
    model_id: str | None = None
    architecture: str | None = None
    feature_version: str | None = None
    test_R2: float | None = None
    test_RMSE: float | None = None
    model_latency_ms: float | None = None
    used_history: int | None = None


class PredictHorizonRequest(PredictRequest):
    horizon_hours: float = Field(default=6.0, ge=0.5, le=24.0)
    interval_minutes: int = Field(default=15, ge=5, le=120)


class LiveCompareRequest(BaseModel):
    route_id: str
    stop_id: str
    direction_id: str | None = None
    prediction_limit: int = Field(default=8, ge=1, le=20)
    vehicle_limit: int = Field(default=100, ge=1, le=200)
    model_id: str | None = None


class LiveEnrichedForecastRequest(BaseModel):
    route_id: str
    stop_id: str
    direction_id: str | None = None
    prediction_limit: int = Field(default=10, ge=1, le=20)
    vehicle_limit: int = Field(default=100, ge=1, le=200)
    model_id: str | None = None


def create_app(bundle_path: str | Path) -> FastAPI:
    runtime = DelayPredictorRuntime.from_bundle_path(bundle_path)
    app = FastAPI(
        title="Boston Bus Equity Realtime Inference",
        version="1.0.0",
    )
    app.state.runtime = runtime

    @app.get("/")
    def dashboard_home() -> FileResponse:
        return FileResponse(asset_path("index.html"), media_type="text/html")

    @app.get("/dashboard")
    def dashboard_redirect() -> RedirectResponse:
        return RedirectResponse(url="/")

    @app.get("/assets/{filename}")
    def dashboard_asset(filename: str) -> FileResponse:
        media_type = "text/css" if filename.endswith(".css") else "application/javascript"
        return FileResponse(asset_path(filename), media_type=media_type)

    @app.get("/figures/{filename}")
    def report_figure(filename: str) -> FileResponse:
        return FileResponse(allowed_figure_path(filename), media_type="image/png")

    @app.get("/health")
    def health() -> dict:
        return runtime.health()

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        # Default V2 runtime path (preserves the original POST /predict shape)
        try:
            prediction = runtime.predict(
                route_id=payload.route_id,
                stop_id=payload.stop_id,
                scheduled_time=payload.scheduled_time,
                scheduled_headway=payload.scheduled_headway,
                direction_id=payload.direction_id,
                trip_id=payload.trip_id,
                vehicle_id=payload.vehicle_id,
                current_stop_sequence=payload.current_stop_sequence,
                vehicle_speed=payload.vehicle_speed,
                vehicle_status=payload.vehicle_status,
                official_predicted_delay_minutes=payload.official_predicted_delay_minutes,
                official_prediction_age_seconds=payload.official_prediction_age_seconds,
            )
        except PredictionInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return PredictResponse(**prediction)

    @app.get("/api/project-summary")
    def dashboard_project_summary() -> dict:
        return project_summary(runtime)

    @app.get("/api/visualizations")
    def dashboard_visualizations() -> dict:
        return visualizations()

    @app.get("/api/model-metrics")
    def dashboard_model_metrics() -> dict:
        return model_metrics()

    @app.get("/api/options")
    def dashboard_options() -> dict:
        return selection_options(runtime)

    @app.get("/api/data-model-notes")
    def dashboard_data_model_notes() -> dict:
        return data_and_model_notes()

    @app.get("/api/defense-qa")
    def dashboard_defense_qa() -> dict:
        return defense_qa()

    @app.get("/api/models")
    def dashboard_models() -> dict:
        return {
            "default": DEFAULT_MODEL_ID,
            "models": list_available_models(),
        }

    @app.post("/api/predict", response_model=PredictResponse)
    def dashboard_predict(payload: PredictRequest) -> PredictResponse:
        # If a specific model was requested, dispatch through the registry
        if payload.model_id and payload.model_id != "v2_mlp_realtime":
            try:
                result = predict_with_registry_model(
                    model_id=payload.model_id,
                    fallback_runtime=runtime,
                    route_id=payload.route_id,
                    stop_id=payload.stop_id,
                    scheduled_time=payload.scheduled_time.isoformat(),
                    direction_id=payload.direction_id,
                    scheduled_headway=payload.scheduled_headway,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return PredictResponse(**result)
        # Default path: V2 MLP realtime bundle through PR #3's runtime
        return predict(payload)

    @app.post("/api/predict-horizon")
    def dashboard_predict_horizon(payload: PredictHorizonRequest) -> dict:
        total_minutes = int(round(payload.horizon_hours * 60))
        step_minutes = int(payload.interval_minutes)
        row_count = min(200, total_minutes // step_minutes + 1)
        rows = []
        used_defaults = set()

        for index in range(row_count):
            scheduled_time = payload.scheduled_time + timedelta(
                minutes=index * step_minutes
            )
            try:
                prediction = runtime.predict(
                    route_id=payload.route_id,
                    stop_id=payload.stop_id,
                    scheduled_time=scheduled_time,
                    scheduled_headway=payload.scheduled_headway,
                    direction_id=payload.direction_id,
                    trip_id=payload.trip_id,
                    vehicle_id=payload.vehicle_id,
                    current_stop_sequence=payload.current_stop_sequence,
                    vehicle_speed=payload.vehicle_speed,
                    vehicle_status=payload.vehicle_status,
                    official_predicted_delay_minutes=payload.official_predicted_delay_minutes,
                    official_prediction_age_seconds=payload.official_prediction_age_seconds,
                )
            except PredictionInputError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            used_defaults.update(prediction["used_defaults"])
            baseline = runtime.historical_baseline_delay(
                route_id=payload.route_id,
                stop_id=payload.stop_id,
                scheduled_time=scheduled_time,
                direction_id=payload.direction_id,
            )
            rows.append(
                {
                    "scheduled_time": scheduled_time.isoformat(),
                    "predicted_delay_minutes": prediction["predicted_delay_minutes"],
                    "historical_baseline_delay_minutes": baseline[
                        "predicted_delay_minutes"
                    ],
                    "historical_baseline_source": baseline["source"],
                    "used_defaults": prediction["used_defaults"],
                }
            )

        return {
            "model": runtime.health()["model"],
            "experiment": runtime.health()["experiment"],
            "horizon_hours": payload.horizon_hours,
            "interval_minutes": step_minutes,
            "used_defaults": sorted(used_defaults),
            "rows": rows,
        }

    @app.post("/api/live-compare")
    def dashboard_live_compare(payload: LiveCompareRequest) -> dict:
        return live_compare(
            runtime=runtime,
            route_id=payload.route_id,
            stop_id=payload.stop_id,
            direction_id=payload.direction_id,
            prediction_limit=payload.prediction_limit,
            vehicle_limit=payload.vehicle_limit,
            model_id=payload.model_id or DEFAULT_MODEL_ID,
        )

    @app.post("/api/live-enriched-forecast")
    def dashboard_live_enriched_forecast(payload: LiveEnrichedForecastRequest) -> dict:
        return live_enriched_forecast(
            runtime=runtime,
            route_id=payload.route_id,
            stop_id=payload.stop_id,
            direction_id=payload.direction_id,
            prediction_limit=payload.prediction_limit,
            vehicle_limit=payload.vehicle_limit,
            model_id=payload.model_id or DEFAULT_MODEL_ID,
        )

    return app
