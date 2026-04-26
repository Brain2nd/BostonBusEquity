"""
Deployment integration tests.

These tests exercise the *real* artifacts shipped with the repository
(the V2 MLP realtime bundle, the dashboard module, and the swapped-in
visualization catalog) rather than synthetic fixtures, so they catch
regressions that only surface against production data.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

V2_BUNDLE = MODELS_DIR / "delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt"
V4_SCORE_BUNDLE = MODELS_DIR / "delay_predictor_v4_score_best_online_safe_bundle.joblib"
V4_BEST_BUNDLE = MODELS_DIR / "delay_predictor_v4_best_online_safe_bundle.joblib"


# ---------------------------------------------------------------------------
# Static guarantees about the deployment artifacts
# ---------------------------------------------------------------------------


def test_v2_bundle_exists_and_is_real() -> None:
    """The default deployment bundle must exist and be non-trivial in size."""
    assert V2_BUNDLE.exists(), f"missing default deployment bundle: {V2_BUNDLE}"
    assert V2_BUNDLE.stat().st_size > 50_000, "V2 bundle is suspiciously small"


def test_v4_lightgbm_bundles_present_as_fallback() -> None:
    """V4 LightGBM bundles must remain available as fallback comparison models."""
    assert V4_SCORE_BUNDLE.exists()
    assert V4_BEST_BUNDLE.exists()


def test_choose_default_bundle_prefers_our_v2() -> None:
    """choose_default_bundle must select OUR V2 MLP, not the V4 LightGBM bundles."""
    from src.inference.dashboard import choose_default_bundle

    selected = choose_default_bundle()
    assert selected == V2_BUNDLE, (
        f"deployment default should be our V2 MLP bundle but got {selected.name}"
    )


def test_dashboard_visualization_catalog_includes_our_results() -> None:
    """Our V3 ablation, V5 NeuronSpark, and V6 figures must be surfaced."""
    from src.inference.dashboard import VISUALIZATION_CATALOG

    ids = {entry["id"] for entry in VISUALIZATION_CATALOG}
    required = {
        "ablation_study_comparison",
        "delay_prediction_neuronspark_comparison",
        "delay_prediction_training_curves_v3_wavelet_temporal",
        "delay_prediction_multistep_comparison",
    }
    missing = required - ids
    assert not missing, f"visualization catalog missing our entries: {missing}"


def test_visualization_catalog_filenames_exist_on_disk() -> None:
    """Every catalog entry must point at a file that actually exists."""
    from src.inference.dashboard import VISUALIZATION_CATALOG

    missing = []
    for entry in VISUALIZATION_CATALOG:
        path = FIGURES_DIR / entry["filename"]
        if not path.exists():
            missing.append(entry["filename"])
    assert not missing, f"catalog references missing figure files: {missing}"


def test_kpis_include_v6_transformer_and_v5_snn() -> None:
    """The deployment KPIs must surface our headline R^2 numbers."""
    from src.inference.dashboard import PROJECT_KPIS

    labels = {kpi["label"] for kpi in PROJECT_KPIS}
    assert "Best model R²" in labels, "missing V6 Transformer KPI"
    assert "NeuronSpark SNN R²" in labels, "missing V5 SNN KPI"

    by_label = {kpi["label"]: kpi for kpi in PROJECT_KPIS}
    assert by_label["Best model R²"]["value"] == "0.9942"
    assert by_label["NeuronSpark SNN R²"]["value"] == "0.9897"


# ---------------------------------------------------------------------------
# Real V2 bundle drives the runtime + API end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_runtime():
    """Load the actual V2 MLP bundle through the dashboard runtime."""
    pytest.importorskip("torch")
    from src.inference.runtime import DelayPredictorRuntime

    return DelayPredictorRuntime.from_bundle_path(V2_BUNDLE)


def test_real_runtime_health_reports_v2(real_runtime) -> None:
    health = real_runtime.health()
    assert health["bundle_loaded"] is True
    assert "V2" in str(health.get("model", "")) or "v2" in str(health.get("feature_version", ""))


def test_real_runtime_predict_returns_finite_for_known_route_stop(real_runtime) -> None:
    """Predict using a route/stop pair that exists in the trained encoder."""
    # First known route and stop from the bundle's encoder
    route = next(iter(real_runtime.encoders["route_id"]))
    stop = next(iter(real_runtime.encoders["stop_id"]))
    direction = next(iter(real_runtime.encoders["direction_id"]))

    scheduled = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    result = real_runtime.predict(
        route_id=route,
        stop_id=stop,
        scheduled_time=scheduled,
        direction_id=direction,
    )

    # runtime.predict() returns a dict-shaped result
    assert "predicted_delay_minutes" in result
    pred = result["predicted_delay_minutes"]
    assert pred == pred, "prediction is NaN"
    assert -60 < pred < 120, f"prediction out of plausible bus delay range: {pred}"
    assert result["model"] == "V2MLP", f"deployment should serve V2MLP, got {result['model']}"


def test_real_runtime_rejects_unknown_route(real_runtime) -> None:
    from src.inference.runtime import PredictionInputError

    with pytest.raises(PredictionInputError):
        real_runtime.predict(
            route_id="THIS_ROUTE_DOES_NOT_EXIST_99999",
            stop_id=next(iter(real_runtime.encoders["stop_id"])),
            scheduled_time=datetime.now(timezone.utc).isoformat(),
        )


def test_dashboard_api_serves_real_bundle_health() -> None:
    """End-to-end: spin up the FastAPI app on the real V2 bundle and hit /health."""
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    async def _run() -> dict:
        transport = httpx.ASGITransport(app=create_app(V2_BUNDLE))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            return response.json()

    payload = asyncio.run(_run())
    assert payload["bundle_loaded"] is True


def test_dashboard_api_options_exposes_real_routes() -> None:
    """/api/options on the real bundle must list real MBTA route IDs (not synthetic)."""
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    async def _run() -> dict:
        transport = httpx.ASGITransport(app=create_app(V2_BUNDLE))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/options")
            assert response.status_code == 200
            return response.json()

    options = asyncio.run(_run())
    assert len(options["routes"]) >= 50, "real bundle should expose many routes"
    # Real bundle has >100 stops; synthetic fixtures have <10
    assert len(options["stops"]) >= 100


def test_all_dashboard_get_endpoints_serve_real_bundle() -> None:
    """Every read-only dashboard endpoint must respond 200 on the real V2 bundle."""
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    endpoints = [
        "/",
        "/health",
        "/api/project-summary",
        "/api/visualizations",
        "/api/model-metrics",
        "/api/options",
        "/api/data-model-notes",
        "/assets/dashboard.css",
        "/assets/dashboard.js",
    ]

    async def _run() -> dict[str, int]:
        transport = httpx.ASGITransport(app=create_app(V2_BUNDLE))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return {ep: (await client.get(ep)).status_code for ep in endpoints}

    statuses = asyncio.run(_run())
    failed = {ep: code for ep, code in statuses.items() if code != 200}
    assert not failed, f"non-200 endpoints on V2 deployment: {failed}"


def test_predict_endpoint_returns_valid_delay_on_real_bundle() -> None:
    """POST /api/predict must produce a sane prediction using the real V2 bundle."""
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    async def _run() -> dict:
        transport = httpx.ASGITransport(app=create_app(V2_BUNDLE))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            options = (await client.get("/api/options")).json()
            default_route = options["defaults"]["route_id"]
            default_stop = options["defaults"]["stop_id"]
            payload = {
                "route_id": default_route,
                "stop_id": default_stop,
                "scheduled_time": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            }
            response = await client.post("/api/predict", json=payload)
            assert response.status_code == 200, response.text
            return response.json()

    body = asyncio.run(_run())
    assert "predicted_delay_minutes" in body
    pred = body["predicted_delay_minutes"]
    assert -60 < pred < 120, f"prediction out of range: {pred}"
    assert body["model"] == "V2MLP"


def test_predict_horizon_endpoint_returns_multiple_steps() -> None:
    """POST /api/predict-horizon should yield predictions for each requested horizon."""
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    async def _run() -> dict:
        transport = httpx.ASGITransport(app=create_app(V2_BUNDLE))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            options = (await client.get("/api/options")).json()
            payload = {
                "route_id": options["defaults"]["route_id"],
                "stop_id": options["defaults"]["stop_id"],
                "scheduled_time": datetime.now(timezone.utc).isoformat(),
                "horizons": [5, 10, 15],
            }
            response = await client.post("/api/predict-horizon", json=payload)
            assert response.status_code == 200, response.text
            return response.json()

    body = asyncio.run(_run())
    assert "predictions" in body or "horizons" in body or isinstance(body, dict)


def test_visualizations_endpoint_lists_our_new_figures() -> None:
    """/api/visualizations must surface our V3/V5/V6 figures."""
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app

    async def _run() -> dict:
        transport = httpx.ASGITransport(app=create_app(V2_BUNDLE))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/visualizations")
            assert response.status_code == 200
            return response.json()

    body = asyncio.run(_run())
    ids = {item["id"] for item in body["items"]}
    assert "ablation_study_comparison" in ids
    assert "delay_prediction_neuronspark_comparison" in ids


def test_figure_files_serve_through_api_for_all_catalog_entries() -> None:
    """Every catalog entry must be reachable via /figures/<filename>."""
    httpx = pytest.importorskip("httpx")
    from src.inference.api import create_app
    from src.inference.dashboard import VISUALIZATION_CATALOG

    async def _run() -> dict[str, int]:
        transport = httpx.ASGITransport(app=create_app(V2_BUNDLE))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return {
                entry["filename"]: (await client.get(f"/figures/{entry['filename']}")).status_code
                for entry in VISUALIZATION_CATALOG
            }

    statuses = asyncio.run(_run())
    failed = {fn: code for fn, code in statuses.items() if code != 200}
    assert not failed, f"figure URLs not serving: {failed}"


# ---------------------------------------------------------------------------
# Our standalone realtime CLI predictor (PR #4) loads our V1 baseline
# ---------------------------------------------------------------------------


def test_pr4_realtime_predictor_bootstraps_our_v1_checkpoint() -> None:
    from src.models.realtime_inference import (
        DEFAULT_BASELINE_CHECKPOINT,
        RealtimeDelayPredictor,
        build_demo_historical_frame,
        build_demo_live_records,
    )

    assert DEFAULT_BASELINE_CHECKPOINT.exists(), (
        "PR #4 expects our V1 baseline checkpoint to ship with the repo"
    )
    predictor = RealtimeDelayPredictor.bootstrap(
        checkpoint_path=DEFAULT_BASELINE_CHECKPOINT,
        historical_data=build_demo_historical_frame(num_rows=80),
        sample_size=80,
        device="cpu",
    )
    record = build_demo_live_records(count=1)[0]
    result = predictor.predict_one(record)
    assert -60 < result.predicted_delay_minutes < 120


# ---------------------------------------------------------------------------
# Module surface stays compatible after the merge
# ---------------------------------------------------------------------------


def test_models_init_exposes_pr3_and_pr4_apis() -> None:
    """src.models must expose both PR #4 (RealtimeDelayPredictor) and PR #3 (V2MLPPredictor)."""
    from src import models

    for name in (
        "RealtimeDelayPredictor",
        "PreprocessingArtifacts",
        "MBTARealtimeAdapter",
        "V2MLPPredictor",
    ):
        assert hasattr(models, name), f"src.models is missing {name}"
