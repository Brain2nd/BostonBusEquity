"""Dashboard helpers for the FastAPI user-facing project app."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.inference.log_mbta_live_snapshots import collect_live_snapshot
from src.inference.mbta_v3_client import MBTAV3Client
from src.inference.plot_mbta_realtime_comparison import apply_runtime_predictions
from src.inference.runtime import DelayPredictorRuntime

DASHBOARD_ASSET_DIR = Path(__file__).parent / "dashboard_assets"
DEFAULT_SCORE_BUNDLE = (
    PROJECT_ROOT / "models" / "delay_predictor_v4_score_best_online_safe_bundle.joblib"
)
DEFAULT_V4_BUNDLE = PROJECT_ROOT / "models" / "delay_predictor_v4_best_online_safe_bundle.joblib"
DEFAULT_V2_BUNDLE = (
    PROJECT_ROOT
    / "models"
    / "delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt"
)
LIVE_FEATURE_BUNDLE = PROJECT_ROOT / "models" / "delay_predictor_v4_tree_realtime_bundle.joblib"
LIVE_ROUTE_STOP_PRIORITIES = [
    ("1", "110"),
    ("1", "75"),
    ("1", "79"),
    ("111", "5547"),
    ("66", "412"),
    ("28", "383"),
]
SWEEP_SUMMARY_PATH = REPORTS_DIR / "delay_prediction_v4_model_sweep_summary.csv"
SWEEP_METRICS_PATH = REPORTS_DIR / "delay_prediction_v4_model_sweep.csv"
MODEL_SCORE_PATH = REPORTS_DIR / "delay_prediction_v4_model_scores.csv"
V5_REPORT_PATH = REPORTS_DIR / "V5_RESIDUAL_DATASET_REPORT.md"

PROJECT_KPIS = [
    {
        "label": "Ridership recovery gap",
        "value": "-32.8%",
        "detail": "Average post-pandemic bus ridership remains below 2016-2019 levels.",
        "source": "Final report Q1",
    },
    {
        "label": "Citywide mean delay",
        "value": "7.51 min",
        "detail": "Average delay across MBTA bus observations in the project analysis.",
        "source": "Final report Q4",
    },
    {
        "label": "On-time performance",
        "value": "31.7%",
        "detail": "Share of bus observations classified as on time in the cleaned dataset.",
        "source": "Final report Q4",
    },
    {
        "label": "Target-route delay gap",
        "value": "+41%",
        "detail": "Livable Streets target routes show higher delays than other routes.",
        "source": "Final report Q5",
    },
    {
        "label": "Best model R²",
        "value": "0.9942",
        "detail": "V6 Transformer (~1.6M params) on full dataset, RMSE=0.46 min.",
        "source": "Q8 delay prediction",
    },
    {
        "label": "NeuronSpark SNN R²",
        "value": "0.9897",
        "detail": "V5 Spiking Neural Network (~1.4M params), outperforms GRU baseline.",
        "source": "Q8 extended research",
    },
]

VISUALIZATION_CATALOG = [
    {
        "id": "mbta_realtime_model_gap_story",
        "title": "Latest Realtime Delay Estimates",
        "filename": "mbta_realtime_model_gap_story.png",
        "category": "Realtime",
        "claim": "The current dashboard compares MBTA official live predictions, the latest V4 local estimate, and a historical baseline for each upcoming trip.",
        "caption": "The top panel compares delay estimates by scheduled time; the lower panel shows Local V4 minus MBTA official disagreement. This is not ground-truth error until actual arrivals are matched.",
    },
    {
        "id": "mbta_realtime_official_vs_model",
        "title": "Latest Official vs Local Poll",
        "filename": "mbta_realtime_official_vs_model.png",
        "category": "Realtime",
        "claim": "A fresh MBTA V3 poll confirms the page is using the latest V4 LightGBM-q35 bundle.",
        "caption": "This figure is regenerated from the current MBTA live API snapshot and should be read as model disagreement rather than measured accuracy.",
    },
    {
        "id": "v4_model_sweep",
        "title": "Latest V4 Model-Family Sweep",
        "filename": "v4_model_sweep.png",
        "category": "Modeling",
        "claim": "The latest sweep compares LightGBM, CatBoost, XGBoost, sklearn boosting, historical baselines, and dummy baselines on true-delay labels.",
        "caption": "The model is evaluated against true delay labels, not against MBTA official predictions.",
    },
    {
        "id": "v4_model_deployability_scores",
        "title": "Latest Deployability Score",
        "filename": "v4_model_deployability_scores.png",
        "category": "Modeling",
        "claim": "The best deployable model is selected by accuracy, stability, online readiness, early-delay behavior, and cost, not MAE alone.",
        "caption": "This gives each candidate a defensible one-number score while keeping the component metrics visible.",
    },
    {
        "id": "v4_optimization_story",
        "title": "Latest Optimization Decision",
        "filename": "v4_optimization_story.png",
        "category": "Modeling",
        "claim": "Static online-safe features help only modestly; large gains require live trip-history or V5 residual labels.",
        "caption": "This is the main speaking figure for explaining why official predictions can outperform a stateless local model.",
    },
    {
        "id": "ablation_study_comparison",
        "title": "V3 Feature-Extraction Ablation",
        "filename": "ablation_study_comparison.png",
        "category": "Modeling",
        "claim": "Combined lag + rolling + FFT + wavelet features achieve the best RMSE (0.9056). Rolling statistics contribute the most among individual methods.",
        "caption": "Ablation isolates each signal-processing method using the same GRU model. All methods achieve R² > 0.975.",
    },
    {
        "id": "delay_prediction_neuronspark_comparison",
        "title": "V5 NeuronSpark SNN vs GRU",
        "filename": "delay_prediction_neuronspark_comparison.png",
        "category": "Modeling",
        "claim": "NeuronSpark SNN (R²=0.9897) outperforms GRU baseline on the full 3.76M-sample dataset.",
        "caption": "Spiking Neural Network with dynamic membrane parameters and K-bit binary spike encoding.",
    },
    {
        "id": "delay_prediction_training_curves_v3_wavelet_temporal",
        "title": "V3 Time-Series Training Curves",
        "filename": "delay_prediction_training_curves_v3_wavelet_temporal.png",
        "category": "Modeling",
        "claim": "Adding lag + signal-processing features lifts R² from -0.07 (V1 baseline) to 0.9846 (V3 GRU).",
        "caption": "Training and validation loss converge smoothly with no overfitting, confirming feature engineering quality.",
    },
    {
        "id": "delay_prediction_multistep_comparison",
        "title": "V4 Multi-Step Prediction",
        "filename": "delay_prediction_multistep_comparison.png",
        "category": "Modeling",
        "claim": "Multi-step Seq2Seq prediction is fundamentally harder (R²~0.08) than single-step (R²=0.98) due to error accumulation.",
        "caption": "Confirms that long-horizon delay forecasting requires external context (weather, traffic) beyond historical delays.",
    },
]


def choose_default_bundle() -> Path:
    """Prefer our team's V2 MLP bundle; fall back to V4 LightGBM if V2 missing."""
    if DEFAULT_V2_BUNDLE.exists():
        return DEFAULT_V2_BUNDLE
    if DEFAULT_SCORE_BUNDLE.exists():
        return DEFAULT_SCORE_BUNDLE
    if DEFAULT_V4_BUNDLE.exists():
        return DEFAULT_V4_BUNDLE
    return DEFAULT_V2_BUNDLE


@lru_cache(maxsize=1)
def _optional_live_feature_runtime() -> DelayPredictorRuntime | None:
    """Load the experimental V4 bundle that includes live vehicle/sequence fields."""

    if not LIVE_FEATURE_BUNDLE.exists():
        return None
    try:
        return DelayPredictorRuntime.from_bundle_path(LIVE_FEATURE_BUNDLE)
    except Exception:
        return None


def asset_path(filename: str) -> Path:
    path = DASHBOARD_ASSET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    return path


def allowed_figure_path(filename: str) -> Path:
    allowed = {item["filename"] for item in VISUALIZATION_CATALOG}
    if filename not in allowed:
        raise FileNotFoundError(filename)
    path = FIGURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    return path


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _read_sweep_summary() -> dict[str, Any]:
    if not SWEEP_SUMMARY_PATH.exists():
        return {}
    try:
        rows = pd.read_csv(SWEEP_SUMMARY_PATH)
    except Exception:
        return {}
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    for key in ["best_validation_mae", "best_test_mae", "best_final_mae", "v2_sample_mae"]:
        row[key] = _safe_float(row.get(key))
    return row


def _read_top_sweep_rows(limit: int = 12) -> list[dict[str, Any]]:
    if not SWEEP_METRICS_PATH.exists():
        return []
    try:
        dataframe = pd.read_csv(SWEEP_METRICS_PATH)
    except Exception:
        return []
    if dataframe.empty:
        return []
    sort_column = "final_2024_2025_to_2026_MAE"
    if sort_column not in dataframe.columns or dataframe[sort_column].isna().all():
        sort_column = "test_MAE"
    numeric_columns = [
        "validation_MAE",
        "test_MAE",
        "final_2024_2025_to_2026_MAE",
        "fit_seconds",
    ]
    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    records = dataframe.sort_values(sort_column, na_position="last").head(limit)
    return records.replace({np.nan: None}).to_dict("records")


def _read_top_score_rows(limit: int = 12) -> list[dict[str, Any]]:
    if not MODEL_SCORE_PATH.exists():
        return []
    try:
        dataframe = pd.read_csv(MODEL_SCORE_PATH)
    except Exception:
        return []
    if dataframe.empty or "composite_score" not in dataframe.columns:
        return []
    numeric_columns = [
        "composite_score",
        "primary_mae",
        "accuracy_score",
        "stability_score",
        "online_readiness_score",
        "early_delay_score",
        "cost_score",
    ]
    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    rows = dataframe.sort_values("composite_score", ascending=False).head(limit)
    return rows.replace({np.nan: None}).to_dict("records")


def _route_api_value(route_id: Any) -> str:
    value = str(route_id)
    if value.isdigit():
        return value.lstrip("0") or "0"
    return value


def _id_sort_key(value: str) -> tuple[int, int | str, str]:
    if value.isdigit():
        return (0, int(value), value)
    return (1, value, value)


def _sort_id_values(values: list[str]) -> list[str]:
    return sorted(values, key=_id_sort_key)


def selection_options(runtime: DelayPredictorRuntime) -> dict[str, Any]:
    """Expose safe route/stop selections from the loaded realtime bundle."""

    route_encoder = runtime.encoders.get("route_id", {})
    stop_encoder = runtime.encoders.get("stop_id", {})
    direction_encoder = runtime.encoders.get("direction_id", {})

    route_stop_map: dict[str, set[str]] = {}
    for key in runtime.stats.get("route_stop", {}):
        route_key, separator, stop_key = str(key).partition("_")
        if not separator:
            continue
        route_value = _route_api_value(route_key)
        if stop_key in stop_encoder:
            route_stop_map.setdefault(route_value, set()).add(stop_key)

    route_stop_options = {
        route: [{"value": stop, "label": stop} for stop in _sort_id_values(list(stop_ids))]
        for route, stop_ids in route_stop_map.items()
        if stop_ids
    }

    routes_by_value: dict[str, dict[str, str]] = {}
    for raw_route in route_encoder:
        api_value = _route_api_value(raw_route)
        if api_value not in route_stop_options:
            continue
        stop_count = len(route_stop_options[api_value])
        route_label = api_value if api_value == str(raw_route) else f"{api_value} ({raw_route})"
        routes_by_value.setdefault(
            api_value,
            {
                "value": api_value,
                "label": f"{route_label} - {stop_count} stops",
                "bundle_value": str(raw_route),
            },
        )

    stops = [{"value": str(stop_id), "label": str(stop_id)} for stop_id in stop_encoder]
    stops = sorted(stops, key=lambda item: _id_sort_key(item["value"]))
    route_values = _sort_id_values(list(routes_by_value))
    routes = [routes_by_value[value] for value in route_values]

    priority_default = next(
        (
            (route, stop)
            for route, stop in LIVE_ROUTE_STOP_PRIORITIES
            if route in routes_by_value
            and any(item["value"] == stop for item in route_stop_options.get(route, []))
        ),
        None,
    )
    default_route = (
        priority_default[0]
        if priority_default is not None
        else ("1" if "1" in routes_by_value else (routes[0]["value"] if routes else ""))
    )
    default_stops = route_stop_options.get(default_route, stops)
    default_stop_values = {item["value"] for item in default_stops}
    default_stop = (
        priority_default[1]
        if priority_default is not None
        else ("110" if "110" in default_stop_values else (
            default_stops[0]["value"] if default_stops else ""
        ))
    )

    directions = [
        {"value": str(direction_id), "label": str(direction_id)}
        for direction_id in _sort_id_values([str(value) for value in direction_encoder])
    ]

    return {
        "routes": routes,
        "stops": stops,
        "route_stop_map": route_stop_options,
        "directions": directions,
        "defaults": {
            "route_id": default_route,
            "stop_id": default_stop,
            "direction_id": "Unknown"
            if any(item["value"] == "Unknown" for item in directions)
            else (directions[0]["value"] if directions else ""),
        },
    }


def project_summary(runtime: DelayPredictorRuntime) -> dict[str, Any]:
    summary = _read_sweep_summary()
    best_final_mae = summary.get("best_final_mae")
    v2_sample_mae = summary.get("v2_sample_mae")
    model_delta = None
    if best_final_mae is not None and v2_sample_mae is not None:
        model_delta = float(best_final_mae - v2_sample_mae)
    kpis = list(PROJECT_KPIS)
    if best_final_mae is not None:
        kpis.append(
            {
                "label": "Best local model MAE",
                "value": f"{best_final_mae:.2f} min",
                "detail": "Best V4 final prior-year retrain evaluated on 2026 true delay labels.",
                "source": "V4 model sweep",
            }
        )
    return {
        "title": "Boston Bus Equity: Realtime Delay Prediction Dashboard",
        "subtitle": "Service equity analysis plus local true-delay prediction for MBTA buses.",
        "kpis": kpis,
        "model": {
            "health": runtime.health(),
            "best_model": summary.get("best_model"),
            "best_feature_profile": summary.get("best_feature_profile"),
            "best_final_mae": best_final_mae,
            "v2_sample_mae": v2_sample_mae,
            "delta_vs_v2": model_delta,
            "accuracy_note": "MAE is measured against true delay actual - scheduled. MBTA official comparison is not ground truth.",
        },
        "data": {
            "processed_parquet": str(PROJECT_ROOT / "data" / "processed" / "arrival_departure.parquet"),
            "raw_years": [2024, 2025, 2026],
            "target_label": "delay_minutes = actual - scheduled",
        },
    }


def visualizations() -> dict[str, Any]:
    items = []
    for item in VISUALIZATION_CATALOG:
        path = FIGURES_DIR / item["filename"]
        enriched = dict(item)
        enriched["available"] = path.exists()
        version = int(path.stat().st_mtime) if path.exists() else 0
        enriched["url"] = f"/figures/{item['filename']}?v={version}"
        items.append(enriched)
    return {"items": items}


def model_metrics() -> dict[str, Any]:
    return {
        "summary": _read_sweep_summary(),
        "sweep_rows": _read_top_sweep_rows(limit=16),
        "score_rows": _read_top_score_rows(limit=16),
        "scoring": {
            "available": MODEL_SCORE_PATH.exists(),
            "csv": str(MODEL_SCORE_PATH),
            "weights": {
                "accuracy": 0.45,
                "stability": 0.20,
                "online_readiness": 0.15,
                "early_delay_behavior": 0.10,
                "cost": 0.10,
            },
        },
        "v5": {
            "status": "collecting_labels" if V5_REPORT_PATH.exists() else "not_started",
            "note": "V5 will correct MBTA official predictions after enough live predictions are matched to later actual arrivals.",
        },
    }


def data_and_model_notes() -> dict[str, Any]:
    return {
        "data_processing": [
            "Download MBTA Bus Arrival/Departure CSV files for selected years.",
            "Convert monthly CSV files into one columnar arrival_departure.parquet file.",
            "Parse scheduled and actual timestamps, normalize route/stop/direction ids, and compute delay_minutes.",
            "Filter extreme delay outliers for model training and split by time so future labels do not leak into training.",
        ],
        "modeling": [
            "The earlier neural baseline used 18 causal features: time flags, encoded route/stop/direction, scheduled headway, and training-period historical delay statistics.",
            "V4 runs a model-family sweep over LightGBM, CatBoost, XGBoost, sklearn boosting, trees, ridge, and dummy baselines.",
            "The prediction chart now prefers MBTA live prediction and vehicle rows over a synthetic time grid, so realtime output changes with current official predictions and vehicle stop sequence when those fields are available.",
            "The live dashboard includes a route-stop-hour historical mean baseline so the local model is not compared only to MBTA official predictions.",
            "The dashboard defaults to the best online-safe V4 bundle when available and falls back to the V2 realtime bundle otherwise.",
            "V5 is intentionally not enabled until live MBTA official predictions are matched to subsequent actual arrivals.",
        ],
        "interpretation": [
            "Official-vs-local live charts show model disagreement, not accuracy.",
            "The main local realtime line is a V5 preview: official live delay plus the latest V4 model's residual from historical baseline; it is useful for demonstration but not an accepted model-quality claim yet.",
            "True accuracy is reported only where actual arrival/departure labels exist.",
            "Scheduled headway is entered in minutes in the UI and converted to the seconds scale used by the trained MBTA historical features.",
            "Large realtime gains likely require vehicle state, current stop sequence, previous-stop delay, and official residual labels.",
        ],
    }


def live_compare(
    runtime: DelayPredictorRuntime,
    route_id: str,
    stop_id: str,
    direction_id: str | int | None = None,
    prediction_limit: int = 8,
    vehicle_limit: int = 100,
) -> dict[str, Any]:
    client = MBTAV3Client()
    requested_route_id = str(route_id)
    requested_stop_id = str(stop_id)
    try:
        snapshots = collect_live_snapshot(
            client=client,
            route_id=route_id,
            stop_id=stop_id,
            direction_id=direction_id,
            prediction_limit=prediction_limit,
            vehicle_limit=vehicle_limit,
        )
    except requests.RequestException as exc:
        return {
            "mode": "network_error",
            "message": f"MBTA API request failed: {exc}",
            "rows": [],
        }
    except Exception as exc:
        return {
            "mode": "unavailable",
            "message": str(exc),
            "rows": [],
        }

    merged = snapshots["merged"]
    if merged.empty:
        fallback = _find_live_prediction_fallback(
            client=client,
            requested=(requested_route_id, requested_stop_id),
            direction_id=direction_id,
            prediction_limit=prediction_limit,
            vehicle_limit=vehicle_limit,
        )
        if fallback is None:
            return {
                "mode": "no_predictions",
                "message": "MBTA returned no current predictions for this route and stop.",
                "rows": [],
                "requested_route_id": requested_route_id,
                "requested_stop_id": requested_stop_id,
            }
        route_id, stop_id, snapshots = fallback
        merged = snapshots["merged"]

    compared = apply_runtime_predictions(merged, runtime=runtime)
    rows = []
    for row in compared.replace({np.nan: None}).to_dict("records"):
        baseline = None
        baseline_source = ""
        if row.get("scheduled_time") is not None and not pd.isna(row.get("scheduled_time")):
            try:
                baseline_result = runtime.historical_baseline_delay(
                    route_id=str(row.get("route_id")),
                    stop_id=str(row.get("stop_id")),
                    scheduled_time=pd.Timestamp(row.get("scheduled_time")).to_pydatetime(),
                    direction_id=None
                    if row.get("direction_id") is None or pd.isna(row.get("direction_id"))
                    else str(row.get("direction_id")),
                )
                baseline = baseline_result["predicted_delay_minutes"]
                baseline_source = baseline_result["source"]
            except Exception:
                baseline = None

        official_delay = _safe_float(row.get("official_delay_minutes"))
        model_delay = _safe_float(row.get("model_predicted_delay_minutes"))
        official_informed_delay = official_delay
        if official_informed_delay is None:
            official_informed_delay = model_delay if model_delay is not None else baseline

        rows.append(
            {
                "observed_at": _timestamp_to_iso(row.get("observed_at")),
                "scheduled_time": _timestamp_to_iso(row.get("scheduled_time")),
                "predicted_time": _timestamp_to_iso(row.get("predicted_time")),
                "route_id": row.get("route_id"),
                "stop_id": row.get("stop_id"),
                "direction_id": row.get("direction_id"),
                "trip_id": row.get("trip_id"),
                "vehicle_id": row.get("vehicle_id"),
                "official_delay_minutes": official_delay,
                "model_predicted_delay_minutes": model_delay,
                "historical_baseline_delay_minutes": baseline,
                "historical_baseline_source": baseline_source,
                "official_informed_delay_minutes": official_informed_delay,
                "scheduled_headway_minutes": _safe_float(row.get("scheduled_headway_minutes")),
                "current_stop_sequence": _safe_float(row.get("current_stop_sequence")),
                "vehicle_speed": _safe_float(row.get("speed")),
                "model_error": row.get("model_error") or "",
                "model_used_defaults": row.get("model_used_defaults") or "",
            }
        )

    comparable = [
        row
        for row in rows
        if row["official_delay_minutes"] is not None
        and row["model_predicted_delay_minutes"] is not None
    ]
    mean_abs_gap = None
    if comparable:
        mean_abs_gap = float(
            np.mean(
                [
                    abs(row["official_delay_minutes"] - row["model_predicted_delay_minutes"])
                    for row in comparable
                ]
            )
        )

    return {
        "mode": "official_vs_model" if comparable else "official_only",
        "message": (
            "Comparison produced. Independent V4 and historical baseline are local "
            "estimates; official-informed forecast uses MBTA live prediction when "
            "available and is not an independent accuracy claim."
        ),
        "mean_abs_gap_minutes": mean_abs_gap,
        "requested_route_id": requested_route_id,
        "requested_stop_id": requested_stop_id,
        "route_id": str(route_id),
        "stop_id": str(stop_id),
        "fallback_used": (str(route_id), str(stop_id)) != (requested_route_id, requested_stop_id),
        "rows": rows,
        "generated_at": datetime.now().isoformat(),
    }


def _find_live_prediction_fallback(
    client: MBTAV3Client,
    requested: tuple[str, str],
    direction_id: str | int | None,
    prediction_limit: int,
    vehicle_limit: int,
) -> tuple[str, str, dict[str, pd.DataFrame]] | None:
    for route_id, stop_id in LIVE_ROUTE_STOP_PRIORITIES:
        if (route_id, stop_id) == requested:
            continue
        try:
            snapshots = collect_live_snapshot(
                client=client,
                route_id=route_id,
                stop_id=stop_id,
                direction_id=direction_id,
                prediction_limit=prediction_limit,
                vehicle_limit=vehicle_limit,
            )
        except Exception:
            continue
        if not snapshots["merged"].empty:
            return route_id, stop_id, snapshots
    return None


def live_enriched_forecast(
    runtime: DelayPredictorRuntime,
    route_id: str,
    stop_id: str,
    direction_id: str | int | None = None,
    prediction_limit: int = 10,
    vehicle_limit: int = 100,
) -> dict[str, Any]:
    """Use MBTA live rows to avoid stateless synthetic-horizon flatlines.

    This keeps the independent V4 model visible, but the dashboard also shows
    official live estimates and an explicitly labeled V5 preview that combines
    the official live signal with the local model's residual from historical
    baseline. The preview is not treated as validated accuracy until matched
    actual labels exist.
    """

    client = MBTAV3Client()
    requested_route_id = str(route_id)
    requested_stop_id = str(stop_id)
    try:
        snapshots = collect_live_snapshot(
            client=client,
            route_id=route_id,
            stop_id=stop_id,
            direction_id=direction_id,
            prediction_limit=prediction_limit,
            vehicle_limit=vehicle_limit,
        )
    except requests.RequestException as exc:
        return {
            "mode": "network_error",
            "message": f"MBTA API request failed: {exc}",
            "rows": [],
        }
    except Exception as exc:
        return {
            "mode": "unavailable",
            "message": str(exc),
            "rows": [],
        }

    merged = snapshots["merged"]
    if merged.empty:
        fallback = _find_live_prediction_fallback(
            client=client,
            requested=(requested_route_id, requested_stop_id),
            direction_id=direction_id,
            prediction_limit=prediction_limit,
            vehicle_limit=vehicle_limit,
        )
        if fallback is None:
            return {
                "mode": "no_predictions",
                "message": "MBTA returned no live predictions, so the dashboard used the synthetic local horizon.",
                "rows": [],
                "requested_route_id": requested_route_id,
                "requested_stop_id": requested_stop_id,
            }
        route_id, stop_id, snapshots = fallback
        merged = snapshots["merged"]

    independent = apply_runtime_predictions(merged, runtime=runtime)
    live_feature_runtime = _optional_live_feature_runtime()
    live_feature = (
        apply_runtime_predictions(merged, runtime=live_feature_runtime)
        if live_feature_runtime is not None
        else None
    )

    rows = []
    for index, row in independent.replace({np.nan: None}).iterrows():
        official_delay = _safe_float(row.get("official_delay_minutes"))
        independent_delay = _safe_float(row.get("model_predicted_delay_minutes"))
        baseline_delay = _safe_float(row.get("historical_baseline_delay_minutes"))
        live_feature_delay = None
        live_feature_defaults = ""
        if live_feature is not None and index in live_feature.index:
            live_row = live_feature.loc[index]
            live_feature_delay = _safe_float(live_row.get("model_predicted_delay_minutes"))
            live_feature_defaults = live_row.get("model_used_defaults") or ""

        live_calibrated = None
        if (
            official_delay is not None
            and independent_delay is not None
            and baseline_delay is not None
        ):
            live_calibrated = official_delay + (independent_delay - baseline_delay)

        rows.append(
            {
                "observed_at": _timestamp_to_iso(row.get("observed_at")),
                "scheduled_time": _timestamp_to_iso(row.get("scheduled_time")),
                "predicted_time": _timestamp_to_iso(row.get("predicted_time")),
                "route_id": row.get("route_id"),
                "stop_id": row.get("stop_id"),
                "direction_id": row.get("direction_id"),
                "trip_id": row.get("trip_id"),
                "vehicle_id": row.get("vehicle_id"),
                "official_delay_minutes": official_delay,
                "independent_v4_delay_minutes": independent_delay,
                "historical_baseline_delay_minutes": baseline_delay,
                "live_feature_v4_delay_minutes": live_feature_delay,
                "live_calibrated_delay_minutes": live_calibrated,
                "current_stop_sequence": _safe_float(row.get("current_stop_sequence")),
                "vehicle_speed": _safe_float(row.get("speed")),
                "scheduled_headway_minutes": _safe_float(row.get("scheduled_headway_minutes")),
                "independent_defaults": row.get("model_used_defaults") or "",
                "live_feature_defaults": live_feature_defaults,
            }
        )

    return {
        "mode": "live_enriched",
        "message": (
            "Using MBTA V3 live predictions/vehicles instead of a synthetic time grid. "
            "V5 preview = MBTA official live delay + independent local residual from "
            "the historical baseline; it is not a validated accuracy claim yet."
        ),
        "live_feature_bundle_loaded": live_feature_runtime is not None,
        "requested_route_id": requested_route_id,
        "requested_stop_id": requested_stop_id,
        "route_id": str(route_id),
        "stop_id": str(stop_id),
        "fallback_used": (str(route_id), str(stop_id)) != (requested_route_id, requested_stop_id),
        "rows": rows,
        "generated_at": datetime.now().isoformat(),
    }


def _timestamp_to_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)
