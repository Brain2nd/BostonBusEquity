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
        "id": "delay_prediction_training_curves_v3_wavelet_temporal",
        "title": "V3 Time-Series Training Curves",
        "filename": "delay_prediction_training_curves_v3_wavelet_temporal.png",
        "category": "Modeling",
        "claim": "Adding lag + signal-processing features lifts R^2 from -0.07 (V1 baseline) to 0.9846 (V3 GRU).",
        "caption": "Training and validation loss converge smoothly with no overfitting, confirming feature engineering quality.",
    },
    {
        "id": "ablation_study_comparison",
        "title": "V3 Feature-Extraction Ablation",
        "filename": "ablation_study_comparison.png",
        "category": "Modeling",
        "claim": "Combined lag + rolling + FFT + wavelet features achieve the best RMSE (0.9056). Rolling statistics contribute the most among individual methods.",
        "caption": "Ablation isolates each signal-processing method using the same GRU model. All methods achieve R^2 > 0.975.",
    },
    {
        "id": "delay_prediction_multistep_comparison",
        "title": "V4 Multi-Step Prediction",
        "filename": "delay_prediction_multistep_comparison.png",
        "category": "Modeling",
        "claim": "Multi-step Seq2Seq prediction is fundamentally harder (R^2 ~ 0.08) than single-step (R^2 = 0.98) due to error accumulation.",
        "caption": "Confirms that long-horizon delay forecasting requires external context (weather, traffic) beyond historical delays.",
    },
    {
        "id": "delay_prediction_neuronspark_comparison",
        "title": "V5 NeuronSpark SNN vs GRU",
        "filename": "delay_prediction_neuronspark_comparison.png",
        "category": "Modeling",
        "claim": "NeuronSpark SNN (R^2 = 0.9897) outperforms GRU baseline on the full 3.76M-sample dataset.",
        "caption": "Spiking Neural Network with dynamic membrane parameters (beta, alpha, V_th) and K-bit binary spike encoding.",
    },
    {
        "id": "delay_prediction_training_curves_v1_baseline_temporal",
        "title": "V1 Baseline Training Curves",
        "filename": "delay_prediction_training_curves_v1_baseline_temporal.png",
        "category": "Modeling",
        "claim": "V1 with only static features fails to converge meaningfully (R^2 = -0.07), motivating temporal feature engineering.",
        "caption": "Validation loss plateaus high; the model cannot beat the sample mean.",
    },
    {
        "id": "delay_prediction_training_curves_v2_lag_features_temporal",
        "title": "V2 Historical Statistics Training Curves",
        "filename": "delay_prediction_training_curves_v2_lag_features_temporal.png",
        "category": "Modeling",
        "claim": "V2 adds historical route/stop averages but R^2 remains negative (-0.11). Static historical baselines do not capture today's conditions.",
        "caption": "Confirms that delay prediction is a short-term dynamics problem, not a static classification.",
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
    """Expose safe route/stop selections from the loaded realtime bundle.

    Bundles trained with the V4 LightGBM pipeline carry a ``stats.route_stop``
    map. Bundles built from the V2 MLP pipeline (our default deployment) do
    not. When that key is missing we fall back to pairing every route in the
    encoder with every stop, so the dropdowns still populate correctly.
    """

    route_encoder = runtime.encoders.get("route_id", {})
    stop_encoder = runtime.encoders.get("stop_id", {})
    direction_encoder = runtime.encoders.get("direction_id", {})

    route_stop_map: dict[str, set[str]] = {}
    for key in runtime.stats.get("route_stop", {}) or {}:
        route_key, separator, stop_key = str(key).partition("_")
        if not separator:
            continue
        route_value = _route_api_value(route_key)
        if stop_key in stop_encoder:
            route_stop_map.setdefault(route_value, set()).add(stop_key)

    if not route_stop_map and route_encoder and stop_encoder:
        # Fallback: bundle has no per-(route, stop) statistics. Allow every
        # encoded stop to be selected for every encoded route so users can
        # still drive the predictor; runtime input validation still rejects
        # truly unseen IDs at predict time.
        all_stop_keys = {str(stop_id) for stop_id in stop_encoder}
        for raw_route in route_encoder:
            route_stop_map[_route_api_value(raw_route)] = set(all_stop_keys)

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
    """Our V1 -> V6 experiment progression on the MBTA delay prediction task.

    Numbers come from the project's own training runs documented in
    reports/DELAY_PREDICTION_COMPARISON_REPORT.md and reports/FINAL_REPORT.md.
    """
    return {
        "summary": {
            "best_model": "Transformer (V6)",
            "best_test_R2": 0.9942,
            "best_test_RMSE": 0.4599,
            "best_test_MAE": 0.0595,
            "improvement_from_baseline_RMSE_reduction_pct": 93,
            "training_protocol": "Strict temporal split: train < 2025, test >= 2025",
            "evaluation_set": "2025-2026 MBTA arrival/departure data",
            "feature_protocol": "Past-only lag/rolling/FFT/wavelet features (no leakage)",
        },
        "experiments": [
            {
                "version": "V1",
                "name": "Baseline (static features only)",
                "best_model": "MLP",
                "RMSE": 6.24,
                "MAE": 4.38,
                "R2": -0.07,
                "key_insight": "Static features alone cannot predict delays. Negative R^2 means worse than mean prediction.",
            },
            {
                "version": "V2",
                "name": "Historical statistics",
                "best_model": "LSTM",
                "RMSE": 6.34,
                "MAE": 4.37,
                "R2": -0.11,
                "key_insight": "Route/stop historical averages do not help. Delays are non-stationary.",
            },
            {
                "version": "V3",
                "name": "Time series features (lag + FFT + wavelet)",
                "best_model": "GRU",
                "RMSE": 0.75,
                "MAE": 0.18,
                "R2": 0.9846,
                "key_insight": "Breakthrough: recent delay history (lag features) lifts R^2 by +1.05.",
            },
            {
                "version": "V4",
                "name": "Multi-step Seq2Seq forecasting",
                "best_model": "Seq2Seq-GRU",
                "RMSE": 5.72,
                "MAE": 3.90,
                "R2": 0.085,
                "key_insight": "Multi-step prediction is fundamentally harder; errors compound autoregressively.",
            },
            {
                "version": "V5",
                "name": "NeuronSpark Spiking Neural Network",
                "best_model": "SNN (D=128, K=8, 1.4M params)",
                "RMSE": 0.6098,
                "MAE": 0.3311,
                "R2": 0.9897,
                "key_insight": "Neuromorphic SNN with dynamic membrane parameters beats GRU on full data.",
            },
            {
                "version": "V6",
                "name": "Transformer attention (best)",
                "best_model": "Transformer (6L, d=128, 1.6M params)",
                "RMSE": 0.4599,
                "MAE": 0.0595,
                "R2": 0.9942,
                "key_insight": "At similar parameter scale, attention outperforms SNN by 25% RMSE.",
            },
        ],
        "ablation_v3_features": [
            {"method": "all combined", "features": 36, "RMSE": 0.9056, "R2": 0.9775},
            {"method": "rolling stats", "features": 20, "RMSE": 0.9091, "R2": 0.9774},
            {"method": "FFT", "features": 18, "RMSE": 0.9387, "R2": 0.9759},
            {"method": "wavelet", "features": 18, "RMSE": 0.9431, "R2": 0.9757},
            {"method": "baseline (lag only)", "features": 12, "RMSE": 0.9436, "R2": 0.9756},
        ],
        "active_deployment_model": {
            "name": "V2 MLP (causal lag features)",
            "reason": (
                "V2 MLP is deployed for realtime inference because its features can be "
                "reconstructed from live MBTA data without future leakage. V3-V6 models "
                "achieve higher R^2 in offline evaluation (see experiments table) but use "
                "rolling/FFT/wavelet windows that need recent delay history per route-stop."
            ),
        },
    }


def data_and_model_notes() -> dict[str, Any]:
    return {
        "data_processing": [
            "MBTA Bus Arrival/Departure data 2020-2026 (161M records, ~18 GB) downloaded via automated script with resume support.",
            "Per-month CSVs converted to a single columnar Parquet file for ~5x compression and ~10x faster reads.",
            "Timestamps parsed as UTC-aware datetimes; delay_minutes = (actual - scheduled).total_seconds() / 60.",
            "Outliers filtered to delay in [-30, 60] minutes (covers >99% of legitimate observations).",
            "Strict temporal split: training year < 2025, test year >= 2025. No random shuffling.",
            "2,910 bus stops mapped to 22 Boston neighborhoods via coordinate matching for the equity analysis.",
        ],
        "modeling": [
            "V1 baseline: 9 static temporal/categorical features. R^2 = -0.07 (worse than mean prediction).",
            "V2 historical: V1 plus route/stop/hour delay means and stds. R^2 = -0.11.",
            "V3 time series: V2 plus 5 lag features, rolling stats, FFT, db4 wavelet, statistical moments. R^2 = 0.9846 (GRU).",
            "V3 ablation study: 6 feature configurations with the same GRU; rolling statistics contribute most.",
            "V4 multi-step Seq2Seq: predicts horizon 1, 3, 5 steps; R^2 drops to 0.07-0.12 from 0.98.",
            "V5 NeuronSpark SNN: K-bit deterministic binary spike encoding, dynamic beta/alpha/V_th. R^2 = 0.9897 on full 3.76M dataset.",
            "V6 Transformer: 6 encoder layers, d_model=128, 8 heads, GELU. R^2 = 0.9942, RMSE = 0.46 min.",
        ],
        "leakage_prevention": [
            "All time series features use only past values: series.shift(k) for lags, delays[i-window:i] for rolling.",
            "Scaler.fit() called on training data only; test data only goes through .transform().",
            "Route/stop/hour historical statistics aggregated from training data; unseen ids in test fall back to global mean.",
            "FFT and wavelet windows explicitly exclude the current index i.",
        ],
        "interpretation": [
            "V3 GRU's R^2 = 0.9846 reflects strong delay autocorrelation (consecutive buses share traffic conditions).",
            "Feature engineering matters more than architecture: MLP/LSTM/GRU all reach R^2 > 0.98 with the V3 feature set.",
            "Multi-step prediction (V4) is hard because external factors (traffic, weather) become unpredictable beyond minutes.",
            "Transformer (V6) beats SNN (V5) at the same parameter count: surrogate-gradient SNN training is less sample-efficient than backprop.",
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
