"""Run a V4 model-family sweep and save the best deployable candidate."""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.models.train_delay_predictor_v4 import (
    DEFAULT_PROCESSED_PATH,
    DEFAULT_V2_BUNDLE,
    _evaluate_v2_bundle_on_test_sample,
    _metric_values,
    _read_processed_dataframe,
)
from src.models.v4_delay_predictor import (
    V4_EXPERIMENT_VERSION,
    V4_FEATURE_COLUMNS,
    V4_RUNTIME_PROFILE_ONLINE_SAFE,
    V4_RUNTIME_PROFILE_OPTIONS,
    V4_TARGET_COLUMN,
    V4Dataset,
    build_v4_encoder,
    build_v4_bundle,
    build_v4_dataset,
    build_v4_feature_frame_from_dataframe,
    evaluate_v4_model,
    make_v4_tree_model,
    normalize_v4_dataframe,
    save_v4_bundle,
)
from src.models.v2_delay_predictor import V2_FEATURE_COLUMNS

DEFAULT_CANDIDATES = [
    "historical_baseline",
    "lightgbm",
    "lightgbm_q35",
    "catboost",
    "catboost_q35",
    "xgboost",
    "xgboost_q35",
    "hist_gradient_boosting",
    "hist_gradient_boosting_q35",
    "hist_gradient_boosting_l2",
    "extra_trees",
    "ridge",
    "dummy",
]
DEFAULT_FEATURE_PROFILES = ["all", "no_ids", "v2_core", "stats_time"]
HIGH_CARDINALITY_FEATURES = {
    "route_encoded",
    "stop_encoded",
    "direction_encoded",
    "half_trip_encoded",
}
LIVE_DEFAULT_FEATURES = {
    "trip_delay_lag_1",
    "trip_delay_lag_2",
    "trip_delay_lag_3",
    "trip_delay_rolling_mean",
    "trip_delay_rolling_std",
    "trip_delay_rolling_max",
    "previous_headway_deviation",
    "vehicle_speed",
    "official_prediction_age_seconds",
}
STATS_TIME_PREFIXES = (
    "route_delay_",
    "stop_delay_",
    "hour_delay_",
    "route_hour_delay_",
    "route_stop_delay_",
    "route_stop_hour_delay_",
    "route_stop_order_delay_",
    "route_direction_hour_delay_",
)
STATS_TIME_FEATURES = {
    "is_weekend",
    "is_rush_hour",
    "scheduled_headway",
    "scheduled_headway_missing",
    "scheduled_headway_log",
    "hour_sin",
    "hour_cos",
    "minute_of_day",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "dow_sin",
    "dow_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
    "time_point_order",
    "current_stop_sequence",
    "stop_sequence_fraction",
}
DEFAULT_OUTPUT_CSV = REPORTS_DIR / "delay_prediction_v4_model_sweep.csv"
DEFAULT_SUMMARY_CSV = REPORTS_DIR / "delay_prediction_v4_model_sweep_summary.csv"
DEFAULT_REPORT = REPORTS_DIR / "V4_MODEL_SWEEP_REPORT.md"
DEFAULT_FIGURE = FIGURES_DIR / "v4_model_sweep.png"
DEFAULT_BEST_BUNDLE = PROJECT_ROOT / "models" / "delay_predictor_v4_best_online_safe_bundle.joblib"


def _optional_dependency_status() -> dict[str, bool]:
    return {
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "catboost": importlib.util.find_spec("catboost") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
    }


def _fit_candidate(
    model_kind: str,
    dataset: Any,
    feature_columns: list[str],
    feature_profile: str,
    random_state: int,
) -> tuple[Any | None, dict[str, Any]]:
    start = time.perf_counter()
    record: dict[str, Any] = {
        "model_kind_requested": model_kind,
        "feature_profile": feature_profile,
        "feature_count": len(feature_columns),
        "status": "ok",
        "error": "",
    }
    try:
        model, resolved_kind = make_v4_tree_model(model_kind, random_state=random_state)
        model.fit(
            dataset.train[feature_columns],
            dataset.train[dataset.target_column],
        )
        for split_name, split_frame in [
            ("train", dataset.train),
            ("validation", dataset.validation),
            ("test", dataset.test),
        ]:
            metrics = _evaluate_model(model, split_frame, feature_columns)
            for metric_name, metric_value in metrics.items():
                record[f"{split_name}_{metric_name}"] = metric_value
        record["model_kind"] = resolved_kind
        record["fit_seconds"] = float(time.perf_counter() - start)
        return model, record
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["fit_seconds"] = float(time.perf_counter() - start)
        return None, record


def _select_feature_columns(feature_columns: list[str], profile: str) -> list[str]:
    if profile == "all":
        return list(feature_columns)
    if profile == "v2_core":
        return [column for column in V2_FEATURE_COLUMNS if column in feature_columns]
    if profile == "no_ids":
        return [
            column
            for column in feature_columns
            if column not in HIGH_CARDINALITY_FEATURES
        ]
    if profile == "stats_time":
        return [
            column
            for column in feature_columns
            if column in STATS_TIME_FEATURES
            or any(column.startswith(prefix) for prefix in STATS_TIME_PREFIXES)
        ]
    raise ValueError(f"Unknown feature profile: {profile}")


def _evaluate_model(model: Any, dataframe: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    y_true = dataframe["delay_minutes"].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(dataframe[feature_columns]), dtype=float)
    metrics = _metric_values(y_true, y_pred)
    early_true = y_true < 0
    early_pred = y_pred < 0
    true_early_count = int(early_true.sum())
    pred_early_count = int(early_pred.sum())
    true_positive = int((early_true & early_pred).sum())
    precision = true_positive / pred_early_count if pred_early_count else 0.0
    recall = true_positive / true_early_count if true_early_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    early_mae = (
        float(np.mean(np.abs(y_true[early_true] - y_pred[early_true])))
        if true_early_count
        else np.nan
    )
    metrics.update(
        {
            "early_recall": float(recall),
            "early_precision": float(precision),
            "early_f1": float(f1),
            "early_MAE": early_mae,
            "early_share": float(true_early_count / len(y_true)) if len(y_true) else 0.0,
            "negative_prediction_rate": float(pred_early_count / len(y_pred))
            if len(y_pred)
            else 0.0,
        }
    )
    return metrics


def _sample_frame(
    dataframe: pd.DataFrame,
    limit: int | None,
    random_state: int,
) -> pd.DataFrame:
    if limit is not None and limit > 0 and len(dataframe) > limit:
        return dataframe.sample(n=limit, random_state=random_state).copy()
    return dataframe.copy()


def _build_year_dataset(
    dataframe: pd.DataFrame,
    train_years: set[int],
    validation_years: set[int],
    test_years: set[int],
    max_train_rows: int | None,
    max_validation_rows: int | None,
    max_test_rows: int | None,
    random_state: int,
    runtime_profile: str,
) -> V4Dataset:
    normalized = normalize_v4_dataframe(dataframe)
    train_raw = _sample_frame(
        normalized[normalized["year"].isin(train_years)],
        max_train_rows,
        random_state,
    )
    validation_raw = _sample_frame(
        normalized[normalized["year"].isin(validation_years)],
        max_validation_rows,
        random_state + 1,
    )
    test_raw = _sample_frame(
        normalized[normalized["year"].isin(test_years)],
        max_test_rows,
        random_state + 2,
    )
    if train_raw.empty or test_raw.empty:
        raise ValueError(
            f"Cannot build year dataset. train_years={train_years}, test_years={test_years}"
        )
    if validation_raw.empty:
        validation_raw = train_raw.sample(
            n=max(1, int(len(train_raw) * 0.15)),
            random_state=random_state + 3,
        ).copy()

    encoders = {
        "route_id": build_v4_encoder(normalized["route_id"], include_unknown=False),
        "stop_id": build_v4_encoder(normalized["stop_id"], include_unknown=False),
        "direction_id": build_v4_encoder(normalized["direction_id"], include_unknown=True),
        "half_trip_id": build_v4_encoder(train_raw["half_trip_id"], include_unknown=True),
    }
    train_features, _, stats = build_v4_feature_frame_from_dataframe(
        train_raw,
        encoders=encoders,
        runtime_profile=runtime_profile,
    )
    validation_features, _, _ = build_v4_feature_frame_from_dataframe(
        validation_raw,
        encoders=encoders,
        stats=stats,
        runtime_profile=runtime_profile,
    )
    test_features, _, _ = build_v4_feature_frame_from_dataframe(
        test_raw,
        encoders=encoders,
        stats=stats,
        runtime_profile=runtime_profile,
    )
    return V4Dataset(
        train=train_features,
        validation=validation_features,
        test=test_features,
        feature_columns=list(V4_FEATURE_COLUMNS),
        target_column=V4_TARGET_COLUMN,
        metadata={
            "encoders": encoders,
            "stats": stats,
            "data_summary": {
                "train_rows": int(len(train_features)),
                "validation_rows": int(len(validation_features)),
                "test_rows": int(len(test_features)),
                "years_present": sorted(int(year) for year in normalized["year"].unique()),
                "runtime_profile": runtime_profile,
                "train_years": sorted(train_years),
                "validation_years": sorted(validation_years),
                "test_years": sorted(test_years),
            },
        },
    )


def _plot_sweep(records: pd.DataFrame, output_path: Path, v2_mae: float | None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = records[records["status"].eq("ok")].copy()
    if ok.empty:
        return output_path
    metric_column = "test_MAE"
    metric_label = "2026 test MAE (minutes)"
    if (
        "final_2024_2025_to_2026_MAE" in ok.columns
        and ok["final_2024_2025_to_2026_MAE"].notna().any()
    ):
        metric_column = "final_2024_2025_to_2026_MAE"
        metric_label = "2026 MAE after 2024+2025 final retrain (minutes)"
    ok["label"] = (
        ok["model_kind"].astype(str).str.replace("_", " ", regex=False)
        + "\n"
        + ok["feature_profile"].astype(str)
    )
    ok = ok.sort_values(metric_column, ascending=True).head(14)

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    fig.patch.set_facecolor("#f2eee6")
    ax.set_facecolor("#fffaf4")
    colors = ["#2f6f4e" if index == 0 else "#1f5f8b" for index in range(len(ok))]
    bars = ax.barh(ok["label"], ok[metric_column], color=colors)
    ax.invert_yaxis()
    for bar in bars:
        ax.text(
            bar.get_width() + 0.04,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    if v2_mae is not None and np.isfinite(v2_mae):
        ax.axvline(v2_mae, color="#bc4749", linestyle="--", linewidth=2.0, label=f"V2 sample MAE {v2_mae:.2f}")
        ax.legend(frameon=False)
    ax.set_xlabel(metric_label)
    ax.set_title("V4 Model Family Sweep", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    fig.text(
        0.01,
        0.015,
        "Lower is better. Final retrain uses 2024+2025 labels and evaluates only on 2026.",
        fontsize=9.5,
        color="#4a5568",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_report(
    output_path: Path,
    records: pd.DataFrame,
    best_record: pd.Series | None,
    best_bundle: Path | None,
    figure_path: Path,
    dependency_status: dict[str, bool],
    data_summary: dict[str, Any],
    v2_result: dict[str, Any] | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = records[records["status"].eq("ok")].copy()
    failed = records[records["status"].ne("ok")].copy()
    v2_mae = np.nan
    if v2_result and v2_result.get("metrics"):
        v2_mae = float(v2_result["metrics"]["MAE"])
    metric_column = "test_MAE"
    if (
        "final_2024_2025_to_2026_MAE" in records.columns
        and records["final_2024_2025_to_2026_MAE"].notna().any()
    ):
        metric_column = "final_2024_2025_to_2026_MAE"
    best_test_mae = (
        float(records[records[metric_column].notna()][metric_column].min())
        if metric_column in records.columns and records[metric_column].notna().any()
        else np.nan
    )
    deployment = "not evaluated"
    if np.isfinite(v2_mae) and np.isfinite(best_test_mae):
        deployment = (
            "replace candidate after final prior-year retrain"
            if best_test_mae < v2_mae
            else "hold V2 default"
        )
    sort_column = metric_column if metric_column in ok.columns else "validation_MAE"

    lines = [
        "# V4 Model Sweep Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiment: `{V4_EXPERIMENT_VERSION}`",
        "",
        "## Dependency Status",
        "",
        f"- lightgbm installed: `{dependency_status['lightgbm']}`",
        f"- catboost installed: `{dependency_status['catboost']}`",
        f"- xgboost installed: `{dependency_status['xgboost']}`",
        "",
        "These optional boosting libraries are not installed in the current environment unless marked `True`; the sweep therefore uses sklearn candidates that can run now.",
        "",
        "## Data",
        "",
        f"- Runtime profile: `{data_summary.get('runtime_profile')}`",
        f"- Train rows: `{data_summary.get('train_rows')}`",
        f"- Validation rows: `{data_summary.get('validation_rows')}`",
        f"- Test rows: `{data_summary.get('test_rows')}`",
        f"- Years present: `{data_summary.get('years_present')}`",
        "",
        "## Results",
        "",
        "```text",
        ok.sort_values(sort_column).head(20).to_string(index=False) if not ok.empty else "No successful candidates.",
        "```",
        "",
        f"Deployment decision: `{deployment}`",
        f"Best bundle: `{best_bundle}`",
        f"Figure: `{figure_path}`",
        "",
        "## Failed Candidates",
        "",
        "```text",
        failed[["model_kind_requested", "error"]].to_string(index=False) if not failed.empty else "None",
        "```",
        "",
        "## Interpretation",
        "",
        "If the best online-safe V4 model still trails V2, the blocker is feature availability more than model class: the stateless realtime API lacks previous-stop delay, current trip adherence, and matched official residual labels. If the `final_2024_2025_to_2026_*` columns beat V2, that is a deployable 2026-style retrain using all prior-year labels, but it should still be described separately from the original 2024-only validation protocol.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_v4_model_sweep(
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    candidates: list[str] | None = None,
    feature_profiles: list[str] | None = None,
    runtime_profile: str = V4_RUNTIME_PROFILE_ONLINE_SAFE,
    max_train_rows: int | None = 50_000,
    max_validation_rows: int | None = 10_000,
    max_test_rows: int | None = 10_000,
    random_state: int = 42,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_summary: Path = DEFAULT_SUMMARY_CSV,
    output_report: Path = DEFAULT_REPORT,
    output_figure: Path = DEFAULT_FIGURE,
    output_bundle: Path = DEFAULT_BEST_BUNDLE,
    v2_bundle: Path | None = DEFAULT_V2_BUNDLE,
    v2_eval_rows: int = 2_000,
    include_validation_in_final: bool = False,
) -> dict[str, Any]:
    candidates = candidates or list(DEFAULT_CANDIDATES)
    feature_profiles = feature_profiles or list(DEFAULT_FEATURE_PROFILES)
    dataframe = _read_processed_dataframe(
        processed_path,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
    )
    dataset = build_v4_dataset(
        dataframe,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
        random_state=random_state,
        runtime_profile=runtime_profile,
    )

    models: dict[str, Any] = {}
    model_features: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []
    for feature_profile in feature_profiles:
        feature_columns = _select_feature_columns(dataset.feature_columns, feature_profile)
        for candidate in candidates:
            model, record = _fit_candidate(
                candidate,
                dataset,
                feature_columns=feature_columns,
                feature_profile=feature_profile,
                random_state=random_state,
            )
            records.append(record)
            if model is not None and record["status"] == "ok":
                model_key = f"{record['model_kind']}::{feature_profile}"
                models[model_key] = model
                model_features[model_key] = feature_columns

    records_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    records_df.to_csv(output_csv, index=False)

    v2_result = None
    v2_mae = None
    if v2_bundle is not None:
        v2_result = _evaluate_v2_bundle_on_test_sample(
            bundle_path=v2_bundle,
            test_dataframe=dataset.test,
            max_rows=v2_eval_rows,
        )
        if v2_result and v2_result.get("metrics"):
            v2_mae = float(v2_result["metrics"]["MAE"])

    ok = records_df[records_df["status"].eq("ok")].copy()
    final_best_record = None
    final_dataset = None
    if include_validation_in_final and not ok.empty:
        final_dataset = _build_year_dataset(
            dataframe=dataframe,
            train_years={2024, 2025},
            validation_years={2025},
            test_years={2026},
            max_train_rows=(max_train_rows or 0) + (max_validation_rows or 0)
            if max_train_rows is not None and max_validation_rows is not None
            else max_train_rows,
            max_validation_rows=max_validation_rows,
            max_test_rows=max_test_rows,
            random_state=random_state,
            runtime_profile=runtime_profile,
        )
        final_records = []
        for _, row in ok.iterrows():
            feature_columns = _select_feature_columns(
                final_dataset.feature_columns,
                str(row["feature_profile"]),
            )
            model, resolved_kind = make_v4_tree_model(
                str(row["model_kind_requested"]),
                random_state=random_state,
            )
            model.fit(
                final_dataset.train[feature_columns],
                final_dataset.train[final_dataset.target_column],
            )
            final_metrics = _evaluate_model(model, final_dataset.test, feature_columns)
            final_record = {
                "model_kind": resolved_kind,
                "model_kind_requested": str(row["model_kind_requested"]),
                "feature_profile": str(row["feature_profile"]),
            }
            for metric_name, metric_value in final_metrics.items():
                final_record[f"final_2024_2025_to_2026_{metric_name}"] = metric_value
            final_records.append(final_record)
        final_df = pd.DataFrame(final_records)
        records_df = records_df.merge(
            final_df,
            on=["model_kind", "model_kind_requested", "feature_profile"],
            how="left",
        )
        records_df.to_csv(output_csv, index=False)
        final_ok = records_df[
            records_df["final_2024_2025_to_2026_MAE"].notna()
        ].copy()
        if not final_ok.empty:
            final_best_record = final_ok.sort_values(
                "final_2024_2025_to_2026_MAE",
                ascending=True,
            ).iloc[0]

    best_record = None
    best_bundle_path = None
    if final_best_record is not None and final_dataset is not None:
        best_record = final_best_record
        best_key = None
        best_feature_columns = _select_feature_columns(
            final_dataset.feature_columns,
            str(best_record["feature_profile"]),
        )
        best_model, _ = make_v4_tree_model(
            str(best_record["model_kind_requested"]),
            random_state=random_state,
        )
        best_model.fit(
            final_dataset.train[best_feature_columns],
            final_dataset.train[final_dataset.target_column],
        )
        best_dataset = replace(final_dataset, feature_columns=list(best_feature_columns))
        metrics = {
            "model_kind": str(best_record["model_kind"]),
            "train": _evaluate_model(best_model, best_dataset.train, best_feature_columns),
            "validation": _evaluate_model(best_model, best_dataset.validation, best_feature_columns),
            "test": _evaluate_model(best_model, best_dataset.test, best_feature_columns),
        }
        bundle = build_v4_bundle(best_model, best_dataset, metrics)
        bundle["feature_profile"] = str(best_record["feature_profile"])
        bundle["training_protocol"] = "final_2024_2025_to_2026"
        best_bundle_path = save_v4_bundle(bundle, output_bundle)
    elif not ok.empty:
        best_record = ok.sort_values(["validation_MAE", "test_MAE"], ascending=True).iloc[0]
        best_key = f"{best_record['model_kind']}::{best_record['feature_profile']}"
        best_model = models[best_key]
        best_dataset = replace(
            dataset,
            feature_columns=list(model_features[best_key]),
        )
        metrics = {
            "model_kind": str(best_record["model_kind"]),
            "train": _row_metrics(best_record, "train"),
            "validation": _row_metrics(best_record, "validation"),
            "test": _row_metrics(best_record, "test"),
        }
        bundle = build_v4_bundle(best_model, best_dataset, metrics)
        bundle["feature_profile"] = str(best_record["feature_profile"])
        best_bundle_path = save_v4_bundle(bundle, output_bundle)

    figure_path = _plot_sweep(records_df, output_figure, v2_mae)
    report_path = _write_report(
        output_path=output_report,
        records=records_df,
        best_record=best_record,
        best_bundle=best_bundle_path,
        figure_path=figure_path,
        dependency_status=_optional_dependency_status(),
        data_summary=dataset.metadata["data_summary"],
        v2_result=v2_result,
    )

    result = {
        "csv": str(output_csv),
        "summary": str(output_summary),
        "report": str(report_path),
        "figure": str(figure_path),
        "best_bundle": str(best_bundle_path) if best_bundle_path else None,
        "best_model": str(best_record["model_kind"]) if best_record is not None else None,
        "best_feature_profile": str(best_record["feature_profile"]) if best_record is not None else None,
        "best_validation_mae": float(best_record["validation_MAE"]) if best_record is not None else None,
        "best_test_mae": float(best_record["test_MAE"]) if best_record is not None else None,
        "best_final_mae": float(best_record["final_2024_2025_to_2026_MAE"])
        if best_record is not None and "final_2024_2025_to_2026_MAE" in best_record
        and pd.notna(best_record["final_2024_2025_to_2026_MAE"])
        else None,
        "v2_sample_mae": v2_mae,
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(output_summary, index=False)
    return result


def _row_metrics(row: pd.Series, split: str) -> dict[str, float]:
    return {
        "MAE": float(row[f"{split}_MAE"]),
        "RMSE": float(row[f"{split}_RMSE"]),
        "R2": float(row[f"{split}_R2"]),
    }


def _optional_limit(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep V4 true-delay model families")
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated model kinds to try",
    )
    parser.add_argument(
        "--feature-profiles",
        default=",".join(DEFAULT_FEATURE_PROFILES),
        help="Comma-separated feature profiles: all,no_ids,v2_core,stats_time",
    )
    parser.add_argument(
        "--runtime-profile",
        default=V4_RUNTIME_PROFILE_ONLINE_SAFE,
        choices=sorted(V4_RUNTIME_PROFILE_OPTIONS),
    )
    parser.add_argument("--max-train-rows", type=int, default=50_000)
    parser.add_argument("--max-validation-rows", type=int, default=10_000)
    parser.add_argument("--max-test-rows", type=int, default=10_000)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--output-bundle", type=Path, default=DEFAULT_BEST_BUNDLE)
    parser.add_argument("--v2-bundle", type=Path, default=DEFAULT_V2_BUNDLE)
    parser.add_argument("--v2-eval-rows", type=int, default=2_000)
    parser.add_argument(
        "--include-validation-in-final",
        action="store_true",
        help="Also refit candidates on 2024+2025 and evaluate 2026 as a deployable final retrain.",
    )
    args = parser.parse_args()

    result = run_v4_model_sweep(
        processed_path=args.processed_path,
        candidates=[candidate.strip() for candidate in args.candidates.split(",") if candidate.strip()],
        feature_profiles=[profile.strip() for profile in args.feature_profiles.split(",") if profile.strip()],
        runtime_profile=args.runtime_profile,
        max_train_rows=_optional_limit(args.max_train_rows),
        max_validation_rows=_optional_limit(args.max_validation_rows),
        max_test_rows=_optional_limit(args.max_test_rows),
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        output_report=args.output_report,
        output_figure=args.output_figure,
        output_bundle=args.output_bundle,
        v2_bundle=args.v2_bundle,
        v2_eval_rows=args.v2_eval_rows,
        include_validation_in_final=args.include_validation_in_final,
    )
    print(result)


if __name__ == "__main__":
    main()
