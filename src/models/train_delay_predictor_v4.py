"""Train the V4 causal tree baseline for true bus-delay prediction."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.inference.runtime import DelayPredictorRuntime, PredictionInputError
from src.models.v4_delay_predictor import (
    V4_BUNDLE_PATH,
    V4_EXPERIMENT_VERSION,
    V4_FEATURE_COLUMNS,
    V4_RUNTIME_PROFILE_ONLINE_SAFE,
    V4_RUNTIME_PROFILE_OPTIONS,
    build_v4_bundle,
    extract_v4_feature_importance,
    save_v4_bundle,
    train_v4_tree_baseline_from_dataframe,
)

DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "arrival_departure.parquet"
DEFAULT_REPORT_PATH = REPORTS_DIR / "DELAY_PREDICTION_V4_OPTIMIZATION_REPORT.md"
DEFAULT_METRICS_PATH = REPORTS_DIR / "delay_prediction_metrics_v4.csv"
DEFAULT_COMPARISON_FIGURE = FIGURES_DIR / "v4_model_comparison.png"
DEFAULT_IMPORTANCE_FIGURE = FIGURES_DIR / "v4_feature_importance.png"
DEFAULT_ACTUAL_FIGURE = FIGURES_DIR / "official_vs_v4_vs_actual.png"
DEFAULT_DASHBOARD_FIGURE = FIGURES_DIR / "v4_optimization_diagnostics.png"
DEFAULT_TEST_PREDICTIONS_PATH = REPORTS_DIR / "v4_test_predictions.csv"
DEFAULT_MAX_TRAIN_ROWS = 20_000
DEFAULT_MAX_VALIDATION_ROWS = 5_000
DEFAULT_MAX_TEST_ROWS = 5_000
DEFAULT_V2_BUNDLE = (
    PROJECT_ROOT
    / "models"
    / "delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt"
)


def _read_processed_dataframe(
    path: Path,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    max_validation_rows: int | None = DEFAULT_MAX_VALIDATION_ROWS,
    max_test_rows: int | None = DEFAULT_MAX_TEST_ROWS,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed arrival/departure parquet not found: {path}")

    preferred_columns = [
        "service_date",
        "route_id",
        "direction_id",
        "half_trip_id",
        "stop_id",
        "time_point_order",
        "scheduled",
        "actual",
        "scheduled_headway",
        "year",
        "month",
    ]
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema.names)
        columns = [column for column in preferred_columns if column in available]

        if all(
            limit is not None
            for limit in [max_train_rows, max_validation_rows, max_test_rows]
        ):
            # Oversample lightly because timestamp parsing and delay clipping
            # can remove rows before the final split-specific cap is applied.
            targets = {
                2024: int((max_train_rows or 0) * 1.35),
                2025: int((max_validation_rows or 0) * 1.35),
                2026: int((max_test_rows or 0) * 1.35),
            }
            collected: dict[int, list[pd.DataFrame]] = {year: [] for year in targets}
            counts = {year: 0 for year in targets}
            row_group_years: dict[int, set[int]] = {}

            for row_group_index in range(parquet_file.num_row_groups):
                year_frame = parquet_file.read_row_group(
                    row_group_index,
                    columns=["year"] if "year" in available else ["service_date"],
                ).to_pandas()
                if "year" not in year_frame.columns:
                    year_frame["service_date"] = pd.to_datetime(
                        year_frame["service_date"],
                        errors="coerce",
                    )
                    year_frame["year"] = year_frame["service_date"].dt.year
                row_group_years[row_group_index] = {
                    int(year)
                    for year in year_frame["year"].dropna().unique()
                    if int(year) in targets
                }

            total_groups = parquet_file.num_row_groups
            for row_group_index in range(total_groups):
                frame = parquet_file.read_row_group(
                    row_group_index,
                    columns=columns,
                ).to_pandas()
                if "year" not in frame.columns:
                    frame["service_date"] = pd.to_datetime(
                        frame["service_date"],
                        errors="coerce",
                    )
                    frame["year"] = frame["service_date"].dt.year

                for year, target in targets.items():
                    needed = target - counts[year]
                    if needed <= 0:
                        continue
                    year_frame = frame[frame["year"].eq(year)]
                    if not year_frame.empty:
                        remaining_groups_for_year = sum(
                            1
                            for index in range(row_group_index, total_groups)
                            if year in row_group_years.get(index, set())
                        )
                        remaining_groups_for_year = max(remaining_groups_for_year, 1)
                        quota = max(1, int(np.ceil(needed / remaining_groups_for_year)))
                        sample_size = min(len(year_frame), quota)
                        if len(year_frame) > sample_size:
                            year_frame = year_frame.sample(
                                n=sample_size,
                                random_state=42 + year + row_group_index,
                            )
                        else:
                            year_frame = year_frame.copy()
                        collected[year].append(year_frame)
                        counts[year] += len(year_frame)

            sampled_frames = [
                part
                for year_frames in collected.values()
                for part in year_frames
            ]
            if sampled_frames:
                return pd.concat(sampled_frames, ignore_index=True)

        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path)


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _overall_metric_records(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        split_metrics = metrics[split]
        records.append(
            {
                "model": f"V4Tree-{metrics['model_kind']}",
                "scope": "overall",
                "split": split,
                "group": "all",
                "n": np.nan,
                "MAE": split_metrics["MAE"],
                "RMSE": split_metrics["RMSE"],
                "R2": split_metrics["R2"],
            }
        )
    return records


def _v4_predictions(model: Any, dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result["v4_predicted_delay_minutes"] = np.asarray(
        model.predict(result[V4_FEATURE_COLUMNS]),
        dtype=float,
    )
    result["absolute_error"] = (
        result["delay_minutes"] - result["v4_predicted_delay_minutes"]
    ).abs()
    return result


def _group_metric_records(predicted: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add_group(
        frame: pd.DataFrame,
        scope: str,
        column: str,
        limit: int | None = None,
    ) -> None:
        grouped = (
            frame.groupby(column, dropna=False)
            .agg(n=("absolute_error", "size"), MAE=("absolute_error", "mean"))
            .reset_index()
            .sort_values("MAE", ascending=False)
        )
        if limit is not None:
            grouped = grouped.head(limit)
        for row in grouped.to_dict("records"):
            records.append(
                {
                    "model": "V4Tree",
                    "scope": scope,
                    "split": "test",
                    "group": str(row[column]),
                    "n": int(row["n"]),
                    "MAE": float(row["MAE"]),
                    "RMSE": np.nan,
                    "R2": np.nan,
                }
            )

    add_group(predicted, "route_mae_top10", "route_id", limit=10)
    add_group(predicted, "stop_mae_top10", "stop_id", limit=10)
    add_group(predicted, "hour_mae", "hour", limit=None)

    buckets = pd.cut(
        predicted["delay_minutes"],
        bins=[-np.inf, -1, 1, 5, 15, np.inf],
        labels=["early", "on_time", "minor_late", "moderate_late", "major_late"],
    )
    bucketed = predicted.assign(delay_bucket=buckets)
    add_group(bucketed, "actual_delay_bucket_mae", "delay_bucket", limit=None)
    return records


def _evaluate_v2_bundle_on_test_sample(
    bundle_path: Path,
    test_dataframe: pd.DataFrame,
    max_rows: int,
) -> dict[str, Any] | None:
    if max_rows <= 0 or not bundle_path.exists():
        return None

    runtime = DelayPredictorRuntime.from_bundle_path(bundle_path)
    sample = test_dataframe.head(max_rows)
    y_true: list[float] = []
    y_pred: list[float] = []
    skipped = 0

    for _, row in sample.iterrows():
        try:
            prediction = runtime.predict(
                route_id=str(row["route_id"]),
                stop_id=str(row["stop_id"]),
                scheduled_time=pd.Timestamp(row["scheduled"]).to_pydatetime(),
                scheduled_headway=float(row["scheduled_headway"])
                if pd.notna(row["scheduled_headway"])
                else None,
                direction_id=str(row["direction_id"])
                if pd.notna(row["direction_id"])
                else None,
            )
        except PredictionInputError:
            skipped += 1
            continue
        y_true.append(float(row["delay_minutes"]))
        y_pred.append(float(prediction["predicted_delay_minutes"]))

    if not y_true:
        return {"skipped": skipped, "metrics": None, "rows": 0}

    return {
        "skipped": skipped,
        "rows": len(y_true),
        "metrics": _metric_values(np.asarray(y_true), np.asarray(y_pred)),
    }


def _compute_importance(
    model: Any,
    validation_dataframe: pd.DataFrame,
    feature_columns: list[str],
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    importance = extract_v4_feature_importance(model, feature_columns)
    if importance["importance"].sum() > 0 or validation_dataframe.empty:
        return importance

    sample = validation_dataframe
    if len(sample) > max_rows:
        sample = sample.sample(n=max_rows, random_state=random_state)
    result = permutation_importance(
        model,
        sample[feature_columns],
        sample["delay_minutes"],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": result.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _plot_model_comparison(
    metrics_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = pd.DataFrame(metrics_records)
    overall = overall[overall["scope"].eq("overall") | overall["scope"].eq("v2_sample")]
    if overall.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.patch.set_facecolor("#f7f4ec")
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        plot_df = overall.dropna(subset=[metric]).copy()
        labels = plot_df["model"] + "\n" + plot_df["split"]
        colors = ["#1f5f8b" if "V4" in label else "#bc4749" for label in labels]
        ax.bar(labels, plot_df[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("V4 True-Delay Model Performance", fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    fig.patch.set_facecolor("#f7f4ec")
    ax.barh(top["feature"], top["importance"], color="#2f6f4e")
    ax.set_title("V4 Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_actual_vs_v4(predicted: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = predicted.sort_values("scheduled").head(400).copy()
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    fig.patch.set_facecolor("#f7f4ec")
    ax.plot(
        sample["scheduled"],
        sample["delay_minutes"],
        label="Actual true delay",
        color="#1f5f8b",
        linewidth=2.0,
    )
    ax.plot(
        sample["scheduled"],
        sample["v4_predicted_delay_minutes"],
        label="V4 model",
        color="#bc4749",
        linewidth=2.0,
        alpha=0.9,
    )
    ax.set_title("Actual Delay vs V4 Prediction (Official Pending Live Labels)")
    ax.set_xlabel("Scheduled time")
    ax.set_ylabel("Delay minutes")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "MBTA official accuracy requires matched live snapshots; this offline view uses true actual labels.",
        fontsize=9,
        color="#4a4e69",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_optimization_dashboard(
    predicted: pd.DataFrame,
    importance: pd.DataFrame,
    metrics_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_records)
    compare_df = metrics_df[metrics_df["scope"].isin(["overall", "v2_sample"])].copy()
    compare_df = compare_df[
        (compare_df["split"].astype(str).str.contains("test"))
        & compare_df["MAE"].notna()
    ].copy()

    plot_df = predicted.copy()
    plot_df["residual"] = (
        plot_df["v4_predicted_delay_minutes"] - plot_df["delay_minutes"]
    )
    plot_df["delay_bucket"] = pd.cut(
        plot_df["delay_minutes"],
        bins=[-np.inf, -1, 1, 5, 15, np.inf],
        labels=["early", "on-time", "1-5 late", "5-15 late", "15+ late"],
    )

    sample = plot_df
    if len(sample) > 2500:
        sample = sample.sample(n=2500, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.patch.set_facecolor("#f2eee6")
    title_color = "#1f2933"
    accent_blue = "#1f5f8b"
    accent_red = "#bc4749"
    accent_green = "#2f6f4e"

    ax = axes[0, 0]
    ax.set_facecolor("#fffaf4")
    if not compare_df.empty:
        labels = compare_df["model"].str.replace("V4Tree-", "V4 ", regex=False)
        colors = [accent_blue if "V4" in label else accent_red for label in labels]
        bars = ax.bar(labels, compare_df["MAE"], color=colors)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_title("True-label MAE on 2026 Test", fontweight="bold", color=title_color)
    ax.set_ylabel("MAE (minutes)")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.set_facecolor("#fffaf4")
    ax.scatter(
        sample["delay_minutes"],
        sample["v4_predicted_delay_minutes"],
        s=16,
        alpha=0.35,
        color=accent_blue,
        edgecolors="none",
    )
    low = float(
        min(sample["delay_minutes"].min(), sample["v4_predicted_delay_minutes"].min())
    )
    high = float(
        max(sample["delay_minutes"].max(), sample["v4_predicted_delay_minutes"].max())
    )
    ax.plot([low, high], [low, high], color=accent_red, linewidth=2.0, label="perfect")
    ax.set_title("Prediction Calibration", fontweight="bold", color=title_color)
    ax.set_xlabel("Actual delay (minutes)")
    ax.set_ylabel("V4 predicted delay (minutes)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.set_facecolor("#fffaf4")
    bucket_mae = (
        plot_df.groupby("delay_bucket", observed=False)["absolute_error"]
        .mean()
        .reset_index()
    )
    ax.bar(bucket_mae["delay_bucket"].astype(str), bucket_mae["absolute_error"], color=accent_green)
    ax.set_title("Where Errors Are Largest", fontweight="bold", color=title_color)
    ax.set_ylabel("MAE (minutes)")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.set_facecolor("#fffaf4")
    top = importance.head(10).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color=accent_green)
    ax.set_title("Most Useful Causal Features", fontweight="bold", color=title_color)
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        "V4 Bus Delay Optimization: Online-safe True-delay Model",
        fontsize=17,
        fontweight="bold",
        color=title_color,
    )
    fig.text(
        0.01,
        0.012,
        "Ground truth is actual - scheduled. MBTA official predictions are a baseline, not labels.",
        fontsize=10,
        color="#4a5568",
    )
    plt.tight_layout(rect=(0, 0.035, 1, 0.94))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    bundle_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    metrics_records: list[dict[str, Any]],
    data_summary: dict[str, Any],
    feature_importance: pd.DataFrame,
    v2_result: dict[str, Any] | None,
    figures: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_records)
    overall = metrics_df[metrics_df["scope"].isin(["overall", "v2_sample"])].copy()
    top_features = feature_importance.head(10)

    v4_test = overall[
        overall["model"].astype(str).str.contains("V4") & overall["split"].eq("test")
    ]
    v4_mae = float(v4_test["MAE"].iloc[0]) if not v4_test.empty else np.nan
    v2_mae = np.nan
    if v2_result and v2_result.get("metrics"):
        v2_mae = float(v2_result["metrics"]["MAE"])
    acceptance = "not evaluated"
    if np.isfinite(v4_mae) and np.isfinite(v2_mae):
        acceptance = "pass" if v4_mae < v2_mae else "hold V2 default"

    lines = [
        "# Delay Prediction V4 Optimization Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Goal",
        "",
        "V4 predicts true delay (`actual - scheduled`). MBTA official predictions are a comparison baseline, not the training target.",
        "",
        "## Data Split",
        "",
        f"- Years present: `{data_summary.get('years_present')}`",
        f"- Train rows: `{data_summary.get('train_rows')}`",
        f"- Validation rows: `{data_summary.get('validation_rows')}`",
        f"- Test rows: `{data_summary.get('test_rows')}`",
        f"- Runtime profile: `{data_summary.get('runtime_profile')}`",
        "",
        "## Metrics",
        "",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        f"Acceptance decision vs current V2 sample: `{acceptance}`",
        "",
        "## Top Features",
        "",
        "```text",
        top_features.to_string(index=False),
        "```",
        "",
        "## Outputs",
        "",
        f"- Bundle: `{bundle_path}`",
        f"- Metrics CSV: `{metrics_path}`",
        f"- Test predictions CSV: `{predictions_path}`",
        f"- Comparison figure: `{figures['comparison']}`",
        f"- Feature importance figure: `{figures['importance']}`",
        f"- Actual-vs-V4 figure: `{figures['actual']}`",
        f"- Optimization dashboard: `{figures['dashboard']}`",
        "",
        "## Interpretation",
        "",
        "If V4 still trails MBTA official live predictions, that is expected: V4 is an independent true-delay model using schedule/history/live vehicle fields only. The later V5 residual model is the correct place to calibrate MBTA official predictions once enough matched live labels exist.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_v4_delay_predictor(
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    output_bundle: Path = V4_BUNDLE_PATH,
    model_kind: str = "auto",
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    max_validation_rows: int | None = DEFAULT_MAX_VALIDATION_ROWS,
    max_test_rows: int | None = DEFAULT_MAX_TEST_ROWS,
    runtime_profile: str = V4_RUNTIME_PROFILE_ONLINE_SAFE,
    v2_bundle: Path | None = DEFAULT_V2_BUNDLE,
    v2_eval_rows: int = 2_000,
    random_state: int = 42,
    report_path: Path = DEFAULT_REPORT_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    test_predictions_path: Path = DEFAULT_TEST_PREDICTIONS_PATH,
    comparison_figure: Path = DEFAULT_COMPARISON_FIGURE,
    importance_figure: Path = DEFAULT_IMPORTANCE_FIGURE,
    actual_figure: Path = DEFAULT_ACTUAL_FIGURE,
    dashboard_figure: Path = DEFAULT_DASHBOARD_FIGURE,
) -> dict[str, Any]:
    dataframe = _read_processed_dataframe(
        processed_path,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
    )
    model, dataset, metrics = train_v4_tree_baseline_from_dataframe(
        dataframe=dataframe,
        model_kind=model_kind,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
        random_state=random_state,
        runtime_profile=runtime_profile,
    )
    bundle = build_v4_bundle(model=model, dataset=dataset, metrics=metrics)
    bundle_path = save_v4_bundle(bundle, output_bundle)

    test_predictions = _v4_predictions(model, dataset.test)
    test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(test_predictions_path, index=False)
    records = _overall_metric_records(metrics)
    records.extend(_group_metric_records(test_predictions))

    v2_result = None
    if v2_bundle is not None:
        v2_result = _evaluate_v2_bundle_on_test_sample(
            bundle_path=v2_bundle,
            test_dataframe=dataset.test,
            max_rows=v2_eval_rows,
        )
        if v2_result and v2_result.get("metrics"):
            records.append(
                {
                    "model": "V2MLP-current-bundle",
                    "scope": "v2_sample",
                    "split": f"test_head_{v2_result['rows']}",
                    "group": "all",
                    "n": v2_result["rows"],
                    **v2_result["metrics"],
                }
            )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(metrics_path, index=False)

    importance = _compute_importance(
        model=model,
        validation_dataframe=dataset.validation,
        feature_columns=dataset.feature_columns,
        max_rows=5_000,
        random_state=random_state,
    )
    _plot_model_comparison(records, comparison_figure)
    _plot_feature_importance(importance, importance_figure)
    _plot_actual_vs_v4(test_predictions, actual_figure)
    _plot_optimization_dashboard(
        predicted=test_predictions,
        importance=importance,
        metrics_records=records,
        output_path=dashboard_figure,
    )
    _write_report(
        path=report_path,
        bundle_path=bundle_path,
        metrics_path=metrics_path,
        predictions_path=test_predictions_path,
        metrics_records=records,
        data_summary=dataset.metadata["data_summary"],
        feature_importance=importance,
        v2_result=v2_result,
        figures={
            "comparison": comparison_figure,
            "importance": importance_figure,
            "actual": actual_figure,
            "dashboard": dashboard_figure,
        },
    )

    return {
        "experiment": V4_EXPERIMENT_VERSION,
        "bundle_path": str(bundle_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "model_kind": metrics["model_kind"],
        "runtime_profile": runtime_profile,
        "test_mae": metrics["test"]["MAE"],
        "v2_comparison": v2_result,
    }


def _optional_limit(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the V4 causal tree baseline and realtime bundle",
    )
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output-bundle", type=Path, default=V4_BUNDLE_PATH)
    parser.add_argument(
        "--model-kind",
        default="auto",
        choices=[
            "auto",
            "lightgbm",
            "catboost",
            "xgboost",
            "hist_gradient_boosting",
            "hist_gradient_boosting_l2",
            "extra_trees",
            "random_forest",
            "gradient_boosting",
            "ridge",
            "dummy",
        ],
    )
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    parser.add_argument(
        "--max-validation-rows",
        type=int,
        default=DEFAULT_MAX_VALIDATION_ROWS,
    )
    parser.add_argument("--max-test-rows", type=int, default=DEFAULT_MAX_TEST_ROWS)
    parser.add_argument(
        "--runtime-profile",
        default=V4_RUNTIME_PROFILE_ONLINE_SAFE,
        choices=sorted(V4_RUNTIME_PROFILE_OPTIONS),
        help="Feature availability profile to train against",
    )
    parser.add_argument("--v2-bundle", type=Path, default=DEFAULT_V2_BUNDLE)
    parser.add_argument("--v2-eval-rows", type=int, default=2_000)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--test-predictions-path",
        type=Path,
        default=DEFAULT_TEST_PREDICTIONS_PATH,
    )
    parser.add_argument("--comparison-figure", type=Path, default=DEFAULT_COMPARISON_FIGURE)
    parser.add_argument("--importance-figure", type=Path, default=DEFAULT_IMPORTANCE_FIGURE)
    parser.add_argument("--actual-figure", type=Path, default=DEFAULT_ACTUAL_FIGURE)
    parser.add_argument("--dashboard-figure", type=Path, default=DEFAULT_DASHBOARD_FIGURE)
    args = parser.parse_args()

    result = train_v4_delay_predictor(
        processed_path=args.processed_path,
        output_bundle=args.output_bundle,
        model_kind=args.model_kind,
        max_train_rows=_optional_limit(args.max_train_rows),
        max_validation_rows=_optional_limit(args.max_validation_rows),
        max_test_rows=_optional_limit(args.max_test_rows),
        runtime_profile=args.runtime_profile,
        v2_bundle=args.v2_bundle,
        v2_eval_rows=args.v2_eval_rows,
        report_path=args.report_path,
        metrics_path=args.metrics_path,
        test_predictions_path=args.test_predictions_path,
        comparison_figure=args.comparison_figure,
        importance_figure=args.importance_figure,
        actual_figure=args.actual_figure,
        dashboard_figure=args.dashboard_figure,
    )
    print(result)


if __name__ == "__main__":
    main()
