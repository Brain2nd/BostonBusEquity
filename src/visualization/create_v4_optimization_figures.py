"""Create presentation-grade figures for V4/V5 optimization findings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from src.config import FIGURES_DIR, REPORTS_DIR

LOCAL_TIMEZONE = ZoneInfo("America/New_York")

DEFAULT_ONLINE_METRICS = REPORTS_DIR / "delay_prediction_metrics_v4.csv"
DEFAULT_HISTORY_METRICS = REPORTS_DIR / "delay_prediction_metrics_v4_history_upper_bound.csv"
DEFAULT_SWEEP_METRICS = REPORTS_DIR / "delay_prediction_v4_model_sweep.csv"
DEFAULT_SWEEP_SUMMARY = REPORTS_DIR / "delay_prediction_v4_model_sweep_summary.csv"
DEFAULT_SCORE_METRICS = REPORTS_DIR / "delay_prediction_v4_model_scores.csv"
DEFAULT_HISTORY_IMPORTANCE = REPORTS_DIR / "DELAY_PREDICTION_V4_HISTORY_UPPER_BOUND_REPORT.md"
DEFAULT_LIVE_CSV = REPORTS_DIR / "mbta_realtime_official_vs_model.csv"
DEFAULT_STORY_FIGURE = FIGURES_DIR / "v4_optimization_story.png"
DEFAULT_LIVE_FIGURE = FIGURES_DIR / "mbta_realtime_model_gap_story.png"
DEFAULT_NOTES = REPORTS_DIR / "V4_OPTIMIZATION_RESEARCH_NOTES.md"


def _read_metrics(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    for column in ["MAE", "RMSE", "R2", "n"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def _metric_value(dataframe: pd.DataFrame, model_contains: str, split_contains: str) -> float:
    rows = dataframe[
        dataframe["model"].astype(str).str.contains(model_contains, regex=False)
        & dataframe["split"].astype(str).str.contains(split_contains, regex=False)
        & dataframe["MAE"].notna()
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["MAE"])


def _feature_importance_from_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    metrics = pd.read_csv(path)
    importance_rows = metrics[metrics["scope"].astype(str).str.contains("feature")]
    if importance_rows.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    return importance_rows


def create_story_figure(
    online_metrics_path: Path = DEFAULT_ONLINE_METRICS,
    history_metrics_path: Path = DEFAULT_HISTORY_METRICS,
    sweep_metrics_path: Path = DEFAULT_SWEEP_METRICS,
    sweep_summary_path: Path = DEFAULT_SWEEP_SUMMARY,
    output_path: Path = DEFAULT_STORY_FIGURE,
    score_metrics_path: Path = DEFAULT_SCORE_METRICS,
) -> Path:
    online = _read_metrics(online_metrics_path)
    history = _read_metrics(history_metrics_path)
    sweep = _read_metrics(sweep_metrics_path) if sweep_metrics_path.exists() else pd.DataFrame()
    scores = pd.read_csv(score_metrics_path) if score_metrics_path.exists() else pd.DataFrame()

    v2_mae = _metric_value(online, "V2MLP", "test_head")
    v2_label = "Current V2\nprior sample"
    if sweep_summary_path.exists():
        summary = pd.read_csv(sweep_summary_path)
        if "v2_sample_mae" in summary.columns and pd.notna(summary["v2_sample_mae"].iloc[0]):
            v2_mae = float(summary["v2_sample_mae"].iloc[0])
            v2_label = "Current V2\nsweep sample"
    online_mae = _metric_value(online, "V4Tree", "test")
    history_mae = _metric_value(history, "V4Tree", "test")
    final_mae = float("nan")
    score_best_mae = float("nan")
    score_best_f1 = float("nan")
    score_best_label = "Score-best\nmodel"
    if not sweep.empty and "final_2024_2025_to_2026_MAE" in sweep.columns:
        final_values = pd.to_numeric(
            sweep["final_2024_2025_to_2026_MAE"],
            errors="coerce",
        ).dropna()
        if not final_values.empty:
            final_mae = float(final_values.min())
    if not scores.empty:
        score_row = scores.sort_values("composite_score", ascending=False).iloc[0]
        score_best_mae = float(score_row["primary_mae"])
        score_best_label = (
            "Latest V4\n"
            + str(score_row["model_kind"]).replace("_", "-")
            + "\n"
            + str(score_row["feature_profile"])
        )
        if "final_2024_2025_to_2026_early_f1" in score_row:
            score_best_f1 = float(score_row["final_2024_2025_to_2026_early_f1"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
    fig.patch.set_facecolor("#f2eee6")
    title_color = "#1f2933"
    blue = "#1f5f8b"
    red = "#bc4749"
    green = "#2f6f4e"
    amber = "#c47f2c"

    ax = axes[0]
    ax.set_facecolor("#fffaf4")
    labels = [
        v2_label,
        "MAE-best\nLightGBM",
        score_best_label,
        "V4 history-aware\nupper bound",
    ]
    values = [v2_mae, final_mae, score_best_mae, history_mae]
    colors = [blue, "#2a9d8f", amber, green]
    labels = [label for label, value in zip(labels, values) if np.isfinite(value)]
    colors = [color for color, value in zip(colors, values) if np.isfinite(value)]
    values = [value for value in values if np.isfinite(value)]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.07,
            f"{value:.2f} min",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.axhline(v2_mae, color=blue, linestyle="--", linewidth=1.4, alpha=0.55)
    ax.set_ylabel("MAE against actual delay (minutes)")
    ax.set_title("Deployment Decision Uses True Labels", fontweight="bold", color=title_color)
    ax.grid(True, axis="y", alpha=0.22)
    ax.text(
        1,
        max(values) * 0.82,
        "MAE-best is accurate\nbut weaker on early buses",
        ha="center",
        va="center",
        fontsize=10,
        color="#5f3b08",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff1cf", edgecolor="#e0b45c"),
    )
    ax.text(
        len(values) - 1,
        max(values) * 0.45,
        "Score-best trades\nsmall MAE loss for\nbetter early-delay F1",
        ha="center",
        va="center",
        fontsize=10,
        color="#1d4d35",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#dff3e8", edgecolor="#86b99e"),
    )

    ax = axes[1]
    ax.set_facecolor("#fffaf4")
    if not scores.empty and "final_2024_2025_to_2026_early_f1" in scores.columns:
        early_rows = scores.copy()
        early_rows["early_f1"] = pd.to_numeric(
            early_rows["final_2024_2025_to_2026_early_f1"],
            errors="coerce",
        )
        early_rows["mae"] = pd.to_numeric(early_rows["primary_mae"], errors="coerce")
        early_rows = early_rows.dropna(subset=["early_f1"]).head(8).sort_values("early_f1")
        labels = (
            early_rows["model_kind"].astype(str).str.replace("_", " ", regex=False)
            + "\n"
            + early_rows["feature_profile"].astype(str)
        )
        ax.barh(labels, early_rows["early_f1"], color=red, alpha=0.86)
        for y_index, (_, row) in enumerate(early_rows.iterrows()):
            ax.text(
                float(row["early_f1"]) + 0.01,
                y_index,
                f"F1 {row['early_f1']:.2f} | MAE {row['mae']:.2f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_xlabel("Early/negative-delay F1")
        ax.set_ylabel("Candidate")
        ax.set_title("Best Realtime Model Must Predict Early Buses Too", fontweight="bold", color=title_color)
    else:
        route_rows = history[history["scope"].eq("route_mae_top10")].copy()
        route_rows = route_rows.sort_values("MAE", ascending=True).tail(8)
        ax.barh(route_rows["group"].astype(str), route_rows["MAE"], color=red, alpha=0.86)
        ax.set_xlabel("MAE (minutes)")
        ax.set_ylabel("Route")
        ax.set_title("Hardest Routes Still Need Better Live Signals", fontweight="bold", color=title_color)
    ax.grid(True, axis="x", alpha=0.22)
    ax.text(
        0.02,
        -0.18,
        f"Interpretation: score-best model early F1 = {score_best_f1:.2f}; live accuracy still needs matched actual arrivals.",
        transform=ax.transAxes,
        fontsize=10,
        color="#4a5568",
    )

    fig.suptitle(
        "Optimizing Bus Delay Prediction: Accuracy plus Early-Bus Coverage",
        fontsize=18,
        fontweight="bold",
        color=title_color,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.92))
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_live_gap_figure(
    live_csv_path: Path = DEFAULT_LIVE_CSV,
    output_path: Path = DEFAULT_LIVE_FIGURE,
) -> Path:
    dataframe = pd.read_csv(live_csv_path)
    for column in ["observed_at", "scheduled_time", "predicted_time"]:
        dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce", utc=True)
        dataframe[column] = dataframe[column].dt.tz_convert(LOCAL_TIMEZONE)
    for column in [
        "official_delay_minutes",
        "model_predicted_delay_minutes",
        "historical_baseline_delay_minutes",
    ]:
        if column not in dataframe.columns:
            dataframe[column] = np.nan
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    dataframe["abs_gap"] = (
        dataframe["official_delay_minutes"] - dataframe["model_predicted_delay_minutes"]
    ).abs()

    latest = dataframe[dataframe["observed_at"].eq(dataframe["observed_at"].max())].copy()
    latest = latest.sort_values("scheduled_time").head(8)
    latest["signed_gap"] = (
        latest["model_predicted_delay_minutes"] - latest["official_delay_minutes"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.2, 7.8))
    axes = fig.add_gridspec(2, 1, height_ratios=[2.15, 1.0], hspace=0.28).subplots()
    fig.patch.set_facecolor("#f2eee6")
    blue = "#1f5f8b"
    red = "#bc4749"
    green = "#2f6f4e"
    amber = "#c47f2c"
    gray = "#4a5568"

    ax = axes[0]
    ax.set_facecolor("#fffaf4")
    x_positions = np.arange(len(latest))
    labels = latest["scheduled_time"].dt.strftime("%H:%M")
    width = 0.25
    official_bars = ax.bar(
        x_positions - width,
        latest["official_delay_minutes"],
        width=width,
        color=blue,
        label="MBTA official",
    )
    model_bars = ax.bar(
        x_positions,
        latest["model_predicted_delay_minutes"],
        width=width,
        color=green,
        label="Latest V4 LightGBM-q35",
    )
    baseline_bars = None
    if latest["historical_baseline_delay_minutes"].notna().any():
        baseline_bars = ax.bar(
            x_positions + width,
            latest["historical_baseline_delay_minutes"],
            width=width,
            color=amber,
            alpha=0.88,
            label="Historical baseline",
        )
    ax.axhline(0, color="#1f2933", linewidth=1.2, alpha=0.72)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delay estimate (minutes)")
    ax.set_title(
        "Latest Live Snapshot: Upcoming Trips By Scheduled Time",
        fontsize=13.5,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.legend(frameon=False, ncols=3, loc="upper right", fontsize=10.2)
    ax.grid(True, axis="y", alpha=0.22)
    ax.text(
        0.01,
        0.04,
        "Below zero = predicted early. Above zero = predicted late.",
        transform=ax.transAxes,
        fontsize=10.5,
        color=gray,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7e6", edgecolor="#e0c58f"),
    )

    for bars in [official_bars, model_bars, baseline_bars]:
        if bars is None:
            continue
        for bar in bars:
            value = float(bar.get_height())
            if not np.isfinite(value):
                continue
            va = "bottom" if value >= 0 else "top"
            offset = 0.08 if value >= 0 else -0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.1f}",
                ha="center",
                va=va,
                fontsize=8.8,
                fontweight="bold",
                color="#1f2933",
            )

    ax = axes[1]
    ax.set_facecolor("#fffaf4")
    gap_colors = np.where(latest["signed_gap"] >= 0, red, blue)
    gap_bars = ax.bar(
        x_positions,
        latest["signed_gap"],
        color=gap_colors,
        width=0.56,
        alpha=0.88,
    )
    ax.axhline(0, color="#1f2933", linewidth=1.2, alpha=0.72)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Scheduled time")
    ax.set_ylabel("Local - official\n(minutes)")
    ax.set_title(
        "Disagreement Per Trip: Positive Means Local V4 Predicts More Delay",
        fontsize=13.5,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.grid(True, axis="y", alpha=0.22)
    for bar in gap_bars:
        value = float(bar.get_height())
        va = "bottom" if value >= 0 else "top"
        offset = 0.08 if value >= 0 else -0.08
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.1f}",
            ha="center",
            va=va,
            fontsize=9.4,
            fontweight="bold",
        )

    mean_gap = float(dataframe["abs_gap"].mean())
    route_id = str(latest["route_id"].iloc[0]) if "route_id" in latest and not latest.empty else "?"
    stop_id = str(latest["stop_id"].iloc[0]) if "stop_id" in latest and not latest.empty else "?"
    observed_at = (
        latest["observed_at"].max().strftime("%Y-%m-%d %H:%M %Z")
        if not latest.empty and pd.notna(latest["observed_at"].max())
        else "latest poll"
    )
    fig.suptitle(
        f"Realtime Delay Estimates: Route {route_id} / Stop {stop_id}",
        fontsize=17.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.012,
        0.018,
        f"Snapshot observed {observed_at}. Mean absolute official/model gap: {mean_gap:.2f} min. This is disagreement, not true error; true accuracy requires matched actual arrivals.",
        fontsize=10.3,
        color=gray,
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.985, hspace=0.46)
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_research_notes(output_path: Path = DEFAULT_NOTES) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V4/V5 Optimization Research Notes",
        "",
        "## What changed",
        "",
        "- Reframed evaluation around true delay labels (`actual - scheduled`), not MBTA official prediction disagreement.",
        "- Added an `online_safe` V4 profile that removes true trip-history labels unavailable to the current stateless HTTP API.",
        "- Kept a `full_history` V4 research model as an upper bound for a future live-history cache.",
        "- Added a model-family sweep across feature profiles and a final 2024+2025 retrain for 2026 deployment testing.",
        "- Added cleaner presentation figures that separate deployment decision, research upper bound, and live model disagreement.",
        "",
        "## Evidence from web research",
        "",
        "- MBTA V3 predictions expose predicted arrival/departure, status, schedule relationship, and stop sequence; this supports live comparison but not true accuracy without later actual labels.",
        "- MBTA V3 vehicles expose live vehicle state such as current stop sequence/status and vehicle relationships; these are appropriate online features.",
        "- Jeong & Rilett (2005) emphasize AVL, schedule adherence, traffic congestion, and dwell time for real-time bus arrival prediction.",
        "- Shalaby & Farhan-style AVL/APC work separates running time and dwell time and uses recent historical/current-day information.",
        "- LightGBM/boosting remains a good tabular baseline, but the feature availability profile matters more than model class here.",
        "",
        "## Decision",
        "",
        "- The raw 2024-only `online_safe` V4 model should not replace V2; it underperformed on true-label test MAE.",
        "- The best current deployable candidate is a constrained V4 HistGradientBoosting model using V2-core causal features and a 2024+2025 final retrain.",
        "- Use the `full_history` V4 result to justify the next engineering step: capture live previous-stop/history labels or train V5 official residual correction once matched labels reach 500+ rows.",
        "",
        "## References",
        "",
        "- MBTA Prediction API: https://hexdocs.pm/mbta_sdk/MBTA.Api.Prediction.html",
        "- MBTA Vehicle API: https://hexdocs.pm/mbta_sdk/MBTA.Api.Vehicle.html",
        "- GTFS Realtime Reference: https://gtfs.org/documentation/realtime/reference/",
        "- Jeong & Rilett 2005: https://journals.sagepub.com/doi/10.1177/0361198105192700123",
        "- Shalaby & Farhan 2004: https://www.sciencedirect.com/science/article/pii/S1077291X22003812",
        "- LightGBM LGBMRegressor docs: https://lightgbm.readthedocs.io/en/v4.0.0/pythonapi/lightgbm.LGBMRegressor.html",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V4 optimization presentation figures")
    parser.add_argument("--online-metrics", type=Path, default=DEFAULT_ONLINE_METRICS)
    parser.add_argument("--history-metrics", type=Path, default=DEFAULT_HISTORY_METRICS)
    parser.add_argument("--sweep-metrics", type=Path, default=DEFAULT_SWEEP_METRICS)
    parser.add_argument("--sweep-summary", type=Path, default=DEFAULT_SWEEP_SUMMARY)
    parser.add_argument("--live-csv", type=Path, default=DEFAULT_LIVE_CSV)
    parser.add_argument("--story-figure", type=Path, default=DEFAULT_STORY_FIGURE)
    parser.add_argument("--live-figure", type=Path, default=DEFAULT_LIVE_FIGURE)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    args = parser.parse_args()

    outputs = {
        "story_figure": str(create_story_figure(args.online_metrics, args.history_metrics, args.sweep_metrics, args.sweep_summary, args.story_figure)),
        "live_figure": str(create_live_gap_figure(args.live_csv, args.live_figure)),
        "notes": str(write_research_notes(args.notes)),
    }
    print(outputs)


if __name__ == "__main__":
    main()
