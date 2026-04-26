"""Poll MBTA V3 realtime predictions and plot official vs local model delays."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.inference.mbta_v3_client import MBTAV3Client
from src.inference.runtime import DelayPredictorRuntime, PredictionInputError

LOCAL_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_FIGURE_PATH = FIGURES_DIR / "mbta_realtime_official_vs_model.png"
DEFAULT_REPORT_PATH = REPORTS_DIR / "MBTA_REALTIME_OFFICIAL_VS_MODEL.md"
DEFAULT_CSV_PATH = REPORTS_DIR / "mbta_realtime_official_vs_model.csv"


def apply_runtime_predictions(
    dataframe: pd.DataFrame,
    runtime: DelayPredictorRuntime | None,
) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["model_predicted_delay_minutes"] = np.nan
    enriched["historical_baseline_delay_minutes"] = np.nan
    enriched["official_informed_delay_minutes"] = np.nan
    enriched["model_error"] = ""
    enriched["model_used_defaults"] = ""

    if runtime is None or enriched.empty:
        return enriched

    for index, row in enriched.iterrows():
        scheduled_time = row.get("scheduled_time")
        if pd.isna(scheduled_time):
            enriched.at[index, "model_error"] = "Missing scheduled_time in MBTA payload"
            continue

        scheduled_headway = row.get("scheduled_headway_minutes")
        if pd.isna(scheduled_headway):
            scheduled_headway = None

        direction_id = row.get("direction_id")
        if pd.isna(direction_id):
            direction_id = None
        elif direction_id is not None:
            direction_id = str(direction_id)
            if direction_id not in runtime.encoders["direction_id"]:
                direction_id = None

        try:
            baseline = runtime.historical_baseline_delay(
                route_id=str(row["route_id"]),
                stop_id=str(row["stop_id"]),
                scheduled_time=pd.Timestamp(scheduled_time).to_pydatetime(),
                direction_id=direction_id,
            )
            enriched.at[index, "historical_baseline_delay_minutes"] = baseline[
                "predicted_delay_minutes"
            ]
            prediction = runtime.predict(
                route_id=str(row["route_id"]),
                stop_id=str(row["stop_id"]),
                scheduled_time=pd.Timestamp(scheduled_time).to_pydatetime(),
                scheduled_headway=None if scheduled_headway is None else float(scheduled_headway),
                direction_id=direction_id,
                trip_id=None if pd.isna(row.get("trip_id")) else str(row.get("trip_id")),
                vehicle_id=None if pd.isna(row.get("vehicle_id")) else str(row.get("vehicle_id")),
                current_stop_sequence=None
                if pd.isna(row.get("current_stop_sequence"))
                else float(row.get("current_stop_sequence")),
                vehicle_speed=None
                if pd.isna(row.get("speed"))
                else float(row.get("speed")),
                vehicle_status=None if pd.isna(row.get("status")) else str(row.get("status")),
                official_predicted_delay_minutes=None
                if pd.isna(row.get("official_delay_minutes"))
                else float(row.get("official_delay_minutes")),
                official_prediction_age_seconds=None,
            )
            enriched.at[index, "model_predicted_delay_minutes"] = prediction[
                "predicted_delay_minutes"
            ]
            if pd.notna(row.get("official_delay_minutes")):
                enriched.at[index, "official_informed_delay_minutes"] = float(
                    row.get("official_delay_minutes")
                )
            else:
                enriched.at[index, "official_informed_delay_minutes"] = prediction[
                    "predicted_delay_minutes"
                ]
            enriched.at[index, "model_used_defaults"] = ",".join(prediction["used_defaults"])
        except PredictionInputError as exc:
            enriched.at[index, "model_error"] = str(exc)

    return enriched


def poll_prediction_snapshots(
    client: MBTAV3Client,
    route_id: str,
    stop_id: str,
    direction_id: str | int | None,
    poll_count: int,
    poll_interval_seconds: float,
    page_limit: int,
) -> pd.DataFrame:
    snapshots: list[pd.DataFrame] = []
    for poll_index in range(poll_count):
        snapshot = client.fetch_predictions_dataframe(
            route_id=route_id,
            stop_id=stop_id,
            direction_id=direction_id,
            limit=page_limit,
        )
        snapshot["poll_index"] = poll_index + 1
        snapshots.append(snapshot)
        if poll_index < poll_count - 1:
            time.sleep(poll_interval_seconds)

    if not snapshots:
        return pd.DataFrame()
    return pd.concat(snapshots, ignore_index=True)


def _rolling_top_prediction(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    rows: list[pd.Series] = []
    for _, group_df in dataframe.groupby("observed_at", sort=True):
        ordered = group_df.sort_values(
            by=["prediction_rank", "scheduled_time", "predicted_time"],
            na_position="last",
        )
        rows.append(ordered.iloc[0])
    return pd.DataFrame(rows)


def _runtime_model_label(runtime: DelayPredictorRuntime | None) -> str:
    if runtime is None:
        return "Local model"
    health = runtime.health()
    if health.get("model") == "V4Tree":
        kind = str(health.get("model_kind") or "tree").replace("_", "-")
        return f"Latest V4 {kind}"
    return f"{health.get('model', 'Local model')} ({health.get('experiment', 'latest')})"


def _plot_comparison_figure(
    dataframe: pd.DataFrame,
    output_path: Path,
    route_id: str,
    stop_id: str,
    compare_enabled: bool,
    model_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    fig.patch.set_facecolor("#f4efe7")

    latest_snapshot = dataframe[dataframe["observed_at"] == dataframe["observed_at"].max()].copy()
    latest_snapshot = latest_snapshot.sort_values("scheduled_time", na_position="last")

    left_ax, right_ax = axes
    left_ax.set_facecolor("#fffaf4")
    right_ax.set_facecolor("#fffaf4")

    if not latest_snapshot.empty:
        left_ax.plot(
            latest_snapshot["scheduled_time"],
            latest_snapshot["official_delay_minutes"],
            marker="o",
            linewidth=2.2,
            color="#1f5f8b",
            label="MBTA official",
        )
        if compare_enabled and latest_snapshot["model_predicted_delay_minutes"].notna().any():
            left_ax.plot(
                latest_snapshot["scheduled_time"],
                latest_snapshot["model_predicted_delay_minutes"],
                marker="s",
                linewidth=2.2,
                color="#2f6f4e",
                label=model_label,
            )
        if latest_snapshot.get("historical_baseline_delay_minutes") is not None and latest_snapshot["historical_baseline_delay_minutes"].notna().any():
            left_ax.plot(
                latest_snapshot["scheduled_time"],
                latest_snapshot["historical_baseline_delay_minutes"],
                linestyle="--",
                linewidth=2.0,
                color="#c47f2c",
                label="Historical baseline",
            )
        left_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        left_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TIMEZONE))

    left_ax.set_title("Latest Snapshot By Scheduled Time")
    left_ax.set_xlabel("Scheduled Time")
    left_ax.set_ylabel("Delay (minutes)")
    left_ax.legend(frameon=False, loc="best")

    rolling_df = _rolling_top_prediction(dataframe)
    if not rolling_df.empty:
        right_ax.plot(
            rolling_df["observed_at"],
            rolling_df["official_delay_minutes"],
            marker="o",
            linewidth=2.2,
            color="#1f5f8b",
            label="MBTA official",
        )
        if compare_enabled and rolling_df["model_predicted_delay_minutes"].notna().any():
            right_ax.plot(
                rolling_df["observed_at"],
                rolling_df["model_predicted_delay_minutes"],
                marker="s",
                linewidth=2.2,
                color="#2f6f4e",
                label=model_label,
            )
        if rolling_df.get("historical_baseline_delay_minutes") is not None and rolling_df["historical_baseline_delay_minutes"].notna().any():
            right_ax.plot(
                rolling_df["observed_at"],
                rolling_df["historical_baseline_delay_minutes"],
                linestyle="--",
                linewidth=2.0,
                color="#c47f2c",
                label="Historical baseline",
            )
        right_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        right_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=LOCAL_TIMEZONE))

    right_ax.set_title("Earliest Upcoming Trip Across Polls")
    right_ax.set_xlabel("Observed Time")
    right_ax.set_ylabel("Delay (minutes)")
    right_ax.legend(frameon=False, loc="best")

    mae_text = "official-only"
    if compare_enabled:
        comparable_df = dataframe.dropna(
            subset=["official_delay_minutes", "model_predicted_delay_minutes"]
        )
        if not comparable_df.empty:
            mae_value = float(
                np.mean(
                    np.abs(
                        comparable_df["official_delay_minutes"]
                        - comparable_df["model_predicted_delay_minutes"]
                    )
                )
            )
            mae_text = f"mean abs gap={mae_value:.2f} min"
        else:
            mae_text = "model comparison unavailable for current bundle ids"

    fig.suptitle(
        f"MBTA Live Predictions vs Latest Local Model: route {route_id} / stop {stop_id}",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.015,
        0.02,
        f"{mae_text}. Local line uses {model_label}; dashed line is a historical baseline. Full protocol is in the report.",
        fontsize=10,
        color="#4a4e69",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    figure_path: Path,
    csv_path: Path,
    dataframe: pd.DataFrame,
    route_id: str,
    stop_id: str,
    direction_id: str | int | None,
    poll_count: int,
    poll_interval_seconds: float,
    page_limit: int,
    bundle_path: Path | None,
    compare_enabled: bool,
    model_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    observed_count = dataframe["observed_at"].nunique() if not dataframe.empty else 0
    lines = [
        "# MBTA Realtime Official vs Local Model",
        "",
        f"Generated: {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Configuration",
        "",
        f"- Route: `{route_id}`",
        f"- Stop: `{stop_id}`",
        f"- Direction: `{direction_id}`",
        f"- Poll count: `{poll_count}`",
        f"- Poll interval seconds: `{poll_interval_seconds}`",
        f"- Page limit: `{page_limit}`",
        f"- Bundle: `{bundle_path}`",
        f"- Local model label: `{model_label}`",
        f"- Figure: `{figure_path}`",
        f"- CSV: `{csv_path}`",
        "",
        "## Summary",
        "",
        f"- Snapshot count: `{observed_count}`",
        f"- Prediction rows: `{len(dataframe)}`",
        f"- Mode: `{'official_vs_model' if compare_enabled else 'official_only'}`",
    ]

    if not dataframe.empty:
        lines.extend(
            [
                f"- First observed at: `{dataframe['observed_at'].min()}`",
                f"- Last observed at: `{dataframe['observed_at'].max()}`",
            ]
        )

    comparable_df = dataframe.dropna(
        subset=["official_delay_minutes", "model_predicted_delay_minutes"]
    )
    if compare_enabled:
        lines.extend(["", "## Comparison Metrics", ""])
        if comparable_df.empty:
            lines.append("- No comparable official/model rows were produced for this route-stop-bundle combination.")
        else:
            mae_value = float(
                np.mean(
                    np.abs(
                        comparable_df["official_delay_minutes"]
                        - comparable_df["model_predicted_delay_minutes"]
                    )
                )
            )
            lines.append(f"- Mean absolute gap: `{mae_value:.3f}` minutes")
            lines.append(
                f"- Comparable rows: `{len(comparable_df)}` / `{len(dataframe)}`"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_mbta_realtime_comparison(
    route_id: str,
    stop_id: str,
    direction_id: str | int | None,
    bundle_path: Path | None,
    figure_path: Path,
    report_path: Path,
    csv_path: Path,
    poll_count: int,
    poll_interval_seconds: float,
    page_limit: int,
) -> dict[str, Any]:
    client = MBTAV3Client()
    runtime = None
    compare_enabled = False
    if bundle_path is not None:
        resolved_bundle_path = bundle_path.resolve()
        if not resolved_bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {resolved_bundle_path}")
        runtime = DelayPredictorRuntime.from_bundle_path(resolved_bundle_path)
        compare_enabled = True
    else:
        resolved_bundle_path = None
    model_label = _runtime_model_label(runtime)

    dataframe = poll_prediction_snapshots(
        client=client,
        route_id=route_id,
        stop_id=stop_id,
        direction_id=direction_id,
        poll_count=poll_count,
        poll_interval_seconds=poll_interval_seconds,
        page_limit=page_limit,
    )
    if dataframe.empty:
        raise ValueError("MBTA API returned no prediction rows for the requested route/stop.")

    dataframe = apply_runtime_predictions(dataframe, runtime=runtime)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)

    _plot_comparison_figure(
        dataframe=dataframe,
        output_path=figure_path,
        route_id=route_id,
        stop_id=stop_id,
        compare_enabled=compare_enabled,
        model_label=model_label,
    )
    _write_report(
        output_path=report_path,
        figure_path=figure_path,
        csv_path=csv_path,
        dataframe=dataframe,
        route_id=route_id,
        stop_id=stop_id,
        direction_id=direction_id,
        poll_count=poll_count,
        poll_interval_seconds=poll_interval_seconds,
        page_limit=page_limit,
        bundle_path=resolved_bundle_path,
        compare_enabled=compare_enabled,
        model_label=model_label,
    )

    return {
        "figure_path": str(figure_path),
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "compare_enabled": compare_enabled,
        "rows": int(len(dataframe)),
        "polls": int(dataframe["observed_at"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll MBTA V3 predictions and plot official vs local model delays",
    )
    parser.add_argument("--route-id", required=True, help="MBTA route id")
    parser.add_argument("--stop-id", required=True, help="MBTA stop id")
    parser.add_argument("--direction-id", default=None, help="Optional direction id")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Optional realtime bundle path for local model comparison",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Path to save the comparison figure",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to save the markdown report",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to save the sampled realtime rows",
    )
    parser.add_argument(
        "--poll-count",
        type=int,
        default=6,
        help="How many API snapshots to collect",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
        help="Seconds to wait between API polls",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=5,
        help="Maximum number of upcoming prediction rows to fetch per poll",
    )
    args = parser.parse_args()

    if args.poll_count <= 0:
        raise ValueError("--poll-count must be positive")
    if args.poll_interval_seconds < 0:
        raise ValueError("--poll-interval-seconds must be non-negative")
    if args.page_limit <= 0:
        raise ValueError("--page-limit must be positive")

    metrics = generate_mbta_realtime_comparison(
        route_id=args.route_id,
        stop_id=args.stop_id,
        direction_id=args.direction_id,
        bundle_path=args.bundle,
        figure_path=args.output_figure.resolve(),
        report_path=args.output_report.resolve(),
        csv_path=args.output_csv.resolve(),
        poll_count=args.poll_count,
        poll_interval_seconds=args.poll_interval_seconds,
        page_limit=args.page_limit,
    )
    print(metrics)


if __name__ == "__main__":
    main()
