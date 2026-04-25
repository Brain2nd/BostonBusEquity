"""Generate a realtime inference prediction trace figure over a service day."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.inference.build_bundle import build_realtime_bundle_from_dataframe
from src.inference.runtime import DelayPredictorRuntime
from src.models.v2_delay_predictor import (
    V2_CHECKPOINT_NAME,
    V2_REALTIME_BUNDLE_NAME,
)

LOCAL_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / V2_REALTIME_BUNDLE_NAME
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / V2_CHECKPOINT_NAME
DEFAULT_FIGURE_PATH = FIGURES_DIR / "realtime_inference_prediction_over_time.png"
DEFAULT_REPORT_PATH = REPORTS_DIR / "REALTIME_INFERENCE_PREDICTION_OVER_TIME.md"
DEFAULT_CSV_PATH = REPORTS_DIR / "realtime_inference_prediction_over_time.csv"


@dataclass(frozen=True)
class PredictionSeriesSpec:
    label: str
    route_id: str
    stop_id: str
    direction_id: str | None = None
    scheduled_headway: float | None = None


def _make_demo_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    hours = [5, 7, 8, 9, 12, 16, 17, 18, 21]
    for year in [2024, 2025]:
        for route_id, route_bias in [("1", 0.3), ("2", 1.5), ("28", 2.6)]:
            for stop_id, stop_bias in [("A", 0.2), ("B", 0.9), ("C", 1.4)]:
                for day in [3, 4, 5, 6]:
                    for hour in hours:
                        scheduled = pd.Timestamp(
                            year=year,
                            month=4,
                            day=day,
                            hour=hour,
                            minute=0,
                            tz="UTC",
                        )
                        rush_bias = 0.8 if hour in {7, 8, 9, 16, 17, 18} else 0.0
                        weekend_bias = 0.5 if scheduled.dayofweek >= 5 else 0.0
                        delay_minutes = route_bias + stop_bias + rush_bias + weekend_bias
                        actual = scheduled + pd.Timedelta(minutes=delay_minutes)
                        headway = 8 if hour in {7, 8, 9, 16, 17, 18} else 14
                        rows.append(
                            {
                                "service_date": scheduled.date().isoformat(),
                                "route_id": route_id,
                                "stop_id": stop_id,
                                "direction_id": "0",
                                "scheduled": scheduled.isoformat(),
                                "actual": actual.isoformat(),
                                "scheduled_headway": headway,
                                "year": year,
                            }
                        )
    return pd.DataFrame(rows)


def _parse_series_definition(raw_value: str) -> PredictionSeriesSpec:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) not in {3, 4, 5}:
        raise ValueError(
            "Series must use 'label,route_id,stop_id[,direction_id][,scheduled_headway]'"
        )

    label, route_id, stop_id = parts[:3]
    direction_id = None
    scheduled_headway = None

    if len(parts) >= 4 and parts[3]:
        direction_id = parts[3]
    if len(parts) == 5 and parts[4]:
        scheduled_headway = float(parts[4])

    return PredictionSeriesSpec(
        label=label,
        route_id=route_id,
        stop_id=stop_id,
        direction_id=direction_id,
        scheduled_headway=scheduled_headway,
    )


def _default_demo_series() -> list[PredictionSeriesSpec]:
    return [
        PredictionSeriesSpec(
            label="Route 1 / Stop A",
            route_id="1",
            stop_id="A",
            direction_id="0",
            scheduled_headway=12,
        ),
        PredictionSeriesSpec(
            label="Route 2 / Stop B",
            route_id="2",
            stop_id="B",
            direction_id="0",
            scheduled_headway=10,
        ),
        PredictionSeriesSpec(
            label="Route 28 / Stop C",
            route_id="28",
            stop_id="C",
            direction_id="0",
            scheduled_headway=8,
        ),
    ]


def _default_series_for_runtime(runtime: DelayPredictorRuntime) -> list[PredictionSeriesSpec]:
    route_ids = sorted(runtime.encoders["route_id"].keys())
    stop_ids = sorted(runtime.encoders["stop_id"].keys())
    direction_candidates = [
        value for value in sorted(runtime.encoders["direction_id"].keys()) if value != "Unknown"
    ]
    direction_id = direction_candidates[0] if direction_candidates else None
    scheduled_headway = float(runtime.stats["scheduled_headway_median"])

    if not route_ids or not stop_ids:
        raise ValueError("Bundle does not contain enough route or stop ids to build default series.")

    candidate_pairs = [(route_ids[0], stop_ids[0])]
    if len(stop_ids) > 1:
        candidate_pairs.append((route_ids[0], stop_ids[1]))
    if len(route_ids) > 1:
        candidate_pairs.append((route_ids[1], stop_ids[min(len(stop_ids) - 1, 1)]))

    series_specs: list[PredictionSeriesSpec] = []
    seen: set[tuple[str, str]] = set()
    for route_id, stop_id in candidate_pairs:
        if (route_id, stop_id) in seen:
            continue
        seen.add((route_id, stop_id))
        series_specs.append(
            PredictionSeriesSpec(
                label=f"Route {route_id} / Stop {stop_id}",
                route_id=route_id,
                stop_id=stop_id,
                direction_id=direction_id,
                scheduled_headway=scheduled_headway,
            )
        )

    return series_specs


def _resolve_runtime(
    bundle_path: Path | None,
    checkpoint_path: Path,
) -> tuple[DelayPredictorRuntime, str, list[str]]:
    warnings_list: list[str] = []
    if bundle_path is not None and bundle_path.exists():
        return DelayPredictorRuntime.from_bundle_path(bundle_path), "bundle", warnings_list

    temp_dir_path = PROJECT_ROOT / "tmp_prediction_trace"
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path, ignore_errors=True)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    temp_bundle_path = temp_dir_path / "demo_realtime_bundle.pt"

    try:
        build_realtime_bundle_from_dataframe(
            dataframe=_make_demo_dataframe(),
            checkpoint_path=checkpoint_path,
            output_path=temp_bundle_path,
        )
        runtime = DelayPredictorRuntime.from_bundle_path(temp_bundle_path)
    finally:
        shutil.rmtree(temp_dir_path, ignore_errors=True)

    warnings_list.append(
        "No realtime bundle was found. Generated a demo bundle from synthetic data for plotting."
    )
    return runtime, "demo_bundle", warnings_list


def build_prediction_trace_dataframe(
    runtime: DelayPredictorRuntime,
    series_specs: list[PredictionSeriesSpec],
    service_date: date,
    start_hour: int,
    end_hour: int,
    frequency_minutes: int,
) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(
        year=service_date.year,
        month=service_date.month,
        day=service_date.day,
        hour=start_hour,
        minute=0,
        tz=LOCAL_TIMEZONE,
    )
    end_timestamp = pd.Timestamp(
        year=service_date.year,
        month=service_date.month,
        day=service_date.day,
        hour=end_hour,
        minute=0,
        tz=LOCAL_TIMEZONE,
    )
    timeline = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        freq=f"{frequency_minutes}min",
        tz=LOCAL_TIMEZONE,
    )

    rows: list[dict[str, Any]] = []
    for series_spec in series_specs:
        for scheduled_time in timeline:
            prediction = runtime.predict(
                route_id=series_spec.route_id,
                stop_id=series_spec.stop_id,
                direction_id=series_spec.direction_id,
                scheduled_headway=series_spec.scheduled_headway,
                scheduled_time=scheduled_time.to_pydatetime(),
            )
            rows.append(
                {
                    "label": series_spec.label,
                    "route_id": series_spec.route_id,
                    "stop_id": series_spec.stop_id,
                    "direction_id": series_spec.direction_id or "Unknown",
                    "scheduled_headway": series_spec.scheduled_headway,
                    "scheduled_time_local": scheduled_time.isoformat(),
                    "predicted_delay_minutes": prediction["predicted_delay_minutes"],
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe["scheduled_time_local"] = pd.to_datetime(dataframe["scheduled_time_local"])
    return dataframe


def _plot_prediction_trace(
    trace_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
    service_date: date,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("#f4efe7")
    ax.set_facecolor("#fffaf4")

    peak_windows = [(7, 9), (16, 19)]
    for start_hour, end_hour in peak_windows:
        start = datetime(
            service_date.year,
            service_date.month,
            service_date.day,
            start_hour,
            tzinfo=LOCAL_TIMEZONE,
        )
        end = datetime(
            service_date.year,
            service_date.month,
            service_date.day,
            end_hour,
            tzinfo=LOCAL_TIMEZONE,
        )
        ax.axvspan(start, end, color="#f6bd60", alpha=0.12, linewidth=0)

    palette = ["#1f5f8b", "#bc4749", "#4d908e", "#6d597a", "#f9844a"]
    for color, (label, group_df) in zip(palette, trace_df.groupby("label", sort=False)):
        ordered = group_df.sort_values("scheduled_time_local")
        ax.plot(
            ordered["scheduled_time_local"],
            ordered["predicted_delay_minutes"],
            label=label,
            color=color,
            linewidth=2.4,
        )
        ax.scatter(
            ordered["scheduled_time_local"],
            ordered["predicted_delay_minutes"],
            color=color,
            s=18,
            alpha=0.75,
        )

    ax.set_title(
        f"Realtime Inference Prediction Trace {title_suffix}",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Scheduled Time (America/New_York)")
    ax.set_ylabel("Predicted Delay (minutes)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=LOCAL_TIMEZONE))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TIMEZONE))
    ax.legend(frameon=False, ncols=1, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.013,
        0.012,
        "Shaded windows highlight rush hours: 07:00-09:00 and 16:00-19:00.",
        fontsize=10,
        color="#4a4e69",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_prediction_trace_report(
    output_path: Path,
    figure_path: Path,
    csv_path: Path,
    trace_df: pd.DataFrame,
    series_specs: list[PredictionSeriesSpec],
    checkpoint_path: Path,
    bundle_mode: str,
    warnings_list: list[str],
    service_date: date,
    frequency_minutes: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = (
        trace_df.groupby("label")["predicted_delay_minutes"]
        .agg(["min", "mean", "max"])
        .reset_index()
    )

    lines = [
        "# Realtime Inference Prediction Trace",
        "",
        f"Generated: {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Configuration",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Bundle mode: `{bundle_mode}`",
        f"- Service date: `{service_date.isoformat()}`",
        f"- Sampling cadence: `{frequency_minutes}` minutes",
        f"- Figure: `{figure_path}`",
        f"- CSV: `{csv_path}`",
        "",
        "## Series",
        "",
    ]

    for series_spec in series_specs:
        lines.append(
            (
                f"- `{series_spec.label}`: route `{series_spec.route_id}`, "
                f"stop `{series_spec.stop_id}`, "
                f"direction `{series_spec.direction_id or 'Unknown'}`, "
                f"headway `{series_spec.scheduled_headway}`"
            )
        )

    if warnings_list:
        lines.extend(
            [
                "",
                "## Warnings",
                "",
            ]
        )
        for warning in warnings_list:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Series | Min (min) | Avg (min) | Max (min) |",
            "|--------|----------:|----------:|----------:|",
        ]
    )

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.min:.3f} | {row.mean:.3f} | {row.max:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- Generated `{len(trace_df)}` realtime predictions across `{len(series_specs)}` series.",
            "- Saved the full prediction trace to CSV for inspection and reuse.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_prediction_trace(
    bundle_path: Path | None,
    checkpoint_path: Path,
    figure_path: Path,
    report_path: Path,
    csv_path: Path,
    service_date: date,
    start_hour: int,
    end_hour: int,
    frequency_minutes: int,
    series_specs: list[PredictionSeriesSpec] | None = None,
) -> dict[str, Any]:
    runtime, bundle_mode, warnings_list = _resolve_runtime(
        bundle_path=bundle_path,
        checkpoint_path=checkpoint_path,
    )

    if series_specs is None:
        if bundle_mode == "demo_bundle":
            series_specs = _default_demo_series()
        else:
            series_specs = _default_series_for_runtime(runtime)

    trace_df = build_prediction_trace_dataframe(
        runtime=runtime,
        series_specs=series_specs,
        service_date=service_date,
        start_hour=start_hour,
        end_hour=end_hour,
        frequency_minutes=frequency_minutes,
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(csv_path, index=False)

    title_suffix = "(Demo Bundle)" if bundle_mode == "demo_bundle" else "(Realtime Bundle)"
    _plot_prediction_trace(
        trace_df=trace_df,
        output_path=figure_path,
        title_suffix=title_suffix,
        service_date=service_date,
    )
    _write_prediction_trace_report(
        output_path=report_path,
        figure_path=figure_path,
        csv_path=csv_path,
        trace_df=trace_df,
        series_specs=series_specs,
        checkpoint_path=checkpoint_path,
        bundle_mode=bundle_mode,
        warnings_list=warnings_list,
        service_date=service_date,
        frequency_minutes=frequency_minutes,
    )

    grouped = trace_df.groupby("label")["predicted_delay_minutes"]
    summary = {
        label: {
            "min": float(np.min(values)),
            "avg": float(np.mean(values)),
            "max": float(np.max(values)),
        }
        for label, values in grouped
    }

    return {
        "bundle_mode": bundle_mode,
        "figure_path": str(figure_path),
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "warnings": warnings_list,
        "series_summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a realtime inference prediction trace figure",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE_PATH,
        help="Path to an existing realtime bundle. Falls back to a demo bundle if missing.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the V2 MLP checkpoint",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Path to save the prediction trace figure",
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
        help="Path to save the underlying prediction trace as CSV",
    )
    parser.add_argument(
        "--service-date",
        type=date.fromisoformat,
        default=datetime.now(LOCAL_TIMEZONE).date(),
        help="Local service date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=5,
        help="First local hour to include in the trace",
    )
    parser.add_argument(
        "--end-hour",
        type=int,
        default=23,
        help="Last local hour to include in the trace",
    )
    parser.add_argument(
        "--frequency-minutes",
        type=int,
        default=30,
        help="Minutes between prediction samples",
    )
    parser.add_argument(
        "--series",
        action="append",
        default=None,
        help=(
            "Optional series definition using "
            "'label,route_id,stop_id[,direction_id][,scheduled_headway]'"
        ),
    )
    args = parser.parse_args()

    if args.start_hour < 0 or args.start_hour > 23:
        raise ValueError("--start-hour must be between 0 and 23")
    if args.end_hour < 0 or args.end_hour > 23:
        raise ValueError("--end-hour must be between 0 and 23")
    if args.end_hour <= args.start_hour:
        raise ValueError("--end-hour must be greater than --start-hour")
    if args.frequency_minutes <= 0:
        raise ValueError("--frequency-minutes must be positive")

    series_specs = None
    if args.series:
        series_specs = [_parse_series_definition(raw_value) for raw_value in args.series]

    result = generate_prediction_trace(
        bundle_path=args.bundle.resolve() if args.bundle is not None else None,
        checkpoint_path=args.checkpoint.resolve(),
        figure_path=args.output_figure.resolve(),
        report_path=args.output_report.resolve(),
        csv_path=args.output_csv.resolve(),
        service_date=args.service_date,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        frequency_minutes=args.frequency_minutes,
        series_specs=series_specs,
    )
    print(result)


if __name__ == "__main__":
    main()
