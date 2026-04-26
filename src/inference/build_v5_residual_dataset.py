"""Build V5 official-residual training labels from live snapshots and actuals."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR

LOCAL_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_SNAPSHOT_DIR = REPORTS_DIR / "live_prediction_snapshots"
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "arrival_departure.parquet"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "live_prediction_residual_dataset.parquet"
DEFAULT_REPORT_PATH = REPORTS_DIR / "V5_RESIDUAL_DATASET_REPORT.md"
DEFAULT_FIGURE_PATH = FIGURES_DIR / "official_vs_v4_vs_actual.png"


def _read_snapshot_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_live_snapshots(snapshot_dir: Path) -> pd.DataFrame:
    paths = sorted(snapshot_dir.glob("merged_*.parquet")) + sorted(
        snapshot_dir.glob("merged_*.csv")
    )
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        frame = _read_snapshot_file(path)
        frame["snapshot_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _read_actuals(processed_path: Path, snapshots: pd.DataFrame) -> pd.DataFrame:
    columns = ["route_id", "stop_id", "scheduled", "actual"]
    years = sorted(
        {
            int(year)
            for year in snapshots["scheduled_time"].dt.year.dropna().unique()
        }
    )
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(processed_path).schema.names)
        read_columns = columns + (["year"] if "year" in available else [])
        filters = [("year", "in", years)] if years and "year" in available else None
        dataframe = pd.read_parquet(
            processed_path,
            columns=read_columns,
            filters=filters,
        )
    except Exception:
        dataframe = pd.read_parquet(processed_path, columns=columns)
    dataframe["scheduled"] = pd.to_datetime(
        dataframe["scheduled"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    dataframe["actual"] = pd.to_datetime(
        dataframe["actual"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    dataframe["route_id"] = dataframe["route_id"].astype(str)
    dataframe["stop_id"] = dataframe["stop_id"].astype(str)
    dataframe["actual_delay_minutes"] = (
        dataframe["actual"] - dataframe["scheduled"]
    ).dt.total_seconds() / 60
    dataframe = dataframe.dropna(subset=["scheduled", "actual_delay_minutes"])
    dataframe = dataframe[
        (dataframe["actual_delay_minutes"] >= -30)
        & (dataframe["actual_delay_minutes"] <= 60)
    ].copy()
    dataframe["scheduled_match_key"] = dataframe["scheduled"].dt.round("min")
    return dataframe.drop_duplicates(
        ["route_id", "stop_id", "scheduled_match_key"],
        keep="first",
    )


def build_residual_dataset(
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
) -> pd.DataFrame:
    snapshots = load_live_snapshots(snapshot_dir)
    if snapshots.empty:
        return pd.DataFrame()

    required = ["route_id", "stop_id", "scheduled_time", "official_delay_minutes"]
    missing = [column for column in required if column not in snapshots.columns]
    if missing:
        raise ValueError(f"Snapshot data is missing required columns: {missing}")

    snapshots = snapshots.copy()
    snapshots["route_id"] = snapshots["route_id"].astype(str)
    snapshots["stop_id"] = snapshots["stop_id"].astype(str)
    snapshots["scheduled_time"] = pd.to_datetime(
        snapshots["scheduled_time"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    snapshots["observed_at"] = pd.to_datetime(
        snapshots["observed_at"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    snapshots["official_delay_minutes"] = pd.to_numeric(
        snapshots["official_delay_minutes"],
        errors="coerce",
    )
    snapshots = snapshots.dropna(
        subset=["scheduled_time", "official_delay_minutes", "route_id", "stop_id"]
    )
    snapshots["scheduled_match_key"] = snapshots["scheduled_time"].dt.round("min")

    actuals = _read_actuals(processed_path, snapshots)
    matched = snapshots.merge(
        actuals[
            [
                "route_id",
                "stop_id",
                "scheduled_match_key",
                "actual",
                "actual_delay_minutes",
            ]
        ],
        on=["route_id", "stop_id", "scheduled_match_key"],
        how="inner",
    )
    if matched.empty:
        return matched

    matched["official_residual_label"] = (
        matched["actual_delay_minutes"] - matched["official_delay_minutes"]
    )
    matched["official_absolute_error"] = (
        matched["actual_delay_minutes"] - matched["official_delay_minutes"]
    ).abs()
    if "model_predicted_delay_minutes" in matched.columns:
        matched["v4_absolute_error"] = (
            matched["actual_delay_minutes"] - matched["model_predicted_delay_minutes"]
        ).abs()
    return matched


def _write_dataframe(dataframe: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".csv")
        dataframe.to_csv(fallback, index=False)
        return fallback


def _plot_matched_labels(dataframe: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    fig.patch.set_facecolor("#f7f4ec")
    if dataframe.empty:
        ax.text(
            0.5,
            0.5,
            "Waiting for matched live actual labels",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_axis_off()
    else:
        sample = dataframe.sort_values("scheduled_time").head(300)
        ax.plot(
            sample["scheduled_time"],
            sample["actual_delay_minutes"],
            label="Actual true delay",
            color="#1f5f8b",
            linewidth=2.0,
        )
        ax.plot(
            sample["scheduled_time"],
            sample["official_delay_minutes"],
            label="MBTA official",
            color="#2f6f4e",
            linewidth=2.0,
        )
        if "model_predicted_delay_minutes" in sample.columns:
            ax.plot(
                sample["scheduled_time"],
                sample["model_predicted_delay_minutes"],
                label="V4 model",
                color="#bc4749",
                linewidth=2.0,
            )
        ax.set_title("Official vs V4 vs Actual Delay")
        ax.set_xlabel("Scheduled time")
        ax.set_ylabel("Delay minutes")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    dataframe: pd.DataFrame,
    output_path: Path,
    report_path: Path,
    figure_path: Path,
    min_samples: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status = "ready" if len(dataframe) >= min_samples else "inconclusive"
    lines = [
        "# V5 Residual Dataset Report",
        "",
        f"Generated: {datetime.now(LOCAL_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        f"- Matched rows: `{len(dataframe)}`",
        f"- Minimum rows for live model acceptance: `{min_samples}`",
        f"- Status: `{status}`",
        f"- Output: `{output_path}`",
        f"- Figure: `{figure_path}`",
    ]
    if not dataframe.empty:
        lines.extend(
            [
                "",
                "## Current Live Label Metrics",
                "",
                f"- MBTA official MAE: `{dataframe['official_absolute_error'].mean():.3f}` minutes",
            ]
        )
        if "v4_absolute_error" in dataframe.columns:
            lines.append(
                f"- V4 model MAE: `{dataframe['v4_absolute_error'].mean():.3f}` minutes"
            )
        lines.append(
            f"- Mean residual label: `{dataframe['official_residual_label'].mean():.3f}` minutes"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_v5_residual_dataset(
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
    min_samples: int = 500,
) -> dict[str, Any]:
    dataset = build_residual_dataset(
        snapshot_dir=snapshot_dir,
        processed_path=processed_path,
    )
    written_output = _write_dataframe(dataset, output_path)
    _plot_matched_labels(dataset, figure_path)
    _write_report(
        dataframe=dataset,
        output_path=written_output,
        report_path=report_path,
        figure_path=figure_path,
        min_samples=min_samples,
    )
    return {
        "rows": int(len(dataset)),
        "status": "ready" if len(dataset) >= min_samples else "inconclusive",
        "output_path": str(written_output),
        "report_path": str(report_path),
        "figure_path": str(figure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match live MBTA snapshots to actual arrivals for V5 residual labels",
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--min-samples", type=int, default=500)
    args = parser.parse_args()

    result = generate_v5_residual_dataset(
        snapshot_dir=args.snapshot_dir,
        processed_path=args.processed_path,
        output_path=args.output,
        report_path=args.report,
        figure_path=args.figure,
        min_samples=args.min_samples,
    )
    print(result)


if __name__ == "__main__":
    main()
