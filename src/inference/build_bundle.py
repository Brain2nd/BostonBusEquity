"""Build a single-file realtime inference bundle for the V2 MLP model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import DATA_PROCESSED, PROJECT_ROOT
from src.models.v2_delay_predictor import (
    V2_CHECKPOINT_NAME,
    V2_EXPERIMENT_VERSION,
    V2_FEATURE_COLUMNS,
    V2_FEATURE_VERSION,
    V2_REALTIME_BUNDLE_NAME,
    build_v2_model_config,
)

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / V2_CHECKPOINT_NAME
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / V2_REALTIME_BUNDLE_NAME
REQUIRED_COLUMNS = [
    "service_date",
    "route_id",
    "stop_id",
    "direction_id",
    "scheduled",
    "scheduled_headway",
]


def _load_processed_dataframe(processed_dir: Path = DATA_PROCESSED) -> pd.DataFrame:
    parquet_path = processed_dir / "arrival_departure.parquet"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"Warning: failed to read {parquet_path}: {exc}")

    csv_candidates = sorted(processed_dir.rglob("arrival_departure*.csv"))
    if not csv_candidates:
        csv_candidates = sorted(processed_dir.rglob("*.csv"))

    if not csv_candidates:
        raise FileNotFoundError(
            "No processed arrival/departure parquet or CSV files were found."
        )

    frames = [pd.read_csv(path) for path in csv_candidates]
    return pd.concat(frames, ignore_index=True)


def _normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    df["scheduled"] = pd.to_datetime(
        df["scheduled"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    if "actual" in df.columns:
        df["actual"] = pd.to_datetime(
            df["actual"],
            format="mixed",
            errors="coerce",
            utc=True,
        )
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")

    if "delay_minutes" not in df.columns:
        if "actual" not in df.columns:
            raise ValueError(
                "Input data must include either 'delay_minutes' or 'actual' to compute delay."
            )
        df["delay_minutes"] = (df["actual"] - df["scheduled"]).dt.total_seconds() / 60
    else:
        df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")

    df["scheduled_headway"] = pd.to_numeric(df["scheduled_headway"], errors="coerce")
    df["route_id"] = df["route_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)
    df["direction_id"] = df["direction_id"].where(df["direction_id"].notna(), None)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = np.nan
    df["year"] = df["year"].fillna(df["service_date"].dt.year)

    df = df.dropna(subset=["scheduled", "service_date", "delay_minutes", "year"])
    df = df[(df["delay_minutes"] >= -30) & (df["delay_minutes"] <= 60)]

    return df


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["hour"] = data["scheduled"].dt.hour
    data["day_of_week"] = data["service_date"].dt.dayofweek
    data["month"] = data["service_date"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_rush_hour"] = (
        ((data["hour"] >= 7) & (data["hour"] <= 9))
        | ((data["hour"] >= 16) & (data["hour"] <= 19))
    ).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["route_hour_key"] = data["route_id"].astype(str) + "_" + data["hour"].astype(str)
    return data


def _build_label_mapping(values: pd.Series, extra_values: list[str] | None = None) -> dict[str, int]:
    classes = sorted({str(value) for value in values if pd.notna(value)})
    for extra_value in extra_values or []:
        if extra_value not in classes:
            classes.append(extra_value)
    return {value: index for index, value in enumerate(classes)}


def _compute_scaler_params(values: np.ndarray) -> dict[str, list[float]]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale == 0, 1.0, scale)
    return {
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
    }


def _compute_feature_scaler_params(
    train_df: pd.DataFrame,
    route_mapping: dict[str, int],
    stop_mapping: dict[str, int],
    direction_mapping: dict[str, int],
    route_stats_map: dict[str, dict[str, float]],
    stop_stats_map: dict[str, dict[str, float]],
    hour_stats_map: dict[str, float],
    route_hour_stats_map: dict[str, float],
    global_mean: float,
    global_std: float,
    headway_median: float,
) -> dict[str, list[float]]:
    """Compute StandardScaler parameters without materializing all 18 features."""
    mean_values: list[float] = []
    scale_values: list[float] = []

    route_mean_map = {
        route_id: stats["mean"] for route_id, stats in route_stats_map.items()
    }
    route_std_map = {
        route_id: stats["std"] for route_id, stats in route_stats_map.items()
    }
    stop_mean_map = {
        stop_id: stats["mean"] for stop_id, stats in stop_stats_map.items()
    }
    stop_std_map = {
        stop_id: stats["std"] for stop_id, stats in stop_stats_map.items()
    }

    def _feature_series(column_name: str) -> pd.Series:
        if column_name == "route_encoded":
            return train_df["route_id"].astype(str).map(route_mapping)
        if column_name == "stop_encoded":
            return train_df["stop_id"].astype(str).map(stop_mapping)
        if column_name == "direction_encoded":
            return (
                train_df["direction_id"]
                .fillna("Unknown")
                .astype(str)
                .map(direction_mapping)
            )
        if column_name == "scheduled_headway":
            return train_df["scheduled_headway"].fillna(headway_median)
        if column_name == "route_delay_mean":
            return train_df["route_id"].astype(str).map(route_mean_map).fillna(global_mean)
        if column_name == "route_delay_std":
            return train_df["route_id"].astype(str).map(route_std_map).fillna(global_std)
        if column_name == "stop_delay_mean":
            return train_df["stop_id"].astype(str).map(stop_mean_map).fillna(global_mean)
        if column_name == "stop_delay_std":
            return train_df["stop_id"].astype(str).map(stop_std_map).fillna(global_std)
        if column_name == "hour_delay_mean":
            return train_df["hour"].map(hour_stats_map).fillna(global_mean)
        if column_name == "route_hour_delay_mean":
            return train_df["route_hour_key"].map(route_hour_stats_map).fillna(global_mean)
        return train_df[column_name]

    for column_name in V2_FEATURE_COLUMNS:
        values = pd.to_numeric(_feature_series(column_name), errors="coerce")
        values = values.fillna(0.0)
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        mean_values.append(mean)
        scale_values.append(scale if scale != 0 else 1.0)

    return {
        "mean": mean_values,
        "scale": scale_values,
    }


def _merge_history_stats(
    df: pd.DataFrame,
    route_stats: pd.DataFrame,
    stop_stats: pd.DataFrame,
    hour_stats: pd.DataFrame,
    route_hour_stats: pd.DataFrame,
    global_mean: float,
    global_std: float,
) -> pd.DataFrame:
    merged = df.merge(route_stats, on="route_id", how="left")
    merged = merged.merge(stop_stats, on="stop_id", how="left")
    merged = merged.merge(hour_stats, on="hour", how="left")
    merged = merged.merge(route_hour_stats, on="route_hour_key", how="left")

    for column in [
        "route_delay_mean",
        "route_delay_std",
        "stop_delay_mean",
        "stop_delay_std",
        "hour_delay_mean",
        "route_hour_delay_mean",
    ]:
        default_value = global_mean if "mean" in column else global_std
        merged[column] = merged[column].fillna(default_value)

    return merged


def build_realtime_bundle_from_dataframe(
    dataframe: pd.DataFrame,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    df = _normalize_dataframe(dataframe)
    if df.empty:
        raise ValueError("No usable rows remained after preprocessing.")

    df = _add_basic_features(df)
    train_mask = df["year"] < 2025
    future_vocab_rows = int((~train_mask).sum())
    train_df = df.loc[train_mask]

    if train_df.empty:
        raise ValueError("Realtime bundle requires at least one training row before 2025.")

    warnings_list: list[str] = []
    if future_vocab_rows == 0:
        warnings_list.append(
            "No 2025+ rows were found; route/stop vocabularies were built from training years only."
        )

    route_stats = (
        train_df.groupby("route_id")["delay_minutes"].agg(["mean", "std"]).reset_index()
    )
    route_stats.columns = ["route_id", "route_delay_mean", "route_delay_std"]
    route_stats["route_delay_std"] = route_stats["route_delay_std"].fillna(0)

    stop_stats = (
        train_df.groupby("stop_id")["delay_minutes"].agg(["mean", "std"]).reset_index()
    )
    stop_stats.columns = ["stop_id", "stop_delay_mean", "stop_delay_std"]
    stop_stats["stop_delay_std"] = stop_stats["stop_delay_std"].fillna(0)

    hour_stats = train_df.groupby("hour")["delay_minutes"].mean().reset_index()
    hour_stats.columns = ["hour", "hour_delay_mean"]

    route_hour_stats = (
        train_df.groupby("route_hour_key")["delay_minutes"].mean().reset_index()
    )
    route_hour_stats.columns = ["route_hour_key", "route_hour_delay_mean"]

    global_mean = float(train_df["delay_minutes"].mean())
    global_std = float(train_df["delay_minutes"].std(ddof=0))

    route_stats_map = {
        str(row.route_id): {
            "mean": float(row.route_delay_mean),
            "std": float(row.route_delay_std),
        }
        for row in route_stats.itertuples(index=False)
    }
    stop_stats_map = {
        str(row.stop_id): {
            "mean": float(row.stop_delay_mean),
            "std": float(row.stop_delay_std),
        }
        for row in stop_stats.itertuples(index=False)
    }
    hour_stats_map = {
        int(row.hour): float(row.hour_delay_mean)
        for row in hour_stats.itertuples(index=False)
    }
    route_hour_stats_map = {
        str(row.route_hour_key): float(row.route_hour_delay_mean)
        for row in route_hour_stats.itertuples(index=False)
    }

    full_direction_series = df["direction_id"].fillna("Unknown").astype(str)
    route_mapping = _build_label_mapping(df["route_id"].astype(str))
    stop_mapping = _build_label_mapping(df["stop_id"].astype(str))
    direction_mapping = _build_label_mapping(
        full_direction_series,
        extra_values=["Unknown"],
    )
    if "Unknown" not in full_direction_series.values:
        warnings_list.append(
            "Direction mapping added an explicit 'Unknown' class for missing realtime inputs."
        )

    headway_median = float(train_df["scheduled_headway"].median())
    scaler_x = _compute_feature_scaler_params(
        train_df=train_df,
        route_mapping=route_mapping,
        stop_mapping=stop_mapping,
        direction_mapping=direction_mapping,
        route_stats_map=route_stats_map,
        stop_stats_map=stop_stats_map,
        hour_stats_map=hour_stats_map,
        route_hour_stats_map=route_hour_stats_map,
        global_mean=global_mean,
        global_std=global_std,
        headway_median=headway_median,
    )
    scaler_y = {
        "mean": [global_mean],
        "scale": [global_std if global_std != 0 else 1.0],
    }

    checkpoint = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing 'model_state_dict'.")

    bundle = {
        "bundle_version": 1,
        "model_name": "V2MLP",
        "experiment": V2_EXPERIMENT_VERSION,
        "feature_version": V2_FEATURE_VERSION,
        "feature_columns": list(V2_FEATURE_COLUMNS),
        "model_config": build_v2_model_config(input_size=len(V2_FEATURE_COLUMNS)),
        "model_state_dict": checkpoint["model_state_dict"],
        "scalers": {
            "x": scaler_x,
            "y": scaler_y,
        },
        "encoders": {
            "route_id": route_mapping,
            "stop_id": stop_mapping,
            "direction_id": direction_mapping,
        },
        "stats": {
            "route": route_stats_map,
            "stop": stop_stats_map,
            "hour": {
                str(int(row.hour)): float(row.hour_delay_mean)
                for row in hour_stats.itertuples(index=False)
            },
            "route_hour": route_hour_stats_map,
            "global_mean": global_mean,
            "global_std": global_std,
            "scheduled_headway_median": headway_median,
        },
        "warnings": warnings_list,
        "data_summary": {
            "train_rows": int(len(train_df)),
            "future_vocab_rows": future_vocab_rows,
            "years_present": sorted(int(year) for year in df["year"].dropna().unique()),
        },
    }

    if output_path is not None:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, output)
        print(f"Saved realtime bundle to: {output}")
        for warning in warnings_list:
            print(f"Warning: {warning}")

    return bundle


def build_realtime_bundle(
    processed_dir: Path | str = DATA_PROCESSED,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    output_path: Path | str = DEFAULT_BUNDLE_PATH,
) -> dict[str, Any]:
    dataframe = _load_processed_dataframe(Path(processed_dir))
    return build_realtime_bundle_from_dataframe(
        dataframe=dataframe,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a realtime V2 inference bundle")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED,
        help="Directory containing arrival_departure parquet or CSV files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to delay_predictor_mlp_v2_lag_features_temporal.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BUNDLE_PATH,
        help="Path to write the realtime bundle",
    )

    args = parser.parse_args()
    build_realtime_bundle(
        processed_dir=args.processed_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
