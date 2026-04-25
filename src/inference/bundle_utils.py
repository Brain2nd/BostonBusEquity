"""
Helpers shared by realtime bundle building and baseline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

V2_FEATURE_COLUMNS = [
    "is_weekend",
    "is_rush_hour",
    "route_encoded",
    "stop_encoded",
    "direction_encoded",
    "scheduled_headway",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "route_delay_mean",
    "route_delay_std",
    "stop_delay_mean",
    "stop_delay_std",
    "hour_delay_mean",
    "route_hour_delay_mean",
]

SOURCE_COLUMNS = [
    "service_date",
    "route_id",
    "stop_id",
    "direction_id",
    "scheduled",
    "actual",
    "scheduled_headway",
    "year",
    "month",
]


@dataclass
class ScalerValues:
    mean: list[float]
    scale: list[float]

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        return (values - mean) / scale


@dataclass
class V2TrainingStatistics:
    route_delay_mean: dict[str, float]
    route_delay_std: dict[str, float]
    stop_delay_mean: dict[str, float]
    stop_delay_std: dict[str, float]
    hour_delay_mean: dict[int, float]
    route_hour_delay_mean: dict[str, float]
    global_mean: float
    global_std: float
    scheduled_headway_median: float

    def to_dict(self) -> dict:
        return {
            "route_delay_mean": self.route_delay_mean,
            "route_delay_std": self.route_delay_std,
            "stop_delay_mean": self.stop_delay_mean,
            "stop_delay_std": self.stop_delay_std,
            "hour_delay_mean": self.hour_delay_mean,
            "route_hour_delay_mean": self.route_hour_delay_mean,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "scheduled_headway_median": self.scheduled_headway_median,
        }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    if "direction" in df.columns and "direction_id" not in df.columns:
        df = df.rename(columns={"direction": "direction_id"})
    return df


def parse_arrival_departure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    required = [col for col in SOURCE_COLUMNS if col in df.columns]
    df = df[required].copy()

    df["route_id"] = df["route_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)
    df["direction_id"] = df.get("direction_id", "Unknown").fillna("Unknown").astype(str)

    df["service_date"] = pd.to_datetime(
        df["service_date"].astype(str).str.replace("\ufeff", "", regex=False),
        errors="coerce",
        format="mixed",
    )
    df["scheduled"] = pd.to_datetime(df["scheduled"], errors="coerce", format="mixed", utc=True)
    df["actual"] = pd.to_datetime(df["actual"], errors="coerce", format="mixed", utc=True)
    df["delay_minutes"] = (df["actual"] - df["scheduled"]).dt.total_seconds() / 60
    df = df.dropna(subset=["service_date", "scheduled", "delay_minutes"])
    df = df[(df["delay_minutes"] >= -30) & (df["delay_minutes"] <= 60)].copy()
    df["year"] = df["service_date"].dt.year
    return df


def add_v2_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["scheduled"].dt.tz_convert("America/New_York").dt.hour
    df["day_of_week"] = df["service_date"].dt.dayofweek
    df["month"] = df["service_date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9))
        | ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def load_arrival_departure_dataframe(
    parquet_path: Path | None = None,
    raw_dir: Path | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    parquet_path = parquet_path or (DATA_PROCESSED / "arrival_departure.parquet")
    raw_dir = raw_dir or (DATA_RAW / "arrival_departure")

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path, columns=SOURCE_COLUMNS)
        return parse_arrival_departure_dataframe(df)

    csv_files = sorted(raw_dir.glob("**/*.csv"))
    if max_files is not None:
        csv_files = csv_files[:max_files]
    if not csv_files:
        raise FileNotFoundError(
            f"No arrival/departure parquet or CSV files found under {parquet_path} or {raw_dir}"
        )

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path, usecols=lambda col: col in SOURCE_COLUMNS or col == "direction")
        frames.append(parse_arrival_departure_dataframe(frame))

    return pd.concat(frames, ignore_index=True)


def split_temporal_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["year"] < 2025].copy()
    test_df = df[df["year"] >= 2025].copy()
    return train_df, test_df


def build_category_mappings(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    all_routes = pd.concat([train_df["route_id"], test_df["route_id"]]).astype(str).unique()
    all_stops = pd.concat([train_df["stop_id"], test_df["stop_id"]]).astype(str).unique()
    all_directions = pd.concat(
        [train_df["direction_id"], test_df["direction_id"]]
    ).fillna("Unknown").astype(str).unique()

    return {
        "route_id": {value: idx for idx, value in enumerate(sorted(all_routes))},
        "stop_id": {value: idx for idx, value in enumerate(sorted(all_stops))},
        "direction_id": {value: idx for idx, value in enumerate(sorted(all_directions))},
    }


def compute_v2_training_statistics(train_df: pd.DataFrame) -> V2TrainingStatistics:
    route_stats = train_df.groupby("route_id")["delay_minutes"].agg(["mean", "std"]).fillna(0)
    stop_stats = train_df.groupby("stop_id")["delay_minutes"].agg(["mean", "std"]).fillna(0)
    hour_stats = train_df.groupby("hour")["delay_minutes"].mean()

    route_hour_df = train_df.copy()
    route_hour_df["route_hour_key"] = (
        route_hour_df["route_id"].astype(str) + "_" + route_hour_df["hour"].astype(str)
    )
    route_hour_stats = route_hour_df.groupby("route_hour_key")["delay_minutes"].mean()

    return V2TrainingStatistics(
        route_delay_mean=route_stats["mean"].to_dict(),
        route_delay_std=route_stats["std"].replace(0, train_df["delay_minutes"].std()).to_dict(),
        stop_delay_mean=stop_stats["mean"].to_dict(),
        stop_delay_std=stop_stats["std"].replace(0, train_df["delay_minutes"].std()).to_dict(),
        hour_delay_mean=hour_stats.to_dict(),
        route_hour_delay_mean=route_hour_stats.to_dict(),
        global_mean=float(train_df["delay_minutes"].mean()),
        global_std=float(train_df["delay_minutes"].std() or 1.0),
        scheduled_headway_median=float(train_df["scheduled_headway"].median()),
    )


def build_v2_feature_frame(
    df: pd.DataFrame,
    mappings: dict[str, dict[str, int]],
    stats: V2TrainingStatistics,
) -> pd.DataFrame:
    df = add_v2_time_features(df)
    df = df.copy()

    df["route_encoded"] = df["route_id"].astype(str).map(mappings["route_id"])
    df["stop_encoded"] = df["stop_id"].astype(str).map(mappings["stop_id"])
    df["direction_encoded"] = (
        df["direction_id"].fillna("Unknown").astype(str).map(mappings["direction_id"])
    )

    df["scheduled_headway"] = df["scheduled_headway"].fillna(stats.scheduled_headway_median)
    df["route_hour_key"] = df["route_id"].astype(str) + "_" + df["hour"].astype(str)

    df["route_delay_mean"] = df["route_id"].map(stats.route_delay_mean).fillna(stats.global_mean)
    df["route_delay_std"] = df["route_id"].map(stats.route_delay_std).fillna(stats.global_std)
    df["stop_delay_mean"] = df["stop_id"].map(stats.stop_delay_mean).fillna(stats.global_mean)
    df["stop_delay_std"] = df["stop_id"].map(stats.stop_delay_std).fillna(stats.global_std)
    df["hour_delay_mean"] = df["hour"].map(stats.hour_delay_mean).fillna(stats.global_mean)
    df["route_hour_delay_mean"] = (
        df["route_hour_key"].map(stats.route_hour_delay_mean).fillna(stats.global_mean)
    )

    if df[V2_FEATURE_COLUMNS].isna().any().any():
        df[V2_FEATURE_COLUMNS] = df[V2_FEATURE_COLUMNS].fillna(0.0)

    return df


def fit_scaler_values(values: np.ndarray) -> ScalerValues:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return ScalerValues(mean=mean.astype(float).tolist(), scale=scale.astype(float).tolist())


def make_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[V2_FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def load_v2_training_frames(max_files: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_arrival_departure_dataframe(max_files=max_files)
    df = add_v2_time_features(df)
    return split_temporal_train_test(df)


def safe_sample(df: pd.DataFrame, limit: int | None, seed: int = 42) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=seed)


def coerce_float_dict(data: dict[str, float] | dict[int, float]) -> dict[str, float]:
    return {str(key): float(value) for key, value in data.items()}


def iter_future_schedule_times(records: Iterable[dict], field: str) -> list[pd.Timestamp]:
    times: list[pd.Timestamp] = []
    for record in records:
        value = record.get(field)
        if value:
            parsed = pd.Timestamp(value)
            if parsed.tzinfo is None:
                parsed = parsed.tz_localize("America/New_York")
            times.append(parsed)
    return times
