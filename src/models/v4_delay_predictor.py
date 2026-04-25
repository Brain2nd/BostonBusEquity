"""V4 causal feature engineering and tree-model bundle helpers.

V4 keeps the realtime path honest: the target is true delay
(`actual - scheduled`), while MBTA official predictions are reserved for
live comparison and the later V5 residual-correction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.models.v2_delay_predictor import V2_FEATURE_COLUMNS

V4_EXPERIMENT_VERSION = "v4_tree_causal_live_features"
V4_FEATURE_VERSION = "v4_causal_trip_history"
V4_BUNDLE_NAME = "delay_predictor_v4_tree_realtime_bundle.joblib"
V4_BUNDLE_PATH = PROJECT_ROOT / "models" / V4_BUNDLE_NAME
V4_REQUIRED_COLUMNS = [
    "service_date",
    "route_id",
    "stop_id",
    "direction_id",
    "scheduled",
    "actual",
    "scheduled_headway",
]
V4_ADDITIONAL_FEATURE_COLUMNS = [
    "time_point_order",
    "half_trip_encoded",
    "minute_of_day",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "trip_delay_lag_1",
    "trip_delay_lag_2",
    "trip_delay_lag_3",
    "trip_delay_rolling_mean",
    "trip_delay_rolling_std",
    "trip_delay_rolling_max",
    "route_stop_delay_mean",
    "route_stop_delay_std",
    "route_stop_hour_delay_mean",
    "route_stop_hour_delay_std",
    "route_stop_order_delay_mean",
    "route_stop_order_delay_std",
    "route_direction_hour_delay_mean",
    "route_direction_hour_delay_std",
    "previous_headway_deviation",
    "scheduled_headway_missing",
    "scheduled_headway_log",
    "stop_sequence_fraction",
    "current_stop_sequence",
    "vehicle_speed",
    "official_prediction_age_seconds",
]
V4_FEATURE_COLUMNS = list(V2_FEATURE_COLUMNS) + V4_ADDITIONAL_FEATURE_COLUMNS
V4_CATEGORICAL_FEATURE_COLUMNS = [
    "route_encoded",
    "stop_encoded",
    "direction_encoded",
    "half_trip_encoded",
]
V4_TARGET_COLUMN = "delay_minutes"
V4_RUNTIME_PROFILE_FULL_HISTORY = "full_history"
V4_RUNTIME_PROFILE_ONLINE_SAFE = "online_safe"
V4_RUNTIME_PROFILE_OPTIONS = {
    V4_RUNTIME_PROFILE_FULL_HISTORY,
    V4_RUNTIME_PROFILE_ONLINE_SAFE,
}


class FeatureFallbackRegressor:
    """Simple interpretable baseline using precomputed historical feature means."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or [
            "route_stop_hour_delay_mean",
            "route_stop_delay_mean",
            "route_hour_delay_mean",
            "hour_delay_mean",
            "route_delay_mean",
            "stop_delay_mean",
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "FeatureFallbackRegressor":
        y_array = np.asarray(y, dtype=float)
        finite = y_array[np.isfinite(y_array)]
        self.fallback_ = float(np.median(finite)) if finite.size else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.full(len(X), getattr(self, "fallback_", 0.0), dtype=float)
        missing = np.ones(len(X), dtype=bool)
        for column in self.columns:
            if column not in X.columns:
                continue
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
            usable = missing & np.isfinite(values)
            predictions[usable] = values[usable]
            missing &= ~usable
            if not missing.any():
                break
        return predictions


V4_ONLINE_DEFAULTED_FEATURE_COLUMNS = [
    "half_trip_encoded",
    "trip_delay_lag_1",
    "trip_delay_lag_2",
    "trip_delay_lag_3",
    "trip_delay_rolling_mean",
    "trip_delay_rolling_std",
    "trip_delay_rolling_max",
    "previous_headway_deviation",
]


@dataclass(frozen=True)
class V4Dataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    metadata: dict[str, Any]


def _safe_std(series: pd.Series) -> float:
    value = float(series.std())
    return value if np.isfinite(value) and value > 0 else 1.0


def _as_string_series(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").astype(str)


def normalize_v4_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw/processed arrival-departure rows for V4 training."""
    missing = [column for column in V4_REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input data is missing required V4 columns: {missing}")

    df = dataframe.copy()
    df["scheduled"] = pd.to_datetime(df["scheduled"], format="mixed", errors="coerce", utc=True)
    df["actual"] = pd.to_datetime(df["actual"], format="mixed", errors="coerce", utc=True)
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    df["delay_minutes"] = (df["actual"] - df["scheduled"]).dt.total_seconds() / 60
    df["scheduled_headway"] = pd.to_numeric(df["scheduled_headway"], errors="coerce")
    df["route_id"] = _as_string_series(df["route_id"])
    df["stop_id"] = _as_string_series(df["stop_id"])
    df["direction_id"] = _as_string_series(df["direction_id"])

    if "half_trip_id" not in df.columns:
        df["half_trip_id"] = (
            df["route_id"].astype(str)
            + "_"
            + df["direction_id"].astype(str)
            + "_"
            + df["service_date"].dt.strftime("%Y%m%d")
            + "_"
            + df["scheduled"].dt.hour.astype("Int64").astype(str)
        )
    df["half_trip_id"] = _as_string_series(df["half_trip_id"])

    if "time_point_order" not in df.columns:
        df["time_point_order"] = np.nan
    df["time_point_order"] = pd.to_numeric(df["time_point_order"], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = np.nan
    df["year"] = df["year"].fillna(df["service_date"].dt.year).astype(int)

    df = df.dropna(subset=["scheduled", "actual", "service_date", "delay_minutes"])
    df = df[(df["delay_minutes"] >= -30) & (df["delay_minutes"] <= 60)]
    return df.reset_index(drop=True)


def add_v4_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["hour"] = df["scheduled"].dt.hour
    df["minute"] = df["scheduled"].dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["day_of_week"] = df["service_date"].dt.dayofweek
    df["day_of_year"] = df["service_date"].dt.dayofyear
    df["month"] = df["service_date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9))
        | ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_of_day_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["minute_of_day_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["route_hour_key"] = df["route_id"] + "_" + df["hour"].astype(str)
    df["route_stop_key"] = df["route_id"] + "_" + df["stop_id"]
    df["route_stop_hour_key"] = df["route_stop_key"] + "_" + df["hour"].astype(str)
    order_key = (
        pd.to_numeric(df["time_point_order"], errors="coerce")
        .fillna(-1)
        .round()
        .astype(int)
        .astype(str)
    )
    df["route_stop_order_key"] = df["route_stop_key"] + "_" + order_key
    df["route_direction_hour_key"] = (
        df["route_id"] + "_" + df["direction_id"] + "_" + df["hour"].astype(str)
    )
    return df


def build_v4_encoder(values: Iterable[Any], include_unknown: bool = True) -> dict[str, int]:
    classes = sorted({str(value) for value in values if pd.notna(value)})
    if include_unknown and "Unknown" not in classes:
        classes.insert(0, "Unknown")
    return {value: index for index, value in enumerate(classes)}


def apply_v4_encoders(
    dataframe: pd.DataFrame,
    encoders: dict[str, dict[str, int]],
) -> pd.DataFrame:
    df = dataframe.copy()
    encoder_sources = {
        "route_encoded": ("route_id", "route_id"),
        "stop_encoded": ("stop_id", "stop_id"),
        "direction_encoded": ("direction_id", "direction_id"),
        "half_trip_encoded": ("half_trip_id", "half_trip_id"),
    }
    for feature_name, (source_column, encoder_name) in encoder_sources.items():
        mapping = encoders[encoder_name]
        df[feature_name] = (
            _as_string_series(df[source_column])
            .map(mapping)
            .fillna(mapping.get("Unknown", -1))
            .astype(int)
        )
    return df


def _stats_frame(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    stats = (
        dataframe.groupby(group_columns, dropna=False)["delay_minutes"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats = stats.rename(
        columns={
            "mean": f"{prefix}_delay_mean",
            "std": f"{prefix}_delay_std",
        }
    )
    stats[f"{prefix}_delay_std"] = stats[f"{prefix}_delay_std"].fillna(0.0)
    return stats


def _stats_map(stats: pd.DataFrame, key_column: str, prefix: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in stats.to_dict("records"):
        result[str(row[key_column])] = {
            "mean": float(row[f"{prefix}_delay_mean"]),
            "std": float(row[f"{prefix}_delay_std"]),
        }
    return result


def build_v4_training_stats(train_df: pd.DataFrame) -> dict[str, Any]:
    global_mean = float(train_df["delay_minutes"].mean())
    global_std = _safe_std(train_df["delay_minutes"])
    headway_median = float(train_df["scheduled_headway"].median())
    if not np.isfinite(headway_median):
        headway_median = 0.0

    route_stats = _stats_frame(train_df, ["route_id"], "route")
    stop_stats = _stats_frame(train_df, ["stop_id"], "stop")
    hour_stats = _stats_frame(train_df, ["hour"], "hour")
    route_hour_stats = _stats_frame(train_df, ["route_hour_key"], "route_hour")
    route_stop_stats = _stats_frame(train_df, ["route_stop_key"], "route_stop")
    route_stop_hour_stats = _stats_frame(
        train_df,
        ["route_stop_hour_key"],
        "route_stop_hour",
    )
    route_stop_order_stats = _stats_frame(
        train_df,
        ["route_stop_order_key"],
        "route_stop_order",
    )
    route_direction_hour_stats = _stats_frame(
        train_df,
        ["route_direction_hour_key"],
        "route_direction_hour",
    )
    route_max_time_point_order = (
        train_df.groupby("route_id", dropna=False)["time_point_order"]
        .max()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_dict()
    )
    global_max_time_point_order = float(
        pd.to_numeric(train_df["time_point_order"], errors="coerce").max()
    )
    if not np.isfinite(global_max_time_point_order) or global_max_time_point_order <= 0:
        global_max_time_point_order = 1.0

    route_max_time_point_order_clean: dict[str, float] = {}
    for route, value in route_max_time_point_order.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value) and numeric_value > 0:
            route_max_time_point_order_clean[str(route)] = numeric_value

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "scheduled_headway_median": headway_median,
        "global_max_time_point_order": global_max_time_point_order,
        "route_max_time_point_order": route_max_time_point_order_clean,
        "route": _stats_map(route_stats, "route_id", "route"),
        "stop": _stats_map(stop_stats, "stop_id", "stop"),
        "hour": {
            str(row["hour"]): {
                "mean": float(row["hour_delay_mean"]),
                "std": float(row["hour_delay_std"]),
            }
            for row in hour_stats.to_dict("records")
        },
        "route_hour": _stats_map(route_hour_stats, "route_hour_key", "route_hour"),
        "route_stop": _stats_map(route_stop_stats, "route_stop_key", "route_stop"),
        "route_stop_hour": _stats_map(
            route_stop_hour_stats,
            "route_stop_hour_key",
            "route_stop_hour",
        ),
        "route_stop_order": _stats_map(
            route_stop_order_stats,
            "route_stop_order_key",
            "route_stop_order",
        ),
        "route_direction_hour": _stats_map(
            route_direction_hour_stats,
            "route_direction_hour_key",
            "route_direction_hour",
        ),
    }


def apply_v4_training_stats(
    dataframe: pd.DataFrame,
    stats: dict[str, Any],
) -> pd.DataFrame:
    df = dataframe.copy()

    def map_stat(map_name: str, key_column: str, output_prefix: str) -> None:
        mapping = stats[map_name]
        df[f"{output_prefix}_delay_mean"] = (
            df[key_column].astype(str).map(lambda key: mapping.get(key, {}).get("mean"))
        )
        df[f"{output_prefix}_delay_std"] = (
            df[key_column].astype(str).map(lambda key: mapping.get(key, {}).get("std"))
        )
        df[f"{output_prefix}_delay_mean"] = df[f"{output_prefix}_delay_mean"].fillna(
            stats["global_mean"]
        )
        df[f"{output_prefix}_delay_std"] = df[f"{output_prefix}_delay_std"].fillna(
            stats["global_std"]
        )

    map_stat("route", "route_id", "route")
    map_stat("stop", "stop_id", "stop")
    map_stat("hour", "hour", "hour")
    map_stat("route_hour", "route_hour_key", "route_hour")
    map_stat("route_stop", "route_stop_key", "route_stop")
    map_stat("route_stop_hour", "route_stop_hour_key", "route_stop_hour")
    map_stat("route_stop_order", "route_stop_order_key", "route_stop_order")
    map_stat("route_direction_hour", "route_direction_hour_key", "route_direction_hour")
    return df


def add_v4_trip_history_features(
    dataframe: pd.DataFrame,
    global_mean: float,
) -> pd.DataFrame:
    """Add lag/rolling features using only previous rows in each trip."""
    df = dataframe.sort_values(
        ["half_trip_id", "scheduled", "time_point_order", "stop_id"],
        na_position="last",
    ).copy()
    grouped = df.groupby("half_trip_id", sort=False)["delay_minutes"]

    shifted_1 = grouped.shift(1)
    df["trip_delay_lag_1"] = shifted_1
    df["trip_delay_lag_2"] = grouped.shift(2)
    df["trip_delay_lag_3"] = grouped.shift(3)

    prior_values = shifted_1
    df["trip_delay_rolling_mean"] = (
        prior_values.groupby(df["half_trip_id"], sort=False)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["trip_delay_rolling_std"] = (
        prior_values.groupby(df["half_trip_id"], sort=False)
        .rolling(window=5, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["trip_delay_rolling_max"] = (
        prior_values.groupby(df["half_trip_id"], sort=False)
        .rolling(window=5, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    for column in [
        "trip_delay_lag_1",
        "trip_delay_lag_2",
        "trip_delay_lag_3",
        "trip_delay_rolling_mean",
        "trip_delay_rolling_max",
    ]:
        df[column] = df[column].fillna(global_mean)
    df["trip_delay_rolling_std"] = df["trip_delay_rolling_std"].fillna(0.0)
    return df.sort_index()


def add_v4_headway_history_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add previous actual headway deviation without using the current actual."""
    df = dataframe.sort_values(["route_id", "stop_id", "scheduled"]).copy()
    group = df.groupby(["route_id", "stop_id"], sort=False)
    actual_headway = group["actual"].diff().dt.total_seconds() / 60
    scheduled_headway = df["scheduled_headway"]
    headway_deviation_current = actual_headway - scheduled_headway
    df["previous_headway_deviation"] = (
        headway_deviation_current.groupby([df["route_id"], df["stop_id"]], sort=False)
        .shift(1)
        .fillna(0.0)
    )
    return df.sort_index()


def finalize_v4_features(dataframe: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    df = dataframe.copy()
    df["scheduled_headway_missing"] = df["scheduled_headway"].isna().astype(int)
    df["scheduled_headway"] = df["scheduled_headway"].fillna(
        stats["scheduled_headway_median"]
    )
    df["scheduled_headway"] = pd.to_numeric(df["scheduled_headway"], errors="coerce").fillna(
        stats["scheduled_headway_median"]
    )
    df["scheduled_headway_log"] = np.log1p(df["scheduled_headway"].clip(lower=0.0))
    df["time_point_order"] = df["time_point_order"].fillna(0.0)
    route_max_order = stats.get("route_max_time_point_order", {})
    denominators = (
        df["route_id"].astype(str)
        .map(lambda route: route_max_order.get(route, stats["global_max_time_point_order"]))
        .astype(float)
        .replace(0.0, stats["global_max_time_point_order"])
    )
    df["stop_sequence_fraction"] = (df["time_point_order"] / denominators).clip(0.0, 1.0)
    df["current_stop_sequence"] = df["time_point_order"]
    df["vehicle_speed"] = 0.0
    df["official_prediction_age_seconds"] = 0.0

    for column in V4_FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def apply_v4_runtime_profile(
    dataframe: pd.DataFrame,
    stats: dict[str, Any],
    encoders: dict[str, dict[str, int]],
    runtime_profile: str = V4_RUNTIME_PROFILE_FULL_HISTORY,
) -> pd.DataFrame:
    """Align offline features with what the realtime runtime can construct.

    `full_history` keeps previous-stop and previous-bus labels for offline
    analysis. `online_safe` matches the current stateless HTTP API: it keeps
    schedule, route/stop/hour, sequence, and optional vehicle fields, but it
    does not train on true trip-history labels that the runtime cannot know.
    """
    if runtime_profile not in V4_RUNTIME_PROFILE_OPTIONS:
        raise ValueError(
            f"Unknown V4 runtime_profile: {runtime_profile}. "
            f"Expected one of {sorted(V4_RUNTIME_PROFILE_OPTIONS)}"
        )
    if runtime_profile == V4_RUNTIME_PROFILE_FULL_HISTORY:
        return dataframe

    df = dataframe.copy()
    unknown_trip_code = encoders["half_trip_id"].get("Unknown", 0)
    global_mean = float(stats["global_mean"])
    df["half_trip_encoded"] = int(unknown_trip_code)
    df["trip_delay_lag_1"] = global_mean
    df["trip_delay_lag_2"] = global_mean
    df["trip_delay_lag_3"] = global_mean
    df["trip_delay_rolling_mean"] = global_mean
    df["trip_delay_rolling_std"] = 0.0
    df["trip_delay_rolling_max"] = global_mean
    df["previous_headway_deviation"] = 0.0
    df["current_stop_sequence"] = df["time_point_order"]
    return df


def build_v4_feature_frame_from_dataframe(
    dataframe: pd.DataFrame,
    encoders: dict[str, dict[str, int]] | None = None,
    stats: dict[str, Any] | None = None,
    runtime_profile: str = V4_RUNTIME_PROFILE_FULL_HISTORY,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, Any]]:
    """Build V4 features for an already normalized dataframe."""
    df = add_v4_time_features(dataframe)
    if encoders is None:
        encoders = {
            "route_id": build_v4_encoder(df["route_id"], include_unknown=False),
            "stop_id": build_v4_encoder(df["stop_id"], include_unknown=False),
            "direction_id": build_v4_encoder(df["direction_id"], include_unknown=True),
            "half_trip_id": build_v4_encoder(df["half_trip_id"], include_unknown=True),
        }
    df = apply_v4_encoders(df, encoders)

    if stats is None:
        stats = build_v4_training_stats(df)
    df = apply_v4_training_stats(df, stats)
    df = add_v4_trip_history_features(df, global_mean=float(stats["global_mean"]))
    df = add_v4_headway_history_features(df)
    df = finalize_v4_features(df, stats)
    df = apply_v4_runtime_profile(
        df,
        stats=stats,
        encoders=encoders,
        runtime_profile=runtime_profile,
    )
    return df, encoders, stats


def split_v4_dataframe(
    dataframe: pd.DataFrame,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = dataframe[dataframe["year"] == 2024].copy()
    validation = dataframe[dataframe["year"] == 2025].copy()
    test = dataframe[dataframe["year"] == 2026].copy()
    if test.empty:
        test = dataframe[dataframe["year"] >= 2025].copy()

    if train.empty:
        raise ValueError("V4 training requires 2024 rows.")
    if validation.empty:
        validation = train.sample(frac=0.15, random_state=random_state).copy()
        train = train.drop(validation.index).copy()
    if test.empty:
        test = validation.copy()

    limits = [
        (max_train_rows, "train"),
        (max_validation_rows, "validation"),
        (max_test_rows, "test"),
    ]
    frames = {"train": train, "validation": validation, "test": test}
    for limit, name in limits:
        if limit is not None and limit > 0 and len(frames[name]) > limit:
            frames[name] = frames[name].sample(n=limit, random_state=random_state).copy()
    return frames["train"], frames["validation"], frames["test"]


def build_v4_dataset(
    dataframe: pd.DataFrame,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
    random_state: int = 42,
    runtime_profile: str = V4_RUNTIME_PROFILE_ONLINE_SAFE,
) -> V4Dataset:
    normalized = normalize_v4_dataframe(dataframe)
    train_raw, validation_raw, test_raw = split_v4_dataframe(
        normalized,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
        random_state=random_state,
    )

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

    metadata = {
        "encoders": encoders,
        "stats": stats,
        "data_summary": {
            "train_rows": int(len(train_features)),
            "validation_rows": int(len(validation_features)),
            "test_rows": int(len(test_features)),
            "years_present": sorted(int(year) for year in normalized["year"].unique()),
            "runtime_profile": runtime_profile,
        },
    }
    return V4Dataset(
        train=train_features,
        validation=validation_features,
        test=test_features,
        feature_columns=list(V4_FEATURE_COLUMNS),
        target_column=V4_TARGET_COLUMN,
        metadata=metadata,
    )


def make_v4_tree_model(model_kind: str = "auto", random_state: int = 42) -> tuple[Any, str]:
    """Create the requested tree model, falling back to sklearn if optional libs are absent."""
    requested = model_kind.lower()
    if requested in {"historical_baseline", "history_baseline", "route_stop_hour_baseline"}:
        return FeatureFallbackRegressor(), "historical_baseline"

    if requested in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMRegressor

            return (
                LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=700,
                    learning_rate=0.04,
                    num_leaves=63,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                "lightgbm",
            )
        except ImportError:
            if requested == "lightgbm":
                raise

    if requested in {"lightgbm_q35", "lightgbm_quantile_35"}:
        from lightgbm import LGBMRegressor

        return (
            LGBMRegressor(
                objective="quantile",
                alpha=0.35,
                n_estimators=700,
                learning_rate=0.04,
                num_leaves=63,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
            ),
            "lightgbm_q35",
        )

    if requested == "catboost":
        from catboost import CatBoostRegressor

        return (
            CatBoostRegressor(
                loss_function="MAE",
                depth=8,
                learning_rate=0.05,
                iterations=700,
                random_seed=random_state,
                verbose=False,
            ),
            "catboost",
        )

    if requested in {"catboost_q35", "catboost_quantile_35"}:
        from catboost import CatBoostRegressor

        return (
            CatBoostRegressor(
                loss_function="Quantile:alpha=0.35",
                depth=8,
                learning_rate=0.05,
                iterations=700,
                random_seed=random_state,
                verbose=False,
            ),
            "catboost_q35",
        )

    if requested == "xgboost":
        from xgboost import XGBRegressor

        return (
            XGBRegressor(
                objective="reg:absoluteerror",
                n_estimators=700,
                learning_rate=0.04,
                max_depth=8,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
            ),
            "xgboost",
        )

    if requested in {"xgboost_q35", "xgboost_quantile_35"}:
        from xgboost import XGBRegressor

        return (
            XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=0.35,
                n_estimators=700,
                learning_rate=0.04,
                max_depth=8,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
            ),
            "xgboost_q35",
        )

    if requested in {"hist_gradient_boosting", "hgb", "hist_gradient_boosting_l1"}:
        return (
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=63,
                l2_regularization=0.01,
                random_state=random_state,
            ),
            "hist_gradient_boosting_l1",
        )

    if requested in {"hist_gradient_boosting_q35", "hgb_q35"}:
        return (
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.35,
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=63,
                l2_regularization=0.01,
                random_state=random_state,
            ),
            "hist_gradient_boosting_q35",
        )

    if requested in {"hist_gradient_boosting_l2", "hgb_l2"}:
        return (
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=63,
                l2_regularization=0.01,
                random_state=random_state,
            ),
            "hist_gradient_boosting_l2",
        )

    if requested in {"extra_trees", "extratrees"}:
        return (
            ExtraTreesRegressor(
                n_estimators=160,
                max_features=0.75,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1,
            ),
            "extra_trees",
        )

    if requested in {"random_forest", "rf"}:
        return (
            RandomForestRegressor(
                n_estimators=120,
                max_features=0.75,
                min_samples_leaf=30,
                random_state=random_state,
                n_jobs=-1,
            ),
            "random_forest",
        )

    if requested in {"gradient_boosting", "gbr"}:
        return (
            GradientBoostingRegressor(
                loss="absolute_error",
                n_estimators=160,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=20,
                random_state=random_state,
            ),
            "gradient_boosting_l1",
        )

    if requested == "ridge":
        return (
            make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=random_state)),
            "ridge",
        )

    if requested == "dummy":
        return (
            DummyRegressor(strategy="median"),
            "dummy_median",
        )

    return (
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=63,
            l2_regularization=0.01,
            random_state=random_state,
        ),
        "hist_gradient_boosting_l1",
    )


def evaluate_v4_model(model: Any, dataframe: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    y_true = dataframe[V4_TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(dataframe[feature_columns]), dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_v4_tree_baseline_from_dataframe(
    dataframe: pd.DataFrame,
    model_kind: str = "auto",
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
    max_test_rows: int | None = None,
    random_state: int = 42,
    runtime_profile: str = V4_RUNTIME_PROFILE_ONLINE_SAFE,
) -> tuple[Any, V4Dataset, dict[str, Any]]:
    dataset = build_v4_dataset(
        dataframe,
        max_train_rows=max_train_rows,
        max_validation_rows=max_validation_rows,
        max_test_rows=max_test_rows,
        random_state=random_state,
        runtime_profile=runtime_profile,
    )
    model, resolved_kind = make_v4_tree_model(model_kind=model_kind, random_state=random_state)

    X_train = dataset.train[dataset.feature_columns]
    y_train = dataset.train[dataset.target_column]
    if resolved_kind == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(dataset.validation[dataset.feature_columns], dataset.validation[dataset.target_column])],
            eval_metric="mae",
            categorical_feature=[
                col for col in V4_CATEGORICAL_FEATURE_COLUMNS if col in dataset.feature_columns
            ],
        )
    elif resolved_kind == "catboost":
        cat_indices = [
            dataset.feature_columns.index(col)
            for col in V4_CATEGORICAL_FEATURE_COLUMNS
            if col in dataset.feature_columns
        ]
        model.fit(
            X_train,
            y_train,
            eval_set=(dataset.validation[dataset.feature_columns], dataset.validation[dataset.target_column]),
            cat_features=cat_indices,
        )
    else:
        model.fit(X_train, y_train)

    metrics = {
        "model_kind": resolved_kind,
        "train": evaluate_v4_model(model, dataset.train, dataset.feature_columns),
        "validation": evaluate_v4_model(model, dataset.validation, dataset.feature_columns),
        "test": evaluate_v4_model(model, dataset.test, dataset.feature_columns),
    }
    return model, dataset, metrics


def extract_v4_feature_importance(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.zeros(len(feature_columns), dtype=float)
    return (
        pd.DataFrame({"feature": feature_columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def build_v4_bundle(
    model: Any,
    dataset: V4Dataset,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bundle_version": "v4.0",
        "model_name": "V4Tree",
        "model_family": "v4_tree",
        "experiment": V4_EXPERIMENT_VERSION,
        "feature_version": V4_FEATURE_VERSION,
        "feature_columns": list(dataset.feature_columns),
        "target_column": dataset.target_column,
        "categorical_feature_columns": list(V4_CATEGORICAL_FEATURE_COLUMNS),
        "model_kind": metrics["model_kind"],
        "model": model,
        "encoders": dataset.metadata["encoders"],
        "stats": dataset.metadata["stats"],
        "data_summary": dataset.metadata["data_summary"],
        "metrics": metrics,
        "feature_importance": extract_v4_feature_importance(
            model,
            dataset.feature_columns,
        ).to_dict("records"),
    }


def save_v4_bundle(bundle: dict[str, Any], output_path: str | Path = V4_BUNDLE_PATH) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path
