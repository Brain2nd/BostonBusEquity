"""
Evaluate simple temporal baselines for delay prediction on the 2025+ split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.inference.bundle_utils import (
    add_v2_time_features,
    build_category_mappings,
    build_v2_feature_frame,
    compute_v2_training_statistics,
    load_arrival_departure_dataframe,
    safe_sample,
    split_temporal_train_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def regression_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "baseline": name,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_baselines(max_files: int | None = None) -> pd.DataFrame:
    df = load_arrival_departure_dataframe(max_files=max_files)
    df = add_v2_time_features(df)
    train_df, test_df = split_temporal_train_test(df)
    if train_df.empty or test_df.empty:
        raise ValueError("Both training (<2025) and test (>=2025) rows are required for baseline evaluation")

    train_df = safe_sample(train_df, 500_000)
    test_df = safe_sample(test_df, 100_000)

    mappings = build_category_mappings(train_df, test_df)
    stats = compute_v2_training_statistics(train_df)
    test_frame = build_v2_feature_frame(test_df, mappings, stats)

    y_true = test_frame["delay_minutes"].to_numpy(dtype=float)
    results = [
        regression_metrics("zero_delay", y_true, np.zeros_like(y_true)),
        regression_metrics("global_mean", y_true, np.full_like(y_true, stats.global_mean)),
        regression_metrics("route_mean", y_true, test_frame["route_delay_mean"].to_numpy(dtype=float)),
        regression_metrics("stop_mean", y_true, test_frame["stop_delay_mean"].to_numpy(dtype=float)),
        regression_metrics("hour_mean", y_true, test_frame["hour_delay_mean"].to_numpy(dtype=float)),
        regression_metrics(
            "route_hour_mean",
            y_true,
            test_frame["route_hour_delay_mean"].to_numpy(dtype=float),
        ),
    ]
    return pd.DataFrame(results).sort_values("RMSE")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    baseline_df = evaluate_baselines()
    output_path = REPORTS_DIR / "delay_prediction_baselines_temporal.csv"
    baseline_df.to_csv(output_path, index=False)
    print(baseline_df.to_string(index=False))
    print(f"\nSaved baseline metrics to: {output_path}")


if __name__ == "__main__":
    main()
