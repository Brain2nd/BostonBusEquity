"""
Build a single-file realtime inference bundle for the V2 causal MLP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.inference.bundle_utils import (
    DATA_PROCESSED,
    V2_FEATURE_COLUMNS,
    add_v2_time_features,
    build_category_mappings,
    build_v2_feature_frame,
    coerce_float_dict,
    compute_v2_training_statistics,
    fit_scaler_values,
    load_arrival_departure_dataframe,
    make_feature_matrix,
    safe_sample,
    split_temporal_train_test,
)
from src.models.v2_mlp import V2_DROPOUT, V2_HIDDEN_SIZES, V2MLPPredictor

DEFAULT_CHECKPOINT = Path("models/delay_predictor_mlp_v2_lag_features_temporal.pt")
DEFAULT_BUNDLE = Path("models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt")


def build_realtime_bundle(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_path: Path = DEFAULT_BUNDLE,
    max_files: int | None = None,
    sample_per_file: int | None = 12_000,
    max_train_rows: int = 500_000,
    max_test_rows: int = 100_000,
    years: list[int] | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    df = load_arrival_departure_dataframe(
        max_files=max_files,
        sample_per_file=sample_per_file,
        years=years,
    )
    df = add_v2_time_features(df)
    train_df, test_df = split_temporal_train_test(df)
    if train_df.empty:
        raise ValueError("No training rows (<2025) available for realtime bundle construction")

    train_df = safe_sample(train_df, max_train_rows)
    test_df = safe_sample(test_df, max_test_rows)

    mappings = build_category_mappings(train_df, test_df)
    stats = compute_v2_training_statistics(train_df)

    train_frame = build_v2_feature_frame(train_df, mappings, stats)
    train_X = make_feature_matrix(train_frame)
    train_y = train_frame["delay_minutes"].to_numpy(dtype="float32").reshape(-1, 1)

    scaler_X = fit_scaler_values(train_X)
    scaler_y = fit_scaler_values(train_y)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = V2MLPPredictor(
        input_size=len(V2_FEATURE_COLUMNS),
        hidden_sizes=V2_HIDDEN_SIZES,
        dropout=V2_DROPOUT,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    bundle = {
        "bundle_version": 1,
        "model_name": "V2MLPPredictor",
        "experiment": "v2_lag_features_temporal",
        "feature_version": "v2_causal_realtime",
        "feature_columns": V2_FEATURE_COLUMNS,
        "model_config": {
            "input_size": len(V2_FEATURE_COLUMNS),
            "hidden_sizes": V2_HIDDEN_SIZES,
            "dropout": V2_DROPOUT,
        },
        "model_state_dict": model.state_dict(),
        "scaler_X": {"mean": scaler_X.mean, "scale": scaler_X.scale},
        "scaler_y": {"mean": scaler_y.mean, "scale": scaler_y.scale},
        "mappings": mappings,
        "statistics": {
            "route_delay_mean": coerce_float_dict(stats.route_delay_mean),
            "route_delay_std": coerce_float_dict(stats.route_delay_std),
            "stop_delay_mean": coerce_float_dict(stats.stop_delay_mean),
            "stop_delay_std": coerce_float_dict(stats.stop_delay_std),
            "hour_delay_mean": coerce_float_dict(stats.hour_delay_mean),
            "route_hour_delay_mean": coerce_float_dict(stats.route_hour_delay_mean),
            "global_mean": float(stats.global_mean),
            "global_std": float(stats.global_std),
            "scheduled_headway_median": float(stats.scheduled_headway_median),
        },
        "build_metadata": {
            "checkpoint_path": str(checkpoint_path),
            "source_parquet_exists": (DATA_PROCESSED / "arrival_departure.parquet").exists(),
            "train_rows_used": int(len(train_df)),
            "test_rows_seen": int(len(test_df)),
            "max_files": max_files,
            "sample_per_file": sample_per_file,
            "years": years,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build realtime inference bundle for V2 MLP")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample-per-file", type=int, default=12_000)
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--max-test-rows", type=int, default=100_000)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    args = parser.parse_args()

    bundle = build_realtime_bundle(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_files=args.max_files,
        sample_per_file=args.sample_per_file,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        years=args.years,
    )
    print(f"Bundle saved to: {args.output}")
    print(f"Train rows used: {bundle['build_metadata']['train_rows_used']:,}")
    print(f"Per-file sample: {bundle['build_metadata']['sample_per_file']}")
    print(f"Years: {bundle['build_metadata']['years']}")
    print(f"Source parquet exists: {bundle['build_metadata']['source_parquet_exists']}")


if __name__ == "__main__":
    main()
