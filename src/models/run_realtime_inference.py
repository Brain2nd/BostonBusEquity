"""CLI for real-time inference and baseline latency benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.models.realtime_inference import (
    DEFAULT_BASELINE_CHECKPOINT,
    MBTARealtimeAdapter,
    RealtimeDelayPredictor,
    build_demo_historical_frame,
    build_demo_live_records,
    benchmark_predictor,
    format_benchmark_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time delay inference with existing project checkpoints.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_BASELINE_CHECKPOINT),
        help="Path to an existing project checkpoint (.pt). Defaults to the baseline MLP.",
    )
    parser.add_argument(
        "--historical-data",
        help="Optional CSV/Parquet file used to rebuild preprocessing artifacts. "
        "If omitted, a lightweight demo dataset is used.",
    )
    parser.add_argument(
        "--artifacts",
        help="Optional path to save or load cached preprocessing artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Maximum number of historical rows to sample when rebuilding artifacts.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference. Use 'auto' to prefer GPU when available.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the baseline latency benchmark instead of printing predictions.",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup iterations.")
    parser.add_argument(
        "--live-json",
        help="Path to a JSON file containing normalized live records or a MBTA-style API payload.",
    )
    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Include feature values in prediction output.",
    )
    return parser.parse_args()


def load_history(path: str | None):
    if not path:
        return build_demo_historical_frame()
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        import pandas as pd

        return pd.read_parquet(target)
    if target.suffix.lower() == ".csv":
        import pandas as pd

        return pd.read_csv(target)
    raise ValueError(f"Unsupported historical data file type: {target.suffix}")


def load_live_records(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return build_demo_live_records()

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    adapter = MBTARealtimeAdapter()
    records = adapter.normalize_records(payload)
    if records:
        return records
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Live JSON file did not contain usable records.")


def build_predictor(args: argparse.Namespace) -> RealtimeDelayPredictor:
    checkpoint_path = Path(args.checkpoint)
    historical_data = load_history(args.historical_data)
    artifacts_path = Path(args.artifacts) if args.artifacts else None

    if artifacts_path and artifacts_path.exists():
        predictor = RealtimeDelayPredictor.from_artifacts_file(
            checkpoint_path=checkpoint_path,
            artifacts_path=artifacts_path,
            device=args.device,
        )
        predictor.warm_state_from_historical_data(historical_data)
        return predictor

    return RealtimeDelayPredictor.bootstrap(
        checkpoint_path=checkpoint_path,
        historical_data=historical_data,
        artifacts_path=artifacts_path,
        device=args.device,
        sample_size=args.sample_size,
    )


def main() -> None:
    args = parse_args()
    predictor = build_predictor(args)
    live_records = load_live_records(args.live_json)

    if args.benchmark:
        summary = benchmark_predictor(
            predictor,
            live_records,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        print(format_benchmark_summary(summary))
        return

    results = predictor.predict_many(live_records)
    payload = [result.to_dict(include_feature_values=args.print_features) for result in results]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
