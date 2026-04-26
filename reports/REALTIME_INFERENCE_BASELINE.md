# Realtime Inference Baseline Report

Generated: 2026-04-24 21:10:50

## Configuration

- Checkpoint: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\models\delay_predictor_mlp_v2_lag_features_temporal.pt`
- Iterations: `200`
- Warmup: `20`
- Platform: `Windows-11-10.0.26200-SP0`
- Figure: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\figures\realtime_inference_latency_baseline.png`

## Latency Summary

| Target | Calls | Min (ms) | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|--------|------:|---------:|---------:|---------:|---------:|---------:|
| runtime.predict | 200 | 0.123 | 0.150 | 0.132 | 0.258 | 0.391 |

## API Benchmark

Skipped: `FastAPI and its dependencies are required for the realtime API. Install `fastapi` before importing src.inference.api.`

## Verification

- Runtime path verified with avg latency `0.150 ms`.
- Runtime path verified with p95 latency `0.258 ms`.
