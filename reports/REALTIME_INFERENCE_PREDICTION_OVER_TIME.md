# Realtime Inference Prediction Trace

Generated: 2026-04-24 21:22:08 EDT

## Configuration

- Checkpoint: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\models\delay_predictor_mlp_v2_lag_features_temporal.pt`
- Bundle mode: `demo_bundle`
- Service date: `2026-04-24`
- Sampling cadence: `30` minutes
- Figure: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\figures\realtime_inference_prediction_over_time.png`
- CSV: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\realtime_inference_prediction_over_time.csv`

## Series

- `Route 1 / Stop A`: route `1`, stop `A`, direction `0`, headway `12`
- `Route 2 / Stop B`: route `2`, stop `B`, direction `0`, headway `10`
- `Route 28 / Stop C`: route `28`, stop `C`, direction `0`, headway `8`

## Warnings

- No realtime bundle was found. Generated a demo bundle from synthetic data for plotting.

## Summary

| Series | Min (min) | Avg (min) | Max (min) |
|--------|----------:|----------:|----------:|
| Route 1 / Stop A | 2.369 | 2.692 | 2.989 |
| Route 2 / Stop B | 2.752 | 3.041 | 3.330 |
| Route 28 / Stop C | 2.983 | 3.369 | 3.977 |

## Verification

- Generated `111` realtime predictions across `3` series.
- Saved the full prediction trace to CSV for inspection and reuse.
