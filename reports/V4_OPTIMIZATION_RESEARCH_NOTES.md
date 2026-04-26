# V4/V5 Optimization Research Notes

## What changed

- Reframed evaluation around true delay labels (`actual - scheduled`), not MBTA official prediction disagreement.
- Added an `online_safe` V4 profile that removes true trip-history labels unavailable to the current stateless HTTP API.
- Kept a `full_history` V4 research model as an upper bound for a future live-history cache.
- Added a model-family sweep across feature profiles and a final 2024+2025 retrain for 2026 deployment testing.
- Added cleaner presentation figures that separate deployment decision, research upper bound, and live model disagreement.

## Evidence from web research

- MBTA V3 predictions expose predicted arrival/departure, status, schedule relationship, and stop sequence; this supports live comparison but not true accuracy without later actual labels.
- MBTA V3 vehicles expose live vehicle state such as current stop sequence/status and vehicle relationships; these are appropriate online features.
- Jeong & Rilett (2005) emphasize AVL, schedule adherence, traffic congestion, and dwell time for real-time bus arrival prediction.
- Shalaby & Farhan-style AVL/APC work separates running time and dwell time and uses recent historical/current-day information.
- LightGBM/boosting remains a good tabular baseline, but the feature availability profile matters more than model class here.

## Decision

- The raw 2024-only `online_safe` V4 model should not replace V2; it underperformed on true-label test MAE.
- The best current deployable candidate is a constrained V4 HistGradientBoosting model using V2-core causal features and a 2024+2025 final retrain.
- Use the `full_history` V4 result to justify the next engineering step: capture live previous-stop/history labels or train V5 official residual correction once matched labels reach 500+ rows.

## References

- MBTA Prediction API: https://hexdocs.pm/mbta_sdk/MBTA.Api.Prediction.html
- MBTA Vehicle API: https://hexdocs.pm/mbta_sdk/MBTA.Api.Vehicle.html
- GTFS Realtime Reference: https://gtfs.org/documentation/realtime/reference/
- Jeong & Rilett 2005: https://journals.sagepub.com/doi/10.1177/0361198105192700123
- Shalaby & Farhan 2004: https://www.sciencedirect.com/science/article/pii/S1077291X22003812
- LightGBM LGBMRegressor docs: https://lightgbm.readthedocs.io/en/v4.0.0/pythonapi/lightgbm.LGBMRegressor.html
