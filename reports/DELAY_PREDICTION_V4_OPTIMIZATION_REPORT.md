# Delay Prediction V4 Optimization Report

Generated: 2026-04-25 06:59:29

## Goal

V4 predicts true delay (`actual - scheduled`). MBTA official predictions are a comparison baseline, not the training target.

## Data Split

- Years present: `[2024, 2025, 2026]`
- Train rows: `280803`
- Validation rows: `46785`
- Test rows: `47818`
- Runtime profile: `online_safe`

## Metrics

```text
                        model     scope          split group      n      MAE     RMSE        R2
V4Tree-hist_gradient_boosting   overall          train   all    NaN 2.640933 4.098661  0.455122
V4Tree-hist_gradient_boosting   overall     validation   all    NaN 4.290608 6.532764 -0.855453
V4Tree-hist_gradient_boosting   overall           test   all    NaN 4.503392 6.556099 -0.954043
         V2MLP-current-bundle v2_sample test_head_5000   all 5000.0 3.301296 4.291643 -0.145398
```

Acceptance decision vs current V2 sample: `hold V2 default`

## Top Features

```text
                   feature  importance
          time_point_order    0.039106
route_stop_hour_delay_mean    0.023258
         scheduled_headway    0.014755
           stop_delay_mean    0.005138
              stop_encoded    0.002789
                   dow_cos    0.000962
                is_weekend    0.000000
             vehicle_speed    0.000000
previous_headway_deviation    0.000000
                 month_sin    0.000000
```

## Outputs

- Bundle: `models\delay_predictor_v4_tree_realtime_bundle.joblib`
- Metrics CSV: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\delay_prediction_metrics_v4.csv`
- Test predictions CSV: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\v4_test_predictions.csv`
- Comparison figure: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\figures\v4_model_comparison.png`
- Feature importance figure: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\figures\v4_feature_importance.png`
- Actual-vs-V4 figure: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\figures\official_vs_v4_vs_actual.png`
- Optimization dashboard: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\figures\v4_optimization_diagnostics.png`

## Interpretation

If V4 still trails MBTA official live predictions, that is expected: V4 is an independent true-delay model using schedule/history/live vehicle fields only. The later V5 residual model is the correct place to calibrate MBTA official predictions once enough matched live labels exist.
