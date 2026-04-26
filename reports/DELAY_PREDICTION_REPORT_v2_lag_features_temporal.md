# Bus Delay Prediction - V2 Lag Features (Temporal Split)

**Date:** 2026-02-13 00:35:50
**Author:** Boston Bus Equity Team

---

## 1. Introduction

Bus delay prediction with enhanced features.
**Data Split:** Train on <2025, Test on >=2025 (Temporal Split)

Improvements over V1:
1. Cyclical encoding for temporal features
2. Historical statistics (route/stop/hour mean/std)
3. All statistics computed ONLY on training data (no leakage)

## 2. Features (18)

is_weekend, is_rush_hour, route_encoded, stop_encoded, direction_encoded, scheduled_headway, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, route_delay_mean, route_delay_std, stop_delay_mean, stop_delay_std, hour_delay_mean, route_hour_delay_mean

## 3. Results

| model   | experiment               |     MSE |    RMSE |     MAE |        R2 |
|:--------|:-------------------------|--------:|--------:|--------:|----------:|
| MLP     | v2_lag_features_temporal | 40.6045 | 6.37216 | 4.40651 | -0.118902 |
| LSTM    | v2_lag_features_temporal | 40.1558 | 6.33686 | 4.37034 | -0.106538 |
| GRU     | v2_lag_features_temporal | 40.439  | 6.35917 | 4.39784 | -0.114343 |

### Best Model

**LSTM** achieved the best RMSE of **6.3369 minutes**.

## 4. Training Time

| Model | Training Time (s) |
|-------|-------------------|
| MLP | 605.84 |
| LSTM | 472.10 |
| GRU | 1056.10 |


## 5. Visualizations

### Training Curves
![Training Curves](figures/delay_prediction_training_curves_v2_lag_features_temporal.png)

---

*Report generated automatically*
