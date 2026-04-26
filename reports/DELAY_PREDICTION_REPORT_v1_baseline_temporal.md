# Bus Delay Prediction - V1 Baseline (Temporal Split)

**Date:** 2026-02-12 23:56:19
**Author:** Boston Bus Equity Team

---

## 1. Introduction

Bus delay prediction using Artificial Neural Networks (ANN).
**Data Split:** Train on <2025, Test on >=2025 (Temporal Split)

Models:
1. MLP (Multi-Layer Perceptron) - Baseline feedforward network
2. LSTM (Long Short-Term Memory) - Hochreiter & Schmidhuber (1997)
3. GRU (Gated Recurrent Unit) - Cho et al. (2014)

## 2. Features (9)

hour, day_of_week, month, is_weekend, is_rush_hour, route_encoded, stop_encoded, direction_encoded, scheduled_headway

## 3. Results

| model   | experiment           |     MSE |    RMSE |     MAE |         R2 |        MAPE |
|:--------|:---------------------|--------:|--------:|--------:|-----------:|------------:|
| MLP     | v1_baseline_temporal | 38.9833 | 6.24366 | 4.37904 | -0.0742275 | 6.96955e+06 |
| LSTM    | v1_baseline_temporal | 39.9716 | 6.32231 | 4.42626 | -0.101462  | 6.83761e+06 |
| GRU     | v1_baseline_temporal | 43.2838 | 6.57905 | 4.58505 | -0.192735  | 7.57559e+06 |

### Best Model

**MLP** achieved the best RMSE of **6.2437 minutes**.

## 4. Training Time

| Model | Training Time (s) |
|-------|-------------------|
| MLP | 594.25 |
| LSTM | 542.81 |
| GRU | 655.11 |


## 5. Visualizations

### Training Curves
![Training Curves](figures/delay_prediction_training_curves_v1_baseline_temporal.png)

---

*Report generated automatically*
