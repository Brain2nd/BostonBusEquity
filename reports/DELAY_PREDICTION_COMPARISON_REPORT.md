# Bus Delay Prediction - Experiment Comparison Report

**Date:** 2026-02-12 22:41:25
**Author:** Boston Bus Equity Team

---

## 1. Experiment Summary

| Experiment | Description | Features | Data Split |
|------------|-------------|----------|------------|
| V1 (Baseline) | MLP/LSTM/GRU with basic features | 9 | Random 70/15/15 |
| V2 (Lag Features) | + Cyclical encoding + Historical stats | 18 | Random 70/15/15 |
| V3 (Wavelet + Temporal) | + Wavelet decomposition + Rolling stats | 28 | **Temporal: <2025 train, >=2025 test** |

## 2. Results Comparison

| model   | experiment          |        MSE |      RMSE |       MAE |       R2 |
|:--------|:--------------------|-----------:|----------:|----------:|---------:|
| MLP     | v3_wavelet_temporal | 0.202562   | 0.450069  | 0.279635  | 0.994408 |
| LSTM    | v3_wavelet_temporal | 0.0937558  | 0.306196  | 0.196834  | 0.997412 |
| GRU     | v3_wavelet_temporal | 0.00971198 | 0.0985494 | 0.0696771 | 0.999732 |

### Best Model per Experiment

- **v3_wavelet_temporal**: GRU (RMSE: 0.0985, R²: 0.9997)


## 3. Key Findings

1. **Temporal split** provides realistic evaluation - training on historical data, testing on future data

2. **Historical statistics** from training data help capture route/stop-specific patterns without leakage

3. **Wavelet decomposition** helps separate signal from noise and captures multi-scale patterns

4. **Lag features** (strictly from past observations) allow the model to learn temporal dependencies

## 4. Visualizations

### Training Curves
- V3: `figures/delay_prediction_training_curves_v3_wavelet_temporal.png`

### Model Comparison
- `figures/delay_prediction_all_experiments_comparison.png`

## 5. Recommendations

For production deployment, the V3 model with temporal split provides the most realistic performance estimates,
as it mimics how the model would perform predicting future delays based on historical patterns.

---

*Report generated automatically*
