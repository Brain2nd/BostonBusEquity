# V4 Model Sweep Report

Generated: 2026-04-25 08:52:12
Experiment: `v4_tree_causal_live_features`

## Dependency Status

- lightgbm installed: `True`
- catboost installed: `True`
- xgboost installed: `True`

These optional boosting libraries are not installed in the current environment unless marked `True`; the sweep therefore uses sklearn candidates that can run now.

## Data

- Runtime profile: `online_safe`
- Train rows: `100000`
- Validation rows: `50000`
- Test rows: `20000`
- Years present: `[2024, 2025, 2026]`

## Results

```text
model_kind_requested feature_profile  feature_count status error  train_MAE  train_RMSE  train_R2  train_early_recall  train_early_precision  train_early_f1  train_early_MAE  train_early_share  train_negative_prediction_rate  validation_MAE  validation_RMSE  validation_R2  validation_early_recall  validation_early_precision  validation_early_f1  validation_early_MAE  validation_early_share  validation_negative_prediction_rate  test_MAE  test_RMSE   test_R2  test_early_recall  test_early_precision  test_early_f1  test_early_MAE  test_early_share  test_negative_prediction_rate   model_kind  fit_seconds  final_2024_2025_to_2026_MAE  final_2024_2025_to_2026_RMSE  final_2024_2025_to_2026_R2  final_2024_2025_to_2026_early_recall  final_2024_2025_to_2026_early_precision  final_2024_2025_to_2026_early_f1  final_2024_2025_to_2026_early_MAE  final_2024_2025_to_2026_early_share  final_2024_2025_to_2026_negative_prediction_rate
        lightgbm_q35         v2_core             18     ok         4.010511    6.567959  0.073081            0.404932                0.60751        0.485954         3.052346            0.19139                         0.12757        3.954415         5.951239      -0.013819                 0.213536                    0.410638             0.280967              4.335665                 0.22488                              0.11694  4.021973   6.200301 -0.043445           0.218257               0.39909       0.282189        4.307594             0.241                         0.1318 lightgbm_q35     4.154695                     3.938519                      6.193354                    -0.03655                              0.318238                                 0.449081                          0.372504                            3.50749                               0.2418                                           0.17135
```

Deployment decision: `replace candidate after final prior-year retrain`
Best bundle: `models\delay_predictor_v4_score_best_online_safe_bundle.joblib`
Figure: `reports\figures\v4_score_best_bundle_train.png`

## Failed Candidates

```text
None
```

## Interpretation

If the best online-safe V4 model still trails V2, the blocker is feature availability more than model class: the stateless realtime API lacks previous-stop delay, current trip adherence, and matched official residual labels. If the `final_2024_2025_to_2026_*` columns beat V2, that is a deployable 2026-style retrain using all prior-year labels, but it should still be described separately from the original 2024-only validation protocol.
