# Model Scoring Guide

Generated: 2026-04-25 10:05:30

## Purpose

The model sweep should not be ranked only by MAE. For a realtime bus-delay dashboard, a model also needs to generalize across years, use features that are available online, detect early/negative delay when measured, and remain cheap enough to retrain.

## Composite Score

- Accuracy score, 40%: lower 2026 true-label MAE is better. If final 2024+2025 -> 2026 metrics exist, they are used; otherwise test metrics are used.
- Stability score, 15%: lower train/test overfit and validation/test gap is better.
- Online readiness score, 15%: online-safe feature profiles receive more credit than feature-heavy profiles that depend on fields often missing at inference time.
- Early-delay score, 20%: rewards negative-delay F1 and early-event MAE. This matters because the user noticed the model under-predicts early arrivals.
- Cost score, 10%: faster and simpler models receive a small bonus.
- Early viability gate: subtracts 25 points and caps the final score at 55 when the test data has meaningful early arrivals but the model almost never predicts negative delay.

Early-delay note: Early-delay score uses measured negative-delay F1 with an early-event MAE tie-breaker.

## Top Models

```text
 rank                 model_kind feature_profile  composite_score  primary_mae  primary_rmse  primary_r2  accuracy_score  stability_score  online_readiness_score  early_delay_score  early_viability_penalty early_viability_gate  cost_score  final_2024_2025_to_2026_early_f1  test_early_f1  final_2024_2025_to_2026_negative_prediction_rate  test_negative_prediction_rate
    1               lightgbm_q35         v2_core        85.349035     3.938519      6.193354   -0.036550       90.619313        96.454015                   100.0          56.075275                      0.0                 pass   84.181528                          0.372504       0.282189                                           0.17135                        0.13180
    2 hist_gradient_boosting_q35         v2_core        85.072052     3.948191      6.199681   -0.038668       89.570650        97.284853                   100.0          55.172993                      0.0                 pass   86.164651                          0.362304       0.266611                                           0.16090                        0.11570
    3               catboost_q35         v2_core        82.468118     3.965743      6.213445   -0.043286       87.667574        94.068420                   100.0          56.326602                      0.0                 pass   70.255045                          0.376200       0.269481                                           0.18005                        0.12860
    4  hist_gradient_boosting_l1         v2_core        81.965037     3.854319      5.947843    0.044001       99.748614        83.975114                   100.0          32.712102                      0.0                 pass   79.269040                          0.207634       0.159654                                           0.05295                        0.04775
    5                   lightgbm         v2_core        80.939723     3.852001      5.952205    0.042598      100.000000        74.180591                   100.0          34.059879                      0.0                 pass   80.006581                          0.217981       0.165372                                           0.05685                        0.04865
    6                   catboost         v2_core        80.023346     3.861185      5.969750    0.036946       99.004244        75.460048                   100.0          34.745524                      0.0                 pass   71.535366                          0.226588       0.189246                                           0.06360                        0.06865
    7                    xgboost         v2_core        74.308781     3.856543      5.954699    0.041796       99.507467        50.873734                   100.0          34.286734                      0.0                 pass   50.173874                          0.224092       0.172743                                           0.06120                        0.05250
    8 hist_gradient_boosting_q35      stats_time        56.087396     4.308560      6.566891   -0.165354       50.497885        42.210854                    90.0          54.182703                      0.0                 pass   52.200733                          0.394726       0.333296                                           0.23980                        0.20245
    9               lightgbm_q35      stats_time        53.986335     4.316195      6.582281   -0.170823       49.670010        32.021830                    90.0          54.854959                      0.0                 pass   48.440650                          0.398657       0.332803                                           0.24960                        0.19890
   10        historical_baseline         v2_core        53.719969     4.418326      6.235905   -0.050841       38.596616        84.237075                   100.0           3.239354                      0.0                 pass   99.978907                          0.046276       0.039538                                           0.01535                        0.01445
   11  hist_gradient_boosting_l1      stats_time        53.535229     4.219690      6.371790   -0.097138       60.133542        22.362413                    90.0          36.531465                      0.0                 pass   53.211570                          0.289377       0.226246                                           0.11310                        0.09315
   12               catboost_q35      stats_time        52.049291     4.343191      6.621116   -0.184679       46.743034        35.068713                    90.0          54.577088                      0.0                 pass   36.763528                          0.396625       0.325289                                           0.25010                        0.19615
   13                   catboost      stats_time        51.577956     4.218271      6.370632   -0.096739       60.287366        16.614374                    90.0          38.600392                      0.0                 pass   37.507751                          0.304795       0.238724                                           0.12730                        0.10375
   14                   lightgbm      stats_time        51.126370     4.224134      6.379735   -0.099876       59.651707         9.829526                    90.0          37.265905                      0.0                 pass   48.380769                          0.296131       0.232770                                           0.11615                        0.09925
   15               dummy_median         v2_core        49.003837     4.037800      6.099936   -0.005516       79.854848       100.000000                   100.0          10.309489                     25.0               capped  100.000000                          0.000000       0.000000                                           0.00000                        0.00000
   16                    xgboost      stats_time        44.972998     4.221697      6.379551   -0.099812       59.915920         0.000000                    90.0          37.533148                      0.0                 pass    0.000000                          0.300348       0.225335                                           0.11745                        0.09095
   17               dummy_median      stats_time        44.000894     4.037800      6.099936   -0.005516       79.854848       100.000000                    90.0          10.309489                     25.0               capped   64.970568                          0.000000       0.000000                                           0.00000                        0.00000
   18        historical_baseline      stats_time        24.536529     4.774303      6.805060   -0.251417        0.000000         8.419394                    90.0          16.397583                      0.0                 pass   64.941036                          0.209616       0.186090                                           0.08785                        0.08250
```

CSV: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\delay_prediction_v4_model_scores.csv`
Figure: `C:\Users\yaobc\Downloads\hw542\BostonBusEquity-main\reports\figures\v4_model_deployability_scores.png`

## Baseline Policy

The sweep already includes a `dummy_median` model as a statistical baseline. The dashboard also exposes a route-stop-hour historical baseline for realtime comparison. Future matched-live evaluation should add MBTA official MAE and V5 residual-correction MAE once actual labels are available.
