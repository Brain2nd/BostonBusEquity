# GitHub Submission Guide

Target repository: `https://github.com/Brain2nd/BostonBusEquity`

## Important Git Safety Note

On this machine, running `git rev-parse --show-toplevel` from the project folder resolves to:

```text
C:/Users/yaobc
```

That means the current project folder is inside a parent Git repository rooted at the Windows user directory. Do **not** run `git add .` from `C:\Users\yaobc` or from this project until the project has its own `.git` folder or is copied into a clean clone of the target GitHub repository.

## Recommended Files To Submit

Core project files:

```text
.gitignore
README.md
requirements.txt
src/
tests/
docs/
```

Model/dashboard artifacts needed for the current demo:

```text
models/delay_predictor_v4_score_best_online_safe_bundle.joblib
models/delay_predictor_v4_best_online_safe_bundle.joblib
models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt
```

Reports and figures:

```text
reports/DELAY_PREDICTION_V4_OPTIMIZATION_REPORT.md
reports/MBTA_REALTIME_OFFICIAL_VS_MODEL.md
reports/MODEL_SCORING_GUIDE.md
reports/PRESENTATION_FIGURE_NOTES.md
reports/REALTIME_INFERENCE_BASELINE.md
reports/REALTIME_INFERENCE_PREDICTION_OVER_TIME.md
reports/V4_MODEL_SWEEP_REPORT.md
reports/V4_OPTIMIZATION_RESEARCH_NOTES.md
reports/V4_SCORE_BEST_BUNDLE_TRAIN_REPORT.md
reports/V5_RESIDUAL_DATASET_REPORT.md
reports/figures/delay_distribution.png
reports/figures/delays_by_route.png
reports/figures/demographic_correlations_heatmap.png
reports/figures/official_vs_v4_vs_actual.png
reports/figures/v4_model_sweep.png
reports/figures/v4_model_deployability_scores.png
reports/figures/v4_optimization_story.png
reports/figures/mbta_realtime_model_gap_story.png
reports/figures/mbta_realtime_official_vs_model.png
```

Optional small metric CSV files, if the graders want raw metric tables:

```text
reports/delay_prediction_v4_model_scores.csv
reports/delay_prediction_v4_model_sweep_summary.csv
reports/delay_prediction_v4_score_best_bundle_train_summary.csv
reports/mbta_realtime_official_vs_model.csv
```

## Files And Folders To Exclude

Do not submit local caches or raw/generated large data:

```text
data/raw/
data/processed/
.pytest_tmp/
pytest-cache-files-*/
tmp*/
catboost_info/
__pycache__/
reports/dashboard_server_*.log
reports/v4_test_predictions.csv
reports/v4_history_upper_bound_test_predictions.csv
reports/tmp_*
reports/figures/tmp_*
models/tmp_*
```

## Safe Submission Workflow

Use a clean clone of the GitHub repository:

```powershell
cd C:\Users\yaobc\Downloads\hw542
git clone https://github.com/Brain2nd/BostonBusEquity.git BostonBusEquity-submit
```

Copy the project files from `BostonBusEquity-main` into `BostonBusEquity-submit`, excluding the folders listed above. Then from inside the clean clone:

```powershell
cd C:\Users\yaobc\Downloads\hw542\BostonBusEquity-submit
git status --short
git add .gitignore README.md requirements.txt docs src tests `
  reports/DELAY_PREDICTION_V4_OPTIMIZATION_REPORT.md `
  reports/MBTA_REALTIME_OFFICIAL_VS_MODEL.md `
  reports/MODEL_SCORING_GUIDE.md `
  reports/PRESENTATION_FIGURE_NOTES.md `
  reports/REALTIME_INFERENCE_BASELINE.md `
  reports/REALTIME_INFERENCE_PREDICTION_OVER_TIME.md `
  reports/V4_MODEL_SWEEP_REPORT.md `
  reports/V4_OPTIMIZATION_RESEARCH_NOTES.md `
  reports/V4_SCORE_BEST_BUNDLE_TRAIN_REPORT.md `
  reports/V5_RESIDUAL_DATASET_REPORT.md `
  reports/figures/delay_distribution.png `
  reports/figures/delays_by_route.png `
  reports/figures/demographic_correlations_heatmap.png `
  reports/figures/official_vs_v4_vs_actual.png `
  reports/figures/v4_model_sweep.png `
  reports/figures/v4_model_deployability_scores.png `
  reports/figures/v4_optimization_story.png `
  reports/figures/mbta_realtime_model_gap_story.png `
  reports/figures/mbta_realtime_official_vs_model.png `
  models/delay_predictor_v4_score_best_online_safe_bundle.joblib `
  models/delay_predictor_v4_best_online_safe_bundle.joblib `
  models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt
git status --short
git commit -m "Add April check-in dashboard and V4 realtime model documentation"
git push origin main
```

If the target repository uses a different default branch, replace `main` with the branch shown by:

```powershell
git branch --show-current
```

## Demo Commands

Start the dashboard:

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.inference.serve `
  --bundle models\delay_predictor_v4_score_best_online_safe_bundle.joblib `
  --host 0.0.0.0 `
  --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

For another device on the same Wi-Fi:

```text
http://192.168.1.179:8000/
```

Run quick validation:

```powershell
C:\Users\yaobc\anaconda3\python.exe -m pytest tests\test_realtime_inference.py tests\test_v4_delay_predictor.py -q --basetemp .pytest_tmp\submission -o cache_dir=.pytest_tmp\cache
```
