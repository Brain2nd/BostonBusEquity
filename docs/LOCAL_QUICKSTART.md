# Local Quickstart

This guide is for teammates or reviewers who want to run the current April check-in deliverables locally with minimal setup.

## 1. What this quickstart covers

The fastest useful local workflow is:

1. install dependencies
2. confirm the required model bundle exists
3. start the dashboard
4. run the small validation command

This guide does **not** require downloading or reprocessing the full historical dataset if the current dashboard bundle is already present.

## 2. Minimum prerequisites

Required:

1. Python environment with the packages in `requirements.txt`
2. repository checkout
3. one of the supported realtime bundles under `models/`

Preferred dashboard bundle:

```text
models/delay_predictor_v4_score_best_online_safe_bundle.joblib
```

Fallback bundles:

```text
models/delay_predictor_v4_best_online_safe_bundle.joblib
models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt
```

## 3. Install dependencies

From the repository root:

```powershell
pip install -r requirements.txt
```

If your environment uses a specific interpreter, replace `pip` with:

```powershell
python -m pip install -r requirements.txt
```

## 4. Start the dashboard

From the repository root:

```powershell
.\tools\start_dashboard.ps1
```

Then open:

```text
http://127.0.0.1:8000/
```

## 5. Run the small validation command

From the repository root:

```powershell
.\tools\run_april_checkin_validation.ps1
```

This is intended as a lightweight local verification step for the current April deliverables.

## 6. What to look at first

If you only have a few minutes, check these in order:

1. dashboard home page at `http://127.0.0.1:8000/`
2. [APRIL_CHECKIN_TECHNICAL_REPORT.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\docs\APRIL_CHECKIN_TECHNICAL_REPORT.md)
3. `reports/figures/mbta_realtime_model_gap_story.png`
4. `reports/figures/v4_model_sweep.png`
5. `reports/MBTA_REALTIME_OFFICIAL_VS_MODEL.md`

## 7. Common issues

### Missing bundle

If the dashboard does not start because no bundle is found, check whether one of the files in `models/` exists:

```text
delay_predictor_v4_score_best_online_safe_bundle.joblib
delay_predictor_v4_best_online_safe_bundle.joblib
delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt
```

### Missing Python package

If startup fails with an import error, install dependencies again:

```powershell
python -m pip install -r requirements.txt
```

### Dashboard loads but live results are unavailable

That usually means MBTA live data is temporarily unavailable, rate-limited, or the selected route-stop pair has no current upcoming prediction rows.

## 8. Full-data work is optional

For the April check-in, the dashboard bundle and reports are the main deliverables. Rebuilding all training artifacts or rerunning full-history processing is optional unless you are actively changing the modeling pipeline.
