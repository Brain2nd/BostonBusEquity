# Boston Bus Equity Implementation Summary (English)

## 1. Overall Status

Based on your requested scope, the requested work has been completed locally.

### Completed

1. The local repository was synced against your specified remote:
   - `git@github.com:lvzhuojun/BostonBusEquity.git`
2. A new branch using your naming convention was created:
   - `ZhuojunLyu/realtime-inference-baseline`
3. The work was committed locally using a standard Git workflow:
   - commit: `38ea99d`
   - message: `Add realtime inference pipeline and delay baselines`
4. A local realtime inference pipeline was added to the project.
5. Baseline delay evaluation was added.
6. Tests, dependency declarations, and README documentation were updated.
7. Local validation was performed in your Conda environment.

### Not done intentionally

1. Nothing was pushed to any remote repository.
2. No remote PR was created.
3. No frontend UI was built for the realtime service.
4. No continuously running MBTA polling service was added.

### Known environment issue

1. A few `pytest-cache-files-*` temporary directories remain in the workspace.
2. They are not project code and were not committed.
3. They currently cause `git status` permission warnings, but they do not affect the actual implementation.

## 2. Mapping to Your Original Requests

Your two core development requests were:

1. Add realtime inference functionality.
2. Add baseline delay testing.

Both have been implemented.

### Realtime inference

The repository now includes a local HTTP inference path with two supported modes:

1. Direct prediction from provided request fields:
   - `route_id`
   - `stop_id`
   - `scheduled_time`
   - `scheduled_headway`
   - `direction_id`
2. MBTA-backed prediction where the service first queries MBTA V3 API data and then converts that live data into model inputs.

The online inference path uses the `V2` MLP model instead of `V3`.

Reason:

1. `V2` uses causal features that can be constructed online from a single stop event.
2. `V3` depends on wavelet and lag-style research features that are better suited for offline experiments than stateless online inference.

### Baseline delay testing

A dedicated baseline evaluation script was added for the temporal split test period (`2025+`).

The following baselines are included:

1. `zero_delay`
2. `global_mean`
3. `route_mean`
4. `stop_mean`
5. `hour_mean`
6. `route_hour_mean`

## 3. Main Files Added or Updated

### Added: realtime inference

1. [src/inference/__init__.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\__init__.py)
2. [src/inference/build_bundle.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\build_bundle.py)
3. [src/inference/bundle_utils.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\bundle_utils.py)
4. [src/inference/runtime.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\runtime.py)
5. [src/inference/api.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\api.py)
6. [src/inference/serve.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\serve.py)
7. [src/inference/mbta_realtime.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\mbta_realtime.py)

### Added: model sharing and baseline evaluation

1. [src/models/v2_mlp.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\models\v2_mlp.py)
2. [src/models/evaluate_delay_baselines.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\models\evaluate_delay_baselines.py)

### Updated: existing training and docs

1. [src/models/train_delay_predictor_v2.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\models\train_delay_predictor_v2.py)
2. [README.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\README.md)
3. [requirements.txt](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\requirements.txt)
4. [pytest.ini](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\pytest.ini)

### Added: tests

1. [tests/test_inference_runtime.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\tests\test_inference_runtime.py)
2. [tests/test_inference_api.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\tests\test_inference_api.py)

## 4. Design Decisions

## 4.1 Why V2 was selected for realtime inference

The `V2` model uses:

1. temporal features
2. route/stop/direction encodings
3. scheduled headway
4. training-period historical aggregates

This makes it suitable for online inference from a single scheduled stop event without maintaining a server-side historical window.

By contrast, `V3` uses wavelet, lag, and rolling features that are much better suited for offline research workflows.

## 4.2 Why a realtime bundle was introduced

The original training workflow kept essential online inference artifacts only inside the training process.

Those artifacts include:

1. encoding mappings
2. scaler parameters
3. route/stop/hour statistics
4. model config
5. checkpoint weights

To make online inference portable and deterministic, these were packaged into a single `.pt` bundle file.

That bundle includes:

1. `model_state_dict`
2. `model_config`
3. `feature_columns`
4. `scaler_X`
5. `scaler_y`
6. route / stop / direction mappings
7. route / stop / hour / route_hour statistics
8. global defaults such as `global_mean`, `global_std`, and `scheduled_headway_median`

## 4.3 How MBTA live data was integrated

The implementation does not run a continuous polling daemon.

Instead, the service performs an on-demand MBTA V3 lookup:

1. query the `schedules` endpoint for upcoming route/stop schedule records
2. query the `predictions` endpoint for MBTA live predicted times
3. extract the next scheduled departure
4. estimate scheduled headway from adjacent schedule records
5. transform the result into project model inputs
6. return both this project’s prediction and the MBTA live delay when available

## 5. Public Interfaces

## 5.1 Build the bundle

Command:

```bash
python -m src.inference.build_bundle
```

Default output:

```text
models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt
```

## 5.2 Start the local API

Command:

```bash
python -m src.inference.serve --bundle models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt --host 127.0.0.1 --port 8000
```

## 5.3 Endpoints

### `GET /health`

Returns:

1. load status
2. model name
3. experiment
4. feature version

### `POST /predict`

Request fields:

1. `route_id`
2. `stop_id`
3. `scheduled_time`
4. `scheduled_headway` optional
5. `direction_id` optional

Response fields:

1. `predicted_delay_minutes`
2. `model`
3. `experiment`
4. `used_defaults`

### `POST /predict/mbta`

Request fields:

1. `route_id`
2. `stop_id`
3. `direction_id` optional
4. `api_key` optional

Additional response fields:

1. `source`
2. `schedule_id`
3. `scheduled_time`
4. `scheduled_headway`
5. `mbta_prediction_departure_time`
6. `mbta_prediction_delay_minutes`

## 6. Baseline Evaluation

Command:

```bash
python -m src.models.evaluate_delay_baselines
```

Output file:

```text
reports/delay_prediction_baselines_temporal.csv
```

The baseline evaluation uses the same temporal split as the model experiments:

1. train: `< 2025`
2. test: `>= 2025`

This keeps baseline and model evaluation directly comparable.

## 7. Local Validation

The following validation was completed locally:

### Completed validation

1. All new modules import successfully.
2. A small-sample realtime bundle smoke test succeeded.
3. Runtime tests passed.
4. API tests passed.
5. Full test run passed:
   - `3 passed`
6. A small-sample baseline smoke test produced RMSE comparisons successfully.

### Environment used

Validation environment:

```text
D:\Anaconda3\envs\506-final
```

Installed dependencies:

1. `torch`
2. `scikit-learn`
3. `fastapi`
4. `uvicorn`
5. `pywavelets`
6. `pytest`
7. `pyarrow`

## 8. Final Answer to “Was everything done?”

If the question is limited to the concrete development items you explicitly asked for, then yes:

1. Realtime inference support: done
2. Baseline delay testing: done
3. Local branch and local commit workflow: done

So the direct answer is:

**Yes, the core implementation work you asked for has been completed locally.**

What was intentionally not done:

1. no push
2. no remote PR
3. no merge into `main`

## 9. Reasonable Next Steps

If you want to continue from here, the next logical options are:

1. add a latency benchmark script for the realtime API
2. add richer API usage examples
3. regenerate baseline outputs using full local processed data
4. clean the Windows permission-blocked `pytest-cache-files-*` directories
5. prepare a PR description without pushing yet
