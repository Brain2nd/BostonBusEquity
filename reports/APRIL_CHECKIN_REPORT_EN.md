# Boston Bus Equity April Check-In Report

## 1. Project Description and Motivation

### 1.1 Project topic

This project studies **MBTA bus service equity and delay prediction in Boston**.  
The April stage extends the earlier offline analysis into a more operational direction by adding:

1. a **realtime delay inference pipeline**
2. a **baseline delay evaluation layer**

The goal is not only to analyze historical patterns, but also to test whether the project can support a realistic live prediction workflow using real MBTA data sources.

### 1.2 Why this stage matters

The earlier project already covered questions such as:

1. delay patterns by route, time, and service context
2. service quality differences across routes and neighborhoods
3. equity-related interpretation of bus performance

The April stage pushes the project one step further:

1. can the current repository model be used in a realtime setting?
2. can official MBTA live API data be connected to the model in a defensible way?
3. can the project provide simple baseline results so that model outputs are easier to interpret?

This makes the stage appropriate for the April check-in because it combines data processing, modeling, and system design into one coherent deliverable.

## 2. Project Timeline

### 2.1 Two-month structure

To fit a realistic course timeline, the broader project is organized into four phases:

| Phase | Time | Focus |
|---|---|---|
| Phase 1 | February | historical data access, cleaning, route and service analysis, initial delay modeling |
| Phase 2 | March | iterative modeling experiments and development of V1 / V2 / V3 model families |
| Phase 3 | April | realtime inference, baseline delay testing, and check-in reporting |
| Phase 4 | Before final deadline | larger validation, final report refinement, final presentation preparation |

### 2.2 April stage subtasks

| Subtask | Estimated duration | Current status |
|---|---:|---|
| define a technically valid realtime plan | 0.5 week | completed |
| connect to official MBTA live API | 0.5 week | completed |
| build the local realtime inference path | 1 week | completed |
| add baseline delay evaluation | 0.5 week | completed |
| produce real-data stage results | 1 week | completed with reportable stage results |
| write bilingual April check-in documentation | 0.5 week | completed |

### 2.3 Timeline rationale

The April scope was intentionally kept focused on:

1. one defensible realtime inference path
2. one baseline comparison layer
3. one reportable set of results

This keeps the scope realistic for the current stage instead of trying to rebuild the entire offline pipeline from scratch.

## 3. Project Goals

### 3.1 Goal 1: Build a high-quality realtime inference path

Definition:

1. accept a live or near-live transit record
2. obtain inputs from the official MBTA API
3. produce a local delay prediction from the project model

Measurable criteria:

1. at least one real live example runs end-to-end
2. the system returns `predicted_delay_minutes`
3. the input comes from the real MBTA V3 API rather than a synthetic mock

### 3.2 Goal 2: Add baseline delay testing

Definition:

1. use a clear temporal split
2. implement multiple simple but defensible baselines
3. export quantitative comparison metrics

Measurable criteria:

1. a baseline result file is generated
2. at least five baselines are included
3. the output includes RMSE, MAE, and R2-style metrics

### 3.3 Goal 3: Produce a check-in-ready stage narrative

Definition:

1. explain what was built
2. explain how data, models, and interfaces connect
3. summarize current results and next steps

Measurable criteria:

1. a formal Chinese and English report exists
2. the report includes data, methods, and results
3. the team can defend design decisions during the check-in

### 3.4 Why these goals are specific and measurable

These goals are not vague statements such as "analyze data."  
They specify:

1. which model is used
2. which live API is used
3. which split logic is used
4. which metrics are reported
5. what counts as a successful stage result

That makes them concrete and measurable.

## 4. Data Collection Plan

### 4.1 Historical data source

The historical data source remains the MBTA bus arrival/departure archive under:

```text
data/raw/arrival_departure
```

Confirmed local status:

1. `61` CSV files
2. total size about `15.282 GB`
3. years covered: `2021-2026`

This means the April stage does **not** switch to a different offline training dataset.

### 4.2 Live data source

The realtime input path uses the **official MBTA V3 API**, mainly:

1. `schedules`
2. `predictions`

Their roles are:

1. `schedules` provides upcoming scheduled service records
2. `predictions` provides MBTA's current live prediction records

### 4.3 Data collection method

Historical data collection:

1. read local MBTA historical CSV archives
2. perform offline ingestion and preprocessing

Realtime data collection:

1. API-based collection
2. query MBTA V3 at inference time
3. no long-running background polling service at this stage

This satisfies the proposal rubric requirement to identify data sources and explain how data is collected.

## 5. Data Processing

### 5.1 Main processing steps

To support both historical modeling logic and live inference, this stage performs:

1. parsing `service_date`, `scheduled`, and `actual`
2. computing `delay_minutes`
3. filtering obvious outliers
4. normalizing `route_id`
5. aligning historical direction semantics with live API direction semantics
6. building temporal and statistical features

### 5.2 Important data decisions

#### Decision 1: use V2 for realtime inference, not V3

Reason:

1. `V2` features are causally constructible online
2. `V3` wavelet and lag-heavy features are better suited to offline experimentation
3. `V2` is the more defensible choice for stateless realtime inference

#### Decision 2: keep the existing repository model

The realtime path still uses the existing repository checkpoint:

```text
models/delay_predictor_mlp_v2_lag_features_temporal.pt
```

This means:

1. the April stage did not retrain a new online-only model
2. the main work was to connect the existing model to live inputs

#### Decision 3: build a realtime bundle instead of loading the raw checkpoint directly

The realtime bundle packages:

1. model weights
2. model config
3. feature ordering
4. scaler parameters
5. route, stop, and direction mappings
6. training-period statistics

This allows a single live request to be transformed into the same feature space as the trained model.

#### Decision 4: normalize route IDs

Historical files may use route IDs such as:

1. `01`
2. `1`

The live API usually returns:

1. `1`

Without canonicalization, route mappings fail. This stage explicitly normalizes route IDs.

#### Decision 5: align direction semantics

Historical data uses:

1. `Inbound`
2. `Outbound`

The live API uses:

1. `0`
2. `1`

This stage adds an explicit mapping so live API direction values can be interpreted consistently with the historical training data.

### 5.3 Real-data subset strategy

Because the full archive is large, the April stage does not attempt a full-history rebuild.  
Instead, it uses a **real-data subset strategy**:

1. still uses real historical CSV files
2. focuses on recent years: `2024`, `2025`, and `2026`
3. samples each file for stage-scale bundle construction and validation

The current successful stage bundle metadata is:

1. `years = [2024, 2025, 2026]`
2. `sample_per_file = 600`
3. `train_rows_used = 7200`
4. `test_rows_seen = 7800`

This gives the project:

1. a real historical data base
2. a manageable stage-scale runtime
3. reportable evidence that the pipeline works on real data

## 6. Implementation Deliverables

### 6.1 Realtime inference modules added

Main code files:

1. `src/inference/build_bundle.py`
2. `src/inference/runtime.py`
3. `src/inference/mbta_realtime.py`
4. `src/inference/api.py`
5. `src/inference/serve.py`
6. `src/inference/bundle_utils.py`
7. `src/models/v2_mlp.py`

Main responsibilities:

1. package the existing V2 checkpoint into a deployable realtime bundle
2. transform live MBTA records into model-ready features
3. load model state, feature metadata, and scalers consistently at inference time
4. expose a local HTTP inference interface

### 6.2 Baseline evaluation module added

Main code file:

1. `src/models/evaluate_delay_baselines.py`

Responsibility:

1. run baseline delay evaluation on the stage split and export a result table

### 6.3 Local API endpoints

The local inference service now supports:

1. `GET /health`
2. `POST /predict`
3. `POST /predict/mbta`

Their roles are:

1. `GET /health` verifies that the service and bundle are loaded
2. `POST /predict` accepts a prepared local payload
3. `POST /predict/mbta` fetches MBTA live inputs and returns a model prediction in one path

### 6.4 Tests and validation artifacts

Stage-related validation assets include:

1. `tests/test_inference_runtime.py`
2. `tests/test_inference_api.py`
3. `pytest.ini`
4. `reports/delay_prediction_baselines_stage2_2024_2026.csv`

## 7. Modeling Methods

### 7.1 Modeling methods attempted in the broader project

Across the broader project, multiple delay models have already been explored:

1. V1 baseline ANN models
2. V2 lag-feature MLP
3. V3 wavelet-temporal models

For the April realtime stage, the operational model used is:

1. `V2 MLP`

### 7.2 Why V2 MLP is appropriate for this stage

The goal of this stage is **high-quality realtime inference**, not simply presenting the best offline score.

Reasons for selecting `V2 MLP`:

1. its features are online-constructible
2. it does not require maintaining a long rolling history window in the service
3. it is easier to interpret and debug in a live path
4. it is technically more defensible for a stateless API workflow

### 7.3 Baseline methods added

The following baselines were added:

1. `zero_delay`
2. `global_mean`
3. `route_mean`
4. `stop_mean`
5. `hour_mean`
6. `route_hour_mean`

These baselines help establish:

1. a lower bound for comparison
2. whether the model learns structure beyond naive heuristics
3. a clearer interpretation of model value

## 8. Visualizations and Check-In Material

### 8.1 Visualization 1: baseline error comparison

The baseline result table can be directly turned into a bar chart for the April check-in.

| Baseline | RMSE |
|---|---:|
| `hour_mean` | 6.0679 |
| `global_mean` | 6.0769 |
| `route_mean` | 6.0985 |
| `stop_mean` | 6.1930 |
| `route_hour_mean` | 6.3473 |
| `zero_delay` | 6.8871 |

Why this visualization is useful:

1. it shows that the baseline layer was actually implemented
2. it supports claims about relative baseline strength
3. it grounds modeling discussion in concrete numbers

### 8.2 Visualization 2: realtime inference example table

The live inference example can be shown as a simple result table.

| Route | Stop | Scheduled Time | Headway | MBTA Live Delay | Model Delay |
|---|---|---|---:|---:|---:|
| 1 | 64 | 2026-04-25 11:09 | 10.0 | -6.2167 | -0.3711 |

Why this visualization is useful:

1. it shows that the live interface is connected
2. it shows that the local model can consume real live inputs
3. it demonstrates direct comparability between local model output and MBTA live prediction

## 9. Preliminary Results

### 9.1 Baseline results

The stage baseline output file is:

```text
reports/delay_prediction_baselines_stage2_2024_2026.csv
```

Results:

| Baseline | RMSE | MAE | R2 |
|---|---:|---:|---:|
| `hour_mean` | 6.0679 | 4.4257 | -0.0258 |
| `global_mean` | 6.0769 | 4.4329 | -0.0288 |
| `route_mean` | 6.0985 | 4.4296 | -0.0361 |
| `stop_mean` | 6.1930 | 4.4530 | -0.0685 |
| `route_hour_mean` | 6.3473 | 4.5926 | -0.1224 |
| `zero_delay` | 6.8871 | 4.6646 | -0.3214 |

Interpretation:

1. `hour_mean` is the strongest baseline on the current stage subset
2. `zero_delay` is the weakest baseline
3. more granular aggregation does not automatically improve performance on the current subset

### 9.2 Realtime inference result

At least one real live example has already run end-to-end.

Input:

1. `route_id = 1`
2. `stop_id = 64`
3. `direction_id = 1`

Retrieved from MBTA V3:

1. `scheduled_time = 2026-04-25T11:09:00-04:00`
2. `scheduled_headway = 10.0`
3. `MBTA live predicted delay = -6.2167 minutes`

Local project model output:

1. `predicted_delay_minutes = -0.3711 minutes`

Interpretation:

1. the official live API path works
2. live inputs can be transformed into model features
3. the local model can produce a realtime prediction
4. the project can compare local model output against MBTA live prediction

This does **not** yet prove that the local model outperforms MBTA's live prediction.  
It proves that the realtime inference chain is functioning on real inputs.

## 10. Results and Interpretation

### 10.1 What the current results show

This stage successfully moves the project from:

1. offline modeling only

to:

2. a working live inference pipeline grounded in real MBTA inputs

The project can now claim:

1. realtime inputs are not mocked
2. the official live API is integrated
3. the local model can produce live predictions
4. baseline comparisons are available

### 10.2 What the current results do not yet show

The current stage does **not** yet prove:

1. that the local model outperforms MBTA live prediction
2. that the system is production-grade
3. that the current subset is equivalent to a full-history result

### 10.3 Why the stage is still strong enough for the April rubric

The April rubric emphasizes:

1. relevant visualizations
2. clear data sources and processing choices
3. at least one modeling method tested for performance
4. understandable results and interpretation

This stage satisfies those criteria because it includes:

1. real historical data and a real official API
2. clearly explained data alignment decisions
3. tested modeling plus baseline evaluation
4. actual numeric outputs and a live example

## 11. Next Steps

The next logical steps are:

1. run multiple live examples across more routes and stops
2. validate direction mapping more broadly
3. scale the realtime bundle and baseline evaluation to larger real-data slices
4. add latency benchmarking for the realtime API
5. compress the results into slide-ready visuals and speaking notes

## 12. Core Takeaways for the April Presentation

The clearest points to emphasize in the presentation are:

1. this stage focused on realtime inference and baseline evaluation
2. the historical data source remains the original MBTA arrival/departure archive
3. the live input source is the official MBTA V3 API
4. the realtime model still uses the existing repository `V2 MLP` checkpoint
5. this stage did not retrain a new model; it connected the existing model to live inputs
6. baseline evaluation now produces real numeric results
7. realtime inference has already produced a real live example result
8. the system can now compare local model predictions against MBTA live predictions

## 13. Summary

This April stage has already achieved the following:

1. identified a technically valid online model path
2. connected the official live API
3. added baseline delay evaluation
4. produced real stage results
5. created a complete narrative for the April check-in

By the April check-in standard, the project now has a strong and defensible stage report foundation.
