# Boston Bus Equity Stage 2 Progress Report

## 1. Stage Goal

This stage is focused on two concrete deliverables for the next check-in:

1. a high-quality realtime inference capability
2. baseline delay evaluation with actual results

This report focuses only on those two goals.

## 2. Did the work run on real data?

Yes, but with an important qualification:

**the current stage result is based on real historical MBTA data plus the official live MBTA API, using a real-data subset strategy for faster stage execution.**

### Confirmed real historical data

The local repository already contains real MBTA bus arrival/departure history:

1. location: `data/raw/arrival_departure`
2. file count: `61` CSV files
3. total size: about `15.282 GB`
4. year coverage: `2021-2026`

These are real source files, not synthetic placeholders.

### Confirmed realtime data source

The live input path uses the **official MBTA V3 API**, primarily:

1. `schedules`
2. `predictions`

This means the realtime path is grounded in the actual external data source the project needs.

## 3. Core Stage 2 Deliverables

## 3.1 Realtime inference pipeline

The project now supports:

1. realtime bundle generation from local historical data
2. loading the existing `V2` MLP checkpoint
3. converting a live stop event into model features
4. local API-based inference
5. direct request-based prediction
6. MBTA-backed live prediction via MBTA V3 lookup plus local model inference

### Why V2 was selected instead of V3

The online path uses `V2 MLP`, not `V3`.

Reason:

1. `V2` uses causal features that can be constructed online from a single event
2. `V3` depends on wavelet, lag, and rolling-window features that are more appropriate for offline research
3. using `V3` as a stateless online model would be methodologically weaker

So for stage quality and technical defensibility, `V2` is the right online choice.

## 3.2 Baseline delay evaluation

The project now includes a baseline evaluation path using the same temporal split logic as the model work:

1. training period: `< 2025`
2. test period: `>= 2025`

Added baselines:

1. `zero_delay`
2. `global_mean`
3. `route_mean`
4. `stop_mean`
5. `hour_mean`
6. `route_hour_mean`

## 4. Real-data subset strategy

Because the raw arrival/departure archive is large, this stage uses a **real-data subset strategy** rather than a full 15GB reprocessing run.

The stage 2 realtime bundle that successfully ran used:

1. `years = [2024, 2025, 2026]`
2. `sample_per_file = 600`
3. `train_rows_used = 7200`
4. `test_rows_seen = 7800`
5. `source_parquet_exists = False`

This means:

1. the stage result is based on real data
2. it is not yet a full-history production-scale run
3. it is appropriate as a credible engineering result for the current monthly check-in

## 5. Baseline results

Stage 2 baseline results file:

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

### Interpretation

1. `hour_mean` is the strongest simple baseline on this stage subset
2. `zero_delay` is clearly the weakest baseline
3. finer-grained `route_hour_mean` did not outperform simple `hour_mean` on this subset

For check-in purposes, this is enough to show that the baseline layer was added and that it produces real quantitative outputs.

## 6. Realtime inference result

## 6.1 Official live API validation

The following has been verified:

1. the MBTA `schedules` endpoint can return upcoming scheduled service
2. the MBTA `predictions` endpoint can return current live predicted times
3. both can be combined to produce a local model inference input

### A real live example

Using:

1. `route_id = 1`
2. `stop_id = 64`
3. `direction_id = 1`

the system retrieved the following real live record:

1. `scheduled_time = 2026-04-25T11:09:00-04:00`
2. `scheduled_headway = 10.0`
3. `mbta_prediction_delay_minutes = -6.2167`

Using the locally built stage 2 realtime bundle, the project model returned:

1. `predicted_delay_minutes = -0.3711`

### Interpretation

This confirms three important points:

1. the official MBTA live data path is working
2. the local model can consume that live input and produce a prediction
3. the project can now compare:
   - the project model prediction
   - the MBTA live prediction

That is one of the most valuable stage outcomes for the second check-in.

## 7. Important issues found and fixed

To make this stage credible, several non-trivial data alignment issues had to be fixed:

### 7.1 Route ID normalization

Historical CSVs sometimes use route IDs such as `01`, while the live API uses `1`.

Without normalization:

1. historical mappings and live inputs fail to match
2. realtime inference produces false `Unknown route_id` failures

This was fixed by canonicalizing route IDs.

### 7.2 Direction ID mismatch

Historical training data uses:

1. `Inbound`
2. `Outbound`

The live MBTA API uses:

1. `0`
2. `1`

This stage added an explicit mapping layer so live API direction values can be converted into training-time direction semantics.

### 7.3 Schedule selection logic

The initial logic could return earlier same-day schedule entries instead of upcoming service.

This was fixed by:

1. adding `filter[min_time]` to the schedule query
2. using only upcoming scheduled records

## 8. Current quality assessment

For a stage check-in, this work is now **reportable and credible**, but it is not yet a final production-grade system.

### Strong enough for the current stage

1. real historical MBTA data is being used
2. the official live MBTA API is connected
3. baseline evaluation produced real numeric results
4. live inference produced a real sample output
5. major data-alignment issues have already been identified and corrected

### Not yet final-grade

1. results currently rely on a real-data subset rather than a full-history run
2. the direction mapping still needs broader validation
3. the realtime path has not yet been evaluated across many live examples

## 9. Recommended message for the second check-in

For the next project update, the strongest points to emphasize are:

1. the work has moved beyond design into actual realtime integration
2. the realtime system uses the official MBTA V3 API
3. the online model choice (`V2`) was made for causal validity, not convenience
4. baseline delay evaluation has been added and produces real results
5. the project can now compare project-model predictions against MBTA live predictions

## 10. Recommended next steps

The next best improvements are:

1. run multiple live examples and summarize them in a table
2. validate the direction mapping more systematically
3. scale the realtime bundle and baseline evaluation to larger real-data slices
4. add realtime latency benchmarking
5. compress this report into a slide-ready summary for the check-in

---

**Bottom line:**

This stage is no longer just a code scaffold.  
It now includes real historical MBTA data usage, real MBTA live API usage, baseline results, and a working live inference example that can support the second project report.
