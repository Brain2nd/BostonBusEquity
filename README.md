# Boston Bus Equity

A data analysis project examining MBTA bus service performance and its impact on Boston residents, with a focus on equity across different communities.

## April Check-In Deliverables

- Technical check-in report: [`docs/APRIL_CHECKIN_TECHNICAL_REPORT.md`](docs/APRIL_CHECKIN_TECHNICAL_REPORT.md)
- GitHub submission guide: [`docs/GITHUB_SUBMISSION_GUIDE.md`](docs/GITHUB_SUBMISSION_GUIDE.md)
- Realtime dashboard command:

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.inference.serve `
  --bundle models\delay_predictor_v4_score_best_online_safe_bundle.joblib `
  --host 0.0.0.0 `
  --port 8000
```

Then open `http://127.0.0.1:8000/`.

## Project Overview

Public transport plays a vital role in the quality of life for Massachusetts and Boston residents in terms of economic development, environment, and equity. This project aims to better understand the impact of bus performance on Boston residents by using MBTA bus data to examine service performance trends by geography.

The MBTA serves over 1 million people daily, with an estimated added economic value of $11.5 billion per year for the greater Boston area.

## Project Focus and Extensions

**Our primary focus is on the extended research (Q8: Delay Prediction)**, which presents significant technical challenges beyond the base project requirements. The base analysis (Q1-Q7) serves as the foundation, while the majority of our semester-long effort is dedicated to advanced machine learning experiments.

### Why We Focus on Extended Research

| Aspect | Base Project (Q1-Q7) | Extended Research (Q8) |
|--------|----------------------|------------------------|
| **Complexity** | Descriptive statistics, visualization | Advanced ML, signal processing, neuromorphic computing |
| **Technical Challenge** | Data aggregation and correlation | Feature engineering (FFT, Wavelet), SNN implementation |
| **Innovation** | Standard data analysis | Novel application of NeuronSpark SNN to transportation data |
| **Time Investment** | ~1 week | ~3 months (semester-long) |

### Data Scope Extension
- **Original**: 2018-2024 MBTA bus arrival/departure data
- **Actual Available**: 2020-2024 (2018-2019 not available on MBTA Open Data Portal)
- **Extended**: Added 2025-2026 data as validation dataset (strict temporal split to prevent data leakage)

### Methodology Enhancement
- **Training Set**: 2020-2024 data (121M records) for model training
- **Test Set**: 2025-2026 data (28M records) for validation
- **Feature Engineering**: Wavelet decomposition, FFT spectral analysis, lag features, rolling statistics
- **Model Architectures**: MLP, LSTM, GRU, Transformer, Spiking Neural Network (NeuronSpark)

### Research Challenges We Address
1. **Temporal Feature Extraction**: How to extract meaningful patterns from delay time series using signal processing techniques?
2. **Neuromorphic Computing**: Can Spiking Neural Networks achieve competitive performance on time series regression tasks?
3. **Model Comparison**: How do different architectures (RNN vs Attention vs SNN) compare at similar parameter scales?
4. **Ablation Study**: Which feature extraction methods contribute most to prediction accuracy?

## Project Goals

### Base Goals (Q1-Q7)
| Goal | Metric | Target | Achieved |
|------|--------|--------|----------|
| Analyze ridership changes pre vs post pandemic | % change in ridership | Quantify change | ✅ -32.8% |
| Measure end-to-end travel times per route | Average minutes | Calculate for all routes | ✅ 28.4 min avg |
| Calculate average wait times | Minutes (on-time vs delayed) | Compare scenarios | ✅ 5 vs 12-15 min |
| Determine citywide average delay | Minutes | Single metric | ✅ 7.51 min |
| Compare target routes vs other routes | Delay difference % | Identify disparity | ✅ +41% higher |
| Identify service level disparities | On-time performance % | Route comparison | ✅ 25.8% vs 32.3% |
| Assess demographic equity impact | Correlation coefficient | Statistical significance | ✅ p=0.96 (no bias) |

### Extended Goals (Q8: Delay Prediction)
| Goal | Metric | Target | Achieved |
|------|--------|--------|----------|
| Build baseline delay prediction model | R² score | > 0 | ✅ R²=0.9846 (GRU) |
| Engineer time series features | RMSE reduction | > 50% vs baseline | ✅ 88% reduction |
| Implement neuromorphic SNN model | R² score | Competitive with GRU | ✅ R²=0.9897 |
| Achieve state-of-the-art performance | R² score | > 0.99 | ✅ R²=0.9942 (Transformer) |

## Key Research Questions

### Base Questions
1. What is the ridership per bus route? How has this changed from pre-pandemic to post-pandemic?
2. What are the end-to-end travel times for each bus route in the city?
3. On average, how long does an individual have to wait for a bus (on time vs. delayed)?
4. What is the average delay time of all routes across the entire city?
5. What is the average delay time of target bus routes (22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14)?
6. Are there disparities in the service levels of different routes?
7. If there are service level disparities, are there differences in the characteristics of the people most impacted (e.g., race, ethnicity, age, income)?

### Extended Question
8. Can we accurately predict bus delays using machine learning with advanced feature engineering?

## Project Structure

```
BostonBusEquity/
├── data/
│   ├── raw/                 # Original, immutable data
│   ├── processed/           # Cleaned, transformed data
│   └── external/            # Data from third-party sources
├── notebooks/               # Jupyter notebooks for analysis
├── src/
│   ├── data/               # Data processing scripts
│   ├── analysis/           # Analysis scripts
│   └── visualization/      # Visualization scripts
├── reports/
│   └── figures/            # Generated graphics and figures
├── docs/                   # Documentation
└── README.md
```

## Data Sources and Collection Methods

### Training Data (2020-2024)
| Dataset | Source | Collection Method | Size |
|---------|--------|-------------------|------|
| Bus Arrival/Departure Times | [MBTA Open Data Portal](https://mbta-massdot.opendata.arcgis.com/search?tags=bus) | Automated download script (`src/data/download_data.py`) | ~18 GB, 161M records |
| Bus Ridership by Trip/Season | [MBTA Open Data Portal](https://mbta-massdot.opendata.arcgis.com/) | Automated download script | ~850 MB, 1.8M records |
| Passenger Survey | [MBTA 2022-2024 Survey](https://gis.data.mass.gov/datasets/MassDOT::mbta-2024-system-wide-passenger-survey/about) | Manual CSV download | ~50 MB |

### Validation Data (2025-2026)
| Dataset | Source | Collection Method | Size |
|---------|--------|-------------------|------|
| Reliability Data 2025 | [MBTA 2025](https://mbta-massdot.opendata.arcgis.com/datasets/924df13d845f4907bb6a6c3ed380d57a/about) | Automated download script | ~2 GB |
| Reliability Data 2026 | [MBTA 2026](https://mbta-massdot.opendata.arcgis.com/datasets/9d8a8cad277545c984c1b25ed10b7d3c/about) | Automated download script | Partial year |

### Demographic Data
| Dataset | Source | Collection Method |
|---------|--------|-------------------|
| Census Data | [2020 Census for Boston](https://data.boston.gov/dataset/2020-census-for-boston) | API download |
| ACS Data | [ACS 2020-2024 5-Year Estimates](https://www.census.gov/programs-surveys/acs/data.html) | Manual download |
| Boston Demographics | [Boston at a Glance 2024](https://www.bostonplans.org/documents/research/population-and-demographics/2024/boston-at-a-glance-2024) | Manual download |

### Data Collection Implementation
```bash
# Automated data download (supports resume)
python src/data/download_data.py

# Check download status
python src/data/download_data.py --status
```

### Additional Sources
- [MBTA V3 API](https://www.mbta.com/developers/v3-api)
- [MBTA Performance Data](https://www.mbta.com/developers)

## Tech Stack

- **Python**: Primary programming language
- **Google Colab**: Collaborative notebook environment
- **Power BI**: Data visualization and dashboards
- **ArcGIS**: Geographic analysis and mapping

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Brain2nd/BostonBusEquity.git
cd BostonBusEquity
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download data (first time only):
```bash
python src/data/download_data.py
```

> **Important Notes for Team Members:**
> - Data files (~20GB total) are NOT stored in Git - each person must download locally
> - `git pull` will NOT overwrite your downloaded data (data folder is in .gitignore)
> - If download is interrupted, just run the script again - it supports resume
> - Check download status anytime: `python src/data/download_data.py --status`

### Manual Data Download

Some datasets require manual download due to authentication or access restrictions:

#### 1. MBTA Passenger Survey Data (Optional, for Q7)

**URL:** https://mbta-massdot.opendata.arcgis.com/datasets/mbta-2024-system-wide-passenger-survey

**Steps:**
1. Open the URL above in your browser
2. Click the **"Download"** button (top right corner)
3. Select **"Download as CSV"** from the dropdown menu
4. Save the downloaded file to `data/raw/survey/`
5. Rename to `MBTA_2024_Passenger_Survey.csv` (optional)

#### 2. Boston Neighborhood Demographics (Optional, for Q7)

**URL:** https://data.boston.gov/dataset/neighborhood-demographics

**Steps:**
1. Open the URL above in your browser
2. Find the CSV resource and click **"Explore"**
3. Click the **"Download"** button
4. Save the downloaded file to `data/raw/census/`

#### Data Download Status

After running the download script, you should have:

| Dataset | Files | Size | Auto-Download |
|---------|-------|------|---------------|
| Arrival/Departure (2020-2026) | 65 | ~18 GB | ✅ Yes |
| Bus Ridership | 10 | ~850 MB | ✅ Yes |
| Census Neighborhood | 1 | ~22 KB | ✅ Yes |
| Passenger Survey | 1 | ~50 MB | ❌ Manual |
| Demographics (ACS) | 1 | ~100 KB | ❌ Manual |

### Running Analysis

```bash
# Full analysis on training data (2020-2024)
python src/run_analysis.py

# Quick test with 2024 data only
python src/run_analysis.py --quick

# Validation analysis (2025-2026)
python src/run_analysis.py --validate
```

## Local Realtime Inference

The repository now includes a minimal local HTTP inference path for realtime delay prediction.
This uses the **V2 MLP checkpoint** (`v2_lag_features_temporal`) because its features are causal
and can be reconstructed online without future leakage.

### Why Not V3 For Realtime?

The V3 wavelet experiment remains an **offline research model**. Its wavelet features are computed
over full route-stop sequences, so the current V3 checkpoint is not exposed as the realtime API model.

### Build the Realtime Bundle

The realtime service expects a single bundle file that combines:
- the V2 MLP checkpoint weights
- feature metadata
- scaler parameters
- route/stop/direction encoders
- route/stop/hour historical statistics

If your environment does not expose `python` on `PATH`, use an explicit interpreter path.

```bash
# Example with a specific interpreter
C:\Users\yaobc\anaconda3\python.exe -m src.inference.build_bundle
```

Optional arguments:

```bash
C:\Users\yaobc\anaconda3\python.exe -m src.inference.build_bundle \
  --processed-dir data/processed \
  --checkpoint models/delay_predictor_mlp_v2_lag_features_temporal.pt \
  --output models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt
```

The bundle builder prefers `data/processed/arrival_departure.parquet` and falls back to processed CSV files.
If 2025/2026 processed data is missing, the builder still works and emits a warning that vocabularies were
built only from the available years.

### Run the Local API

```bash
C:\Users\yaobc\anaconda3\python.exe -m src.inference.serve \
  --bundle models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt \
  --host 127.0.0.1 \
  --port 8000
```

### API Endpoints

`GET /health`

- Returns whether the bundle is loaded plus model and feature metadata.

`POST /predict`

- Required JSON fields: `route_id`, `stop_id`, `scheduled_time`
- Optional JSON fields: `scheduled_headway`, `direction_id`
- `scheduled_time` accepts ISO8601. If no timezone is provided, it is interpreted as `America/New_York`.
- Missing `scheduled_headway` falls back to the training median.
- Missing `direction_id` falls back to `"Unknown"`.
- Unknown `route_id` or `stop_id` returns HTTP `422`.

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' `
  -Body '{
    "route_id": "1",
    "stop_id": "70001",
    "scheduled_time": "2026-04-24T08:15:00",
    "scheduled_headway": 12
  }'
```

### Benchmark And Verification

You can generate a latency baseline figure and markdown report with:

```bash
C:\Users\yaobc\anaconda3\python.exe -m src.inference.benchmark_latency
```

Outputs:
- `reports/figures/realtime_inference_latency_baseline.png`
- `reports/REALTIME_INFERENCE_BASELINE.md`

If `fastapi` is not installed in the active environment, the benchmark still verifies
the `runtime.predict` path and notes that the API benchmark was skipped.

### Prediction Trace Figure

You can also generate a "predicted delay over time" figure for the realtime inference path:

```bash
C:\Users\yaobc\anaconda3\python.exe -m src.inference.plot_predictions_over_time
```

Outputs:
- `reports/figures/realtime_inference_prediction_over_time.png`
- `reports/REALTIME_INFERENCE_PREDICTION_OVER_TIME.md`
- `reports/realtime_inference_prediction_over_time.csv`

If `models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt` is present,
the script uses that bundle directly. Otherwise it falls back to a synthetic demo bundle
and labels the figure/report accordingly.

You can override the default plotted series with repeated `--series` flags:

```bash
C:\Users\yaobc\anaconda3\python.exe -m src.inference.plot_predictions_over_time ^
  --series "Route 22 Downtown,22,70001,0,10" ^
  --series "Route 28 Roxbury,28,70002,0,8"
```

### MBTA V3 Realtime Polling And Comparison

The MBTA V3 API exposes official realtime arrival predictions. This repository now includes
an integration script that can poll those predictions and optionally compare them against the
local V2 delay model if a realtime bundle is available.

Official predictions only:

```bash
C:\Users\yaobc\anaconda3\python.exe -m src.inference.plot_mbta_realtime_comparison ^
  --route-id 111 ^
  --stop-id 5547 ^
  --direction-id 1 ^
  --poll-count 6 ^
  --poll-interval-seconds 30
```

Official predictions vs local V2 model:

```bash
C:\Users\yaobc\anaconda3\python.exe -m src.inference.plot_mbta_realtime_comparison ^
  --route-id 111 ^
  --stop-id 5547 ^
  --direction-id 1 ^
  --bundle models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt ^
  --poll-count 6 ^
  --poll-interval-seconds 30
```

Outputs:
- `reports/figures/mbta_realtime_official_vs_model.png`
- `reports/MBTA_REALTIME_OFFICIAL_VS_MODEL.md`
- `reports/mbta_realtime_official_vs_model.csv`

Notes:
- The script reads `MBTA_API_KEY` from the environment if available, but MBTA allows limited experimentation without a key.
- The left panel shows the latest snapshot by scheduled trip time.
- The right panel shows the earliest upcoming trip tracked across polling snapshots.
- If no bundle is provided, the script runs in `official_only` mode and skips local model comparison.
- If MBTA numeric direction ids are not present in the bundle encoder, comparison mode falls back to the bundle's `Unknown` direction default.

## V4 Strong Baseline And V5 Live Residual Path

The live comparison figure should not be read as true model accuracy by itself:
MBTA official predictions are another model output, not ground truth. The V4
baseline trains against the real label `actual - scheduled`; MBTA official
predictions are held out as a comparison baseline and later as the input to a
V5 residual-correction model.

### Train V4 Tree Bundle

V4 uses causal V2 features plus trip/history/headway/live-vehicle fields. It
uses `2024` for training, `2025` for validation, and `2026` for testing. If
LightGBM is installed, `--model-kind auto` uses LightGBM; otherwise it falls
back to sklearn's histogram gradient boosting model.

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.models.train_delay_predictor_v4 `
  --processed-path data/processed/arrival_departure.parquet `
  --output-bundle models/delay_predictor_v4_tree_realtime_bundle.joblib `
  --model-kind auto
```

By default this command runs a smoke-sized training sample (`2024=20k`,
`2025=5k`, `2026=5k`) so it finishes quickly on a laptop. Use
`--max-train-rows 0 --max-validation-rows 0 --max-test-rows 0` for a full
parquet run.

The default V4 runtime profile is `online_safe`: features that require true
previous-stop delays are disabled unless the service has a live trip-history
cache. This prevents an optimistic offline score from being deployed as a
stateless realtime model.

Outputs:
- `models/delay_predictor_v4_tree_realtime_bundle.joblib`
- `reports/DELAY_PREDICTION_V4_OPTIMIZATION_REPORT.md`
- `reports/delay_prediction_metrics_v4.csv`
- `reports/v4_test_predictions.csv`
- `reports/figures/v4_model_comparison.png`
- `reports/figures/v4_feature_importance.png`
- `reports/figures/official_vs_v4_vs_actual.png`
- `reports/figures/v4_optimization_diagnostics.png`

### Sweep V4 Model Families

For optimization, run a model-family and feature-profile sweep instead of
trusting one tree model. The final retrain option keeps `2026` as the holdout,
but refits deployable candidates on all prior labels (`2024+2025`):

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.models.sweep_delay_predictor_v4 `
  --max-train-rows 50000 `
  --max-validation-rows 50000 `
  --max-test-rows 10000 `
  --candidates dummy,ridge,hist_gradient_boosting,hist_gradient_boosting_l2,extra_trees `
  --feature-profiles all,no_ids,v2_core,stats_time `
  --include-validation-in-final
```

Current best sweep result after installing LightGBM/CatBoost/XGBoost:
`lightgbm` with `v2_core` features, final `2026` MAE `3.85` minutes versus the
comparable V2 sample MAE `4.18` minutes. The improvement is useful but modest; the larger gap remains live
trip-history state and V5 matched official-residual labels.

For realtime presentation, the project also scores each candidate with a
deployability rubric: 40% MAE, 15% year-to-year stability, 15% online readiness,
20% early/negative-delay behavior, and 10% training cost. Under that score,
`lightgbm_q35` with `v2_core` features is the preferred dashboard model because
it raises early-delay F1 while keeping final `2026` MAE below the V2 sample
baseline.

Outputs:
- `models/delay_predictor_v4_best_online_safe_bundle.joblib`
- `models/delay_predictor_v4_score_best_online_safe_bundle.joblib`
- `reports/delay_prediction_v4_model_sweep.csv`
- `reports/delay_prediction_v4_model_scores.csv`
- `reports/delay_prediction_v4_model_sweep_summary.csv`
- `reports/V4_MODEL_SWEEP_REPORT.md`
- `reports/MODEL_SCORING_GUIDE.md`
- `reports/figures/v4_model_sweep.png`
- `reports/figures/v4_model_deployability_scores.png`

### User-Facing Realtime Dashboard

The realtime FastAPI service also serves an English presentation dashboard at
`/`. It includes project KPIs, existing project figures, interactive model
metrics, manual local prediction, MBTA official-vs-local live comparison, and
plain-language descriptions of the data processing and modeling code.

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.inference.serve `
  --bundle models/delay_predictor_v4_score_best_online_safe_bundle.joblib `
  --host 127.0.0.1 `
  --port 8000
```

Open `http://127.0.0.1:8000/`.

Presentation figures that summarize the optimization decision:

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.visualization.create_v4_optimization_figures
```

Outputs:
- `reports/figures/v4_optimization_story.png`
- `reports/figures/mbta_realtime_model_gap_story.png`
- `reports/V4_OPTIMIZATION_RESEARCH_NOTES.md`

The existing HTTP API can load either the old V2 `.pt` bundle or the V4
`.joblib` bundle:

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.inference.serve `
  --bundle models/delay_predictor_v4_tree_realtime_bundle.joblib `
  --host 127.0.0.1 `
  --port 8000
```

### Log Live Data For V5

V5 is not enabled until there are enough matched live labels. First collect
MBTA `/predictions` and `/vehicles` snapshots:

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.inference.log_mbta_live_snapshots `
  --route-id 111 `
  --stop-id 5547 `
  --direction-id 1 `
  --poll-count 6 `
  --poll-interval-seconds 30
```

Then match snapshots to actual arrival/departure labels:

```powershell
C:\Users\yaobc\anaconda3\python.exe -m src.inference.build_v5_residual_dataset `
  --snapshot-dir reports/live_prediction_snapshots `
  --processed-path data/processed/arrival_departure.parquet `
  --min-samples 500
```

The V5 target is `actual_delay - official_delay`; when enough matched rows
exist, the calibrated prediction is `official_delay + residual_correction`.
Until then the report marks the result as `inconclusive`.

## References

- [Livable Streets Report](https://www.livablestreets.info/)
- [MBTA Guide to Ridership Data](https://www.mbta.com/)
- [64 Hours Documentary](https://www.youtube.com/)
- [Bus Network Redesign Phase 1](https://www.mbta.com/news/2024-10-07/phase-1-bus-network-redesign-launches-december-15-bring-more-frequent-service)

## Project Timeline

### Completed: Base Project (Feb 10 - Feb 16, 2026)

All base project requirements (Q1-Q7) completed within one week:

| Date | Task | Member | Status |
|------|------|--------|--------|
| Feb 10 | Data collection scripts, repository setup | zztangbu@bu.edu | ✅ |
| Feb 10-11 | MBTA data download (2020-2026, ~20GB) | zztangbu@bu.edu | ✅ |
| Feb 11-12 | Data cleaning, datetime parsing, delay calculation | All members | ✅ |
| Feb 12-13 | Q1-Q3: Ridership, travel time, wait time analysis | zztangbu@bu.edu, lzj2729@bu.edu | ✅ |
| Feb 13-14 | Q4-Q5: Citywide delays, target routes analysis | lzj2729@bu.edu, ljf628@bu.edu | ✅ |
| Feb 14-15 | Q6-Q7: Service disparities, demographic impact | ljf628@bu.edu, yaobc@bu.edu | ✅ |
| Feb 15-16 | Visualizations, final report draft | All members | ✅ |

### Completed: Extended Experiments Phase 1 (Feb 12 - Feb 16, 2026)

| Date | Task | Member | Status |
|------|------|--------|--------|
| Feb 12 | V1-V2: Baseline MLP/LSTM/GRU models | zztangbu@bu.edu | ✅ |
| Feb 13 | V3: Time series features (FFT, Wavelet, Lag) | zztangbu@bu.edu | ✅ |
| Feb 13 | V3-Ablation: Feature extraction comparison | zztangbu@bu.edu | ✅ |
| Feb 14-16 | V5-NeuronSpark: SNN implementation | zztangbu@bu.edu | ✅ |
| Feb 16 | V6-Transformer: Attention-based model | zztangbu@bu.edu | ✅ |

**Current Best Results**:
| Model | RMSE (min) | R² | Parameters |
|-------|------------|-----|------------|
| Transformer | 0.46 | 0.9942 | ~1.6M |
| NeuronSpark SNN | 0.61 | 0.9897 | ~1.4M |
| GRU + Time Series | 0.75 | 0.9846 | ~150K |

### Planned: Extended Experiments Phase 2 (Feb 17 - Mar 2026)

| Date | Task | Member | Deliverable |
|------|------|--------|-------------|
| Feb 17-28 | NeuronSpark v7 training on full dataset (3.76M samples) | zztangbu@bu.edu | Improved SNN model |
| Mar 1-7 | Additional feature extraction methods (EMD, STFT) | zztangbu@bu.edu | Feature comparison |
| Mar 8-14 | Cross-validation and hyperparameter tuning | zztangbu@bu.edu, lzj2729@bu.edu | Validation results |
| **Mar Check-In** | Present feature engineering and model comparison | All members | Check-in meeting |

### Planned: Extended Experiments Phase 3 (Apr 2026)

| Date | Task | Member | Deliverable |
|------|------|--------|-------------|
| Apr 1-7 | Ensemble methods exploration | zztangbu@bu.edu | Ensemble model |
| Apr 8-14 | Comprehensive ablation study on all features | zztangbu@bu.edu, ljf628@bu.edu | Ablation report |
| Apr 15-21 | Advanced visualizations (interactive plots, feature importance) | yaobc@bu.edu | Visualization dashboard |
| **Apr Check-In** | Present complete model analysis | All members | Check-in meeting |

### Final Deliverables (May 2026)

| Date | Task | Member | Deliverable |
|------|------|--------|-------------|
| Apr 22-28 | Final report writing and results consolidation | zztangbu@bu.edu | `reports/FINAL_REPORT.md` |
| Apr 28-30 | Presentation video recording | All members | YouTube link |
| **May 1** | **Final Report Due** | All members | GitHub repo + video |

---

## Extended Experiments: Neuromorphic Computing

### Feature Engineering
| Method | Description |
|--------|-------------|
| Wavelet Decomposition | Daubechies db4 for multi-scale temporal patterns |
| FFT Features | Spectral analysis for frequency domain |
| Lag Features | Previous delay values with rolling statistics |

### Spiking Neural Network (NeuronSpark)
- K-bit deterministic binary spike encoding
- Dynamic membrane parameters (β, α, V_th)
- Parallel scan algorithms for PLIF neuron dynamics

### Code Locations
| File | Description |
|------|-------------|
| `src/models/train_delay_predictor_v3_fixed.py` | Time series feature engineering |
| `src/models/snn_delay_model.py` | NeuronSpark SNN architecture |
| `src/models/train_snn_delay.py` | SNN training pipeline |
| `NeuronSpark/` | NeuronSpark reference implementation |

### Results Locations
| File | Description |
|------|-------------|
| `reports/FINAL_REPORT.md` | Section 8: Delay Prediction Model |
| `reports/DELAY_PREDICTION_COMPARISON_REPORT.md` | Model comparison |
| `reports/figures/` | Visualizations and training curves |

## Team

Boston University CS506 - Spring 2026

SPARK - HUB - CS506

### Authors

- zztangbu@bu.edu
- lzj2729@bu.edu
- ljf628@bu.edu
- yaobc@bu.edu

## License

This project is for educational purposes as part of BU CS506 coursework.
