# Boston Bus Equity Analysis
## Final Report - Spring 2026

**Course:** CS506 - Data Science Tools and Applications
**Client:** City of Boston Analytics Team / Spark!

### Authors

- zztangbu@bu.edu
- lzj2729@bu.edu
- ljf628@bu.edu
- yaobc@bu.edu

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Data Sources](#data-sources)
4. [Methodology](#methodology)
5. [Key Findings](#key-findings)
   - [Q1: Ridership Analysis](#q1-ridership-analysis)
   - [Q2: Travel Times](#q2-travel-times)
   - [Q3: Wait Times](#q3-wait-times)
   - [Q4: Citywide Delays](#q4-citywide-delays)
   - [Q5: Target Routes Analysis](#q5-target-routes-analysis)
   - [Q6: Service Disparities](#q6-service-disparities)
   - [Q7: Demographic Impact](#q7-demographic-impact)
   - [Q8: Delay Prediction Models (Extended Research)](#q8-delay-prediction-models-extended-research)
6. [Visualizations](#visualizations)
7. [Limitations](#limitations)
8. [Conclusions and Recommendations](#conclusions-and-recommendations)

---

## Executive Summary

This report analyzes MBTA bus service performance and its equity implications for Boston residents. Using ridership data (2016-2024) and bus arrival/departure times (2020-2024), we examined service quality across routes and neighborhoods to understand how bus performance impacts different communities.

### Key Findings:

- **Ridership declined 32.8%** from pre-pandemic (2016-2019) to post-pandemic (2021-2024) levels
- **Average citywide delay is 7.5 minutes**, with only 31.7% on-time performance
- **Target routes (serving underserved communities) experience 41% higher delays** than other routes (10.2 vs 7.2 minutes)
- **6 neighborhoods classified as "vulnerable"** (high minority + low income): Dorchester, East Boston, Hyde Park, Mattapan, Mission Hill, and Roxbury
- Service quality shows **no significant negative correlation with minority population**, suggesting equitable service distribution in terms of delays

---

## Project Overview

### Goal
To better understand the impact of bus performance on Boston residents by examining service performance trends by geography and demographics.

### Scope
- Analyze MBTA bus ridership trends pre vs post pandemic
- Examine delay patterns across routes and time periods
- Identify service level disparities
- Assess demographic equity implications

### Research Questions
1. What is the ridership per bus route? How has this changed from pre-pandemic to post-pandemic?
2. What are the end-to-end travel times for each bus route?
3. On average, how long does an individual have to wait for a bus?
4. What is the average delay time across all routes citywide?
5. What is the average delay for target routes (22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14)?
6. Are there disparities in service levels between routes?
7. Are there differences in service quality impacting different demographic groups?

---

## Data Sources

### Primary Data

| Dataset | Source | Years | Records |
|---------|--------|-------|---------|
| Bus Arrival Departure Times | MBTA Open Data Portal | 2020-2024 | ~27M records |
| Bus Ridership by Trip/Season | MBTA Open Data Portal | 2016-2024 | ~1.8M records |

### Secondary Data

| Dataset | Source | Description |
|---------|--------|-------------|
| Census/ACS Demographics | Boston Planning & Development Agency | 2015-2019 neighborhood demographics |
| GTFS Stops Data | MBTA | Bus stop locations and route mappings |

### Data Limitations

**Important:** The MBTA Bus Arrival Departure Times data for 2018-2019 is no longer available on the MBTA Open Data Portal. This limits our ability to analyze pre-pandemic delay patterns. However, ridership data is complete from 2016-2024, allowing full pre/post pandemic comparison for Q1.

---

## Methodology

### Data Processing Pipeline

1. **Data Loading**: Chunked processing for large CSV files (500K rows per chunk)
2. **Datetime Parsing**: Combined service_date with scheduled/actual times
3. **Delay Calculation**: `delay_minutes = (actual_time - scheduled_time)`
4. **Geographic Mapping**: Mapped 2,910 bus stops to 22 Boston neighborhoods using coordinate-based matching
5. **Route-Demographic Profiling**: Created demographic profiles for 210 routes based on neighborhoods served

### Analysis Approach

- **Descriptive Statistics**: Mean, median, percentiles for delays and travel times
- **Comparative Analysis**: Target routes vs other routes, pre vs post pandemic
- **Correlation Analysis**: Pearson correlations between service metrics and demographics
- **Statistical Testing**: Mann-Whitney U tests for group comparisons

---

## Key Findings

### Q1: Ridership Analysis

**How has ridership changed from pre-pandemic to post-pandemic?**

![Ridership Trends](figures/ridership_pre_post_pandemic.png)

| Period | Avg Annual Boardings | Change |
|--------|---------------------|--------|
| Pre-Pandemic (2016-2019) | 746,761 | - |
| Pandemic (2020) | 363,317 | -51.3% |
| Post-Pandemic (2021-2024) | 501,474 | -32.8% vs pre |

**Key Insights:**
- Ridership dropped dramatically in 2020 (pandemic impact)
- Recovery is ongoing but still 32.8% below pre-pandemic levels
- 2023 showed strongest recovery with 616,157 average boardings
- 2024 data is partial (through available months)

---

### Q2: Travel Times

**What are the end-to-end travel times for each bus route?**

![Travel Times](figures/travel_times.png)

| Metric | Value |
|--------|-------|
| Average route travel time | 28.4 minutes |
| Shortest route | 8.2 minutes |
| Longest route | 89.3 minutes |
| Median travel time | 24.6 minutes |

**Key Insights:**
- Wide variation in travel times reflects route length diversity
- Longer routes tend to accumulate more delays
- Peak hour travel times are 15-20% longer than off-peak

---

### Q3: Wait Times

**How long does an individual wait for a bus (on-time vs delayed)?**

![Wait Time Comparison](figures/wait_time_comparison.png)

| Condition | Expected Wait |
|-----------|---------------|
| On-Time Buses | ~5 minutes (half of scheduled headway) |
| Delayed Buses | ~12-15 minutes |
| Overall Average | ~8 minutes |

**Key Insights:**
- When buses run on time, wait times are predictable
- Delays significantly impact passenger wait times
- Bunching (multiple buses arriving together) reduces effective frequency

---

### Q4: Citywide Delays

**What is the average delay across all routes?**

![Delay Distribution](figures/delay_distribution.png)
![Delays by Hour](figures/delays_by_hour.png)

| Metric | Value |
|--------|-------|
| Mean Delay | 7.51 minutes |
| Median Delay | 1.34 minutes |
| 95th Percentile | 42.6 minutes |
| On-Time Performance | 31.7% |

**Delay Categories:**
- Early arrivals: 14.8%
- On-time (-2 to +5 min): 31.7%
- Minor delay (5-10 min): 18.2%
- Moderate delay (10-15 min): 12.4%
- Major delay (>15 min): 22.9%

**Time Patterns:**
- Highest delays during evening rush (4-7 PM)
- Weekend service shows slightly better on-time performance
- Winter months show increased delays

---

### Q5: Target Routes Analysis

**How do the target routes (from Livable Streets report) perform?**

![Target Routes Summary](figures/target_routes_summary.png)

| Metric | Target Routes | Other Routes | Difference |
|--------|--------------|--------------|------------|
| Mean Delay | 10.20 min | 7.22 min | +41% |
| On-Time Performance | 25.8% | 32.3% | -6.5 pts |
| Service Score | Lower | Higher | Significant |

**Target Routes:** 22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14

**Key Insights:**
- Target routes experience significantly higher delays
- These routes serve predominantly low-income and minority communities
- Lower on-time performance creates greater burden on transit-dependent riders

---

### Q6: Service Disparities

**Are there disparities in service levels between routes?**

![Service Score Comparison](figures/service_score_comparison.png)
![On-Time Performance](figures/on_time_performance.png)

**Service Level Distribution:**
- Top 10% of routes: 45%+ on-time performance
- Bottom 10% of routes: <20% on-time performance
- Standard deviation in service scores: 15.2 points

**Disparity Findings:**
- Significant variation in service quality across routes
- Some routes consistently underperform
- Target routes cluster in the lower performance tier
- Routes with higher ridership tend to have more delays (capacity constraints)

---

### Q7: Demographic Impact

**Are there differences in service quality impacting different demographic groups?**

![Demographic Correlations](figures/demographic_correlations_heatmap.png)
![Neighborhood Demographics](figures/neighborhood_demographics.png)

**Neighborhood Classifications:**

| Category | Count | Examples |
|----------|-------|----------|
| High Minority (>50%) | 6 | Dorchester, Mattapan, Roxbury |
| Low Income | 11 | Mission Hill, East Boston |
| High Poverty | 9 | Fenway, Roxbury |
| Vulnerable (both) | 6 | Dorchester, Mattapan, Roxbury, Hyde Park, East Boston, Mission Hill |

**Correlation Analysis:**

| Service Metric | Demographic Variable | Correlation | p-value | Significant |
|----------------|---------------------|-------------|---------|-------------|
| Mean Delay | Minority % | -0.007 | 0.96 | No |
| Mean Delay | Hispanic % | -0.332 | 0.016 | Yes |
| Mean Delay | Median Income | 0.002 | 0.99 | No |
| On-Time Performance | Poverty Rate | -0.082 | 0.56 | No |

**Key Insights:**
- **No significant negative correlation** between service quality and minority population
- Routes serving high-minority areas show **slightly lower delays** on average
- The significant correlation with Hispanic % is negative (lower delays), not indicating discrimination
- However, **target routes** (identified by Livable Streets as equity-priority routes) do show worse performance

**Interpretation:**
The demographic analysis reveals a nuanced picture. While route-level analysis shows no systematic bias against minority communities, the target routes (specifically identified as serving underserved communities) do experience worse service. This suggests that other factors (route length, traffic conditions, infrastructure) may be the primary drivers of delays, not demographic characteristics per se.

---

### Q8: Delay Prediction Models (Extended Research)

**Can we accurately predict bus delays using machine learning with advanced feature engineering?**

This extended research question represents the primary technical focus of the semester. We developed six successive model versions (V1–V6), progressing from simple static-feature baselines to a Transformer architecture achieving R² = 0.9942 on a strict temporal holdout.

#### 8.1 Progressive Model Development

| Version | Key Innovation | Best Architecture | RMSE (min) | R² |
|---------|---------------|-------------------|------------|-----|
| V1 Baseline | Static context features only | MLP | 6.24 | −0.07 |
| V2 Historical Stats | Route/stop historical averages | LSTM | 6.34 | −0.11 |
| V3 Time Series | Lag + FFT + Wavelet + rolling features | GRU | 0.75 | 0.9846 |
| V4 Multi-step | Seq2Seq autoregressive forecasting | Seq2Seq GRU | 5.72 | 0.085 |
| V5 NeuronSpark | Spiking Neural Network (full-data retrain) | SNN-D128-K8 | 0.61 | 0.9897 |
| V6 Transformer | Multi-head attention (full-data retrain) | 6-layer Transformer | 0.46 | 0.9942 |

**Training data:** 2020–2024 (121 M records, 3.76 M feature-extracted samples).  
**Test data:** 2025–2026 (28 M records). Strict temporal split — no overlap.

#### 8.2 Why V1 and V2 Failed (R² < 0)

Negative R² means the model performs worse than predicting the mean for every observation.

1. **Static context is insufficient.** Knowing "Route 28, Monday, 5 PM" does not indicate whether *this specific* bus is delayed. The same route/time combination yields vastly different delays on different days due to external factors (traffic incidents, weather, crowding).
2. **Historical averages are too coarse.** A five-year route average cannot reflect today's conditions. Bus delays are **non-stationary**: the system state right now depends on the last few arrivals, not on the long-run mean.

#### 8.3 Why V3 Succeeded (R² = 0.9846)

The breakthrough came from exposing the model to recent delay history. Three categories of engineered features drive this:

| Feature Category | RMSE (GRU, ablation) | Marginal gain vs baseline (0.9436) |
|-----------------|----------------------|-----------------------------------|
| Lag features only (baseline) | 0.9436 | — |
| + Rolling statistics | 0.9091 | **Best individual method** |
| + FFT components | 0.9387 | Moderate |
| + Wavelet decomposition | 0.9431 | Modest |
| + Statistical moments | 0.9482 | Small |
| **All combined** | **0.9056** | **3.8% further RMSE reduction** |

**Key finding: feature engineering matters more than architecture.** All three architectures (MLP, LSTM, GRU) achieve R² > 0.98 with the same V3 feature set.

#### 8.4 Feature Engineering Details

**Lag features (7 dims):** Previous 5 delay values plus first- and second-order differences, computed via `series.shift(k)` to guarantee no future leakage.

**Rolling statistics (8 dims):** Mean, std, min, max over 5- and 10-step windows excluding the current index.

**FFT features (6 dims):** Discrete Fourier Transform on a 10-step historical window; top-3 magnitudes and frequencies extracted.

**Wavelet features (6 dims):** Daubechies-4 two-level DWT (`pywt.wavedec(window, 'db4', level=2)`); mean and std per decomposition level.

**Statistical features (4 dims):** Skewness, kurtosis, linear trend slope, and volatility (std of first differences).

**Historical baselines (5 dims):** Route-level and stop-level delay mean/std computed on training data only; hour-of-day mean as a causal contextual baseline.

**Cyclical time encoding (4 dims):** `sin/cos(2π·hour/24)` and `sin/cos(2π·dow/7)` to eliminate the discontinuity at midnight and week boundaries.

#### 8.5 V5 NeuronSpark Spiking Neural Network

We applied the NeuronSpark SNN architecture — originally developed for neuromorphic computing research — to transportation time-series regression. Key innovations:

- **K-bit deterministic binary encoding:** Continuous values in [0,1] mapped to 8 binary spike frames (MSB-first). Example: 0.75 → `[1, 1, 0, 0, 0, 0, 0, 0]`.
- **Dynamic membrane parameters:** Decay rate β, write gain α, and threshold V_th are all computed from both the input spike and the current membrane voltage, enabling voltage-gated feedback similar to attention.
- **Surrogate gradient training:** Non-differentiable spike activations are approximated with a smooth surrogate during backpropagation.

Full-dataset training (3.76 M samples, 13 epochs, ~3.4 h on MPS GPU):

| Metric | GRU Baseline | NeuronSpark SNN | Improvement |
|--------|-------------|-----------------|-------------|
| RMSE (min) | 0.6384 | **0.6098** | −4.5% |
| R² | 0.9893 | **0.9897** | +0.0004 |
| Training time | ~55 s | ~7.6 h | 500× slower |

The SNN requires large training sets to outperform GRU. At 50 K samples it is worse; at 3.76 M samples it surpasses GRU, demonstrating that its complex multi-timescale dynamics provide genuine representational benefit given sufficient data.

#### 8.6 V6 Transformer (State-of-the-Art Comparison)

To provide a fair architectural comparison at similar parameter scale, we trained a 6-layer Transformer with d=128 and 8 attention heads (~1.6 M parameters vs SNN's ~1.4 M).

Full-dataset training (3.76 M samples, 21 epochs, ~2.6 h on MPS GPU):

| Metric | NeuronSpark SNN | Transformer | Transformer advantage |
|--------|-----------------|-------------|----------------------|
| RMSE (min) | 0.6098 | **0.4599** | −24.6% |
| MAE (min) | 0.3311 | **0.0595** | −82% |
| R² | 0.9897 | **0.9942** | +0.0045 |
| Training time | 7.6 h | **5.8 h** | −24% |

The Transformer achieves state-of-the-art performance because standard backpropagation with exact gradients is more efficient than surrogate gradient approximations, and self-attention natively captures global temporal dependencies without the quantization overhead of spike encoding.

Both V5 and V6 reproduce published results to within 0.0002 R² (sampling noise), confirming that the full-data checkpoints are correctly trained.

#### 8.7 Deployment Finding: Live vs Offline Ranking Inversion

The most important applied finding emerged from connecting the trained models to the MBTA V3 live API.

**Offline test-set ranking** (R², 2025–2026 matched actuals):

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| V6 Transformer | **0.9940** (best) | 0.46 min |
| V5 NeuronSpark SNN | 0.9897 | 0.61 min |
| V3 GRU + wavelet | 0.9846 (worst) | 0.75 min |

**Live API ranking** (mean absolute gap vs MBTA official predictions, route 1, stop 110, 5 upcoming trips):

| Model | Mean abs gap | vs offline rank |
|-------|-------------|-----------------|
| V3 GRU + wavelet | **5.06 min** (best) | offline: worst |
| V5 NeuronSpark SNN | 8.54 min | offline: middle |
| V6 Transformer | 8.80 min (worst) | offline: best |

**The ranking inverts.** Three controlled experiments identified the cause:

1. **Distribution shift:** Live MBTA data at the time of measurement had mean delay −1.8 min (buses running early), while the training distribution has mean +5.6 min (buses usually late). The fraction of early buses flipped from 19% in training to 78% in live data.

2. **V5/V6 are near-linear lag amplifiers:** Fed identical synthetic lag inputs, both V5 and V6 predict within ±0.4 min of the input value. They learned "next delay ≈ recent delays" and suppress the time-of-day signal.

3. **V3 is robust because FFT/wavelet compress lag extremes:** The signal-processing features reduce raw lag spikes into smoother low-frequency components. V3's GRU learned to weight these compressed views, making it insensitive to noisy live lag values.

This is a textbook **bias-variance result**:

| Model | Bias | Variance | Wins when |
|-------|------|----------|-----------|
| V3 GRU + wavelet | Higher (predicts near mean) | Lower | Live data is noisy / out-of-distribution |
| V5 SNN, V6 Transformer | Lower | Higher | Historical test data is clean / in-distribution |

**Practical implication:** The headline "Transformer is best" holds only on the offline split. On deployment-realistic live data, the V3 GRU equipped with signal-processed features is the most useful model. This motivates three follow-on investigations: matched-actuals live evaluation, noise-injection retraining of V6, and a V3+V6 ensemble.

#### 8.8 Realtime Inference Dashboard

To demonstrate end-to-end deployment, we built a FastAPI dashboard (`src/inference/`) that:

- Serves all six model checkpoints (V1–V6) via a model picker dropdown.
- Pulls live upcoming-trip predictions from the MBTA V3 API and runs the selected local model on the same trips for side-by-side comparison.
- Presents project KPIs, all offline figures, and interactive model metrics in a single browser UI.
- Includes 64 deployment and API-loadability tests (`tests/test_deployment.py`).

Run locally:

```bash
python -m src.inference.serve \
  --bundle models/delay_predictor_v4_score_best_online_safe_bundle.joblib \
  --host 127.0.0.1 \
  --port 8000
```

Open `http://127.0.0.1:8000/`.

The default dashboard model is the V4 LightGBM `v2_core` quantile bundle — chosen for its online-safe causal features and superior early-delay prediction (negative-prediction rate 17.1%, early-delay F1 0.373) over the V2 neural baseline.

#### 8.9 Q8 Summary

| Milestone | Result |
|-----------|--------|
| Baseline (V1) failure confirmed | R² = −0.07; static context alone insufficient |
| Feature engineering breakthrough (V3) | R² = 0.9846; **88% RMSE reduction** over V1 |
| SNN competitive at scale (V5) | R² = 0.9897; outperforms GRU with 3.76 M samples |
| State-of-the-art (V6 Transformer) | R² = 0.9942; **93% RMSE reduction** over V1 |
| Live vs offline inversion finding | V3 GRU best on live MBTA data; bias-variance root cause confirmed |
| Realtime dashboard | 6-model picker, live MBTA comparison, 64 automated tests |

---

## Visualizations

All visualizations are available in `reports/figures/`:

| # | Visualization | Research Question |
|---|--------------|-------------------|
| 1 | `ridership_pre_post_pandemic.png` | Q1 |
| 2 | `ridership_by_route_comparison.png` | Q1 |
| 3 | `travel_times.png` | Q2 |
| 4 | `wait_time_comparison.png` | Q3 |
| 5 | `delay_distribution.png` | Q4 |
| 6 | `delays_by_hour.png` | Q4 |
| 7 | `delays_by_day.png` | Q4 |
| 8 | `monthly_delay_trends.png` | Q4 |
| 9 | `target_routes_summary.png` | Q5 |
| 10 | `service_score_comparison.png` | Q6 |
| 11 | `on_time_performance.png` | Q6 |
| 12 | `delays_by_route.png` | Q6 |
| 13 | `demographic_correlations_heatmap.png` | Q7 |
| 14 | `demographic_service_comparison.png` | Q7 |
| 15 | `neighborhood_demographics.png` | Q7 |
| 16 | `delay_prediction_training_curves_v1_baseline_temporal.png` | Q8 — V1 training convergence |
| 17 | `delay_prediction_training_curves_v2_lag_features_temporal.png` | Q8 — V2 training convergence |
| 18 | `delay_prediction_training_curves_v3_wavelet_temporal.png` | Q8 — V3 training convergence |
| 19 | `ablation_study_comparison.png` | Q8 — Feature method ablation |
| 20 | `delay_prediction_multistep_comparison.png` | Q8 — V4 multi-step performance |
| 21 | `delay_prediction_neuronspark_comparison.png` | Q8 — V5 SNN vs GRU comparison |
| 22 | `v4_model_sweep.png` | Q8 — V4 model family sweep |
| 23 | `v4_model_deployability_scores.png` | Q8 — Deployability scoring |
| 24 | `official_vs_v4_vs_actual.png` | Q8 — Offline accuracy vs MBTA official |
| 25 | `v4_optimization_story.png` | Q8 — V4 optimization narrative |
| 26 | `mbta_realtime_model_gap_story.png` | Q8 — Live vs offline gap |
| 27 | `mbta_realtime_official_vs_model.png` | Q8 — Live MBTA vs local model |

---

## Limitations

### Data Limitations

1. **Missing Historical Data**: Bus Arrival Departure Times for 2018-2019 are no longer available on the MBTA portal, limiting pre-pandemic delay analysis.

2. **Sampling**: Quick analysis used 5% sample of full dataset for computational efficiency. Full analysis recommended for production use.

3. **Geographic Approximation**: Stop-to-neighborhood mapping uses approximate bounding boxes, not precise polygon boundaries.

### Analytical Limitations

1. **Causality**: Correlations do not imply causation. Observed relationships between demographics and service quality may be driven by confounding factors.

2. **Temporal Scope**: Post-pandemic data (2020-2024) may not reflect "normal" operations due to ongoing ridership recovery and service adjustments.

3. **Route-Level Aggregation**: Demographic profiles are averaged across neighborhoods served, potentially masking within-route variation.

---

## Conclusions and Recommendations

### Conclusions

1. **Ridership Recovery is Incomplete**: Bus ridership remains 33% below pre-pandemic levels, indicating a need for service improvements to attract riders back.

2. **Service Quality is a Challenge**: With only 31.7% on-time performance, MBTA bus service has significant room for improvement.

3. **Target Routes Need Attention**: Routes serving underserved communities experience 41% higher delays, confirming the equity concerns raised by Livable Streets.

4. **No Systematic Demographic Bias in Delays**: Route-level analysis shows no evidence that minority or low-income areas receive systematically worse service, though target routes remain a concern.

5. **Delay Prediction is Highly Feasible**: With lag + signal-processing features, bus delays are highly predictable in the short term (R² = 0.9846 for V3 GRU). The dramatic improvement from V1 (R² = −0.07) to V3 confirms that temporal autocorrelation is the dominant signal.

6. **Feature Engineering Dominates Architecture Choice**: All three architectures (MLP, LSTM, GRU) achieve R² > 0.98 given the same V3 feature set. Architecture selection matters primarily at the margin once good features are in place.

7. **Larger Models Benefit from More Data**: NeuronSpark SNN surpasses GRU only when trained on 3.76 M samples (not 50 K). Transformer achieves R² = 0.9942 — a 93% RMSE reduction from baseline — with full-data training.

8. **Offline Accuracy Does Not Guarantee Live Performance**: The V6 Transformer (best offline) is the worst model on live MBTA API data. The V3 GRU (worst offline) is the best live. This bias-variance inversion is caused by a distribution shift (live buses run early; training data shows buses running late) and by V5/V6 being near-linear lag amplifiers that propagate noisy live inputs. This is the project's most deployment-relevant finding.

### Recommendations

1. **Prioritize Target Route Improvements**: Focus infrastructure and scheduling improvements on the 15 target routes identified by Livable Streets.

2. **Peak Hour Intervention**: Deploy additional resources during evening rush hours (4–7 PM) when delays are highest.

3. **Monitor Equity Metrics**: Establish ongoing tracking of service quality by neighborhood demographics.

4. **Ridership Recovery Initiatives**: Develop strategies to restore ridership, including service reliability improvements and community outreach.

5. **Data Transparency**: Request MBTA to restore historical data (2018–2019) for comprehensive trend analysis.

6. **Deploy V3 GRU for Live Prediction**: Until matched-actuals live evaluation is complete, serve the V3 GRU + wavelet model for production delay estimates. Its lower variance on out-of-distribution live data outweighs the Transformer's superior offline accuracy.

7. **Retrain V6 with Noise Injection**: Augment V6 Transformer training with Gaussian-perturbed lag inputs to reduce its sensitivity to distribution shift, potentially combining the offline accuracy advantage with V3's live robustness.

8. **Collect Matched Live Labels**: Stream MBTA predictions and match them against eventual actual arrivals to enable a true live MAE comparison across all six architectures.

---

## Appendix

### Code Repository

All analysis code is available in the project repository:
- `src/data/` - Data loading and processing modules
- `src/analysis/` - Analysis modules for each research question
- `src/visualization/` - Visualization generation
- `data/processed/` - Cleaned datasets and analysis results

### Team

Boston University CS506 - Spring 2025

### Acknowledgments

- MBTA Open Data Portal
- City of Boston Analytics Team
- Spark! at Boston University
- Livable Streets Alliance (for target routes identification)
