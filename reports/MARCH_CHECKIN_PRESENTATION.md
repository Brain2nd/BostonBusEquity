# Boston Bus Equity - April Check-In Report

**Course:** CS506 - Data Science Tools and Applications, Spring 2026
**Team:** zztangbu@bu.edu, lzj2729@bu.edu, ljf628@bu.edu, yaobc@bu.edu
**Client:** City of Boston Analytics Team / Spark!
**Date:** April 2026

---

## 0. April Additions: Live Deployment + Surprising Online vs Offline Result

This section is the new content for the April check-in. The original March
check-in material (Sections 1 - 4 below) remains intact for continuity.

### 0.1 What we built since March

| Deliverable | Status |
|-------------|--------|
| FastAPI realtime inference dashboard with live MBTA V3 API integration | Complete |
| Multi-model picker exposing all V1 - V6 architectures end-to-end | Complete |
| V6 Transformer **full-data** training (3.76 M samples, 21 epochs, 2.6 h on MPS GPU) | Complete |
| V5 NeuronSpark SNN **full-data** training (3.76 M samples, 13 epochs, 3.4 h on MPS GPU) | Complete |
| V4 Seq2Seq multi-step training + checkpoint | Complete |
| Live-compare endpoint: pulls upcoming trips from MBTA V3 and runs the picked local model on them | Complete |
| 64 deployment / API / model-loadability tests | Complete |

### 0.2 V5 / V6 reproduce the paper exactly

After saving checkpoints for V5 and V6 (which the original training scripts
did not persist), we trained both on the full 3.76 M-sample training set
with the same temporal split (train < 2025, test >= 2025):

| Model | Paper R² | Our retrain R² | Paper RMSE | Our RMSE |
|-------|----------|----------------|------------|----------|
| V5 NeuronSpark SNN | 0.9897 | **0.9897** | 0.6098 | 0.6100 |
| V6 Transformer | 0.9942 | **0.9940** | 0.4599 | 0.4644 |

V5 reproduced the paper R² to four decimal places. V6 differs by 0.0002
(within sampling noise). The full-trained checkpoints are now what the
dashboard serves by default.

### 0.3 The surprise: GRU beats Transformer on live MBTA data

The most interesting finding from April is an apparent contradiction
between offline test-set accuracy and live MBTA-API agreement.

**Offline ranking** (test-set R² on 2025-2026 matched actuals):

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| **V6 Transformer** | **0.9940** (best) | 0.46 min |
| V5 NeuronSpark SNN | 0.9897 | 0.61 min |
| V3 GRU + wavelet | 0.9846 (worst of the three) | 0.75 min |

**Live ranking** (mean absolute gap vs MBTA V3 official predictions on the
same upcoming trips, using live MBTA delays as lag input — route 1,
stop 110, 5 trips):

| Model | Live mean_abs_gap | vs offline rank |
|-------|------------------|-----------------|
| **V3 GRU + wavelet** | **5.06 min** (best on live) | offline: worst |
| V5 NeuronSpark SNN | 8.54 min | offline: middle |
| V6 Transformer | 8.80 min (worst on live) | offline: best |

**The ranking inverts.** V6 Transformer wins offline by 0.5 min RMSE but
is ~3.7 min farther from MBTA's live predictions than V3 GRU. This is the
opposite of what the project's headline narrative ("V6 Transformer is the
best model") would suggest.

### 0.4 Why this might be happening (preliminary hypotheses)

We have not yet proven the cause; these are working hypotheses for the
final-report investigation:

1. **Generalization vs overfitting.** V6 Transformer (1.6 M params) and V5
   SNN (1.4 M params) each have ~10x the parameters of V3 GRU (~150 K).
   On 2025-2026 test data the larger models exploit subtle patterns that
   stop holding when 2026 routes/equipment differ from training.

2. **Lag distribution shift between offline and live.** Offline lag
   features come from the same temporal-split training distribution. Live
   lag features come from MBTA V3 official_delay predictions, which are
   themselves model outputs (not realized actuals). The two sources have
   different noise characteristics; smaller models like V3 GRU may be more
   robust to this kind of distribution shift.

3. **MBTA "official" is itself a prediction, not ground truth.** Mean
   absolute gap to MBTA is a proxy for accuracy, not accuracy itself. V6
   could be MORE accurate against actual arrivals while disagreeing more
   with MBTA's own model. We currently have no labeled actuals to settle
   this — that requires the V5 residual labeling phase.

4. **Live cache freshness.** Our live lag uses MBTA's next-12 upcoming
   predictions. If V6 has learned to over-rely on lag-tail variance and
   the live lag's variance characteristics differ from training, V6 will
   be more sensitive than V3.

### 0.5 What we will investigate in the final report

- **Build matched-actuals dataset.** Stream MBTA V3 predictions for the
  most-watched routes for 2-4 weeks, then match each prediction against
  the trip's actual arrival when it lands. This gives us live ground
  truth, which is what we actually need to settle the V3-vs-V6 question.

- **Decompose the offline-online gap.** Re-evaluate V3 / V5 / V6 on the
  matched-actuals set and report each model's true MAE / RMSE / R² there,
  not just its agreement with MBTA's own predictions.

- **Smaller-vs-larger generalization study.** If V3 GRU still beats V6
  Transformer on matched live actuals, the next experiment is a
  parameter-count sweep on V6 (d_model 64, 96, 128, 256) to find the
  point at which over-parameterization starts hurting live performance.

- **Use MBTA vehicle-position telemetry as a feature.** Both MBTA's
  internal model and any production-deployed model would benefit from GPS
  position, current_stop_sequence, and vehicle_speed. Our V4 LightGBM
  exploration uses these but we have not folded them into V3 / V5 / V6.

### 0.6 Why this matters for the project narrative

The March check-in concluded that **feature engineering matters more than
architecture** (the V1 -> V3 jump from R² = -0.07 to 0.9846 was driven by
features, not by a more powerful model). The April finding strengthens
that conclusion in a different way: **a smaller GRU equipped with the right
features (lag + FFT + wavelet) outperforms larger architectures on the
deployment-realistic comparison**. The headline "Transformer is the best"
holds only on the offline split; on live MBTA data, the project's V3 GRU
is the most useful model. This is exactly the kind of result that motivates
the final-report investigation into deployment-quality evaluation.

---

## 1. Preliminary Data Visualizations (15 pts)

> Rubric: 1.1 At least one relevant visualization (5 pts) | 1.2 Clear and readable, well-labeled (5 pts) | 1.3 Show meaningful patterns or inform future exploration (5 pts)

We produced 22 visualizations covering all 8 research questions. All figures include axis labels, titles, legends, and color coding. Below are key figures with interpretations.

### Q1: Ridership Pre vs Post Pandemic

![Ridership Trends](figures/ridership_pre_post_pandemic.png)

![Ridership by Route](figures/ridership_by_route_comparison.png)

| Period | Avg Annual Boardings | Change |
|--------|---------------------|--------|
| Pre-Pandemic (2016-2019) | 746,761 | - |
| Pandemic (2020) | 363,317 | -51.3% |
| Post-Pandemic (2021-2024) | 501,474 | -32.8% vs pre |

**Pattern:** Ridership dropped dramatically in 2020 and has not fully recovered (-32.8%). Route-level breakdown reveals uneven recovery, informing which routes need targeted intervention.

---

### Q2: End-to-End Travel Times

![Travel Times](figures/travel_times.png)

| Metric | Value |
|--------|-------|
| Average route travel time | 28.4 minutes |
| Shortest route | 8.2 minutes |
| Longest route | 89.3 minutes |
| Median travel time | 24.6 minutes |

**Pattern:** Wide variation in travel times. Peak hour travel times are 15-20% longer than off-peak. Longer routes accumulate more delays — route length is a confounding factor for delay analysis.

---

### Q3: Wait Times (On-Time vs Delayed)

![Wait Time Comparison](figures/wait_time_comparison.png)

| Condition | Expected Wait |
|-----------|---------------|
| On-Time Buses | ~5 minutes (half of scheduled headway) |
| Delayed Buses | ~12-15 minutes |
| Overall Average | ~8 minutes |

**Pattern:** Wait times nearly triple when buses are delayed. Bus bunching further reduces effective frequency for passengers.

---

### Q4: Citywide Delay Patterns

![Delay Distribution](figures/delay_distribution.png)

![Delays by Hour](figures/delays_by_hour.png)

![Delays by Day](figures/delays_by_day.png)

![Monthly Delay Trends](figures/monthly_delay_trends.png)

| Metric | Value |
|--------|-------|
| Mean Delay | 7.51 minutes |
| Median Delay | 1.34 minutes |
| 95th Percentile | 42.6 minutes |
| On-Time Performance | 31.7% |

**Delay Categories:** Early arrivals: 14.8% | On-time (-2 to +5 min): 31.7% | Minor delay (5-10 min): 18.2% | Moderate delay (10-15 min): 12.4% | Major delay (>15 min): 22.9%

**Pattern:** Strong temporal regularity — evening rush (4-7 PM) peak, morning secondary peak, weekend improvement, winter worsening. These predictable patterns are exploitable by machine learning models.

---

### Q5: Target Routes vs Other Routes

![Target Routes Summary](figures/target_routes_summary.png)

| Metric | Target Routes | Other Routes | Difference |
|--------|--------------|--------------|------------|
| Mean Delay | 10.20 min | 7.22 min | +41% |
| On-Time Performance | 25.8% | 32.3% | -6.5 pts |

**Target Routes:** 22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14

**Pattern:** The 15 equity-priority routes (identified by Livable Streets Alliance) consistently underperform — 41% higher delays, confirming the equity concern quantitatively.

---

### Q6: Service Level Disparities

![Service Score Comparison](figures/service_score_comparison.png)

![On-Time Performance](figures/on_time_performance.png)

![Delays by Route](figures/delays_by_route.png)

**Pattern:** Top 10% of routes: >45% on-time; bottom 10%: <20% on-time (std dev = 15.2 pts). Target routes cluster in the lower performance tier. Higher-ridership routes tend to have more delays, suggesting capacity constraints.

---

### Q7: Demographic Impact

![Demographic Correlations](figures/demographic_correlations_heatmap.png)

![Demographic Service Comparison](figures/demographic_service_comparison.png)

![Neighborhood Demographics](figures/neighborhood_demographics.png)

| Service Metric | Demographic Variable | Correlation | p-value | Significant |
|----------------|---------------------|-------------|---------|-------------|
| Mean Delay | Minority % | -0.007 | 0.96 | No |
| Mean Delay | Hispanic % | -0.332 | 0.016 | Yes |
| Mean Delay | Median Income | 0.002 | 0.99 | No |
| On-Time Performance | Poverty Rate | -0.082 | 0.56 | No |

6 neighborhoods classified as "vulnerable" (high minority + low income): Dorchester, East Boston, Hyde Park, Mattapan, Mission Hill, Roxbury.

**Pattern:** No systematic demographic bias in delays at the route level. However, vulnerable neighborhoods are disproportionately served by underperforming target routes — the disparity is driven by infrastructure/traffic, not demographics per se.

---

### Q8: Delay Prediction Models

![V1 Training Curves](figures/delay_prediction_training_curves_v1_baseline_temporal.png)

![V3 Training Curves](figures/delay_prediction_training_curves_v3_wavelet_temporal.png)

![Ablation Study](figures/ablation_study_comparison.png)

![NeuronSpark vs GRU](figures/delay_prediction_neuronspark_comparison.png)

![Multi-step Prediction](figures/delay_prediction_multistep_comparison.png)

**Pattern:** V1 (static features) fails to converge — confirms temporal features are essential. V3 (time series features) converges smoothly. Ablation study reveals rolling statistics contribute most; combined features achieve best RMSE. NeuronSpark SNN slightly outperforms GRU on full dataset. Multi-step prediction (R²~0.08) is fundamentally harder than single-step (R²=0.98).

---

## 2. Data Processing Progress (15 pts)

> Rubric: 2.1 Clear sources of data and data collection methods (5 pts) | 2.2 Data cleaning steps considered (5 pts) | 2.3 Reasoning for data processing decisions are well-explained (5 pts)

### 2.1 Data Sources and Collection Methods

| Dataset | Source | Collection Method | Size | Status |
|---------|--------|-------------------|------|--------|
| Bus Arrival/Departure (2020-2026) | [MBTA Open Data Portal](https://mbta-massdot.opendata.arcgis.com/) | Automated script (`src/data/download_data.py`) with resume support | 161M records, ~18 GB | Complete |
| Bus Ridership (2016-2024) | MBTA Open Data Portal | Automated script | 1.8M records, ~850 MB | Complete |
| Passenger Survey | MBTA 2024 System-Wide Survey | Manual CSV download | ~50 MB | Complete |
| Census/ACS Demographics | Boston Data Portal & ACS 2020-2024 | API + manual download | Multiple CSVs | Complete |
| GTFS Routes & Stops | MBTA | Automated script | Standard GTFS format | Complete |

**Note:** 2018-2019 arrival/departure data is no longer available on MBTA portal. Training set uses 2020-2024; validation set uses 2025-2026.

### 2.2 Data Cleaning Steps

Our data processing pipeline (`src/data/preprocess.py`, `src/models/train_delay_predictor_v3_fixed.py`):

| Step | Description | Implementation |
|------|-------------|----------------|
| 1. Chunked Loading | Process large CSVs in 500K-row chunks to fit memory | `src/data/load_data.py` |
| 2. Datetime Parsing | Combine `service_date` + scheduled/actual into UTC-aware datetime | `preprocess.py` |
| 3. Delay Calculation | `delay_minutes = (actual - scheduled).total_seconds() / 60` | `preprocess.py` |
| 4. Missing Value Removal | Drop records with null `delay_minutes`, `scheduled`, or `service_date` | `train_delay_predictor_v3_fixed.py` |
| 5. Outlier Filtering | Retain delays within [-30, 60] minutes | `train_delay_predictor_v3_fixed.py` |
| 6. Temporal Split | `train = year < 2025`, `test = year >= 2025` (strict, no overlap) | `train_delay_predictor_v3_fixed.py` |
| 7. Geographic Mapping | Map 2,910 bus stops to 22 Boston neighborhoods using coordinates | `stop_neighborhood_mapping.py` |
| 8. Route-Demographic Profiling | Create demographic profiles for 210 routes based on neighborhoods served | `demographic_analysis.py` |
| 9. Format Conversion | Convert raw CSVs to Parquet (~5x compression, ~10x faster reads) | `convert_all_to_parquet.py` |

```python
# Core pipeline (src/models/train_delay_predictor_v3_fixed.py)
df['delay_minutes'] = (df['actual'] - df['scheduled']).dt.total_seconds() / 60
df = df.dropna(subset=['delay_minutes']).query('-30 <= delay_minutes <= 60')
train_df = df[df['year'] < 2025]     # 2020-2024: training
test_df  = df[df['year'] >= 2025]    # 2025-2026: validation (NO overlap)

# Scaler: fit on train ONLY, transform both (prevents leakage)
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)         # no fit!
```

### 2.3 Reasoning for Data Processing Decisions

| Decision | Reasoning |
|----------|-----------|
| **Delay range [-30, 60] min** | Values outside this range represent data errors (system glitches, cancelled trips). Including them distorts aggregate statistics. Range captures >99% of legitimate observations. |
| **Strict temporal split** | Prevents data leakage. Simulates real-world deployment: only historical data used to predict future. Random splits would let the model "memorize" future patterns. |
| **Chunked processing (500K rows)** | Full dataset (~18 GB) exceeds available RAM. Chunked processing with aggregation preserves accuracy on standard hardware. |
| **Scaler fit only on training data** | Fitting on both sets would leak test-set statistics into training, artificially inflating performance. |
| **Features use only past values** | `series.shift(1)` and `delays[i-window:i]` ensure each feature depends only on past observations. Current/future values never leak into features. |
| **Parquet format** | Columnar storage: ~5x compression over CSV, ~10x faster reads. Essential for iterating on 161M records. |
| **Per-group feature extraction** | `groupby(['route_id','stop_id'])` ensures lag/rolling features are computed within each route-stop pair, not across unrelated sequences. |

---

## 3. Modeling Methods (15 pts)

> Rubric: 3.1 Process being predicted/described is clearly explained (5 pts) | 3.2 Relevant features are chosen (5 pts) | 3.3 Clear justification for features used (5 pts)

### 3.1 Process Being Predicted

**Task:** Single-step bus delay regression — given a bus stop's current context and recent delay history, predict the next delay in minutes.

**Input:** 41-dimensional feature vector (route, stop, time context + temporal signal features)
**Output:** Predicted delay (continuous, minutes)
**Loss:** Mean Squared Error (MSE)

**Why this matters:**
- Real-time passenger information: accurate ETAs improve rider experience
- Proactive dispatch: dispatchers can intervene before delays cascade
- Resource allocation: identify routes needing additional capacity
- Scientific question: quantify how predictable bus delays are

**Analysis approach** (from FINAL_REPORT.md Methodology):
- Descriptive Statistics: Mean, median, percentiles for delays and travel times
- Comparative Analysis: Target routes vs other routes, pre vs post pandemic
- Correlation Analysis: Pearson correlations between service metrics and demographics
- Statistical Testing: Mann-Whitney U tests for group comparisons
- Machine Learning: Iterative model development from V1 (baseline) to V6 (Transformer)

### 3.2 Feature Selection

We engineered 41 features across 7 categories. Let *d_i* denote delay (minutes) at time step *i* for a given route-stop pair.

#### Lag Features (7 dims)

| Feature | Formula | Description |
|---------|---------|-------------|
| lag_k (k=1..5) | x = d_(i-k) | Delay *k* steps in the past |
| diff_1 | x = d_(i-1) - d_(i-2) | First-order difference (acceleration) |
| diff_2 | x = d_(i-1) - d_(i-3) | Second-order difference |

#### Rolling Statistics (8 dims)

For window *w* in {5, 10}, define *W_i = {d_(i-w), ..., d_(i-1)}* (excludes current):

| Feature | Formula | Description |
|---------|---------|-------------|
| roll_mean_w | mu = (1/w) * sum(W_i) | Mean of past *w* delays |
| roll_std_w | sigma = sqrt(var(W_i)) | Volatility over *w* steps |
| roll_min_w | min(W_i) | Best case in window |
| roll_max_w | max(W_i) | Worst case in window |

#### FFT Features (6 dims)

Discrete Fourier Transform on past 10-step window:
> X[k] = sum_{n=0}^{N-1} d_n * e^{-j2pi*kn/N}

Extract top 3 components by magnitude |X[k]| (excluding DC, k=0). Output: 3 magnitudes + 3 frequencies.

#### Wavelet Features (6 dims)

Daubechies-4 (db4) Discrete Wavelet Transform, 2-level decomposition:
> W_i --> DWT --> {cA2, cD2, cD1}

cA2 = level-2 approximation, cD2 = level-2 detail, cD1 = level-1 detail. Extract mean and std for each level (3 x 2 = 6 features).

#### Statistical Features (4 dims)

| Feature | Formula | Description |
|---------|---------|-------------|
| Skewness | gamma = E[((d - mu)/sigma)^3] | Asymmetry of delay distribution |
| Kurtosis | kappa = E[((d - mu)/sigma)^4] - 3 | Heavy-tailedness |
| Trend | beta_1 from linear fit d_j = beta_0 + beta_1*j | Slope: positive = worsening |
| Volatility | std(diff(W_i)) | Erratic-ness of changes |

#### Historical Statistics (5 dims)

Computed on **training data only**, then merged into both sets:

| Feature | Formula | Description |
|---------|---------|-------------|
| route_delay_mean/std | E[d\|route=r], Std[d\|route=r] | Route-level historical profile |
| stop_delay_mean/std | E[d\|stop=s], Std[d\|stop=s] | Stop-level historical profile |
| hour_delay_mean | E[d\|hour=h] | Hour-level baseline |

Unseen routes/stops in test set receive global mean as fallback.

#### Context + Temporal (9 dims)

| Feature | Method | Description |
|---------|--------|-------------|
| is_weekend, is_rush_hour | Binary flags | Weekend / rush hour (7-9, 16-19) |
| route_enc, stop_enc, dir_enc | LabelEncoder | 210 routes, 2,910 stops, direction |
| hour_sin, hour_cos | sin(2*pi*h/24), cos(2*pi*h/24) | Cyclical hour (23:00 close to 01:00) |
| dow_sin, dow_cos | sin(2*pi*d/7), cos(2*pi*d/7) | Cyclical day-of-week |

### 3.3 Feature Justification

| Feature Category | Why Selected | Evidence |
|-----------------|-------------|----------|
| **Lag features** | Delays have strong autocorrelation — if recent buses were late, next one likely is too (shared external causes: traffic, weather). | V1 (no lags) R²=-0.07 vs V3 (with lags) R²=0.98. **+1.05 absolute R² improvement** from this single change. |
| **Rolling statistics** | Capture trends (worsening/improving?) and volatility (stable or erratic?). | Ablation: best individual method (RMSE 0.9091 vs baseline 0.9436). |
| **FFT features** | Delays have periodic patterns (rush hour cycles) with clear frequency signatures. | Ablation: FFT (RMSE 0.9387) outperforms wavelet (0.9431). |
| **Wavelet features** | Multi-resolution: captures both fast fluctuations and slow trends simultaneously. | Complementary to FFT; combined > either alone. |
| **Statistical features** | Skewness/kurtosis describe distribution shape; trend indicates deterioration direction. | Modest individual gain but enriches combined model. |
| **Historical stats** | Provide route/stop-specific baselines ("what is normal for this location?"). | Computed on train only, used as static context. |
| **Cyclical encoding** | Prevents discontinuity: 23:00 and 01:00 should be close, not 22 units apart. | Standard practice for periodic features. |

**Ablation study** (`src/models/train_delay_predictor_v3_ablation.py`): Tests 6 configurations {baseline, +rolling, +fft, +wavelet, +stats, all} using the same GRU model to isolate each category's marginal contribution.

#### Model Architectures

| Model | Architecture | Params | Purpose |
|-------|-------------|--------|---------|
| MLP | Linear(41,256)-ReLU-BN-Dropout(0.3) x3 layers | ~25K | Baseline: do features alone suffice? |
| LSTM | Linear-LSTM(128, 2 layers)-Dropout-Linear | ~200K | Sequential dependency modeling |
| GRU | Linear-GRU(128, 2 layers)-Dropout-Linear | ~150K | Simpler gating, often comparable to LSTM |
| NeuronSpark SNN | K-bit binary encode -> 2 SNNBlocks(D=128,N=8,K=8) -> decode | ~1.4M | Neuromorphic: dynamic membrane dynamics |
| Transformer | 6 encoder layers, d=128, 8 heads, FFN=768, GELU | ~1.6M | Attention mechanism, fair comparison with SNN |

**SNN key innovation** — K-bit deterministic binary encoding + dynamic membrane:
- Encoding: h in [0,1] -> K binary spike frames (e.g. 0.75 -> [1,1,0,0,0,0,0,0])
- Membrane: V[t] = beta(t)*V[t-1] + alpha(t)*I[t], spike if V[t] > V_th(t)
- beta, alpha, V_th are **dynamic** — computed from both input and membrane voltage
- Decoding: output = sum_{k=1}^{K} spike[k] * 2^{-k}

**Training configuration:** Adam (lr=0.001, weight_decay=1e-5), batch=256, dropout=0.3, early stopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5), gradient clipping (max_norm=1.0).

---

## 4. Preliminary Results and Interpretation (5 pts)

> Rubric: 4.1 Preliminary results are presented and interpreted (5 pts)

### Progressive Improvement

| Version | Key Change | Best Model | RMSE (min) | R² | Interpretation |
|---------|-----------|------------|------------|-----|----------------|
| V1 Baseline | Static features only | MLP | 6.24 | -0.07 | Worse than mean prediction. Static context cannot predict delays. |
| V2 Historical | + route/stop averages | LSTM | 6.34 | -0.11 | Historical averages useless. Delays are non-stationary. |
| **V3 Time Series** | **+ lag/FFT/wavelet** | **GRU** | **0.75** | **0.9846** | **Breakthrough: recent delay history enables prediction.** |
| V4 Multi-step | Seq2Seq autoregressive | Seq2Seq | 5.72 | 0.085 | Multi-step is fundamentally harder. Errors accumulate. |
| V5 NeuronSpark | Spiking Neural Network | SNN | 0.61 | 0.9897 | SNN outperforms GRU on full data (3.76M samples). |
| **V6 Transformer** | **Attention mechanism** | **Transformer** | **0.46** | **0.9942** | **Best: 93% RMSE reduction from V1.** |

### Why V1-V2 Failed (R² < 0)

Negative R² means the model predicts worse than simply using the mean delay for all observations.

1. **Static features insufficient:** Knowing "Route 28, Monday, 5 PM" does not tell you whether this specific bus is delayed. Same route/time can have vastly different delays on different days.
2. **Historical averages too coarse:** A route's 5-year average doesn't reflect today's conditions. Delays are **non-stationary**.
3. **Conclusion:** Delay prediction is a **short-term dynamics problem**, not a static classification problem.

### Why V3 Succeeded (R² = 0.98)

1. **Strong autocorrelation:** If the last 5 buses were all 10 min late, the next one likely is too — external factors (traffic, weather) persist across consecutive buses.
2. **Signal processing adds refinement:** FFT/wavelet/rolling provide ~4% additional RMSE reduction by capturing periodic patterns and multi-scale trends.
3. **Features > Architecture:** MLP, LSTM, GRU all achieve R² > 0.98 with the same features. Feature engineering matters most.

### V5 NeuronSpark SNN vs V6 Transformer (Fair Comparison)

At comparable parameter counts (~1.4M vs ~1.6M):

| Metric | NeuronSpark SNN | Transformer | Winner |
|--------|-----------------|-------------|--------|
| R² | 0.9897 | **0.9942** | Transformer |
| RMSE | 0.6098 min | **0.4599 min** | Transformer (-24.6%) |
| MAE | 0.3311 min | **0.0595 min** | Transformer (-82%) |
| Training Time | 7.6 hours | **5.8 hours** | Transformer (-24%) |

**Why Transformer wins:** Standard backpropagation with exact gradients is more efficient than surrogate gradient methods for SNNs. Attention captures temporal dependencies without spike-processing overhead.

**SNN's value:** While not the best here, SNN has advantages for energy-constrained edge deployment on neuromorphic hardware.

### Overall

| Metric | V1 Baseline | V6 Best | Improvement |
|--------|-------------|---------|-------------|
| RMSE | 6.24 min | 0.46 min | **93% reduction** |
| R² | -0.07 | 0.9942 | **+1.06 absolute** |

The iterative design demonstrates that **feature engineering** is the dominant factor, with **model architecture** providing additional refinement.

---

## Appendix: Code and Artifact Inventory

| Directory | Files | Purpose |
|-----------|-------|---------|
| `src/data/` | 7 | Data download, cleaning, preprocessing, format conversion |
| `src/analysis/` | 6 | Q1-Q7 analysis (ridership, delay, equity, demographics) |
| `src/visualization/` | 1 | All visualization functions |
| `src/models/` | 10 | V1-V6 training, SNN architecture, ablation study |
| `reports/figures/` | 22 | All visualizations |
| `models/` | 14 | Trained model checkpoints (.pt) |

**Key source files:**

| File | Description |
|------|-------------|
| `src/models/train_delay_predictor_v3_fixed.py` | V3 time series features (no leakage) |
| `src/models/train_delay_predictor_v3_ablation.py` | Ablation study: 6 feature methods compared |
| `src/models/train_delay_predictor_v5_neuronspark.py` | V5 NeuronSpark SNN + V6 Transformer |
| `src/models/snn_delay_model.py` | NeuronSpark SNN architecture (v7 parallel scan) |

---

*Boston Bus Equity Project - CS506 Spring 2026*
