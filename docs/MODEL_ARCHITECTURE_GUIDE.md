# Model Architecture Guide — Boston Bus Equity Delay Prediction

This document describes every model version developed for the Q8 delay prediction task, from the
V1 baseline to the V6 Transformer. It is intended to help readers understand *what* each model
does, *why* it was built, and *what we learned* from it.

All models share the same temporal train/test split:

| Split | Years | Records |
|-------|-------|---------|
| Training | 2020–2024 | ~121 M raw / 3.76 M feature-extracted |
| Validation (V4) | 2025 | ~28 M raw |
| Test | 2025–2026 | ~28 M raw |

---

## V1 — Static Baseline

**Goal:** Establish a lower bound. Can route/time context alone predict delays?

**Input features (10 dims):**

```
hour, day_of_week, month          — calendar context
is_weekend, is_rush_hour          — binary flags (rush = 7–9, 16–19)
hour_sin, hour_cos                — cyclical hour encoding
dow_sin, dow_cos                  — cyclical day-of-week encoding
route_enc                         — label-encoded route id
stop_enc                          — label-encoded stop id
```

**Architecture:**

```
Input(10) → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(1)
```

| Architecture | Parameters | RMSE (min) | MAE (min) | R² |
|-------------|-----------|------------|-----------|-----|
| MLP | ~25 K | 6.24 | 4.38 | −0.07 |
| LSTM | ~200 K | 6.32 | 4.43 | −0.10 |
| GRU | ~150 K | 6.58 | 4.59 | −0.19 |

**Finding:** R² < 0 means the model is worse than a constant mean predictor.
Static context cannot capture whether *this specific bus right now* is running late.
Source: `src/models/train_delay_predictor.py`

---

## V2 — Historical Statistics Baseline

**Goal:** Add route/stop-level historical averages. Does knowing the typical delay for
a given route and stop help?

**Additional features over V1:**

```
route_delay_mean, route_delay_std      — computed on train only, applied to both
stop_delay_mean, stop_delay_std        — same principle
route_stop_delay_mean                  — combined route-stop pair
```

| Architecture | Parameters | RMSE (min) | MAE (min) | R² |
|-------------|-----------|------------|-----------|-----|
| MLP | ~25 K | 6.37 | 4.41 | −0.12 |
| LSTM | ~200 K | 6.34 | 4.37 | −0.11 |
| GRU | ~150 K | 6.36 | 4.40 | −0.11 |

**Finding:** Historical averages do not help. Delays are non-stationary: the mean over
five years tells you nothing about today's congestion.
Source: `src/models/train_delay_predictor_v2.py`

The V2 MLP checkpoint (`delay_predictor_mlp_v2_lag_features_temporal.pt`) is kept as
the **realtime bundle base** because its 18 causal features (route/stop/direction
encoders + time flags + historical stats) are fully online-safe (computable before the
bus arrives), even though the offline test score is poor.

---

## V3 — Time Series Feature Engineering

**Goal:** The real breakthrough. Provide the model with recent delay history.

**Feature categories (41 total dims):**

### Lag Features (7 dims)
```python
lag_1 = delay[i-1]          # Most recent delay
lag_2 = delay[i-2]
lag_3 = delay[i-3]
lag_4 = delay[i-4]
lag_5 = delay[i-5]
diff_1 = delay[i-1] - delay[i-2]   # 1st-order difference
diff_2 = delay[i-1] - delay[i-3]   # 2nd-order difference
```

### Rolling Statistics (8 dims)
```python
# Windows: w ∈ {5, 10}, computed on past values only
roll_mean_w  = mean(delay[i-w : i])
roll_std_w   = std(delay[i-w : i])
roll_min_w   = min(delay[i-w : i])
roll_max_w   = max(delay[i-w : i])
```

### FFT Features (6 dims)
```python
# Discrete Fourier Transform on 10-step historical window
window = delay[i-10 : i]
X = numpy.fft.fft(window)
# Extract top-3 magnitudes and their corresponding frequencies
```

Mathematical form: X[k] = Σ d_n · e^(−j2πkn/N), k = 0..N−1

### Wavelet Features (6 dims)
```python
import pywt
coeffs = pywt.wavedec(window, 'db4', level=2)
# cA2: level-2 approximation (slow trend)
# cD2: level-2 detail (medium oscillation)
# cD1: level-1 detail (fast noise)
# Extract mean and std per level → 3 × 2 = 6 features
```

### Statistical Features (4 dims)
```python
skewness   = scipy.stats.skew(window)
kurtosis   = scipy.stats.kurtosis(window)
trend      = linear_regression_slope(window)   # positive = worsening
volatility = std(diff(window))                 # erratic-ness
```

### Historical Statistics (5 dims, train-computed)
```python
route_delay_mean, route_delay_std   # route-level profile
stop_delay_mean,  stop_delay_std    # stop-level profile
hour_delay_mean                     # hourly baseline
```

### Context + Time (9 dims)
```python
is_weekend, is_rush_hour            # binary flags
route_enc, stop_enc, direction_enc  # LabelEncoder
hour_sin, hour_cos                  # cyclical hour
dow_sin,  dow_cos                   # cyclical day-of-week
```

**Architecture (shared across MLP / LSTM / GRU):**

```
                MLP                      LSTM / GRU
Input(41) → Linear(256) → BN → ReLU   Input(41) → Linear(128)
          → Linear(128) → BN → ReLU             → LSTM/GRU(128, 2 layers)
          → Linear(64)  → BN → ReLU             → Dropout(0.3)
          → Dropout(0.3) → Linear(1)            → Linear(64) → Linear(1)
```

| Architecture | Parameters | RMSE (min) | MAE (min) | R² |
|-------------|-----------|------------|-----------|-----|
| MLP | ~25 K | 0.79 | 0.29 | 0.9830 |
| LSTM | ~200 K | 0.80 | 0.29 | 0.9822 |
| **GRU** | **~150 K** | **0.75** | **0.18** | **0.9846** |

**Ablation study** (GRU, 500 K training samples):

| Feature set | RMSE | ∆ vs baseline |
|-------------|------|---------------|
| Baseline (lags only) | 0.9436 | — |
| + Rolling statistics | 0.9091 | −3.7% ← best individual |
| + FFT | 0.9387 | −0.5% |
| + Wavelet | 0.9431 | −0.05% |
| + Statistical moments | 0.9482 | +0.5% alone |
| **All combined** | **0.9056** | **−4.0%** |

**Key lesson:** Features matter more than architecture. MLP and GRU both hit R² ≈ 0.98.
Source: `src/models/train_delay_predictor_v3_fixed.py`, `train_delay_predictor_v3_ablation.py`

---

## V4 — Multi-step Seq2Seq and Tabular Tree Ensemble

V4 splits into two distinct experiments with the same version label.

### V4a — Seq2Seq Autoregressive (Multi-step Prediction)

**Goal:** Predict a sequence of future delays, not just the next one.

```
Encoder:   GRU(input=1, hidden=128, layers=2)
           Reads 10 past delay values
           Produces context vector h

Decoder:   GRU(input=1, hidden=128, layers=2)
           Autoregressively produces horizon=1/3/5 steps
           Step t input = prediction at step t-1

Output:    Linear(128, 1) per step
```

| Horizon | RMSE (min) | MAE (min) | R² |
|---------|------------|-----------|-----|
| 1 step | 5.81 | 3.94 | 0.116 |
| 3 steps | 5.72 | 3.90 | 0.085 |
| 5 steps | 5.81 | 3.95 | 0.072 |

**Finding:** Multi-step prediction is fundamentally harder. Without rich engineered
features feeding the decoder, error accumulates and R² collapses to ~0.08.
Source: `src/models/train_delay_predictor_v4_multistep.py`

### V4b — LightGBM/CatBoost/XGBoost Online-Safe Sweep

**Goal:** Build a deployable production model using gradient boosting on causal tabular features.

**Training protocol:**
- Train on 2024 data (or 2024+2025 for final retrain)
- Validate on 2025 data
- Test on 2026 data (held out throughout)

**Feature profile `v2_core` (18 online-safe dims):**

```
route_enc, stop_enc, direction_enc    — entity encodings
hour, day_of_week, month              — calendar context
is_weekend, is_rush_hour              — binary flags
scheduled_headway                     — planned gap between trips
route_delay_mean, route_delay_std     — route historical profile
stop_delay_mean,  stop_delay_std      — stop historical profile
hour_delay_mean                       — hour baseline
route_stop_delay_mean                 — combined pair baseline
```

All features are available *before* the bus departs — no future leakage.

**Sweep results (best candidates):**

| Model | Feature profile | 2026 MAE (min) | Score |
|-------|----------------|----------------|-------|
| LightGBM `v2_core` | v2_core | 3.852 | — |
| **LightGBM q35 `v2_core`** | v2_core | **3.939** | **Best score** |
| HistGradientBoosting | v2_core | 4.012 | — |
| Dummy median | — | 4.038 | Baseline |
| Historical baseline | v2_core | 4.418 | — |

**Deployability score** (40% MAE, 15% year-to-year stability, 15% online readiness,
20% early/negative-delay behavior, 10% training cost):

The selected dashboard model is **LightGBM quantile 0.35 with `v2_core` features**:
- 2026 MAE: 3.939 min (beats dummy baseline 4.038)
- Early-delay F1: 0.373
- Negative prediction rate: 17.1% (vs ~0% for a naive positive-bias model)
- Bundle: `models/delay_predictor_v4_score_best_online_safe_bundle.joblib`

Source: `src/models/train_delay_predictor_v4.py`, `src/models/sweep_delay_predictor_v4.py`

---

## V5 — NeuronSpark Spiking Neural Network

**Goal:** Apply neuromorphic computing to transportation time-series regression.

### Architecture

```
Input(14) ─── Encoder ───► [spike frames, K=8 timesteps per input]
                              │
                    ┌─────────▼─────────┐
                    │     SNNBlock 1     │
                    │  D=128, N=8, K=8  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │     SNNBlock 2     │
                    │  D=128, N=8, K=8  │
                    └─────────┬─────────┘
                              │
                           Decoder
                              │
                           Output(1)
```

**Binary encoding:** Continuous input h ∈ [0,1] → K binary spike frames via
MSB-first fixed-point representation.  
Example: h=0.75 → [1, 1, 0, 0, 0, 0, 0, 0]

**SNNBlock internals (6 parallel pathways per block):**

```
I[t]   = W_in · spike[t]                           (input current)
β(t)   = σ(W_β · spike[t] + W_β^V · V[t] + b_β)   (dynamic decay)
α(t)   = softplus(W_α · spike[t] + W_α^V · V[t] + b_α)  (write gain)
V_th(t)= V_min + |W_th · spike[t] + W_th^V · V[t] + b_th|  (threshold)
gate(t)= σ(W_gate · spike[t])                       (gating)
skip(t)= W_skip · spike[t]                          (skip connection)

# SelectivePLIF membrane dynamics:
V[t]   = β(t) · V[t-1] + α(t) · I[t]
s[t]   = Heaviside(V[t] - V_th(t))                 (spike)
V[t]  -= V_th(t) · s[t]                            (soft reset)
```

**Binary decoding:** output = Σ_{k=1}^{K} spike[k] · 2^{−k}

**Model configurations:**

| Config | D | N | K | Blocks | Parameters |
|--------|---|---|---|--------|-----------|
| NeuronSpark-D64-K8 | 64 | 8 | 8 | 2 | 348,547 |
| **NeuronSpark-D128-K8** | **128** | **8** | **8** | **2** | **1,384,835** |

**Full-dataset results (3.76 M samples, 13 epochs, ~3.4 h on MPS GPU):**

| Model | RMSE (min) | MAE (min) | R² |
|-------|------------|-----------|-----|
| GRU baseline | 0.6384 | 0.1855 | 0.9893 |
| **NeuronSpark-D128-K8** | **0.6098** | **0.3311** | **0.9897** |

**Key insight:** SNN requires large training sets. At 50 K samples, GRU wins. At
3.76 M samples, SNN surpasses GRU — its complex multi-timescale membrane dynamics
learn meaningful patterns only with sufficient data.

**Training configuration:**

```
Optimizer   : Adam (lr=0.001, weight_decay=1e-5)
Batch size  : 256
Dropout     : 0.3
Max epochs  : 50 (early stopping, patience=10)
LR schedule : ReduceLROnPlateau (factor=0.5, patience=5)
Grad clip   : max_norm=1.0
Checkpoint  : models/delay_neuronspark_v5_quick.pt
```

Source: `src/models/train_delay_predictor_v5_neuronspark.py`, `src/models/snn_delay_model.py`,
`src/models/train_v5_v6_quick.py`

---

## V6 — Transformer (State-of-the-Art)

**Goal:** Fair comparison with V5 at similar parameter scale using standard attention.

### Architecture

```
Input(14) → Linear(128) → Positional Encoding (learnable)
                         │
              ┌──────────▼──────────┐
              │  TransformerEncoder  │
              │  6 layers            │
              │  d_model = 128       │
              │  nhead   = 8         │
              │  dim_ffn  = 768      │
              │  activation = GELU   │
              │  dropout   = 0.1     │
              └──────────┬──────────┘
                         │
                    Linear(128, 1)
                         │
                      Output
```

Total parameters: **1,595,649**

**Full-dataset results (3.76 M samples, 21 epochs, ~2.6 h on MPS GPU):**

| Model | Parameters | RMSE (min) | MAE (min) | R² |
|-------|-----------|------------|-----------|-----|
| NeuronSpark-D128-K8 | 1,384,835 | 0.6098 | 0.3311 | 0.9897 |
| **Transformer-6L-128d** | **1,595,649** | **0.4599** | **0.0595** | **0.9942** |

Transformer advantage over SNN: RMSE −24.6%, MAE −82%, R² +0.0045, training −24% time.

**Why Transformer wins offline:**
1. Exact gradients (no surrogate approximation for spikes).
2. Self-attention captures global temporal context natively.
3. Continuous activations (GELU) yield smoother optimization landscape.
4. No K-bit quantization error in the encoding step.

**Training configuration:**

```
Optimizer   : Adam (lr=0.001, weight_decay=1e-5)
Batch size  : 256
Dropout     : 0.1
Max epochs  : 50 (early stopping, patience=10)
LR schedule : ReduceLROnPlateau (factor=0.5, patience=5)
Grad clip   : max_norm=1.0
Checkpoint  : models/delay_transformer_v6_quick.pt
```

Source: `src/models/train_v5_v6_quick.py`

---

## Head-to-Head Summary

| Version | Best R² | Best RMSE | Params | Key insight |
|---------|---------|-----------|--------|-------------|
| V1 Static | −0.07 | 6.24 | ~25 K | Calendar context insufficient |
| V2 Historical | −0.11 | 6.34 | ~25 K | Long-run averages insufficient |
| V3 GRU + features | **0.9846** | **0.75** | ~150 K | Lag + FFT + wavelet → breakthrough |
| V4a Seq2Seq | 0.116 | 5.72 | ~150 K | Multi-step accumulates error |
| V4b LightGBM | 3.939 MAE | — | — | Online-safe production model |
| V5 SNN | 0.9897 | 0.61 | 1.4 M | Neuromorphic competitive at scale |
| **V6 Transformer** | **0.9942** | **0.46** | 1.6 M | Best offline; worst on live data |

**Overall RMSE progression (best model per version):**

```
V1   ████████████████████████████████████████████████████  6.24 min
V2   ████████████████████████████████████████████████████  6.34 min
V4a  ████████████████████████████████████████████████  5.72 min
V3   █████  0.75 min
V5   ████  0.61 min
V6   ███  0.46 min
     ─────────────────────────────────────────────────────
     (93% total RMSE reduction, V1 → V6)
```

---

## Live vs Offline Ranking Inversion (April 2026 Finding)

When the V3/V5/V6 models were connected to the MBTA V3 live predictions API, the
offline ranking inverted completely:

| Rank | Offline (test R²) | Live (mean abs gap vs MBTA) |
|------|------------------|---------------------------|
| 1st | V6 Transformer (0.9940) | V3 GRU + wavelet (5.06 min) |
| 2nd | V5 SNN (0.9897) | V5 SNN (8.54 min) |
| 3rd | V3 GRU (0.9846) | V6 Transformer (8.80 min) |

**Root cause (three controlled experiments):**

1. **Distribution shift.** Training mean = +5.6 min (late); live mean = −1.8 min (early).
   78% of live trips run early vs 19% in training.

2. **V5/V6 are lag amplifiers.** On synthetic constant-lag inputs, both V5 and V6 output
   within ±0.4 min of the input. They learned "next ≈ recent" and suppress time-of-day.

3. **V3 is robust.** FFT/wavelet features compress noisy lag spikes into smooth low-frequency
   components. V3's GRU outputs near its training mean regardless of extreme live lag inputs.

**Unified explanation:** Bias-variance tradeoff between clean offline data and noisy live data.

---

## Realtime Inference System

The inference layer (`src/inference/`) connects all model checkpoints to a FastAPI web service:

```
Browser / curl
      │
      ▼
FastAPI (serve.py)
      │
      ├── GET  /           → dashboard (index.html + dashboard.js)
      ├── GET  /health     → bundle metadata + model info
      ├── POST /predict    → single-trip delay prediction
      ├── GET  /live-compare → MBTA V3 API + local model side-by-side
      └── GET  /figures/*  → project visualization PNGs
            │
            ▼
      runtime.py (ModelRuntime)
            │
            ├── V2 bundle (.pt) ──► PyTorch MLP forward
            └── V4 bundle (.joblib) → LightGBM predict
                  │
                  ▼
            MBTARealtimeAdapter
            (route alias normalization, lag cache,
             MBTA V3 /predictions + /vehicles API)
```

**Supported bundle formats:**

| Format | Models | File |
|--------|--------|------|
| `.pt` PyTorch bundle | V2 MLP (realtime features) | `delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt` |
| `.joblib` sklearn/LightGBM bundle | V4b LightGBM (v2_core features) | `delay_predictor_v4_score_best_online_safe_bundle.joblib` |

The dashboard also exposes all six V1–V6 checkpoints via a model picker for comparison.
The V4b LightGBM bundle is the default because it is the only model with an online-safe
production-quality feature pipeline and deployability score validation.

---

*Boston Bus Equity Project — CS506 Spring 2026*  
*See `reports/DELAY_PREDICTION_COMPARISON_REPORT.md` for full numeric tables.*  
*See `reports/MARCH_CHECKIN_PRESENTATION.md` (Section 0) for the April live-vs-offline analysis.*
