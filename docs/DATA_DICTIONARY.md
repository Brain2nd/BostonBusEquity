# Data Dictionary — Boston Bus Equity

This document describes every dataset used in the project: field names, data types,
value ranges, and known quality issues. It is intended as a reference for anyone
running the analysis pipeline or adding new models.

---

## 1. MBTA Bus Arrival / Departure Times

**Primary source for all delay labels and Q4–Q8 analysis.**

| Source | MBTA Open Data Portal |
|--------|----------------------|
| Portal URL | https://mbta-massdot.opendata.arcgis.com/search?tags=bus |
| Years available | 2020–2026 (2018–2019 no longer published) |
| Format | CSV per year, converted to `arrival_departure.parquet` |
| Raw size | ~18 GB, 161 M records total |
| Download script | `src/data/download_data.py` |
| Conversion script | `src/data/convert_all_to_parquet.py` |

### 1.1 Field Reference

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `service_date` | `DATE` (YYYY-MM-DD) | Operating date. A trip starting after midnight still uses the prior calendar date. | `2023-06-15` |
| `route_id` | `STRING` | MBTA route identifier. May include leading zeros (`"001"` = Route 1). Normalized to strip leading zeros in model pipelines. | `"1"`, `"22"`, `"111"` |
| `trip_id` | `STRING` | Unique trip instance within a service day. Format `{route_id}-{service_date}-{sequence}`. | `"1-20230615-0041"` |
| `direction_id` | `INT` (0 or 1) | Inbound (1) or outbound (0), as defined per route. | `0`, `1` |
| `stop_id` | `STRING` | MBTA stop identifier. Numeric string, may have `.0` suffix in some exports. Normalized to plain integer string. | `"110"`, `"5547"` |
| `stop_sequence` | `INT` | Ordered position of this stop within the trip. Used to determine which stop comes before/after for lag features. | `1`, `7`, `23` |
| `scheduled` | `DATETIME` (UTC or local) | Scheduled arrival/departure time, ISO8601. Needs parsing with `pandas.to_datetime`. | `2023-06-15T08:15:00` |
| `actual` | `DATETIME` (UTC or local) | Actual arrival/departure time recorded by MBTA sensors. `NULL` if the trip was cancelled or data was not recorded. | `2023-06-15T08:22:37` |
| `scheduled_headway` | `FLOAT` (minutes) | Planned time gap from the previous trip at the same stop. Missing in some early years; imputed with route/hour median. | `12.0` |
| `actual_headway` | `FLOAT` (minutes) | Actual time gap from previous trip. Available in some years; not used as a model feature (future leakage risk for stateless requests). | `18.5` |

### 1.2 Derived Field: `delay_minutes`

```python
delay_minutes = (actual - scheduled).total_seconds() / 60
```

- Positive: bus is late.
- Negative: bus arrived early (early departures are recorded).
- `NULL` if either `actual` or `scheduled` is missing.

**Model training range:** `−30 ≤ delay_minutes ≤ 60`. Values outside this range are
treated as data artifacts (cancelled trips re-opened, sensor misfires, midnight
wraparound parsing errors) and dropped before training.

| Range | Share of records | Action |
|-------|-----------------|--------|
| < −30 min | < 0.1% | Dropped — sensor artifact |
| −30 to 0 min | ~14.8% | Kept — early arrivals |
| 0 to 5 min | ~16.9% | Kept — on time |
| 5 to 15 min | ~30.6% | Kept — minor/moderate delay |
| 15 to 60 min | ~37.2% | Kept — major delay |
| > 60 min | ~0.4% | Dropped — cancelled/resumed trip |

### 1.3 Record Counts by Year

| Year | Raw records | After cleaning |
|------|-------------|----------------|
| 2020 | 19,197,828 | ~18.4 M |
| 2021 | 28,916,111 | ~27.8 M |
| 2022 | 28,301,238 | ~27.2 M |
| 2023 | 27,095,791 | ~26.1 M |
| 2024 | 27,049,203 | ~26.0 M |
| 2025 | 28,115,881 | ~27.0 M |
| 2026 | ~2,439,311 | ~2.3 M (partial year) |

### 1.4 Known Quality Issues

1. **Missing 2018–2019 data.** These years are no longer published on the MBTA portal.
   The project uses 2020 as the earliest year for delay analysis.
2. **Partial 2026 data.** As of the project cut-off, 2026 data covers January only.
3. **Leading zero inconsistency.** Route `"001"` and `"1"` refer to the same route.
   All pipelines normalize via `str(int(x)).strip()`.
4. **Stop ID decimal suffix.** Some CSV exports include `"110.0"` instead of `"110"`.
   Normalized by stripping `.0` suffixes.
5. **Timezone inconsistency.** Some years ship in local Eastern time (EST/EDT); others
   in UTC. The preprocessing pipeline converts all timestamps to UTC-aware before
   computing delay.

---

## 2. MBTA Bus Ridership by Trip/Season

**Primary source for Q1 (ridership pre/post pandemic) analysis.**

| Source | MBTA Open Data Portal |
|--------|----------------------|
| Years available | 2016–2024 |
| Format | CSV per year |
| Raw size | ~850 MB, 1.8 M records |
| Download script | `src/data/download_data.py` |

### 2.1 Field Reference

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `service_date` | `DATE` | Operating date | `2019-03-14` |
| `route_id` | `STRING` | MBTA route identifier | `"22"` |
| `direction_id` | `INT` | Inbound (1) / outbound (0) | `0` |
| `trip_id` | `STRING` | Unique trip identifier | `"22-20190314-0012"` |
| `stop_id` | `STRING` | MBTA stop identifier | `"383"` |
| `load` | `INT` | Passenger count at this stop | `17` |
| `season` | `STRING` | Service season code | `"Fall_2023"` |

### 2.2 Pre/Post Pandemic Definition

| Period | Years | Avg annual boardings |
|--------|-------|---------------------|
| Pre-pandemic | 2016–2019 | 746,761 |
| Pandemic | 2020 | 363,317 (−51.3%) |
| Post-pandemic | 2021–2024 | 501,474 (−32.8% vs pre) |

---

## 3. MBTA GTFS Stops and Routes

**Used for geographic mapping and route vocabulary.**

| Source | MBTA |
|--------|------|
| Format | Standard GTFS (stops.txt, routes.txt, shapes.txt) |
| Download script | `src/data/download_data.py` |

### 3.1 Field Reference — stops.txt

| Field | Type | Description |
|-------|------|-------------|
| `stop_id` | `STRING` | Unique stop identifier (matches arrival/departure `stop_id`) |
| `stop_name` | `STRING` | Human-readable stop name |
| `stop_lat` | `FLOAT` | WGS84 latitude |
| `stop_lon` | `FLOAT` | WGS84 longitude |
| `wheelchair_boarding` | `INT` | 0=no info, 1=accessible, 2=not accessible |

Boston has **2,910 bus stops** mapped to **22 neighborhoods** using coordinate-based
spatial join to neighborhood polygon boundaries.

---

## 4. Demographic and Census Data

**Used for Q7 (equity analysis) and route-demographic profiling.**

### 4.1 Sources

| Dataset | Source | Coverage |
|---------|--------|----------|
| 2020 Census for Boston | https://data.boston.gov/dataset/2020-census-for-boston | Block-level counts |
| ACS 2020–2024 5-Year Estimates | https://www.census.gov/programs-surveys/acs/ | Tract-level demographics |
| Boston Neighborhood Demographics | Boston Planning & Development Agency | Neighborhood summaries |

### 4.2 Key Fields (Neighborhood Level)

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `neighborhood` | `STRING` | Boston neighborhood name (22 neighborhoods) | BPDA |
| `total_population` | `INT` | Total residents | Census 2020 |
| `minority_pct` | `FLOAT` | % non-white residents | ACS 5-year |
| `hispanic_pct` | `FLOAT` | % Hispanic/Latino residents | ACS 5-year |
| `median_household_income` | `FLOAT` (USD) | Median annual household income | ACS 5-year |
| `poverty_rate` | `FLOAT` | % residents below federal poverty line | ACS 5-year |
| `transit_dependency_rate` | `FLOAT` | % households with no vehicle | ACS 5-year |

### 4.3 Vulnerability Classification

A neighborhood is classified as **"vulnerable"** if it has both:
- `minority_pct > 50%` (high minority concentration), AND
- Income or poverty metric below citywide median.

Six neighborhoods meet both criteria: **Dorchester, East Boston, Hyde Park,
Mattapan, Mission Hill, Roxbury.**

---

## 5. MBTA Passenger Survey (2022–2024)

**Used for supplementary demographic validation in Q7.**

| Source | MBTA 2024 System-Wide Passenger Survey |
|--------|---------------------------------------|
| URL | https://gis.data.mass.gov/datasets/MassDOT::mbta-2024-system-wide-passenger-survey |
| Format | CSV (~50 MB) |
| Download | Manual (see README) |

### 5.1 Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `route_id` | `STRING` | Route surveyed |
| `respondent_race` | `STRING` | Self-reported race/ethnicity |
| `respondent_income` | `STRING` (band) | Household income bracket |
| `trip_purpose` | `STRING` | Work / school / medical / other |
| `frequency_of_use` | `STRING` | Daily / several-per-week / occasional |

---

## 6. MBTA V3 Live API

**Used for realtime inference and the April live-vs-offline comparison.**

| Source | MBTA V3 API |
|--------|-------------|
| Base URL | https://api-v3.mbta.com |
| Auth | Optional `MBTA_API_KEY` env var; limited unauthenticated access allowed |
| Client code | `src/inference/mbta_v3_client.py` |

### 6.1 Endpoints Used

| Endpoint | Fields consumed | Use |
|----------|----------------|-----|
| `/predictions` | `route_id`, `stop_id`, `direction_id`, `arrival_time`, `departure_time`, `trip_id` | Official realtime arrival predictions for upcoming trips |
| `/vehicles` | `trip_id`, `stop_id`, `current_stop_sequence`, `current_status`, `speed` | Live vehicle position, used for enriched lag features |

### 6.2 Official Prediction Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `arrival_time` | `DATETIME` (ISO8601) | MBTA's predicted arrival at the stop |
| `departure_time` | `DATETIME` (ISO8601) | MBTA's predicted departure |
| `trip_id` | `STRING` | Links prediction to a scheduled trip |
| `status` | `STRING` | `"Arriving"`, `"Approaching"`, `null` |

**Important:** MBTA official predictions are **not ground truth**. They are another
model's output. Comparing our model against them measures *disagreement*, not accuracy.
True accuracy requires matching predictions against eventual actual arrival records
from the historical dataset.

---

## 7. Processed Parquet File — `arrival_departure.parquet`

The canonical processed dataset used by all model training and inference scripts.

| Location | `data/processed/arrival_departure.parquet` |
|----------|--------------------------------------------|
| Size | ~3 GB (columnar compression from ~18 GB raw) |
| Created by | `src/data/convert_all_to_parquet.py` |
| Compression ratio | ~5–6× vs raw CSV |
| Read speedup | ~10× vs CSV for column-selective queries |

### 7.1 Schema

| Column | DType | Derived from |
|--------|-------|-------------|
| `service_date` | `date32` | Raw `service_date` |
| `route_id` | `string` | Normalized (leading zeros stripped) |
| `trip_id` | `string` | Raw |
| `direction_id` | `int8` | Raw |
| `stop_id` | `string` | Normalized (decimal suffix stripped) |
| `stop_sequence` | `int16` | Raw |
| `scheduled` | `timestamp[us, UTC]` | Parsed + converted to UTC |
| `actual` | `timestamp[us, UTC]` | Parsed + converted to UTC |
| `delay_minutes` | `float32` | `(actual − scheduled).seconds / 60` |
| `scheduled_headway` | `float32` | Raw (median-imputed where missing) |
| `year` | `int16` | Extracted from `service_date` |
| `hour` | `int8` | Extracted from `scheduled` |
| `day_of_week` | `int8` | 0=Monday, 6=Sunday |
| `month` | `int8` | 1–12 |

---

## 8. Model Bundles

These are serialized files combining model weights, scalers, encoders, and
historical statistics for deployment.

| File | Format | Contents | Used by |
|------|--------|----------|---------|
| `delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt` | PyTorch | V2 MLP weights + scaler + route/stop/direction encoders + route-stop-hour stats | `src/inference/runtime.py` (V2 path) |
| `delay_predictor_v4_best_online_safe_bundle.joblib` | joblib | V4 LightGBM (best MAE) + v2_core feature pipeline | `src/inference/runtime.py` (V4 path) |
| `delay_predictor_v4_score_best_online_safe_bundle.joblib` | joblib | V4 LightGBM quantile 0.35 (best deployability score) + v2_core feature pipeline | Dashboard default |
| `delay_neuronspark_v5_quick.pt` | PyTorch | V5 NeuronSpark SNN weights (full-data retrain) | Dashboard model picker |
| `delay_transformer_v6_quick.pt` | PyTorch | V6 Transformer weights (full-data retrain) | Dashboard model picker |
| `delay_seq2seq_v4_quick.pt` | PyTorch | V4 Seq2Seq GRU weights (multi-step) | Dashboard model picker |

---

## 9. Target Routes (Equity Priority)

The 15 routes identified by the Livable Streets Alliance as serving underserved
communities and requiring equity-focused improvements:

```
14, 15, 17, 22, 23, 24, 26, 28, 29, 31, 33, 42, 44, 45, 111
```

These routes experience on average **10.20 min mean delay** vs **7.22 min** for other
routes (+41%). Their on-time performance is **25.8%** vs **32.3%** system-wide.

---

*Boston Bus Equity Project — CS506 Spring 2026*  
*For pipeline usage, see `README.md`. For model details, see `docs/MODEL_ARCHITECTURE_GUIDE.md`.*
