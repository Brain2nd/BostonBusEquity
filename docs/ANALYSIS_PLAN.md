# Boston Bus Equity - Analysis Plan

## Data Overview

### Available Data Fields (Arrival/Departure)
| Field | Description |
|-------|-------------|
| service_date | Date of service (YYYY-MM-DD) |
| route_id | Bus route identifier |
| direction_id | Inbound/Outbound |
| half_trip_id | Unique trip identifier |
| stop_id | Bus stop identifier |
| time_point_id | Time point identifier |
| time_point_order | Order of stop in route |
| point_type | Startpoint/Midpoint/Endpoint |
| standard_type | Schedule type |
| scheduled | Scheduled arrival time |
| actual | Actual arrival time |
| scheduled_headway | Planned time between buses |
| headway | Actual time between buses |

### Data Split
- **Training Set**: 2020-2024 (5 years, ~15.2 GB)
- **Validation Set**: 2025-2026 (1+ years, ~3 GB)

### Target Routes (Livable Streets Report)
Routes serving predominantly low-income and minority communities:
`22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14`

---

## Research Questions & Analysis Approach

### Q1: Ridership per bus route (pre vs post pandemic)
**Data Required**: Ridership data (manual download needed)
**Approach**:
- Load ridership data by route and time period
- Define periods: Pre-pandemic (2020-01 to 2020-03), Post-pandemic (2021-07+)
- Calculate ridership change percentage per route
- Visualize with bar charts comparing periods

**Output**: `reports/figures/ridership_comparison.png`

---

### Q2: End-to-end travel times for each route
**Data Required**: Arrival/Departure times
**Approach**:
```
For each route:
    For each trip (half_trip_id):
        start_time = actual time at point_type='Startpoint'
        end_time = actual time at point_type='Endpoint'
        travel_time = end_time - start_time

    avg_travel_time = mean(all trip travel_times)
```
**Metrics**:
- Average travel time per route
- Travel time variability (std dev)
- Peak vs off-peak comparison

**Output**: `reports/figures/travel_times_by_route.png`

---

### Q3: Average wait time (on-time vs delayed)
**Data Required**: Arrival/Departure times (headway field)
**Approach**:
```
For each stop arrival:
    delay = actual - scheduled (in minutes)

    if delay <= 1 minute:
        category = "on_time"
    elif delay <= 5 minutes:
        category = "minor_delay"
    elif delay <= 10 minutes:
        category = "moderate_delay"
    else:
        category = "major_delay"

Expected wait time = headway / 2 + delay_adjustment
```
**Metrics**:
- % on-time arrivals
- Average delay by category
- Expected wait time distribution

**Output**: `reports/figures/wait_time_distribution.png`

---

### Q4: Average delay time - all routes citywide
**Data Required**: Arrival/Departure times
**Approach**:
```
delay_minutes = (actual - scheduled).total_seconds() / 60

Aggregate by:
- Overall average
- By month/year (trend analysis)
- By time of day (rush hour analysis)
- By day of week
```
**Metrics**:
- Mean delay (minutes)
- Median delay
- 90th percentile delay
- % severely delayed (>10 min)

**Output**: `reports/figures/citywide_delay_trends.png`

---

### Q5: Average delay time - target routes
**Data Required**: Arrival/Departure times, filtered to target routes
**Approach**:
```
target_routes = [22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14]

For each target route:
    Calculate same metrics as Q4

Compare target routes vs non-target routes
```
**Metrics**:
- Delay comparison: target vs non-target
- Ranking of target routes by delay
- Statistical significance test

**Output**: `reports/figures/target_routes_delay.png`

---

### Q6: Service level disparities between routes
**Data Required**: Arrival/Departure times
**Approach**:
Define "Service Level Score" combining:
1. On-time performance (% within 1 min of schedule)
2. Average delay
3. Headway consistency (actual vs scheduled headway)
4. Service frequency

```
service_score = w1*on_time_pct + w2*(1/avg_delay) + w3*headway_consistency + w4*frequency
```

**Analysis**:
- Calculate service score per route
- Identify top/bottom performing routes
- Statistical disparity analysis (ANOVA/Kruskal-Wallis)

**Output**: `reports/figures/service_level_comparison.png`

---

### Q7: Demographic impact analysis (Equity)
**Data Required**:
- Service level scores (from Q6)
- Census/ACS demographic data
- Route-to-neighborhood mapping

**Approach**:
```
1. Map routes to census tracts/neighborhoods
2. Get demographic data per area:
   - Race/ethnicity breakdown
   - Median household income
   - Age distribution
   - % below poverty line

3. Correlation analysis:
   - Service score vs % minority population
   - Service score vs median income
   - Service score vs poverty rate

4. Regression model:
   service_score ~ income + minority_pct + poverty_rate + ...
```

**Metrics**:
- Correlation coefficients
- Regression coefficients with significance
- Equity index comparison

**Output**: `reports/figures/equity_analysis.png`

---

## Implementation Plan

### Phase 1: Data Preprocessing (src/data/)
1. `load_data.py` - Load and combine CSV files
2. `clean_data.py` - Handle missing values, parse timestamps
3. `feature_engineering.py` - Calculate delays, categorize, add time features

### Phase 2: Core Analysis (src/analysis/)
1. `delay_analysis.py` - Q3, Q4, Q5 (delay metrics)
2. `travel_time_analysis.py` - Q2 (end-to-end times)
3. `service_level_analysis.py` - Q6 (disparity analysis)
4. `equity_analysis.py` - Q7 (demographic correlation)

### Phase 3: Visualization (src/visualization/)
1. `plot_delays.py` - Delay charts and trends
2. `plot_comparisons.py` - Route comparisons
3. `plot_equity.py` - Equity visualizations
4. `create_dashboard.py` - Interactive dashboard (optional)

### Phase 4: Validation
- Run analysis on 2025-2026 data
- Compare patterns with training period
- Validate model predictions

---

## File Structure
```
src/
├── data/
│   ├── download_data.py      [DONE] - Data download with resume support
│   ├── preprocess.py         [DONE] - Basic preprocessing functions
│   └── load_data.py          [DONE] - Chunked data loading for large files
├── analysis/
│   ├── delay_analysis.py     [DONE] - Q3, Q4, Q5 delay metrics
│   ├── travel_time.py        [DONE] - Q2 end-to-end travel times
│   ├── service_level.py      [DONE] - Q6 service disparities
│   └── equity.py             [DONE] - Q7 demographic analysis
├── visualization/
│   └── plots.py              [DONE] - All visualization functions
├── config.py                 [DONE] - Project configuration
└── run_analysis.py           [DONE] - Main analysis runner
```

---

## Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- matplotlib/seaborn: Visualization
- scipy: Statistical tests
- geopandas: Geographic analysis (for Q7)

---

## Timeline Estimate
This document outlines the analysis approach. Implementation order:
1. Data loading and preprocessing
2. Delay analysis (Q3-Q5) - core metrics
3. Travel time analysis (Q2)
4. Service level analysis (Q6)
5. Equity analysis (Q7) - requires additional demographic data
6. Validation with 2025-2026 data
