# Boston Bus Equity - Project Implementation Plan

## Phase 1: Data Collection & Infrastructure

### 1.1 Data Download
- [ ] MBTA Bus Arrival/Departure Times (2018-2024) - Training data
- [ ] MBTA Bus Arrival/Departure Times (2025) - Validation data
- [ ] MBTA Bus Ridership by Route/Stop
- [ ] MBTA 2022-2024 Passenger Survey (Pooled)
- [ ] Boston 2020 Census Data
- [ ] ACS 2020-2024 5-Year Estimates

### 1.2 Data Storage Structure
```
data/
├── raw/
│   ├── arrival_departure/
│   │   ├── 2018/
│   │   ├── 2019/
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   ├── 2024/
│   │   └── 2025/          # Validation set
│   ├── ridership/
│   ├── survey/
│   └── census/
├── processed/
│   ├── train/             # 2018-2024
│   └── validation/        # 2025
└── external/
```

## Phase 2: Data Preprocessing

### 2.1 Data Cleaning
- Handle missing values
- Remove duplicates
- Standardize date/time formats
- Validate data integrity

### 2.2 Feature Engineering
- Calculate delay times (actual - scheduled)
- Aggregate by route, stop, time period
- Merge with demographic data by geographic area

### 2.3 Data Integration
- Link bus stops to census tracts
- Map routes to neighborhoods
- Join ridership with demographic characteristics

## Phase 3: Base Questions Analysis

### Q1: Ridership per Route (Pre vs Post Pandemic)
- **Data**: Ridership data 2018-2024
- **Method**: Time series analysis, before/after comparison
- **Output**: Line charts, summary statistics

### Q2: End-to-End Travel Times
- **Data**: Arrival/departure times
- **Method**: Calculate total trip duration per route
- **Output**: Distribution plots, route comparison table

### Q3: Average Wait Time (On-time vs Delayed)
- **Data**: Arrival/departure times + schedules
- **Method**: Compare scheduled vs actual arrival
- **Output**: Histogram, summary by route

### Q4: Average Delay Time (City-wide)
- **Data**: All routes arrival/departure
- **Method**: Aggregate delay statistics
- **Output**: Summary statistics, trend over time

### Q5: Target Routes Delay Analysis
- **Routes**: 22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14
- **Data**: Filtered arrival/departure for target routes
- **Method**: Detailed delay analysis per route
- **Output**: Comparative bar charts, heatmaps

### Q6: Service Level Disparities
- **Data**: Delay data by route
- **Method**: Statistical comparison across routes
- **Output**: Ranking, disparity metrics

### Q7: Demographic Impact Analysis
- **Data**: Delay data + Census/ACS demographics
- **Method**: Correlation analysis, geographic mapping
- **Output**: Choropleth maps, demographic breakdown charts

## Phase 4: Validation with 2025 Data

### 4.1 Model Validation
- Apply patterns discovered from 2018-2024 to predict 2025 metrics
- Compare predictions with actual 2025 data
- Calculate validation metrics (MAE, RMSE, etc.)

### 4.2 Trend Verification
- Verify if identified trends continue in 2025
- Check if equity disparities persist
- Document any significant changes

## Phase 5: Visualization & Reporting

### 5.1 Required Visualizations (Minimum 5-7)
1. Ridership trends over time (pre/post pandemic)
2. Delay time distribution by route
3. Target routes performance comparison
4. Geographic heatmap of delays
5. Demographic disparity analysis
6. Service level comparison across routes
7. Validation results (2025 predictions vs actuals)

### 5.2 Final Deliverables
- Clean datasets (uploaded to Google Drive & GitHub)
- All code with documentation
- Final Report with visualizations
- Presentation slides

## Timeline Milestones

1. **Data Review + Initial Questions** - Data collection complete
2. **Data Preprocessing** - Clean data ready for analysis
3. **Early Insights Presentation** - Initial findings
4. **Base Questions Answered** - All 7 questions addressed
5. **Final Report** - Complete documentation
6. **End of Semester Showcase** - Final presentation
