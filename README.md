# Boston Bus Equity

A data analysis project examining MBTA bus service performance and its impact on Boston residents, with a focus on equity across different communities.

## Project Overview

Public transport plays a vital role in the quality of life for Massachusetts and Boston residents in terms of economic development, environment, and equity. This project aims to better understand the impact of bus performance on Boston residents by using MBTA bus data to examine service performance trends by geography.

The MBTA serves over 1 million people daily, with an estimated added economic value of $11.5 billion per year for the greater Boston area.

## Key Research Questions

### Base Questions
1. What is the ridership per bus route? How has this changed from pre-pandemic to post-pandemic?
2. What are the end-to-end travel times for each bus route in the city?
3. On average, how long does an individual have to wait for a bus (on time vs. delayed)?
4. What is the average delay time of all routes across the entire city?
5. What is the average delay time of target bus routes (22, 29, 15, 45, 28, 44, 42, 17, 23, 31, 26, 111, 24, 33, 14)?
6. Are there disparities in the service levels of different routes?
7. If there are service level disparities, are there differences in the characteristics of the people most impacted (e.g., race, ethnicity, age, income)?

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

## Data Sources

- **Ridership Data**: [Bus Ridership by Trip, Season, Route/Line, and Stop](https://mbta-massdot.opendata.arcgis.com/)
- **Reliability Data**: [MBTA Bus Arrival Departure Times 2018-2024](https://mbta-massdot.opendata.arcgis.com/)
- **Survey Data**: [MBTA 2023 System-Wide Passenger Survey](https://www.mbta.com/)
- **Census Data**: [2020 Census for Boston](https://data.boston.gov/dataset/2020-census-for-boston)
- **Additional Sources**:
  - [MBTA V3 API](https://www.mbta.com/developers/v3-api)
  - [MBTA Performance Data](https://www.mbta.com/developers)
  - ACS Means of Transportation by Travel Time to Work

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

## References

- [Livable Streets Report](https://www.livablestreets.info/)
- [MBTA Guide to Ridership Data](https://www.mbta.com/)
- [64 Hours Documentary](https://www.youtube.com/)
- [Bus Network Redesign Phase 1](https://www.mbta.com/news/2024-10-07/phase-1-bus-network-redesign-launches-december-15-bring-more-frequent-service)

## Team

Boston University CS506 - Spring 2025

SPARK - HUB - CS506

## License

This project is for educational purposes as part of BU CS506 coursework.
