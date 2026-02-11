"""
Configuration file for Boston Bus Equity Project

Contains constants, target routes, and shared configurations.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Target bus routes from Livable Streets report
# These routes serve predominantly low-income and minority communities
TARGET_ROUTES = [
    "22", "29", "15", "45", "28", "44", "42",
    "17", "23", "31", "26", "111", "24", "33", "14"
]

# Data split configuration
# Note: 2018-2019 data not available on ArcGIS, starting from 2020
TRAIN_YEARS = ["2020", "2021", "2022", "2023", "2024"]
VALIDATION_YEARS = ["2025"]
LATEST_YEAR = "2026"  # Partial data available

# Pandemic period definition
PRE_PANDEMIC_END = "2020-03-01"
POST_PANDEMIC_START = "2021-07-01"

# Time periods for analysis
# Note: Pre-pandemic data limited due to data availability (2020-01 to 2020-03)
TIME_PERIODS = {
    "pre_pandemic": ("2020-01-01", "2020-03-01"),
    "pandemic": ("2020-03-01", "2021-07-01"),
    "post_pandemic": ("2021-07-01", "2024-12-31"),
}

# Delay thresholds (in minutes)
DELAY_THRESHOLDS = {
    "on_time": 1,        # Within 1 minute of scheduled time
    "minor_delay": 5,    # 1-5 minutes late
    "moderate_delay": 10, # 5-10 minutes late
    "major_delay": 15,   # >15 minutes late
}

# Visualization settings
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 150,
    "style": "seaborn-v0_8-whitegrid",
    "color_palette": "Set2",
}

# MBTA API configuration (optional, for real-time data)
MBTA_API_BASE_URL = "https://api-v3.mbta.com"
# Note: API key is optional but recommended for higher rate limits
# Set MBTA_API_KEY environment variable if needed
