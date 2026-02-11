"""
Data Loading Module for Boston Bus Equity Project

Handles loading raw CSV data from various year/month formats,
with support for chunked processing of large datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Generator
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_RAW, TARGET_ROUTES,
    TRAIN_YEARS, VALIDATION_YEARS
)

# Base directory for arrival/departure data
ARRIVAL_DEPARTURE_DIR = DATA_RAW / "arrival_departure"


def find_csv_files(years: List[str] = None) -> List[Path]:
    """
    Find all CSV files for specified years.

    The data files are organized in subdirectories by year,
    with varying naming conventions.

    Args:
        years: List of years to find files for. If None, find all.

    Returns:
        List of Path objects for CSV files.
    """
    if years is None:
        years = TRAIN_YEARS + VALIDATION_YEARS

    csv_files = []

    for year in years:
        # Look for year-specific subdirectories
        year_patterns = [
            f"MBTA_Bus_Arrival_Departure_Times_{year}",
            f"*{year}*"
        ]

        for pattern in year_patterns:
            for subdir in ARRIVAL_DEPARTURE_DIR.glob(pattern):
                if subdir.is_dir():
                    csv_files.extend(sorted(subdir.glob("*.csv")))

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in csv_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return sorted(unique_files)


def parse_mbta_datetime(date_col: pd.Series, time_col: pd.Series) -> pd.Series:
    """
    Parse MBTA datetime format efficiently.

    The data has service_date (YYYY-MM-DD) and time as 1900-01-01T...
    We need to combine the actual date with the time portion.

    Args:
        date_col: Series with service dates (YYYY-MM-DD)
        time_col: Series with times (1900-01-01THH:MM:SSZ format)

    Returns:
        Series with combined datetime
    """
    # Parse date with explicit format
    dates = pd.to_datetime(date_col, format='%Y-%m-%d', errors='coerce')

    # Extract time components from ISO format (1900-01-01THH:MM:SSZ)
    # More efficient: extract HH:MM:SS directly from string
    time_str = time_col.astype(str).str.extract(r'T(\d{2}):(\d{2}):(\d{2})')
    hours = pd.to_numeric(time_str[0], errors='coerce').fillna(0).astype(int)
    minutes = pd.to_numeric(time_str[1], errors='coerce').fillna(0).astype(int)
    seconds = pd.to_numeric(time_str[2], errors='coerce').fillna(0).astype(int)

    # Combine using timedelta (vectorized, much faster)
    combined = dates + pd.to_timedelta(hours, unit='h') + \
               pd.to_timedelta(minutes, unit='m') + \
               pd.to_timedelta(seconds, unit='s')

    return combined


def load_single_file(file_path: Path,
                     target_routes_only: bool = False) -> pd.DataFrame:
    """
    Load a single CSV file with proper parsing.

    Args:
        file_path: Path to CSV file
        target_routes_only: If True, filter to target routes only

    Returns:
        DataFrame with parsed data
    """
    df = pd.read_csv(file_path, low_memory=False)

    # Convert route_id to string for consistent comparison
    df['route_id'] = df['route_id'].astype(str)

    # Filter to target routes if requested
    if target_routes_only:
        df = df[df['route_id'].isin(TARGET_ROUTES)]

    # Parse datetime columns
    if 'service_date' in df.columns:
        df['service_date'] = pd.to_datetime(df['service_date'], errors='coerce')

    if 'scheduled' in df.columns and 'service_date' in df.columns:
        df['scheduled_datetime'] = parse_mbta_datetime(
            df['service_date'].astype(str),
            df['scheduled']
        )

    if 'actual' in df.columns and 'service_date' in df.columns:
        df['actual_datetime'] = parse_mbta_datetime(
            df['service_date'].astype(str),
            df['actual']
        )

    # Calculate delay in minutes
    if 'scheduled_datetime' in df.columns and 'actual_datetime' in df.columns:
        df['delay_minutes'] = (
            df['actual_datetime'] - df['scheduled_datetime']
        ).dt.total_seconds() / 60

    # Extract year and month for easier filtering
    if 'service_date' in df.columns:
        df['year'] = df['service_date'].dt.year
        df['month'] = df['service_date'].dt.month
        df['day_of_week'] = df['service_date'].dt.dayofweek

    # Extract hour from actual time for time-of-day analysis
    if 'actual_datetime' in df.columns:
        df['hour'] = df['actual_datetime'].dt.hour

    return df


def load_data_chunked(years: List[str] = None,
                      target_routes_only: bool = False,
                      chunk_size: int = 500000) -> Generator[pd.DataFrame, None, None]:
    """
    Load data in chunks for memory-efficient processing.

    Args:
        years: Years to load
        target_routes_only: Filter to target routes
        chunk_size: Rows per chunk

    Yields:
        DataFrame chunks
    """
    csv_files = find_csv_files(years)

    for file_path in csv_files:
        print(f"Loading {file_path.name}...")

        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            # Convert route_id to string
            chunk['route_id'] = chunk['route_id'].astype(str)

            # Filter if needed
            if target_routes_only:
                chunk = chunk[chunk['route_id'].isin(TARGET_ROUTES)]
                if chunk.empty:
                    continue

            # Parse datetime
            if 'service_date' in chunk.columns:
                chunk['service_date'] = pd.to_datetime(chunk['service_date'], errors='coerce')

            if 'scheduled' in chunk.columns and 'service_date' in chunk.columns:
                chunk['scheduled_datetime'] = parse_mbta_datetime(
                    chunk['service_date'].astype(str),
                    chunk['scheduled']
                )

            if 'actual' in chunk.columns and 'service_date' in chunk.columns:
                chunk['actual_datetime'] = parse_mbta_datetime(
                    chunk['service_date'].astype(str),
                    chunk['actual']
                )

            # Calculate delay
            if 'scheduled_datetime' in chunk.columns and 'actual_datetime' in chunk.columns:
                chunk['delay_minutes'] = (
                    chunk['actual_datetime'] - chunk['scheduled_datetime']
                ).dt.total_seconds() / 60

            # Add time features
            if 'service_date' in chunk.columns:
                chunk['year'] = chunk['service_date'].dt.year
                chunk['month'] = chunk['service_date'].dt.month
                chunk['day_of_week'] = chunk['service_date'].dt.dayofweek

            if 'actual_datetime' in chunk.columns:
                chunk['hour'] = chunk['actual_datetime'].dt.hour

            yield chunk


def load_data(years: List[str] = None,
              target_routes_only: bool = False,
              sample_frac: float = None) -> pd.DataFrame:
    """
    Load all data for specified years into a single DataFrame.

    WARNING: This may use significant memory for full datasets.
    Consider using load_data_chunked() for large data processing.

    Args:
        years: Years to load. If None, load all training years.
        target_routes_only: If True, filter to target routes only
        sample_frac: If set, randomly sample this fraction of data

    Returns:
        Combined DataFrame
    """
    if years is None:
        years = TRAIN_YEARS

    csv_files = find_csv_files(years)

    if not csv_files:
        print(f"No CSV files found for years: {years}")
        return pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files to load")

    dfs = []
    for file_path in csv_files:
        print(f"  Loading {file_path.name}...")
        df = load_single_file(file_path, target_routes_only)

        if sample_frac is not None and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} total rows")

    return combined


def get_data_summary(years: List[str] = None) -> dict:
    """
    Get summary statistics about available data.

    Args:
        years: Years to summarize

    Returns:
        Dictionary with summary info
    """
    csv_files = find_csv_files(years)

    summary = {
        'total_files': len(csv_files),
        'files_by_year': {},
        'total_size_mb': 0
    }

    for f in csv_files:
        year = None
        for y in TRAIN_YEARS + VALIDATION_YEARS:
            if y in str(f):
                year = y
                break

        if year:
            if year not in summary['files_by_year']:
                summary['files_by_year'][year] = {'count': 0, 'size_mb': 0}
            summary['files_by_year'][year]['count'] += 1
            size_mb = f.stat().st_size / (1024 * 1024)
            summary['files_by_year'][year]['size_mb'] += size_mb
            summary['total_size_mb'] += size_mb

    return summary


if __name__ == "__main__":
    # Test data loading
    print("=" * 60)
    print("Data Loading Test")
    print("=" * 60)

    # Get summary
    print("\nData Summary:")
    summary = get_data_summary()
    print(f"Total files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_mb']:.1f} MB")

    for year, info in sorted(summary['files_by_year'].items()):
        print(f"  {year}: {info['count']} files, {info['size_mb']:.1f} MB")

    # Test loading a small sample
    print("\nLoading sample data (1% of 2024)...")
    sample_df = load_data(years=["2024"], sample_frac=0.01)

    if not sample_df.empty:
        print(f"\nSample data shape: {sample_df.shape}")
        print(f"Columns: {list(sample_df.columns)}")
        print(f"\nDelay statistics (minutes):")
        print(sample_df['delay_minutes'].describe())

        print(f"\nRoutes in sample: {sample_df['route_id'].nunique()}")
        print(f"Target routes in sample: {sample_df[sample_df['route_id'].isin(TARGET_ROUTES)]['route_id'].nunique()}")
