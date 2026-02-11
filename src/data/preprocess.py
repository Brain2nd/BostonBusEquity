"""
Data Preprocessing Script for Boston Bus Equity Project

This script handles data cleaning, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_RAW, DATA_PROCESSED, TARGET_ROUTES,
    TRAIN_YEARS, VALIDATION_YEARS, DELAY_THRESHOLDS
)


def load_arrival_departure_data(years: list = None) -> pd.DataFrame:
    """
    Load and combine arrival/departure data for specified years.

    Args:
        years: List of years to load. If None, load all available.

    Returns:
        Combined DataFrame with all years' data.
    """
    if years is None:
        years = TRAIN_YEARS + VALIDATION_YEARS

    dfs = []
    for year in years:
        file_path = DATA_RAW / "arrival_departure" / year / f"bus_arrival_departure_{year}.csv"
        if file_path.exists():
            print(f"Loading {year} data...")
            df = pd.read_csv(file_path)
            df["year"] = year
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def clean_arrival_departure_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean arrival/departure data.

    Args:
        df: Raw arrival/departure DataFrame

    Returns:
        Cleaned DataFrame
    """
    print("Cleaning arrival/departure data...")

    # Make a copy
    df = df.copy()

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {initial_rows - len(df)} duplicate rows")

    # Convert timestamp columns (assuming epoch milliseconds)
    time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    for col in time_columns:
        if df[col].dtype in ['int64', 'float64']:
            # Convert from epoch milliseconds to datetime
            df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')

    # Handle missing values
    missing_before = df.isnull().sum().sum()
    # For numeric columns, we'll keep NaN for now (handle in analysis)
    # For categorical columns, fill with 'Unknown'
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    print(f"  Missing values: {missing_before}")

    return df


def calculate_delay(df: pd.DataFrame,
                    scheduled_col: str = 'scheduled_time',
                    actual_col: str = 'actual_time') -> pd.DataFrame:
    """
    Calculate delay time in minutes.

    Args:
        df: DataFrame with scheduled and actual times
        scheduled_col: Name of scheduled time column
        actual_col: Name of actual time column

    Returns:
        DataFrame with delay column added
    """
    df = df.copy()

    if scheduled_col in df.columns and actual_col in df.columns:
        # Calculate delay in minutes
        df['delay_minutes'] = (
            (df[actual_col] - df[scheduled_col]).dt.total_seconds() / 60
        )

        # Categorize delays
        df['delay_category'] = pd.cut(
            df['delay_minutes'],
            bins=[-np.inf, DELAY_THRESHOLDS['on_time'],
                  DELAY_THRESHOLDS['minor_delay'],
                  DELAY_THRESHOLDS['moderate_delay'],
                  DELAY_THRESHOLDS['major_delay'], np.inf],
            labels=['early', 'on_time', 'minor_delay', 'moderate_delay', 'major_delay']
        )

    return df


def filter_target_routes(df: pd.DataFrame, route_col: str = 'route_id') -> pd.DataFrame:
    """
    Filter data to include only target routes.

    Args:
        df: DataFrame with route information
        route_col: Name of route column

    Returns:
        Filtered DataFrame
    """
    if route_col in df.columns:
        # Convert to string for comparison
        df[route_col] = df[route_col].astype(str)
        return df[df[route_col].isin(TARGET_ROUTES)]
    return df


def add_time_features(df: pd.DataFrame, time_col: str = 'actual_time') -> pd.DataFrame:
    """
    Add time-based features for analysis.

    Args:
        df: DataFrame with datetime column
        time_col: Name of time column

    Returns:
        DataFrame with additional time features
    """
    df = df.copy()

    if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['month'] = df[time_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        # Rush hour flags
        df['is_morning_rush'] = df['hour'].between(7, 9)
        df['is_evening_rush'] = df['hour'].between(16, 19)
        df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']

    return df


def split_train_validation(df: pd.DataFrame) -> tuple:
    """
    Split data into training (2018-2024) and validation (2025) sets.

    Args:
        df: Combined DataFrame

    Returns:
        Tuple of (train_df, validation_df)
    """
    train_df = df[df['year'].isin(TRAIN_YEARS)]
    validation_df = df[df['year'].isin(VALIDATION_YEARS)]

    print(f"Training set: {len(train_df)} rows ({TRAIN_YEARS})")
    print(f"Validation set: {len(validation_df)} rows ({VALIDATION_YEARS})")

    return train_df, validation_df


def save_processed_data(train_df: pd.DataFrame, validation_df: pd.DataFrame):
    """Save processed data to disk."""
    train_dir = DATA_PROCESSED / "train"
    val_dir = DATA_PROCESSED / "validation"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_dir / "arrival_departure_train.csv", index=False)
    validation_df.to_csv(val_dir / "arrival_departure_validation.csv", index=False)

    print(f"Saved training data to {train_dir}")
    print(f"Saved validation data to {val_dir}")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("Boston Bus Equity - Data Preprocessing")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading arrival/departure data...")
    df = load_arrival_departure_data()

    if df.empty:
        print("No data found. Please run download_data.py first.")
        return

    print(f"Loaded {len(df)} total rows")

    # Clean data
    print("\n[2/5] Cleaning data...")
    df = clean_arrival_departure_data(df)

    # Add features
    print("\n[3/5] Adding time features...")
    df = add_time_features(df)

    # Calculate delays (column names may vary - adjust as needed)
    print("\n[4/5] Calculating delays...")
    # Note: Actual column names depend on the data schema
    # df = calculate_delay(df)

    # Split data
    print("\n[5/5] Splitting train/validation...")
    train_df, validation_df = split_train_validation(df)

    # Save
    save_processed_data(train_df, validation_df)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
