"""
Travel Time Analysis Module for Boston Bus Equity Project

Analyzes end-to-end travel times for bus routes.
Addresses research question Q2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_RAW, DATA_PROCESSED, TARGET_ROUTES,
    TRAIN_YEARS, VALIDATION_YEARS
)
from data.load_data import load_data_chunked, find_csv_files


def calculate_trip_travel_time(trip_df: pd.DataFrame) -> Optional[float]:
    """
    Calculate travel time for a single trip.

    Args:
        trip_df: DataFrame containing all stops for one trip

    Returns:
        Travel time in minutes, or None if cannot be calculated
    """
    if trip_df.empty:
        return None

    # Find start and end points
    start_points = trip_df[trip_df['point_type'] == 'Startpoint']
    end_points = trip_df[trip_df['point_type'] == 'Endpoint']

    if start_points.empty or end_points.empty:
        return None

    # Use actual times if available
    if 'actual_datetime' in trip_df.columns:
        start_time = start_points['actual_datetime'].iloc[0]
        end_time = end_points['actual_datetime'].iloc[0]
    elif 'actual' in trip_df.columns:
        start_time = pd.to_datetime(start_points['actual'].iloc[0], errors='coerce')
        end_time = pd.to_datetime(end_points['actual'].iloc[0], errors='coerce')
    else:
        return None

    if pd.isna(start_time) or pd.isna(end_time):
        return None

    # Calculate travel time in minutes
    travel_time = (end_time - start_time).total_seconds() / 60

    # Validate: travel time should be positive and reasonable
    # Most bus routes take between 10 and 180 minutes
    if travel_time <= 0 or travel_time > 300:
        return None

    return travel_time


def analyze_travel_times_by_route(years: List[str] = None,
                                  target_only: bool = False) -> pd.DataFrame:
    """
    Analyze travel times grouped by route.

    Args:
        years: Years to analyze
        target_only: If True, only analyze target routes

    Returns:
        DataFrame with per-route travel time metrics
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Analyzing travel times by route for years: {years}")

    route_travel_times = defaultdict(list)

    for chunk in load_data_chunked(years, target_routes_only=target_only):
        if 'half_trip_id' not in chunk.columns or 'actual_datetime' not in chunk.columns:
            continue

        # Optimized: Use vectorized operations instead of iterating
        # Filter to start and end points only
        endpoints = chunk[chunk['point_type'].isin(['Startpoint', 'Endpoint'])].copy()
        if endpoints.empty:
            continue

        # Pivot to get start and end times side by side
        pivot = endpoints.pivot_table(
            index=['route_id', 'half_trip_id'],
            columns='point_type',
            values='actual_datetime',
            aggfunc='first'
        )

        if 'Startpoint' not in pivot.columns or 'Endpoint' not in pivot.columns:
            continue

        # Calculate travel time in minutes (vectorized)
        pivot['travel_time'] = (pivot['Endpoint'] - pivot['Startpoint']).dt.total_seconds() / 60

        # Filter valid travel times (positive and reasonable)
        valid = pivot[(pivot['travel_time'] > 0) & (pivot['travel_time'] <= 300)]

        # Aggregate by route
        for route_id in valid.index.get_level_values('route_id').unique():
            times = valid.loc[route_id, 'travel_time'].tolist()
            route_travel_times[route_id].extend(times)

    # Calculate statistics for each route
    results = []
    for route_id, times in route_travel_times.items():
        if not times:
            continue

        times_series = pd.Series(times)
        results.append({
            'route_id': route_id,
            'is_target_route': route_id in TARGET_ROUTES,
            'trip_count': len(times),
            'mean_travel_time': times_series.mean(),
            'median_travel_time': times_series.median(),
            'std_travel_time': times_series.std(),
            'min_travel_time': times_series.min(),
            'max_travel_time': times_series.max(),
            'p25_travel_time': times_series.quantile(0.25),
            'p75_travel_time': times_series.quantile(0.75),
            'p90_travel_time': times_series.quantile(0.90),
        })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('mean_travel_time', ascending=False)

    return df_results


def analyze_travel_times_by_time_of_day(years: List[str] = None,
                                        target_only: bool = False) -> pd.DataFrame:
    """
    Analyze travel times by hour of day (peak vs off-peak).

    Args:
        years: Years to analyze
        target_only: If True, only analyze target routes

    Returns:
        DataFrame with hourly travel time metrics
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Analyzing travel times by time of day for years: {years}")

    hourly_times = defaultdict(list)

    for chunk in load_data_chunked(years, target_routes_only=target_only):
        if 'half_trip_id' not in chunk.columns or 'actual_datetime' not in chunk.columns:
            continue

        # Optimized: Use vectorized operations
        endpoints = chunk[chunk['point_type'].isin(['Startpoint', 'Endpoint'])].copy()
        if endpoints.empty:
            continue

        # Pivot to get start and end times
        pivot = endpoints.pivot_table(
            index=['route_id', 'half_trip_id'],
            columns='point_type',
            values='actual_datetime',
            aggfunc='first'
        )

        if 'Startpoint' not in pivot.columns or 'Endpoint' not in pivot.columns:
            continue

        # Calculate travel time and hour
        pivot['travel_time'] = (pivot['Endpoint'] - pivot['Startpoint']).dt.total_seconds() / 60
        pivot['hour'] = pivot['Startpoint'].dt.hour

        # Filter valid travel times
        valid = pivot[(pivot['travel_time'] > 0) & (pivot['travel_time'] <= 300) & pivot['hour'].notna()]

        # Group by hour
        for hour in valid['hour'].dropna().unique():
            hour_int = int(hour)
            times = valid[valid['hour'] == hour]['travel_time'].tolist()
            hourly_times[hour_int].extend(times)

    # Calculate statistics
    results = []
    for hour, times in sorted(hourly_times.items()):
        if not times:
            continue

        times_series = pd.Series(times)

        # Determine period
        if 7 <= hour <= 9:
            period = 'morning_rush'
        elif 16 <= hour <= 19:
            period = 'evening_rush'
        elif 22 <= hour or hour <= 5:
            period = 'night'
        else:
            period = 'off_peak'

        results.append({
            'hour': hour,
            'period': period,
            'trip_count': len(times),
            'mean_travel_time': times_series.mean(),
            'median_travel_time': times_series.median(),
            'std_travel_time': times_series.std(),
        })

    return pd.DataFrame(results)


def calculate_scheduled_vs_actual_travel_time(years: List[str] = None,
                                              target_only: bool = False) -> pd.DataFrame:
    """
    Compare scheduled vs actual travel times.

    Args:
        years: Years to analyze
        target_only: If True, only analyze target routes

    Returns:
        DataFrame with scheduled vs actual comparison
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Comparing scheduled vs actual travel times for years: {years}")

    route_comparisons = defaultdict(lambda: {'scheduled': [], 'actual': []})

    for chunk in load_data_chunked(years, target_routes_only=target_only):
        if 'half_trip_id' not in chunk.columns:
            continue

        has_scheduled = 'scheduled_datetime' in chunk.columns
        has_actual = 'actual_datetime' in chunk.columns

        if not has_scheduled and not has_actual:
            continue

        # Optimized: Use vectorized operations
        endpoints = chunk[chunk['point_type'].isin(['Startpoint', 'Endpoint'])].copy()
        if endpoints.empty:
            continue

        # Pivot for actual times
        if has_actual:
            pivot_actual = endpoints.pivot_table(
                index=['route_id', 'half_trip_id'],
                columns='point_type',
                values='actual_datetime',
                aggfunc='first'
            )
            if 'Startpoint' in pivot_actual.columns and 'Endpoint' in pivot_actual.columns:
                pivot_actual['actual_time'] = (pivot_actual['Endpoint'] - pivot_actual['Startpoint']).dt.total_seconds() / 60
                valid_actual = pivot_actual[(pivot_actual['actual_time'] > 0) & (pivot_actual['actual_time'] <= 300)]

                for route_id in valid_actual.index.get_level_values('route_id').unique():
                    times = valid_actual.loc[route_id, 'actual_time'].tolist()
                    route_comparisons[route_id]['actual'].extend(times)

        # Pivot for scheduled times
        if has_scheduled:
            pivot_sched = endpoints.pivot_table(
                index=['route_id', 'half_trip_id'],
                columns='point_type',
                values='scheduled_datetime',
                aggfunc='first'
            )
            if 'Startpoint' in pivot_sched.columns and 'Endpoint' in pivot_sched.columns:
                pivot_sched['sched_time'] = (pivot_sched['Endpoint'] - pivot_sched['Startpoint']).dt.total_seconds() / 60
                valid_sched = pivot_sched[(pivot_sched['sched_time'] > 0) & (pivot_sched['sched_time'] <= 300)]

                for route_id in valid_sched.index.get_level_values('route_id').unique():
                    times = valid_sched.loc[route_id, 'sched_time'].tolist()
                    route_comparisons[route_id]['scheduled'].extend(times)

    # Calculate comparison metrics
    results = []
    for route_id, data in route_comparisons.items():
        if not data['scheduled'] or not data['actual']:
            continue

        sched = pd.Series(data['scheduled'])
        actual = pd.Series(data['actual'])

        results.append({
            'route_id': route_id,
            'is_target_route': route_id in TARGET_ROUTES,
            'mean_scheduled': sched.mean(),
            'mean_actual': actual.mean(),
            'travel_time_diff': actual.mean() - sched.mean(),
            'travel_time_diff_pct': ((actual.mean() - sched.mean()) / sched.mean() * 100
                                     if sched.mean() > 0 else None),
        })

    return pd.DataFrame(results)


def run_travel_time_analysis(years: List[str] = None,
                             output_dir: Path = None) -> Dict:
    """
    Run complete travel time analysis.

    Args:
        years: Years to analyze
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    if years is None:
        years = TRAIN_YEARS

    if output_dir is None:
        output_dir = DATA_PROCESSED / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Running Travel Time Analysis")
    print("=" * 60)

    results = {}

    # By route
    print("\n[1/3] Analyzing travel times by route...")
    by_route = analyze_travel_times_by_route(years)
    by_route.to_csv(output_dir / "travel_times_by_route.csv", index=False)
    results['by_route'] = by_route
    print(f"  Analyzed {len(by_route)} routes")

    # By time of day
    print("\n[2/3] Analyzing travel times by time of day...")
    by_hour = analyze_travel_times_by_time_of_day(years)
    by_hour.to_csv(output_dir / "travel_times_by_hour.csv", index=False)
    results['by_hour'] = by_hour

    # Scheduled vs actual
    print("\n[3/3] Comparing scheduled vs actual...")
    sched_vs_actual = calculate_scheduled_vs_actual_travel_time(years)
    sched_vs_actual.to_csv(output_dir / "travel_times_scheduled_vs_actual.csv", index=False)
    results['scheduled_vs_actual'] = sched_vs_actual

    print("\n" + "=" * 60)
    print("Travel Time Analysis Complete")
    print("=" * 60)

    if not by_route.empty:
        print(f"\nAverage travel time: {by_route['mean_travel_time'].mean():.1f} minutes")
        print(f"Longest route: {by_route.iloc[0]['route_id']} "
              f"({by_route.iloc[0]['mean_travel_time']:.1f} min)")

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    results = run_travel_time_analysis(years=TRAIN_YEARS)
