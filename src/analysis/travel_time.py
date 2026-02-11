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
        if 'half_trip_id' not in chunk.columns:
            continue

        # Group by trip and calculate travel time
        for (route_id, trip_id), trip_df in chunk.groupby(['route_id', 'half_trip_id']):
            travel_time = calculate_trip_travel_time(trip_df)
            if travel_time is not None:
                route_travel_times[route_id].append(travel_time)

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
        if 'half_trip_id' not in chunk.columns:
            continue

        for (route_id, trip_id), trip_df in chunk.groupby(['route_id', 'half_trip_id']):
            travel_time = calculate_trip_travel_time(trip_df)
            if travel_time is None:
                continue

            # Get hour of trip start
            start_points = trip_df[trip_df['point_type'] == 'Startpoint']
            if start_points.empty:
                continue

            if 'hour' in trip_df.columns:
                hour = start_points['hour'].iloc[0]
            elif 'actual_datetime' in trip_df.columns:
                start_dt = start_points['actual_datetime'].iloc[0]
                if pd.notna(start_dt):
                    hour = start_dt.hour
                else:
                    continue
            else:
                continue

            hourly_times[hour].append(travel_time)

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

        for (route_id, trip_id), trip_df in chunk.groupby(['route_id', 'half_trip_id']):
            start_points = trip_df[trip_df['point_type'] == 'Startpoint']
            end_points = trip_df[trip_df['point_type'] == 'Endpoint']

            if start_points.empty or end_points.empty:
                continue

            # Scheduled travel time
            if 'scheduled_datetime' in trip_df.columns:
                sched_start = start_points['scheduled_datetime'].iloc[0]
                sched_end = end_points['scheduled_datetime'].iloc[0]

                if pd.notna(sched_start) and pd.notna(sched_end):
                    sched_time = (sched_end - sched_start).total_seconds() / 60
                    if 0 < sched_time < 300:
                        route_comparisons[route_id]['scheduled'].append(sched_time)

            # Actual travel time
            actual_time = calculate_trip_travel_time(trip_df)
            if actual_time is not None:
                route_comparisons[route_id]['actual'].append(actual_time)

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
