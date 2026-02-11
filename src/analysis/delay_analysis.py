"""
Delay Analysis Module for Boston Bus Equity Project

Analyzes bus delay patterns across routes, time periods, and demographics.
Addresses research questions Q3, Q4, Q5.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_RAW, DATA_PROCESSED, TARGET_ROUTES,
    TRAIN_YEARS, VALIDATION_YEARS, DELAY_THRESHOLDS,
    TIME_PERIODS, FIGURES_DIR
)
from data.load_data import load_data_chunked, load_data, find_csv_files


def clean_delay_values(delay_minutes: pd.Series) -> pd.Series:
    """
    Clean delay values by handling midnight crossovers.

    Buses running past midnight may have extreme negative delays
    (e.g., -1438 minutes = 24 hours off).

    Args:
        delay_minutes: Raw delay values in minutes

    Returns:
        Cleaned delay values
    """
    # Handle midnight crossover: if delay is > 720 minutes (12 hours),
    # it's likely a midnight issue - subtract 24 hours
    # If delay is < -720 minutes, add 24 hours
    cleaned = delay_minutes.copy()
    cleaned = cleaned.where(cleaned <= 720, cleaned - 1440)
    cleaned = cleaned.where(cleaned >= -720, cleaned + 1440)

    return cleaned


def categorize_delay(delay_minutes: float) -> str:
    """
    Categorize delay into buckets.

    Args:
        delay_minutes: Delay in minutes

    Returns:
        Category string
    """
    if pd.isna(delay_minutes):
        return 'unknown'
    elif delay_minutes < -1:
        return 'early'
    elif delay_minutes <= DELAY_THRESHOLDS['on_time']:
        return 'on_time'
    elif delay_minutes <= DELAY_THRESHOLDS['minor_delay']:
        return 'minor_delay'
    elif delay_minutes <= DELAY_THRESHOLDS['moderate_delay']:
        return 'moderate_delay'
    else:
        return 'major_delay'


def calculate_delay_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive delay metrics from a DataFrame.

    Args:
        df: DataFrame with delay_minutes column

    Returns:
        Dictionary of metrics
    """
    if 'delay_minutes' not in df.columns or df['delay_minutes'].isna().all():
        return {}

    delays = clean_delay_values(df['delay_minutes'])
    valid_delays = delays.dropna()

    if len(valid_delays) == 0:
        return {}

    metrics = {
        'count': len(valid_delays),
        'mean_delay': valid_delays.mean(),
        'median_delay': valid_delays.median(),
        'std_delay': valid_delays.std(),
        'min_delay': valid_delays.min(),
        'max_delay': valid_delays.max(),
        'p25_delay': valid_delays.quantile(0.25),
        'p75_delay': valid_delays.quantile(0.75),
        'p90_delay': valid_delays.quantile(0.90),
        'p95_delay': valid_delays.quantile(0.95),
    }

    # Calculate category percentages
    categories = valid_delays.apply(categorize_delay)
    category_counts = categories.value_counts()
    total = len(categories)

    for cat in ['early', 'on_time', 'minor_delay', 'moderate_delay', 'major_delay']:
        count = category_counts.get(cat, 0)
        metrics[f'{cat}_count'] = count
        metrics[f'{cat}_pct'] = (count / total * 100) if total > 0 else 0

    # On-time performance (within 1 minute of schedule)
    metrics['on_time_performance'] = metrics['on_time_pct'] + metrics.get('early_pct', 0)

    return metrics


def analyze_delays_by_route(years: List[str] = None,
                            target_only: bool = False) -> pd.DataFrame:
    """
    Analyze delays grouped by route.

    Args:
        years: Years to analyze
        target_only: If True, only analyze target routes

    Returns:
        DataFrame with per-route delay metrics
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Analyzing delays by route for years: {years}")

    route_metrics = defaultdict(lambda: {
        'delays': [],
        'total_count': 0
    })

    for chunk in load_data_chunked(years, target_routes_only=target_only):
        if 'delay_minutes' not in chunk.columns:
            continue

        chunk['delay_minutes'] = clean_delay_values(chunk['delay_minutes'])

        for route_id, group in chunk.groupby('route_id'):
            valid_delays = group['delay_minutes'].dropna()
            route_metrics[route_id]['delays'].extend(valid_delays.tolist())
            route_metrics[route_id]['total_count'] += len(group)

    # Calculate final metrics for each route
    results = []
    for route_id, data in route_metrics.items():
        if not data['delays']:
            continue

        delays = pd.Series(data['delays'])
        metrics = calculate_delay_metrics(pd.DataFrame({'delay_minutes': delays}))
        metrics['route_id'] = route_id
        metrics['is_target_route'] = route_id in TARGET_ROUTES
        results.append(metrics)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values('mean_delay', ascending=False)

    return df_results


def analyze_delays_by_time(years: List[str] = None,
                           target_only: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Analyze delays by time dimensions (hour, day of week, month).

    Args:
        years: Years to analyze
        target_only: If True, only analyze target routes

    Returns:
        Dictionary with DataFrames for each time dimension
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Analyzing delays by time for years: {years}")

    hourly_data = defaultdict(list)
    dow_data = defaultdict(list)
    monthly_data = defaultdict(list)

    for chunk in load_data_chunked(years, target_routes_only=target_only):
        if 'delay_minutes' not in chunk.columns:
            continue

        chunk['delay_minutes'] = clean_delay_values(chunk['delay_minutes'])

        # By hour
        if 'hour' in chunk.columns:
            for hour, group in chunk.groupby('hour'):
                valid = group['delay_minutes'].dropna()
                hourly_data[hour].extend(valid.tolist())

        # By day of week
        if 'day_of_week' in chunk.columns:
            for dow, group in chunk.groupby('day_of_week'):
                valid = group['delay_minutes'].dropna()
                dow_data[dow].extend(valid.tolist())

        # By month
        if 'month' in chunk.columns and 'year' in chunk.columns:
            for (year, month), group in chunk.groupby(['year', 'month']):
                valid = group['delay_minutes'].dropna()
                monthly_data[(year, month)].extend(valid.tolist())

    # Convert to DataFrames
    results = {}

    # Hourly
    hourly_results = []
    for hour, delays in sorted(hourly_data.items()):
        metrics = calculate_delay_metrics(pd.DataFrame({'delay_minutes': delays}))
        metrics['hour'] = hour
        hourly_results.append(metrics)
    results['hourly'] = pd.DataFrame(hourly_results)

    # Day of week
    dow_results = []
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for dow, delays in sorted(dow_data.items()):
        metrics = calculate_delay_metrics(pd.DataFrame({'delay_minutes': delays}))
        metrics['day_of_week'] = dow
        metrics['day_name'] = dow_names[int(dow)] if 0 <= dow <= 6 else 'Unknown'
        dow_results.append(metrics)
    results['day_of_week'] = pd.DataFrame(dow_results)

    # Monthly
    monthly_results = []
    for (year, month), delays in sorted(monthly_data.items()):
        metrics = calculate_delay_metrics(pd.DataFrame({'delay_minutes': delays}))
        metrics['year'] = year
        metrics['month'] = month
        metrics['year_month'] = f"{int(year)}-{int(month):02d}"
        monthly_results.append(metrics)
    results['monthly'] = pd.DataFrame(monthly_results)

    return results


def compare_target_vs_other_routes(years: List[str] = None) -> Dict:
    """
    Compare delay metrics between target routes and other routes.

    Args:
        years: Years to analyze

    Returns:
        Dictionary with comparison metrics
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Comparing target routes vs other routes for years: {years}")

    target_delays = []
    other_delays = []

    for chunk in load_data_chunked(years, target_routes_only=False):
        if 'delay_minutes' not in chunk.columns:
            continue

        chunk['delay_minutes'] = clean_delay_values(chunk['delay_minutes'])

        target_mask = chunk['route_id'].isin(TARGET_ROUTES)
        target_valid = chunk.loc[target_mask, 'delay_minutes'].dropna()
        other_valid = chunk.loc[~target_mask, 'delay_minutes'].dropna()

        target_delays.extend(target_valid.tolist())
        other_delays.extend(other_valid.tolist())

    # Calculate metrics for each group
    target_metrics = calculate_delay_metrics(
        pd.DataFrame({'delay_minutes': target_delays})
    )
    other_metrics = calculate_delay_metrics(
        pd.DataFrame({'delay_minutes': other_delays})
    )

    # Calculate differences
    comparison = {
        'target_routes': target_metrics,
        'other_routes': other_metrics,
        'difference': {}
    }

    for key in target_metrics:
        if isinstance(target_metrics[key], (int, float)):
            comparison['difference'][key] = target_metrics[key] - other_metrics.get(key, 0)

    return comparison


def calculate_wait_time_stats(years: List[str] = None,
                              target_only: bool = False) -> Dict:
    """
    Calculate expected wait time statistics.

    Expected wait time = scheduled_headway / 2 + delay_adjustment

    Args:
        years: Years to analyze
        target_only: If True, only analyze target routes

    Returns:
        Dictionary with wait time statistics
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Calculating wait time statistics for years: {years}")

    wait_times = []
    on_time_waits = []
    delayed_waits = []

    for chunk in load_data_chunked(years, target_routes_only=target_only):
        if 'headway' not in chunk.columns or 'delay_minutes' not in chunk.columns:
            continue

        chunk['delay_minutes'] = clean_delay_values(chunk['delay_minutes'])

        # Filter valid headway values (positive and reasonable)
        valid = chunk[
            (chunk['headway'].notna()) &
            (chunk['headway'] > 0) &
            (chunk['headway'] < 120)  # Less than 2 hours
        ].copy()

        if valid.empty:
            continue

        # Expected wait = headway/2 for random arrival
        # Actual wait adjusted by delay
        valid['expected_wait'] = valid['headway'] / 2
        valid['actual_wait'] = valid['expected_wait'] + valid['delay_minutes'].clip(lower=0)

        wait_times.extend(valid['actual_wait'].dropna().tolist())

        # Separate on-time vs delayed
        on_time = valid[valid['delay_minutes'] <= DELAY_THRESHOLDS['on_time']]
        delayed = valid[valid['delay_minutes'] > DELAY_THRESHOLDS['on_time']]

        on_time_waits.extend(on_time['actual_wait'].dropna().tolist())
        delayed_waits.extend(delayed['actual_wait'].dropna().tolist())

    results = {
        'all': {
            'count': len(wait_times),
            'mean': np.mean(wait_times) if wait_times else None,
            'median': np.median(wait_times) if wait_times else None,
            'std': np.std(wait_times) if wait_times else None,
        },
        'on_time': {
            'count': len(on_time_waits),
            'mean': np.mean(on_time_waits) if on_time_waits else None,
            'median': np.median(on_time_waits) if on_time_waits else None,
        },
        'delayed': {
            'count': len(delayed_waits),
            'mean': np.mean(delayed_waits) if delayed_waits else None,
            'median': np.median(delayed_waits) if delayed_waits else None,
        }
    }

    return results


def run_full_delay_analysis(years: List[str] = None,
                            output_dir: Path = None) -> Dict:
    """
    Run complete delay analysis and save results.

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
    print("Running Full Delay Analysis")
    print("=" * 60)

    results = {}

    # Q4: Citywide delay analysis
    print("\n[1/5] Analyzing citywide delays...")
    citywide_by_route = analyze_delays_by_route(years, target_only=False)
    citywide_by_route.to_csv(output_dir / "citywide_delays_by_route.csv", index=False)
    results['citywide_by_route'] = citywide_by_route
    print(f"  Analyzed {len(citywide_by_route)} routes")

    # Q5: Target routes analysis
    print("\n[2/5] Analyzing target routes...")
    target_by_route = analyze_delays_by_route(years, target_only=True)
    target_by_route.to_csv(output_dir / "target_routes_delays.csv", index=False)
    results['target_routes'] = target_by_route
    print(f"  Analyzed {len(target_by_route)} target routes")

    # Target vs other comparison
    print("\n[3/5] Comparing target vs other routes...")
    comparison = compare_target_vs_other_routes(years)
    results['target_vs_other'] = comparison

    # Save comparison summary
    comparison_df = pd.DataFrame({
        'metric': list(comparison['target_routes'].keys()),
        'target_routes': list(comparison['target_routes'].values()),
        'other_routes': [comparison['other_routes'].get(k) for k in comparison['target_routes'].keys()],
        'difference': [comparison['difference'].get(k) for k in comparison['target_routes'].keys()]
    })
    comparison_df.to_csv(output_dir / "target_vs_other_comparison.csv", index=False)

    # Time-based analysis
    print("\n[4/5] Analyzing delays by time...")
    time_analysis = analyze_delays_by_time(years, target_only=False)
    for name, df in time_analysis.items():
        df.to_csv(output_dir / f"delays_by_{name}.csv", index=False)
    results['time_analysis'] = time_analysis

    # Q3: Wait time analysis
    print("\n[5/5] Calculating wait times...")
    wait_stats = calculate_wait_time_stats(years)
    results['wait_times'] = wait_stats

    # Save summary
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    print("\nKey Findings:")
    if 'target_routes' in results and not results['target_routes'].empty:
        target_mean = results['target_routes']['mean_delay'].mean()
        print(f"  Target routes avg delay: {target_mean:.2f} minutes")

    if 'target_vs_other' in results:
        target_delay = results['target_vs_other']['target_routes'].get('mean_delay', 0)
        other_delay = results['target_vs_other']['other_routes'].get('mean_delay', 0)
        diff = target_delay - other_delay
        print(f"  Target routes vs others: {diff:+.2f} minutes")

    if 'wait_times' in results:
        on_time_wait = results['wait_times']['on_time'].get('mean')
        delayed_wait = results['wait_times']['delayed'].get('mean')
        if on_time_wait and delayed_wait:
            print(f"  Wait time (on-time): {on_time_wait:.2f} minutes")
            print(f"  Wait time (delayed): {delayed_wait:.2f} minutes")

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    # Run analysis on training data
    results = run_full_delay_analysis(years=TRAIN_YEARS)
