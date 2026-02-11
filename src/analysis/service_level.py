"""
Service Level Analysis Module for Boston Bus Equity Project

Analyzes disparities in service levels between different routes.
Addresses research question Q6.
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
    TRAIN_YEARS, VALIDATION_YEARS, DELAY_THRESHOLDS
)
from data.load_data import load_data_chunked, find_csv_files
from analysis.delay_analysis import clean_delay_values, categorize_delay


def calculate_headway_consistency(scheduled_headway: pd.Series,
                                  actual_headway: pd.Series) -> Dict:
    """
    Calculate headway consistency metrics.

    Args:
        scheduled_headway: Series of scheduled headways
        actual_headway: Series of actual headways

    Returns:
        Dictionary with consistency metrics
    """
    valid_mask = (
        scheduled_headway.notna() &
        actual_headway.notna() &
        (scheduled_headway > 0) &
        (actual_headway > 0) &
        (scheduled_headway < 120) &  # Less than 2 hours
        (actual_headway < 120)
    )

    if valid_mask.sum() == 0:
        return {}

    scheduled = scheduled_headway[valid_mask]
    actual = actual_headway[valid_mask]

    # Calculate deviation
    deviation = actual - scheduled
    deviation_pct = (deviation / scheduled * 100).replace([np.inf, -np.inf], np.nan).dropna()

    return {
        'headway_count': len(scheduled),
        'mean_scheduled_headway': scheduled.mean(),
        'mean_actual_headway': actual.mean(),
        'mean_headway_deviation': deviation.mean(),
        'std_headway_deviation': deviation.std(),
        'headway_consistency': 1 - (deviation.abs().mean() / scheduled.mean()),  # 0-1 score
        'pct_headway_on_target': (deviation.abs() <= 2).mean() * 100,  # Within 2 min
    }


def calculate_service_score(metrics: Dict,
                            weights: Dict = None) -> float:
    """
    Calculate composite service level score.

    Higher score = better service.

    Args:
        metrics: Dictionary with various service metrics
        weights: Custom weights for scoring components

    Returns:
        Composite service score (0-100)
    """
    if weights is None:
        weights = {
            'on_time_performance': 0.35,
            'avg_delay_score': 0.25,
            'headway_consistency': 0.20,
            'service_reliability': 0.20,
        }

    score = 0
    total_weight = 0

    # On-time performance (0-100)
    if 'on_time_performance' in metrics:
        score += weights['on_time_performance'] * metrics['on_time_performance']
        total_weight += weights['on_time_performance']

    # Average delay score (inverse, normalized)
    if 'mean_delay' in metrics:
        # Convert delay to 0-100 score (lower delay = higher score)
        delay = metrics['mean_delay']
        delay_score = max(0, 100 - abs(delay) * 10)  # 10 min delay = 0 score
        score += weights['avg_delay_score'] * delay_score
        total_weight += weights['avg_delay_score']

    # Headway consistency (0-100)
    if 'headway_consistency' in metrics:
        score += weights['headway_consistency'] * (metrics['headway_consistency'] * 100)
        total_weight += weights['headway_consistency']

    # Service reliability (inverse of std deviation)
    if 'std_delay' in metrics:
        std = metrics['std_delay']
        reliability_score = max(0, 100 - std * 10)
        score += weights['service_reliability'] * reliability_score
        total_weight += weights['service_reliability']

    if total_weight == 0:
        return 0

    return score / total_weight


def analyze_service_levels_by_route(years: List[str] = None) -> pd.DataFrame:
    """
    Analyze service levels for each route.

    Args:
        years: Years to analyze

    Returns:
        DataFrame with service level metrics per route
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Analyzing service levels by route for years: {years}")

    route_data = defaultdict(lambda: {
        'delays': [],
        'scheduled_headways': [],
        'actual_headways': [],
        'total_observations': 0
    })

    for chunk in load_data_chunked(years, target_routes_only=False):
        if 'delay_minutes' not in chunk.columns:
            continue

        chunk['delay_minutes'] = clean_delay_values(chunk['delay_minutes'])

        for route_id, group in chunk.groupby('route_id'):
            # Delays
            valid_delays = group['delay_minutes'].dropna()
            route_data[route_id]['delays'].extend(valid_delays.tolist())
            route_data[route_id]['total_observations'] += len(group)

            # Headways
            if 'scheduled_headway' in group.columns and 'headway' in group.columns:
                valid_mask = (
                    group['scheduled_headway'].notna() &
                    group['headway'].notna()
                )
                route_data[route_id]['scheduled_headways'].extend(
                    group.loc[valid_mask, 'scheduled_headway'].tolist()
                )
                route_data[route_id]['actual_headways'].extend(
                    group.loc[valid_mask, 'headway'].tolist()
                )

    # Calculate metrics for each route
    results = []
    for route_id, data in route_data.items():
        if not data['delays']:
            continue

        delays = pd.Series(data['delays'])

        # Basic delay metrics
        metrics = {
            'route_id': route_id,
            'is_target_route': route_id in TARGET_ROUTES,
            'total_observations': data['total_observations'],
            'mean_delay': delays.mean(),
            'median_delay': delays.median(),
            'std_delay': delays.std(),
        }

        # On-time performance
        categories = delays.apply(categorize_delay)
        on_time_count = (categories == 'on_time').sum() + (categories == 'early').sum()
        metrics['on_time_performance'] = (on_time_count / len(categories) * 100
                                          if len(categories) > 0 else 0)

        # Delay category percentages
        for cat in ['early', 'on_time', 'minor_delay', 'moderate_delay', 'major_delay']:
            cat_count = (categories == cat).sum()
            metrics[f'{cat}_pct'] = cat_count / len(categories) * 100 if len(categories) > 0 else 0

        # Headway consistency
        if data['scheduled_headways'] and data['actual_headways']:
            headway_metrics = calculate_headway_consistency(
                pd.Series(data['scheduled_headways']),
                pd.Series(data['actual_headways'])
            )
            metrics.update(headway_metrics)

        # Calculate composite service score
        metrics['service_score'] = calculate_service_score(metrics)

        results.append(metrics)

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        # Add ranking
        df_results['service_rank'] = df_results['service_score'].rank(ascending=False)
        df_results = df_results.sort_values('service_score', ascending=False)

    return df_results


def identify_service_disparities(service_df: pd.DataFrame,
                                 threshold_pct: float = 20) -> Dict:
    """
    Identify significant disparities in service levels.

    Args:
        service_df: DataFrame from analyze_service_levels_by_route
        threshold_pct: Percentage deviation from mean to consider significant

    Returns:
        Dictionary with disparity analysis
    """
    if service_df.empty:
        return {}

    mean_score = service_df['service_score'].mean()
    std_score = service_df['service_score'].std()

    threshold_low = mean_score - (mean_score * threshold_pct / 100)
    threshold_high = mean_score + (mean_score * threshold_pct / 100)

    below_avg = service_df[service_df['service_score'] < threshold_low]
    above_avg = service_df[service_df['service_score'] > threshold_high]

    # Target routes analysis
    target_df = service_df[service_df['is_target_route']]
    other_df = service_df[~service_df['is_target_route']]

    target_mean = target_df['service_score'].mean() if not target_df.empty else None
    other_mean = other_df['service_score'].mean() if not other_df.empty else None

    return {
        'mean_service_score': mean_score,
        'std_service_score': std_score,
        'min_service_score': service_df['service_score'].min(),
        'max_service_score': service_df['service_score'].max(),
        'below_average_routes': below_avg['route_id'].tolist(),
        'below_average_count': len(below_avg),
        'above_average_routes': above_avg['route_id'].tolist(),
        'above_average_count': len(above_avg),
        'worst_routes': service_df.nsmallest(5, 'service_score')[['route_id', 'service_score']].to_dict('records'),
        'best_routes': service_df.nlargest(5, 'service_score')[['route_id', 'service_score']].to_dict('records'),
        'target_routes_mean_score': target_mean,
        'other_routes_mean_score': other_mean,
        'target_vs_other_diff': (target_mean - other_mean) if target_mean and other_mean else None,
    }


def analyze_temporal_service_trends(years: List[str] = None) -> pd.DataFrame:
    """
    Analyze service level trends over time.

    Args:
        years: Years to analyze

    Returns:
        DataFrame with monthly service metrics
    """
    if years is None:
        years = TRAIN_YEARS

    print(f"Analyzing service level trends for years: {years}")

    monthly_data = defaultdict(lambda: {'delays': [], 'on_time_count': 0, 'total': 0})

    for chunk in load_data_chunked(years, target_routes_only=False):
        if 'delay_minutes' not in chunk.columns:
            continue

        chunk['delay_minutes'] = clean_delay_values(chunk['delay_minutes'])

        if 'year' not in chunk.columns or 'month' not in chunk.columns:
            continue

        for (year, month), group in chunk.groupby(['year', 'month']):
            key = f"{int(year)}-{int(month):02d}"
            valid_delays = group['delay_minutes'].dropna()
            monthly_data[key]['delays'].extend(valid_delays.tolist())

            # Count on-time
            on_time = (valid_delays <= DELAY_THRESHOLDS['on_time']).sum()
            monthly_data[key]['on_time_count'] += on_time
            monthly_data[key]['total'] += len(valid_delays)

    # Calculate monthly metrics
    results = []
    for year_month, data in sorted(monthly_data.items()):
        if not data['delays']:
            continue

        delays = pd.Series(data['delays'])
        results.append({
            'year_month': year_month,
            'observation_count': data['total'],
            'mean_delay': delays.mean(),
            'median_delay': delays.median(),
            'on_time_performance': (data['on_time_count'] / data['total'] * 100
                                    if data['total'] > 0 else 0),
            'p90_delay': delays.quantile(0.90),
        })

    return pd.DataFrame(results)


def run_service_level_analysis(years: List[str] = None,
                               output_dir: Path = None) -> Dict:
    """
    Run complete service level analysis.

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
    print("Running Service Level Analysis")
    print("=" * 60)

    results = {}

    # Service levels by route
    print("\n[1/3] Analyzing service levels by route...")
    service_by_route = analyze_service_levels_by_route(years)
    service_by_route.to_csv(output_dir / "service_levels_by_route.csv", index=False)
    results['by_route'] = service_by_route
    print(f"  Analyzed {len(service_by_route)} routes")

    # Identify disparities
    print("\n[2/3] Identifying service disparities...")
    disparities = identify_service_disparities(service_by_route)
    results['disparities'] = disparities

    # Save disparities summary
    disparity_df = pd.DataFrame([{
        'metric': k,
        'value': str(v) if isinstance(v, list) else v
    } for k, v in disparities.items()])
    disparity_df.to_csv(output_dir / "service_disparities_summary.csv", index=False)

    # Temporal trends
    print("\n[3/3] Analyzing temporal trends...")
    trends = analyze_temporal_service_trends(years)
    trends.to_csv(output_dir / "service_level_trends.csv", index=False)
    results['trends'] = trends

    print("\n" + "=" * 60)
    print("Service Level Analysis Complete")
    print("=" * 60)

    if disparities:
        print(f"\nKey Findings:")
        print(f"  Mean service score: {disparities['mean_service_score']:.1f}/100")
        print(f"  Routes below average: {disparities['below_average_count']}")
        print(f"  Routes above average: {disparities['above_average_count']}")

        if disparities['target_vs_other_diff'] is not None:
            diff = disparities['target_vs_other_diff']
            direction = "better" if diff > 0 else "worse"
            print(f"  Target routes are {abs(diff):.1f} points {direction} than others")

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    results = run_service_level_analysis(years=TRAIN_YEARS)
