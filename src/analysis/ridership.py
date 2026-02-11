"""
Ridership Analysis Module for Boston Bus Equity Project

Analyzes bus ridership trends, particularly pre vs post pandemic comparison.
Addresses research question Q1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TARGET_ROUTES

# Ridership data directory
RIDERSHIP_DIR = DATA_RAW / "ridership"


def find_ridership_files() -> List[Path]:
    """Find all ridership CSV files."""
    files = []
    for subdir in RIDERSHIP_DIR.iterdir():
        if subdir.is_dir():
            files.extend(sorted(subdir.glob("*.csv")))
    return files


def load_ridership_data() -> pd.DataFrame:
    """
    Load all ridership data files.

    Returns:
        DataFrame with all ridership data combined
    """
    files = find_ridership_files()
    if not files:
        print("Warning: No ridership files found")
        return pd.DataFrame()

    all_data = []
    for file_path in files:
        print(f"Loading {file_path.name}...")
        df = pd.read_csv(file_path)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # Parse season to extract year
    # Format: "Fall 2019", "Spring 2024"
    year_extracted = combined['season'].str.extract(r'(\d{4})')
    combined['year'] = pd.to_numeric(year_extracted[0], errors='coerce')
    combined['season_name'] = combined['season'].str.extract(r'(Fall|Spring|Winter|Summer)')

    # Drop rows without valid year
    combined = combined.dropna(subset=['year'])
    combined['year'] = combined['year'].astype(int)

    # Convert route_id to string for consistent comparison
    combined['route_id'] = combined['route_id'].astype(str)

    return combined


def analyze_ridership_by_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze total ridership by route.

    Args:
        df: Ridership DataFrame

    Returns:
        DataFrame with ridership per route
    """
    route_ridership = df.groupby('route_id').agg({
        'boardings': 'sum',
        'alightings': 'sum',
        'sample_size': 'sum'
    }).reset_index()

    route_ridership['total_ridership'] = route_ridership['boardings'] + route_ridership['alightings']
    route_ridership['is_target_route'] = route_ridership['route_id'].isin(TARGET_ROUTES)

    return route_ridership.sort_values('total_ridership', ascending=False)


def analyze_pre_post_pandemic(df: pd.DataFrame) -> Dict:
    """
    Compare ridership before and after pandemic.

    Pre-pandemic: 2016-2019 (all data before March 2020)
    Post-pandemic: 2021-2024 (recovery period, excluding 2020)

    Args:
        df: Ridership DataFrame

    Returns:
        Dictionary with comparison results
    """
    # Define periods
    pre_pandemic = df[df['year'] <= 2019].copy()
    post_pandemic = df[df['year'] >= 2021].copy()
    pandemic_year = df[df['year'] == 2020].copy()

    results = {
        'pre_pandemic': {},
        'post_pandemic': {},
        'pandemic_2020': {},
        'by_route': None,
        'by_year': None
    }

    # Overall statistics
    for name, data in [('pre_pandemic', pre_pandemic),
                       ('post_pandemic', post_pandemic),
                       ('pandemic_2020', pandemic_year)]:
        if not data.empty:
            results[name] = {
                'total_boardings': data['boardings'].sum(),
                'total_alightings': data['alightings'].sum(),
                'avg_boardings_per_stop': data['boardings'].mean(),
                'years': sorted(data['year'].unique().tolist()),
                'routes_count': data['route_id'].nunique()
            }

    # By route comparison
    route_comparison = []
    for route_id in df['route_id'].unique():
        route_pre = pre_pandemic[pre_pandemic['route_id'] == route_id]['boardings'].sum()
        route_post = post_pandemic[post_pandemic['route_id'] == route_id]['boardings'].sum()

        # Normalize by number of years
        pre_years = pre_pandemic['year'].nunique() if not pre_pandemic.empty else 1
        post_years = post_pandemic['year'].nunique() if not post_pandemic.empty else 1

        avg_pre = route_pre / pre_years if pre_years > 0 else 0
        avg_post = route_post / post_years if post_years > 0 else 0

        change_pct = ((avg_post - avg_pre) / avg_pre * 100) if avg_pre > 0 else 0

        route_comparison.append({
            'route_id': route_id,
            'is_target_route': route_id in TARGET_ROUTES,
            'pre_pandemic_total': route_pre,
            'post_pandemic_total': route_post,
            'pre_pandemic_avg_yearly': avg_pre,
            'post_pandemic_avg_yearly': avg_post,
            'change_percent': change_pct
        })

    results['by_route'] = pd.DataFrame(route_comparison).sort_values('change_percent')

    # By year trend
    yearly = df.groupby('year').agg({
        'boardings': 'sum',
        'alightings': 'sum',
        'route_id': 'nunique'
    }).reset_index()
    yearly.columns = ['year', 'total_boardings', 'total_alightings', 'routes_count']
    results['by_year'] = yearly

    return results


def analyze_ridership_by_day_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ridership patterns by day type (weekday, saturday, sunday).

    Args:
        df: Ridership DataFrame

    Returns:
        DataFrame with ridership by day type
    """
    day_type_stats = df.groupby(['year', 'day_type_name']).agg({
        'boardings': 'sum',
        'alightings': 'sum'
    }).reset_index()

    return day_type_stats


def analyze_target_routes_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detailed ridership analysis for target routes.

    Args:
        df: Ridership DataFrame

    Returns:
        DataFrame with target routes ridership details
    """
    target_df = df[df['route_id'].isin(TARGET_ROUTES)].copy()

    if target_df.empty:
        print("Warning: No target routes found in ridership data")
        return pd.DataFrame()

    # By route and year
    route_year = target_df.groupby(['route_id', 'year']).agg({
        'boardings': 'sum',
        'alightings': 'sum',
        'stop_id': 'nunique'
    }).reset_index()
    route_year.columns = ['route_id', 'year', 'total_boardings', 'total_alightings', 'stops_served']

    return route_year


def run_ridership_analysis(output_dir: Path = None) -> Dict:
    """
    Run complete ridership analysis.

    Args:
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    if output_dir is None:
        output_dir = DATA_PROCESSED / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Running Ridership Analysis (Q1)")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading ridership data...")
    df = load_ridership_data()

    if df.empty:
        print("Error: No ridership data available")
        return {}

    print(f"  Loaded {len(df):,} records")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Routes: {df['route_id'].nunique()}")

    results = {}

    # Overall ridership by route
    print("\n[2/4] Analyzing ridership by route...")
    by_route = analyze_ridership_by_route(df)
    by_route.to_csv(output_dir / "ridership_by_route.csv", index=False)
    results['by_route'] = by_route
    print(f"  Top 5 routes by ridership:")
    for _, row in by_route.head(5).iterrows():
        print(f"    Route {row['route_id']}: {row['total_ridership']:,.0f}")

    # Pre vs post pandemic comparison
    print("\n[3/4] Comparing pre vs post pandemic...")
    pandemic_comparison = analyze_pre_post_pandemic(df)
    results['pandemic_comparison'] = pandemic_comparison

    # Save comparison results
    if pandemic_comparison['by_route'] is not None:
        pandemic_comparison['by_route'].to_csv(
            output_dir / "ridership_pre_post_pandemic_by_route.csv", index=False
        )

    if pandemic_comparison['by_year'] is not None:
        pandemic_comparison['by_year'].to_csv(
            output_dir / "ridership_by_year.csv", index=False
        )

    # Print summary
    pre = pandemic_comparison.get('pre_pandemic', {})
    post = pandemic_comparison.get('post_pandemic', {})

    if pre and post:
        pre_total = pre.get('total_boardings', 0)
        post_total = post.get('total_boardings', 0)

        # Normalize by years
        pre_years = len(pre.get('years', [1]))
        post_years = len(post.get('years', [1]))

        pre_avg = pre_total / pre_years if pre_years > 0 else 0
        post_avg = post_total / post_years if post_years > 0 else 0

        change = ((post_avg - pre_avg) / pre_avg * 100) if pre_avg > 0 else 0

        print(f"  Pre-pandemic ({pre.get('years', [])}): {pre_avg:,.0f} avg yearly boardings")
        print(f"  Post-pandemic ({post.get('years', [])}): {post_avg:,.0f} avg yearly boardings")
        print(f"  Change: {change:+.1f}%")

    # Target routes analysis
    print("\n[4/4] Analyzing target routes ridership...")
    target_ridership = analyze_target_routes_ridership(df)
    if not target_ridership.empty:
        target_ridership.to_csv(output_dir / "ridership_target_routes.csv", index=False)
        results['target_routes'] = target_ridership
        print(f"  Analyzed {target_ridership['route_id'].nunique()} target routes")

    print("\n" + "=" * 60)
    print("Ridership Analysis Complete")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    results = run_ridership_analysis()
