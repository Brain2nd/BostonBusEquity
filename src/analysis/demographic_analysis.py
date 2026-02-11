"""
Demographic Analysis Module for Boston Bus Equity Project

Analyzes bus service performance by demographic characteristics.
Addresses research question Q7: impact on communities by race, ethnicity, age, income.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED, TARGET_ROUTES

# Import our data modules
sys.path.append(str(Path(__file__).parent.parent / "data"))
from census_data import load_all_demographics, classify_neighborhood_demographics
from stop_neighborhood_mapping import get_stop_neighborhood_dict, create_stop_neighborhood_mapping


def load_service_data_by_stop() -> pd.DataFrame:
    """
    Load delay statistics aggregated by stop.

    Returns:
        DataFrame with delay stats per stop
    """
    results_dir = DATA_PROCESSED / "analysis_results"

    # Try to load existing stop-level analysis
    stop_file = results_dir / "delay_by_stop.csv"
    if stop_file.exists():
        return pd.read_csv(stop_file)

    return pd.DataFrame()


def load_delay_data() -> pd.DataFrame:
    """
    Load processed delay data.

    Returns:
        DataFrame with delay data
    """
    quick_file = DATA_PROCESSED / "quick_analysis" / "delay_sample.csv"
    if quick_file.exists():
        return pd.read_csv(quick_file)

    # Try full processed data
    processed_file = DATA_PROCESSED / "combined_data.csv"
    if processed_file.exists():
        return pd.read_csv(processed_file)

    return pd.DataFrame()


def load_route_delay_data() -> pd.DataFrame:
    """
    Load delay data aggregated by route.

    Returns:
        DataFrame with delay stats per route
    """
    route_file = DATA_PROCESSED / "quick_analysis" / "citywide_delays_by_route.csv"
    if route_file.exists():
        return pd.read_csv(route_file)

    return pd.DataFrame()


def load_route_neighborhood_mapping() -> pd.DataFrame:
    """
    Load the route to neighborhood mapping.

    Returns:
        DataFrame with route_id and neighborhood columns
    """
    mapping_file = DATA_PROCESSED / "route_neighborhood_mapping.csv"
    if mapping_file.exists():
        return pd.read_csv(mapping_file)

    return pd.DataFrame()


def create_route_demographic_profile(route_neighborhoods: pd.DataFrame,
                                     demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Create demographic profile for each route based on neighborhoods served.

    Args:
        route_neighborhoods: Route to neighborhood mapping
        demographics: Demographics by neighborhood

    Returns:
        DataFrame with demographic profile per route
    """
    if route_neighborhoods.empty or demographics.empty:
        return pd.DataFrame()

    # Merge route-neighborhoods with demographics
    merged = route_neighborhoods.merge(demographics, on='neighborhood', how='left')

    # Filter to routes with demographic data
    merged = merged[merged['total_pop'].notna()]

    if merged.empty:
        return pd.DataFrame()

    # Aggregate demographics per route (weighted by population)
    numeric_cols = ['minority_pct', 'black_pct', 'hispanic_pct', 'asian_pct',
                    'median_income', 'low_income_pct', 'poverty_rate']

    route_profiles = merged.groupby('route_id').agg({
        'neighborhood': 'count',
        'total_pop': 'sum',
        **{col: 'mean' for col in numeric_cols if col in merged.columns}
    }).reset_index()

    route_profiles = route_profiles.rename(columns={'neighborhood': 'neighborhoods_served'})

    # Classify routes
    if 'minority_pct' in route_profiles.columns:
        route_profiles['serves_high_minority'] = route_profiles['minority_pct'] > 0.5

    if 'median_income' in route_profiles.columns:
        median_income_threshold = route_profiles['median_income'].median()
        route_profiles['serves_low_income'] = route_profiles['median_income'] < median_income_threshold

    if 'poverty_rate' in route_profiles.columns:
        poverty_threshold = route_profiles['poverty_rate'].mean()
        route_profiles['serves_high_poverty'] = route_profiles['poverty_rate'] > poverty_threshold

    return route_profiles


def analyze_route_service_by_demographics(route_delays: pd.DataFrame,
                                          route_demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Join route service data with demographic profiles.

    Args:
        route_delays: Service/delay data by route
        route_demographics: Demographic profile per route

    Returns:
        DataFrame with merged data
    """
    if route_delays.empty or route_demographics.empty:
        return pd.DataFrame()

    # Ensure route_id types match
    route_delays['route_id'] = route_delays['route_id'].astype(str)
    route_demographics['route_id'] = route_demographics['route_id'].astype(str)

    merged = route_delays.merge(route_demographics, on='route_id', how='inner')

    return merged


def aggregate_delays_by_neighborhood(delay_df: pd.DataFrame,
                                    stop_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Aggregate delay statistics by neighborhood.

    Args:
        delay_df: DataFrame with delay data containing stop_id
        stop_mapping: Dict mapping stop_id to neighborhood

    Returns:
        DataFrame with delay stats per neighborhood
    """
    if delay_df.empty or 'stop_id' not in delay_df.columns:
        return pd.DataFrame()

    # Map stops to neighborhoods
    delay_df = delay_df.copy()
    delay_df['stop_id'] = delay_df['stop_id'].astype(str)
    delay_df['neighborhood'] = delay_df['stop_id'].map(stop_mapping)

    # Filter to mapped stops
    mapped = delay_df[delay_df['neighborhood'].notna()]

    if mapped.empty:
        return pd.DataFrame()

    # Aggregate by neighborhood
    neighborhood_stats = mapped.groupby('neighborhood').agg({
        'delay_minutes': ['mean', 'median', 'std', 'count']
    }).reset_index()

    neighborhood_stats.columns = ['neighborhood', 'mean_delay', 'median_delay',
                                   'std_delay', 'sample_count']

    # Calculate on-time performance (within 5 min of schedule)
    on_time = mapped.groupby('neighborhood').apply(
        lambda x: (x['delay_minutes'].between(-2, 5)).mean()
    ).reset_index(name='on_time_pct')

    neighborhood_stats = neighborhood_stats.merge(on_time, on='neighborhood')

    return neighborhood_stats


def analyze_service_by_demographics(neighborhood_stats: pd.DataFrame,
                                    demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Join neighborhood service stats with demographic data.

    Args:
        neighborhood_stats: Service performance by neighborhood
        demographics: Demographic data by neighborhood

    Returns:
        DataFrame with merged service and demographic data
    """
    if neighborhood_stats.empty or demographics.empty:
        return pd.DataFrame()

    merged = neighborhood_stats.merge(demographics, on='neighborhood', how='inner')

    return merged


def calculate_demographic_correlations(merged_df: pd.DataFrame) -> Dict:
    """
    Calculate correlations between service metrics and demographics.

    Args:
        merged_df: Merged service and demographic data

    Returns:
        Dictionary with correlation results
    """
    if merged_df.empty or len(merged_df) < 5:
        return {'error': 'Insufficient data for correlation analysis'}

    service_metrics = ['mean_delay', 'median_delay', 'on_time_pct']
    demographic_vars = ['minority_pct', 'black_pct', 'hispanic_pct',
                        'median_income', 'low_income_pct', 'poverty_rate']

    results = {
        'sample_size': len(merged_df),
        'correlations': {},
        'significant_findings': []
    }

    for service in service_metrics:
        if service not in merged_df.columns:
            continue

        results['correlations'][service] = {}

        for demo in demographic_vars:
            if demo not in merged_df.columns:
                continue

            valid = merged_df[[service, demo]].dropna()
            if len(valid) < 5:
                continue

            corr, p_value = stats.pearsonr(valid[service], valid[demo])

            results['correlations'][service][demo] = {
                'correlation': round(corr, 3),
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05
            }

            if p_value < 0.05:
                direction = "positive" if corr > 0 else "negative"
                results['significant_findings'].append({
                    'service': service,
                    'demographic': demo,
                    'correlation': round(corr, 3),
                    'p_value': round(p_value, 4),
                    'interpretation': f"{direction.capitalize()} correlation: "
                                    f"r={corr:.3f}, p={p_value:.4f}"
                })

    return results


def compare_vulnerable_vs_other(merged_df: pd.DataFrame) -> Dict:
    """
    Compare service quality between vulnerable and other neighborhoods.

    Vulnerable = high minority AND low income neighborhoods.

    Args:
        merged_df: Merged service and demographic data

    Returns:
        Dictionary with comparison results
    """
    if merged_df.empty or 'vulnerable' not in merged_df.columns:
        return {'error': 'Cannot classify neighborhoods'}

    vulnerable = merged_df[merged_df['vulnerable']]
    other = merged_df[~merged_df['vulnerable']]

    if len(vulnerable) < 2 or len(other) < 2:
        return {'error': 'Insufficient data for comparison'}

    results = {
        'vulnerable_count': len(vulnerable),
        'other_count': len(other),
        'comparisons': {}
    }

    metrics = ['mean_delay', 'median_delay', 'on_time_pct']

    for metric in metrics:
        if metric not in merged_df.columns:
            continue

        v_values = vulnerable[metric].dropna()
        o_values = other[metric].dropna()

        if len(v_values) < 2 or len(o_values) < 2:
            continue

        # Mann-Whitney U test
        try:
            stat, p_value = stats.mannwhitneyu(v_values, o_values, alternative='two-sided')

            results['comparisons'][metric] = {
                'vulnerable_mean': round(v_values.mean(), 2),
                'other_mean': round(o_values.mean(), 2),
                'difference': round(v_values.mean() - o_values.mean(), 2),
                'vulnerable_median': round(v_values.median(), 2),
                'other_median': round(o_values.median(), 2),
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05
            }
        except Exception as e:
            results['comparisons'][metric] = {'error': str(e)}

    return results


def run_demographic_analysis(output_dir: Path = None) -> Dict:
    """
    Run complete demographic equity analysis (Q7).

    Uses route-level analysis with neighborhood demographics.

    Args:
        output_dir: Directory to save results

    Returns:
        Dictionary with all analysis results
    """
    if output_dir is None:
        output_dir = DATA_PROCESSED / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Running Demographic Equity Analysis (Q7)")
    print("=" * 60)

    results = {}

    # Step 1: Load demographics
    print("\n[1/5] Loading demographic data...")
    demographics = load_all_demographics()
    if demographics.empty:
        print("  Error: No demographic data available")
        return {'error': 'No demographic data'}

    demographics = classify_neighborhood_demographics(demographics)
    print(f"  Loaded demographics for {len(demographics)} neighborhoods")
    demographics.to_csv(output_dir / "demographic_neighborhood_summary.csv", index=False)

    results['demographics_summary'] = {
        'total_neighborhoods': len(demographics),
        'high_minority_count': int(demographics['high_minority'].sum()),
        'low_income_count': int(demographics['low_income'].sum()),
        'vulnerable_count': int(demographics['vulnerable'].sum())
    }

    # Step 2: Load route-neighborhood mapping
    print("\n[2/5] Loading route-neighborhood mapping...")
    route_neighborhoods = load_route_neighborhood_mapping()

    if route_neighborhoods.empty:
        print("  Warning: No route-neighborhood mapping found")
        print("  Run stop_neighborhood_mapping first")
        return results

    print(f"  Loaded {len(route_neighborhoods)} route-neighborhood pairs")

    # Step 3: Create route demographic profiles
    print("\n[3/5] Creating route demographic profiles...")
    route_profiles = create_route_demographic_profile(route_neighborhoods, demographics)

    if route_profiles.empty:
        print("  Warning: Could not create route profiles")
        return results

    print(f"  Created profiles for {len(route_profiles)} routes")
    route_profiles.to_csv(output_dir / "route_demographic_profiles.csv", index=False)

    # Step 4: Load route delay data and merge
    print("\n[4/5] Loading route delay data...")
    route_delays = load_route_delay_data()

    if route_delays.empty:
        print("  Warning: No route delay data found")
        print("  Route demographic profiles saved")
        results['route_profiles'] = route_profiles
        return results

    print(f"  Loaded delay data for {len(route_delays)} routes")

    # Merge route delays with demographic profiles
    merged = analyze_route_service_by_demographics(route_delays, route_profiles)
    merged.to_csv(output_dir / "route_service_demographics.csv", index=False)
    results['merged_data'] = merged
    print(f"  Merged data for {len(merged)} routes")

    # Step 5: Statistical analysis
    print("\n[5/5] Running statistical analysis...")

    # Correlation analysis
    service_metrics = ['mean_delay', 'median_delay', 'on_time_performance']
    demographic_vars = ['minority_pct', 'black_pct', 'hispanic_pct',
                        'median_income', 'low_income_pct', 'poverty_rate']

    correlations = {
        'sample_size': len(merged),
        'correlations': {},
        'significant_findings': []
    }

    for service in service_metrics:
        if service not in merged.columns:
            continue

        correlations['correlations'][service] = {}

        for demo in demographic_vars:
            if demo not in merged.columns:
                continue

            valid = merged[[service, demo]].dropna()
            if len(valid) < 10:
                continue

            corr, p_value = stats.pearsonr(valid[service], valid[demo])

            correlations['correlations'][service][demo] = {
                'correlation': round(corr, 3),
                'p_value': round(p_value, 4),
                'significant': p_value < 0.05
            }

            if p_value < 0.05:
                direction = "positive" if corr > 0 else "negative"
                correlations['significant_findings'].append({
                    'service': service,
                    'demographic': demo,
                    'correlation': round(corr, 3),
                    'p_value': round(p_value, 4),
                    'interpretation': f"{direction.capitalize()} correlation: r={corr:.3f}, p={p_value:.4f}"
                })

    results['correlations'] = correlations

    if correlations['significant_findings']:
        print("\n  Significant correlations found:")
        for finding in correlations['significant_findings']:
            print(f"    - {finding['service']} vs {finding['demographic']}: "
                  f"r={finding['correlation']:.3f}, p={finding['p_value']:.4f}")

    # Compare routes serving high-minority vs other areas
    if 'serves_high_minority' in merged.columns:
        high_minority = merged[merged['serves_high_minority']]
        other = merged[~merged['serves_high_minority']]

        if len(high_minority) >= 5 and len(other) >= 5:
            comparison = {'high_minority_count': len(high_minority), 'other_count': len(other), 'comparisons': {}}

            for metric in ['mean_delay', 'on_time_performance']:
                if metric not in merged.columns:
                    continue

                hm_values = high_minority[metric].dropna()
                o_values = other[metric].dropna()

                if len(hm_values) >= 5 and len(o_values) >= 5:
                    stat, p_value = stats.mannwhitneyu(hm_values, o_values, alternative='two-sided')

                    comparison['comparisons'][metric] = {
                        'high_minority_mean': round(hm_values.mean(), 2),
                        'other_mean': round(o_values.mean(), 2),
                        'difference': round(hm_values.mean() - o_values.mean(), 2),
                        'p_value': round(p_value, 4),
                        'significant': p_value < 0.05
                    }

            results['high_minority_comparison'] = comparison

            print("\n  High-Minority Areas vs Other Areas:")
            for metric, data in comparison['comparisons'].items():
                sig = "*" if data.get('significant') else ""
                print(f"    {metric}: high-minority={data['high_minority_mean']}, "
                      f"other={data['other_mean']} (diff={data['difference']}{sig})")

    # Save correlation results
    corr_records = []
    for service, demo_corrs in correlations.get('correlations', {}).items():
        for demo, data in demo_corrs.items():
            corr_records.append({
                'service_metric': service,
                'demographic_variable': demo,
                **data
            })
    if corr_records:
        pd.DataFrame(corr_records).to_csv(output_dir / "demographic_correlations.csv", index=False)

    print("\n" + "=" * 60)
    print("Demographic Analysis Complete")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    results = run_demographic_analysis()

    if 'correlations' in results and 'significant_findings' in results['correlations']:
        print("\n" + "=" * 60)
        print("Key Findings Summary")
        print("=" * 60)

        for finding in results['correlations']['significant_findings']:
            print(f"\n• {finding['interpretation']}")
            print(f"  Service: {finding['service']}, Demographic: {finding['demographic']}")
