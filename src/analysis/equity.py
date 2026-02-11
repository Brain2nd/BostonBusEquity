"""
Equity Analysis Module for Boston Bus Equity Project

Analyzes the relationship between bus service quality and
demographic characteristics of served communities.
Addresses research question Q7.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, TARGET_ROUTES,
    TRAIN_YEARS, FIGURES_DIR
)


# Expected demographic data structure
# Census data should have columns like:
# - tract_id or neighborhood
# - total_population
# - pct_white, pct_black, pct_hispanic, pct_asian
# - median_household_income
# - pct_below_poverty
# - pct_no_vehicle


def load_demographic_data(file_path: Path = None) -> Optional[pd.DataFrame]:
    """
    Load demographic data from census/ACS files.

    Args:
        file_path: Path to demographic data file

    Returns:
        DataFrame with demographic data, or None if not found
    """
    if file_path is None:
        # Try to find demographic data in expected locations
        possible_paths = [
            DATA_EXTERNAL / "demographics.csv",
            DATA_RAW / "census" / "demographics.csv",
            DATA_RAW / "census" / "boston_census_2020.csv",
        ]

        for path in possible_paths:
            if path.exists():
                file_path = path
                break

    if file_path is None or not file_path.exists():
        print("Warning: Demographic data file not found.")
        print("Please download census data and save to:")
        print(f"  {DATA_EXTERNAL / 'demographics.csv'}")
        return None

    return pd.read_csv(file_path)


def load_route_geography_mapping(file_path: Path = None) -> Optional[pd.DataFrame]:
    """
    Load route-to-neighborhood/census-tract mapping.

    This mapping connects bus routes to the geographic areas they serve.

    Args:
        file_path: Path to mapping file

    Returns:
        DataFrame with route-geography mapping
    """
    if file_path is None:
        possible_paths = [
            DATA_EXTERNAL / "route_geography.csv",
            DATA_RAW / "route_mapping.csv",
        ]

        for path in possible_paths:
            if path.exists():
                file_path = path
                break

    if file_path is None or not file_path.exists():
        print("Warning: Route geography mapping not found.")
        print("Consider creating a mapping that connects routes to neighborhoods.")
        return None

    return pd.read_csv(file_path)


def create_route_demographic_profile(route_geography: pd.DataFrame,
                                     demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Create demographic profiles for each route based on served areas.

    Args:
        route_geography: Route to geography mapping
        demographics: Demographics by geography

    Returns:
        DataFrame with demographic profile per route
    """
    # This is a simplified version - actual implementation would need
    # proper geographic matching logic

    # Merge route geography with demographics
    merged = route_geography.merge(demographics, how='left')

    # Aggregate by route (weighted by population or simple average)
    numeric_cols = demographics.select_dtypes(include=[np.number]).columns.tolist()

    route_profiles = merged.groupby('route_id')[numeric_cols].mean().reset_index()

    return route_profiles


def calculate_correlation_analysis(service_df: pd.DataFrame,
                                   demographic_df: pd.DataFrame) -> Dict:
    """
    Calculate correlations between service metrics and demographics.

    Args:
        service_df: Service level data by route
        demographic_df: Demographic data by route

    Returns:
        Dictionary with correlation results
    """
    # Merge service and demographic data
    merged = service_df.merge(demographic_df, on='route_id', how='inner')

    if merged.empty:
        return {'error': 'No matching routes between service and demographic data'}

    # Define service metrics of interest
    service_metrics = [
        'service_score', 'mean_delay', 'on_time_performance',
        'headway_consistency'
    ]

    # Define demographic variables of interest
    demographic_vars = [
        'pct_minority', 'pct_below_poverty', 'median_household_income',
        'pct_no_vehicle', 'pct_transit_commuters'
    ]

    results = {
        'sample_size': len(merged),
        'correlations': {},
        'significant_findings': []
    }

    for service_metric in service_metrics:
        if service_metric not in merged.columns:
            continue

        results['correlations'][service_metric] = {}

        for demo_var in demographic_vars:
            if demo_var not in merged.columns:
                continue

            # Remove NaN values for correlation
            valid_data = merged[[service_metric, demo_var]].dropna()

            if len(valid_data) < 3:
                continue

            # Calculate Pearson correlation
            corr, p_value = stats.pearsonr(
                valid_data[service_metric],
                valid_data[demo_var]
            )

            results['correlations'][service_metric][demo_var] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            # Track significant findings
            if p_value < 0.05:
                direction = "positive" if corr > 0 else "negative"
                results['significant_findings'].append({
                    'service_metric': service_metric,
                    'demographic_var': demo_var,
                    'correlation': corr,
                    'p_value': p_value,
                    'interpretation': (
                        f"{direction} correlation between {service_metric} "
                        f"and {demo_var} (r={corr:.3f}, p={p_value:.4f})"
                    )
                })

    return results


def run_equity_analysis_with_target_routes(service_df: pd.DataFrame) -> Dict:
    """
    Analyze equity implications using target routes as proxy for
    underserved communities.

    Target routes (from Livable Streets report) serve predominantly
    low-income and minority communities.

    Args:
        service_df: Service level data by route

    Returns:
        Dictionary with equity analysis results
    """
    if service_df.empty:
        return {'error': 'No service data provided'}

    target_df = service_df[service_df['is_target_route']]
    other_df = service_df[~service_df['is_target_route']]

    if target_df.empty or other_df.empty:
        return {'error': 'Insufficient data for comparison'}

    metrics_to_compare = [
        'service_score', 'mean_delay', 'on_time_performance',
        'median_delay', 'std_delay'
    ]

    results = {
        'target_route_count': len(target_df),
        'other_route_count': len(other_df),
        'comparisons': {},
        'statistical_tests': {}
    }

    for metric in metrics_to_compare:
        if metric not in service_df.columns:
            continue

        target_values = target_df[metric].dropna()
        other_values = other_df[metric].dropna()

        if len(target_values) < 2 or len(other_values) < 2:
            continue

        # Basic comparison
        results['comparisons'][metric] = {
            'target_mean': target_values.mean(),
            'other_mean': other_values.mean(),
            'difference': target_values.mean() - other_values.mean(),
            'target_median': target_values.median(),
            'other_median': other_values.median(),
        }

        # Statistical test (Mann-Whitney U for non-parametric comparison)
        try:
            stat, p_value = stats.mannwhitneyu(
                target_values, other_values, alternative='two-sided'
            )
            results['statistical_tests'][metric] = {
                'test': 'Mann-Whitney U',
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            results['statistical_tests'][metric] = {'error': str(e)}

    # Summary interpretation
    results['summary'] = interpret_equity_results(results)

    return results


def interpret_equity_results(results: Dict) -> str:
    """
    Generate human-readable interpretation of equity analysis results.

    Args:
        results: Results from equity analysis

    Returns:
        Summary interpretation string
    """
    interpretations = []

    if 'comparisons' in results and 'service_score' in results['comparisons']:
        diff = results['comparisons']['service_score']['difference']
        if diff < -5:
            interpretations.append(
                f"Target routes (serving underserved communities) have notably "
                f"lower service scores ({abs(diff):.1f} points below other routes)."
            )
        elif diff > 5:
            interpretations.append(
                f"Target routes (serving underserved communities) have higher "
                f"service scores ({diff:.1f} points above other routes)."
            )
        else:
            interpretations.append(
                "Service scores are relatively similar between target routes "
                "and other routes."
            )

    if 'comparisons' in results and 'mean_delay' in results['comparisons']:
        delay_diff = results['comparisons']['mean_delay']['difference']
        if delay_diff > 1:
            interpretations.append(
                f"Target routes experience {delay_diff:.1f} minutes more delay "
                "on average compared to other routes."
            )
        elif delay_diff < -1:
            interpretations.append(
                f"Target routes experience {abs(delay_diff):.1f} minutes less delay "
                "on average compared to other routes."
            )

    if 'statistical_tests' in results:
        significant_metrics = [
            m for m, t in results['statistical_tests'].items()
            if isinstance(t, dict) and t.get('significant')
        ]
        if significant_metrics:
            interpretations.append(
                f"Statistically significant differences found in: "
                f"{', '.join(significant_metrics)}"
            )

    return " ".join(interpretations) if interpretations else "Insufficient data for interpretation."


def run_full_equity_analysis(service_df: pd.DataFrame,
                             output_dir: Path = None) -> Dict:
    """
    Run complete equity analysis.

    Args:
        service_df: Service level data by route
        output_dir: Directory to save results

    Returns:
        Dictionary with all equity analysis results
    """
    if output_dir is None:
        output_dir = DATA_PROCESSED / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Running Equity Analysis")
    print("=" * 60)

    results = {}

    # Target route analysis (always available)
    print("\n[1/3] Analyzing target routes as equity proxy...")
    target_analysis = run_equity_analysis_with_target_routes(service_df)
    results['target_route_analysis'] = target_analysis

    # Save target route comparison
    if 'comparisons' in target_analysis:
        comparison_df = pd.DataFrame([
            {'metric': m, **v}
            for m, v in target_analysis['comparisons'].items()
        ])
        comparison_df.to_csv(output_dir / "equity_target_route_comparison.csv", index=False)

    # Try to load and use demographic data
    print("\n[2/3] Loading demographic data...")
    demographics = load_demographic_data()

    if demographics is not None:
        print(f"  Loaded {len(demographics)} geographic areas")

        route_geo = load_route_geography_mapping()
        if route_geo is not None:
            print("\n[3/3] Running demographic correlation analysis...")
            route_profiles = create_route_demographic_profile(route_geo, demographics)
            correlation_results = calculate_correlation_analysis(service_df, route_profiles)
            results['correlation_analysis'] = correlation_results

            # Save correlation results
            if 'correlations' in correlation_results:
                corr_records = []
                for service_metric, demo_corrs in correlation_results['correlations'].items():
                    for demo_var, corr_data in demo_corrs.items():
                        corr_records.append({
                            'service_metric': service_metric,
                            'demographic_variable': demo_var,
                            **corr_data
                        })
                if corr_records:
                    pd.DataFrame(corr_records).to_csv(
                        output_dir / "equity_correlations.csv", index=False
                    )
        else:
            print("  Skipping demographic correlation (no route mapping)")
    else:
        print("  Demographic data not available")
        print("  Using target routes as proxy for underserved communities")

    print("\n" + "=" * 60)
    print("Equity Analysis Complete")
    print("=" * 60)

    if 'target_route_analysis' in results:
        print(f"\nSummary: {results['target_route_analysis'].get('summary', 'N/A')}")

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    # Load service level data
    service_file = DATA_PROCESSED / "analysis_results" / "service_levels_by_route.csv"

    if service_file.exists():
        service_df = pd.read_csv(service_file)
        results = run_full_equity_analysis(service_df)
    else:
        print("Service level data not found. Please run service_level.py first.")
