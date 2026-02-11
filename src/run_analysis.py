"""
Main Analysis Runner for Boston Bus Equity Project

This script runs the complete analysis pipeline:
1. Data loading and preprocessing
2. Delay analysis (Q3, Q4, Q5)
3. Travel time analysis (Q2)
4. Service level analysis (Q6)
5. Equity analysis (Q7)
6. Visualization generation

Usage:
    python run_analysis.py              # Run full analysis
    python run_analysis.py --quick      # Run with sample data
    python run_analysis.py --validate   # Run on validation set
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED, FIGURES_DIR,
    TRAIN_YEARS, VALIDATION_YEARS, TARGET_ROUTES
)
from data.load_data import get_data_summary
from analysis.delay_analysis import run_full_delay_analysis
from analysis.travel_time import run_travel_time_analysis
from analysis.service_level import run_service_level_analysis
from analysis.equity import run_full_equity_analysis
from visualization.plots import generate_all_visualizations


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_full_analysis(years: list = None, output_dir: Path = None):
    """
    Run the complete analysis pipeline.

    Args:
        years: Years to analyze
        output_dir: Directory for output files
    """
    if years is None:
        years = TRAIN_YEARS

    if output_dir is None:
        output_dir = DATA_PROCESSED / "analysis_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    print_header("BOSTON BUS EQUITY ANALYSIS")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Years: {years}")
    print(f"Target Routes: {TARGET_ROUTES}")
    print(f"Output Directory: {output_dir}")

    # Data summary
    print_header("DATA SUMMARY")
    summary = get_data_summary(years)
    print(f"Total files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_mb']:.1f} MB")
    for year, info in sorted(summary['files_by_year'].items()):
        print(f"  {year}: {info['count']} files ({info['size_mb']:.1f} MB)")

    all_results = {}

    # Delay Analysis (Q3, Q4, Q5)
    print_header("DELAY ANALYSIS (Q3, Q4, Q5)")
    delay_results = run_full_delay_analysis(years, output_dir)
    all_results['delay_analysis'] = delay_results

    # Travel Time Analysis (Q2)
    print_header("TRAVEL TIME ANALYSIS (Q2)")
    travel_results = run_travel_time_analysis(years, output_dir)
    all_results['travel_time_analysis'] = travel_results

    # Service Level Analysis (Q6)
    print_header("SERVICE LEVEL ANALYSIS (Q6)")
    service_results = run_service_level_analysis(years, output_dir)
    all_results['service_analysis'] = service_results

    # Equity Analysis (Q7)
    print_header("EQUITY ANALYSIS (Q7)")
    if 'by_route' in service_results:
        equity_results = run_full_equity_analysis(
            service_results['by_route'], output_dir
        )
        all_results['equity_analysis'] = equity_results
    else:
        print("Skipping equity analysis - service data not available")

    # Generate Visualizations
    print_header("VISUALIZATION GENERATION")
    figures_dir = FIGURES_DIR
    generate_all_visualizations(all_results, figures_dir)

    # Final Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print_header("ANALYSIS COMPLETE")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Figures saved to: {figures_dir}")

    # Print key findings
    print_header("KEY FINDINGS SUMMARY")

    if 'delay_analysis' in all_results:
        da = all_results['delay_analysis']
        if 'target_vs_other' in da:
            tvo = da['target_vs_other']
            target_delay = tvo['target_routes'].get('mean_delay', 0)
            other_delay = tvo['other_routes'].get('mean_delay', 0)
            print(f"\nDelay Comparison:")
            print(f"  - Target routes avg delay: {target_delay:.2f} minutes")
            print(f"  - Other routes avg delay: {other_delay:.2f} minutes")
            print(f"  - Difference: {target_delay - other_delay:+.2f} minutes")

        if 'wait_times' in da:
            wt = da['wait_times']
            print(f"\nWait Time Analysis:")
            print(f"  - On-time buses: {wt['on_time'].get('mean', 0):.1f} min avg wait")
            print(f"  - Delayed buses: {wt['delayed'].get('mean', 0):.1f} min avg wait")

    if 'service_analysis' in all_results:
        sa = all_results['service_analysis']
        if 'disparities' in sa:
            disp = sa['disparities']
            print(f"\nService Level Disparities:")
            print(f"  - Mean service score: {disp.get('mean_service_score', 0):.1f}/100")
            print(f"  - Routes below average: {disp.get('below_average_count', 0)}")
            if disp.get('target_vs_other_diff') is not None:
                diff = disp['target_vs_other_diff']
                direction = "higher" if diff > 0 else "lower"
                print(f"  - Target routes {abs(diff):.1f} points {direction}")

    if 'equity_analysis' in all_results:
        ea = all_results['equity_analysis']
        if 'target_route_analysis' in ea:
            print(f"\nEquity Analysis:")
            print(f"  {ea['target_route_analysis'].get('summary', 'N/A')}")

    return all_results


def run_quick_analysis():
    """Run analysis on a small sample for testing."""
    print_header("QUICK ANALYSIS MODE (Sample Data)")
    print("Running on 2024 data only with limited scope...")

    output_dir = DATA_PROCESSED / "quick_analysis"
    return run_full_analysis(years=["2024"], output_dir=output_dir)


def run_validation_analysis():
    """Run analysis on validation set (2025-2026)."""
    print_header("VALIDATION ANALYSIS MODE")
    print(f"Running on validation years: {VALIDATION_YEARS}")

    output_dir = DATA_PROCESSED / "validation_results"
    return run_full_analysis(years=VALIDATION_YEARS, output_dir=output_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Boston Bus Equity Analysis"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick analysis on sample data'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run analysis on validation set (2025-2026)'
    )
    parser.add_argument(
        '--years', nargs='+',
        help='Specific years to analyze (e.g., --years 2023 2024)'
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_analysis()
    elif args.validate:
        run_validation_analysis()
    elif args.years:
        run_full_analysis(years=args.years)
    else:
        run_full_analysis()


if __name__ == "__main__":
    main()
