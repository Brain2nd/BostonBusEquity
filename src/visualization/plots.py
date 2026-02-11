"""
Visualization Module for Boston Bus Equity Project

Creates visualizations for all research questions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TARGET_ROUTES, FIGURES_DIR, VIZ_CONFIG
)

# Set default style
plt.style.use(VIZ_CONFIG['style'])


def setup_figure(figsize: tuple = None, dpi: int = None):
    """
    Create and configure a figure with default settings.

    Args:
        figsize: Figure size (width, height)
        dpi: Dots per inch

    Returns:
        Figure and Axes objects
    """
    if figsize is None:
        figsize = VIZ_CONFIG['figure_size']
    if dpi is None:
        dpi = VIZ_CONFIG['dpi']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def save_figure(fig, filename: str, output_dir: Path = None):
    """
    Save figure to file.

    Args:
        fig: Figure to save
        filename: Output filename
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    fig.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_delay_distribution(delay_df: pd.DataFrame,
                            output_dir: Path = None) -> None:
    """
    Plot distribution of delays across all routes.

    Args:
        delay_df: DataFrame with delay data
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    delays = delay_df['mean_delay'].dropna()
    axes[0].hist(delays, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(delays.mean(), color='red', linestyle='--',
                    label=f'Mean: {delays.mean():.1f} min')
    axes[0].axvline(delays.median(), color='green', linestyle='--',
                    label=f'Median: {delays.median():.1f} min')
    axes[0].set_xlabel('Average Delay (minutes)')
    axes[0].set_ylabel('Number of Routes')
    axes[0].set_title('Distribution of Average Delays by Route')
    axes[0].legend()

    # Box plot comparing target vs other routes
    target_delays = delay_df[delay_df['is_target_route']]['mean_delay']
    other_delays = delay_df[~delay_df['is_target_route']]['mean_delay']

    bp = axes[1].boxplot([target_delays.dropna(), other_delays.dropna()],
                         labels=['Target Routes', 'Other Routes'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff7f7f')
    bp['boxes'][1].set_facecolor('#7fbfff')

    axes[1].set_ylabel('Average Delay (minutes)')
    axes[1].set_title('Delay Comparison: Target vs Other Routes')

    plt.tight_layout()
    save_figure(fig, 'delay_distribution.png', output_dir)


def plot_delays_by_route(delay_df: pd.DataFrame,
                         top_n: int = 20,
                         output_dir: Path = None) -> None:
    """
    Plot delays for top N routes.

    Args:
        delay_df: DataFrame with delay data per route
        top_n: Number of routes to show
        output_dir: Output directory
    """
    fig, ax = setup_figure(figsize=(14, 8))

    # Sort and get top routes
    sorted_df = delay_df.nlargest(top_n, 'mean_delay')

    colors = ['#ff7f7f' if r in TARGET_ROUTES else '#7fbfff'
              for r in sorted_df['route_id']]

    bars = ax.barh(sorted_df['route_id'].astype(str), sorted_df['mean_delay'],
                   color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Average Delay (minutes)')
    ax.set_ylabel('Route')
    ax.set_title(f'Top {top_n} Routes by Average Delay')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff7f7f', edgecolor='black', label='Target Routes'),
        Patch(facecolor='#7fbfff', edgecolor='black', label='Other Routes')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.invert_yaxis()
    plt.tight_layout()
    save_figure(fig, 'delays_by_route.png', output_dir)


def plot_delays_by_hour(hourly_df: pd.DataFrame,
                        output_dir: Path = None) -> None:
    """
    Plot delay patterns by hour of day.

    Args:
        hourly_df: DataFrame with hourly delay data
        output_dir: Output directory
    """
    fig, ax = setup_figure()

    ax.plot(hourly_df['hour'], hourly_df['mean_delay'],
            marker='o', linewidth=2, markersize=8)
    ax.fill_between(hourly_df['hour'], hourly_df['mean_delay'], alpha=0.3)

    # Mark rush hours
    ax.axvspan(7, 9, alpha=0.2, color='red', label='Morning Rush')
    ax.axvspan(16, 19, alpha=0.2, color='orange', label='Evening Rush')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Delay (minutes)')
    ax.set_title('Average Delay by Hour of Day')
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'delays_by_hour.png', output_dir)


def plot_delays_by_day_of_week(dow_df: pd.DataFrame,
                               output_dir: Path = None) -> None:
    """
    Plot delay patterns by day of week.

    Args:
        dow_df: DataFrame with day-of-week delay data
        output_dir: Output directory
    """
    fig, ax = setup_figure(figsize=(10, 6))

    colors = ['#ff9999' if d in [5, 6] else '#66b3ff'
              for d in dow_df['day_of_week']]

    bars = ax.bar(dow_df['day_name'], dow_df['mean_delay'],
                  color=colors, edgecolor='black', alpha=0.8)

    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Delay (minutes)')
    ax.set_title('Average Delay by Day of Week')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#66b3ff', label='Weekday'),
        Patch(facecolor='#ff9999', label='Weekend')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    save_figure(fig, 'delays_by_day.png', output_dir)


def plot_monthly_trends(monthly_df: pd.DataFrame,
                        output_dir: Path = None) -> None:
    """
    Plot monthly delay trends.

    Args:
        monthly_df: DataFrame with monthly delay data
        output_dir: Output directory
    """
    fig, ax = setup_figure(figsize=(14, 6))

    # Convert year_month to datetime for proper plotting
    monthly_df = monthly_df.copy()
    monthly_df['date'] = pd.to_datetime(monthly_df['year_month'] + '-01')

    ax.plot(monthly_df['date'], monthly_df['mean_delay'],
            marker='o', linewidth=2, markersize=4, label='Mean Delay')

    ax.fill_between(monthly_df['date'], monthly_df['mean_delay'], alpha=0.3)

    ax.set_xlabel('Month')
    ax.set_ylabel('Average Delay (minutes)')
    ax.set_title('Monthly Delay Trends (2020-2024)')

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'monthly_delay_trends.png', output_dir)


def plot_on_time_performance(service_df: pd.DataFrame,
                             output_dir: Path = None) -> None:
    """
    Plot on-time performance by route.

    Args:
        service_df: DataFrame with service level data
        output_dir: Output directory
    """
    fig, ax = setup_figure(figsize=(14, 8))

    # Sort by on-time performance
    sorted_df = service_df.nsmallest(20, 'on_time_performance')

    colors = ['#ff7f7f' if r in TARGET_ROUTES else '#7fbfff'
              for r in sorted_df['route_id']]

    bars = ax.barh(sorted_df['route_id'].astype(str),
                   sorted_df['on_time_performance'],
                   color=colors, edgecolor='black', alpha=0.8)

    ax.axvline(x=service_df['on_time_performance'].mean(),
               color='red', linestyle='--',
               label=f"Average: {service_df['on_time_performance'].mean():.1f}%")

    ax.set_xlabel('On-Time Performance (%)')
    ax.set_ylabel('Route')
    ax.set_title('Bottom 20 Routes by On-Time Performance')
    ax.legend()

    ax.invert_yaxis()
    plt.tight_layout()
    save_figure(fig, 'on_time_performance.png', output_dir)


def plot_service_score_comparison(service_df: pd.DataFrame,
                                  output_dir: Path = None) -> None:
    """
    Plot service score comparison between target and other routes.

    Args:
        service_df: DataFrame with service level data
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Service score distribution
    target_scores = service_df[service_df['is_target_route']]['service_score']
    other_scores = service_df[~service_df['is_target_route']]['service_score']

    axes[0].hist([target_scores, other_scores], bins=20,
                 label=['Target Routes', 'Other Routes'],
                 alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Service Score')
    axes[0].set_ylabel('Number of Routes')
    axes[0].set_title('Service Score Distribution')
    axes[0].legend()

    # Box plot
    bp = axes[1].boxplot([target_scores.dropna(), other_scores.dropna()],
                         labels=['Target Routes', 'Other Routes'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('#ff7f7f')
    bp['boxes'][1].set_facecolor('#7fbfff')

    axes[1].set_ylabel('Service Score')
    axes[1].set_title('Service Score: Target vs Other Routes')

    # Add means
    target_mean = target_scores.mean()
    other_mean = other_scores.mean()
    axes[1].axhline(y=target_mean, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=other_mean, color='blue', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, 'service_score_comparison.png', output_dir)


def plot_travel_times(travel_df: pd.DataFrame,
                      output_dir: Path = None) -> None:
    """
    Plot travel time analysis.

    Args:
        travel_df: DataFrame with travel time data
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distribution of travel times
    axes[0].hist(travel_df['mean_travel_time'].dropna(), bins=30,
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(travel_df['mean_travel_time'].mean(), color='red',
                    linestyle='--', label=f"Mean: {travel_df['mean_travel_time'].mean():.1f} min")
    axes[0].set_xlabel('Average Travel Time (minutes)')
    axes[0].set_ylabel('Number of Routes')
    axes[0].set_title('Distribution of Route Travel Times')
    axes[0].legend()

    # Top 15 longest routes
    top_routes = travel_df.nlargest(15, 'mean_travel_time')

    colors = ['#ff7f7f' if r in TARGET_ROUTES else '#7fbfff'
              for r in top_routes['route_id']]

    axes[1].barh(top_routes['route_id'].astype(str),
                 top_routes['mean_travel_time'],
                 color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Average Travel Time (minutes)')
    axes[1].set_ylabel('Route')
    axes[1].set_title('Top 15 Longest Routes')
    axes[1].invert_yaxis()

    plt.tight_layout()
    save_figure(fig, 'travel_times.png', output_dir)


def plot_wait_time_comparison(wait_stats: Dict,
                              output_dir: Path = None) -> None:
    """
    Plot wait time comparison (on-time vs delayed).

    Args:
        wait_stats: Dictionary with wait time statistics
        output_dir: Output directory
    """
    fig, ax = setup_figure(figsize=(10, 6))

    categories = ['On-Time Buses', 'Delayed Buses', 'Overall']
    means = [
        wait_stats['on_time'].get('mean', 0),
        wait_stats['delayed'].get('mean', 0),
        wait_stats['all'].get('mean', 0)
    ]

    colors = ['#66c2a5', '#fc8d62', '#8da0cb']
    bars = ax.bar(categories, means, color=colors, edgecolor='black', alpha=0.8)

    ax.set_ylabel('Expected Wait Time (minutes)')
    ax.set_title('Expected Wait Time: On-Time vs Delayed Buses')

    # Add value labels
    for bar, mean in zip(bars, means):
        if mean:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'wait_time_comparison.png', output_dir)


def plot_target_routes_summary(delay_df: pd.DataFrame,
                               output_dir: Path = None) -> None:
    """
    Plot summary of target routes performance.

    Args:
        delay_df: DataFrame with delay data for target routes
        output_dir: Output directory
    """
    target_df = delay_df[delay_df['route_id'].isin(TARGET_ROUTES)]

    if target_df.empty:
        print("  No target route data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Mean delay by target route
    sorted_target = target_df.sort_values('mean_delay', ascending=True)
    axes[0, 0].barh(sorted_target['route_id'].astype(str),
                    sorted_target['mean_delay'],
                    color='#ff7f7f', edgecolor='black', alpha=0.8)
    axes[0, 0].set_xlabel('Average Delay (minutes)')
    axes[0, 0].set_title('Average Delay by Target Route')

    # On-time performance
    sorted_otp = target_df.sort_values('on_time_performance', ascending=False)
    if 'on_time_performance' in sorted_otp.columns:
        axes[0, 1].barh(sorted_otp['route_id'].astype(str),
                        sorted_otp['on_time_performance'],
                        color='#66c2a5', edgecolor='black', alpha=0.8)
        axes[0, 1].set_xlabel('On-Time Performance (%)')
        axes[0, 1].set_title('On-Time Performance by Target Route')

    # Delay category breakdown
    if all(col in target_df.columns for col in ['on_time_pct', 'minor_delay_pct', 'moderate_delay_pct', 'major_delay_pct']):
        categories = ['on_time_pct', 'minor_delay_pct', 'moderate_delay_pct', 'major_delay_pct']
        cat_labels = ['On Time', 'Minor (1-5 min)', 'Moderate (5-10 min)', 'Major (>10 min)']

        bottom = np.zeros(len(target_df))
        colors = ['#66c2a5', '#fc8d62', '#e78ac3', '#e5c494']

        for cat, label, color in zip(categories, cat_labels, colors):
            if cat in target_df.columns:
                axes[1, 0].bar(target_df['route_id'].astype(str),
                               target_df[cat], bottom=bottom,
                               label=label, color=color, alpha=0.8)
                bottom += target_df[cat].values

        axes[1, 0].set_xlabel('Route')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].set_title('Delay Category Breakdown by Target Route')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=45)

    # Service score if available
    if 'service_score' in target_df.columns:
        sorted_score = target_df.sort_values('service_score', ascending=True)
        axes[1, 1].barh(sorted_score['route_id'].astype(str),
                        sorted_score['service_score'],
                        color='#8da0cb', edgecolor='black', alpha=0.8)
        axes[1, 1].set_xlabel('Service Score')
        axes[1, 1].set_title('Service Score by Target Route')

    plt.tight_layout()
    save_figure(fig, 'target_routes_summary.png', output_dir)


def generate_all_visualizations(results: Dict, output_dir: Path = None) -> None:
    """
    Generate all visualizations from analysis results.

    Args:
        results: Dictionary with all analysis results
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    # Delay visualizations
    if 'delay_analysis' in results:
        delay_results = results['delay_analysis']

        print("\n[1/6] Delay distribution plots...")
        if 'citywide_by_route' in delay_results:
            plot_delay_distribution(delay_results['citywide_by_route'], output_dir)
            plot_delays_by_route(delay_results['citywide_by_route'], output_dir=output_dir)

        if 'time_analysis' in delay_results:
            if 'hourly' in delay_results['time_analysis']:
                plot_delays_by_hour(delay_results['time_analysis']['hourly'], output_dir)
            if 'day_of_week' in delay_results['time_analysis']:
                plot_delays_by_day_of_week(delay_results['time_analysis']['day_of_week'], output_dir)
            if 'monthly' in delay_results['time_analysis']:
                plot_monthly_trends(delay_results['time_analysis']['monthly'], output_dir)

        if 'wait_times' in delay_results:
            plot_wait_time_comparison(delay_results['wait_times'], output_dir)

    # Service level visualizations
    if 'service_analysis' in results:
        service_results = results['service_analysis']

        print("\n[2/6] Service level plots...")
        if 'by_route' in service_results:
            plot_on_time_performance(service_results['by_route'], output_dir)
            plot_service_score_comparison(service_results['by_route'], output_dir)

    # Travel time visualizations
    if 'travel_time_analysis' in results:
        travel_results = results['travel_time_analysis']

        print("\n[3/6] Travel time plots...")
        if 'by_route' in travel_results:
            plot_travel_times(travel_results['by_route'], output_dir)

    # Target routes summary
    print("\n[4/6] Target routes summary...")
    if 'delay_analysis' in results and 'citywide_by_route' in results['delay_analysis']:
        plot_target_routes_summary(results['delay_analysis']['citywide_by_route'], output_dir)

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)


def plot_ridership_trends(ridership_df: pd.DataFrame,
                          output_dir: Path = None) -> None:
    """
    Plot ridership trends pre vs post pandemic (Q1).

    Args:
        ridership_df: DataFrame with yearly ridership data
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Yearly ridership trend
    years = ridership_df['year']
    boardings = ridership_df['total_boardings']

    # Color by period
    colors = []
    for y in years:
        if y <= 2019:
            colors.append('#2ecc71')  # Green for pre-pandemic
        elif y == 2020:
            colors.append('#e74c3c')  # Red for pandemic year
        else:
            colors.append('#3498db')  # Blue for post-pandemic

    axes[0].bar(years.astype(str), boardings / 1e6, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Total Boardings (Millions)')
    axes[0].set_title('Annual Bus Ridership Trend')
    axes[0].tick_params(axis='x', rotation=45)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Pre-Pandemic (2016-2019)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Pandemic (2020)'),
        Patch(facecolor='#3498db', edgecolor='black', label='Post-Pandemic (2021-2024)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')

    # Pre vs Post comparison
    pre_pandemic = ridership_df[ridership_df['year'] <= 2019]['total_boardings'].mean()
    post_pandemic = ridership_df[ridership_df['year'] >= 2021]['total_boardings'].mean()

    categories = ['Pre-Pandemic\n(2016-2019)', 'Post-Pandemic\n(2021-2024)']
    values = [pre_pandemic / 1e6, post_pandemic / 1e6]
    colors_bar = ['#2ecc71', '#3498db']

    bars = axes[1].bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.8)

    # Add percentage change annotation
    pct_change = ((post_pandemic - pre_pandemic) / pre_pandemic) * 100
    axes[1].annotate(f'{pct_change:+.1f}%',
                     xy=(1, values[1]), xytext=(1.2, (values[0] + values[1]) / 2),
                     fontsize=14, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))

    axes[1].set_ylabel('Average Annual Boardings (Millions)')
    axes[1].set_title('Pre vs Post Pandemic Ridership Comparison')

    # Add value labels
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}M', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'ridership_pre_post_pandemic.png', output_dir)


def plot_ridership_by_route(route_df: pd.DataFrame,
                            top_n: int = 20,
                            output_dir: Path = None) -> None:
    """
    Plot ridership by route with pre/post pandemic comparison.

    Args:
        route_df: DataFrame with route ridership data
        top_n: Number of top routes to show
        output_dir: Output directory
    """
    fig, ax = setup_figure(figsize=(14, 10))

    # Sort and get top routes by total ridership change
    sorted_df = route_df.nlargest(top_n, 'pre_pandemic_avg_yearly')

    y_pos = range(len(sorted_df))
    width = 0.35

    pre_vals = sorted_df['pre_pandemic_avg_yearly'] / 1e3
    post_vals = sorted_df['post_pandemic_avg_yearly'] / 1e3

    bars1 = ax.barh([y - width/2 for y in y_pos], pre_vals,
                    width, label='Pre-Pandemic', color='#2ecc71', alpha=0.8)
    bars2 = ax.barh([y + width/2 for y in y_pos], post_vals,
                    width, label='Post-Pandemic', color='#3498db', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df['route_id'].astype(str))
    ax.set_xlabel('Average Annual Boardings (Thousands)')
    ax.set_ylabel('Route')
    ax.set_title(f'Top {top_n} Routes: Pre vs Post Pandemic Ridership')
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    save_figure(fig, 'ridership_by_route_comparison.png', output_dir)


def plot_demographic_correlations(corr_df: pd.DataFrame,
                                  output_dir: Path = None) -> None:
    """
    Plot demographic correlations heatmap (Q7).

    Args:
        corr_df: DataFrame with correlation results
        output_dir: Output directory
    """
    fig, ax = setup_figure(figsize=(12, 8))

    # Pivot the data for heatmap
    pivot_df = corr_df.pivot(index='service_metric', columns='demographic_variable', values='correlation')

    # Create heatmap
    im = ax.imshow(pivot_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')

    # Set ticks
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_df.index)

    # Add correlation values as text
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

    ax.set_title('Service Metrics vs Demographic Variables\n(Correlation Coefficients)')

    plt.tight_layout()
    save_figure(fig, 'demographic_correlations_heatmap.png', output_dir)


def plot_demographic_comparison(merged_df: pd.DataFrame,
                                output_dir: Path = None) -> None:
    """
    Plot service comparison between high-minority and other areas (Q7).

    Args:
        merged_df: DataFrame with merged service and demographic data
        output_dir: Output directory
    """
    if 'serves_high_minority' not in merged_df.columns:
        print("  No minority classification available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    high_minority = merged_df[merged_df['serves_high_minority']]
    other = merged_df[~merged_df['serves_high_minority']]

    # Mean delay comparison
    categories = ['High-Minority\nAreas', 'Other\nAreas']

    # Plot 1: Mean Delay
    vals = [high_minority['mean_delay'].mean(), other['mean_delay'].mean()]
    colors = ['#e74c3c', '#3498db']
    bars = axes[0].bar(categories, vals, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('Average Delay (minutes)')
    axes[0].set_title('Mean Delay by Area Type')
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: On-Time Performance
    if 'on_time_performance' in merged_df.columns:
        vals = [high_minority['on_time_performance'].mean(), other['on_time_performance'].mean()]
        bars = axes[1].bar(categories, vals, color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_ylabel('On-Time Performance (%)')
        axes[1].set_title('On-Time Performance by Area Type')
        for bar, val in zip(bars, vals):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Scatter plot - Minority % vs Delay
    if 'minority_pct' in merged_df.columns:
        axes[2].scatter(merged_df['minority_pct'] * 100, merged_df['mean_delay'],
                        alpha=0.6, edgecolors='black', s=50)
        axes[2].set_xlabel('Minority Population (%)')
        axes[2].set_ylabel('Mean Delay (minutes)')
        axes[2].set_title('Delay vs Minority Population')

        # Add trend line
        z = np.polyfit(merged_df['minority_pct'] * 100, merged_df['mean_delay'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged_df['minority_pct'].min() * 100, merged_df['minority_pct'].max() * 100, 100)
        axes[2].plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend')
        axes[2].legend()

    plt.tight_layout()
    save_figure(fig, 'demographic_service_comparison.png', output_dir)


def plot_neighborhood_demographics(demo_df: pd.DataFrame,
                                   output_dir: Path = None) -> None:
    """
    Plot neighborhood demographic summary (Q7).

    Args:
        demo_df: DataFrame with neighborhood demographics
        output_dir: Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Sort by minority percentage for visualization
    sorted_df = demo_df.sort_values('minority_pct', ascending=True)

    # Plot 1: Minority percentage by neighborhood
    colors = ['#e74c3c' if v else '#3498db' for v in sorted_df['high_minority']]
    axes[0, 0].barh(sorted_df['neighborhood'], sorted_df['minority_pct'] * 100,
                    color=colors, edgecolor='black', alpha=0.8)
    axes[0, 0].axvline(x=50, color='red', linestyle='--', label='50% Threshold')
    axes[0, 0].set_xlabel('Minority Population (%)')
    axes[0, 0].set_title('Minority Population by Neighborhood')
    axes[0, 0].legend()

    # Plot 2: Median income by neighborhood
    sorted_income = demo_df.sort_values('median_income', ascending=True)
    colors = ['#e74c3c' if v else '#3498db' for v in sorted_income['low_income']]
    axes[0, 1].barh(sorted_income['neighborhood'], sorted_income['median_income'] / 1000,
                    color=colors, edgecolor='black', alpha=0.8)
    axes[0, 1].set_xlabel('Median Income ($K)')
    axes[0, 1].set_title('Median Household Income by Neighborhood')

    # Plot 3: Poverty rate by neighborhood
    sorted_poverty = demo_df.sort_values('poverty_rate', ascending=False)
    colors = ['#e74c3c' if v else '#3498db' for v in sorted_poverty['high_poverty']]
    axes[1, 0].barh(sorted_poverty['neighborhood'], sorted_poverty['poverty_rate'] * 100,
                    color=colors, edgecolor='black', alpha=0.8)
    axes[1, 0].set_xlabel('Poverty Rate (%)')
    axes[1, 0].set_title('Poverty Rate by Neighborhood')

    # Plot 4: Classification summary
    classifications = ['High Minority', 'Low Income', 'High Poverty', 'Vulnerable']
    counts = [
        demo_df['high_minority'].sum(),
        demo_df['low_income'].sum(),
        demo_df['high_poverty'].sum(),
        demo_df['vulnerable'].sum()
    ]
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#c0392b']
    bars = axes[1, 1].bar(classifications, counts, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 1].set_ylabel('Number of Neighborhoods')
    axes[1, 1].set_title(f'Neighborhood Classifications (Total: {len(demo_df)})')

    # Add value labels
    for bar, count in zip(bars, counts):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'neighborhood_demographics.png', output_dir)


if __name__ == "__main__":
    # Test with sample data
    print("Run this module through the main analysis script.")
