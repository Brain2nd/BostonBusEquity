"""
Analysis modules for Boston Bus Equity Project
"""

from .delay_analysis import (
    analyze_delays_by_route,
    analyze_delays_by_time,
    compare_target_vs_other_routes,
    calculate_wait_time_stats,
    run_full_delay_analysis
)

from .travel_time import (
    analyze_travel_times_by_route,
    analyze_travel_times_by_time_of_day,
    run_travel_time_analysis
)

from .service_level import (
    analyze_service_levels_by_route,
    identify_service_disparities,
    run_service_level_analysis
)

from .equity import (
    run_equity_analysis_with_target_routes,
    run_full_equity_analysis
)
