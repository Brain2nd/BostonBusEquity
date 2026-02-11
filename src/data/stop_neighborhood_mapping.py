"""
Stop to Neighborhood Mapping Module

Maps MBTA bus stops to Boston neighborhoods using coordinate-based matching.
Uses approximate neighborhood boundaries based on known coordinates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED

# GTFS stops file
GTFS_DIR = DATA_RAW / "gtfs"
STOPS_FILE = GTFS_DIR / "stops.txt"

# Approximate Boston neighborhood boundaries (lat_min, lat_max, lon_min, lon_max)
# These are rough approximations based on neighborhood geography
NEIGHBORHOOD_BOUNDS = {
    'Allston': (42.345, 42.365, -71.145, -71.125),
    'Back Bay': (42.345, 42.355, -71.090, -71.065),
    'Beacon Hill': (42.355, 42.365, -71.075, -71.055),
    'Brighton': (42.335, 42.365, -71.175, -71.140),
    'Charlestown': (42.370, 42.390, -71.075, -71.045),
    'Dorchester': (42.270, 42.320, -71.085, -71.035),
    'Downtown': (42.350, 42.365, -71.065, -71.050),
    'East Boston': (42.365, 42.395, -71.040, -70.985),
    'Fenway': (42.335, 42.350, -71.110, -71.085),
    'Hyde Park': (42.245, 42.275, -71.145, -71.105),
    'Jamaica Plain': (42.295, 42.325, -71.130, -71.095),
    'Longwood': (42.330, 42.345, -71.115, -71.095),
    'Mattapan': (42.260, 42.285, -71.105, -71.065),
    'Mission Hill': (42.325, 42.340, -71.110, -71.090),
    'North End': (42.360, 42.370, -71.060, -71.045),
    'Roslindale': (42.275, 42.300, -71.145, -71.115),
    'Roxbury': (42.310, 42.340, -71.095, -71.065),
    'South Boston': (42.325, 42.350, -71.060, -71.025),
    'South Boston Waterfront': (42.335, 42.360, -71.055, -71.030),
    'South End': (42.335, 42.350, -71.085, -71.060),
    'West End': (42.360, 42.370, -71.070, -71.055),
    'West Roxbury': (42.270, 42.295, -71.185, -71.145),
}

# Center points for each neighborhood (for distance-based fallback)
NEIGHBORHOOD_CENTERS = {
    'Allston': (42.355, -71.135),
    'Back Bay': (42.350, -71.078),
    'Beacon Hill': (42.359, -71.065),
    'Brighton': (42.350, -71.158),
    'Charlestown': (42.380, -71.060),
    'Dorchester': (42.295, -71.060),
    'Downtown': (42.357, -71.058),
    'East Boston': (42.380, -71.015),
    'Fenway': (42.343, -71.098),
    'Hyde Park': (42.260, -71.125),
    'Jamaica Plain': (42.310, -71.112),
    'Longwood': (42.338, -71.105),
    'Mattapan': (42.272, -71.085),
    'Mission Hill': (42.332, -71.100),
    'North End': (42.365, -71.053),
    'Roslindale': (42.288, -71.130),
    'Roxbury': (42.325, -71.080),
    'South Boston': (42.338, -71.043),
    'South Boston Waterfront': (42.348, -71.042),
    'South End': (42.342, -71.073),
    'West End': (42.365, -71.063),
    'West Roxbury': (42.282, -71.165),
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the haversine distance between two points in km."""
    R = 6371  # Earth's radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def point_in_bounds(lat: float, lon: float, bounds: Tuple[float, float, float, float]) -> bool:
    """Check if a point is within the given bounds."""
    lat_min, lat_max, lon_min, lon_max = bounds
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def find_neighborhood(lat: float, lon: float) -> Optional[str]:
    """
    Find the neighborhood for a given latitude/longitude.

    First tries bounding box match, then falls back to nearest center.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Neighborhood name or None if outside Boston
    """
    # First, try bounding box match
    for neighborhood, bounds in NEIGHBORHOOD_BOUNDS.items():
        if point_in_bounds(lat, lon, bounds):
            return neighborhood

    # Fallback: find nearest neighborhood center (within 3km)
    min_dist = float('inf')
    nearest = None

    for neighborhood, (center_lat, center_lon) in NEIGHBORHOOD_CENTERS.items():
        dist = haversine_distance(lat, lon, center_lat, center_lon)
        if dist < min_dist:
            min_dist = dist
            nearest = neighborhood

    # Only return if within 3km of a neighborhood center
    if min_dist <= 3.0:
        return nearest

    return None


def load_stops() -> pd.DataFrame:
    """Load GTFS stops data."""
    if not STOPS_FILE.exists():
        print(f"Warning: Stops file not found: {STOPS_FILE}")
        return pd.DataFrame()

    return pd.read_csv(STOPS_FILE)


def create_stop_neighborhood_mapping() -> pd.DataFrame:
    """
    Create mapping from stop_id to neighborhood.

    Returns:
        DataFrame with stop_id and neighborhood columns
    """
    stops = load_stops()
    if stops.empty:
        return pd.DataFrame()

    # Filter to Boston municipality stops
    boston_stops = stops[stops['municipality'] == 'Boston'].copy()

    # Map each stop to a neighborhood
    neighborhoods = []
    for _, row in boston_stops.iterrows():
        neighborhood = find_neighborhood(row['stop_lat'], row['stop_lon'])
        neighborhoods.append(neighborhood)

    boston_stops['neighborhood'] = neighborhoods

    # Create mapping DataFrame
    mapping = boston_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'neighborhood']].copy()

    # Report coverage
    mapped = mapping['neighborhood'].notna().sum()
    total = len(mapping)
    print(f"Mapped {mapped} of {total} Boston stops to neighborhoods ({mapped/total*100:.1f}%)")

    return mapping


def get_stop_neighborhood_dict() -> Dict[str, str]:
    """
    Get a dictionary mapping stop_id to neighborhood.

    Returns:
        Dict with stop_id as key, neighborhood as value
    """
    mapping = create_stop_neighborhood_mapping()
    if mapping.empty:
        return {}

    # Convert to dict, excluding unmapped stops
    valid = mapping[mapping['neighborhood'].notna()]
    return dict(zip(valid['stop_id'].astype(str), valid['neighborhood']))


def save_stop_mapping(output_dir: Path = None) -> Path:
    """
    Save stop-neighborhood mapping to CSV.

    Args:
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = DATA_PROCESSED

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = create_stop_neighborhood_mapping()
    if mapping.empty:
        print("No mapping to save")
        return None

    output_path = output_dir / "stop_neighborhood_mapping.csv"
    mapping.to_csv(output_path, index=False)
    print(f"Saved stop mapping to: {output_path}")

    return output_path


def analyze_stops_by_neighborhood(mapping: pd.DataFrame = None) -> pd.DataFrame:
    """
    Analyze the distribution of stops by neighborhood.

    Args:
        mapping: Stop mapping DataFrame (loads if not provided)

    Returns:
        DataFrame with stop counts per neighborhood
    """
    if mapping is None:
        mapping = create_stop_neighborhood_mapping()

    if mapping.empty:
        return pd.DataFrame()

    # Count stops per neighborhood
    counts = mapping.groupby('neighborhood').agg({
        'stop_id': 'count'
    }).reset_index()
    counts.columns = ['neighborhood', 'stop_count']

    return counts.sort_values('stop_count', ascending=False)


if __name__ == "__main__":
    print("=" * 60)
    print("Stop to Neighborhood Mapping")
    print("=" * 60)

    # Create and save mapping
    mapping = create_stop_neighborhood_mapping()

    if not mapping.empty:
        print("\n--- Stops per Neighborhood ---")
        by_neighborhood = analyze_stops_by_neighborhood(mapping)
        print(by_neighborhood.to_string())

        print("\n--- Unmapped Stops Sample ---")
        unmapped = mapping[mapping['neighborhood'].isna()]
        if not unmapped.empty:
            print(f"Total unmapped: {len(unmapped)}")
            print(unmapped[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].head(10).to_string())

        # Save mapping
        save_stop_mapping()
