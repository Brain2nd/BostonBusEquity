"""
Convert All Datasets to Parquet Format
======================================

This script converts all raw datasets to Parquet format with proper subset distinction:

1. arrival_departure - Bus arrival/departure times (2020-2026)
2. ridership - Bus ridership by trip/season (2016-2024)
3. survey - 2024 System-Wide Passenger Survey
4. gtfs_stops - GTFS bus stop locations
5. gtfs_routes - GTFS route information

Each dataset maintains its own schema and is saved as a separate Parquet file.

Author: Boston Bus Equity Team
Date: February 2025
"""

import pandas as pd
import os
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)


def convert_arrival_departure():
    """
    Convert Bus Arrival/Departure Times data to Parquet.

    Schema:
    - service_date: date
    - route_id: string
    - direction_id: int
    - half_trip_id: string
    - stop_id: int
    - time_point_id: string
    - time_point_order: int
    - point_type: string
    - standard_type: string
    - scheduled: datetime
    - actual: datetime
    - scheduled_headway: float
    - actual_headway: float
    - year: int (derived)
    - month: int (derived)
    """
    print("\n" + "="*60)
    print("Converting Arrival/Departure Data")
    print("="*60)

    arrival_dir = RAW_DATA_DIR / "arrival_departure"
    if not arrival_dir.exists():
        print("  Arrival/Departure directory not found, skipping...")
        return None

    all_csvs = list(arrival_dir.glob("**/*.csv"))
    print(f"  Found {len(all_csvs)} CSV files")

    if len(all_csvs) == 0:
        return None

    all_dfs = []

    for csv_path in sorted(all_csvs):
        try:
            # Determine year from path
            year = None
            for part in csv_path.parts:
                if '2020' in part: year = 2020
                elif '2021' in part: year = 2021
                elif '2022' in part: year = 2022
                elif '2023' in part: year = 2023
                elif '2024' in part: year = 2024
                elif '2025' in part: year = 2025
                elif '2026' in part: year = 2026

            # Read CSV
            df = pd.read_csv(csv_path, low_memory=False)

            # Clean BOM character from column names
            df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]

            # Normalize column names (2020 uses 'direction', later years use 'direction_id')
            if 'direction' in df.columns and 'direction_id' not in df.columns:
                df = df.rename(columns={'direction': 'direction_id'})

            # Clean service_date
            if 'service_date' in df.columns:
                df['service_date'] = df['service_date'].astype(str).str.replace('\ufeff', '', regex=False)
                df['service_date'] = pd.to_datetime(df['service_date'], errors='coerce', format='mixed')

            # Add year column
            df['year'] = year

            # Extract month if possible
            if 'service_date' in df.columns:
                df['month'] = df['service_date'].dt.month

            all_dfs.append(df)
            print(f"  Processed: {csv_path.name} ({len(df):,} rows)")

        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")

    if len(all_dfs) == 0:
        return None

    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop rows with invalid dates
    combined = combined.dropna(subset=['service_date'])

    # Save to Parquet
    output_path = PROCESSED_DIR / "arrival_departure.parquet"
    combined.to_parquet(output_path, index=False, compression='snappy')

    print(f"\n  Total records: {len(combined):,}")
    print(f"  Columns: {list(combined.columns)}")
    print(f"  Years: {sorted(combined['year'].unique())}")
    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return combined


def convert_ridership():
    """
    Convert Bus Ridership data to Parquet.

    Schema:
    - season: string (e.g., "Fall 2023")
    - route_id: string
    - route_variant: string
    - direction_id: int
    - trip_start_time: time
    - day_type_id: string
    - day_type_name: string
    - stop_name: string
    - stop_id: int
    - stop_sequence: int
    - boardings: float
    - alightings: float
    - load_: float
    - sample_size: int
    - year: int (derived)
    - season_name: string (derived)
    """
    print("\n" + "="*60)
    print("Converting Ridership Data")
    print("="*60)

    ridership_dir = RAW_DATA_DIR / "ridership" / "MBTA_Bus_Ridership_by_Trip_Season_Route_Line_and_Stop"
    if not ridership_dir.exists():
        print("  Ridership directory not found, skipping...")
        return None

    all_csvs = list(ridership_dir.glob("*.csv"))
    print(f"  Found {len(all_csvs)} CSV files")

    if len(all_csvs) == 0:
        return None

    all_dfs = []

    for csv_path in sorted(all_csvs):
        try:
            df = pd.read_csv(csv_path, low_memory=False)

            # Clean BOM
            df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]

            # Extract year and season from filename or season column
            if 'season' in df.columns:
                # Parse year from season string like "Fall 2023"
                df['year'] = df['season'].str.extract(r'(\d{4})').astype(float).astype('Int64')
                df['season_name'] = df['season'].str.extract(r'(Fall|Spring|Summer|Winter)')

            all_dfs.append(df)
            print(f"  Processed: {csv_path.name} ({len(df):,} rows)")

        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")

    if len(all_dfs) == 0:
        return None

    # Combine all
    combined = pd.concat(all_dfs, ignore_index=True)

    # Save to Parquet
    output_path = PROCESSED_DIR / "ridership.parquet"
    combined.to_parquet(output_path, index=False, compression='snappy')

    print(f"\n  Total records: {len(combined):,}")
    print(f"  Columns: {list(combined.columns)}")
    print(f"  Years: {sorted(combined['year'].dropna().unique())}")
    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return combined


def convert_survey():
    """
    Convert 2024 Passenger Survey data to Parquet.

    Schema:
    - aggregation_level: string
    - service_mode: string
    - reporting_group: string
    - measure_group: string
    - measure: string
    - category: string
    - weighted_percent: float
    - ObjectId: int
    """
    print("\n" + "="*60)
    print("Converting Survey Data")
    print("="*60)

    survey_path = RAW_DATA_DIR / "survey" / "MBTA_2024_System-Wide_Passenger_Survey.csv"
    if not survey_path.exists():
        print("  Survey file not found, skipping...")
        return None

    try:
        df = pd.read_csv(survey_path, encoding='utf-8-sig')  # Handle BOM

        # Clean column names
        df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]

        # Add year column
        df['year'] = 2024

        # Save to Parquet
        output_path = PROCESSED_DIR / "survey.parquet"
        df.to_parquet(output_path, index=False, compression='snappy')

        print(f"  Total records: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def convert_gtfs_stops():
    """
    Convert GTFS Stops data to Parquet.

    Schema:
    - stop_id: int
    - stop_code: string
    - stop_name: string
    - stop_desc: string
    - platform_code: string
    - platform_name: string
    - stop_lat: float
    - stop_lon: float
    - zone_id: string
    - stop_address: string
    - stop_url: string
    - level_id: string
    - location_type: int
    - parent_station: string
    - wheelchair_boarding: int
    - municipality: string
    - on_street: string
    - at_street: string
    - vehicle_type: int
    """
    print("\n" + "="*60)
    print("Converting GTFS Stops Data")
    print("="*60)

    stops_path = RAW_DATA_DIR / "gtfs" / "stops.txt"
    if not stops_path.exists():
        print("  GTFS stops file not found, skipping...")
        return None

    try:
        df = pd.read_csv(stops_path, low_memory=False)

        # Save to Parquet
        output_path = PROCESSED_DIR / "gtfs_stops.parquet"
        df.to_parquet(output_path, index=False, compression='snappy')

        print(f"  Total records: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def convert_gtfs_routes():
    """
    Convert GTFS Routes data to Parquet.

    Schema:
    - route_id: string
    - agency_id: int
    - route_short_name: string
    - route_long_name: string
    - route_desc: string
    - route_type: int
    - route_url: string
    - route_color: string
    - route_text_color: string
    - route_sort_order: int
    - route_fare_class: string
    - line_id: string
    - listed_route: string
    - network_id: string
    """
    print("\n" + "="*60)
    print("Converting GTFS Routes Data")
    print("="*60)

    routes_path = RAW_DATA_DIR / "gtfs" / "routes.txt"
    if not routes_path.exists():
        print("  GTFS routes file not found, skipping...")
        return None

    try:
        df = pd.read_csv(routes_path, low_memory=False)

        # Save to Parquet
        output_path = PROCESSED_DIR / "gtfs_routes.parquet"
        df.to_parquet(output_path, index=False, compression='snappy')

        print(f"  Total records: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def convert_census():
    """
    Convert Census/Demographics data to Parquet.

    Reads from Excel files in census directory.
    """
    print("\n" + "="*60)
    print("Converting Census/Demographics Data")
    print("="*60)

    census_dir = RAW_DATA_DIR / "census"
    if not census_dir.exists():
        print("  Census directory not found, skipping...")
        return None

    # Try the 2015-2019 neighborhood tables
    xlsm_path = census_dir / "2015-2019_neighborhood_tables_2021.12.21.xlsm"

    if xlsm_path.exists():
        try:
            # Read all sheets
            xlsx = pd.ExcelFile(xlsm_path)
            print(f"  Found sheets: {xlsx.sheet_names}")

            # Try to find a demographics summary sheet
            df = None
            for sheet_name in xlsx.sheet_names:
                if 'demo' in sheet_name.lower() or 'summary' in sheet_name.lower() or 'pop' in sheet_name.lower():
                    df = pd.read_excel(xlsm_path, sheet_name=sheet_name)
                    print(f"  Using sheet: {sheet_name}")
                    break

            # If no matching sheet, try first sheet
            if df is None and len(xlsx.sheet_names) > 0:
                df = pd.read_excel(xlsm_path, sheet_name=0)
                print(f"  Using first sheet: {xlsx.sheet_names[0]}")

            if df is not None:
                # Clean column names
                df.columns = [str(col).replace('\n', ' ').strip() for col in df.columns]

                # Save to Parquet
                output_path = PROCESSED_DIR / "census_demographics.parquet"
                df.to_parquet(output_path, index=False, compression='snappy')

                print(f"  Total records: {len(df):,}")
                print(f"  Columns: {list(df.columns)[:10]}...")
                print(f"  Saved to: {output_path}")
                print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

                return df

        except Exception as e:
            print(f"  Error reading Excel: {e}")

    return None


def main():
    """Main function to convert all datasets."""
    print("="*60)
    print("Boston Bus Equity - Dataset Conversion to Parquet")
    print("="*60)
    print(f"\nSource: {RAW_DATA_DIR}")
    print(f"Destination: {PROCESSED_DIR}")

    results = {}

    # Convert each dataset
    results['arrival_departure'] = convert_arrival_departure()
    results['ridership'] = convert_ridership()
    results['survey'] = convert_survey()
    results['gtfs_stops'] = convert_gtfs_stops()
    results['gtfs_routes'] = convert_gtfs_routes()
    results['census'] = convert_census()

    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)

    for name, df in results.items():
        if df is not None:
            parquet_path = PROCESSED_DIR / f"{name}.parquet"
            if parquet_path.exists():
                size_mb = parquet_path.stat().st_size / 1024 / 1024
                print(f"  {name}: {len(df):,} records ({size_mb:.2f} MB)")
            else:
                # Try alternate names
                if name == 'census':
                    parquet_path = PROCESSED_DIR / "census_demographics.parquet"
                    if parquet_path.exists():
                        size_kb = parquet_path.stat().st_size / 1024
                        print(f"  {name}: {len(df):,} records ({size_kb:.2f} KB)")
        else:
            print(f"  {name}: SKIPPED (not found or error)")

    # List all created files
    print("\n" + "="*60)
    print("CREATED PARQUET FILES")
    print("="*60)

    for parquet_file in sorted(PROCESSED_DIR.glob("*.parquet")):
        size = parquet_file.stat().st_size
        if size > 1024 * 1024:
            print(f"  {parquet_file.name}: {size / 1024 / 1024:.2f} MB")
        else:
            print(f"  {parquet_file.name}: {size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
