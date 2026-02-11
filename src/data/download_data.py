"""
Data Download Script for Boston Bus Equity Project

This script downloads all required datasets from MBTA Open Data Portal,
Boston Open Data, and Census Bureau.

Features:
- Resume support for interrupted downloads
- Progress bar with download speed
- Automatic extraction of ZIP files

Data Sources:
- MBTA Bus Arrival/Departure Times: ArcGIS CSV Collection (ZIP files)
- MBTA Ridership: ArcGIS
- MBTA Passenger Survey: ArcGIS
- Boston Census: data.boston.gov

Usage:
    python download_data.py              # Download all data
    python download_data.py --status     # Check download status
    python download_data.py --years 2024 2025  # Download specific years
"""

import os
import sys
import zipfile
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# ArcGIS Item IDs for MBTA Bus Arrival/Departure Times
# Download URL format: https://www.arcgis.com/sharing/rest/content/items/{ITEM_ID}/data
MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS = {
    "2020": "4c1293151c6c4a069d49e6b85ee68ea4",
    "2021": "2d415555f63b431597721151a7e07a3e",
    "2022": "ef464a75666349f481353f16514c06d0",
    "2023": "b7b36fdb7b3a4728af2fccc78c2ca5b7",
    "2024": "96c77138c3144906bce93d0257531b6a",  # Training data
    "2025": "924df13d845f4907bb6a6c3ed380d57a",  # Validation data
    "2026": "9d8a8cad277545c984c1b25ed10b7d3c",  # Validation data (partial year)
}

# Additional data sources with direct download URLs
ADDITIONAL_DATA_SOURCES = {
    "ridership": {
        "item_id": "8daf4a33925a4df59183f860826d29ee",
        "filename": "MBTA_Bus_Ridership_by_Trip.zip",
        "description": "Bus Ridership by Trip, Season, Route/Line, and Stop"
    },
    "census_neighborhood": {
        "url": "https://data.boston.gov/dataset/7846ff3b-b738-47a3-a57e-19de2c753220/resource/5800a0a2-6acd-41a3-9fe0-1bf7b038750d/download/redistricting_data_tract20_nbhd_hhpopsize_ab-1.csv",
        "filename": "boston_2020_census_neighborhood.csv",
        "description": "2020 Census for Boston by Neighborhood"
    },
}

# Manual download required for these datasets (require authentication or special access)
MANUAL_DOWNLOAD_SOURCES = {
    "survey_2024": {
        "url": "https://mbta-massdot.opendata.arcgis.com/datasets/mbta-2024-system-wide-passenger-survey",
        "description": "MBTA 2024 System-Wide Passenger Survey",
        "save_to": "survey/"
    },
    "census_demographics": {
        "url": "https://data.boston.gov/dataset/neighborhood-demographics",
        "description": "Boston Neighborhood Demographics (ACS)",
        "save_to": "census/"
    }
}


def create_directories():
    """Create all necessary data directories."""
    directories = [
        DATA_RAW / "arrival_departure",
        DATA_RAW / "ridership",
        DATA_RAW / "survey",
        DATA_RAW / "census",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("Directory structure created.")


def get_remote_file_size(url: str) -> int:
    """Get the size of remote file via HEAD request."""
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return int(response.headers.get('content-length', 0))
    except Exception:
        return 0


def download_file_with_resume(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """
    Download a file with resume support and progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
        desc: Description for progress bar

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get remote file size
        remote_size = get_remote_file_size(url)

        # Check if partial file exists
        local_size = 0
        mode = 'wb'
        headers = {}

        if output_path.exists():
            local_size = output_path.stat().st_size
            if local_size == remote_size and remote_size > 0:
                print(f"  File already complete: {output_path.name}")
                return True
            elif local_size < remote_size:
                # Resume download
                headers['Range'] = f'bytes={local_size}-'
                mode = 'ab'
                print(f"  Resuming from {local_size / 1024 / 1024:.1f} MB...")

        # Start download
        response = requests.get(url, stream=True, timeout=30, headers=headers)

        # Check if server supports resume
        if response.status_code == 206:  # Partial content
            total_size = remote_size
        elif response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            local_size = 0  # Server doesn't support resume, start over
            mode = 'wb'
        else:
            print(f"  Error: HTTP {response.status_code}")
            return False

        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=local_size,
                unit='B',
                unit_scale=True,
                desc=desc,
                ncols=80
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except requests.exceptions.RequestException as e:
        print(f"  Network error: {e}")
        print("  You can resume by running the script again.")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_and_extract_zip(item_id: str, year: str, output_dir: Path) -> bool:
    """
    Download and extract a ZIP file from ArcGIS with resume support.

    Args:
        item_id: ArcGIS item ID
        year: Year of the data
        output_dir: Directory to extract files to

    Returns:
        True if successful, False otherwise
    """
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data"
    zip_path = output_dir / f"MBTA_Bus_Arrival_Departure_Times_{year}.zip"

    print(f"\n[{year}] Downloading bus arrival/departure data...")

    if not download_file_with_resume(url, zip_path, f"{year} data"):
        return False

    # Verify ZIP file integrity
    if not zip_path.exists():
        print(f"  Error: ZIP file not found")
        return False

    # Check if already extracted
    csv_files = list(output_dir.glob(f"*{year}*.csv"))
    if csv_files:
        print(f"  Already extracted: {csv_files[0].name}")
        # Remove ZIP to save space
        if zip_path.exists():
            zip_path.unlink()
        return True

    # Extract ZIP
    print(f"  Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Remove ZIP file after successful extraction
        zip_path.unlink()

        # Find extracted file
        csv_files = list(output_dir.glob(f"*{year}*.csv"))
        if csv_files:
            size_mb = csv_files[0].stat().st_size / (1024 * 1024)
            print(f"  Extracted: {csv_files[0].name} ({size_mb:.1f} MB)")

        return True
    except zipfile.BadZipFile:
        print(f"  Error: Corrupted ZIP file, deleting for re-download...")
        zip_path.unlink()
        return False
    except Exception as e:
        # Try using system unzip command as fallback (supports deflate64)
        print(f"  Python zipfile failed: {e}")
        print(f"  Trying system unzip command...")
        try:
            extract_dir = output_dir / f"MBTA_Bus_Arrival_Departure_Times_{year}"
            result = subprocess.run(
                ["unzip", "-o", str(zip_path), "-d", str(extract_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Remove ZIP file after successful extraction
                zip_path.unlink()
                # Find extracted files
                csv_files = list(extract_dir.glob("*.csv"))
                if csv_files:
                    total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
                    print(f"  Extracted {len(csv_files)} files ({total_size:.1f} MB total)")
                return True
            else:
                print(f"  System unzip failed: {result.stderr}")
                return False
        except FileNotFoundError:
            print(f"  System unzip command not found")
            return False


def download_bus_arrival_departure(years: list = None):
    """
    Download bus arrival/departure data for specified years.

    Args:
        years: List of years to download. If None, download all available.
    """
    if years is None:
        years = list(MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS.keys())

    output_dir = DATA_RAW / "arrival_departure"
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for year in years:
        if year not in MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS:
            print(f"\n[{year}] Not available on MBTA Open Data Portal")
            continue

        # Check if already downloaded and extracted (check both root and subdirs)
        existing_csv = list(output_dir.glob(f"*{year}*.csv")) + \
                       list(output_dir.glob(f"*{year}*/*.csv"))
        if existing_csv:
            total_size = sum(f.stat().st_size for f in existing_csv) / (1024 * 1024)
            print(f"\n[{year}] Already exists: {len(existing_csv)} files ({total_size:.1f} MB)")
            success_count += 1
            continue

        item_id = MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS[year]
        if download_and_extract_zip(item_id, year, output_dir):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n  Summary: {success_count} succeeded, {fail_count} failed")
    if fail_count > 0:
        print("  Run the script again to resume failed downloads.")


def download_ridership_data():
    """Download bus ridership data from ArcGIS."""
    print("\n" + "=" * 50)
    print("Ridership Data")
    print("=" * 50)

    output_dir = DATA_RAW / "ridership"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = ADDITIONAL_DATA_SOURCES["ridership"]
    item_id = source["item_id"]
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data"
    zip_path = output_dir / source["filename"]

    # Check if already extracted (check both root and subdirectories)
    csv_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("**/*.csv"))
    if csv_files:
        total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
        print(f"  Already downloaded: {len(csv_files)} CSV files ({total_size:.1f} MB)")
        return True

    print(f"  Downloading: {source['description']}...")
    if not download_file_with_resume(url, zip_path, "Ridership"):
        return False

    # Extract ZIP
    print("  Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        zip_path.unlink()
        csv_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("**/*.csv"))
        total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
        print(f"  Extracted {len(csv_files)} files ({total_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def download_survey_data():
    """Download MBTA passenger survey data (requires manual download)."""
    print("\n" + "=" * 50)
    print("Passenger Survey Data")
    print("=" * 50)

    output_dir = DATA_RAW / "survey"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    csv_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("**/*.csv"))
    if csv_files:
        total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
        print(f"  Already downloaded: {len(csv_files)} files ({total_size:.1f} MB)")
        return True

    # Manual download required
    source = MANUAL_DOWNLOAD_SOURCES["survey_2024"]
    print(f"  Manual download required:")
    print(f"    1. Visit: {source['url']}")
    print(f"    2. Click 'Download' -> CSV")
    print(f"    3. Save to: {output_dir}")
    return False


def download_census_data():
    """Download Boston census and demographic data."""
    print("\n" + "=" * 50)
    print("Census & Demographic Data")
    print("=" * 50)

    output_dir = DATA_RAW / "census"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-downloadable sources
    auto_sources = ["census_neighborhood"]
    success_count = 0

    for source_key in auto_sources:
        source = ADDITIONAL_DATA_SOURCES[source_key]
        output_path = output_dir / source["filename"]

        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"  Already downloaded: {source['filename']} ({size_kb:.1f} KB)")
            success_count += 1
            continue

        print(f"  Downloading: {source['description']}...")
        if download_file_with_resume(source["url"], output_path, source_key):
            size_kb = output_path.stat().st_size / 1024
            print(f"  Downloaded: {source['filename']} ({size_kb:.1f} KB)")
            success_count += 1
        else:
            print(f"  Failed to download: {source['filename']}")

    # Check for other census files
    all_csv = list(output_dir.glob("*.csv"))
    if len(all_csv) > success_count:
        print(f"  Found {len(all_csv)} total census files")

    # Print manual download info if needed
    if len(all_csv) < 2:
        print(f"\n  For additional demographic data, visit:")
        print(f"    {MANUAL_DOWNLOAD_SOURCES['census_demographics']['url']}")
        print(f"    Save to: {output_dir}")


def check_data_status():
    """Check and report data download status."""
    print("\n" + "=" * 50)
    print("Data Status")
    print("=" * 50)

    # Check arrival/departure data
    ad_dir = DATA_RAW / "arrival_departure"
    if ad_dir.exists():
        # Check both root and subdirectories
        csv_files = list(ad_dir.glob("*.csv")) + list(ad_dir.glob("**/*.csv"))
        zip_files = list(ad_dir.glob("*.zip"))

        # Group by year
        years_found = {}
        for f in csv_files:
            for year in MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS.keys():
                if year in str(f):
                    if year not in years_found:
                        years_found[year] = []
                    years_found[year].append(f)

        total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)

        print(f"\nArrival/Departure data:")
        print(f"  Years available: {len(years_found)}/{len(MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS)}")
        print(f"  Total CSV files: {len(csv_files)}")
        print(f"  Total size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")

        for year in sorted(years_found.keys()):
            files = years_found[year]
            size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
            print(f"    {year}: {len(files)} files ({size_mb:.1f} MB)")

        if zip_files:
            print(f"  Partial (ZIP): {len(zip_files)} files")

        missing_years = set(MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS.keys()) - set(years_found.keys())
        if missing_years:
            print(f"  Missing years: {sorted(missing_years)}")
    else:
        print("\nArrival/Departure data: Not downloaded")

    # Check other data (including subdirectories)
    for data_type in ["ridership", "survey", "census"]:
        data_dir = DATA_RAW / data_type
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("**/*.csv"))
            if csv_files:
                total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
                print(f"\n{data_type.title()} data: {len(csv_files)} files ({total_size:.1f} MB)")
            else:
                print(f"\n{data_type.title()} data: No CSV files found")
        else:
            print(f"\n{data_type.title()} data: Not downloaded")


def main():
    """Main function to download all data."""
    print("=" * 60)
    print("Boston Bus Equity - Data Download Script")
    print("=" * 60)
    print("Features: Resume support for interrupted downloads")

    # Create directories
    create_directories()

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            check_data_status()
            return
        elif sys.argv[1] == "--years":
            years = sys.argv[2:]
            if not years:
                print("Error: Please specify years, e.g., --years 2024 2025")
                return
            download_bus_arrival_departure(years)
            check_data_status()
            return
        elif sys.argv[1] == "--help":
            print(__doc__)
            return

    # Download all data
    print("\n[1/4] Downloading bus arrival/departure data...")
    print("Available years: 2020-2026")
    print("Training set: 2020-2024 | Validation set: 2025-2026")
    download_bus_arrival_departure()

    print("\n[2/4] Ridership data instructions...")
    download_ridership_data()

    print("\n[3/4] Survey data instructions...")
    download_survey_data()

    print("\n[4/4] Census data instructions...")
    download_census_data()

    # Show status
    check_data_status()

    print("\n" + "=" * 60)
    print("Download complete! Run again to resume any failed downloads.")
    print("=" * 60)


if __name__ == "__main__":
    main()
