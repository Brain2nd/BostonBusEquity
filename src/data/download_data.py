"""
Data Download Script for Boston Bus Equity Project

This script downloads all required datasets from MBTA Open Data Portal,
Boston Open Data, and Census Bureau.

Data Sources:
- MBTA Bus Arrival/Departure Times: ArcGIS CSV Collection (ZIP files)
- MBTA Ridership: ArcGIS
- MBTA Passenger Survey: ArcGIS
- Boston Census: data.boston.gov
"""

import os
import sys
import zipfile
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
    "2026": "9d8a8cad277545c984c1b25ed10b7d3c",  # Latest (partial)
}

# Other MBTA datasets (need to find item IDs)
MBTA_RIDERSHIP_ITEM = None  # TODO: Find correct item ID
MBTA_SURVEY_ITEM = None  # TODO: Find correct item ID

# Boston Census data URLs
BOSTON_CENSUS_URLS = {
    "neighborhoods": "https://data.boston.gov/dataset/2020-census-for-boston",
    # Direct download links need to be found from the website
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


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
        desc: Description for progress bar

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_and_extract_zip(item_id: str, year: str, output_dir: Path) -> bool:
    """
    Download and extract a ZIP file from ArcGIS.

    Args:
        item_id: ArcGIS item ID
        year: Year of the data
        output_dir: Directory to extract files to

    Returns:
        True if successful, False otherwise
    """
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data"
    zip_path = output_dir / f"temp_{year}.zip"

    print(f"\nDownloading {year} data...")

    if not download_file(url, zip_path, f"Downloading {year}"):
        return False

    # Extract ZIP
    print(f"Extracting {year} data...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        # Remove ZIP file after extraction
        zip_path.unlink()
        print(f"Successfully extracted {year} data")
        return True
    except Exception as e:
        print(f"Error extracting {year} data: {e}")
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

    for year in years:
        if year not in MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS:
            print(f"Year {year} not available")
            continue

        # Check if already downloaded
        existing_files = list(output_dir.glob(f"*{year}*.csv"))
        if existing_files:
            print(f"Data for {year} already exists: {existing_files[0].name}")
            continue

        item_id = MBTA_BUS_ARRIVAL_DEPARTURE_ITEMS[year]
        download_and_extract_zip(item_id, year, output_dir)


def download_ridership_data():
    """Download bus ridership data."""
    print("\n" + "=" * 50)
    print("Ridership Data")
    print("=" * 50)
    print("Please download manually from:")
    print("  https://mbta-massdot.opendata.arcgis.com/")
    print("  Search for: 'Bus Ridership by Trip'")
    print(f"  Save to: {DATA_RAW / 'ridership'}")


def download_survey_data():
    """Download MBTA passenger survey data."""
    print("\n" + "=" * 50)
    print("Passenger Survey Data")
    print("=" * 50)
    print("Please download manually from:")
    print("  https://mbta-massdot.opendata.arcgis.com/")
    print("  Search for: 'MBTA 2024 System-Wide Passenger Survey'")
    print(f"  Save to: {DATA_RAW / 'survey'}")


def download_census_data():
    """Download Boston census data."""
    print("\n" + "=" * 50)
    print("Census Data")
    print("=" * 50)
    print("Please download manually from:")
    print("  - 2020 Census: https://data.boston.gov/dataset/2020-census-for-boston")
    print("  - ACS 2020-2024: https://data.census.gov/")
    print(f"  Save to: {DATA_RAW / 'census'}")


def check_data_status():
    """Check and report data download status."""
    print("\n" + "=" * 50)
    print("Data Status")
    print("=" * 50)

    # Check arrival/departure data
    ad_dir = DATA_RAW / "arrival_departure"
    if ad_dir.exists():
        csv_files = list(ad_dir.glob("*.csv"))
        print(f"\nArrival/Departure data: {len(csv_files)} files")
        for f in sorted(csv_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}: {size_mb:.1f} MB")
    else:
        print("\nArrival/Departure data: Not downloaded")

    # Check other data
    for data_type in ["ridership", "survey", "census"]:
        data_dir = DATA_RAW / data_type
        if data_dir.exists():
            files = [f for f in data_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
            print(f"\n{data_type.title()} data: {len(files)} files")
        else:
            print(f"\n{data_type.title()} data: Not downloaded")


def main():
    """Main function to download all data."""
    print("=" * 60)
    print("Boston Bus Equity - Data Download Script")
    print("=" * 60)

    # Create directories
    create_directories()

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            check_data_status()
            return
        elif sys.argv[1] == "--years":
            years = sys.argv[2:]
            download_bus_arrival_departure(years)
            return

    # Download data
    print("\n[1/4] Downloading bus arrival/departure data...")
    print("Available years: 2020-2026")
    print("Note: 2018-2019 data not available on ArcGIS")
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
    print("Data download script complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
