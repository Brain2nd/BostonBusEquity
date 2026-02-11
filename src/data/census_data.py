"""
Census/Demographic Data Module for Boston Bus Equity Project

Parses Boston neighborhood demographic data from ACS 2015-2019 Excel files.
Used for Q7: demographic disparities analysis (race, ethnicity, age, income).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED

# Census data directory
CENSUS_DIR = DATA_RAW / "census"
CENSUS_EXCEL = CENSUS_DIR / "2015-2019_neighborhood_tables_2021.12.21.xlsm"

# Valid Boston neighborhoods (excludes source notes)
BOSTON_NEIGHBORHOODS = [
    'Allston', 'Back Bay', 'Beacon Hill', 'Brighton', 'Charlestown',
    'Dorchester', 'Downtown', 'East Boston', 'Fenway', 'Hyde Park',
    'Jamaica Plain', 'Longwood', 'Mattapan', 'Mission Hill', 'North End',
    'Roslindale', 'Roxbury', 'South Boston', 'South Boston Waterfront',
    'South End', 'West End', 'West Roxbury'
]


def load_race_data() -> pd.DataFrame:
    """
    Load race/ethnicity data by neighborhood.

    Returns:
        DataFrame with race/ethnicity breakdown per neighborhood
    """
    if not CENSUS_EXCEL.exists():
        print(f"Warning: Census file not found: {CENSUS_EXCEL}")
        return pd.DataFrame()

    df = pd.read_excel(CENSUS_EXCEL, sheet_name='Race', header=2)

    # Rename columns based on actual structure
    df.columns = ['neighborhood', 'total_pop', 'white', 'white_pct',
                  'black', 'black_pct', 'hispanic', 'hispanic_pct',
                  'asian', 'asian_pct', 'other', 'other_pct']

    # Filter to Boston neighborhoods only
    df = df[df['neighborhood'].isin(BOSTON_NEIGHBORHOODS)].copy()

    # Convert numeric columns
    for col in ['total_pop', 'white', 'black', 'hispanic', 'asian', 'other']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['white_pct', 'black_pct', 'hispanic_pct', 'asian_pct', 'other_pct']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate minority percentage (non-white)
    df['minority_pct'] = 1 - df['white_pct']

    return df.reset_index(drop=True)


def load_income_data() -> pd.DataFrame:
    """
    Load household income data by neighborhood.

    Returns:
        DataFrame with income distribution per neighborhood
    """
    if not CENSUS_EXCEL.exists():
        return pd.DataFrame()

    df = pd.read_excel(CENSUS_EXCEL, sheet_name='Household Income', header=2)

    # Rename columns based on actual structure
    df.columns = ['neighborhood', 'median_income', 'total_households',
                  'under_15k', 'under_15k_pct', '15k_25k', '15k_25k_pct',
                  '25k_35k', '25k_35k_pct', '35k_50k', '35k_50k_pct',
                  '50k_75k', '50k_75k_pct', '75k_100k', '75k_100k_pct',
                  '100k_150k', '100k_150k_pct', 'over_150k', 'over_150k_pct']

    # Filter to Boston neighborhoods
    df = df[df['neighborhood'].isin(BOSTON_NEIGHBORHOODS)].copy()

    # Convert numeric columns
    df['median_income'] = pd.to_numeric(df['median_income'], errors='coerce')
    df['total_households'] = pd.to_numeric(df['total_households'], errors='coerce')

    # Calculate low income percentage (under $35k)
    df['low_income_pct'] = (
        pd.to_numeric(df['under_15k_pct'], errors='coerce') +
        pd.to_numeric(df['15k_25k_pct'], errors='coerce') +
        pd.to_numeric(df['25k_35k_pct'], errors='coerce')
    )

    return df.reset_index(drop=True)


def load_poverty_data() -> pd.DataFrame:
    """
    Load poverty rate data by neighborhood.

    Returns:
        DataFrame with poverty rates per neighborhood
    """
    if not CENSUS_EXCEL.exists():
        return pd.DataFrame()

    df = pd.read_excel(CENSUS_EXCEL, sheet_name='Poverty Rates', header=2)

    # Rename columns
    df.columns = ['neighborhood', 'pop_for_poverty', 'in_poverty',
                  'poverty_rate', 'pct_of_boston_poor']

    # Filter to Boston neighborhoods
    df = df[df['neighborhood'].isin(BOSTON_NEIGHBORHOODS)].copy()

    # Convert numeric columns
    for col in ['pop_for_poverty', 'in_poverty', 'poverty_rate', 'pct_of_boston_poor']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.reset_index(drop=True)


def load_age_data() -> pd.DataFrame:
    """
    Load age distribution data by neighborhood.

    Returns:
        DataFrame with age breakdown per neighborhood
    """
    if not CENSUS_EXCEL.exists():
        return pd.DataFrame()

    df = pd.read_excel(CENSUS_EXCEL, sheet_name='Age', header=2)

    # Get first neighborhood column and identify structure
    # Age sheet typically has: neighborhood, total, then age groups
    # We'll extract key age groups

    # Filter to Boston neighborhoods
    df = df[df.iloc[:, 0].isin(BOSTON_NEIGHBORHOODS)].copy()

    # Simplify: just get total and rename first column
    result = pd.DataFrame()
    result['neighborhood'] = df.iloc[:, 0]
    result['total_pop'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

    return result.reset_index(drop=True)


def load_all_demographics() -> pd.DataFrame:
    """
    Load and merge all demographic data into a single DataFrame.

    Returns:
        DataFrame with comprehensive demographic data per neighborhood
    """
    # Load individual datasets
    race = load_race_data()
    income = load_income_data()
    poverty = load_poverty_data()

    if race.empty:
        print("Warning: No race data available")
        return pd.DataFrame()

    # Start with race as base
    demographics = race[['neighborhood', 'total_pop', 'white_pct', 'black_pct',
                         'hispanic_pct', 'asian_pct', 'minority_pct']].copy()

    # Merge income
    if not income.empty:
        income_cols = income[['neighborhood', 'median_income', 'low_income_pct']]
        demographics = demographics.merge(income_cols, on='neighborhood', how='left')

    # Merge poverty
    if not poverty.empty:
        poverty_cols = poverty[['neighborhood', 'poverty_rate']]
        demographics = demographics.merge(poverty_cols, on='neighborhood', how='left')

    return demographics


def classify_neighborhood_demographics(demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Classify neighborhoods by demographic characteristics for equity analysis.

    Args:
        demographics: DataFrame from load_all_demographics()

    Returns:
        DataFrame with demographic classifications added
    """
    df = demographics.copy()

    # High minority neighborhood (>50% non-white)
    df['high_minority'] = df['minority_pct'] > 0.5

    # Low income neighborhood (median income below Boston median)
    boston_median = df['median_income'].median()
    df['low_income'] = df['median_income'] < boston_median

    # High poverty neighborhood (poverty rate above Boston average)
    boston_poverty = df['poverty_rate'].mean()
    df['high_poverty'] = df['poverty_rate'] > boston_poverty

    # Vulnerable neighborhood (high minority AND low income)
    df['vulnerable'] = df['high_minority'] & df['low_income']

    return df


def save_demographics_csv(output_dir: Path = None) -> Path:
    """
    Save processed demographic data to CSV.

    Args:
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = DATA_PROCESSED

    output_dir.mkdir(parents=True, exist_ok=True)

    demographics = load_all_demographics()
    if demographics.empty:
        print("No demographic data to save")
        return None

    demographics = classify_neighborhood_demographics(demographics)

    output_path = output_dir / "neighborhood_demographics.csv"
    demographics.to_csv(output_path, index=False)
    print(f"Saved demographic data to: {output_path}")

    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Loading Census/Demographic Data")
    print("=" * 60)

    # Load and display demographics
    demographics = load_all_demographics()

    if demographics.empty:
        print("No demographic data available")
    else:
        demographics = classify_neighborhood_demographics(demographics)

        print(f"\nLoaded demographics for {len(demographics)} neighborhoods")
        print("\nColumns:", list(demographics.columns))

        print("\n" + "=" * 60)
        print("Demographic Summary")
        print("=" * 60)

        print(f"\nTotal population covered: {demographics['total_pop'].sum():,.0f}")

        print("\n--- Race/Ethnicity ---")
        print(f"Avg White %: {demographics['white_pct'].mean()*100:.1f}%")
        print(f"Avg Black %: {demographics['black_pct'].mean()*100:.1f}%")
        print(f"Avg Hispanic %: {demographics['hispanic_pct'].mean()*100:.1f}%")
        print(f"Avg Asian %: {demographics['asian_pct'].mean()*100:.1f}%")

        print("\n--- Income ---")
        print(f"Median income range: ${demographics['median_income'].min():,.0f} - ${demographics['median_income'].max():,.0f}")
        print(f"Boston median: ${demographics['median_income'].median():,.0f}")

        print("\n--- Poverty ---")
        print(f"Poverty rate range: {demographics['poverty_rate'].min()*100:.1f}% - {demographics['poverty_rate'].max()*100:.1f}%")

        print("\n--- Classifications ---")
        print(f"High minority neighborhoods: {demographics['high_minority'].sum()}")
        print(f"Low income neighborhoods: {demographics['low_income'].sum()}")
        print(f"High poverty neighborhoods: {demographics['high_poverty'].sum()}")
        print(f"Vulnerable neighborhoods: {demographics['vulnerable'].sum()}")

        print("\n--- Vulnerable Neighborhoods ---")
        vulnerable = demographics[demographics['vulnerable']][['neighborhood', 'minority_pct', 'median_income', 'poverty_rate']]
        print(vulnerable.to_string())

        # Save to CSV
        save_demographics_csv()
