"""
PyTorch Dataset for MBTA Bus Delay Prediction
==============================================

This module provides efficient data loading and preprocessing for deep learning
models predicting bus delays. Supports:
- Time series windowing for sequence models (LSTM, Transformer)
- Feature engineering (temporal, route, historical)
- Train/val/test splitting by time
- Efficient caching with HDF5/Parquet

Author: Boston Bus Equity Team
Date: February 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq


class BusDelayDataProcessor:
    """
    Preprocesses raw MBTA arrival/departure data for deep learning.

    Handles:
    - Loading and combining multiple CSV files
    - Calculating delays
    - Feature engineering
    - Saving to efficient Parquet format
    """

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Target routes for special attention
        self.target_routes = ['22', '29', '15', '45', '28', '44', '42', '17',
                              '23', '31', '26', '111', '24', '33', '14']

    def load_and_process_all(self, years: List[int] = None,
                             sample_frac: float = 1.0) -> pd.DataFrame:
        """
        Load all arrival/departure data and process for ML.

        Args:
            years: List of years to load (default: all available)
            sample_frac: Fraction of data to sample (for faster iteration)

        Returns:
            Processed DataFrame with features
        """
        if years is None:
            years = [2020, 2021, 2022, 2023, 2024, 2025]

        all_data = []

        for year in years:
            year_dir = self.data_dir / f"MBTA_Bus_Arrival_Departure_Times_{year}"
            if not year_dir.exists():
                print(f"Skipping {year}: directory not found")
                continue

            csv_files = list(year_dir.glob("*.csv"))
            print(f"Loading {year}: {len(csv_files)} files")

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8-sig')
                    # Normalize column names (2020 uses 'direction', 2021+ uses 'direction_id')
                    if 'direction' in df.columns and 'direction_id' not in df.columns:
                        df = df.rename(columns={'direction': 'direction_id'})
                    if sample_frac < 1.0:
                        df = df.sample(frac=sample_frac, random_state=42)
                    all_data.append(df)
                except Exception as e:
                    print(f"  Error loading {csv_file.name}: {e}")

        if not all_data:
            raise ValueError("No data loaded!")

        print(f"\nCombining {len(all_data)} files...")
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Total records: {len(combined):,}")

        # Process features
        processed = self._engineer_features(combined)

        return processed

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML model."""
        df = df.copy()

        # Parse dates (handle BOM character and various formats)
        df['service_date'] = df['service_date'].astype(str).str.strip()
        # Only remove BOM character, not all non-ASCII
        df['service_date'] = df['service_date'].str.replace('\ufeff', '', regex=False)
        df['service_date'] = df['service_date'].str.replace('<U+FEFF>', '', regex=False)
        df['service_date'] = pd.to_datetime(df['service_date'], errors='coerce', format='mixed')

        # Parse scheduled and actual times (handle multiple formats)
        df['scheduled'] = df['scheduled'].astype(str).str.replace('T', ' ').str.replace('Z', '')
        df['actual'] = df['actual'].astype(str).str.replace('T', ' ').str.replace('Z', '')
        df['scheduled'] = pd.to_datetime(df['scheduled'], errors='coerce', format='mixed')
        df['actual'] = pd.to_datetime(df['actual'], errors='coerce', format='mixed')

        # Calculate delay in minutes
        df['delay_minutes'] = (df['actual'] - df['scheduled']).dt.total_seconds() / 60

        # Filter outliers and drop NAs
        df = df[(df['delay_minutes'] >= -30) & (df['delay_minutes'] <= 120)]
        df = df.dropna(subset=['delay_minutes', 'service_date', 'scheduled'])

        # Temporal features
        df['year'] = df['service_date'].dt.year
        df['month'] = df['service_date'].dt.month
        df['day'] = df['service_date'].dt.day
        df['day_of_week'] = df['service_date'].dt.dayofweek  # 0=Monday
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['week_of_year'] = df['service_date'].dt.isocalendar().week.astype(int)

        # Hour from scheduled time
        df['hour'] = df['scheduled'].dt.hour
        df['minute'] = df['scheduled'].dt.minute

        # Time periods
        df['time_period'] = pd.cut(df['hour'],
                                    bins=[0, 6, 9, 15, 19, 22, 24],
                                    labels=['night', 'am_peak', 'midday', 'pm_peak', 'evening', 'night2'],
                                    ordered=False)
        df['time_period'] = df['time_period'].replace('night2', 'night')

        # Route features
        df['route_id'] = df['route_id'].astype(str)
        df['is_target_route'] = df['route_id'].isin(self.target_routes).astype(int)

        # Direction encoding
        df['direction_encoded'] = (df['direction_id'] == 'Outbound').astype(int)

        # Point type encoding
        point_type_map = {'Startpoint': 0, 'Midpoint': 1, 'Endpoint': 2}
        df['point_type_encoded'] = df['point_type'].map(point_type_map).fillna(1)

        # Headway features (if available)
        if 'scheduled_headway' in df.columns:
            df['scheduled_headway'] = pd.to_numeric(df['scheduled_headway'], errors='coerce')
        if 'headway' in df.columns:
            df['actual_headway'] = pd.to_numeric(df['headway'], errors='coerce')

        # Create datetime index for time series
        df['datetime'] = df['service_date'] + pd.to_timedelta(df['hour'], unit='h') + pd.to_timedelta(df['minute'], unit='m')

        # Sort by time
        df = df.sort_values(['datetime', 'route_id', 'stop_id'])

        print(f"Processed {len(df):,} records with {len(df.columns)} features")

        return df

    def save_to_parquet(self, df: pd.DataFrame, filename: str = "bus_delays_processed.parquet"):
        """Save processed data to Parquet for efficient loading."""
        output_path = self.output_dir / filename

        # Select columns to save
        columns_to_save = [
            'service_date', 'datetime', 'route_id', 'direction_id', 'stop_id',
            'delay_minutes', 'year', 'month', 'day', 'day_of_week', 'is_weekend',
            'week_of_year', 'hour', 'minute', 'time_period', 'is_target_route',
            'direction_encoded', 'point_type_encoded'
        ]

        # Add optional columns if present
        for col in ['scheduled_headway', 'actual_headway', 'time_point_order']:
            if col in df.columns:
                columns_to_save.append(col)

        df_save = df[[c for c in columns_to_save if c in df.columns]]
        df_save.to_parquet(output_path, index=False, compression='snappy')

        print(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path


class BusDelayTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for bus delay time series prediction.

    Creates sequences of historical delays to predict future delays.
    Supports:
    - Variable sequence length
    - Multi-step forecasting
    - Route-specific or aggregate predictions
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        seq_length: int = 24,  # Hours of history
        pred_length: int = 1,   # Hours to predict
        route_id: Optional[str] = None,  # None = all routes
        features: List[str] = None,
        target: str = 'delay_minutes',
        aggregate_by: str = 'hour',  # 'hour', '15min', 'trip'
        normalize: bool = True,
        train_split: float = 0.7,
        val_split: float = 0.15,
        mode: str = 'train'  # 'train', 'val', 'test'
    ):
        """
        Initialize the dataset.

        Args:
            data: DataFrame or path to parquet file
            seq_length: Number of time steps for input sequence
            pred_length: Number of time steps to predict
            route_id: Specific route to filter (None for all)
            features: List of feature columns to use
            target: Target column name
            aggregate_by: Time aggregation level
            normalize: Whether to normalize features
            train_split: Fraction for training
            val_split: Fraction for validation
            mode: Dataset mode ('train', 'val', 'test')
        """
        # Load data
        if isinstance(data, str):
            data = pd.read_parquet(data)

        self.df = data.copy()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.target = target
        self.normalize = normalize
        self.mode = mode

        # Filter by route if specified
        if route_id:
            self.df = self.df[self.df['route_id'] == route_id]

        # Aggregate data
        self._aggregate_data(aggregate_by)

        # Define features
        if features is None:
            self.features = [
                'delay_minutes', 'hour', 'day_of_week', 'is_weekend',
                'month', 'is_target_route'
            ]
        else:
            self.features = features

        # Prepare sequences
        self._prepare_sequences(train_split, val_split)

        # Normalize
        if normalize:
            self._normalize_data()

    def _aggregate_data(self, aggregate_by: str):
        """Aggregate raw data to desired time resolution."""
        if aggregate_by == 'hour':
            # Aggregate to hourly level
            self.df['time_key'] = self.df['datetime'].dt.floor('H')
            agg_dict = {
                'delay_minutes': 'mean',
                'hour': 'first',
                'day_of_week': 'first',
                'is_weekend': 'first',
                'month': 'first',
                'year': 'first',
                'is_target_route': 'mean'  # Proportion of target routes
            }
            self.df = self.df.groupby('time_key').agg(agg_dict).reset_index()
            self.df = self.df.rename(columns={'time_key': 'datetime'})

        elif aggregate_by == '15min':
            self.df['time_key'] = self.df['datetime'].dt.floor('15T')
            agg_dict = {
                'delay_minutes': 'mean',
                'hour': 'first',
                'day_of_week': 'first',
                'is_weekend': 'first',
                'month': 'first',
                'is_target_route': 'mean'
            }
            self.df = self.df.groupby('time_key').agg(agg_dict).reset_index()
            self.df = self.df.rename(columns={'time_key': 'datetime'})

        # Sort by time
        self.df = self.df.sort_values('datetime').reset_index(drop=True)

        # Fill missing timestamps with interpolation
        self.df = self.df.set_index('datetime')
        if aggregate_by == 'hour':
            self.df = self.df.resample('H').mean()
        elif aggregate_by == '15min':
            self.df = self.df.resample('15T').mean()
        self.df = self.df.interpolate(method='linear', limit=6)
        self.df = self.df.fillna(method='bfill').fillna(method='ffill')
        self.df = self.df.reset_index()

    def _prepare_sequences(self, train_split: float, val_split: float):
        """Create sequences for time series prediction."""
        # Create feature matrix
        feature_cols = [c for c in self.features if c in self.df.columns]
        self.feature_data = self.df[feature_cols].values.astype(np.float32)
        self.target_data = self.df[self.target].values.astype(np.float32)
        self.timestamps = self.df['datetime'].values

        # Total samples
        total_len = len(self.df) - self.seq_length - self.pred_length + 1

        # Split by time (not random!)
        train_end = int(total_len * train_split)
        val_end = int(total_len * (train_split + val_split))

        if self.mode == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif self.mode == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        else:  # test
            self.start_idx = val_end
            self.end_idx = total_len

        print(f"Dataset mode: {self.mode}")
        print(f"Samples: {self.end_idx - self.start_idx}")
        print(f"Features: {len(feature_cols)}")

    def _normalize_data(self):
        """Normalize features using training statistics."""
        # Always calculate stats from training portion of data
        train_features = self.feature_data[:int(len(self.feature_data) * 0.7)]
        self.feature_mean = train_features.mean(axis=0)
        self.feature_std = train_features.std(axis=0) + 1e-8

        train_target = self.target_data[:int(len(self.target_data) * 0.7)]
        self.target_mean = train_target.mean()
        self.target_std = train_target.std() + 1e-8

        # Normalize
        self.feature_data = (self.feature_data - self.feature_mean) / self.feature_std
        self.target_data_normalized = (self.target_data - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            x: Input sequence [seq_length, num_features]
            y: Target values [pred_length]
        """
        real_idx = self.start_idx + idx

        # Input sequence
        x = self.feature_data[real_idx:real_idx + self.seq_length]

        # Target (next pred_length steps)
        if self.normalize:
            y = self.target_data_normalized[real_idx + self.seq_length:
                                            real_idx + self.seq_length + self.pred_length]
        else:
            y = self.target_data[real_idx + self.seq_length:
                                 real_idx + self.seq_length + self.pred_length]

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def get_timestamp(self, idx: int) -> np.datetime64:
        """Get timestamp for prediction target."""
        real_idx = self.start_idx + idx
        return self.timestamps[real_idx + self.seq_length]

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        if self.normalize:
            return y * self.target_std + self.target_mean
        return y


class BusDelayRouteDataset(Dataset):
    """
    PyTorch Dataset for route-level delay prediction.

    Predicts delays for specific routes considering:
    - Route-specific historical patterns
    - Cross-route interactions
    - Spatial features (stop sequences)
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        routes: List[str] = None,
        seq_length: int = 48,
        pred_length: int = 24,
        mode: str = 'train'
    ):
        if isinstance(data, str):
            data = pd.read_parquet(data)

        self.df = data.copy()
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.mode = mode

        # Use target routes if not specified
        if routes is None:
            routes = ['22', '29', '15', '45', '28', '44', '42', '17',
                     '23', '31', '26', '111', '24', '33', '14']
        self.routes = routes

        # Create route-level aggregated time series
        self._prepare_route_data()

    def _prepare_route_data(self):
        """Create route-level time series."""
        # Filter to target routes
        self.df = self.df[self.df['route_id'].isin(self.routes)]

        # Aggregate by hour and route
        self.df['time_key'] = self.df['datetime'].dt.floor('H')

        route_data = self.df.groupby(['time_key', 'route_id']).agg({
            'delay_minutes': 'mean'
        }).reset_index()

        # Pivot to have routes as columns
        self.pivot_df = route_data.pivot(
            index='time_key',
            columns='route_id',
            values='delay_minutes'
        )

        # Fill missing with interpolation
        self.pivot_df = self.pivot_df.interpolate(method='linear', limit=6)
        self.pivot_df = self.pivot_df.fillna(0)

        # Convert to numpy
        self.data = self.pivot_df.values.astype(np.float32)
        self.timestamps = self.pivot_df.index.values
        self.route_names = self.pivot_df.columns.tolist()

        # Normalize
        self.mean = self.data.mean(axis=0, keepdims=True)
        self.std = self.data.std(axis=0, keepdims=True) + 1e-8
        self.data_normalized = (self.data - self.mean) / self.std

        # Split
        total_len = len(self.data) - self.seq_length - self.pred_length + 1
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)

        if self.mode == 'train':
            self.start_idx, self.end_idx = 0, train_end
        elif self.mode == 'val':
            self.start_idx, self.end_idx = train_end, val_end
        else:
            self.start_idx, self.end_idx = val_end, total_len

        print(f"Route Dataset - Mode: {self.mode}, Samples: {self.end_idx - self.start_idx}")
        print(f"Routes: {len(self.route_names)}, Time steps: {len(self.data)}")

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: [seq_length, num_routes]
            y: [pred_length, num_routes]
        """
        real_idx = self.start_idx + idx

        x = self.data_normalized[real_idx:real_idx + self.seq_length]
        y = self.data_normalized[real_idx + self.seq_length:
                                 real_idx + self.seq_length + self.pred_length]

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Convert back to original scale."""
        return y * torch.FloatTensor(self.std) + torch.FloatTensor(self.mean)


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    seq_length: int = 24,
    pred_length: int = 1,
    num_workers: int = 4,
    dataset_type: str = 'timeseries'  # 'timeseries' or 'route'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        data_path: Path to parquet file
        batch_size: Batch size
        seq_length: Input sequence length
        pred_length: Prediction length
        num_workers: Number of data loading workers
        dataset_type: Type of dataset to create

    Returns:
        train_loader, val_loader, test_loader
    """
    if dataset_type == 'timeseries':
        DatasetClass = BusDelayTimeSeriesDataset
    else:
        DatasetClass = BusDelayRouteDataset

    train_dataset = DatasetClass(
        data=data_path,
        seq_length=seq_length,
        pred_length=pred_length,
        mode='train'
    )

    val_dataset = DatasetClass(
        data=data_path,
        seq_length=seq_length,
        pred_length=pred_length,
        mode='val'
    )

    test_dataset = DatasetClass(
        data=data_path,
        seq_length=seq_length,
        pred_length=pred_length,
        mode='test'
    )

    # Copy normalization stats from training
    if hasattr(train_dataset, 'feature_mean'):
        val_dataset.feature_mean = train_dataset.feature_mean
        val_dataset.feature_std = train_dataset.feature_std
        val_dataset.target_mean = train_dataset.target_mean
        val_dataset.target_std = train_dataset.target_std
        test_dataset.feature_mean = train_dataset.feature_mean
        test_dataset.feature_std = train_dataset.feature_std
        test_dataset.target_mean = train_dataset.target_mean
        test_dataset.target_std = train_dataset.target_std

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    import sys

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw" / "arrival_departure"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

    print("="*60)
    print("MBTA Bus Delay Dataset Preparation")
    print("="*60)

    # Process data
    processor = BusDelayDataProcessor(
        data_dir=str(DATA_RAW),
        output_dir=str(DATA_PROCESSED)
    )

    # Load with sampling for quick test
    print("\nLoading data (10% sample for quick test)...")
    df = processor.load_and_process_all(
        years=[2023, 2024],
        sample_frac=0.1
    )

    # Save to parquet
    parquet_path = processor.save_to_parquet(df)

    # Create dataset
    print("\n" + "="*60)
    print("Creating PyTorch Dataset...")
    print("="*60)

    train_dataset = BusDelayTimeSeriesDataset(
        data=str(parquet_path),
        seq_length=24,
        pred_length=1,
        mode='train'
    )

    # Test loading
    x, y = train_dataset[0]
    print(f"\nSample input shape: {x.shape}")
    print(f"Sample target shape: {y.shape}")

    # Create DataLoaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=str(parquet_path),
        batch_size=32,
        seq_length=24,
        pred_length=1
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
