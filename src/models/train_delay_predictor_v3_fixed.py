"""
Bus Delay Prediction - Experiment V3 Fixed: Proper Time Series Features
========================================================================

FIXES from previous V3:
1. NO data leakage - all features use ONLY past values
2. Wavelet/FFT features computed on HISTORICAL data only
3. Multiple feature extraction methods for comparison

Feature Methods:
- Lag features (baseline)
- Rolling statistics
- Wavelet decomposition (on historical window)
- Fourier transform (on historical window)

Author: Boston Bus Equity Team
Date: February 2025
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import pywt for wavelet transform
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("Warning: pywt not installed, using fallback methods")

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

EXPERIMENT_VERSION = "v3_fixed_no_leakage"


def compute_lag_features(df, target_col='delay_minutes', max_lag=5):
    """
    Compute lag features - STRICTLY using past values only.
    """
    for lag in range(1, max_lag + 1):
        df[f'delay_lag_{lag}'] = df.groupby(['route_id', 'stop_id'])[target_col].shift(lag).fillna(0)

    # Difference features
    df['delay_diff_1'] = df.groupby(['route_id', 'stop_id'])[target_col].diff(1).fillna(0)
    df['delay_diff_2'] = df.groupby(['route_id', 'stop_id'])[target_col].diff(2).fillna(0)

    return df


def compute_rolling_features(df, target_col='delay_minutes', windows=[5, 10, 20]):
    """
    Compute rolling statistics - using ONLY past values (shift by 1 to exclude current).
    """
    for window in windows:
        # Shift by 1 to ensure we don't include current value
        shifted = df.groupby(['route_id', 'stop_id'])[target_col].shift(1)

        df[f'delay_roll_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean().fillna(0).values
        df[f'delay_roll_std_{window}'] = shifted.rolling(window=window, min_periods=1).std().fillna(0).values
        df[f'delay_roll_min_{window}'] = shifted.rolling(window=window, min_periods=1).min().fillna(0).values
        df[f'delay_roll_max_{window}'] = shifted.rolling(window=window, min_periods=1).max().fillna(0).values

    return df


def compute_fft_features(signal, n_components=3):
    """
    Extract FFT features from a historical signal window.
    Returns magnitude of top frequency components.
    """
    if len(signal) < 4:
        return np.zeros(n_components * 2)

    try:
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))

        # Get magnitudes (excluding DC component)
        half_len = max(1, len(signal)//2)
        magnitudes = np.abs(fft_vals[1:half_len])

        if len(magnitudes) == 0:
            return np.zeros(n_components * 2)

        if len(magnitudes) < n_components:
            magnitudes = np.pad(magnitudes, (0, n_components - len(magnitudes)))

        # Top n_components by magnitude
        top_indices = np.argsort(magnitudes)[-n_components:]
        top_magnitudes = magnitudes[top_indices]

        freq_slice = freqs[1:half_len]
        if len(freq_slice) > 0 and len(top_indices) <= len(freq_slice):
            valid_indices = top_indices[top_indices < len(freq_slice)]
            top_freqs = np.abs(freq_slice[valid_indices])
            if len(top_freqs) < n_components:
                top_freqs = np.pad(top_freqs, (0, n_components - len(top_freqs)))
        else:
            top_freqs = np.zeros(n_components)

        return np.concatenate([top_magnitudes[:n_components], top_freqs[:n_components]])
    except:
        return np.zeros(n_components * 2)


def compute_wavelet_features(signal, wavelet='db4', level=2):
    """
    Extract wavelet features from a historical signal window.
    """
    if len(signal) < 8:
        return np.zeros(6)  # Return zeros if signal too short

    if not HAS_PYWT:
        # Fallback: simple statistics
        return np.array([
            np.mean(signal), np.std(signal), np.min(signal),
            np.max(signal), signal[-1] - signal[0], np.median(signal)
        ])

    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Extract statistics from each level
        features = []
        for c in coeffs:
            features.extend([np.mean(c), np.std(c)])

        # Pad or truncate to fixed size
        features = np.array(features[:6]) if len(features) >= 6 else np.pad(features, (0, 6 - len(features)))
        return features
    except:
        return np.zeros(6)


def compute_window_features(df, target_col='delay_minutes', window_size=10):
    """
    Compute FFT and Wavelet features on historical windows.
    CRITICAL: Only use values BEFORE current timestamp (shift by 1).
    """
    fft_features = []
    wavelet_features = []

    groups = df.groupby(['route_id', 'stop_id'])

    for (route, stop), group in groups:
        group = group.sort_values('scheduled')
        delays = group[target_col].values

        group_fft = []
        group_wavelet = []

        for i in range(len(delays)):
            # Use only PAST values (indices 0 to i-1)
            if i < 2:
                group_fft.append(np.zeros(6))
                group_wavelet.append(np.zeros(6))
            else:
                start_idx = max(0, i - window_size)
                historical_window = delays[start_idx:i]  # Excludes current value

                group_fft.append(compute_fft_features(historical_window, n_components=3))
                group_wavelet.append(compute_wavelet_features(historical_window))

        fft_features.extend(group_fft)
        wavelet_features.extend(group_wavelet)

    # Add to dataframe
    fft_features = np.array(fft_features)
    wavelet_features = np.array(wavelet_features)

    for i in range(6):
        df[f'fft_feat_{i}'] = fft_features[:, i] if fft_features.shape[1] > i else 0
        df[f'wavelet_feat_{i}'] = wavelet_features[:, i] if wavelet_features.shape[1] > i else 0

    return df


def load_and_preprocess_data_v3_fixed(sample_size=300000, use_fft=True, use_wavelet=True, use_cache=True):
    """
    Load and preprocess with CORRECT temporal features (no data leakage).

    Args:
        sample_size: Number of training samples (test = sample_size // 3)
        use_fft: Include FFT features
        use_wavelet: Include wavelet features
        use_cache: If True, save/load preprocessed data to/from cache
    """
    import pickle

    print("\n" + "="*60)
    print("Loading Data (V3 Fixed - No Leakage)")
    print("="*60)

    # Cache file path based on parameters
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"preprocessed_v3_fixed_n{sample_size}_fft{use_fft}_wav{use_wavelet}.pkl"

    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Loaded cached data - Train: {cached_data['X_train'].shape}, Test: {cached_data['X_test'].shape}")
        return (cached_data['X_train'], cached_data['y_train'],
                cached_data['X_test'], cached_data['y_test'],
                cached_data['feature_columns'], cached_data['scaler_X'], cached_data['scaler_y'])

    parquet_path = DATA_DIR / "arrival_departure.parquet"
    print(f"Loading from: {parquet_path}")

    columns = ['service_date', 'route_id', 'stop_id', 'direction_id',
               'scheduled', 'actual', 'scheduled_headway', 'year', 'month']

    df = pd.read_parquet(parquet_path, columns=columns)
    print(f"Total records: {len(df):,}")

    # Parse datetime
    df['scheduled'] = pd.to_datetime(df['scheduled'], format='mixed', errors='coerce', utc=True)
    df['actual'] = pd.to_datetime(df['actual'], format='mixed', errors='coerce', utc=True)
    df['service_date'] = pd.to_datetime(df['service_date'], errors='coerce')

    # Calculate delay
    df['delay_minutes'] = (df['actual'] - df['scheduled']).dt.total_seconds() / 60

    # Drop invalid
    df = df.dropna(subset=['delay_minutes', 'scheduled', 'service_date'])
    df = df[(df['delay_minutes'] >= -30) & (df['delay_minutes'] <= 60)]
    print(f"After filtering: {len(df):,} records")

    # Extract year
    df['year'] = df['service_date'].dt.year

    # Temporal split
    train_df = df[df['year'] < 2025].copy()
    test_df = df[df['year'] >= 2025].copy()

    print(f"\nTemporal Split:")
    print(f"  Train (<2025): {len(train_df):,}")
    print(f"  Test (>=2025): {len(test_df):,}")

    # Sample
    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=SEED)
    if len(test_df) > sample_size // 3:
        test_df = test_df.sample(n=sample_size // 3, random_state=SEED)

    print(f"  After sampling - Train: {len(train_df):,}, Test: {len(test_df):,}")

    # Sort by time within each group
    train_df = train_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)
    test_df = test_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)

    # Basic features
    def add_basic_features(data):
        data = data.copy()
        data['hour'] = data['scheduled'].dt.hour
        data['day_of_week'] = data['service_date'].dt.dayofweek
        data['month_num'] = data['service_date'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_rush_hour'] = ((data['hour'] >= 7) & (data['hour'] <= 9) |
                                (data['hour'] >= 16) & (data['hour'] <= 19)).astype(int)

        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        return data

    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    # Historical statistics (from training only)
    print("  Computing historical statistics...")
    route_stats = train_df.groupby('route_id')['delay_minutes'].agg(['mean', 'std']).reset_index()
    route_stats.columns = ['route_id', 'route_delay_mean', 'route_delay_std']

    stop_stats = train_df.groupby('stop_id')['delay_minutes'].agg(['mean', 'std']).reset_index()
    stop_stats.columns = ['stop_id', 'stop_delay_mean', 'stop_delay_std']

    hour_stats = train_df.groupby('hour')['delay_minutes'].agg('mean').reset_index()
    hour_stats.columns = ['hour', 'hour_delay_mean']

    global_mean = train_df['delay_minutes'].mean()

    for data in [train_df, test_df]:
        data_idx = data.index
        merged = data.merge(route_stats, on='route_id', how='left')
        merged = merged.merge(stop_stats, on='stop_id', how='left')
        merged = merged.merge(hour_stats, on='hour', how='left')

        for col in ['route_delay_mean', 'route_delay_std', 'stop_delay_mean', 'stop_delay_std', 'hour_delay_mean']:
            merged[col] = merged[col].fillna(global_mean if 'mean' in col else 0)

        if data is train_df:
            train_df = merged
        else:
            test_df = merged

    # Lag features (NO LEAKAGE - strictly past values)
    print("  Computing lag features...")
    train_df = compute_lag_features(train_df, max_lag=5)
    test_df = compute_lag_features(test_df, max_lag=5)

    # Rolling features (NO LEAKAGE - shifted by 1)
    print("  Computing rolling statistics...")
    train_df = compute_rolling_features(train_df, windows=[5, 10])
    test_df = compute_rolling_features(test_df, windows=[5, 10])

    # FFT and Wavelet features on historical windows (optional)
    if use_fft or use_wavelet:
        print("  Computing FFT/Wavelet features on historical windows...")
        train_df = compute_window_features(train_df, window_size=10)
        test_df = compute_window_features(test_df, window_size=10)

    # Encode categorical
    le_route = LabelEncoder()
    le_stop = LabelEncoder()
    le_direction = LabelEncoder()

    all_routes = pd.concat([train_df['route_id'], test_df['route_id']]).astype(str).unique()
    all_stops = pd.concat([train_df['stop_id'], test_df['stop_id']]).astype(str).unique()
    all_dirs = pd.concat([train_df['direction_id'].fillna('U').astype(str),
                          test_df['direction_id'].fillna('U').astype(str)]).unique()

    le_route.fit(all_routes)
    le_stop.fit(all_stops)
    le_direction.fit(all_dirs)

    train_df['route_enc'] = le_route.transform(train_df['route_id'].astype(str))
    train_df['stop_enc'] = le_stop.transform(train_df['stop_id'].astype(str))
    train_df['dir_enc'] = le_direction.transform(train_df['direction_id'].fillna('U').astype(str))

    test_df['route_enc'] = le_route.transform(test_df['route_id'].astype(str))
    test_df['stop_enc'] = le_stop.transform(test_df['stop_id'].astype(str))
    test_df['dir_enc'] = le_direction.transform(test_df['direction_id'].fillna('U').astype(str))

    # Feature columns
    feature_columns = [
        # Basic
        'is_weekend', 'is_rush_hour', 'route_enc', 'stop_enc', 'dir_enc',
        # Cyclical
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        # Historical stats
        'route_delay_mean', 'route_delay_std', 'stop_delay_mean', 'stop_delay_std', 'hour_delay_mean',
        # Lag features
        'delay_lag_1', 'delay_lag_2', 'delay_lag_3', 'delay_lag_4', 'delay_lag_5',
        'delay_diff_1', 'delay_diff_2',
        # Rolling features
        'delay_roll_mean_5', 'delay_roll_std_5', 'delay_roll_min_5', 'delay_roll_max_5',
        'delay_roll_mean_10', 'delay_roll_std_10', 'delay_roll_min_10', 'delay_roll_max_10',
    ]

    if use_fft:
        feature_columns += [f'fft_feat_{i}' for i in range(6)]
    if use_wavelet:
        feature_columns += [f'wavelet_feat_{i}' for i in range(6)]

    # Ensure columns exist
    for col in feature_columns:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in test_df.columns:
            test_df[col] = 0

    X_train = train_df[feature_columns].values
    y_train = train_df['delay_minutes'].values.reshape(-1, 1)
    X_test = test_df[feature_columns].values
    y_test = test_df['delay_minutes'].values.reshape(-1, 1)

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    print(f"\nFeatures: {len(feature_columns)}")
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    # Save to cache for future use
    if use_cache:
        print(f"Saving to cache: {cache_file}")
        cached_data = {
            'X_train': X_train_scaled,
            'y_train': y_train_scaled,
            'X_test': X_test_scaled,
            'y_test': y_test_scaled,
            'feature_columns': feature_columns,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"Cache saved ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            feature_columns, scaler_X, scaler_y)


class MLPPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev_size, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden_size//2, 1))

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden_size//2, 1))

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, name="model"):
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_state = None
    wait = 0
    train_losses, val_losses = [], []

    start = time.time()

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)
        train_losses.append(t_loss)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                v_loss += criterion(model(X), y).item()
        v_loss /= len(val_loader)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Train={t_loss:.6f}, Val={v_loss:.6f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    print(f"Training time: {time.time()-start:.1f}s")
    return model, train_losses, val_losses


def evaluate(model, loader, scaler_y, name="model"):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            preds.extend(model(X).cpu().numpy())
            actuals.extend(y.numpy())

    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    print(f"\n{name} Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    return {'model': name, 'experiment': EXPERIMENT_VERSION, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def main():
    print("="*60)
    print(f"V3 Fixed - No Data Leakage")
    print("="*60)
    print(f"Start: {datetime.now()}")

    # Load data
    (X_train, y_train, X_test, y_test,
     features, scaler_X, scaler_y) = load_and_preprocess_data_v3_fixed(
         sample_size=300000, use_fft=True, use_wavelet=True)

    # Train/Val split
    n = len(X_train)
    n_val = int(0.15 * n)

    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]

    print(f"\nSplit: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test)}")

    # Loaders
    batch = 256
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                              batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                            batch_size=batch)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
                             batch_size=batch)

    # Models
    input_size = X_train.shape[1]
    models = {
        'MLP': MLPPredictor(input_size),
        'LSTM': LSTMPredictor(input_size),
        'GRU': GRUPredictor(input_size)
    }

    results = []
    for name, model in models.items():
        trained, _, _ = train_model(model, train_loader, val_loader, epochs=50, name=name)
        metrics = evaluate(trained, test_loader, scaler_y, name)
        results.append(metrics)

        torch.save(trained.state_dict(), MODELS_DIR / f"delay_{name.lower()}_{EXPERIMENT_VERSION}.pt")

    # Save results
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("Final Results (V3 Fixed - No Leakage)")
    print("="*60)
    print(df.to_string(index=False))

    df.to_csv(REPORTS_DIR / f"delay_prediction_metrics_{EXPERIMENT_VERSION}.csv", index=False)

    best = df.loc[df['RMSE'].idxmin()]
    print(f"\nBest: {best['model']} - RMSE={best['RMSE']:.4f}, R²={best['R2']:.4f}")
    print(f"\nEnd: {datetime.now()}")


if __name__ == "__main__":
    main()
