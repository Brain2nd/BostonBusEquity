"""
Bus Delay Prediction - V3 Ablation Study: Multiple Feature Extraction Methods
==============================================================================

Compare different signal processing methods for time series feature extraction:
1. Baseline (only lag features)
2. Rolling statistics
3. Fourier Transform (FFT)
4. Wavelet Transform (DWT)
5. EMD (Empirical Mode Decomposition) - if available
6. All combined

All methods ensure NO data leakage - only use past values.

Author: Boston Bus Equity Team
Date: February 2025
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

# Optional imports
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("pywt not available")

try:
    from PyEMD import EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False
    print("PyEMD not available")

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# =============================================================================
# Feature Extraction Methods (All using ONLY historical data)
# =============================================================================

class FeatureExtractor:
    """Base class for feature extraction - ensures no data leakage."""

    @staticmethod
    def extract_lag_features(delays: np.ndarray, max_lag: int = 5) -> Dict[str, np.ndarray]:
        """Lag features using shift (strictly past values)."""
        n = len(delays)
        features = {}

        for lag in range(1, max_lag + 1):
            lagged = np.zeros(n)
            lagged[lag:] = delays[:-lag]
            features[f'lag_{lag}'] = lagged

        # Differences
        diff1 = np.zeros(n)
        diff1[1:] = delays[1:] - delays[:-1]
        features['diff_1'] = diff1

        return features

    @staticmethod
    def extract_rolling_features(delays: np.ndarray, windows: List[int] = [5, 10, 20]) -> Dict[str, np.ndarray]:
        """Rolling statistics using ONLY past values (shift by 1)."""
        n = len(delays)
        features = {}

        for w in windows:
            roll_mean = np.zeros(n)
            roll_std = np.zeros(n)
            roll_min = np.zeros(n)
            roll_max = np.zeros(n)

            for i in range(1, n):
                start = max(0, i - w)
                window_data = delays[start:i]  # Excludes current
                if len(window_data) > 0:
                    roll_mean[i] = np.mean(window_data)
                    roll_std[i] = np.std(window_data) if len(window_data) > 1 else 0
                    roll_min[i] = np.min(window_data)
                    roll_max[i] = np.max(window_data)

            features[f'roll_mean_{w}'] = roll_mean
            features[f'roll_std_{w}'] = roll_std
            features[f'roll_min_{w}'] = roll_min
            features[f'roll_max_{w}'] = roll_max

        return features

    @staticmethod
    def extract_fft_features(delays: np.ndarray, window: int = 10, n_components: int = 3) -> Dict[str, np.ndarray]:
        """FFT features on historical window (excludes current value)."""
        n = len(delays)
        features = {f'fft_mag_{i}': np.zeros(n) for i in range(n_components)}
        features.update({f'fft_phase_{i}': np.zeros(n) for i in range(n_components)})

        for i in range(window, n):
            hist_window = delays[i-window:i]  # Excludes current

            try:
                fft_result = np.fft.fft(hist_window)
                freqs = np.fft.fftfreq(len(hist_window))

                # Get magnitudes and phases (exclude DC)
                mags = np.abs(fft_result[1:len(hist_window)//2+1])
                phases = np.angle(fft_result[1:len(hist_window)//2+1])

                # Top components by magnitude
                if len(mags) >= n_components:
                    top_idx = np.argsort(mags)[-n_components:]
                    for j, idx in enumerate(top_idx):
                        features[f'fft_mag_{j}'][i] = mags[idx]
                        features[f'fft_phase_{j}'][i] = phases[idx]
            except:
                pass

        return features

    @staticmethod
    def extract_wavelet_features(delays: np.ndarray, window: int = 10, wavelet: str = 'db4', level: int = 2) -> Dict[str, np.ndarray]:
        """Wavelet features on historical window."""
        n = len(delays)
        n_features = (level + 1) * 2  # mean and std for each level
        features = {f'wavelet_{i}': np.zeros(n) for i in range(n_features)}

        if not HAS_PYWT:
            return features

        for i in range(window, n):
            hist_window = delays[i-window:i]

            try:
                coeffs = pywt.wavedec(hist_window, wavelet, level=level)
                feat_idx = 0
                for c in coeffs:
                    features[f'wavelet_{feat_idx}'][i] = np.mean(c)
                    features[f'wavelet_{feat_idx+1}'][i] = np.std(c) if len(c) > 1 else 0
                    feat_idx += 2
                    if feat_idx >= n_features:
                        break
            except:
                pass

        return features

    @staticmethod
    def extract_emd_features(delays: np.ndarray, window: int = 20, n_imfs: int = 3) -> Dict[str, np.ndarray]:
        """EMD (Empirical Mode Decomposition) features."""
        n = len(delays)
        features = {f'imf_{i}_mean': np.zeros(n) for i in range(n_imfs)}
        features.update({f'imf_{i}_energy': np.zeros(n) for i in range(n_imfs)})

        if not HAS_EMD:
            return features

        emd = EMD()

        for i in range(window, n):
            hist_window = delays[i-window:i]

            try:
                imfs = emd.emd(hist_window, max_imf=n_imfs)
                for j in range(min(n_imfs, len(imfs))):
                    features[f'imf_{j}_mean'][i] = np.mean(imfs[j])
                    features[f'imf_{j}_energy'][i] = np.sum(imfs[j]**2)
            except:
                pass

        return features

    @staticmethod
    def extract_statistical_features(delays: np.ndarray, window: int = 10) -> Dict[str, np.ndarray]:
        """Additional statistical features on historical window."""
        n = len(delays)
        features = {
            'skewness': np.zeros(n),
            'kurtosis': np.zeros(n),
            'trend': np.zeros(n),
            'volatility': np.zeros(n),
        }

        for i in range(window, n):
            hist = delays[i-window:i]

            # Skewness
            mean_val = np.mean(hist)
            std_val = np.std(hist)
            if std_val > 0:
                features['skewness'][i] = np.mean(((hist - mean_val) / std_val) ** 3)
                features['kurtosis'][i] = np.mean(((hist - mean_val) / std_val) ** 4) - 3

            # Trend (linear regression slope)
            x = np.arange(len(hist))
            if len(hist) > 1:
                features['trend'][i] = np.polyfit(x, hist, 1)[0]

            # Volatility (rolling std of differences)
            if len(hist) > 1:
                diffs = np.diff(hist)
                features['volatility'][i] = np.std(diffs)

        return features


def load_base_data(sample_size: int = 200000):
    """Load and prepare base data with temporal split."""
    print("\n" + "="*60)
    print("Loading Base Data")
    print("="*60)

    parquet_path = DATA_DIR / "arrival_departure.parquet"
    df = pd.read_parquet(parquet_path,
                         columns=['service_date', 'route_id', 'stop_id', 'direction_id',
                                  'scheduled', 'actual', 'scheduled_headway'])

    print(f"Total: {len(df):,}")

    # Parse and filter
    df['scheduled'] = pd.to_datetime(df['scheduled'], format='mixed', errors='coerce', utc=True)
    df['actual'] = pd.to_datetime(df['actual'], format='mixed', errors='coerce', utc=True)
    df['service_date'] = pd.to_datetime(df['service_date'], errors='coerce')
    df['delay_minutes'] = (df['actual'] - df['scheduled']).dt.total_seconds() / 60
    df = df.dropna(subset=['delay_minutes', 'scheduled', 'service_date'])
    df = df[(df['delay_minutes'] >= -30) & (df['delay_minutes'] <= 60)]
    df['year'] = df['service_date'].dt.year

    # Temporal split
    train_df = df[df['year'] < 2025].copy()
    test_df = df[df['year'] >= 2025].copy()

    # Sample
    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=SEED)
    if len(test_df) > sample_size // 3:
        test_df = test_df.sample(n=sample_size // 3, random_state=SEED)

    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    # Sort
    train_df = train_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)
    test_df = test_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)

    return train_df, test_df


def add_base_features(df, train_stats=None):
    """Add basic features without signal processing."""
    df = df.copy()

    df['hour'] = df['scheduled'].dt.hour
    df['dow'] = df['service_date'].dt.dayofweek
    df['month'] = df['service_date'].dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                     (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)

    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    return df


def prepare_features_for_method(train_df, test_df, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare features based on specified method."""
    print(f"\n  Preparing features for method: {method}")

    extractor = FeatureExtractor()

    def extract_for_df(df, method):
        all_features = {}

        # Always include lag features as baseline
        for (route, stop), group in df.groupby(['route_id', 'stop_id']):
            delays = group['delay_minutes'].values
            indices = group.index

            # Lag features (always included)
            lag_feats = extractor.extract_lag_features(delays, max_lag=5)

            if method in ['rolling', 'all']:
                roll_feats = extractor.extract_rolling_features(delays, windows=[5, 10])
                lag_feats.update(roll_feats)

            if method in ['fft', 'all']:
                fft_feats = extractor.extract_fft_features(delays, window=10, n_components=3)
                lag_feats.update(fft_feats)

            if method in ['wavelet', 'all'] and HAS_PYWT:
                wav_feats = extractor.extract_wavelet_features(delays, window=10)
                lag_feats.update(wav_feats)

            if method in ['emd', 'all'] and HAS_EMD:
                emd_feats = extractor.extract_emd_features(delays, window=20)
                lag_feats.update(emd_feats)

            if method in ['stats', 'all']:
                stat_feats = extractor.extract_statistical_features(delays, window=10)
                lag_feats.update(stat_feats)

            for feat_name, feat_vals in lag_feats.items():
                if feat_name not in all_features:
                    all_features[feat_name] = np.zeros(len(df))
                all_features[feat_name][indices] = feat_vals

        return all_features

    # Extract features
    train_signal_feats = extract_for_df(train_df, method)
    test_signal_feats = extract_for_df(test_df, method)

    # Combine with base features
    base_cols = ['is_weekend', 'is_rush', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

    feature_names = base_cols + list(train_signal_feats.keys())

    X_train = np.column_stack([train_df[base_cols].values] +
                              [train_signal_feats[k] for k in train_signal_feats.keys()])
    X_test = np.column_stack([test_df[base_cols].values] +
                             [test_signal_feats.get(k, np.zeros(len(test_df))) for k in train_signal_feats.keys()])

    y_train = train_df['delay_minutes'].values.reshape(-1, 1)
    y_test = test_df['delay_minutes'].values.reshape(-1, 1)

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Features: {len(feature_names)}, Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_test, y_test, feature_names


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden)
        self.gru = nn.GRU(hidden, hidden, 2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden//2, 1))

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def train_and_evaluate(X_train, y_train, X_test, y_test, method_name: str):
    """Train GRU model and evaluate."""
    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    y_train_s = scaler_y.fit_transform(y_train)
    X_test_s = scaler_X.transform(X_test)
    y_test_s = scaler_y.transform(y_test)

    # Split train/val
    n = len(X_train_s)
    n_val = int(0.15 * n)
    X_tr, y_tr = X_train_s[:-n_val], y_train_s[:-n_val]
    X_val, y_val = X_train_s[-n_val:], y_train_s[-n_val:]

    # Loaders
    batch = 256
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                              batch_size=batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                            batch_size=batch)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_s), torch.FloatTensor(y_test_s)),
                             batch_size=batch)

    # Model
    model = SimpleGRU(X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # Train
    best_loss = float('inf')
    best_state = None
    patience = 10
    wait = 0

    print(f"\n  Training {method_name}...")
    start = time.time()

    for epoch in range(50):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                v_loss += criterion(model(X), y).item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    train_time = time.time() - start

    # Evaluate
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            preds.extend(model(X).cpu().numpy())
            actuals.extend(y.numpy())

    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    print(f"    {method_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f} ({train_time:.1f}s)")

    return {
        'method': method_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'train_time': train_time,
        'n_features': X_train.shape[1]
    }


def main():
    print("="*60)
    print("Ablation Study: Feature Extraction Methods")
    print("="*60)
    print(f"Start: {datetime.now()}")
    print(f"PyWT: {HAS_PYWT}, PyEMD: {HAS_EMD}")

    # Load data
    train_df, test_df = load_base_data(sample_size=200000)

    # Add base features
    train_df = add_base_features(train_df)
    test_df = add_base_features(test_df)

    # Methods to compare
    methods = [
        'baseline',  # Only lag features
        'rolling',   # Lag + rolling statistics
        'fft',       # Lag + FFT features
        'wavelet',   # Lag + Wavelet features
        'stats',     # Lag + statistical features
        'all',       # All combined
    ]

    if HAS_EMD:
        methods.insert(-1, 'emd')  # Insert before 'all'

    results = []

    for method in methods:
        X_train, y_train, X_test, y_test, feat_names = prepare_features_for_method(
            train_df, test_df, method)
        result = train_and_evaluate(X_train, y_train, X_test, y_test, method)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('RMSE')
    print(df_results.to_string(index=False))

    # Save
    df_results.to_csv(REPORTS_DIR / "delay_prediction_ablation_study.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods_sorted = df_results['method'].values
    rmse_vals = df_results['RMSE'].values
    r2_vals = df_results['R2'].values
    n_feats = df_results['n_features'].values

    axes[0].barh(methods_sorted, rmse_vals, color='steelblue')
    axes[0].set_xlabel('RMSE (minutes)')
    axes[0].set_title('RMSE by Method')
    axes[0].invert_yaxis()

    axes[1].barh(methods_sorted, r2_vals, color='seagreen')
    axes[1].set_xlabel('R²')
    axes[1].set_title('R² by Method')
    axes[1].invert_yaxis()

    axes[2].barh(methods_sorted, n_feats, color='coral')
    axes[2].set_xlabel('Number of Features')
    axes[2].set_title('Feature Count by Method')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_study_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nBest method: {df_results.iloc[0]['method']} (RMSE={df_results.iloc[0]['RMSE']:.4f})")
    print(f"\nEnd: {datetime.now()}")


if __name__ == "__main__":
    main()
