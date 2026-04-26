"""
Bus Delay Prediction - V5 Spiking Neural Network (NeuronSpark-inspired)
========================================================================

Implements a Spiking Neural Network for bus delay prediction, inspired by
the NeuronSpark project's selective state space model with dynamic membrane properties.

Key Features:
- Dynamic membrane parameters (β, α, V_th) that depend on input AND membrane potential
- Multi-timescale neuron groups for capturing different temporal patterns
- Spike-based information encoding and decoding
- Comparison with traditional GRU/LSTM baselines

Author: Boston Bus Equity Team
Date: February 2026
"""

import os
import sys
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# SpikingJelly imports
try:
    from spikingjelly.activation_based import neuron, surrogate, functional
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False
    print("Warning: spikingjelly not installed. Run: pip install spikingjelly")

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
print(f"SpikingJelly available: {HAS_SPIKINGJELLY}")


# =============================================================================
# SNN Components (NeuronSpark-inspired)
# =============================================================================

class DynamicLIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with dynamic parameters.

    Unlike standard LIF where β, V_th are fixed, here they depend on:
    - Current input x
    - Membrane potential V (feedback)

    This implements the key insight from NeuronSpark:
    β(t) = σ(W_β^x · x + W_β^V · V + b_β)
    V_th(t) = |W_th^x · x + W_th^V · V + b_th|
    """

    def __init__(self, input_size: int, hidden_size: int,
                 tau_range: Tuple[float, float] = (2.0, 20.0),
                 surrogate_function=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection
        self.W_in = nn.Linear(input_size, hidden_size)

        # Dynamic β (decay) parameters
        self.W_beta_x = nn.Linear(input_size, hidden_size)
        self.W_beta_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_beta = nn.Parameter(torch.zeros(hidden_size))

        # Dynamic threshold parameters
        self.W_th_x = nn.Linear(input_size, hidden_size)
        self.W_th_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_th = nn.Parameter(torch.ones(hidden_size) * 0.3)  # Base threshold

        # Surrogate gradient for spike
        if surrogate_function is None:
            self.surrogate = surrogate.ATan(alpha=2.0)
        else:
            self.surrogate = surrogate_function

        # Initialize with multi-timescale β distribution (HiPPO-inspired)
        self._initialize_multitimescale(tau_range)

        # Membrane potential (will be set during forward)
        self.v = None

    def _initialize_multitimescale(self, tau_range):
        """Initialize parameters for multi-timescale dynamics."""
        tau_min, tau_max = tau_range
        # β = 1 - 1/τ, so larger τ means larger β (slower decay)
        beta_min = 1 - 1/tau_min  # ~0.5 for fast
        beta_max = 1 - 1/tau_max  # ~0.95 for slow

        # Initialize b_beta to achieve distributed timescales
        target_betas = torch.linspace(beta_min, beta_max, self.hidden_size)
        # Inverse sigmoid to get initial bias
        init_bias = torch.log(target_betas / (1 - target_betas + 1e-8))
        self.b_beta.data = init_bias

        # Small initialization for weight matrices
        nn.init.xavier_uniform_(self.W_beta_x.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_beta_v.weight, gain=0.05)
        nn.init.xavier_uniform_(self.W_th_x.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_th_v.weight, gain=0.05)

    def reset(self):
        """Reset membrane potential."""
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one timestep.

        Args:
            x: Input tensor [batch, input_size]

        Returns:
            spike: Binary spike output [batch, hidden_size]
        """
        batch_size = x.size(0)

        # Initialize membrane potential if needed
        if self.v is None:
            self.v = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Compute dynamic parameters
        # β depends on input and membrane potential
        beta_input = self.W_beta_x(x) + self.W_beta_v(self.v) + self.b_beta
        beta = torch.sigmoid(beta_input)  # [0, 1]

        # Dynamic threshold (always positive)
        th_input = self.W_th_x(x) + self.W_th_v(self.v) + self.b_th
        v_th = torch.abs(th_input) + 0.1  # Minimum threshold of 0.1

        # Input current
        current = self.W_in(x)

        # LIF dynamics: V[t] = β * V[t-1] + (1-β) * I[t]
        self.v = beta * self.v + (1 - beta) * current

        # Spike generation with surrogate gradient
        spike = self.surrogate(self.v - v_th)

        # Soft reset: V = V - V_th * spike
        self.v = self.v - v_th * spike.detach()

        return spike


class SNNBlock(nn.Module):
    """
    SNN Block with dynamic LIF neurons.

    Architecture:
    Input → DynamicLIF → Dropout → Linear → Output
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.3):
        super().__init__()

        self.lif = DynamicLIFNeuron(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def reset(self):
        self.lif.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_size]
        Returns:
            out: [batch, output_size]
        """
        spike = self.lif(x)
        spike = self.dropout(spike)
        out = self.output_proj(spike)
        return out


class SNNDelayPredictor(nn.Module):
    """
    Spiking Neural Network for bus delay prediction.

    Architecture:
    1. Input encoding: continuous → spike trains (K timesteps per feature)
    2. SNN processing: multiple SNN blocks with dynamic neurons
    3. Output decoding: spike trains → continuous prediction
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_blocks: int = 2, K: int = 8, dropout: float = 0.3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.K = K  # Number of SNN timesteps per input
        self.num_blocks = num_blocks

        # Input encoder: maps continuous features to spike-friendly representation
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Sigmoid()  # [0, 1] for rate coding
        )

        # SNN blocks
        self.snn_blocks = nn.ModuleList([
            SNNBlock(hidden_size, hidden_size, hidden_size, dropout)
            for _ in range(num_blocks)
        ])

        # Output decoder: aggregates spikes over K timesteps
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def reset(self):
        """Reset all SNN blocks."""
        for block in self.snn_blocks:
            block.reset()

    def encode_to_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous input to spike trains using rate coding.

        Args:
            x: [batch, input_size] continuous features

        Returns:
            spikes: [K, batch, hidden_size] spike trains
        """
        # Get rate representation
        rate = self.encoder(x)  # [batch, hidden_size], values in [0, 1]

        # Generate K spike frames using Bernoulli sampling
        spikes = []
        for _ in range(self.K):
            spike_frame = (torch.rand_like(rate) < rate).float()
            spikes.append(spike_frame)

        return torch.stack(spikes, dim=0)  # [K, batch, hidden_size]

    def decode_spikes(self, spike_history: torch.Tensor) -> torch.Tensor:
        """
        Decode spike history to continuous output.

        Args:
            spike_history: [K, batch, hidden_size] accumulated outputs

        Returns:
            output: [batch, 1] predicted value
        """
        # Average over timesteps (spike rate)
        avg_activity = spike_history.mean(dim=0)  # [batch, hidden_size]
        return self.decoder(avg_activity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, input_size] input features

        Returns:
            pred: [batch, 1] predicted delay
        """
        self.reset()

        # Encode input to spike trains
        spike_input = self.encode_to_spikes(x)  # [K, batch, hidden_size]

        # Process through SNN blocks over K timesteps
        output_history = []
        for t in range(self.K):
            h = spike_input[t]  # [batch, hidden_size]

            for block in self.snn_blocks:
                h = block(h)

            output_history.append(h)

        output_history = torch.stack(output_history, dim=0)  # [K, batch, hidden_size]

        # Decode to prediction
        pred = self.decode_spikes(output_history)

        return pred.squeeze(-1)


class SimplifiedSNNPredictor(nn.Module):
    """
    Simplified SNN using SpikingJelly's built-in neurons.
    More stable for training.
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, K: int = 8, dropout: float = 0.3):
        super().__init__()

        self.K = K
        self.hidden_size = hidden_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )

        # SNN layers using SpikingJelly
        if HAS_SPIKINGJELLY:
            self.snn_layers = nn.ModuleList()
            for i in range(num_layers):
                self.snn_layers.append(nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
                    nn.Dropout(dropout)
                ))
        else:
            # Fallback to regular ReLU if SpikingJelly not available
            self.snn_layers = nn.ModuleList()
            for i in range(num_layers):
                self.snn_layers.append(nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_size]
        Returns:
            pred: [batch]
        """
        # Reset neuron states
        if HAS_SPIKINGJELLY:
            functional.reset_net(self)

        # Encode
        h = self.encoder(x)  # [batch, hidden_size]
        h = torch.sigmoid(h)  # Rate coding

        # Accumulate over K timesteps
        output_sum = torch.zeros_like(h)

        for t in range(self.K):
            # Add noise for spike generation
            spike_input = (torch.rand_like(h) < h).float()

            # Process through SNN layers
            out = spike_input
            for layer in self.snn_layers:
                out = layer(out)

            output_sum = output_sum + out

        # Average and decode
        avg_output = output_sum / self.K
        pred = self.decoder(avg_output)

        return pred.squeeze(-1)


# =============================================================================
# Hybrid Model: SNN + GRU
# =============================================================================

class HybridSNNGRU(nn.Module):
    """
    Hybrid model combining SNN's spike-based processing with GRU's
    sequence modeling capability.

    Architecture:
    1. SNN encoder: extract spike-based features
    2. GRU: model temporal dependencies
    3. Decoder: predict delay
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 K: int = 4, dropout: float = 0.3):
        super().__init__()

        self.K = K
        self.hidden_size = hidden_size

        # SNN feature extractor
        self.snn_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )

        if HAS_SPIKINGJELLY:
            self.snn_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            )
        else:
            self.snn_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_size]
        Returns:
            pred: [batch]
        """
        if HAS_SPIKINGJELLY:
            functional.reset_net(self)

        batch_size = x.size(0)

        # SNN encoding over K timesteps
        h = self.snn_encoder(x)
        h = torch.sigmoid(h)

        spike_sequence = []
        for t in range(self.K):
            spike_input = (torch.rand_like(h) < h).float()
            spike_out = self.snn_layer(spike_input)
            spike_sequence.append(spike_out)

        # Stack into sequence: [batch, K, hidden]
        spike_seq = torch.stack(spike_sequence, dim=1)

        # GRU processing
        gru_out, _ = self.gru(spike_seq)

        # Use last hidden state
        final_hidden = gru_out[:, -1, :]

        # Decode
        pred = self.decoder(final_hidden)

        return pred.squeeze(-1)


# =============================================================================
# Data Loading (reuse from V3)
# =============================================================================

class FeatureExtractor:
    """Feature extraction - same as V3."""

    @staticmethod
    def extract_lag_features(delays: np.ndarray, max_lag: int = 5) -> Dict[str, np.ndarray]:
        n = len(delays)
        features = {}
        for lag in range(1, max_lag + 1):
            lagged = np.zeros(n)
            lagged[lag:] = delays[:-lag]
            features[f'lag_{lag}'] = lagged
        diff1 = np.zeros(n)
        diff1[1:] = delays[1:] - delays[:-1]
        features['diff_1'] = diff1
        return features

    @staticmethod
    def extract_rolling_features(delays: np.ndarray, windows: List[int] = [5, 10]) -> Dict[str, np.ndarray]:
        n = len(delays)
        features = {}
        for w in windows:
            roll_mean = np.zeros(n)
            roll_std = np.zeros(n)
            for i in range(1, n):
                start = max(0, i - w)
                window_data = delays[start:i]
                if len(window_data) > 0:
                    roll_mean[i] = np.mean(window_data)
                    roll_std[i] = np.std(window_data) if len(window_data) > 1 else 0
            features[f'roll_mean_{w}'] = roll_mean
            features[f'roll_std_{w}'] = roll_std
        return features


def load_and_prepare_data(sample_size: int = 100000):
    """Load data with V3-style features."""

    print("\n" + "="*60)
    print("Loading Data for SNN Experiment")
    print("="*60)

    parquet_path = DATA_DIR / "arrival_departure.parquet"
    df = pd.read_parquet(parquet_path,
                         columns=['service_date', 'route_id', 'stop_id',
                                  'scheduled', 'actual'])

    print(f"Total records: {len(df):,}")

    # Sample to reduce memory
    if len(df) > 5000000:
        df = df.sample(n=5000000, random_state=SEED)

    # Parse dates
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

    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    del df
    gc.collect()

    # Sort
    train_df = train_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)
    test_df = test_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)

    # Add base features
    for df_part in [train_df, test_df]:
        df_part['hour'] = df_part['scheduled'].dt.hour
        df_part['dow'] = df_part['service_date'].dt.dayofweek
        df_part['is_weekend'] = (df_part['dow'] >= 5).astype(int)
        df_part['is_rush'] = ((df_part['hour'] >= 7) & (df_part['hour'] <= 9) |
                              (df_part['hour'] >= 16) & (df_part['hour'] <= 19)).astype(int)
        df_part['hour_sin'] = np.sin(2 * np.pi * df_part['hour'] / 24)
        df_part['hour_cos'] = np.cos(2 * np.pi * df_part['hour'] / 24)

    # Extract lag and rolling features
    extractor = FeatureExtractor()

    def extract_features_for_df(df_part):
        all_features = {}
        for (route, stop), group in df_part.groupby(['route_id', 'stop_id']):
            if len(group) < 15:
                continue
            delays = group['delay_minutes'].values
            indices = group.index

            lag_feats = extractor.extract_lag_features(delays, max_lag=5)
            roll_feats = extractor.extract_rolling_features(delays, windows=[5, 10])
            lag_feats.update(roll_feats)

            for feat_name, feat_vals in lag_feats.items():
                if feat_name not in all_features:
                    all_features[feat_name] = np.zeros(len(df_part))
                all_features[feat_name][indices] = feat_vals

        return all_features

    print("Extracting features...")
    train_features = extract_features_for_df(train_df)
    test_features = extract_features_for_df(test_df)

    # Combine features
    base_cols = ['is_weekend', 'is_rush', 'hour_sin', 'hour_cos']
    feature_names = base_cols + list(train_features.keys())

    X_train = np.column_stack([train_df[base_cols].values] +
                              [train_features[k] for k in train_features.keys()])
    X_test = np.column_stack([test_df[base_cols].values] +
                             [test_features.get(k, np.zeros(len(test_df))) for k in train_features.keys()])

    y_train = train_df['delay_minutes'].values
    y_test = test_df['delay_minutes'].values

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Sample
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    test_sample = min(sample_size // 4, len(X_test))
    if len(X_test) > test_sample:
        idx = np.random.choice(len(X_test), test_sample, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    print(f"Final - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Features: {len(feature_names)}")

    gc.collect()

    return X_train, y_train, X_test, y_test, feature_names


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       model_class, model_name: str, **model_kwargs):
    """Train and evaluate a model."""

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print("="*60)

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Split train/val
    n_val = int(0.15 * len(X_train_s))
    X_tr, y_tr = X_train_s[:-n_val], y_train_s[:-n_val]
    X_val, y_val = X_train_s[-n_val:], y_train_s[-n_val:]

    # DataLoaders
    batch_size = 256
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_s), torch.FloatTensor(y_test_s)),
        batch_size=batch_size
    )

    # Model
    input_size = X_train.shape[1]
    model = model_class(input_size=input_size, **model_kwargs).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # Training
    best_loss = float('inf')
    best_state = None
    patience = 10
    wait = 0

    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(50):
        # Train
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s")

    # Evaluate
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            pred = model(X_batch)
            all_preds.extend(pred.cpu().numpy())
            all_actuals.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Inverse transform
    preds_inv = scaler_y.inverse_transform(all_preds.reshape(-1, 1)).ravel()
    actuals_inv = scaler_y.inverse_transform(all_actuals.reshape(-1, 1)).ravel()

    # Metrics
    rmse = np.sqrt(mean_squared_error(actuals_inv, preds_inv))
    mae = mean_absolute_error(actuals_inv, preds_inv)
    r2 = r2_score(actuals_inv, preds_inv)

    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f} minutes")
    print(f"  MAE:  {mae:.4f} minutes")
    print(f"  R²:   {r2:.4f}")

    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_time': train_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'n_params': n_params
    }


def main():
    print("="*60)
    print("V5: SNN-based Delay Prediction (NeuronSpark-inspired)")
    print("="*60)
    print(f"Start: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"SpikingJelly: {HAS_SPIKINGJELLY}")

    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_and_prepare_data(
        sample_size=100000
    )

    # Models to compare
    models = [
        # SNN models
        (SimplifiedSNNPredictor, 'SNN-Simple', {'hidden_size': 128, 'num_layers': 2, 'K': 8}),
        (HybridSNNGRU, 'SNN-GRU-Hybrid', {'hidden_size': 128, 'K': 4}),
        (SNNDelayPredictor, 'SNN-Dynamic', {'hidden_size': 128, 'num_blocks': 2, 'K': 8}),
    ]

    all_results = []

    for model_class, model_name, kwargs in models:
        try:
            result = train_and_evaluate(
                X_train, y_train, X_test, y_test,
                model_class, model_name, **kwargs
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SNN EXPERIMENT RESULTS SUMMARY")
    print("="*60)

    summary_data = []
    for r in all_results:
        summary_data.append({
            'Model': r['model'],
            'RMSE': r['rmse'],
            'MAE': r['mae'],
            'R2': r['r2'],
            'Params': r['n_params'],
            'Time (s)': r['train_time']
        })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Best model
    if len(all_results) > 0:
        best_idx = df_summary['RMSE'].idxmin()
        best = df_summary.iloc[best_idx]
        print(f"\nBest SNN Model: {best['Model']}")
        print(f"  RMSE: {best['RMSE']:.4f}, R²: {best['R2']:.4f}")

    # Save results
    df_summary.to_csv(REPORTS_DIR / "delay_prediction_snn_results.csv", index=False)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training curves
    ax1 = axes[0]
    for r in all_results:
        ax1.plot(r['train_losses'], label=f"{r['model']} (train)", linestyle='-')
        ax1.plot(r['val_losses'], label=f"{r['model']} (val)", linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar comparison
    ax2 = axes[1]
    models_list = [r['model'] for r in all_results]
    rmses = [r['rmse'] for r in all_results]
    r2s = [r['r2'] for r in all_results]

    x = np.arange(len(models_list))
    width = 0.35

    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width/2, rmses, width, label='RMSE', color='steelblue')
    bars2 = ax2_twin.bar(x + width/2, r2s, width, label='R²', color='coral')

    ax2.set_xlabel('Model')
    ax2.set_ylabel('RMSE (minutes)', color='steelblue')
    ax2_twin.set_ylabel('R²', color='coral')
    ax2.set_title('Model Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_list, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delay_prediction_snn_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nEnd: {datetime.now()}")
    print(f"Results saved to: {REPORTS_DIR / 'delay_prediction_snn_results.csv'}")
    print(f"Figure saved to: {FIGURES_DIR / 'delay_prediction_snn_comparison.png'}")


if __name__ == "__main__":
    main()
