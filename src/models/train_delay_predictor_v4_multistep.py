"""
Bus Delay Prediction - V4 Multi-step Forecasting (Proper Sequence Model)
=========================================================================

Multi-step prediction using proper sequence-to-sequence architecture.
Input: sequence of past delay values (seq_len time steps)
Output: sequence of future delay predictions (horizon time steps)

Key difference from previous version:
- Model sees actual sequence of past delays, not just extracted features
- True seq2seq architecture with autoregressive decoding option

Author: Boston Bus Equity Team
Date: February 2025
"""

import os
import sys
import gc
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
# Seq2Seq Models for Multi-step Prediction
# =============================================================================

class Seq2SeqGRU(nn.Module):
    """
    Sequence-to-Sequence GRU for multi-step delay prediction.

    Architecture:
    - Encoder: GRU that processes input sequence of delay values
    - Decoder: GRU that generates future predictions step by step
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 128,
                 num_layers: int = 2, horizon: int = 5, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder GRU (autoregressive)
        self.decoder = nn.GRU(
            input_size=1,  # Takes previous prediction as input
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x, teacher_forcing_ratio=0.0):
        """
        x: [batch, seq_len, input_size] - input sequence
        Returns: [batch, horizon] - predictions for future steps
        """
        batch_size = x.size(0)

        # Encode input sequence
        _, hidden = self.encoder(x)  # hidden: [num_layers, batch, hidden_size]

        # Decode: generate predictions autoregressively
        predictions = []

        # Start with last known value
        decoder_input = x[:, -1:, :1]  # [batch, 1, 1]

        for step in range(self.horizon):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            pred = self.output_proj(decoder_output[:, -1, :])  # [batch, 1]
            predictions.append(pred)

            # Next input is current prediction
            decoder_input = pred.unsqueeze(1)  # [batch, 1, 1]

        return torch.cat(predictions, dim=1)  # [batch, horizon]


class Seq2SeqAttention(nn.Module):
    """
    Sequence-to-Sequence with Attention mechanism.

    The attention mechanism allows the decoder to focus on relevant parts
    of the input sequence when generating each prediction step.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 128,
                 num_layers: int = 2, horizon: int = 5, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # Encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=hidden_size + 1,  # context + previous prediction
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)

        # Encode
        encoder_outputs, hidden = self.encoder(x)  # [batch, seq_len, hidden]

        predictions = []
        decoder_input = x[:, -1:, :1]  # [batch, 1, 1] - last value

        for step in range(self.horizon):
            # Compute attention weights
            # Expand hidden state to match encoder outputs
            h = hidden[-1].unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)  # [batch, seq_len, hidden]
            attn_input = torch.cat([encoder_outputs, h], dim=2)  # [batch, seq_len, hidden*2]
            attn_weights = torch.softmax(self.attention(attn_input), dim=1)  # [batch, seq_len, 1]

            # Context vector
            context = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)  # [batch, 1, hidden]

            # Decoder input: context + previous value
            decoder_in = torch.cat([context, decoder_input], dim=2)  # [batch, 1, hidden+1]

            decoder_output, hidden = self.decoder(decoder_in, hidden)
            pred = self.output_proj(decoder_output[:, -1, :])  # [batch, 1]
            predictions.append(pred)

            decoder_input = pred.unsqueeze(1)  # [batch, 1, 1]

        return torch.cat(predictions, dim=1)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for sequence prediction.
    Uses dilated causal convolutions to capture long-range dependencies.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 128,
                 num_layers: int = 4, horizon: int = 5, dropout: float = 0.3):
        super().__init__()
        self.horizon = horizon

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=3,
                                   padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.conv_layers = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_size, horizon)

    def forward(self, x, teacher_forcing_ratio=0.0):
        # x: [batch, seq_len, input_size]
        x = x.transpose(1, 2)  # [batch, input_size, seq_len]
        x = self.conv_layers(x)  # [batch, hidden_size, seq_len]
        x = x[:, :, -1]  # Take last time step: [batch, hidden_size]
        return self.output_proj(x)  # [batch, horizon]


# =============================================================================
# Data Preparation
# =============================================================================

def load_data_sequences(seq_len: int = 10, horizon: int = 5, sample_size: int = 50000):
    """
    Load data and create sequences for multi-step prediction.

    For each route-stop combination:
    - Input: sequence of seq_len past delay values
    - Output: sequence of horizon future delay values

    NO data leakage: sequences are created within route-stop groups,
    and train/test split is temporal (train < 2025, test >= 2025).
    """

    print("\n" + "="*60)
    print(f"Loading Data (seq_len={seq_len}, horizon={horizon})")
    print("="*60)

    parquet_path = DATA_DIR / "arrival_departure.parquet"
    df = pd.read_parquet(parquet_path,
                         columns=['service_date', 'route_id', 'stop_id',
                                  'scheduled', 'actual'])

    print(f"Total records: {len(df):,}")

    # Sample first to reduce memory
    if len(df) > 5000000:
        df = df.sample(n=5000000, random_state=SEED)
    print(f"Sampled to: {len(df):,}")

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

    print(f"Train records: {len(train_df):,}")
    print(f"Test records: {len(test_df):,}")

    del df
    gc.collect()

    # Sort by time within route-stop groups
    train_df = train_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)
    test_df = test_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)

    def create_sequences(df_part, seq_len, horizon):
        """Create input-output sequence pairs from a dataframe."""
        X_list = []
        y_list = []

        for (route, stop), group in df_part.groupby(['route_id', 'stop_id']):
            if len(group) < seq_len + horizon:
                continue

            delays = group['delay_minutes'].values

            # Create sequences
            for i in range(len(delays) - seq_len - horizon + 1):
                X_list.append(delays[i:i+seq_len])
                y_list.append(delays[i+seq_len:i+seq_len+horizon])

        return np.array(X_list), np.array(y_list)

    print("\nCreating sequences...")
    X_train, y_train = create_sequences(train_df, seq_len, horizon)
    X_test, y_test = create_sequences(test_df, seq_len, horizon)

    del train_df, test_df
    gc.collect()

    # Sample if too large
    if len(X_train) > sample_size:
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    test_sample = min(sample_size // 4, len(X_test))
    if len(X_test) > test_sample:
        idx = np.random.choice(len(X_test), test_sample, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    # Reshape for model: [batch, seq_len, 1]
    X_train = X_train.reshape(-1, seq_len, 1)
    X_test = X_test.reshape(-1, seq_len, 1)

    print(f"Train sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")

    gc.collect()

    return X_train, y_train, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       model_class, model_name: str, horizon: int):
    """Train and evaluate a multi-step prediction model."""

    print(f"\n{'='*60}")
    print(f"Training {model_name} (horizon={horizon})")
    print("="*60)

    # Scale data
    # Flatten for scaling
    train_flat = X_train.reshape(-1, 1)
    test_flat = X_test.reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(train_flat)

    X_train_s = scaler.transform(train_flat).reshape(X_train.shape)
    X_test_s = scaler.transform(test_flat).reshape(X_test.shape)
    y_train_s = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # Train/validation split
    n_val = int(0.15 * len(X_train_s))
    X_tr, y_tr = X_train_s[:-n_val], y_train_s[:-n_val]
    X_val, y_val = X_train_s[-n_val:], y_train_s[-n_val:]

    # Data loaders
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

    # Initialize model
    model = model_class(
        input_size=1,
        hidden_size=128,
        num_layers=2,
        horizon=horizon,
        dropout=0.3
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # Training loop
    best_loss = float('inf')
    best_state = None
    patience = 10
    wait = 0

    start_time = time.time()

    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()

        val_loss /= len(val_loader)
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

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s")

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            all_preds.extend(output.cpu().numpy())
            all_actuals.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Inverse transform
    preds_inv = scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    actuals_inv = scaler.inverse_transform(all_actuals.reshape(-1, 1)).reshape(all_actuals.shape)

    # Metrics per prediction step
    print(f"\nResults (Horizon={horizon}):")
    print("-" * 50)

    metrics = {}
    for step in range(horizon):
        rmse = np.sqrt(mean_squared_error(actuals_inv[:, step], preds_inv[:, step]))
        mae = mean_absolute_error(actuals_inv[:, step], preds_inv[:, step])
        r2 = r2_score(actuals_inv[:, step], preds_inv[:, step])
        metrics[f'step_{step+1}'] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"  step_{step+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # Overall metrics
    overall_rmse = np.sqrt(mean_squared_error(actuals_inv.flatten(), preds_inv.flatten()))
    overall_mae = mean_absolute_error(actuals_inv.flatten(), preds_inv.flatten())
    overall_r2 = r2_score(actuals_inv.flatten(), preds_inv.flatten())
    metrics['overall'] = {'RMSE': overall_rmse, 'MAE': overall_mae, 'R2': overall_r2}
    print(f"  overall: RMSE={overall_rmse:.4f}, MAE={overall_mae:.4f}, R²={overall_r2:.4f}")

    return {
        'model': model_name,
        'horizon': horizon,
        'metrics': metrics,
        'train_time': train_time,
        'preds': preds_inv,
        'actuals': actuals_inv
    }


def main():
    print("="*60)
    print("V4: Multi-step Delay Prediction")
    print("="*60)
    print(f"Start: {datetime.now()}")
    print(f"Device: {DEVICE}")

    # Configuration
    seq_len = 10  # Input sequence length (past delays)
    horizons = [1, 3, 5]  # Prediction horizons
    sample_size = 50000  # Samples per horizon

    # Models to test
    models = [
        (Seq2SeqGRU, 'Seq2Seq-GRU'),
        # (Seq2SeqAttention, 'Seq2Seq-Attention'),  # Optional: more compute
        # (TemporalConvNet, 'TCN'),  # Optional: convolutional approach
    ]

    all_results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"Experiment: Seq2Seq-GRU, Horizon={horizon}")
        print("="*60)

        # Load data for this horizon
        X_train, y_train, X_test, y_test = load_data_sequences(
            seq_len=seq_len, horizon=horizon, sample_size=sample_size
        )

        for model_class, model_name in models:
            result = train_and_evaluate(
                X_train, y_train, X_test, y_test,
                model_class, model_name, horizon
            )
            all_results.append(result)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("MULTI-STEP PREDICTION RESULTS SUMMARY")
    print("="*60)

    summary = []
    for r in all_results:
        overall = r['metrics']['overall']
        summary.append({
            'Model': r['model'],
            'Horizon': r['horizon'],
            'RMSE': overall['RMSE'],
            'MAE': overall['MAE'],
            'R2': overall['R2'],
            'Train Time (s)': r['train_time']
        })

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    # Find best
    best_idx = df_summary['RMSE'].idxmin()
    best = df_summary.iloc[best_idx]
    print(f"\nBest Model: {best['Model']} at Horizon={best['Horizon']}")
    print(f"  RMSE: {best['RMSE']:.4f}, R²: {best['R2']:.4f}")

    # Save results
    df_summary.to_csv(REPORTS_DIR / "delay_prediction_multistep_results.csv", index=False)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE vs Horizon
    ax1 = axes[0, 0]
    for model_name in df_summary['Model'].unique():
        data = df_summary[df_summary['Model'] == model_name]
        ax1.plot(data['Horizon'], data['RMSE'], marker='o', linewidth=2, markersize=8, label=model_name)
    ax1.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax1.set_ylabel('RMSE (minutes)', fontsize=12)
    ax1.set_title('RMSE vs Prediction Horizon', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # R² vs Horizon
    ax2 = axes[0, 1]
    for model_name in df_summary['Model'].unique():
        data = df_summary[df_summary['Model'] == model_name]
        ax2.plot(data['Horizon'], data['R2'], marker='o', linewidth=2, markersize=8, label=model_name)
    ax2.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('R² vs Prediction Horizon', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Per-step RMSE for longest horizon
    ax3 = axes[1, 0]
    max_horizon = max(horizons)
    h_max_results = [r for r in all_results if r['horizon'] == max_horizon]
    for r in h_max_results:
        step_rmses = [r['metrics'][f'step_{i+1}']['RMSE'] for i in range(max_horizon)]
        ax3.plot(range(1, max_horizon+1), step_rmses, marker='o', linewidth=2, markersize=8, label=r['model'])
    ax3.set_xlabel('Prediction Step', fontsize=12)
    ax3.set_ylabel('RMSE (minutes)', fontsize=12)
    ax3.set_title(f'RMSE by Step (Horizon={max_horizon})', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bar comparison
    ax4 = axes[1, 1]
    models_list = df_summary['Model'].unique()
    width = 0.35
    x = np.arange(len(horizons))
    for i, model_name in enumerate(models_list):
        data = df_summary[df_summary['Model'] == model_name]
        rmses = [data[data['Horizon'] == h]['RMSE'].values[0] if len(data[data['Horizon'] == h]) > 0 else 0 for h in horizons]
        ax4.bar(x + i*width, rmses, width, label=model_name)
    ax4.set_xlabel('Prediction Horizon', fontsize=12)
    ax4.set_ylabel('RMSE (minutes)', fontsize=12)
    ax4.set_title('Model Comparison by Horizon', fontsize=14)
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels(horizons)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delay_prediction_multistep_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nEnd: {datetime.now()}")
    print(f"Results saved to: {REPORTS_DIR / 'delay_prediction_multistep_results.csv'}")
    print(f"Figure saved to: {FIGURES_DIR / 'delay_prediction_multistep_comparison.png'}")


if __name__ == "__main__":
    main()
