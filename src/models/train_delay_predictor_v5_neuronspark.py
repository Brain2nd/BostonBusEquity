"""
Bus Delay Prediction - V5 NeuronSpark SNN Model (Correct Implementation)
=========================================================================

Correctly adapts NeuronSpark's SNN architecture for bus delay regression.

Key Design (from NeuronSpark):
- K-bit deterministic binary encoding: float [0,1] → K binary spike frames
- SNNBlock with dynamic membrane parameters (β, α, V_th)
- K-bit binary decoding: K spike frames → float [0,1]

Encoding example (K=8):
  0.75 = 0.5 + 0.25 = 2^{-1} + 2^{-2}
  → [1, 1, 0, 0, 0, 0, 0, 0] (MSB-first)

Author: Boston Bus Equity Team
Date: February 2026
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add NeuronSpark to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
NEURONSPARK_PATH = PROJECT_ROOT / "NeuronSpark"
sys.path.insert(0, str(NEURONSPARK_PATH))

# Import NeuronSpark components
try:
    from atomic_ops import SNNBlock
    from spikingjelly.activation_based import functional
    HAS_NEURONSPARK = True
    print("NeuronSpark modules loaded successfully!")
except ImportError as e:
    HAS_NEURONSPARK = False
    print(f"Warning: Could not import NeuronSpark: {e}")

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# =============================================================================
# NeuronSpark-based Delay Predictor (Correct Implementation)
# =============================================================================

class SNNDelayRegressor(nn.Module):
    """
    Bus delay regressor using NeuronSpark's SNNBlock.

    Architecture (following NeuronSpark exactly):
    1. Input → Linear → sigmoid → [0,1]^D
    2. K-bit binary encoding: [0,1] → K frames of {0,1}^D spikes (DETERMINISTIC)
    3. K timesteps through L SNNBlocks (serial within each timestep)
    4. K-bit binary decoding: K spike frames → [0,1]^D
    5. Linear → delay prediction

    The key insight: binary encoding is DETERMINISTIC, not stochastic!
    For h ∈ [0,1), extract bits MSB-first:
      bit[k] = 1 if residual >= 0.5 else 0
      residual = (residual - bit[k] * 0.5) * 2
    """

    def __init__(
        self,
        input_size: int,
        D: int = 64,
        N: int = 8,
        K: int = 8,
        num_blocks: int = 2,
        v_th_min: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.D = D
        self.N = N
        self.K = K
        self.num_blocks = num_blocks

        # Encoder: features → D-dim [0,1] representation
        self.encode_proj = nn.Linear(input_size, D)

        # SNN Blocks (from NeuronSpark)
        self.blocks = nn.ModuleList([
            SNNBlock(D=D, N=N, v_th_min=v_th_min)
            for _ in range(num_blocks)
        ])

        # Decoder: [0,1]^D → delay prediction
        self.decode_head = nn.Linear(D, 1)

        # Binary weights: 2^{-1}, 2^{-2}, ..., 2^{-K} (MSB-first)
        self.register_buffer(
            'bit_weights',
            torch.tensor([2.0 ** (-(k + 1)) for k in range(K)])
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encode_proj.weight)
        nn.init.zeros_(self.encode_proj.bias)
        nn.init.xavier_uniform_(self.decode_head.weight)
        nn.init.zeros_(self.decode_head.bias)

    def _encode_binary(self, h: torch.Tensor) -> List[torch.Tensor]:
        """
        K-bit deterministic binary encoding (MSB-first).

        Args:
            h: Values in [0, 1], shape (batch, D)

        Returns:
            List of K spike frames, each (batch, D) with values {0, 1}

        Example:
            h = 0.75 = 0.5 + 0.25 = 2^{-1} + 2^{-2}
            → frames = [1, 1, 0, 0, 0, 0, 0, 0]
        """
        frames = []
        residual = h.clone()

        for k in range(self.K):
            bit = (residual >= 0.5).float()  # Deterministic threshold
            frames.append(bit)
            residual = (residual - bit * 0.5) * 2.0

        return frames

    def _decode_binary(self, spike_frames: List[torch.Tensor]) -> torch.Tensor:
        """
        K-bit binary decoding.

        Args:
            spike_frames: List of K frames, each (batch, D) with {0, 1}

        Returns:
            Decoded values in [0, 1], shape (batch, D)

        Formula: ŷ_d = Σ_{k=1}^{K} spike_d[k] · 2^{-k}
        """
        decoded = torch.zeros_like(spike_frames[0])
        for k, frame in enumerate(spike_frames):
            decoded = decoded + frame * self.bit_weights[k]
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (v7: uses forward_parallel for efficiency).

        Args:
            x: Input features, shape (batch, input_size)

        Returns:
            pred: Predicted delay, shape (batch,)
        """
        # Reset all neuron states (V → 0)
        for block in self.blocks:
            functional.reset_net(block)

        # 1. Encode input to [0, 1]^D
        h = torch.sigmoid(self.encode_proj(x))  # (batch, D)

        # 2. K-bit binary encoding (DETERMINISTIC)
        spike_frames_in = self._encode_binary(h)  # List of K frames, each (batch, D)

        # 3. Stack frames: List[(batch, D)] -> (K, batch, D)
        spike_seq = torch.stack(spike_frames_in, dim=0)  # (K, batch, D)

        # 4. Process through SNN blocks using parallel scan (v7)
        for block in self.blocks:
            if hasattr(block, 'forward_parallel'):
                spike_seq = block.forward_parallel(spike_seq)  # (K, batch, D)
            else:
                # Fallback to single-step for compatibility
                spike_frames_out = []
                for k in range(self.K):
                    spike = spike_seq[k]
                    spike = block(spike)
                    spike_frames_out.append(spike)
                spike_seq = torch.stack(spike_frames_out, dim=0)

        # 5. Unstack: (K, batch, D) -> List of K frames
        spike_frames_out = [spike_seq[k] for k in range(self.K)]

        # 6. Binary decode K output spike frames → [0,1]^D
        decoded = self._decode_binary(spike_frames_out)  # (batch, D)

        # 7. Output head → delay prediction
        pred = self.decode_head(decoded)  # (batch, 1)

        return pred.squeeze(-1)


class TransformerRegressor(nn.Module):
    """
    Transformer-based delay regressor with similar parameter count to NeuronSpark.

    NeuronSpark-D128-K8 has ~1.38M parameters.
    Target: ~1.4M parameters for fair comparison.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = 8,  # Match K timesteps
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features, shape (batch, input_size)

        Returns:
            pred: Predicted delay, shape (batch,)
        """
        batch_size = x.shape[0]

        # Project input and expand to sequence
        h = self.input_proj(x)  # (batch, d_model)
        h = h.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, d_model)

        # Add positional encoding
        h = h + self.pos_embedding

        # Transformer encoding
        h = self.transformer(h)  # (batch, seq_len, d_model)

        # Pool over sequence (mean pooling)
        h = h.mean(dim=1)  # (batch, d_model)

        # Output prediction
        pred = self.output_head(h)  # (batch, 1)

        return pred.squeeze(-1)


class GRUBaseline(nn.Module):
    """GRU baseline model for comparison."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).unsqueeze(1)
        out, _ = self.gru(h)
        pred = self.decoder(out[:, -1, :])
        return pred.squeeze(-1)


# =============================================================================
# Data Loading
# =============================================================================

class FeatureExtractor:
    @staticmethod
    def extract_lag_features(delays: np.ndarray, max_lag: int = 5):
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
    def extract_rolling_features(delays: np.ndarray, windows: List[int] = [5, 10]):
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
    print("\n" + "="*60)
    print("Loading Data for NeuronSpark SNN Experiment")
    print("="*60)

    parquet_path = DATA_DIR / "arrival_departure.parquet"
    df = pd.read_parquet(parquet_path,
                         columns=['service_date', 'route_id', 'stop_id',
                                  'scheduled', 'actual'])

    print(f"Total records: {len(df):,}")

    if len(df) > 5000000:
        df = df.sample(n=5000000, random_state=SEED)

    df['scheduled'] = pd.to_datetime(df['scheduled'], format='mixed', errors='coerce', utc=True)
    df['actual'] = pd.to_datetime(df['actual'], format='mixed', errors='coerce', utc=True)
    df['service_date'] = pd.to_datetime(df['service_date'], errors='coerce')
    df['delay_minutes'] = (df['actual'] - df['scheduled']).dt.total_seconds() / 60

    df = df.dropna(subset=['delay_minutes', 'scheduled', 'service_date'])
    df = df[(df['delay_minutes'] >= -30) & (df['delay_minutes'] <= 60)]
    df['year'] = df['service_date'].dt.year

    train_df = df[df['year'] < 2025].copy()
    test_df = df[df['year'] >= 2025].copy()

    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    del df
    gc.collect()

    train_df = train_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)
    test_df = test_df.sort_values(['route_id', 'stop_id', 'scheduled']).reset_index(drop=True)

    for df_part in [train_df, test_df]:
        df_part['hour'] = df_part['scheduled'].dt.hour
        df_part['dow'] = df_part['service_date'].dt.dayofweek
        df_part['is_weekend'] = (df_part['dow'] >= 5).astype(int)
        df_part['is_rush'] = ((df_part['hour'] >= 7) & (df_part['hour'] <= 9) |
                              (df_part['hour'] >= 16) & (df_part['hour'] <= 19)).astype(int)
        df_part['hour_sin'] = np.sin(2 * np.pi * df_part['hour'] / 24)
        df_part['hour_cos'] = np.cos(2 * np.pi * df_part['hour'] / 24)

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

    base_cols = ['is_weekend', 'is_rush', 'hour_sin', 'hour_cos']
    feature_names = base_cols + list(train_features.keys())

    X_train = np.column_stack([train_df[base_cols].values] +
                              [train_features[k] for k in train_features.keys()])
    X_test = np.column_stack([test_df[base_cols].values] +
                             [test_features.get(k, np.zeros(len(test_df))) for k in train_features.keys()])

    y_train = train_df['delay_minutes'].values
    y_test = test_df['delay_minutes'].values

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Use all data if sample_size is None
    if sample_size is not None:
        if len(X_train) > sample_size:
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]

        test_sample = min(sample_size // 4, len(X_test))
        if len(X_test) > test_sample:
            idx = np.random.choice(len(X_test), test_sample, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]

    print(f"Final - X_train: {X_train.shape}, X_test: {X_test.shape}")

    gc.collect()

    return X_train, y_train, X_test, y_test, feature_names


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       model_class, model_name: str, **model_kwargs):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print("="*60)

    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    n_val = int(0.15 * len(X_train_s))
    X_tr, y_tr = X_train_s[:-n_val], y_train_s[:-n_val]
    X_val, y_val = X_train_s[-n_val:], y_train_s[-n_val:]

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

    input_size = X_train.shape[1]
    model = model_class(input_size=input_size, **model_kwargs).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_state = None
    patience = 10
    wait = 0

    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(50):
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
            print(f"  Epoch {epoch+1}: train={epoch_loss:.6f}, val={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s")

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

    preds_inv = scaler_y.inverse_transform(all_preds.reshape(-1, 1)).ravel()
    actuals_inv = scaler_y.inverse_transform(all_actuals.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(actuals_inv, preds_inv))
    mae = mean_absolute_error(actuals_inv, preds_inv)
    r2 = r2_score(actuals_inv, preds_inv)

    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f} min")
    print(f"  MAE:  {mae:.4f} min")
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
    print("V5: NeuronSpark SNN for Bus Delay Prediction")
    print("="*60)
    print(f"Start: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"NeuronSpark available: {HAS_NEURONSPARK}")

    if not HAS_NEURONSPARK:
        print("ERROR: NeuronSpark not available. Cannot proceed.")
        return

    X_train, y_train, X_test, y_test, feature_names = load_and_prepare_data(
        sample_size=None  # Use all data
    )

    # NeuronSpark v7 with parallel scan
    models = [
        (SNNDelayRegressor, 'NeuronSpark-v7-D128-K8', {'D': 128, 'N': 8, 'K': 8, 'num_blocks': 2}),
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
            import traceback
            traceback.print_exc()
            continue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("NEURONSPARK SNN EXPERIMENT RESULTS")
    print("="*60)

    if len(all_results) == 0:
        print("No results to display.")
        return

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

    best_idx = df_summary['RMSE'].idxmin()
    best = df_summary.iloc[best_idx]
    print(f"\nBest Model: {best['Model']}")
    print(f"  RMSE: {best['RMSE']:.4f}, R²: {best['R2']:.4f}")

    df_summary.to_csv(REPORTS_DIR / "delay_prediction_neuronspark_results.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for r in all_results:
        ax1.plot(r['train_losses'], label=f"{r['model']} (train)", linestyle='-')
        ax1.plot(r['val_losses'], label=f"{r['model']} (val)", linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    models_list = [r['model'] for r in all_results]
    rmses = [r['rmse'] for r in all_results]
    r2s = [r['r2'] for r in all_results]

    x = np.arange(len(models_list))
    width = 0.35

    ax2_twin = ax2.twinx()
    ax2.bar(x - width/2, rmses, width, label='RMSE', color='steelblue')
    ax2_twin.bar(x + width/2, r2s, width, label='R²', color='coral')

    ax2.set_xlabel('Model')
    ax2.set_ylabel('RMSE (min)', color='steelblue')
    ax2_twin.set_ylabel('R²', color='coral')
    ax2.set_title('NeuronSpark vs GRU Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_list, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delay_prediction_neuronspark_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nEnd: {datetime.now()}")
    print(f"Results saved to: {REPORTS_DIR / 'delay_prediction_neuronspark_results.csv'}")


if __name__ == "__main__":
    main()
