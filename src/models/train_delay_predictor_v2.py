"""
Bus Delay Prediction - Experiment V2: Lag Features + Cyclical Encoding
=======================================================================

Improvements over V1:
1. Cyclical encoding for temporal features (hour, day_of_week, month)
2. Historical statistics computed ONLY on training data (no leakage)
3. Temporal split: Train <2025, Test >=2025

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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Experiment version
EXPERIMENT_VERSION = "v2_lag_features_temporal"


def load_and_preprocess_data_v2_temporal(sample_size: int = 500000):
    """
    Load and preprocess data with enhanced features.

    TEMPORAL SPLIT: Train <2025, Test >=2025
    Historical statistics computed ONLY on training data.
    """
    print("\n" + "="*60)
    print("Loading and Preprocessing Data (V2 - Temporal Split)")
    print("="*60)

    parquet_path = DATA_DIR / "arrival_departure.parquet"
    print(f"Loading from: {parquet_path}")

    columns = ['service_date', 'route_id', 'stop_id', 'direction_id',
               'scheduled', 'actual', 'scheduled_headway', 'year', 'month']

    df = pd.read_parquet(parquet_path, columns=columns)
    print(f"Total records: {len(df):,}")

    # Parse datetime
    print("Parsing datetime columns...")
    df['scheduled'] = pd.to_datetime(df['scheduled'], format='mixed', errors='coerce', utc=True)
    df['actual'] = pd.to_datetime(df['actual'], format='mixed', errors='coerce', utc=True)
    df['service_date'] = pd.to_datetime(df['service_date'], errors='coerce')

    # Calculate delay
    df['delay_minutes'] = (df['actual'] - df['scheduled']).dt.total_seconds() / 60

    # Drop invalid
    df = df.dropna(subset=['delay_minutes', 'scheduled', 'service_date'])

    # Filter outliers
    df = df[(df['delay_minutes'] >= -30) & (df['delay_minutes'] <= 60)]
    print(f"After filtering outliers: {len(df):,} records")

    # Extract year for temporal split
    df['year'] = df['service_date'].dt.year

    # =========================================
    # TEMPORAL SPLIT: Train <2025, Test >=2025
    # =========================================
    train_df = df[df['year'] < 2025].copy()
    test_df = df[df['year'] >= 2025].copy()

    print(f"\nTemporal Split:")
    print(f"  Train (before 2025): {len(train_df):,} records")
    print(f"  Test (2025+): {len(test_df):,} records")

    # Sample if too large
    max_train = sample_size
    max_test = 100000

    if len(train_df) > max_train:
        train_df = train_df.sample(n=max_train, random_state=SEED)
        print(f"  Train sampled to: {len(train_df):,}")

    if len(test_df) > max_test:
        test_df = test_df.sample(n=max_test, random_state=SEED)
        print(f"  Test sampled to: {len(test_df):,}")

    # Basic features
    def add_basic_features(data):
        data = data.copy()
        data['hour'] = data['scheduled'].dt.hour
        data['day_of_week'] = data['service_date'].dt.dayofweek
        data['month'] = data['service_date'].dt.month
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_rush_hour'] = ((data['hour'] >= 7) & (data['hour'] <= 9) |
                                (data['hour'] >= 16) & (data['hour'] <= 19)).astype(int)

        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        return data

    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    # =========================================
    # V2: Historical statistics ONLY on training data
    # =========================================
    print("  Computing historical statistics on TRAINING data only...")

    # Route-level stats from training
    route_stats = train_df.groupby('route_id')['delay_minutes'].agg(['mean', 'std']).reset_index()
    route_stats.columns = ['route_id', 'route_delay_mean', 'route_delay_std']
    route_stats['route_delay_std'] = route_stats['route_delay_std'].fillna(0)

    # Stop-level stats from training
    stop_stats = train_df.groupby('stop_id')['delay_minutes'].agg(['mean', 'std']).reset_index()
    stop_stats.columns = ['stop_id', 'stop_delay_mean', 'stop_delay_std']
    stop_stats['stop_delay_std'] = stop_stats['stop_delay_std'].fillna(0)

    # Hour-level stats from training
    hour_stats = train_df.groupby('hour')['delay_minutes'].agg('mean').reset_index()
    hour_stats.columns = ['hour', 'hour_delay_mean']

    # Route+Hour combination stats
    train_df['route_hour_key'] = train_df['route_id'].astype(str) + '_' + train_df['hour'].astype(str)
    route_hour_stats = train_df.groupby('route_hour_key')['delay_minutes'].agg('mean').reset_index()
    route_hour_stats.columns = ['route_hour_key', 'route_hour_delay_mean']

    # Global mean for unknown categories
    global_mean = train_df['delay_minutes'].mean()
    global_std = train_df['delay_minutes'].std()

    # Merge stats into training data
    train_df = train_df.merge(route_stats, on='route_id', how='left')
    train_df = train_df.merge(stop_stats, on='stop_id', how='left')
    train_df = train_df.merge(hour_stats, on='hour', how='left')
    train_df = train_df.merge(route_hour_stats, on='route_hour_key', how='left')

    # Merge stats into test data
    test_df['route_hour_key'] = test_df['route_id'].astype(str) + '_' + test_df['hour'].astype(str)
    test_df = test_df.merge(route_stats, on='route_id', how='left')
    test_df = test_df.merge(stop_stats, on='stop_id', how='left')
    test_df = test_df.merge(hour_stats, on='hour', how='left')
    test_df = test_df.merge(route_hour_stats, on='route_hour_key', how='left')

    # Fill missing with global stats
    for col in ['route_delay_mean', 'route_delay_std', 'stop_delay_mean', 'stop_delay_std',
                'hour_delay_mean', 'route_hour_delay_mean']:
        train_df[col] = train_df[col].fillna(global_mean if 'mean' in col else global_std)
        test_df[col] = test_df[col].fillna(global_mean if 'mean' in col else global_std)

    # Encode categorical
    le_route = LabelEncoder()
    le_stop = LabelEncoder()
    le_direction = LabelEncoder()

    all_routes = pd.concat([train_df['route_id'], test_df['route_id']]).astype(str).unique()
    all_stops = pd.concat([train_df['stop_id'], test_df['stop_id']]).astype(str).unique()
    all_directions = pd.concat([
        train_df['direction_id'].fillna('Unknown').astype(str),
        test_df['direction_id'].fillna('Unknown').astype(str)
    ]).unique()

    le_route.fit(all_routes)
    le_stop.fit(all_stops)
    le_direction.fit(all_directions)

    train_df['route_encoded'] = le_route.transform(train_df['route_id'].astype(str))
    train_df['stop_encoded'] = le_stop.transform(train_df['stop_id'].astype(str))
    train_df['direction_encoded'] = le_direction.transform(train_df['direction_id'].fillna('Unknown').astype(str))

    test_df['route_encoded'] = le_route.transform(test_df['route_id'].astype(str))
    test_df['stop_encoded'] = le_stop.transform(test_df['stop_id'].astype(str))
    test_df['direction_encoded'] = le_direction.transform(test_df['direction_id'].fillna('Unknown').astype(str))

    # Scheduled headway
    headway_median = train_df['scheduled_headway'].median()
    train_df['scheduled_headway'] = train_df['scheduled_headway'].fillna(headway_median)
    test_df['scheduled_headway'] = test_df['scheduled_headway'].fillna(headway_median)

    # =========================================
    # Feature selection (V2)
    # =========================================
    feature_columns = [
        # Original features
        'is_weekend', 'is_rush_hour',
        'route_encoded', 'stop_encoded', 'direction_encoded', 'scheduled_headway',
        # V2: Cyclical encoding
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        # V2: Historical statistics (from training data only)
        'route_delay_mean', 'route_delay_std',
        'stop_delay_mean', 'stop_delay_std',
        'hour_delay_mean', 'route_hour_delay_mean'
    ]

    X_train = train_df[feature_columns].values
    y_train = train_df['delay_minutes'].values.reshape(-1, 1)

    X_test = test_df[feature_columns].values
    y_test = test_df['delay_minutes'].values.reshape(-1, 1)

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Standardize - fit only on training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    print(f"\nFeature matrix shape: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"\nDelay Statistics (Training, minutes):")
    print(f"  Mean: {y_train.mean():.2f}")
    print(f"  Std: {y_train.std():.2f}")

    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            feature_columns, scaler_X, scaler_y, y_train, y_test)


# =============================================================================
# Models
# =============================================================================

class MLPPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class GRUPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


def train_model(model, train_loader, val_loader, epochs: int = 50, lr: float = 0.001,
                patience: int = 10, model_name: str = "model"):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print("="*60)

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = MODELS_DIR / f"{model_name}_{EXPERIMENT_VERSION}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time
    }, model_path)
    print(f"Model saved to: {model_path}")

    return model, train_losses, val_losses, training_time


def evaluate_model(model, test_loader, scaler_y, model_name: str = "model"):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_orig = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

    mse = mean_squared_error(actuals_orig, predictions_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_orig, predictions_orig)
    r2 = r2_score(actuals_orig, predictions_orig)

    metrics = {
        'model': model_name,
        'experiment': EXPERIMENT_VERSION,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    print(f"\n{model_name} (V2 Temporal) Evaluation Results:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f} minutes")
    print(f"  MAE:  {mae:.4f} minutes")
    print(f"  R²:   {r2:.4f}")

    return metrics, predictions_orig, actuals_orig


def generate_report(all_metrics, results, feature_columns, save_path):
    """Generate experiment report."""
    metrics_df = pd.DataFrame(all_metrics)
    best_idx = metrics_df['RMSE'].idxmin()
    best_model = metrics_df.loc[best_idx, 'model']
    best_rmse = metrics_df.loc[best_idx, 'RMSE']

    report = f"""# Bus Delay Prediction - V2 Lag Features (Temporal Split)

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Author:** Boston Bus Equity Team

---

## 1. Introduction

Bus delay prediction with enhanced features.
**Data Split:** Train on <2025, Test on >=2025 (Temporal Split)

Improvements over V1:
1. Cyclical encoding for temporal features
2. Historical statistics (route/stop/hour mean/std)
3. All statistics computed ONLY on training data (no leakage)

## 2. Features ({len(feature_columns)})

{', '.join(feature_columns)}

## 3. Results

{metrics_df.to_markdown(index=False)}

### Best Model

**{best_model}** achieved the best RMSE of **{best_rmse:.4f} minutes**.

## 4. Training Time

| Model | Training Time (s) |
|-------|-------------------|
"""
    for name, data in results.items():
        report += f"| {name} | {data['training_time']:.2f} |\n"

    report += """

## 5. Visualizations

### Training Curves
![Training Curves](figures/delay_prediction_training_curves_v2_lag_features_temporal.png)

---

*Report generated automatically*
"""

    with open(save_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {save_path}")


def main():
    print("="*60)
    print(f"Bus Delay Prediction - Experiment {EXPERIMENT_VERSION}")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n*** Using TEMPORAL SPLIT: Train <2025, Test >=2025 ***\n")

    # Load data with temporal split
    (X_train, y_train, X_test, y_test,
     feature_columns, scaler_X, scaler_y, y_train_orig, y_test_orig) = load_and_preprocess_data_v2_temporal()

    # Split training into train/val (85/15)
    n_train = len(X_train)
    n_val = int(0.15 * n_train)
    n_train_actual = n_train - n_val

    X_train_final = X_train[:n_train_actual]
    y_train_final = y_train[:n_train_actual]
    X_val = X_train[n_train_actual:]
    y_val = y_train[n_train_actual:]

    print(f"\nFinal Dataset split:")
    print(f"  Train: {len(X_train_final):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test (2025+): {len(X_test):,}")

    # Create data loaders
    batch_size = 256

    train_dataset = TensorDataset(torch.FloatTensor(X_train_final), torch.FloatTensor(y_train_final))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Models
    input_size = X_train.shape[1]
    models = {
        'MLP': MLPPredictor(input_size=input_size, hidden_sizes=[128, 64, 32], dropout=0.2),
        'LSTM': LSTMPredictor(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2),
        'GRU': GRUPredictor(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)
    }

    # Train and evaluate
    results = {}
    all_metrics = []

    for model_name, model in models.items():
        trained_model, train_losses, val_losses, training_time = train_model(
            model, train_loader, val_loader,
            epochs=50, lr=0.001, patience=10,
            model_name=f"delay_predictor_{model_name.lower()}"
        )

        metrics, predictions, actuals = evaluate_model(
            trained_model, test_loader, scaler_y, model_name
        )

        results[model_name] = {
            'model': trained_model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_time': training_time,
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals
        }

        all_metrics.append(metrics)

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + "="*60)
    print("V2 Lag Features Temporal Split Results Summary")
    print("="*60)
    print(metrics_df.to_string(index=False))

    metrics_path = REPORTS_DIR / f"delay_prediction_metrics_{EXPERIMENT_VERSION}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    # Training curves
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (model_name, data) in zip(axes, results.items()):
        ax.plot(data['train_losses'], label='Train Loss', linewidth=2)
        ax.plot(data['val_losses'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{model_name} (V2 Temporal) Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"delay_prediction_training_curves_{EXPERIMENT_VERSION}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Generate report
    generate_report(all_metrics, results, feature_columns, REPORTS_DIR / f"DELAY_PREDICTION_REPORT_{EXPERIMENT_VERSION}.md")

    # Save config
    config = {
        'experiment': EXPERIMENT_VERSION,
        'date': datetime.now().isoformat(),
        'temporal_split': 'Train: <2025, Test: >=2025',
        'features': feature_columns,
        'n_features': len(feature_columns),
        'models': list(models.keys()),
        'batch_size': batch_size,
        'epochs': 50,
        'device': str(DEVICE),
        'train_size': len(X_train_final),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'results': {name: data['metrics'] for name, data in results.items()}
    }

    config_path = MODELS_DIR / f"delay_prediction_config_{EXPERIMENT_VERSION}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*60)
    print("Experiment V2 (Temporal Split) Complete!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    best_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'model']
    best_rmse = metrics_df['RMSE'].min()
    best_r2 = metrics_df['R2'].max()
    print(f"\nBest Model: {best_model}")
    print(f"  RMSE: {best_rmse:.4f} minutes")
    print(f"  R²: {best_r2:.4f}")


if __name__ == "__main__":
    main()
