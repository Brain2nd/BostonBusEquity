#!/usr/bin/env python3
"""
Train NeuronSpark v7 Delay Prediction Model

Uses same data preprocessing as v3_fixed (wavelet, FFT features)
Directly uses SNNDelayModel copied from NeuronSpark/model.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from snn_delay_model import SNNDelayModel
from train_delay_predictor_v3_fixed import load_and_preprocess_data_v3_fixed

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

EXPERIMENT_VERSION = "v7_neuronspark"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, name="Model"):
    """Train model with early stopping."""
    print(f"\n{'=' * 60}")
    print(f"Training {name}")
    print("=" * 60)

    model = model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(-1)
            optimizer.zero_grad()
            output = model(X_batch, y_batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += output.loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(-1)
                output = model(X_batch, y_batch)
                val_loss += output.loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.1f}s")

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return model, param_count, training_time


def evaluate(model, loader, scaler_y, name="model"):
    """Evaluate model (aligned with v3_fixed format)."""
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            preds.extend(output.prediction.cpu().numpy())
            actuals.extend(y_batch.numpy())

    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    print(f"\n{name} Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    return {'model': name, 'experiment': EXPERIMENT_VERSION, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def main():
    print("=" * 60)
    print("NeuronSpark v7 Delay Prediction")
    print("=" * 60)
    print(f"Start: {datetime.now()}")
    print(f"Device: {DEVICE}")

    # Load data with v3_fixed preprocessing (wavelet, FFT features)
    # 3.76M samples (full training set) for fair comparison
    (X_train, y_train, X_test, y_test,
     feature_columns, scaler_X, scaler_y) = load_and_preprocess_data_v3_fixed(
        sample_size=None,  # Use full dataset
        use_fft=True,
        use_wavelet=True,
        use_cache=True
    )

    print(f"\nFeatures: {len(feature_columns)}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Train/Val split (same as v3_fixed)
    n = len(X_train)
    n_val = int(0.15 * n)

    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]

    print(f"\nSplit: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test)}")

    # Loaders
    batch_size = 2048
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size
    )

    input_size = X_train.shape[1]

    # Model config aligned with Transformer (~1.6M parameters)
    # Transformer: 1,595,649 parameters
    # NeuronSpark: D=128, N=8, K=8, num_layers=2 ~ 1.6M parameters
    model = SNNDelayModel(
        input_size=input_size,
        D=128,
        N=8,
        K=8,
        num_layers=2,
        D_ff=384,
        v_th_min=0.1,
    )

    model_name = "NeuronSpark-v7"
    trained_model, params, train_time = train_model(
        model, train_loader, val_loader,
        epochs=50, lr=0.001, patience=10, name=model_name
    )

    # Evaluate
    metrics = evaluate(trained_model, test_loader, scaler_y, model_name)
    metrics['Params'] = params
    metrics['Time(s)'] = train_time

    # Save model (aligned with v3_fixed format)
    model_path = MODELS_DIR / f"delay_{model_name.lower().replace('-', '_')}_{EXPERIMENT_VERSION}.pt"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Save results (aligned with v3_fixed format)
    results_df = pd.DataFrame([metrics])
    print("\n" + "=" * 60)
    print(f"Final Results ({EXPERIMENT_VERSION})")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_path = REPORTS_DIR / f"delay_prediction_metrics_{EXPERIMENT_VERSION}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")

    print(f"\nBest: {metrics['model']} - RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    print(f"\nEnd: {datetime.now()}")


if __name__ == "__main__":
    main()
