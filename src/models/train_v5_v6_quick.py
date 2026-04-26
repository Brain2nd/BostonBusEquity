"""
Quick training script for V5 NeuronSpark SNN and V6 Transformer.

The original V5 / V6 training scripts in this directory train on the full
3.76M-sample dataset (~5-8 hours each) and crucially never call torch.save,
so no shipped checkpoints exist for those models. This script reuses the
exact same architectures from train_delay_predictor_v5_neuronspark.py but:
  1. Trains on a small sample (~50 K) so it finishes in minutes
  2. Saves the resulting state_dict to models/ so the dashboard registry
     can load and demo them

The resulting models will not match the headline R^2 numbers (0.9897 SNN,
0.9942 Transformer) — those require full-data training. They DO produce
real predictions and let users compare V5 / V6 architectures alongside
V1 / V2 / V3 in the dashboard model picker.

Usage:
    python -m src.models.train_v5_v6_quick --sample-size 50000
    python -m src.models.train_v5_v6_quick --sample-size 50000 --skip-snn
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.train_delay_predictor_v5_neuronspark import (
    SNNDelayRegressor,
    TransformerRegressor,
    load_and_prepare_data,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def train_one_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_name: str,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
) -> dict:
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    Xs_train = scaler_X.fit_transform(X_train).astype(np.float32)
    Xs_test = scaler_X.transform(X_test).astype(np.float32)
    ys_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    ys_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)

    n_val = max(1, int(0.15 * len(Xs_train)))
    X_tr, y_tr = Xs_train[:-n_val], ys_train[:-n_val]
    X_val, y_val = Xs_train[-n_val:], ys_train[-n_val:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xs_test), torch.from_numpy(ys_test)),
        batch_size=batch_size,
    )

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)

    best_loss, best_state, wait = float("inf"), None, 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: train={epoch_loss:.6f}  val={val_loss:.6f}")

    model.load_state_dict(best_state)
    elapsed = time.time() - start

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds.extend(model(X_batch).cpu().numpy())
            actuals.extend(y_batch.numpy())

    preds_inv = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    actuals_inv = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(actuals_inv, preds_inv)))
    mae = float(mean_absolute_error(actuals_inv, preds_inv))
    r2 = float(r2_score(actuals_inv, preds_inv))

    print(f"\nResults: RMSE={rmse:.4f}  MAE={mae:.4f}  R^2={r2:.4f}  ({elapsed:.1f}s)")

    return {
        "state_dict": model.state_dict(),
        "scaler_X_mean": scaler_X.mean_.astype(np.float32),
        "scaler_X_scale": scaler_X.scale_.astype(np.float32),
        "scaler_y_mean": scaler_y.mean_.astype(np.float32),
        "scaler_y_scale": scaler_y.scale_.astype(np.float32),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_params": n_params,
        "input_size": int(X_train.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50000,
                        help="Number of training samples (smaller = faster, less accurate)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--skip-snn", action="store_true", help="Skip V5 NeuronSpark training")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip V6 Transformer training")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading data with sample_size={args.sample_size}...")
    X_train, y_train, X_test, y_test, _ = load_and_prepare_data(sample_size=args.sample_size)
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    if not args.skip_transformer:
        transformer = TransformerRegressor(
            input_size=X_train.shape[1],
            d_model=128, nhead=8, num_layers=4,
            dim_feedforward=512, dropout=0.1, seq_len=8,
        )
        result = train_one_model(transformer, X_train, y_train, X_test, y_test,
                                 model_name="V6 Transformer", epochs=args.epochs)
        bundle = {
            "model_state_dict": result["state_dict"],
            "model_kind": "transformer",
            "feature_version": "v6",
            "input_size": result["input_size"],
            "test_rmse": result["rmse"],
            "test_mae": result["mae"],
            "test_r2": result["r2"],
            "scaler_X": {"mean": result["scaler_X_mean"], "scale": result["scaler_X_scale"]},
            "scaler_y": {"mean": result["scaler_y_mean"], "scale": result["scaler_y_scale"]},
            "trained_at": datetime.now().isoformat(),
            "trained_sample_size": args.sample_size,
            "note": "Quick-train demo checkpoint; full-data run reaches R^2=0.9942.",
        }
        out = MODELS_DIR / "delay_transformer_v6_quick.pt"
        torch.save(bundle, out)
        print(f"Saved {out}")

    if not args.skip_snn:
        snn = SNNDelayRegressor(
            input_size=X_train.shape[1],
            D=128, N=8, K=8, num_blocks=2,
        )
        result = train_one_model(snn, X_train, y_train, X_test, y_test,
                                 model_name="V5 NeuronSpark SNN", epochs=args.epochs)
        bundle = {
            "model_state_dict": result["state_dict"],
            "model_kind": "neuronspark_snn",
            "feature_version": "v5",
            "input_size": result["input_size"],
            "test_rmse": result["rmse"],
            "test_mae": result["mae"],
            "test_r2": result["r2"],
            "scaler_X": {"mean": result["scaler_X_mean"], "scale": result["scaler_X_scale"]},
            "scaler_y": {"mean": result["scaler_y_mean"], "scale": result["scaler_y_scale"]},
            "trained_at": datetime.now().isoformat(),
            "trained_sample_size": args.sample_size,
            "snn_config": {"D": 128, "N": 8, "K": 8, "num_blocks": 2},
            "note": "Quick-train demo checkpoint; full-data run reaches R^2=0.9897.",
        }
        out = MODELS_DIR / "delay_neuronspark_v5_quick.pt"
        torch.save(bundle, out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
