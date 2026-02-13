"""
Deep Learning Models for MBTA Bus Delay Prediction
===================================================

Implements multiple architectures for time series forecasting:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer
- Temporal Convolutional Network (TCN)

Author: Boston Bus Equity Team
Date: February 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LSTMPredictor(nn.Module):
    """
    LSTM-based delay predictor.

    Architecture:
        Input -> LSTM layers -> FC layers -> Output
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Predictions [batch, output_size]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]

        # FC layers
        out = self.fc(h_n)

        return out


class GRUPredictor(nn.Module):
    """
    GRU-based delay predictor.

    Simpler than LSTM, often similar performance.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based delay predictor.

    Uses self-attention to capture long-range dependencies.
    """

    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        output_size: int = 1,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            [batch, output_size]
        """
        # Project to d_model dimensions
        x = self.input_proj(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Use last time step or mean pooling
        x = x[:, -1, :]  # Last time step

        # Output
        out = self.fc(x)

        return out


class TCNBlock(nn.Module):
    """Temporal Convolutional Block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )

        self.chomp1 = padding
        self.chomp2 = padding

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[:, :, :-self.chomp1] if self.chomp1 > 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.chomp2] if self.chomp2 > 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for delay prediction.

    Uses dilated causal convolutions for efficient sequence modeling.
    """

    def __init__(
        self,
        input_size: int = 6,
        num_channels: list = [32, 64, 64],
        kernel_size: int = 3,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(TCNBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            [batch, output_size]
        """
        # TCN expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        # TCN forward
        x = self.tcn(x)

        # Use last time step
        x = x[:, :, -1]

        # Output
        out = self.fc(x)

        return out


class EnsemblePredictor(nn.Module):
    """
    Ensemble of multiple models.

    Combines LSTM, Transformer, and TCN predictions.
    """

    def __init__(
        self,
        input_size: int = 6,
        output_size: int = 1,
        hidden_size: int = 64
    ):
        super().__init__()

        self.lstm = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=output_size
        )

        self.transformer = TransformerPredictor(
            input_size=input_size,
            d_model=hidden_size,
            nhead=4,
            num_layers=2,
            output_size=output_size
        )

        self.tcn = TCNPredictor(
            input_size=input_size,
            num_channels=[32, 64],
            output_size=output_size
        )

        # Learnable weights for ensemble
        self.weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get predictions from each model
        lstm_out = self.lstm(x)
        trans_out = self.transformer(x)
        tcn_out = self.tcn(x)

        # Weighted average
        weights = F.softmax(self.weights, dim=0)
        out = weights[0] * lstm_out + weights[1] * trans_out + weights[2] * tcn_out

        return out


def get_model(
    model_name: str,
    input_size: int = 6,
    output_size: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_name: One of 'lstm', 'gru', 'transformer', 'tcn', 'ensemble'
        input_size: Number of input features
        output_size: Number of output predictions
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model
    """
    models = {
        'lstm': LSTMPredictor,
        'gru': GRUPredictor,
        'transformer': TransformerPredictor,
        'tcn': TCNPredictor,
        'ensemble': EnsemblePredictor
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name](input_size=input_size, output_size=output_size, **kwargs)


# Test models
if __name__ == "__main__":
    print("="*60)
    print("Testing Deep Learning Models")
    print("="*60)

    # Test input
    batch_size = 32
    seq_len = 24
    input_size = 6

    x = torch.randn(batch_size, seq_len, input_size)

    models_to_test = ['lstm', 'gru', 'transformer', 'tcn', 'ensemble']

    for model_name in models_to_test:
        print(f"\n{model_name.upper()}:")
        model = get_model(model_name, input_size=input_size, output_size=1)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Forward pass
        out = model(x)
        print(f"  Output shape: {out.shape}")

    print("\n" + "="*60)
    print("All models working!")
    print("="*60)
