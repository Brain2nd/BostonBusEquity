"""
SNNDelayModel v7: NeuronSpark-based Delay Prediction Model (Parallel Scan)

Copied from NeuronSpark/model.py with minimal changes:
  - embed_tokens (Embedding) -> input_proj (Linear): continuous features instead of discrete tokens
  - Output layer: vocab_size logits -> 1D regression output

Architecture:
  features -> input_proj -> sigmoid -> K-bit binary encoding -> K spike frames
  -> L SNNDecoderLayers (SNNBlock + SNNFFN, parallel scan)
  -> K spike frames binary decoding -> [0,1]^D
  -> decode_proj -> RMSNorm -> output_head -> delay prediction
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.checkpoint import checkpoint

# Add NeuronSpark to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
NEURONSPARK_PATH = PROJECT_ROOT / "NeuronSpark"
sys.path.insert(0, str(NEURONSPARK_PATH))

from atomic_ops import SNNDecoderLayer


@dataclass
class SNNModelOutput:
    """Model output container."""
    loss: Optional[torch.Tensor] = None
    prediction: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    """RMSNorm normalization layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SNNDelayModel(nn.Module):
    """
    NeuronSpark v7 based delay prediction model (parallel scan).

    Args:
        input_size: Input feature dimension (default 14)
        D: Visible dimension
        N: State expansion factor
        K: SNN timesteps per sample (v7 default 16)
        num_layers: Number of SNN decoder layers
        D_ff: FFN intermediate dimension
        v_th_min: Dynamic threshold lower bound
    """

    def __init__(
        self,
        input_size: int = 14,
        D: int = 1024,
        N: int = 8,
        K: int = 16,
        num_layers: int = 20,
        D_ff: int = 3072,
        v_th_min: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff

        # Input projection (replaces Embedding)
        self.input_proj = nn.Linear(input_size, D)
        self.norm = RMSNorm(D)

        # Trainable encode/decode projections
        self.encode_proj = nn.Linear(D, D)
        self.decode_proj = nn.Linear(D, D)

        # SNN Decoder Layers
        self.layers = nn.ModuleList([
            SNNDecoderLayer(
                D=D, N=N, D_ff=D_ff, v_th_min=v_th_min,
                block_output_v_threshold=0.3 if i == 0 else 0.05,
                ffn_output_v_threshold=0.5 if i == 0 else 0.15,
                num_layers=num_layers,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])

        # Output head (regression)
        self.output_head = nn.Linear(D, 1)

        # K-bit binary weights
        self.register_buffer(
            'bit_weights',
            torch.tensor([2.0 ** (-(k + 1)) for k in range(K)]),
        )
        # K-bit encoding scale factors (for parallel encoding)
        self.register_buffer(
            'bit_scales',
            torch.tensor([2.0 ** (k + 1) for k in range(K)]),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize all trainable weights."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.encode_proj.weight)
        nn.init.zeros_(self.encode_proj.bias)
        nn.init.xavier_uniform_(self.decode_proj.weight)
        nn.init.zeros_(self.decode_proj.bias)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features to spike frame sequence.

        Args:
            x: (batch, input_size)

        Returns:
            spike_seq: (K, batch, D) - K spike frames
        """
        batch = x.shape[0]

        # 1. Input projection: (batch, input_size) -> (batch, D)
        emb = self.input_proj(x)

        # 2. Encode: (batch, D) -> (batch, D)
        h = torch.sigmoid(self.encode_proj(emb))

        # 3. K-bit parallel binary encoding -> (batch, K, D)
        #    Math: h in [0,1] binary expansion bit_k = floor(2^{k+1}*h) mod 2
        #    STE: forward = bit_hard (binary {0,1}), backward = dh/dh = 1 (identity)
        scaled = h.unsqueeze(1) * self.bit_scales.view(1, self.K, 1)  # (batch, K, D)
        bit_hard = torch.floor(scaled) % 2  # {0.0, 1.0}
        h_k = h.unsqueeze(1)  # (batch, 1, D), broadcasts to (batch, K, D)
        bits = h_k + (bit_hard - h_k).detach()  # STE: forward=bit_hard, backward=dh/dh=1

        # 4. (batch, K, D) -> (K, batch, D)
        spike_seq = bits.permute(1, 0, 2)

        return spike_seq

    def _decode_features(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """
        Decode spike frame sequence to continuous representation.

        Args:
            spike_seq: (K, batch, D)

        Returns:
            decoded: (batch, D)
        """
        # (K, batch, D) -> (batch, K, D)
        spike_seq = spike_seq.permute(1, 0, 2)

        # Binary weighted sum: sum(spike * 2^{-k})
        decoded = torch.einsum('bkd,k->bd', spike_seq, self.bit_weights)

        return decoded  # (batch, D)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        Forward pass (v7: parallel scan).

        Args:
            x: (batch, input_size) feature input
            targets: (batch, 1) target delay values

        Returns:
            SNNModelOutput:
                If targets provided: .loss = MSE loss
                If targets is None: .prediction = (batch, 1)
        """
        batch = x.shape[0]

        # Reset all layer neuron states
        for layer_module in self.layers:
            functional.reset_net(layer_module)

        # 1. Encode features
        spike_seq = self._encode_features(x)  # (K, batch, D)

        # 2. Layer-wise parallel processing (gradient checkpoint: recompute per layer, save memory)
        def _layer_forward(layer_mod, s):
            functional.reset_net(layer_mod)
            return layer_mod.forward_parallel(s)

        for layer_module in self.layers:
            spike_seq = checkpoint(
                _layer_forward, layer_module, spike_seq,
                use_reentrant=False,
            )

        # 3. Decode
        decoded = self._decode_features(spike_seq)  # (batch, D)

        # 4. Projection -> RMSNorm -> output head
        h = self.decode_proj(decoded)  # (batch, D)
        h = self.norm(h)               # (batch, D)
        prediction = self.output_head(h)  # (batch, 1)

        if targets is not None:
            loss = F.mse_loss(prediction, targets)
            return SNNModelOutput(loss=loss, prediction=prediction)

        return SNNModelOutput(prediction=prediction)
