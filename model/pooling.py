"""
Pooling modules for aggregating patch/token features.
"""

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """Simple mean pooling over patch tokens."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) patch token features

        Returns:
            (B, D) pooled features
        """
        return x.mean(dim=1)


class CLSPooling(nn.Module):
    """Use CLS token (first token) as representation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token features (CLS is at index 0)

        Returns:
            (B, D) CLS token features
        """
        return x[:, 0]


class AttentionPooling(nn.Module):
    """Learnable attention pooling with a single query.

    A learnable query attends to all patch tokens to produce
    a single aggregated representation.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) patch token features

        Returns:
            (B, D) attention-pooled features
        """
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        out, _ = self.attn(query, x, x)  # (B, 1, D)
        out = self.norm(out.squeeze(1))  # (B, D)
        return out


def build_pooling(pooling_type: str, hidden_dim: int, num_heads: int = 8) -> nn.Module:
    """Factory function for pooling modules.

    Args:
        pooling_type: One of "mean", "cls", "attention"
        hidden_dim: Feature dimension (required for attention pooling)
        num_heads: Number of attention heads (for attention pooling)

    Returns:
        Pooling module
    """
    if pooling_type == "mean":
        return MeanPooling()
    elif pooling_type == "cls":
        return CLSPooling()
    elif pooling_type == "attention":
        return AttentionPooling(hidden_dim, num_heads)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")
