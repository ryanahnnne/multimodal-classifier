"""
Classification head modules.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP classification head for binary classification.

    Architecture: input_dim -> hidden_dim -> 1
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) features

        Returns:
            (B,) logits (raw, before sigmoid)
        """
        return self.net(x).squeeze(-1)
