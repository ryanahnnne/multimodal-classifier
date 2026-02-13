"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear layers.

    Wraps an existing linear layer with trainable low-rank matrices
    while keeping the original weights frozen.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_linear(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: int = 8,
    target_layers: int = 6,
    dropout: float = 0.05,
) -> nn.Module:
    """Apply LoRA to the last N transformer blocks of the vision encoder.

    Targets query, key, value projection layers in attention.

    Args:
        model: Vision model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        target_layers: Number of last transformer blocks to apply LoRA
        dropout: Dropout rate for LoRA

    Returns:
        Model with LoRA applied
    """
    # Find all transformer layers
    # SigLIP2 structure: model.vision_model.encoder.layers[i].self_attn.{q_proj, k_proj, v_proj}
    encoder_layers = None

    # Try different possible structures
    if hasattr(model, "vision_model"):
        if hasattr(model.vision_model, "encoder"):
            encoder_layers = model.vision_model.encoder.layers
    elif hasattr(model, "encoder"):
        if hasattr(model.encoder, "layers"):
            encoder_layers = model.encoder.layers

    if encoder_layers is None:
        print(
            "Warning: Could not find transformer layers for LoRA. "
            "Printing model structure for debugging:"
        )
        for name, _ in model.named_modules():
            print(f"  {name}")
        raise ValueError("Cannot find encoder layers. Check model structure above.")

    total_layers = len(encoder_layers)
    start_layer = max(0, total_layers - target_layers)

    lora_param_count = 0

    for i in range(start_layer, total_layers):
        layer = encoder_layers[i]
        attn = layer.self_attn

        # Replace q, k, v projections with LoRA versions
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            if hasattr(attn, proj_name):
                original = getattr(attn, proj_name)
                lora_layer = LoRALinear(original, rank, alpha, dropout)
                setattr(attn, proj_name, lora_layer)
                lora_param_count += rank * (original.in_features + original.out_features)

    print(f"Applied LoRA (rank={rank}) to layers {start_layer}-{total_layers - 1}")
    print(f"  LoRA trainable parameters: {lora_param_count:,}")

    return model
