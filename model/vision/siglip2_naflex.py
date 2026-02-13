"""
SigLIP2 NaFlex visual backbone.
Supports variable aspect ratio via NaFlex architecture.
Uses fixed-size inputs for compatibility with existing data pipeline,
but passes spatial_shapes for proper NaFlex processing.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel
from omegaconf import DictConfig

from ..base import VisionBackbone
from ..lora import apply_lora_to_model


class SigLIP2NaFlexBackbone(VisionBackbone):
    """SigLIP2 NaFlex vision transformer backbone.

    NaFlex supports native aspect ratios, but here we use fixed-size inputs
    for compatibility with the existing data pipeline (torchvision transforms).
    spatial_shapes is computed from input dimensions at forward time.

    Supports:
        - google/siglip2-so400m-patch16-naflex (1152-dim)
        - google/siglip2-base-patch16-naflex (768-dim)
    """

    def __init__(
        self,
        name: str = "google/siglip2-so400m-patch16-naflex",
        image_size: int = 384,
        hidden_dim: int = 1152,
        patch_size: int = 16,
        freeze: bool = True,
        finetune_layers: int = 0,
        lora: DictConfig = None,
        output_feature_type: str = "pooled",
    ):
        super().__init__()
        self._name = name
        self._image_size = image_size
        self._output_dim = hidden_dim
        self._patch_size = patch_size
        self._output_feature_type = output_feature_type

        print(f"Loading SigLIP2 NaFlex backbone: {name}")
        self.model = AutoModel.from_pretrained(name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("  Backbone frozen.")

        # Partial unfreezing: unfreeze top N transformer layers
        if freeze and finetune_layers != 0:
            encoder_layers = self.model.vision_model.encoder.layers
            total = len(encoder_layers)
            if finetune_layers == -1:
                start = 0
            else:
                start = max(0, total - finetune_layers)
            for i in range(start, total):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
            print(f"  Unfroze layers {start}-{total - 1} ({total - start} layers)")

        # Apply LoRA if configured
        if lora is not None and lora.enabled:
            self.model = apply_lora_to_model(
                self.model,
                rank=lora.rank,
                alpha=lora.alpha,
                target_layers=lora.target_layers,
                dropout=lora.dropout,
            )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def image_size(self) -> int:
        return self._image_size

    def get_normalization(self) -> Tuple[List[float], List[float]]:
        """SigLIP2 uses 0.5 normalization."""
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features from the NaFlex vision backbone.

        Args:
            pixel_values: (B, 3, H, W) normalized images

        Returns:
            (B, N, D) patch token features or (B, D) pooled features
        """
        B, _, H, W = pixel_values.shape

        # Compute spatial_shapes from input dimensions
        h_patches = H // self._patch_size
        w_patches = W // self._patch_size
        spatial_shapes = torch.tensor(
            [[h_patches, w_patches]], device=pixel_values.device
        ).expand(B, -1)

        outputs = self.model.vision_model(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
        )

        if self._output_feature_type == "patch":
            return outputs.last_hidden_state  # (B, N, D)
        else:
            return outputs.pooler_output  # (B, D)
