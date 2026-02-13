"""
SigLIP2 visual backbone.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel
from omegaconf import DictConfig

from ..base import VisionBackbone
from ..lora import apply_lora_to_model


class SigLIP2Backbone(VisionBackbone):
    """SigLIP2 vision transformer backbone.

    Supports:
        - google/siglip2-so400m-patch14-384 (1152-dim)
        - google/siglip2-base-patch16-384 (768-dim)
        - google/siglip2-base-patch16-256 (768-dim)
    """

    def __init__(
        self,
        name: str = "google/siglip2-so400m-patch14-384",
        image_size: int = 384,
        hidden_dim: int = 1152,
        freeze: bool = True,
        finetune_layers: int = 0,
        lora: DictConfig = None,
        output_feature_type: str = "pooled",
    ):
        super().__init__()
        self._name = name
        self._image_size = image_size
        self._output_dim = hidden_dim
        self._output_feature_type = output_feature_type

        print(f"Loading SigLIP2 backbone: {name}")
        self.model = AutoModel.from_pretrained(name)

        for param in self.model.parameters():
            param.requires_grad = False

        if not freeze:
            encoder_layers = self.model.vision_model.encoder.layers
            total = len(encoder_layers)
            if finetune_layers <= 0:
                start = 0
            else:
                start = max(0, total - finetune_layers)
            for i in range(start, total):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
            print(f"  Unfroze layers {start}-{total - 1} ({total - start} layers)")

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
        """SigLIP2 uses different normalization than ImageNet."""
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features from the vision backbone.

        Args:
            pixel_values: (B, 3, H, W) normalized images

        Returns:
            (B, N, D) patch token features or (B, D) pooled features depending on output_feature_type
        """
        outputs = self.model.vision_model(pixel_values=pixel_values)
        if self._output_feature_type == "patch":
            return outputs.last_hidden_state  # (B, N, D)
        else:
            return outputs.pooler_output  # (B, D)
