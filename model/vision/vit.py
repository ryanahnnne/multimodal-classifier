"""
Vision Transformer (ViT) visual backbone using timm.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

from ..base import VisionBackbone


class ViTBackbone(VisionBackbone):
    """Vision Transformer backbone using timm.

    Supports various ViT models from timm:
        - vit_base_patch16_224
        - vit_base_patch16_384
        - vit_large_patch16_224
        - vit_base_patch32_224
        - etc.
    """

    # Common ViT models and their hidden dimensions
    HIDDEN_DIMS = {
        "vit_tiny_patch16_224": 192,
        "vit_small_patch16_224": 384,
        "vit_base_patch16_224": 768,
        "vit_base_patch16_384": 768,
        "vit_base_patch32_224": 768,
        "vit_large_patch16_224": 1024,
        "vit_large_patch16_384": 1024,
        "vit_huge_patch14_224": 1280,
    }

    def __init__(
        self,
        name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        image_size: int = 224,
        hidden_dim: int = 768,
        freeze: bool = True,
        finetune_layers: int = 0,
        lora=None,  # Not supported for timm ViT
    ):
        super().__init__()

        if timm is None:
            raise ImportError(
                "timm is required for ViT backbone. Install with: pip install timm"
            )

        self._name = name
        self._image_size = image_size
        self._output_dim = hidden_dim

        print(f"Loading ViT backbone: {name}")
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("  Backbone frozen.")

        # Partial unfreezing: unfreeze top N transformer blocks
        if freeze and finetune_layers != 0:
            blocks = self.model.blocks
            total = len(blocks)
            if finetune_layers == -1:
                start = 0
            else:
                start = max(0, total - finetune_layers)
            for i in range(start, total):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            # Also unfreeze norm layer if unfreezing any blocks
            if hasattr(self.model, "norm"):
                for param in self.model.norm.parameters():
                    param.requires_grad = True
            print(f"  Unfroze blocks {start}-{total - 1} ({total - start} blocks)")

        if lora is not None and lora.enabled:
            print("  Warning: LoRA is not supported for timm ViT. Ignoring.")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def image_size(self) -> int:
        return self._image_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            pixel_values: (B, 3, H, W) input images

        Returns:
            (B, N, D) patch token features (including CLS token at index 0)
        """
        # timm ViT forward_features returns (B, N+1, D) including CLS token
        # We use forward_features to get all tokens
        x = self.model.forward_features(pixel_values)

        # Some timm models return (B, D) after pooling, check shape
        if x.dim() == 2:
            # Already pooled, reshape to (B, 1, D)
            x = x.unsqueeze(1)

        return x
