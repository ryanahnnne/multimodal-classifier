"""
EfficientNet visual backbone using timm.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

from ..base import VisionBackbone


class EfficientNetBackbone(VisionBackbone):
    """EfficientNet backbone using timm.

    Supports various EfficientNet models:
        - efficientnet_b0 to efficientnet_b7
        - efficientnetv2_s, efficientnetv2_m, efficientnetv2_l
    """

    # Common EfficientNet models and their hidden dimensions
    HIDDEN_DIMS = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "efficientnet_b5": 2048,
        "efficientnet_b6": 2304,
        "efficientnet_b7": 2560,
        "efficientnetv2_s": 1280,
        "efficientnetv2_m": 1280,
        "efficientnetv2_l": 1280,
    }

    def __init__(
        self,
        name: str = "efficientnet_b0",
        pretrained: bool = True,
        image_size: int = 224,
        hidden_dim: int = 1280,
        freeze: bool = True,
        finetune_layers: int = 0,
        lora=None,  # Not supported for EfficientNet
    ):
        super().__init__()

        if timm is None:
            raise ImportError(
                "timm is required for EfficientNet backbone. Install with: pip install timm"
            )

        self._name = name
        self._image_size = image_size
        self._output_dim = hidden_dim

        print(f"Loading EfficientNet backbone: {name}")
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            features_only=False,
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("  Backbone frozen.")

        # Partial unfreezing: unfreeze top N blocks
        if freeze and finetune_layers != 0:
            blocks = list(self.model.blocks)
            total = len(blocks)
            if finetune_layers == -1:
                start = 0
            else:
                start = max(0, total - finetune_layers)
            for i in range(start, total):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            # Also unfreeze final conv + norm
            if hasattr(self.model, "conv_head"):
                for param in self.model.conv_head.parameters():
                    param.requires_grad = True
            if hasattr(self.model, "bn2"):
                for param in self.model.bn2.parameters():
                    param.requires_grad = True
            print(f"  Unfroze blocks {start}-{total - 1} ({total - start} blocks)")

        if lora is not None and lora.enabled:
            print("  Warning: LoRA is not supported for EfficientNet. Ignoring.")

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
            (B, N, D) spatial features reshaped for pooling
        """
        # forward_features returns (B, C, H', W')
        x = self.model.forward_features(pixel_values)

        # Handle case where output is already (B, D)
        if x.dim() == 2:
            return x.unsqueeze(1)  # (B, 1, D)

        # Reshape from (B, C, H, W) to (B, N, C)
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1)
