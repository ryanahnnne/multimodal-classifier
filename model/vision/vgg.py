"""
VGG visual backbone.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from ..base import VisionBackbone


class VGGBackbone(VisionBackbone):
    """VGG visual backbone.

    Supports: vgg11, vgg13, vgg16, vgg19 (with batch normalization variants)
    """

    MODELS = {
        "vgg11": (models.vgg11, models.VGG11_Weights.IMAGENET1K_V1, 512),
        "vgg11_bn": (models.vgg11_bn, models.VGG11_BN_Weights.IMAGENET1K_V1, 512),
        "vgg13": (models.vgg13, models.VGG13_Weights.IMAGENET1K_V1, 512),
        "vgg13_bn": (models.vgg13_bn, models.VGG13_BN_Weights.IMAGENET1K_V1, 512),
        "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, 512),
        "vgg16_bn": (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1, 512),
        "vgg19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1, 512),
        "vgg19_bn": (models.vgg19_bn, models.VGG19_BN_Weights.IMAGENET1K_V1, 512),
    }

    def __init__(
        self,
        name: str = "vgg16",
        pretrained: bool = True,
        image_size: int = 224,
        hidden_dim: int = 512,
        freeze: bool = True,
        finetune_layers: int = 0,
        lora=None,  # Not supported for VGG
    ):
        super().__init__()

        if name not in self.MODELS:
            raise ValueError(
                f"Unknown VGG model: {name}. " f"Available: {list(self.MODELS.keys())}"
            )

        self._name = name
        self._image_size = image_size
        model_fn, weights, dim = self.MODELS[name]
        self._output_dim = dim

        print(f"Loading VGG backbone: {name}")
        base = model_fn(weights=weights if pretrained else None)

        # Use only the features part (convolutional layers)
        self.features = base.features

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            print("  Backbone frozen.")

        # Partial unfreezing: unfreeze top N convolutional blocks
        if freeze and finetune_layers != 0:
            children = list(self.features.children())
            total = len(children)
            if finetune_layers == -1:
                start = 0
            else:
                start = max(0, total - finetune_layers)
            for i in range(start, total):
                for param in children[i].parameters():
                    param.requires_grad = True
            print(f"  Unfroze layer modules {start}-{total - 1} ({total - start} modules)")

        if lora is not None and lora.enabled:
            print("  Warning: LoRA is not supported for VGG. Ignoring.")

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
        # Output: (B, C, H', W')
        x = self.features(pixel_values)
        B, C, H, W = x.shape
        # Reshape to (B, N, C) where N = H' * W'
        return x.view(B, C, H * W).permute(0, 2, 1)
