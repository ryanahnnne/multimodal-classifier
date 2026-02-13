"""
ResNet visual backbone.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from ..base import VisionBackbone


class ResNetBackbone(VisionBackbone):
    """ResNet visual backbone.

    Supports: resnet18, resnet34, resnet50, resnet101, resnet152
    """

    MODELS = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
        "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2, 2048),
    }

    # ResNet layer groups: [conv1+bn1, layer1, layer2, layer3, layer4]
    LAYER_GROUPS = ["0", "1", "2", "3", "4", "5", "6", "7"]

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        image_size: int = 224,
        hidden_dim: int = 2048,
        freeze: bool = True,
        finetune_layers: int = 0,
        lora=None,  # Not supported for ResNet
    ):
        super().__init__()

        if name not in self.MODELS:
            raise ValueError(
                f"Unknown ResNet model: {name}. "
                f"Available: {list(self.MODELS.keys())}"
            )

        self._name = name
        self._image_size = image_size
        model_fn, weights, dim = self.MODELS[name]
        self._output_dim = dim

        print(f"Loading ResNet backbone: {name}")
        base = model_fn(weights=weights if pretrained else None)

        # Remove FC layer, keep feature extractor up to avgpool
        # We want spatial features, so remove avgpool and fc
        self.features = nn.Sequential(*list(base.children())[:-2])

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            print("  Backbone frozen.")

        # Partial unfreezing: unfreeze top N layer groups
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
            print(f"  Unfroze layer groups {start}-{total - 1} ({total - start} groups)")

        if lora is not None and lora.enabled:
            print("  Warning: LoRA is not supported for ResNet. Ignoring.")

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
