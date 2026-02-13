"""
Visual backbone modules.
"""

from .siglip2 import SigLIP2Backbone
from .resnet import ResNetBackbone
from .vgg import VGGBackbone
from .vit import ViTBackbone
from .efficientnet import EfficientNetBackbone

__all__ = [
    "SigLIP2Backbone",
    "ResNetBackbone",
    "VGGBackbone",
    "ViTBackbone",
    "EfficientNetBackbone",
]
