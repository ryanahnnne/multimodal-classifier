"""
Model package for Ad Image Binary Classifier.
Provides modular components: visual backbones, text encoders, pooling, and heads.
"""

from .classifier import MultimodalClassifier, build_classifier
from .pooling import MeanPooling, CLSPooling, AttentionPooling, build_pooling
from .head import ClassificationHead
from .lora import LoRALinear, apply_lora_to_model

__all__ = [
    "MultimodalClassifier",
    "build_classifier",
    "MeanPooling",
    "CLSPooling",
    "AttentionPooling",
    "build_pooling",
    "ClassificationHead",
    "LoRALinear",
    "apply_lora_to_model",
]
