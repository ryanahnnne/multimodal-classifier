"""
Abstract base classes for model components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class VisionBackbone(ABC, nn.Module):
    """Abstract base class for vision feature extractors."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        pass

    @property
    @abstractmethod
    def image_size(self) -> int:
        """Return the expected input image size."""
        pass

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision features.

        Args:
            pixel_values: (B, 3, H, W) input images

        Returns:
            (B, N, D) patch/token features for pooling
        """
        pass

    def get_normalization(self) -> Tuple[List[float], List[float]]:
        """Return (mean, std) normalization values for this backbone.

        Default: ImageNet normalization.
        Override in subclasses for different normalization.
        """
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class TextEncoder(ABC, nn.Module):
    """Abstract base class for text feature extractors."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        pass

    @abstractmethod
    def forward(
        self, texts: Dict[str, List[str]], device: torch.device
    ) -> torch.Tensor:
        """Encode text to features.

        Args:
            texts: Dict of language -> list of strings
            device: Target device

        Returns:
            (B, D) text features
        """
        pass
