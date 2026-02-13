"""
Multimodal fusion modules for combining visual and text features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Simple concatenation fusion with optional projection.

    When projection_dim > 0, both modalities are projected to the same
    dimension before concatenation (current baseline behavior).
    When projection_dim = 0, raw features are L2-normalized and concatenated
    directly, preserving full information.

    Output dim:
        - projection_dim > 0: projection_dim * 2
        - projection_dim = 0: visual_dim + text_dim
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        projection_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if projection_dim > 0:
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
            )
            self.output_dim = projection_dim * 2
        else:
            self.visual_proj = None
            self.text_proj = None
            self.output_dim = visual_dim + text_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if self.visual_proj is not None:
            visual = self.visual_proj(visual)
            text = self.text_proj(text)

        visual = F.normalize(visual, p=2, dim=-1, eps=1e-8)
        text = F.normalize(text, p=2, dim=-1, eps=1e-8)

        return torch.cat([visual, text], dim=-1)


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (GMU) for visual-text feature fusion.

    Reference: Arevalo et al., "Gated Multimodal Units for Information Fusion" (2017)

    Each modality is independently transformed to a common dimension,
    then a learned sigmoid gate determines the per-sample, per-dimension
    contribution of each modality. This allows the model to dynamically
    suppress uninformative text features (e.g., no_text_embedding) and
    preserve the strong visual signal when text is unreliable.

    Architecture:
        h_v = tanh(W_v * visual)
        h_t = tanh(W_t * text)
        gate = sigmoid(W_g * [visual, text])
        output = gate * h_v + (1 - gate) * h_t
        output = LayerNorm(output)
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.visual_transform = nn.Linear(visual_dim, output_dim)
        self.text_transform = nn.Linear(text_dim, output_dim)
        self.gate = nn.Linear(visual_dim + text_dim, output_dim)

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize gate bias slightly positive so gate starts
        # favoring visual features (the stronger modality)
        nn.init.constant_(self.gate.bias, 0.5)

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        h_v = torch.tanh(self.visual_transform(visual))
        h_t = torch.tanh(self.text_transform(text))

        gate = torch.sigmoid(self.gate(torch.cat([visual, text], dim=-1)))

        fused = gate * h_v + (1 - gate) * h_t
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


def build_fusion(
    fusion_type: str,
    visual_dim: int,
    text_dim: int,
    **kwargs,
) -> nn.Module:
    """Build fusion module from config.

    Args:
        fusion_type: "concat" or "gated"
        visual_dim: Visual feature dimension
        text_dim: Text feature dimension
        **kwargs: Additional arguments passed to fusion module

    Returns:
        Fusion module with .output_dim attribute
    """
    if fusion_type == "concat":
        return ConcatFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            projection_dim=kwargs.get("projection_dim", 0),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif fusion_type == "gated":
        return GatedFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            output_dim=kwargs.get("projection_dim", 512),
            dropout=kwargs.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
