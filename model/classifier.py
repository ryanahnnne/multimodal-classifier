"""
Main classifier that composes vision encoder, text encoder, pooling, fusion, and head.
"""

from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate

from .pooling import build_pooling
from .head import ClassificationHead
from .fusion import build_fusion


class MultimodalClassifier(nn.Module):
    """Multimodal binary classifier.

    Architecture (multimodal):
        Image -> Vision Encoder -> Pooling -> ┐
                                                ├-> Fusion -> Classification Head
        Text  -> Text Encoder    -> ───────────┘

    Architecture (visual-only):
        Image -> Vision Encoder -> Pooling -> [Projection] -> L2 Norm -> Classification Head

    Architecture (text-only):
        Text -> Text Encoder -> [Projection] -> L2 Norm -> Classification Head

    Supports:
        - Multiple vision encoder (SigLIP2, ResNet, VGG, ViT, EfficientNet)
        - Optional text encoder (BGE-M3, LaBSE, E5-Large)
        - Multiple pooling strategies (mean, cls, attention)
        - Multiple fusion strategies (gated, concat) for multimodal features
        - Optional vision/text feature projection
        - Text-only mode (vision_encoder=None) for modality ablation
    """

    def __init__(
        self,
        vision_encoder: Optional[nn.Module],
        text_encoder: Optional[nn.Module],
        pooling: Optional[nn.Module],
        classifier: nn.Module,
        fusion: Optional[nn.Module] = None,
        visual_projection: Optional[nn.Module] = None,
        text_projection: Optional[nn.Module] = None,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.pooling = pooling
        self.classifier = classifier
        self.fusion = fusion
        self.visual_projection = visual_projection
        self.text_projection = text_projection
        self.normalize_features = normalize_features

        self.use_text = text_encoder is not None

        # Print parameter summary
        self._print_param_summary()

    def train(self, mode: bool = True):
        """Override to keep frozen modules in eval mode.

        Prevents frozen BatchNorm from updating running statistics
        and frozen Dropout from being active during training.

        Uses recurse=True so that parent modules whose entire subtree
        is frozen (including parameter-less children like Dropout) are
        correctly switched to eval mode.
        """
        super().train(mode)
        if mode:
            for module in self.modules():
                if module is self:
                    continue
                params = list(module.parameters(recurse=True))
                if params and all(not p.requires_grad for p in params):
                    module.eval()
        return self

    def _print_param_summary(self):
        """Print trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"\nModel Parameter Summary:")
        print(f"  Total:     {total:>12,}")
        print(f"  Trainable: {trainable:>12,}")
        print(f"  Frozen:    {frozen:>12,}")
        print(f"  Trainable %: {100 * trainable / total:.2f}%")

    def encode_text(
        self, texts: Dict[str, List[str]], device: torch.device
    ) -> Optional[torch.Tensor]:
        """Encode OCR texts to tensor.

        This method should be called before forward() to allow DataParallel
        to scatter the tensor across GPUs.

        Args:
            texts: Dict of language -> list of strings
            device: Target device

        Returns:
            (B, text_dim) text features or None if text encoder is disabled
        """
        if not self.use_text or self.text_encoder is None:
            return None
        return self.text_encoder(texts, device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized images (ignored in text-only mode)
            text_features: Optional (B, text_dim) pre-encoded text features.
                           Must be encoded by encode_text() before calling forward.

        Returns:
            (B,) logits (raw, before sigmoid)
        """
        # Text-only mode (no vision encoder)
        if self.vision_encoder is None:
            if text_features is None:
                raise ValueError("text_features required in text-only mode")
            if text_features.dtype != torch.float32:
                text_features = text_features.to(torch.float32)
            if self.text_projection is not None:
                text_features = self.text_projection(text_features)
            if self.normalize_features:
                text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-8)
            return self.classifier(text_features)

        # Vision features
        vis_features = self.vision_encoder(pixel_values)
        if vis_features.dim() == 3:
            # Patch features from encoder (B, N, D)
            pooled = self.pooling(vis_features)  # (B, D)
        else:
            # Pooled features from encoder (B, D)
            pooled = vis_features

        # Multimodal fusion or visual-only path
        if self.fusion is not None and self.use_text and text_features is not None:
            # Ensure text_features has same dtype (for AMP compatibility)
            if text_features.dtype != pooled.dtype:
                text_features = text_features.to(pooled.dtype)

            # Fusion module handles projection, gating, and combination
            features = self.fusion(pooled, text_features)
        else:
            # Visual-only: optional projection + normalize
            if self.visual_projection is not None:
                pooled = self.visual_projection(pooled)
            if self.normalize_features:
                pooled = F.normalize(pooled, p=2, dim=-1, eps=1e-8)
            features = pooled

        # Classification
        logits = self.classifier(features)  # (B,)

        return logits


def _build_projection(input_dim: int, output_dim: int, dropout: float = 0.1) -> nn.Module:
    """Build a projection layer with linear -> GELU -> dropout -> linear."""
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(output_dim, output_dim),
    )


def build_classifier(cfg: DictConfig):
    """Build classifier from Hydra config.

    Modality modes (train.modality):
        - "multimodal": vision_encoder + text_encoder + fusion
        - "vision_only": vision_encoder only (no text encoder loaded)
        - "text_only": text_encoder only (no vision_encoder loaded)

    Args:
        cfg: Full Hydra configuration

    Returns:
        AdImageClassifier instance
    """
    modality = cfg.train.get("modality", "multimodal")

    # Auto-detect modality from encoder configs if not explicitly set
    has_vis = cfg.vision_encoder.get("_target_") is not None
    has_text = cfg.text_encoder.get("_target_") is not None
    if modality == "multimodal":
        if not has_text:
            modality = "vision_only"
        elif not has_vis:
            modality = "text_only"

    projection_cfg = cfg.train.get("projection", {})
    projection_dim = projection_cfg.get("dim", 0)
    projection_dropout = projection_cfg.get("dropout", 0.1)

    vision_encoder = None
    text_encoder = None
    pooling = None
    fusion = None
    visual_projection = None
    text_projection = None

    # --- Visual encoder ---
    if modality in ("multimodal", "vision_only"):
        is_siglip = "siglip" in cfg.vision_encoder.get("name", "").lower()
        backbone_kwargs = dict(
            freeze=cfg.train.freeze_backbone,
            finetune_layers=cfg.train.get("finetune_layers", 0),
            lora=cfg.train.lora,
        )
        if is_siglip and cfg.train.get("output_feature_type"):
            backbone_kwargs["output_feature_type"] = cfg.train.output_feature_type
        vision_encoder = instantiate(cfg.vision_encoder, **backbone_kwargs)

        pooling = build_pooling(
            pooling_type=cfg.pooling.type,
            hidden_dim=cfg.vision_encoder.hidden_dim,
            num_heads=cfg.pooling.get("num_heads", 8),
        )

    # --- Text encoder ---
    if modality in ("multimodal", "text_only"):
        text_encoder = instantiate(cfg.text_encoder)

    # --- Head input dim ---
    if modality == "multimodal":
        # Fusion
        fusion_cfg = cfg.get("fusion", {})
        fusion_type = fusion_cfg.get("type", "gated")
        fusion_kwargs = {k: v for k, v in fusion_cfg.items() if k != "type"}
        fusion = build_fusion(
            fusion_type=fusion_type,
            visual_dim=cfg.vision_encoder.hidden_dim,
            text_dim=cfg.text_encoder.output_dim,
            **fusion_kwargs,
        )
        head_input_dim = fusion.output_dim
        print(f"\nModality: multimodal | Fusion: {fusion_type} (output_dim={head_input_dim})")

    elif modality == "vision_only":
        visual_dim = cfg.vision_encoder.hidden_dim
        if projection_dim > 0:
            visual_projection = _build_projection(visual_dim, projection_dim, projection_dropout)
            head_input_dim = projection_dim
        else:
            head_input_dim = visual_dim
        print(f"\nModality: vision_only (dim={head_input_dim})")

    elif modality == "text_only":
        text_dim = cfg.text_encoder.output_dim
        if projection_dim > 0:
            text_projection = _build_projection(text_dim, projection_dim, projection_dropout)
            head_input_dim = projection_dim
        else:
            head_input_dim = text_dim
        print(f"\nModality: text_only (dim={head_input_dim})")

    else:
        raise ValueError(f"Unknown modality: {modality}. Use 'multimodal', 'vision_only', or 'text_only'.")

    # Classification head
    classifier = ClassificationHead(
        input_dim=head_input_dim,
        hidden_dim=cfg.train.get("head_hidden_dim", 512),
        dropout=cfg.train.get("head_dropout", 0.1),
    )

    normalize_features = projection_cfg.get("normalize", False)

    return MultimodalClassifier(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        pooling=pooling,
        classifier=classifier,
        fusion=fusion,
        visual_projection=visual_projection,
        text_projection=text_projection,
        normalize_features=normalize_features,
    )
