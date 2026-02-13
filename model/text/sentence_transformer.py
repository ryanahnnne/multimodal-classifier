"""
Generic SentenceTransformer text encoder.
Supports LaBSE, Multilingual-E5-Large, and other sentence-transformers models.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from ..base import TextEncoder


class SentenceTransformerEncoder(TextEncoder):
    """Generic sentence-transformers text encoder.

    Works with any model supported by the sentence-transformers library:
    - sentence-transformers/LaBSE (768-dim)
    - intfloat/multilingual-e5-large (1024-dim)
    """


    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "sentence-transformers/LaBSE",
        output_dim: int = 768,
        normalize_embeddings: bool = True,
        use_no_text_embedding: bool = True,
        query_prefix: str = "",
    ):
        super().__init__()
        self.enabled = enabled
        self._output_dim = output_dim
        self.normalize_embeddings = normalize_embeddings
        self.use_no_text_embedding = use_no_text_embedding
        self.query_prefix = query_prefix

        if not enabled:
            return

        print(f"Loading SentenceTransformer text encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Learnable embedding for no-text images
        if use_no_text_embedding:
            self.no_text_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def train(self, mode: bool = True):
        """Override train to keep encoder in eval mode."""
        super().train(mode)
        if self.enabled and hasattr(self, "encoder"):
            self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        texts: Dict[str, List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            texts: Dict with "text_info" key mapping to list of text strings (length B).
                   Empty string = no text detected.
            device: Target device

        Returns:
            (B, output_dim) text features
        """
        if not self.enabled:
            raise RuntimeError("Text encoder is disabled")

        combined_texts = [t.strip() for t in texts["text_info"]]
        if self.query_prefix:
            combined_texts = [self.query_prefix + t if t else t for t in combined_texts]
        batch_size = len(combined_texts)

        # Identify samples with no text
        empty_mask = torch.tensor(
            [len(t) == 0 for t in combined_texts],
            dtype=torch.bool,
            device=device,
        )

        # Handle case where all samples have no text
        if empty_mask.all():
            if self.use_no_text_embedding:
                return self.no_text_embedding.unsqueeze(0).expand(batch_size, -1)
            else:
                return torch.zeros(batch_size, self._output_dim, device=device)

        # Encode non-empty texts
        non_empty_texts = [t for t in combined_texts if t]
        non_empty_embeddings = self.encoder.encode(
            non_empty_texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        )

        # Build full embedding tensor (use index_copy_ for deterministic compatibility)
        embeddings = torch.zeros(
            batch_size, self._output_dim, device=device, dtype=non_empty_embeddings.dtype
        )
        non_empty_idx = (~empty_mask).nonzero(as_tuple=True)[0]
        embeddings.index_copy_(0, non_empty_idx, non_empty_embeddings.to(device))

        # Replace empty text embeddings with learnable no_text_embedding
        if self.use_no_text_embedding and empty_mask.any():
            empty_idx = empty_mask.nonzero(as_tuple=True)[0]
            embeddings.index_copy_(0, empty_idx, self.no_text_embedding.unsqueeze(0).expand(empty_idx.shape[0], -1))

        return embeddings
