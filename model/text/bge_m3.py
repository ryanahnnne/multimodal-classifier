"""
BGE-M3 multilingual text encoder.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..base import TextEncoder


class BGEM3Encoder(TextEncoder):
    """BGE-M3 multilingual dense embedding encoder.

    Includes a learnable 'no_text' embedding for images without any detected text.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "BAAI/bge-m3",
        output_dim: int = 1024,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        use_no_text_embedding: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self._output_dim = output_dim
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_no_text_embedding = use_no_text_embedding

        if not enabled:
            return

        print(f"Loading BGE-M3 text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze text encoder and set to eval mode
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
        # Always keep the frozen encoder in eval mode
        if self.enabled and hasattr(self, 'encoder'):
            self.encoder.eval()
        return self

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode a batch of text strings using BGE-M3 dense embeddings."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        outputs = self.encoder(**tokens)

        # BGE-M3 uses CLS token (first token) for dense embedding
        embeddings = outputs.last_hidden_state[:, 0]

        # Normalize embeddings (recommended for BGE-M3)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)

        return embeddings  # (B, output_dim)

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
        batch_size = len(combined_texts)

        # Identify samples with no text at all
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
        non_empty_embeddings = self._encode_texts(non_empty_texts, device)

        # Build full embedding tensor (use index_copy_ for deterministic compatibility)
        embeddings = torch.zeros(batch_size, self._output_dim, device=device)
        non_empty_idx = (~empty_mask).nonzero(as_tuple=True)[0]
        embeddings.index_copy_(0, non_empty_idx, non_empty_embeddings)

        # Replace empty text embeddings with learnable no_text_embedding
        if self.use_no_text_embedding and empty_mask.any():
            empty_idx = empty_mask.nonzero(as_tuple=True)[0]
            embeddings.index_copy_(0, empty_idx, self.no_text_embedding.unsqueeze(0).expand(empty_idx.shape[0], -1))

        return embeddings
