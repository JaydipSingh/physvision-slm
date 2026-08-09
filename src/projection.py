"""
Vision-Language Projection Module
=================================
Maps vision encoder features into the language model's embedding space.

Two modes:
  1. Full projection (for SigLIP 1152-dim → LM d_model)
  2. Compressed projection (reduce num_patches via pooling)

This is the ONLY trainable component during Stage 1 alignment.
"""
import torch
import torch.nn as nn
import math


class VisionProjection(nn.Module):
    """
    Projects vision features into language model embedding space.

    Maps: (batch, num_vision_tokens, vision_dim)
       → (batch, num_output_tokens, text_dim)

    Architecture: Linear → GELU → Linear (+ optional token compression)
    """

    def __init__(
        self,
        vision_dim: int = 1152,     # SigLIP hidden dim
        text_dim: int = 384,         # Language model d_model
        num_vision_tokens: int = 729, # SigLIP patch count (27×27 for 384px/14)
        num_output_tokens: int = 64,  # Compressed to this many tokens
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_vision_tokens = num_vision_tokens
        self.num_output_tokens = num_output_tokens

        # Token compression (reduce sequence length via learned pooling)
        if num_vision_tokens != num_output_tokens:
            self.compress = nn.Sequential(
                # Transpose → pool along token dimension → transpose back
                nn.Linear(num_vision_tokens, num_output_tokens),
            )
            self.use_compression = True
        else:
            self.use_compression = False

        # Dimension projection (2-layer MLP)
        hidden_dim = max(vision_dim, text_dim) * 2
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, num_vision_tokens, vision_dim)

        Returns:
            (batch, num_output_tokens, text_dim)
        """
        x = vision_features  # (B, V, D_v)

        # Token compression (reduce V → T)
        if self.use_compression:
            # (B, V, D_v) → (B, D_v, V) → Linear → (B, D_v, T) → (B, T, D_v)
            x = x.transpose(1, 2)  # (B, D_v, V)
            x = self.compress[0](x)  # (B, D_v, T)
            x = x.transpose(1, 2)  # (B, T, D_v)

        # Dimension projection
        x = self.proj(x)  # (B, T, D_t)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SimpleProjection(nn.Module):
    """
    Simpler projection: Average pool vision tokens → single vector → expand.
    Uses less memory, suitable for very small LMs (35M).

    Maps: (batch, num_vision_tokens, vision_dim) → (batch, num_output_tokens, text_dim)
    Via: avg_pool → Linear → repeat
    """

    def __init__(
        self,
        vision_dim: int = 256,      # VisionEncoderLite output
        text_dim: int = 384,        # Language model d_model
        num_output_tokens: int = 16, # Few tokens for small LM
    ):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.text_dim = text_dim

        self.proj = nn.Sequential(
            nn.Linear(vision_dim, text_dim * num_output_tokens),
            nn.GELU(),
            nn.LayerNorm(text_dim * num_output_tokens),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, num_patches, vision_dim)

        Returns:
            (batch, num_output_tokens, text_dim)
        """
        # Global average pooling over patches
        x = vision_features.mean(dim=1)  # (B, D_v)

        # Project to full output size
        x = self.proj(x)  # (B, T * D_t)

        # Reshape to token sequence
        batch = x.shape[0]
        x = x.view(batch, self.num_output_tokens, self.text_dim)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
