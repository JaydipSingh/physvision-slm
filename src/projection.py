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
    DEPRECATED — global-average-pool projection.

    This collapses ALL spatial patches into a single vector via mean(dim=1)
    before projecting, which DESTROYS the spatial information (tumor location,
    count, size) that image questions depend on. A model trained with this
    projection cannot distinguish options that require reading the image, so it
    falls back to a text-only prior. Kept only for reference / ablation.

    Use SpatialProjection instead.
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
        # Global average pooling over patches (destroys spatial info!)
        x = vision_features.mean(dim=1)  # (B, D_v)
        x = self.proj(x)  # (B, T * D_t)
        batch = x.shape[0]
        x = x.view(batch, self.num_output_tokens, self.text_dim)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SpatialProjection(nn.Module):
    """
    Spatial-preserving projection for the lite config.

    Keeps EVERY patch as its own vision token so spatial information (where the
    tumor is, how many, how big) survives into the language model. Adds a
    learned positional embedding per patch so the LM can tell patches apart.

    Maps: (batch, num_patches, vision_dim) -> (batch, num_patches, text_dim)

    Architecture: per-patch MLP (Linear -> GELU -> Linear -> LayerNorm)
    + learned patch position embedding. No pooling / no averaging.
    """

    def __init__(
        self,
        vision_dim: int = 256,       # VisionEncoderLite output channels
        text_dim: int = 384,         # Language model d_model
        num_patches: int = 64,       # CNN produces 8x8 = 64 patches
        hidden_mult: int = 2,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_patches = num_patches
        # Exposed so the model knows how many vision tokens this emits.
        self.num_output_tokens = num_patches

        hidden_dim = text_dim * hidden_mult
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
            nn.LayerNorm(text_dim),
        )
        # Learned position embedding per patch (spatial identity).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, text_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, num_patches, vision_dim)

        Returns:
            (batch, num_patches, text_dim)  — one token per patch, spatial info kept
        """
        B, V, _ = vision_features.shape
        x = self.proj(vision_features)          # (B, V, text_dim)
        if V == self.num_patches:
            x = x + self.pos_embed              # add spatial position identity
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
