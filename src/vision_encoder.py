"""
Vision Encoder Module — SigLIP Feature Extraction
==================================================
Wraps a pre-trained SigLIP (or CLIP) vision encoder for extracting
image features. The encoder is kept FROZEN during training.

Supports:
  - SigLIP-SO400M/14 (default, best for scientific images)
  - CLIP ViT-B/16 (fallback, smaller)
  - Any HuggingFace vision model

Usage:
  encoder = VisionEncoder(model_name="google/siglip-so400m-patch14-384")
  features = encoder(images)  # (batch, num_patches, hidden_dim)
"""
import torch
import torch.nn as nn
from typing import Optional


class VisionEncoder(nn.Module):
    """
    Frozen vision encoder that extracts patch-level features from images.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str = "cpu",
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self._hidden_dim = None
        self._num_patches = None

        # Load model
        self._load_model(model_name, device)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def _load_model(self, model_name: str, device: str):
        """Load the vision model from HuggingFace."""
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Detect hidden dimension from a dummy forward pass
        dummy = torch.randn(1, 3, 384, 384).to(device)
        with torch.no_grad():
            if hasattr(self.model, 'vision_model'):
                out = self.model.vision_model(dummy)
                features = out.last_hidden_state
            elif hasattr(self.model, 'get_image_features'):
                features = self.model.get_image_features(dummy)
                if features.dim() == 2:
                    features = features.unsqueeze(1)
            else:
                out = self.model(dummy)
                features = out.last_hidden_state

        self._hidden_dim = features.shape[-1]
        self._num_patches = features.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    def preprocess(self, images) -> torch.Tensor:
        """Preprocess PIL images or numpy arrays for the model."""
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract features from preprocessed images.

        Args:
            pixel_values: (batch, 3, H, W) tensor

        Returns:
            (batch, num_patches, hidden_dim) tensor
        """
        if hasattr(self.model, 'vision_model'):
            out = self.model.vision_model(pixel_values)
            features = out.last_hidden_state
        elif hasattr(self.model, 'get_image_features'):
            features = self.model.get_image_features(pixel_values)
            if features.dim() == 2:
                features = features.unsqueeze(1)
        else:
            out = self.model(pixel_values)
            features = out.last_hidden_state

        if not self.use_cls_token and features.shape[1] > 1:
            # Remove CLS token if present (keep only patch tokens)
            features = features[:, 1:, :]

        return features


class VisionEncoderLite(nn.Module):
    """
    Lightweight vision encoder using a simple CNN backbone.
    For use when SigLIP is too large for constrained hardware.
    Trainable (not frozen) — learns to extract features from PET/sinogram images.

    ~5M parameters, suitable for 18GB M3 Pro alongside TinyLMv3.
    """

    def __init__(self, output_dim: int = 384, num_output_tokens: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.num_output_tokens = num_output_tokens

        # Simple CNN backbone (ResNet-like)
        self.features = nn.Sequential(
            # 128x128 → 64x64
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 64x64 → 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 32x32 → 16x16
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 16x16 → 8x8
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 8x8 = 64 spatial positions × 256 channels
        self.proj = nn.Linear(256, output_dim)

    @property
    def hidden_dim(self) -> int:
        return self.output_dim

    @property
    def num_patches(self) -> int:
        return self.num_output_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, 128, 128) tensor

        Returns:
            (batch, 64, output_dim) tensor
        """
        # Resize to 128x128 if needed
        if pixel_values.shape[-1] != 128:
            pixel_values = nn.functional.interpolate(
                pixel_values, size=(128, 128), mode='bilinear', align_corners=False)

        x = self.features(pixel_values)  # (batch, 256, 8, 8)
        batch = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (batch, 64, 256)
        x = self.proj(x)  # (batch, 64, output_dim)
        return x
