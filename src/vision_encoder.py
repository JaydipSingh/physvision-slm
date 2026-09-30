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
        from transformers import AutoModel

        # Load ONLY the image processor, not the full AutoProcessor. The full
        # processor also loads the model's TEXT tokenizer (SigLIP's needs
        # protobuf/sentencepiece), which we never use — we only feed images.
        self.processor = None
        try:
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception:
            # Image preprocessing is not required for our pipeline (the dataset
            # already yields normalized tensors that we resize in forward()),
            # so a missing processor is non-fatal.
            self.processor = None

        # Load in float32 explicitly. On MPS, mixing dtypes in a matmul triggers
        # a hard Metal assertion, so the vision tower must match the rest of the
        # model (also float32).
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32).to(device)
        self.model = self.model.float()
        self.model.eval()

        # Determine the model's expected input resolution from the processor so
        # the dummy forward (and real inputs) match. SigLIP-SO400M/14 uses 384.
        self.input_size = self._detect_input_size()

        # Detect hidden dimension and patch count from a dummy forward pass.
        dummy = torch.randn(1, 3, self.input_size, self.input_size).to(device)
        with torch.no_grad():
            features = self._extract(dummy)

        self._hidden_dim = features.shape[-1]
        self._num_patches = features.shape[1]

    def _detect_input_size(self) -> int:
        """Best-effort detection of the encoder's expected square input size."""
        size = 384
        if self.processor is None:
            # Fall back to the model config's image_size if available.
            try:
                vcfg = getattr(self.model.config, "vision_config", self.model.config)
                size = getattr(vcfg, "image_size", size)
            except Exception:
                pass
            return int(size)
        try:
            ip = getattr(self.processor, "image_processor", self.processor)
            s = getattr(ip, "size", None)
            if isinstance(s, dict):
                size = s.get("height") or s.get("width") or s.get("shortest_edge") or size
            elif isinstance(s, int):
                size = s
        except Exception:
            pass
        return int(size)

    def _extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision tower and return (B, num_patches, hidden_dim)."""
        if hasattr(self.model, 'vision_model'):
            out = self.model.vision_model(pixel_values)
            features = out.last_hidden_state
        elif hasattr(self.model, 'get_image_features'):
            features = self.model.get_image_features(pixel_values)
            if features.dim() == 2:
                features = features.unsqueeze(1)
        else:
            out = self.model(pixel_values)
            features = getattr(out, "last_hidden_state", out)
        return features

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
        Extract features from images.

        Args:
            pixel_values: (batch, 3, H, W) tensor. If H/W differ from the
                encoder's expected input size, the batch is resized (the dataset
                renders 128x128 while SigLIP expects e.g. 384x384).

        Returns:
            (batch, num_patches, hidden_dim) tensor
        """
        target = getattr(self, "input_size", 384)
        if pixel_values.shape[-1] != target or pixel_values.shape[-2] != target:
            pixel_values = nn.functional.interpolate(
                pixel_values, size=(target, target),
                mode="bilinear", align_corners=False)

        features = self._extract(pixel_values)

        # SigLIP produces patch tokens only (no CLS). Guard the CLS-strip so we
        # do not accidentally drop a real patch when num_patches was detected
        # without a CLS token.
        if not self.use_cls_token and features.shape[1] == (self._num_patches + 1):
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
