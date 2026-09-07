"""
PhysVision Multimodal Model
============================
Combines vision encoder + projection + language model into a single
multimodal architecture for visual question answering.

Supports two configurations:
  A) VisionEncoderLite + SimpleProjection + TinyLMv3 (35M total, ultra-light)
  B) SigLIP + VisionProjection + Qwen2-0.5B (500M total, production)

Usage:
  model = PhysVisionModel.from_config("lite")  # Config A
  model = PhysVisionModel.from_config("full")  # Config B

  answer_logits = model(pixel_values, input_ids)
"""
import sys
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent paths for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision_encoder import VisionEncoder, VisionEncoderLite
from src.projection import VisionProjection, SimpleProjection, SpatialProjection


class PhysVisionModel(nn.Module):
    """
    Full multimodal model: Image → Vision Encoder → Projection → Language Model → Answer

    The model concatenates vision tokens (from image) with text tokens (from question)
    and feeds the combined sequence to the language model for next-token prediction.

    Sequence format:
        [VISION_TOKENS] [TEXT_TOKENS (question)] [ANSWER_TOKENS]
        ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
        from projection  from tokenizer            predicted by LM
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        projection: nn.Module,
        language_model: nn.Module,
        vocab_size: int,
        text_dim: int,
        num_vision_tokens: int,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projection = projection
        self.language_model = language_model
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.num_vision_tokens = num_vision_tokens
        self.max_seq_len = max_seq_len

        # Vision token type embedding (to distinguish vision from text tokens)
        self.vision_token_type = nn.Parameter(torch.zeros(1, 1, text_dim))

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal visual QA.

        Args:
            pixel_values: (batch, 3, H, W) image tensor
            input_ids: (batch, seq_len) text token IDs (question + answer)
            labels: (batch, seq_len) target token IDs for loss computation
                    (set to -100 for positions that shouldn't contribute to loss)

        Returns:
            dict with:
                - "logits": (batch, total_seq_len, vocab_size)
                - "loss": scalar (if labels provided)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. Extract vision features.
        #    If the encoder is frozen (e.g. SigLIP), run under no_grad for
        #    efficiency. If it is trainable (VisionEncoderLite CNN), it MUST
        #    run with gradients enabled or it never learns to see.
        encoder_trainable = any(
            p.requires_grad for p in self.vision_encoder.parameters())
        if encoder_trainable:
            vision_features = self.vision_encoder(pixel_values)
        else:
            with torch.no_grad():
                vision_features = self.vision_encoder(pixel_values)

        # 2. Project to text embedding space
        vision_tokens = self.projection(vision_features)  # (B, num_vision_tokens, text_dim)
        vision_tokens = vision_tokens + self.vision_token_type

        # 3. Get text embeddings from language model
        if hasattr(self.language_model, 'embed'):
            text_embeddings = self.language_model.embed(input_ids)  # (B, seq_len, text_dim)
        elif hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'embed_tokens'):
            text_embeddings = self.language_model.model.embed_tokens(input_ids)
        else:
            raise ValueError("Cannot find embedding layer in language model")

        # 4. Concatenate: [vision_tokens | text_tokens]
        combined = torch.cat([vision_tokens, text_embeddings], dim=1)  # (B, V+T, d)

        # 5. Truncate to max sequence length
        if combined.shape[1] > self.max_seq_len:
            combined = combined[:, :self.max_seq_len, :]

        # 6. Pass through language model transformer layers
        if hasattr(self.language_model, 'layers'):
            # TinyLMv3: custom transformer
            h = self.language_model.drop(combined)
            for layer in self.language_model.layers:
                h = layer(h, self.language_model.rope_cos, self.language_model.rope_sin)
            h = self.language_model.norm_f(h)
            logits = self.language_model.head(h)
        elif hasattr(self.language_model, 'transformer'):
            # TinyLMv2: nn.TransformerEncoder
            h = combined
            causal_mask = self.language_model._generate_causal_mask(
                combined.shape[1], device)
            h = self.language_model.transformer(h, mask=causal_mask)
            h = self.language_model.ln_f(h)
            logits = self.language_model.head(h)
        else:
            # HuggingFace model (Qwen2, etc.)
            outputs = self.language_model(inputs_embeds=combined)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # 7. Compute loss (only on text portion, after vision tokens)
        result = {"logits": logits}

        if labels is not None:
            # Shift logits and labels for causal LM loss
            # Vision token positions don't have labels → pad labels with -100
            vision_pad = torch.full(
                (batch_size, self.num_vision_tokens), -100,
                dtype=torch.long, device=device
            )
            full_labels = torch.cat([vision_pad, labels], dim=1)

            # Truncate to match logits length
            if full_labels.shape[1] > logits.shape[1]:
                full_labels = full_labels[:, :logits.shape[1]]
            elif full_labels.shape[1] < logits.shape[1]:
                pad = torch.full(
                    (batch_size, logits.shape[1] - full_labels.shape[1]), -100,
                    dtype=torch.long, device=device
                )
                full_labels = torch.cat([full_labels, pad], dim=1)

            # Causal LM loss: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = full_labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate_answer(
        self,
        pixel_values: torch.Tensor,
        question_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """
        Generate answer tokens given an image and question.

        Args:
            pixel_values: (1, 3, H, W) single image
            question_ids: (1, seq_len) question token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature

        Returns:
            (1, generated_len) tensor of generated token IDs
        """
        device = question_ids.device

        # Get vision + text embeddings
        vision_features = self.vision_encoder(pixel_values)
        vision_tokens = self.projection(vision_features) + self.vision_token_type

        if hasattr(self.language_model, 'embed'):
            text_emb = self.language_model.embed(question_ids)
        else:
            text_emb = self.language_model.model.embed_tokens(question_ids)

        combined = torch.cat([vision_tokens, text_emb], dim=1)

        generated_ids = []
        for _ in range(max_new_tokens):
            if combined.shape[1] > self.max_seq_len:
                combined = combined[:, -self.max_seq_len:, :]

            # Forward through LM
            if hasattr(self.language_model, 'layers'):
                h = combined
                for layer in self.language_model.layers:
                    h = layer(h, self.language_model.rope_cos, self.language_model.rope_sin)
                h = self.language_model.norm_f(h)
                logits = self.language_model.head(h)
            else:
                outputs = self.language_model(inputs_embeds=combined)
                logits = outputs.logits

            # Sample next token
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Stop at EOS (token 2)
            if next_token.item() in [0, 2]:
                break

            generated_ids.append(next_token.item())

            # Append next token embedding
            if hasattr(self.language_model, 'embed'):
                next_emb = self.language_model.embed(next_token.unsqueeze(0))
            else:
                next_emb = self.language_model.model.embed_tokens(next_token.unsqueeze(0))
            combined = torch.cat([combined, next_emb], dim=1)

        return torch.tensor([generated_ids], dtype=torch.long, device=device)

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config_name: str = "lite", device: str = "cpu", **kwargs):
        """
        Factory method to create model from named configuration.

        Configs:
          "lite" — VisionEncoderLite + SimpleProjection + TinyLMv3 (~40M total)
          "full" — SigLIP + VisionProjection + Qwen2-0.5B (~900M total)
        """
        if config_name == "lite":
            return cls._build_lite(device, **kwargs)
        elif config_name == "full":
            return cls._build_full(device, **kwargs)
        else:
            raise ValueError(f"Unknown config: {config_name}. Use 'lite' or 'full'.")

    @classmethod
    def _build_lite(cls, device: str, lm_checkpoint: Optional[str] = None, **kwargs):
        """Build the lightweight config: ~40M parameters total."""
        # Vision encoder (trainable CNN, ~5M params) — emits 64 spatial patches
        vision = VisionEncoderLite(output_dim=256, num_output_tokens=64)

        # Projection: spatial-preserving (one token per patch, keeps location).
        # NOTE: SimpleProjection averaged all patches into one vector and made
        # the model blind to image content; SpatialProjection fixes that.
        proj = SpatialProjection(vision_dim=256, text_dim=384, num_patches=64)

        # Language model (TinyLMv3). Prefer the standalone, dependency-free
        # architecture module; the pretrained checkpoint loads into it because
        # parameter names are identical to paper 1.
        try:
            from src.tinylm_v3 import TinyLMv3
        except ImportError:
            # Fallback: import from paper 1 training script (requires its deps)
            slm_v2_path = ROOT.parent / "slm_v2"
            sys.path.insert(0, str(slm_v2_path))
            try:
                from stream_train_v3 import TinyLMv3  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Cannot import TinyLMv3 from src.tinylm_v3 or slm_v2.") from e

        # 64 vision tokens + up to 128 text tokens = 192; use 208 for headroom.
        num_vision_tokens = 64
        max_seq_len = 208
        lm = TinyLMv3(vocab_size=32000, d_model=384, n_layers=6, n_heads=6,
                      max_seq_len=max_seq_len)

        if lm_checkpoint:
            ckpt = torch.load(lm_checkpoint, map_location="cpu", weights_only=False)
            # RoPE buffers are position-derived and deterministic; skip if the
            # saved max_seq_len differs. All other weights load normally.
            state = ckpt["model"]
            state = {k: v for k, v in state.items()
                     if not k.startswith("rope_")}
            lm.load_state_dict(state, strict=False)

        model = cls(
            vision_encoder=vision,
            projection=proj,
            language_model=lm,
            vocab_size=32000,
            text_dim=384,
            num_vision_tokens=num_vision_tokens,
            max_seq_len=max_seq_len,
        ).to(device)

        return model

    @classmethod
    def _build_full(cls, device: str, **kwargs):
        """Build the full config: ~900M parameters total (mostly frozen)."""
        # Vision encoder (frozen SigLIP, ~400M params)
        vision = VisionEncoder(
            model_name="google/siglip-so400m-patch14-384",
            device=device,
        )

        # Projection (learned, ~5M params)
        proj = VisionProjection(
            vision_dim=vision.hidden_dim,
            text_dim=896,  # Qwen2-0.5B hidden size
            num_vision_tokens=vision.num_patches,
            num_output_tokens=64,
        )

        # Language model (Qwen2-0.5B, LoRA applied separately)
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

        model = cls(
            vision_encoder=vision,
            projection=proj,
            language_model=lm,
            vocab_size=151936,  # Qwen2 vocab size
            text_dim=896,
            num_vision_tokens=64,
            max_seq_len=512,
        ).to(device)

        return model
