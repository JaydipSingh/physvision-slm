"""
TinyLMv3 Architecture (standalone)
==================================
Self-contained copy of the TinyLMv3 decoder-only transformer from paper 1
(slm_v2/stream_train_v3.py), with NO training dependencies (no `arxiv`,
`requests`, tokenizer subprocess, etc.).

This lets the PhysVision multimodal model import the architecture cleanly
on any machine. Weights trained by paper 1 load into this class directly
because the module/parameter names are identical.

Only dependencies: torch.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope(dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute Rotary Position Embedding frequencies."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to input tensor. x: (batch, n_heads, seq, head_dim)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit Q/K/V/O projections."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (used in LLaMA, Mistral)."""
    def __init__(self, d_model: int, hidden_mult: float = 2.667):
        super().__init__()
        hidden_dim = int(d_model * hidden_mult)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block: pre-norm, custom attention, SwiGLU FFN."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLMv3(nn.Module):
    """
    Custom decoder-only transformer for domain SLM training.
    Matches paper 1 exactly so pretrained checkpoints load directly.
    """
    def __init__(self, vocab_size=32000, d_model=384, n_layers=6, n_heads=6,
                 max_seq_len=512, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Precompute RoPE
        head_dim = d_model // n_heads
        rope_cos, rope_sin = precompute_rope(head_dim, max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.embed(x))
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        h = self.norm_f(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
