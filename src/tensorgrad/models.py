"""Reference models used by the demos, tests and benchmarks."""

from __future__ import annotations

import numpy as np

from . import functional as F
from . import nn
from .nn.layers import normal_
from .tensor import Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: tuple[int, ...] = (512, 256), n_classes: int = 10,
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvNet(nn.Module):
    """Two conv blocks (conv-BN-ReLU-pool) followed by a small classifier head."""

    def __init__(self, in_channels: int = 1, n_classes: int = 10, width: int = 32, dropout: float = 0.25):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1, bias=False), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * w * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, causal=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.proj(F.gelu(self.fc(self.ln2(x)))))


class CharTransformer(nn.Module):
    """A small GPT-style decoder over characters (learned positional embeddings)."""

    def __init__(self, vocab_size: int, block_size: int = 128, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d_model, std=0.02)
        self.pos = nn.Embedding(block_size, d_model, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # GPT-2 style init: N(0, 0.02) for weights, scaled residual projections.
        for name, p in self.named_parameters():
            if p.ndim == 2 and "weight" in name and "tok" not in name and "pos" not in name:
                std = 0.02 / np.sqrt(2 * n_layers) if name.endswith("proj.weight") else 0.02
                p.data = normal_(p.shape, std)
            elif name.endswith("bias") and "ln" not in name:
                p.data = np.zeros_like(p.data)

    def forward(self, idx: np.ndarray) -> Tensor:
        b, t = idx.shape
        x = self.drop(self.tok(idx) + self.pos(np.arange(t)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def generate(self, idx: np.ndarray, n_new: int, temperature: float = 1.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        from .tensor import no_grad

        rng = rng or np.random.default_rng(0)
        was_training = self.training
        self.eval()
        with no_grad():
            for _ in range(n_new):
                ctx = idx[:, -self.block_size :]
                logits = self(ctx).data[:, -1, :] / temperature
                p = np.exp(logits - logits.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                nxt = np.array([[rng.choice(p.shape[-1], p=row)] for row in p])
                idx = np.concatenate([idx, nxt], axis=1)
        self.train(was_training)
        return idx


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, d_embed: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_embed, std=0.1)
        self.lstm = nn.LSTM(d_embed, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.block_size = 10**9

    def forward(self, idx: np.ndarray) -> Tensor:
        out, _ = self.lstm(self.emb(idx))
        return self.head(out)
