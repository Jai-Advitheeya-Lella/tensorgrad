"""Standard layers built on :mod:`tensorgrad.functional`."""

from __future__ import annotations

import math
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .. import functional as F
from ..tensor import Tensor, get_default_dtype, get_rng
from .module import Module, Parameter

# --------------------------------------------------------------------------- #
# Initialisers
# --------------------------------------------------------------------------- #


def uniform_(shape: Sequence[int], bound: float) -> np.ndarray:
    return get_rng().uniform(-bound, bound, size=tuple(shape)).astype(get_default_dtype())


def normal_(shape: Sequence[int], std: float) -> np.ndarray:
    return (get_rng().standard_normal(tuple(shape)) * std).astype(get_default_dtype())


def kaiming_uniform(shape: Sequence[int], fan_in: int, gain: float = math.sqrt(2.0)) -> np.ndarray:
    return uniform_(shape, gain * math.sqrt(3.0 / fan_in))


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, i: int) -> Module:
        return list(self._modules.values())[i]

    def __len__(self) -> int:
        return len(self._modules)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = ()) -> None:
        super().__init__()
        for m in modules:
            self.append(m)

    def append(self, m: Module) -> None:
        setattr(self, str(len(self._modules)), m)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, i: int) -> Module:
        return list(self._modules.values())[i]

    def __len__(self) -> int:
        return len(self._modules)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


# --------------------------------------------------------------------------- #
# Parametric layers
# --------------------------------------------------------------------------- #


class Linear(Module):
    """``y = x W^T + b`` with PyTorch's default init (U(-1/sqrt(in), 1/sqrt(in)))."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(uniform_((out_features, in_features), bound))
        self.bias = Parameter(uniform_((out_features,), bound)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = Parameter(uniform_((out_channels, in_channels, kernel_size, kernel_size), bound))
        self.bias = Parameter(uniform_((out_channels,), bound)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, std: float = 1.0) -> None:
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(normal_((num_embeddings, embedding_dim), std))

    def forward(self, idx: Any) -> Tensor:
        return F.embedding(idx, self.weight)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class LayerNorm(Module):
    def __init__(self, normalized_shape: int | Sequence[int], eps: float = 1e-5) -> None:
        super().__init__()
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.normalized_shape, self.eps = shape, eps
        self.weight = Parameter(np.ones(shape, dtype=get_default_dtype()))
        self.bias = Parameter(np.zeros(shape, dtype=get_default_dtype()))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class _BatchNorm(Module):
    expected_ndim: tuple[int, ...] = ()

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        dt = get_default_dtype()
        self.weight = Parameter(np.ones(num_features, dtype=dt))
        self.bias = Parameter(np.zeros(num_features, dtype=dt))
        self.register_buffer("running_mean", np.zeros(num_features, dtype=dt))
        self.register_buffer("running_var", np.ones(num_features, dtype=dt))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in self.expected_ndim:
            raise ValueError(f"{type(self).__name__} expects ndim in {self.expected_ndim}, got {x.ndim}")
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            training=self.training, momentum=self.momentum, eps=self.eps)

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"


class BatchNorm1d(_BatchNorm):
    expected_ndim = (2, 3)


class BatchNorm2d(_BatchNorm):
    expected_ndim = (4,)


# --------------------------------------------------------------------------- #
# Parameter-free layers
# --------------------------------------------------------------------------- #


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size}"


class AvgPool2d(MaxPool2d):
    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


# --------------------------------------------------------------------------- #
# Sequence layers
# --------------------------------------------------------------------------- #


class MultiHeadAttention(Module):
    """Multi-head self-attention with a fused QKV projection."""

    def __init__(self, d_model: int, n_heads: int, causal: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.causal, self.dropout = d_model, n_heads, causal, dropout
        self.qkv = Linear(d_model, 3 * d_model)
        self.proj = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        h = self.n_heads
        qkv = self.qkv(x).reshape(b, t, 3, h, d // h).permute(2, 0, 3, 1, 4)  # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, causal=self.causal,
                                           dropout_p=self.dropout, training=self.training)
        y = y.permute(0, 2, 1, 3).reshape(b, t, d)
        return self.proj(y)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, causal={self.causal}"


class LSTM(Module):
    """Single-layer, batch-first LSTM with PyTorch's gate layout (i, f, g, o)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        bound = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(uniform_((4 * hidden_size, input_size), bound))
        self.weight_hh = Parameter(uniform_((4 * hidden_size, hidden_size), bound))
        self.bias_ih = Parameter(uniform_((4 * hidden_size,), bound))
        self.bias_hh = Parameter(uniform_((4 * hidden_size,), bound))

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
                ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, _ = x.shape
        hs = self.hidden_size
        if state is None:
            h = Tensor(np.zeros((b, hs), dtype=x.dtype))
            c = Tensor(np.zeros((b, hs), dtype=x.dtype))
        else:
            h, c = state
        # Input projection for all time steps in one GEMM.
        xw = F.linear(x, self.weight_ih, self.bias_ih + self.bias_hh)  # (B, T, 4H)
        outs = []
        for step in range(t):
            gates = xw[:, step] + h @ self.weight_hh.transpose()
            i = gates[:, 0:hs].sigmoid()
            f = gates[:, hs : 2 * hs].sigmoid()
            g = gates[:, 2 * hs : 3 * hs].tanh()
            o = gates[:, 3 * hs :].sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            outs.append(h)
        return F.stack(outs, axis=1), (h, c)

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}"


# --------------------------------------------------------------------------- #
# Losses as modules
# --------------------------------------------------------------------------- #


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.reduction, self.label_smoothing = reduction, label_smoothing

    def forward(self, logits: Tensor, target: Any) -> Tensor:
        return F.cross_entropy(logits, target, self.reduction, self.label_smoothing)


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Any) -> Tensor:
        return F.mse_loss(pred, target, self.reduction)
