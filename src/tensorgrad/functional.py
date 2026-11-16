"""Stateless neural-network functions (the ``F`` namespace).

Most of these are thin wrappers around fused primitives in :mod:`ops`; the
losses and attention are *composed* from primitives, so their gradients come
for free from the autodiff engine.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import ops
from .tensor import Tensor, get_rng

# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #


def relu(x: Tensor) -> Tensor:
    return ops.ReLU.apply(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return ops.LeakyReLU.apply(x, slope=negative_slope)


def gelu(x: Tensor) -> Tensor:
    """GELU, tanh approximation (``torch.nn.functional.gelu(x, approximate='tanh')``)."""
    return ops.GELU.apply(x)


def silu(x: Tensor) -> Tensor:
    return ops.SiLU.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    return ops.Sigmoid.apply(x)


def tanh(x: Tensor) -> Tensor:
    return ops.Tanh.apply(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return ops.Softmax.apply(x, axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return ops.LogSoftmax.apply(x, axis=axis)


def logsumexp(x: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    return ops.LogSumExp.apply(x, axis=axis, keepdims=keepdims)


# --------------------------------------------------------------------------- #
# Tensor utilities
# --------------------------------------------------------------------------- #


def concat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    return ops.Concat.apply(*tensors, axis=axis)


cat = concat


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    return ops.Stack.apply(*tensors, axis=axis)


def where(cond: Any, a: Any, b: Any) -> Tensor:
    cond_arr = cond.data if isinstance(cond, Tensor) else np.asarray(cond)
    ref = a if isinstance(a, Tensor) else b
    dtype = ref.dtype if isinstance(ref, Tensor) else None
    a_t = a if isinstance(a, Tensor) else Tensor(np.asarray(a, dtype=dtype))
    b_t = b if isinstance(b, Tensor) else Tensor(np.asarray(b, dtype=dtype))
    return ops.Where.apply(a_t, b_t, cond=cond_arr.astype(bool))


def maximum(a: Tensor, b: Any) -> Tensor:
    return a.maximum(b)


def minimum(a: Tensor, b: Any) -> Tensor:
    return a.minimum(b)


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``x @ weight.T + bias`` with ``weight`` of shape (out, in), like PyTorch."""
    out = x @ weight.transpose()
    return out + bias if bias is not None else out


def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride: Any = 1,
           padding: Any = 0) -> Tensor:
    out = ops.Conv2d.apply(x, weight, stride=stride, padding=padding)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)
    return out


def max_pool2d(x: Tensor, kernel_size: Any, stride: Any = None, padding: Any = 0) -> Tensor:
    return ops.MaxPool2d.apply(x, kernel_size=kernel_size, stride=stride, padding=padding)


def avg_pool2d(x: Tensor, kernel_size: Any, stride: Any = None, padding: Any = 0) -> Tensor:
    return ops.AvgPool2d.apply(x, kernel_size=kernel_size, stride=stride, padding=padding)


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Inverted dropout: zero with prob ``p`` and rescale survivors by ``1/(1-p)``."""
    if not training or p == 0.0:
        return x
    if p >= 1.0:
        return x * 0.0
    keep = (get_rng().random(x.shape) >= p).astype(x.dtype) / (1.0 - p)
    return x * Tensor(keep)


def layer_norm(x: Tensor, normalized_shape: Sequence[int], weight: Tensor | None = None,
               bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    k = len(tuple(normalized_shape))
    axes = tuple(range(x.ndim - k, x.ndim))
    w = weight if weight is not None else Tensor(np.ones(normalized_shape, dtype=x.dtype))
    b = bias if bias is not None else Tensor(np.zeros(normalized_shape, dtype=x.dtype))
    return ops.Normalize.apply(x, w, b, axes=axes, eps=eps)


def batch_norm(x: Tensor, running_mean: np.ndarray | None, running_var: np.ndarray | None,
               weight: Tensor | None = None, bias: Tensor | None = None, training: bool = True,
               momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """Batch normalisation over every axis except the channel axis (axis 1).

    In training mode the batch statistics are used and the running buffers are
    updated in place (running variance uses the unbiased estimator, as in
    PyTorch). In eval mode the running statistics are used.
    """
    c = x.shape[1]
    bshape = (1, c) + (1,) * (x.ndim - 2)
    w = (weight if weight is not None else Tensor(np.ones(c, dtype=x.dtype))).reshape(bshape)
    b = (bias if bias is not None else Tensor(np.zeros(c, dtype=x.dtype))).reshape(bshape)
    if training:
        axes = (0,) + tuple(range(2, x.ndim))
        out = ops.Normalize.apply(x, w, b, axes=axes, eps=eps)
        if running_mean is not None and running_var is not None:
            n = x.size // c
            mean = x.data.mean(axis=axes)
            var = x.data.var(axis=axes) * (n / max(n - 1, 1))
            running_mean *= 1.0 - momentum
            running_mean += momentum * mean
            running_var *= 1.0 - momentum
            running_var += momentum * var
        return out
    if running_mean is None or running_var is None:
        raise ValueError("eval-mode batch_norm needs running statistics")
    scale = Tensor((1.0 / np.sqrt(running_var + eps)).astype(x.dtype).reshape(bshape))
    shift = Tensor(running_mean.astype(x.dtype).reshape(bshape))
    return (x - shift) * scale * w + b


def embedding(idx: Any, weight: Tensor) -> Tensor:
    """Row lookup ``weight[idx]``; the backward scatter-adds into the used rows."""
    idx_arr = idx.data if isinstance(idx, Tensor) else np.asarray(idx)
    return weight[idx_arr.astype(np.int64)]


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False,
                                 dropout_p: float = 0.0, training: bool = True) -> Tensor:
    """``softmax(q k^T / sqrt(d) + mask) v`` over the last two axes."""
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(d))
    if causal:
        tq, tk = q.shape[-2], k.shape[-2]
        mask = np.triu(np.ones((tq, tk), dtype=bool), k=1 + tk - tq)
        scores = where(mask, Tensor(np.asarray(-1e9, dtype=scores.dtype)), scores)
    attn = softmax(scores, axis=-1)
    attn = dropout(attn, dropout_p, training)
    return attn @ v


def pad(x: Tensor, pad_width: Sequence[tuple[int, int]], value: float = 0.0) -> Tensor:
    return x.pad(pad_width, value)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction {reduction!r}")


def nll_loss(log_probs: Tensor, target: Any, reduction: str = "mean") -> Tensor:
    """Negative log-likelihood for (N, C) log-probabilities and integer targets."""
    t = (target.data if isinstance(target, Tensor) else np.asarray(target)).astype(np.int64)
    picked = log_probs[np.arange(log_probs.shape[0]), t]
    return _reduce(-picked, reduction)


def cross_entropy(logits: Tensor, target: Any, reduction: str = "mean",
                  label_smoothing: float = 0.0) -> Tensor:
    """Softmax cross-entropy for (N, C) logits; extra leading dims are flattened."""
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
        target = (target.data if isinstance(target, Tensor) else np.asarray(target)).reshape(-1)
    logp = log_softmax(logits, axis=-1)
    loss = nll_loss(logp, target, reduction="none")
    if label_smoothing > 0.0:
        smooth = -logp.mean(axis=-1)
        loss = loss * (1.0 - label_smoothing) + smooth * label_smoothing
    return _reduce(loss, reduction)


def mse_loss(pred: Tensor, target: Any, reduction: str = "mean") -> Tensor:
    diff = pred - target
    return _reduce(diff * diff, reduction)


def l1_loss(pred: Tensor, target: Any, reduction: str = "mean") -> Tensor:
    return _reduce((pred - target).abs(), reduction)


def binary_cross_entropy_with_logits(logits: Tensor, target: Any, reduction: str = "mean") -> Tensor:
    """Stable BCE: ``max(x, 0) - x*y + log(1 + exp(-|x|))``."""
    loss = relu(logits) - logits * target + (1.0 + (-logits.abs()).exp()).log()
    return _reduce(loss, reduction)
