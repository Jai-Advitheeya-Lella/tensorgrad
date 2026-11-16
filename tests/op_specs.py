"""Op specifications shared by the finite-difference and PyTorch-parity tests.

Each spec gives the tensorgrad function, the equivalent PyTorch function, the
input shapes and the sampling domain of each input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import tensorgrad.functional as F

_rng = np.random.default_rng(1234)
# Constant (non-differentiable) arguments shared by both implementations.
COND = _rng.random((3, 4)) > 0.5
BOOL_MASK = np.array([True, False, True, True, False])
CE_TARGET = np.array([0, 3, 1, 4, 2, 3])
CE_TARGET_3D = np.array([[0, 4, 2], [1, 1, 3]])
EMB_IDX = np.array([[1, 3, 3], [0, 9, 1]])
BCE_TARGET = (_rng.random((4, 3)) > 0.5).astype(np.float64)
MSE_TARGET = _rng.standard_normal((4, 3))


@dataclass
class OpSpec:
    name: str
    tg: Callable[..., Any]
    th: Callable[..., Any]
    shapes: list[tuple[int, ...]]
    domains: list[tuple[float, float]] | None = None  # per input, default (-1, 1)
    atol: float = 1e-6
    extra: dict[str, Any] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator) -> list[np.ndarray]:
        doms = self.domains or [(-1.0, 1.0)] * len(self.shapes)
        return [rng.uniform(lo, hi, size=s) for s, (lo, hi) in zip(self.shapes, doms)]


def _th():
    import torch
    import torch.nn.functional as TF

    return torch, TF


def T(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a lambda that takes (torch, torch.nn.functional, *tensors)."""

    def wrapped(*xs: Any) -> Any:
        torch, TF = _th()
        return fn(torch, TF, *xs)

    return wrapped


POS = (0.5, 2.0)

SPECS: list[OpSpec] = [
    # ---- binary elementwise + broadcasting
    OpSpec("add", lambda a, b: a + b, T(lambda t, f, a, b: a + b), [(3, 4), (3, 4)]),
    OpSpec("add_broadcast_trailing", lambda a, b: a + b, T(lambda t, f, a, b: a + b), [(2, 3, 4), (4,)]),
    OpSpec("add_broadcast_both", lambda a, b: a + b, T(lambda t, f, a, b: a + b), [(2, 1, 4), (3, 1)]),
    OpSpec("sub_broadcast", lambda a, b: a - b, T(lambda t, f, a, b: a - b), [(3, 4), (3, 1)]),
    OpSpec("mul_broadcast", lambda a, b: a * b, T(lambda t, f, a, b: a * b), [(2, 3, 4), (1, 3, 1)]),
    OpSpec("div", lambda a, b: a / b, T(lambda t, f, a, b: a / b), [(3, 4), (3, 4)], [(-1, 1), POS]),
    OpSpec("rsub_scalar", lambda a: 2.0 - a, T(lambda t, f, a: 2.0 - a), [(3, 4)]),
    OpSpec("rdiv_scalar", lambda a: 1.0 / a, T(lambda t, f, a: 1.0 / a), [(3, 4)], [POS]),
    OpSpec("pow_scalar", lambda a: a**3, T(lambda t, f, a: a**3), [(3, 4)]),
    OpSpec("pow_fractional", lambda a: a**0.5, T(lambda t, f, a: a**0.5), [(3, 4)], [POS]),
    OpSpec("pow_tensor", lambda a, b: a**b, T(lambda t, f, a, b: a**b), [(3, 4), (3, 4)], [POS, (-1, 1)]),
    OpSpec("maximum", lambda a, b: a.maximum(b), T(lambda t, f, a, b: t.maximum(a, b)), [(3, 4), (3, 4)]),
    OpSpec("minimum", lambda a, b: a.minimum(b), T(lambda t, f, a, b: t.minimum(a, b)), [(3, 4), (3, 4)]),
    # ---- unary elementwise
    OpSpec("neg", lambda a: -a, T(lambda t, f, a: -a), [(3, 4)]),
    OpSpec("exp", lambda a: a.exp(), T(lambda t, f, a: a.exp()), [(3, 4)]),
    OpSpec("log", lambda a: a.log(), T(lambda t, f, a: a.log()), [(3, 4)], [POS]),
    OpSpec("sqrt", lambda a: a.sqrt(), T(lambda t, f, a: a.sqrt()), [(3, 4)], [POS]),
    OpSpec("tanh", lambda a: a.tanh(), T(lambda t, f, a: a.tanh()), [(3, 4)], [(-3, 3)]),
    OpSpec("sigmoid", lambda a: a.sigmoid(), T(lambda t, f, a: a.sigmoid()), [(3, 4)], [(-6, 6)]),
    OpSpec("relu", F.relu, T(lambda t, f, a: f.relu(a)), [(3, 4)]),
    OpSpec("leaky_relu", lambda a: F.leaky_relu(a, 0.1), T(lambda t, f, a: f.leaky_relu(a, 0.1)), [(3, 4)]),
    OpSpec("gelu", F.gelu, T(lambda t, f, a: f.gelu(a, approximate="tanh")), [(3, 4)], [(-3, 3)]),
    OpSpec("silu", F.silu, T(lambda t, f, a: f.silu(a)), [(3, 4)], [(-3, 3)]),
    OpSpec("abs", lambda a: a.abs(), T(lambda t, f, a: a.abs()), [(3, 4)]),
    OpSpec("sin", lambda a: a.sin(), T(lambda t, f, a: a.sin()), [(3, 4)], [(-3, 3)]),
    OpSpec("cos", lambda a: a.cos(), T(lambda t, f, a: a.cos()), [(3, 4)], [(-3, 3)]),
    OpSpec("clip", lambda a: a.clip(-0.5, 0.5), T(lambda t, f, a: a.clamp(-0.5, 0.5)), [(3, 4)]),
    # ---- matmul (all NumPy promotion / broadcasting cases)
    OpSpec("matmul_2d", lambda a, b: a @ b, T(lambda t, f, a, b: a @ b), [(3, 4), (4, 5)]),
    OpSpec("matmul_batched", lambda a, b: a @ b, T(lambda t, f, a, b: a @ b), [(2, 3, 4), (4, 5)]),
    OpSpec("matmul_batch_broadcast", lambda a, b: a @ b, T(lambda t, f, a, b: a @ b),
           [(2, 1, 3, 4), (5, 4, 2)]),
    OpSpec("matmul_vec_mat", lambda a, b: a @ b, T(lambda t, f, a, b: a @ b), [(4,), (4, 5)]),
    OpSpec("matmul_mat_vec", lambda a, b: a @ b, T(lambda t, f, a, b: a @ b), [(2, 3, 4), (4,)]),
    OpSpec("matmul_vec_vec", lambda a, b: a @ b, T(lambda t, f, a, b: a @ b), [(4,), (4,)]),
    OpSpec("linear", F.linear, T(lambda t, f, x, w, b: f.linear(x, w, b)), [(2, 3, 4), (5, 4), (5,)]),
    # ---- reductions
    OpSpec("sum_all", lambda a: a.sum(), T(lambda t, f, a: a.sum()), [(3, 4)]),
    OpSpec("sum_axis", lambda a: a.sum(1), T(lambda t, f, a: a.sum(1)), [(2, 3, 4)]),
    OpSpec("sum_axes_keepdims", lambda a: a.sum((0, 2), keepdims=True),
           T(lambda t, f, a: a.sum((0, 2), keepdim=True)), [(2, 3, 4)]),
    OpSpec("mean_axis", lambda a: a.mean(-1), T(lambda t, f, a: a.mean(-1)), [(2, 3, 4)]),
    OpSpec("mean_all", lambda a: a.mean(), T(lambda t, f, a: a.mean()), [(2, 3, 4)]),
    OpSpec("max_axis", lambda a: a.max(1), T(lambda t, f, a: a.amax(1)), [(2, 3, 4)]),
    OpSpec("max_all", lambda a: a.max(), T(lambda t, f, a: a.amax()), [(3, 4)]),
    OpSpec("min_axis_keepdims", lambda a: a.min(0, keepdims=True),
           T(lambda t, f, a: a.amin(0, keepdim=True)), [(3, 4)]),
    OpSpec("var_axis", lambda a: a.var(1), T(lambda t, f, a: a.var(1, unbiased=False)), [(3, 5)]),
    OpSpec("std_unbiased", lambda a: a.std(0, correction=1), T(lambda t, f, a: a.std(0, unbiased=True)),
           [(5, 3)]),
    OpSpec("logsumexp", lambda a: a.logsumexp(1), T(lambda t, f, a: a.logsumexp(1)), [(3, 5)], [(-4, 4)]),
    OpSpec("logsumexp_keepdims", lambda a: a.logsumexp(0, keepdims=True),
           T(lambda t, f, a: a.logsumexp(0, keepdim=True)), [(3, 5)], [(-4, 4)]),
    OpSpec("softmax", F.softmax, T(lambda t, f, a: f.softmax(a, -1)), [(3, 5)], [(-4, 4)]),
    OpSpec("softmax_axis0", lambda a: F.softmax(a, 0), T(lambda t, f, a: f.softmax(a, 0)), [(3, 5)]),
    OpSpec("log_softmax", F.log_softmax, T(lambda t, f, a: f.log_softmax(a, -1)), [(3, 5)], [(-4, 4)]),
    # ---- movement / indexing
    OpSpec("reshape", lambda a: a.reshape(4, 6), T(lambda t, f, a: a.reshape(4, 6)), [(2, 3, 4)]),
    OpSpec("transpose_2d", lambda a: a.T, T(lambda t, f, a: a.T), [(3, 4)]),
    OpSpec("transpose_swap", lambda a: a.transpose(0, 2), T(lambda t, f, a: a.transpose(0, 2)), [(2, 3, 4)]),
    OpSpec("permute", lambda a: a.permute(2, 0, 1), T(lambda t, f, a: a.permute(2, 0, 1)), [(2, 3, 4)]),
    OpSpec("getitem_slice", lambda a: a[1:, ::2], T(lambda t, f, a: a[1:, ::2]), [(3, 5)]),
    OpSpec("getitem_int_dups", lambda a: a[np.array([0, 2, 0, 2])],
           T(lambda t, f, a: a[t.tensor([0, 2, 0, 2])]), [(3, 4)]),
    OpSpec("getitem_pairs", lambda a: a[np.array([0, 1, 1]), np.array([2, 3, 3])],
           T(lambda t, f, a: a[t.tensor([0, 1, 1]), t.tensor([2, 3, 3])]), [(3, 4)]),
    OpSpec("getitem_bool", lambda a: a[BOOL_MASK], T(lambda t, f, a: a[t.from_numpy(BOOL_MASK)]), [(5, 2)]),
    OpSpec("concat", lambda a, b, c: F.concat([a, b, c], axis=1),
           T(lambda t, f, a, b, c: t.cat([a, b, c], dim=1)), [(2, 1, 3), (2, 4, 3), (2, 2, 3)]),
    OpSpec("stack", lambda a, b: F.stack([a, b], axis=1), T(lambda t, f, a, b: t.stack([a, b], dim=1)),
           [(3, 4), (3, 4)]),
    OpSpec("expand", lambda a: a.expand(2, 3, 4), T(lambda t, f, a: a.expand(2, 3, 4)), [(3, 1)]),
    OpSpec("unsqueeze_squeeze", lambda a: a.unsqueeze(1).squeeze(), T(lambda t, f, a: a.unsqueeze(1).squeeze()),
           [(3, 4)]),
    OpSpec("flatten", lambda a: a.flatten(1), T(lambda t, f, a: a.flatten(1)), [(2, 3, 4)]),
    OpSpec("pad", lambda a: a.pad(((1, 0), (2, 1))), T(lambda t, f, a: f.pad(a, (2, 1, 1, 0))), [(3, 4)]),
    OpSpec("where", lambda a, b: F.where(COND, a, b), T(lambda t, f, a, b: t.where(t.from_numpy(COND), a, b)),
           [(3, 4), (4,)]),
    # ---- convolution & pooling
    OpSpec("conv2d_pad1", lambda x, w, b: F.conv2d(x, w, b, padding=1),
           T(lambda t, f, x, w, b: f.conv2d(x, w, b, padding=1)), [(2, 3, 6, 6), (4, 3, 3, 3), (4,)]),
    OpSpec("conv2d_stride2", lambda x, w: F.conv2d(x, w, stride=2),
           T(lambda t, f, x, w: f.conv2d(x, w, stride=2)), [(2, 2, 8, 7), (3, 2, 3, 3)]),
    OpSpec("conv2d_5x5_rect", lambda x, w: F.conv2d(x, w, stride=(1, 2), padding=(2, 1)),
           T(lambda t, f, x, w: f.conv2d(x, w, stride=(1, 2), padding=(2, 1))), [(1, 2, 6, 9), (2, 2, 5, 5)]),
    OpSpec("max_pool2d", lambda x: F.max_pool2d(x, 2), T(lambda t, f, x: f.max_pool2d(x, 2)), [(2, 3, 6, 6)]),
    OpSpec("max_pool2d_overlap_pad", lambda x: F.max_pool2d(x, 3, 2, 1),
           T(lambda t, f, x: f.max_pool2d(x, 3, 2, 1)), [(2, 2, 7, 7)]),
    OpSpec("avg_pool2d", lambda x: F.avg_pool2d(x, 2), T(lambda t, f, x: f.avg_pool2d(x, 2)), [(2, 3, 6, 6)]),
    OpSpec("avg_pool2d_pad", lambda x: F.avg_pool2d(x, 3, 1, 1),
           T(lambda t, f, x: f.avg_pool2d(x, 3, 1, 1)), [(1, 2, 5, 5)]),
    # ---- normalisation
    OpSpec("layer_norm", lambda x, w, b: F.layer_norm(x, (6,), w, b),
           T(lambda t, f, x, w, b: f.layer_norm(x, (6,), w, b)), [(4, 6), (6,), (6,)]),
    OpSpec("layer_norm_2d", lambda x, w, b: F.layer_norm(x, (3, 4), w, b),
           T(lambda t, f, x, w, b: f.layer_norm(x, (3, 4), w, b)), [(2, 3, 4), (3, 4), (3, 4)]),
    OpSpec("batch_norm_2d_train", lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True),
           T(lambda t, f, x, w, b: f.batch_norm(x, None, None, w, b, training=True)),
           [(4, 3, 5, 5), (3,), (3,)]),
    OpSpec("batch_norm_1d_train", lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True),
           T(lambda t, f, x, w, b: f.batch_norm(x, None, None, w, b, training=True)), [(8, 5), (5,), (5,)]),
    # ---- losses
    OpSpec("cross_entropy", lambda x: F.cross_entropy(x, CE_TARGET),
           T(lambda t, f, x: f.cross_entropy(x, t.from_numpy(CE_TARGET))), [(6, 5)], [(-3, 3)]),
    OpSpec("cross_entropy_smooth", lambda x: F.cross_entropy(x, CE_TARGET, label_smoothing=0.1),
           T(lambda t, f, x: f.cross_entropy(x, t.from_numpy(CE_TARGET), label_smoothing=0.1)),
           [(6, 5)], [(-3, 3)]),
    OpSpec("cross_entropy_3d", lambda x: F.cross_entropy(x, CE_TARGET_3D),
           T(lambda t, f, x: f.cross_entropy(x.reshape(-1, 5), t.from_numpy(CE_TARGET_3D).reshape(-1))),
           [(2, 3, 5)], [(-3, 3)]),
    OpSpec("nll_loss_sum", lambda x: F.nll_loss(F.log_softmax(x), CE_TARGET, reduction="sum"),
           T(lambda t, f, x: f.nll_loss(f.log_softmax(x, -1), t.from_numpy(CE_TARGET), reduction="sum")),
           [(6, 5)]),
    OpSpec("mse_loss", lambda x: F.mse_loss(x, MSE_TARGET),
           T(lambda t, f, x: f.mse_loss(x, t.from_numpy(MSE_TARGET))), [(4, 3)]),
    OpSpec("l1_loss", lambda x: F.l1_loss(x, MSE_TARGET),
           T(lambda t, f, x: f.l1_loss(x, t.from_numpy(MSE_TARGET))), [(4, 3)]),
    OpSpec("bce_with_logits", lambda x: F.binary_cross_entropy_with_logits(x, BCE_TARGET),
           T(lambda t, f, x: f.binary_cross_entropy_with_logits(x, t.from_numpy(BCE_TARGET))),
           [(4, 3)], [(-5, 5)]),
    # ---- composite layers
    OpSpec("embedding", lambda w: F.embedding(EMB_IDX, w), T(lambda t, f, w: f.embedding(t.from_numpy(EMB_IDX), w)),
           [(10, 4)]),
    OpSpec("attention_causal", lambda q, k, v: F.scaled_dot_product_attention(q, k, v, causal=True),
           T(lambda t, f, q, k, v: f.scaled_dot_product_attention(q, k, v, is_causal=True)),
           [(2, 2, 5, 4), (2, 2, 5, 4), (2, 2, 5, 4)]),
    OpSpec("attention_full", lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
           T(lambda t, f, q, k, v: f.scaled_dot_product_attention(q, k, v)),
           [(1, 3, 4), (1, 6, 4), (1, 6, 2)]),
]

SPEC_IDS = [s.name for s in SPECS]
