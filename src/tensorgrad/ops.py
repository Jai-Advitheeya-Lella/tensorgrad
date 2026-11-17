"""Differentiable primitives.

Each class is a :class:`~tensorgrad.tensor.Function` with a NumPy forward
and an analytic NumPy backward. Gradients returned with a broadcast shape are
reduced automatically by the engine, so binary ops simply return ``g * ...``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .tensor import Function

# --------------------------------------------------------------------------- #
# Binary elementwise
# --------------------------------------------------------------------------- #


class Add(Function):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def backward(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g, g


class Sub(Function):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def backward(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g, -g


class Mul(Function):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        return a * b

    def backward(self, g: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        ga = g * self.b if self.needs_input_grad[0] else None
        gb = g * self.a if self.needs_input_grad[1] else None
        return ga, gb


class Div(Function):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        return a / b

    def backward(self, g: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        ga = g / self.b if self.needs_input_grad[0] else None
        gb = -g * self.a / (self.b * self.b) if self.needs_input_grad[1] else None
        return ga, gb


class Pow(Function):
    """``a ** b`` with both operands differentiable (``a > 0`` for d/db)."""

    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        self.out = a**b
        return self.out

    def backward(self, g: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        ga = g * self.b * self.a ** (self.b - 1) if self.needs_input_grad[0] else None
        gb = None
        if self.needs_input_grad[1]:
            safe_log = np.log(np.where(self.a > 0, self.a, 1.0))
            gb = g * self.out * safe_log
        return ga, gb


class PowScalar(Function):
    def forward(self, a: np.ndarray, exponent: float) -> np.ndarray:
        self.a, self.p = a, exponent
        return a**exponent

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self.p * self.a ** (self.p - 1)


class Maximum(Function):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.mask = a >= b
        return np.maximum(a, b)

    def backward(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g * self.mask, g * ~self.mask


class Minimum(Function):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.mask = a <= b
        return np.minimum(a, b)

    def backward(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g * self.mask, g * ~self.mask


class MatMul(Function):
    """Batched matmul with NumPy semantics (1-D promotion + batch broadcasting)."""

    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a, self.b = a, b
        return a @ b

    def backward(self, g: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        a, b = self.a, self.b
        if a.ndim == 1 and b.ndim == 1:  # inner product -> scalar output
            return g * b, g * a
        a2 = a[None, :] if a.ndim == 1 else a
        b2 = b[:, None] if b.ndim == 1 else b
        # Re-insert the dimensions that 1-D promotion removed from the output.
        g2 = g
        if a.ndim == 1:
            g2 = np.expand_dims(g2, -2)
        if b.ndim == 1:
            g2 = np.expand_dims(g2, -1)
        ga = gb = None
        if self.needs_input_grad[0]:
            ga = g2 @ np.swapaxes(b2, -1, -2)
            ga = _sum_to(ga, a2.shape).reshape(a.shape)
        if self.needs_input_grad[1]:
            gb = np.swapaxes(a2, -1, -2) @ g2
            gb = _sum_to(gb, b2.shape).reshape(b.shape)
        return ga, gb


def _sum_to(x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    extra = x.ndim - len(shape)
    if extra > 0:
        x = x.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and x.shape[i] != 1)
    return x.sum(axis=axes, keepdims=True) if axes else x


class Where(Function):
    """``where(cond, a, b)``; ``cond`` is a constant boolean array."""

    def forward(self, a: np.ndarray, b: np.ndarray, cond: np.ndarray) -> np.ndarray:
        self.cond = cond
        return np.where(cond, a, b)

    def backward(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zero = np.zeros((), dtype=g.dtype)
        return np.where(self.cond, g, zero), np.where(self.cond, zero, g)


# --------------------------------------------------------------------------- #
# Unary elementwise
# --------------------------------------------------------------------------- #


class Cast(Function):
    def forward(self, a: np.ndarray, dtype: np.dtype) -> np.ndarray:
        self.src = a.dtype
        return a.astype(dtype)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g.astype(self.src)


class Neg(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        return -a

    def backward(self, g: np.ndarray) -> np.ndarray:
        return -g


class Exp(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.out = np.exp(a)
        return self.out

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self.out


class Log(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        return np.log(a)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g / self.a


class Sqrt(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.out = np.sqrt(a)
        return self.out

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g / (2.0 * self.out)


class Tanh(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.out = np.tanh(a)
        return self.out

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * (1.0 - self.out * self.out)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable in both tails.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class Sigmoid(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.out = _sigmoid(a)
        return self.out

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self.out * (1.0 - self.out)


class ReLU(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.mask = a > 0
        return a * self.mask

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self.mask


class LeakyReLU(Function):
    def forward(self, a: np.ndarray, slope: float = 0.01) -> np.ndarray:
        self.scale = np.where(a > 0, 1.0, slope).astype(a.dtype)
        return a * self.scale

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self.scale


_GELU_C = float(np.sqrt(2.0 / np.pi))  # Python float: keeps float32 inputs float32 (NEP 50)


class GELU(Function):
    """GELU with the tanh approximation (Hendrycks & Gimpel, 2016)."""

    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        self.t = np.tanh(_GELU_C * (a + 0.044715 * a**3))
        return 0.5 * a * (1.0 + self.t)

    def backward(self, g: np.ndarray) -> np.ndarray:
        a, t = self.a, self.t
        dt = (1.0 - t * t) * _GELU_C * (1.0 + 3 * 0.044715 * a * a)
        return g * (0.5 * (1.0 + t) + 0.5 * a * dt)


class SiLU(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        self.s = _sigmoid(a)
        return a * self.s

    def backward(self, g: np.ndarray) -> np.ndarray:
        s = self.s
        return g * (s * (1.0 + self.a * (1.0 - s)))


class Abs(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.sign = np.sign(a)
        return np.abs(a)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self.sign


class Sin(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        return np.sin(a)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * np.cos(self.a)


class Cos(Function):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        return np.cos(a)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return -g * np.sin(self.a)


class Clip(Function):
    def forward(self, a: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
        self.mask = np.ones(a.shape, dtype=bool)
        if lo is not None:
            self.mask &= a >= lo
        if hi is not None:
            self.mask &= a <= hi
        return np.clip(a, lo, hi)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self.mask


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #


def _expand_reduced(g: np.ndarray, shape: tuple[int, ...], axis: tuple[int, ...] | None,
                    keepdims: bool) -> np.ndarray:
    """Broadcast a reduced gradient back to the input shape."""
    if axis is None:
        return np.broadcast_to(np.reshape(g, (1,) * len(shape)), shape)
    if not keepdims:
        g = np.expand_dims(g, tuple(a % len(shape) for a in axis))
    return np.broadcast_to(g, shape)


class Sum(Function):
    def forward(self, a: np.ndarray, axis: tuple[int, ...] | None, keepdims: bool) -> np.ndarray:
        self.shape, self.axis, self.keepdims = a.shape, axis, keepdims
        return np.asarray(a.sum(axis=axis, keepdims=keepdims))

    def backward(self, g: np.ndarray) -> np.ndarray:
        return _expand_reduced(g, self.shape, self.axis, self.keepdims)


class Mean(Function):
    def forward(self, a: np.ndarray, axis: tuple[int, ...] | None, keepdims: bool) -> np.ndarray:
        self.shape, self.axis, self.keepdims = a.shape, axis, keepdims
        self.n = a.size if axis is None else int(np.prod([a.shape[i] for i in axis]))
        return np.asarray(a.mean(axis=axis, keepdims=keepdims))

    def backward(self, g: np.ndarray) -> np.ndarray:
        return _expand_reduced(g / self.n, self.shape, self.axis, self.keepdims)


class Max(Function):
    """Max reduction. Ties split the gradient evenly (like ``torch.amax``)."""

    def forward(self, a: np.ndarray, axis: tuple[int, ...] | None, keepdims: bool) -> np.ndarray:
        self.axis, self.keepdims, self.shape = axis, keepdims, a.shape
        m = a.max(axis=axis, keepdims=True)
        mask = (a == m).astype(a.dtype)
        self.mask = mask / mask.sum(axis=axis, keepdims=True)
        return np.asarray(m if keepdims else (m.reshape(()) if axis is None else np.squeeze(m, axis)))

    def backward(self, g: np.ndarray) -> np.ndarray:
        return _expand_reduced(g, self.shape, self.axis, self.keepdims) * self.mask


class LogSumExp(Function):
    def forward(self, a: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
        self.axis, self.keepdims = axis, keepdims
        m = a.max(axis=axis, keepdims=True)
        e = np.exp(a - m)
        s = e.sum(axis=axis, keepdims=True)
        self.softmax = e / s
        out = m + np.log(s)
        return out if keepdims else np.squeeze(out, axis)

    def backward(self, g: np.ndarray) -> np.ndarray:
        if not self.keepdims:
            g = np.expand_dims(g, self.axis)
        return g * self.softmax


class Softmax(Function):
    def forward(self, a: np.ndarray, axis: int) -> np.ndarray:
        self.axis = axis
        e = np.exp(a - a.max(axis=axis, keepdims=True))
        self.out = e / e.sum(axis=axis, keepdims=True)
        return self.out

    def backward(self, g: np.ndarray) -> np.ndarray:
        y = self.out
        return y * (g - (g * y).sum(axis=self.axis, keepdims=True))


class LogSoftmax(Function):
    def forward(self, a: np.ndarray, axis: int) -> np.ndarray:
        self.axis = axis
        shifted = a - a.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out = shifted - lse
        self.softmax = np.exp(out)
        return out

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g - self.softmax * g.sum(axis=self.axis, keepdims=True)


# --------------------------------------------------------------------------- #
# Movement / shape ops
# --------------------------------------------------------------------------- #


class Reshape(Function):
    def forward(self, a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        self.in_shape = a.shape
        return a.reshape(shape)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g.reshape(self.in_shape)


class Permute(Function):
    def forward(self, a: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        self.inv = tuple(np.argsort(axes))
        return a.transpose(axes)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g.transpose(self.inv)


class Expand(Function):
    def forward(self, a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        shape = tuple(a.shape[i - (len(shape) - a.ndim)] if s == -1 else s for i, s in enumerate(shape))
        return np.broadcast_to(a, shape)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g  # the engine's unbroadcast does the reduction


def _is_advanced(idx: Any) -> bool:
    items = idx if isinstance(idx, tuple) else (idx,)
    return any(isinstance(i, (list, np.ndarray)) for i in items)


def _has_int_array(idx: Any) -> bool:
    items = idx if isinstance(idx, tuple) else (idx,)
    return any(
        isinstance(i, list) or (isinstance(i, np.ndarray) and i.dtype.kind in "iu") for i in items
    )


class GetItem(Function):
    """Basic slicing and advanced (integer / boolean) indexing."""

    def forward(self, a: np.ndarray, idx: Any) -> np.ndarray:
        self.idx, self.shape, self.dtype = idx, a.shape, a.dtype
        return a[idx]

    def backward(self, g: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape, dtype=g.dtype)
        if _has_int_array(self.idx):
            np.add.at(out, self.idx, g)  # integer indices may repeat -> accumulate
        else:
            out[self.idx] = g
        return out


class Concat(Function):
    def forward(self, *arrays: np.ndarray, axis: int = 0) -> np.ndarray:
        self.axis = axis
        self.splits = np.cumsum([a.shape[axis] for a in arrays])[:-1]
        return np.concatenate(arrays, axis=axis)

    def backward(self, g: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(np.split(g, self.splits, axis=self.axis))


class Stack(Function):
    def forward(self, *arrays: np.ndarray, axis: int = 0) -> np.ndarray:
        self.axis = axis
        return np.stack(arrays, axis=axis)

    def backward(self, g: np.ndarray) -> tuple[np.ndarray, ...]:
        moved = np.moveaxis(g, self.axis, 0)
        return tuple(moved[i] for i in range(moved.shape[0]))


class Pad(Function):
    def forward(self, a: np.ndarray, pad_width: tuple[tuple[int, int], ...], value: float) -> np.ndarray:
        self.slices = tuple(slice(lo, lo + n) for (lo, _), n in zip(pad_width, a.shape))
        return np.pad(a, pad_width, mode="constant", constant_values=value)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g[self.slices]


# --------------------------------------------------------------------------- #
# Normalisation (fused forward/backward)
# --------------------------------------------------------------------------- #


class Normalize(Function):
    """``(x - mean) / sqrt(var + eps) * gamma + beta`` over ``axes``.

    Shared by LayerNorm (axes = trailing dims) and BatchNorm (axes = batch and
    spatial dims). Statistics are treated as functions of ``x``, so the
    backward pass is the full batch-statistics gradient:

    ``dx = rstd / N * (N * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))``.
    """

    def forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                axes: tuple[int, ...], eps: float) -> np.ndarray:
        self.axes = axes
        self.n = int(np.prod([x.shape[a] for a in axes]))
        mean = x.mean(axis=axes, keepdims=True)
        xc = x - mean
        var = (xc * xc).mean(axis=axes, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + eps)
        self.xhat = xc * self.rstd
        self.gamma = gamma
        self.batch_mean, self.batch_var = mean, var
        return self.xhat * gamma + beta

    def backward(self, g: np.ndarray) -> tuple[np.ndarray | None, ...]:
        xhat, axes = self.xhat, self.axes
        dgamma = g * xhat if self.needs_input_grad[1] else None
        dbeta = g if self.needs_input_grad[2] else None
        dx = None
        if self.needs_input_grad[0]:
            dxhat = g * self.gamma
            s1 = dxhat.sum(axis=axes, keepdims=True)
            s2 = (dxhat * xhat).sum(axis=axes, keepdims=True)
            dx = (self.rstd / self.n) * (self.n * dxhat - s1 - xhat * s2)
        return dx, dgamma, dbeta


# --------------------------------------------------------------------------- #
# Convolution and pooling
# --------------------------------------------------------------------------- #


def _pair(v: int | Sequence[int]) -> tuple[int, int]:
    return (int(v), int(v)) if isinstance(v, (int, np.integer)) else (int(v[0]), int(v[1]))


def _windows(x: np.ndarray, kh: int, kw: int, sh: int, sw: int, oh: int, ow: int) -> np.ndarray:
    """Zero-copy view of shape (N, C, OH, OW, KH, KW) over sliding windows."""
    n, c = x.shape[:2]
    s_n, s_c, s_h, s_w = x.strides
    return as_strided(
        x,
        shape=(n, c, oh, ow, kh, kw),
        strides=(s_n, s_c, s_h * sh, s_w * sw, s_h, s_w),
        writeable=False,
    )


def _pad_hw(x: np.ndarray, ph: int, pw: int, value: float = 0.0) -> np.ndarray:
    if ph == 0 and pw == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=value)


def _col2im(dcols: np.ndarray, padded_shape: tuple[int, ...], sh: int, sw: int) -> np.ndarray:
    """Adjoint of im2col: add (C, KH, KW, N, OH, OW) column grads back into an NCHW image.

    One strided add per kernel offset (KH*KW of them). The accumulator is kept
    in (C, N, H, W) order so both operands of every add share a layout.
    """
    c, kh, kw, n, oh, ow = dcols.shape
    _, _, hp, wp = padded_shape
    out = np.zeros((c, n, hp, wp), dtype=dcols.dtype)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw] += dcols[:, i, j]
    return out.transpose(1, 0, 2, 3)


def _unpad_hw(x: np.ndarray, ph: int, pw: int) -> np.ndarray:
    h, w = x.shape[2:]
    return x[:, :, ph : h - ph, pw : w - pw]


class Conv2d(Function):
    """2-D cross-correlation via im2col + one GEMM per pass.

    Forward: copy every receptive field into a column matrix ``cols`` of shape
    ``(C*KH*KW, N*OH*OW)`` and compute ``Y = W cols`` with ``W`` the kernel
    flattened to ``(OC, C*KH*KW)``. Backward: ``dW = dY cols^T`` and
    ``dcols = W^T dY`` (one GEMM each), then col2im scatters ``dcols`` back
    into the (padded) input.
    """

    def forward(self, x: np.ndarray, w: np.ndarray, stride: Any = 1, padding: Any = 0) -> np.ndarray:
        sh, sw = _pair(stride)
        ph, pw = _pair(padding)
        n, c, _, _ = x.shape
        oc, c2, kh, kw = w.shape
        if c != c2:
            raise ValueError(f"conv2d: input has {c} channels, weight expects {c2}")
        xp = _pad_hw(x, ph, pw)
        hp, wp = xp.shape[2:]
        oh, ow = (hp - kh) // sh + 1, (wp - kw) // sw + 1
        # (N, C, OH, OW, KH, KW) view -> (C, KH, KW, N, OH, OW) copy = im2col
        win = _windows(xp, kh, kw, sh, sw, oh, ow).transpose(1, 4, 5, 0, 2, 3)
        cols = np.ascontiguousarray(win).reshape(c * kh * kw, n * oh * ow)
        w2 = w.reshape(oc, -1)
        out = w2 @ cols  # (OC, N*OH*OW)
        self.cols = cols if self.needs_input_grad[1] else None
        self.w2, self.w_shape = w2, w.shape
        self.geom = (n, c, hp, wp, oh, ow, kh, kw, sh, sw, ph, pw)
        return out.reshape(oc, n, oh, ow).transpose(1, 0, 2, 3)

    def backward(self, g: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        n, c, hp, wp, oh, ow, kh, kw, sh, sw, ph, pw = self.geom
        oc = self.w_shape[0]
        g2 = g.transpose(1, 0, 2, 3).reshape(oc, n * oh * ow)
        dw = dx = None
        if self.needs_input_grad[1]:
            dw = (g2 @ self.cols.T).reshape(self.w_shape)
        if self.needs_input_grad[0]:
            dcols = (self.w2.T @ g2).reshape(c, kh, kw, n, oh, ow)
            dx = _unpad_hw(_col2im(dcols, (n, c, hp, wp), sh, sw), ph, pw)
        return dx, dw


class MaxPool2d(Function):
    def forward(self, x: np.ndarray, kernel_size: Any, stride: Any = None, padding: Any = 0) -> np.ndarray:
        kh, kw = _pair(kernel_size)
        sh, sw = _pair(stride if stride is not None else kernel_size)
        ph, pw = _pair(padding)
        xp = _pad_hw(x, ph, pw, value=-np.inf)
        n, c, hp, wp = xp.shape
        oh, ow = (hp - kh) // sh + 1, (wp - kw) // sw + 1
        win = _windows(xp, kh, kw, sh, sw, oh, ow).reshape(n, c, oh, ow, kh * kw)
        self.argmax = win.argmax(axis=-1)
        self.geom = (n, c, hp, wp, oh, ow, kh, kw, sh, sw, ph, pw)
        return np.take_along_axis(win, self.argmax[..., None], axis=-1)[..., 0]

    def backward(self, g: np.ndarray) -> np.ndarray:
        n, c, hp, wp, oh, ow, kh, kw, sh, sw, ph, pw = self.geom
        out = np.zeros((n, c, hp, wp), dtype=g.dtype)
        for i in range(kh):
            for j in range(kw):
                hit = self.argmax == i * kw + j
                out[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw] += g * hit
        return _unpad_hw(out, ph, pw)


class AvgPool2d(Function):
    """Average pooling; zero padding counts towards the average (PyTorch default)."""

    def forward(self, x: np.ndarray, kernel_size: Any, stride: Any = None, padding: Any = 0) -> np.ndarray:
        kh, kw = _pair(kernel_size)
        sh, sw = _pair(stride if stride is not None else kernel_size)
        ph, pw = _pair(padding)
        xp = _pad_hw(x, ph, pw)
        n, c, hp, wp = xp.shape
        oh, ow = (hp - kh) // sh + 1, (wp - kw) // sw + 1
        self.geom = (n, c, hp, wp, oh, ow, kh, kw, sh, sw, ph, pw)
        return _windows(xp, kh, kw, sh, sw, oh, ow).mean(axis=(4, 5))

    def backward(self, g: np.ndarray) -> np.ndarray:
        n, c, hp, wp, oh, ow, kh, kw, sh, sw, ph, pw = self.geom
        out = np.zeros((n, c, hp, wp), dtype=g.dtype)
        share = g / (kh * kw)
        for i in range(kh):
            for j in range(kw):
                out[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw] += share
        return _unpad_hw(out, ph, pw)
