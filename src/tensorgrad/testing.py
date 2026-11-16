"""Finite-difference gradient checking."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .tensor import Tensor, no_grad


def numerical_grad(fn: Callable[..., Tensor], inputs: Sequence[Tensor], weight: np.ndarray,
                   wrt: int, eps: float = 1e-6) -> np.ndarray:
    """Central-difference estimate of d<fn(inputs), weight>/d inputs[wrt]."""
    x = inputs[wrt]
    grad = np.zeros_like(x.data, dtype=np.float64)
    flat = x.data.reshape(-1)  # view: perturbs x in place
    with no_grad():
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            plus = float((fn(*inputs).data * weight).sum())
            flat[i] = orig - eps
            minus = float((fn(*inputs).data * weight).sum())
            flat[i] = orig
            grad.reshape(-1)[i] = (plus - minus) / (2 * eps)
    return grad


def gradcheck(fn: Callable[..., Tensor], inputs: Sequence[Tensor], eps: float = 1e-6,
              atol: float = 1e-6, rtol: float = 1e-4, seed: int = 0) -> None:
    """Assert that autodiff gradients match central finite differences.

    The output is contracted with a random weight tensor so that every output
    element contributes a different upstream gradient (a scalar ``sum`` would
    hide errors that cancel). Inputs should be float64 for meaningful checks.
    """
    rng = np.random.default_rng(seed)
    for t in inputs:
        t.grad = None
    out = fn(*inputs)
    weight = rng.standard_normal(out.shape)
    (out * Tensor(weight.astype(out.dtype))).sum().backward()
    for k, x in enumerate(inputs):
        if not x.requires_grad:
            continue
        analytic = np.zeros_like(x.data) if x.grad is None else x.grad
        numeric = numerical_grad(fn, inputs, weight, k, eps)
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            err = np.max(np.abs(analytic - numeric))
            raise AssertionError(f"gradcheck failed for input {k}: max abs err {err:.3e}")
