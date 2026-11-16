"""Training utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..tensor import Tensor


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """Rescale gradients in place so their global L2 norm is at most ``max_norm``.

    Returns the norm *before* clipping (as ``torch.nn.utils.clip_grad_norm_`` does).
    """
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total = float(np.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in params)))
    scale = max_norm / (total + eps)
    if scale < 1.0:
        for p in params:
            p.grad = p.grad * scale
    return total
