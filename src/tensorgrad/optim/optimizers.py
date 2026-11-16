"""First-order optimisers with PyTorch-compatible update rules and param groups."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..tensor import Tensor


class Optimizer:
    """Base class. ``params`` is an iterable of tensors or of dicts (param groups).

    Each group is a dict with a ``"params"`` list plus hyper-parameters; any
    hyper-parameter missing from a group falls back to the constructor default.
    """

    def __init__(self, params: Iterable[Any], defaults: dict[str, Any]) -> None:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        groups = params if isinstance(params[0], dict) else [{"params": params}]
        self.defaults = defaults
        self.param_groups: list[dict[str, Any]] = []
        for g in groups:
            group = {**defaults, **g, "params": list(g["params"])}
            self.param_groups.append(group)
        self.state: dict[int, dict[str, Any]] = {}

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._update(p, p.grad, self.state.setdefault(id(p), {}), group)

    def _update(self, p: Tensor, g: np.ndarray, state: dict[str, Any], group: dict[str, Any]) -> None:
        raise NotImplementedError

    @property
    def lr(self) -> float:
        return self.param_groups[0]["lr"]


class SGD(Optimizer):
    """SGD with optional (Nesterov) momentum and L2 weight decay.

    ``b <- mu * b + g``; plain momentum steps along ``b``, Nesterov along
    ``g + mu * b`` (Sutskever et al., 2013 formulation used by PyTorch).
    """

    def __init__(self, params: Iterable[Any], lr: float = 1e-2, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0) -> None:
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay))

    def _update(self, p: Tensor, g: np.ndarray, state: dict[str, Any], group: dict[str, Any]) -> None:
        if group["weight_decay"]:
            g = g + group["weight_decay"] * p.data
        mu = group["momentum"]
        if mu:
            buf = state.get("momentum_buffer")
            buf = g.copy() if buf is None else mu * buf + g
            state["momentum_buffer"] = buf
            g = g + mu * buf if group["nesterov"] else buf
        p.data = p.data - group["lr"] * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015). ``weight_decay`` here is coupled L2 regularisation."""

    def __init__(self, params: Iterable[Any], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    decoupled = False

    def _update(self, p: Tensor, g: np.ndarray, state: dict[str, Any], group: dict[str, Any]) -> None:
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        if wd and not self.decoupled:
            g = g + wd * p.data
        if not state:
            state["step"] = 0
            state["m"] = np.zeros_like(p.data)
            state["v"] = np.zeros_like(p.data)
        state["step"] += 1
        t = state["step"]
        m = state["m"] = b1 * state["m"] + (1 - b1) * g
        v = state["v"] = b2 * state["v"] + (1 - b2) * (g * g)
        data = p.data
        if wd and self.decoupled:
            data = data * (1 - lr * wd)
        bc1, bc2 = 1 - b1**t, 1 - b2**t
        denom = np.sqrt(v) / np.sqrt(bc2) + eps
        p.data = (data - (lr / bc1) * m / denom).astype(p.dtype, copy=False)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""

    decoupled = True

    def __init__(self, params: Iterable[Any], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
