"""Learning-rate schedules (closed-form, evaluated at the current step count)."""

from __future__ import annotations

import math
from typing import Callable

from .optimizers import Optimizer


class LRScheduler:
    """Base class: call :meth:`step` once per epoch (or per iteration) after ``optimizer.step()``."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_step = 0
        self._apply()

    def get_lr(self, base_lr: float, step: int) -> float:  # pragma: no cover
        raise NotImplementedError

    def _apply(self) -> None:
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.get_lr(base, self.last_step)

    def step(self) -> None:
        self.last_step += 1
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer)

    def get_lr(self, base_lr: float, step: int) -> float:
        return base_lr * self.gamma ** (step // self.step_size)


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float) -> None:
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self, base_lr: float, step: int) -> float:
        return base_lr * self.gamma**step


class CosineAnnealingLR(LRScheduler):
    """``eta_min + (base - eta_min) * (1 + cos(pi * t / T_max)) / 2``."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer)

    def get_lr(self, base_lr: float, step: int) -> float:
        t = min(step, self.T_max)
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2


class LambdaLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda: Callable[[int], float]) -> None:
        self.lr_lambda = lr_lambda
        super().__init__(optimizer)

    def get_lr(self, base_lr: float, step: int) -> float:
        return base_lr * self.lr_lambda(step)


class WarmupCosineLR(LRScheduler):
    """Linear warm-up for ``warmup_steps`` then cosine decay to ``min_ratio * base`` at ``total_steps``."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int,
                 min_ratio: float = 0.1) -> None:
        self.warmup_steps, self.total_steps, self.min_ratio = warmup_steps, total_steps, min_ratio
        super().__init__(optimizer)

    def get_lr(self, base_lr: float, step: int) -> float:
        if step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return base_lr * (self.min_ratio + (1 - self.min_ratio) * cos)
