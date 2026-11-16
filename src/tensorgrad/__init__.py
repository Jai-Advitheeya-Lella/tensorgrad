"""tensorgrad: a small deep-learning framework written from scratch in NumPy."""

from . import data, datasets, functional, nn, optim
from .serialization import load, save
from .tensor import (
    Function,
    Tensor,
    enable_grad,
    get_default_dtype,
    is_grad_enabled,
    manual_seed,
    no_grad,
    set_default_dtype,
    tensor,
)

F = functional
concat = functional.concat
stack = functional.stack
where = functional.where

__version__ = "0.1.0"

__all__ = [
    "Tensor", "Function", "tensor", "no_grad", "enable_grad", "is_grad_enabled", "manual_seed",
    "set_default_dtype", "get_default_dtype", "nn", "optim", "functional", "F", "data", "datasets",
    "save", "load", "concat", "stack", "where",
]
