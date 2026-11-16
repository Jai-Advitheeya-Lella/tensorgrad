"""Core :class:`Tensor` type and the reverse-mode autodiff engine.

Every differentiable operation is a :class:`Function` subclass with a NumPy
``forward`` and a NumPy ``backward``. Calling ``Function.apply`` runs the
forward pass and, when gradients are required, records the function instance
on the output tensor (``out._ctx``). The recorded functions form a dynamic
DAG that :meth:`Tensor.backward` walks in reverse topological order.

Broadcasting is handled once, centrally: whenever a backward function returns
a gradient whose shape differs from its input (because the input was
broadcast in the forward pass), the engine sums it back down with
:func:`unbroadcast`. Individual ops therefore never need to care about it.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import numpy as np

Shape = tuple[int, ...]
Axis = int | Sequence[int] | None

# --------------------------------------------------------------------------- #
# Global state: default dtype, grad mode and the random generator.
# --------------------------------------------------------------------------- #

_DEFAULT_DTYPE: np.dtype = np.dtype(np.float32)
_GRAD_ENABLED: bool = True
_RNG: np.random.Generator = np.random.default_rng(0)


def set_default_dtype(dtype: Any) -> None:
    """Set the floating-point dtype used when tensors are built from Python data."""
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype)


def get_default_dtype() -> np.dtype:
    return _DEFAULT_DTYPE


def manual_seed(seed: int) -> None:
    """Seed the global generator used for initialisation, ``randn`` and dropout."""
    global _RNG
    _RNG = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _RNG


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class _GradMode(contextlib.ContextDecorator):
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._prev: list[bool] = []

    def __enter__(self) -> None:
        global _GRAD_ENABLED
        self._prev.append(_GRAD_ENABLED)
        _GRAD_ENABLED = self.enabled

    def __exit__(self, *exc: object) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev.pop()


def no_grad() -> _GradMode:
    """Context manager / decorator that disables graph construction."""
    return _GradMode(False)


def enable_grad() -> _GradMode:
    return _GradMode(True)


# --------------------------------------------------------------------------- #
# Autograd machinery
# --------------------------------------------------------------------------- #


def unbroadcast(grad: np.ndarray, shape: Shape) -> np.ndarray:
    """Reduce ``grad`` to ``shape`` by summing over the broadcast dimensions.

    NumPy broadcasting (1) prepends size-1 dimensions and (2) stretches
    size-1 dimensions. The adjoint of both is a sum, which is what we do here.
    """
    if grad.shape == tuple(shape):
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return np.asarray(grad).reshape(shape)


class Function:
    """Base class for differentiable operations.

    Subclasses implement ``forward(*arrays, **kwargs) -> ndarray`` and
    ``backward(grad) -> ndarray | tuple[ndarray | None, ...]`` (one gradient per
    tensor input, ``None`` for inputs that need none). Anything the backward
    pass needs is stored on ``self`` during ``forward``.
    """

    def __init__(self, *inputs: "Tensor") -> None:
        self.inputs: tuple[Tensor, ...] = inputs
        self.needs_input_grad: tuple[bool, ...] = tuple(t.requires_grad for t in inputs)

    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> Any:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Any, **kwargs: Any) -> "Tensor":
        tensors = tuple(x if isinstance(x, Tensor) else Tensor(x) for x in inputs)
        fn = cls(*tensors)
        out = fn.forward(*(t.data for t in tensors), **kwargs)
        requires_grad = _GRAD_ENABLED and any(fn.needs_input_grad)
        return Tensor._from_op(out, fn if requires_grad else None, requires_grad)

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"<{self.name}>"


# --------------------------------------------------------------------------- #
# Tensor
# --------------------------------------------------------------------------- #


def _as_array(value: Any, dtype: np.dtype | None) -> np.ndarray:
    if isinstance(value, Tensor):
        value = value.data
    if isinstance(value, np.ndarray):
        return value if dtype is None else value.astype(dtype, copy=False)
    arr = np.asarray(value, dtype=dtype)
    if dtype is None and arr.dtype == np.float64:
        arr = arr.astype(_DEFAULT_DTYPE)
    return arr


class Tensor:
    """An n-dimensional array that records the operations applied to it.

    Args:
        data: array-like payload. NumPy arrays keep their dtype; Python floats
            and lists become the default dtype (float32 unless changed).
        requires_grad: if True, gradients w.r.t. this tensor are accumulated
            into ``.grad`` by :meth:`backward`.
        dtype: optional dtype override.
        name: optional label used by the graph visualiser.
    """

    __array_priority__ = 1000  # make `ndarray <op> Tensor` dispatch to Tensor
    __hash__ = object.__hash__

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: Any = None,
        name: str | None = None,
    ) -> None:
        self.data: np.ndarray = _as_array(data, None if dtype is None else np.dtype(dtype))
        if requires_grad and self.data.dtype.kind != "f":
            raise TypeError("only floating-point tensors can require gradients")
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = requires_grad
        self._ctx: Function | None = None
        self._retain_grad: bool = False
        self.name: str | None = name

    @classmethod
    def _from_op(cls, data: np.ndarray, ctx: Function | None, requires_grad: bool) -> "Tensor":
        t = cls.__new__(cls)
        t.data = np.asarray(data)
        t.grad = None
        t.requires_grad = requires_grad
        t._ctx = ctx
        t._retain_grad = False
        t.name = None
        return t

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def is_leaf(self) -> bool:
        return self._ctx is None

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        body = np.array2string(self.data, precision=4, separator=", ", prefix="tensor(")
        extra = ""
        if self._ctx is not None:
            extra = f", grad_fn={self._ctx.name}"
        elif self.requires_grad:
            extra = ", requires_grad=True"
        return f"tensor({body}{extra})"

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return self.data.item()

    def tolist(self) -> Any:
        return self.data.tolist()

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def astype(self, dtype: Any) -> "Tensor":
        return _ops.Cast.apply(self, dtype=np.dtype(dtype))

    def float(self) -> "Tensor":
        return self.astype(np.float32)

    def double(self) -> "Tensor":
        return self.astype(np.float64)

    def requires_grad_(self, flag: bool = True) -> "Tensor":
        self.requires_grad = flag
        return self

    def retain_grad(self) -> "Tensor":
        """Keep ``.grad`` for this non-leaf tensor after ``backward``."""
        self._retain_grad = True
        return self

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------------------ #
    # Autodiff
    # ------------------------------------------------------------------ #
    def _toposort(self) -> list["Tensor"]:
        """Post-order DFS over tensors that require grad (iterative, no recursion limit)."""
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            if node._ctx is not None:
                for parent in node._ctx.inputs:
                    if parent.requires_grad and id(parent) not in visited:
                        stack.append((parent, False))
        return order

    def backward(self, grad: Any = None) -> None:
        """Back-propagate from this tensor, accumulating into leaf ``.grad`` fields.

        Gradients accumulate across calls (like PyTorch), so call
        ``optimizer.zero_grad()`` between steps. The graph is kept, so calling
        backward twice on the same output is allowed and doubles the grads.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            seed = np.ones_like(self.data)
        else:
            seed = _as_array(grad, self.dtype)
            if seed.shape != self.shape:
                raise ValueError(f"grad shape {seed.shape} != tensor shape {self.shape}")

        grads: dict[int, np.ndarray] = {id(self): seed}
        for node in reversed(self._toposort()):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            ctx = node._ctx
            if ctx is None or node._retain_grad:
                node.grad = g.copy() if node.grad is None else node.grad + g
            if ctx is None:
                continue
            in_grads = ctx.backward(g)
            if not isinstance(in_grads, tuple):
                in_grads = (in_grads,)
            for parent, pg, needed in zip(ctx.inputs, in_grads, ctx.needs_input_grad):
                if not needed or pg is None:
                    continue
                pg = np.asarray(pg)
                if pg.shape != parent.data.shape:
                    pg = unbroadcast(pg, parent.data.shape)
                if pg.dtype != parent.data.dtype:
                    pg = pg.astype(parent.data.dtype)
                key = id(parent)
                grads[key] = grads[key] + pg if key in grads else pg

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @staticmethod
    def zeros(*shape: int, dtype: Any = None, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(_shape(shape), dtype=dtype or _DEFAULT_DTYPE), requires_grad)

    @staticmethod
    def ones(*shape: int, dtype: Any = None, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(_shape(shape), dtype=dtype or _DEFAULT_DTYPE), requires_grad)

    @staticmethod
    def full(shape: Shape, value: float, dtype: Any = None, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.full(shape, value, dtype=dtype or _DEFAULT_DTYPE), requires_grad)

    @staticmethod
    def randn(*shape: int, dtype: Any = None, requires_grad: bool = False) -> "Tensor":
        data = _RNG.standard_normal(_shape(shape)).astype(dtype or _DEFAULT_DTYPE)
        return Tensor(data, requires_grad)

    @staticmethod
    def rand(*shape: int, dtype: Any = None, requires_grad: bool = False) -> "Tensor":
        data = _RNG.random(_shape(shape)).astype(dtype or _DEFAULT_DTYPE)
        return Tensor(data, requires_grad)

    @staticmethod
    def arange(*args: int, dtype: Any = None) -> "Tensor":
        return Tensor(np.arange(*args, dtype=dtype))

    @staticmethod
    def eye(n: int, dtype: Any = None, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.eye(n, dtype=dtype or _DEFAULT_DTYPE), requires_grad)

    def zeros_like(self) -> "Tensor":
        return Tensor(np.zeros_like(self.data))

    def ones_like(self) -> "Tensor":
        return Tensor(np.ones_like(self.data))

    # ------------------------------------------------------------------ #
    # Arithmetic (operators)
    # ------------------------------------------------------------------ #
    def _lift(self, other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.dtype if self.dtype.kind == "f" else None))

    def __add__(self, other: Any) -> "Tensor":
        return _ops.Add.apply(self, self._lift(other))

    def __radd__(self, other: Any) -> "Tensor":
        return _ops.Add.apply(self._lift(other), self)

    def __sub__(self, other: Any) -> "Tensor":
        return _ops.Sub.apply(self, self._lift(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return _ops.Sub.apply(self._lift(other), self)

    def __mul__(self, other: Any) -> "Tensor":
        return _ops.Mul.apply(self, self._lift(other))

    def __rmul__(self, other: Any) -> "Tensor":
        return _ops.Mul.apply(self._lift(other), self)

    def __truediv__(self, other: Any) -> "Tensor":
        return _ops.Div.apply(self, self._lift(other))

    def __rtruediv__(self, other: Any) -> "Tensor":
        return _ops.Div.apply(self._lift(other), self)

    def __neg__(self) -> "Tensor":
        return _ops.Neg.apply(self)

    def __pow__(self, exponent: Any) -> "Tensor":
        if isinstance(exponent, Tensor):
            return _ops.Pow.apply(self, exponent)
        return _ops.PowScalar.apply(self, exponent=float(exponent))

    def __rpow__(self, base: Any) -> "Tensor":
        return _ops.Pow.apply(self._lift(base), self)

    def __matmul__(self, other: Any) -> "Tensor":
        return _ops.MatMul.apply(self, self._lift(other))

    def __rmatmul__(self, other: Any) -> "Tensor":
        return _ops.MatMul.apply(self._lift(other), self)

    def __getitem__(self, idx: Any) -> "Tensor":
        return _ops.GetItem.apply(self, idx=_unwrap_index(idx))

    # Comparisons produce non-differentiable boolean tensors.
    def __lt__(self, other: Any) -> "Tensor":
        return Tensor(self.data < _raw(other))

    def __le__(self, other: Any) -> "Tensor":
        return Tensor(self.data <= _raw(other))

    def __gt__(self, other: Any) -> "Tensor":
        return Tensor(self.data > _raw(other))

    def __ge__(self, other: Any) -> "Tensor":
        return Tensor(self.data >= _raw(other))

    def eq(self, other: Any) -> "Tensor":
        return Tensor(self.data == _raw(other))

    # ------------------------------------------------------------------ #
    # Elementwise math
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        return _ops.Exp.apply(self)

    def log(self) -> "Tensor":
        return _ops.Log.apply(self)

    def sqrt(self) -> "Tensor":
        return _ops.Sqrt.apply(self)

    def tanh(self) -> "Tensor":
        return _ops.Tanh.apply(self)

    def sigmoid(self) -> "Tensor":
        return _ops.Sigmoid.apply(self)

    def relu(self) -> "Tensor":
        return _ops.ReLU.apply(self)

    def abs(self) -> "Tensor":
        return _ops.Abs.apply(self)

    def sin(self) -> "Tensor":
        return _ops.Sin.apply(self)

    def cos(self) -> "Tensor":
        return _ops.Cos.apply(self)

    def clip(self, min: float | None = None, max: float | None = None) -> "Tensor":
        return _ops.Clip.apply(self, lo=min, hi=max)

    def maximum(self, other: Any) -> "Tensor":
        return _ops.Maximum.apply(self, self._lift(other))

    def minimum(self, other: Any) -> "Tensor":
        return _ops.Minimum.apply(self, self._lift(other))

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #
    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        return _ops.Sum.apply(self, axis=_norm_axis(axis), keepdims=keepdims)

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        return _ops.Mean.apply(self, axis=_norm_axis(axis), keepdims=keepdims)

    def max(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        return _ops.Max.apply(self, axis=_norm_axis(axis), keepdims=keepdims)

    def min(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        return -((-self).max(axis=axis, keepdims=keepdims))

    def var(self, axis: Axis = None, keepdims: bool = False, correction: int = 0) -> "Tensor":
        axis_n = _norm_axis(axis)
        n = self.size if axis_n is None else int(np.prod([self.shape[a] for a in axis_n]))
        centered = self - self.mean(axis=axis_n, keepdims=True)
        return (centered * centered).sum(axis=axis_n, keepdims=keepdims) / (n - correction)

    def std(self, axis: Axis = None, keepdims: bool = False, correction: int = 0) -> "Tensor":
        return self.var(axis=axis, keepdims=keepdims, correction=correction).sqrt()

    def logsumexp(self, axis: int = -1, keepdims: bool = False) -> "Tensor":
        return _ops.LogSumExp.apply(self, axis=axis, keepdims=keepdims)

    def softmax(self, axis: int = -1) -> "Tensor":
        return _ops.Softmax.apply(self, axis=axis)

    def log_softmax(self, axis: int = -1) -> "Tensor":
        return _ops.LogSoftmax.apply(self, axis=axis)

    def argmax(self, axis: int | None = None) -> "Tensor":
        return Tensor(self.data.argmax(axis=axis))

    # ------------------------------------------------------------------ #
    # Shape manipulation
    # ------------------------------------------------------------------ #
    def reshape(self, *shape: Any) -> "Tensor":
        return _ops.Reshape.apply(self, shape=_shape(shape))

    view = reshape

    def flatten(self, start_dim: int = 0) -> "Tensor":
        start = start_dim % max(self.ndim, 1)
        return self.reshape(*self.shape[:start], -1)

    def transpose(self, *axes: Any) -> "Tensor":
        """``x.transpose()`` reverses all axes; ``x.transpose(a, b)`` swaps two axes."""
        if not axes:
            perm = tuple(reversed(range(self.ndim)))
        elif len(axes) == 2:
            a, b = (ax % self.ndim for ax in axes)
            p = list(range(self.ndim))
            p[a], p[b] = p[b], p[a]
            perm = tuple(p)
        else:
            perm = tuple(axes)
        return _ops.Permute.apply(self, axes=perm)

    def permute(self, *axes: int) -> "Tensor":
        return _ops.Permute.apply(self, axes=_shape(axes))

    def expand(self, *shape: int) -> "Tensor":
        return _ops.Expand.apply(self, shape=_shape(shape))

    def unsqueeze(self, axis: int) -> "Tensor":
        axis = axis if axis >= 0 else axis + self.ndim + 1
        return self.reshape(*self.shape[:axis], 1, *self.shape[axis:])

    def squeeze(self, axis: int | None = None) -> "Tensor":
        if axis is None:
            return self.reshape(*(s for s in self.shape if s != 1))
        axis %= self.ndim
        if self.shape[axis] != 1:
            return self
        return self.reshape(*self.shape[:axis], *self.shape[axis + 1 :])

    def pad(self, pad_width: Sequence[tuple[int, int]], value: float = 0.0) -> "Tensor":
        return _ops.Pad.apply(self, pad_width=tuple(tuple(p) for p in pad_width), value=value)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _shape(shape: Sequence[Any]) -> Shape:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(int(s) for s in shape[0])
    return tuple(int(s) for s in shape)


def _norm_axis(axis: Axis) -> tuple[int, ...] | None:
    if axis is None:
        return None
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


def _raw(x: Any) -> Any:
    return x.data if isinstance(x, Tensor) else x


def _unwrap_index(idx: Any) -> Any:
    if isinstance(idx, Tensor):
        return idx.data
    if isinstance(idx, tuple):
        return tuple(i.data if isinstance(i, Tensor) else i for i in idx)
    return idx


def tensor(data: Any, requires_grad: bool = False, dtype: Any = None) -> Tensor:
    """Functional constructor mirroring ``torch.tensor``."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def iter_graph(root: Tensor) -> Iterator[Tensor]:
    """Yield every tensor reachable from ``root`` (including constants)."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        if t._ctx is not None:
            stack.extend(t._ctx.inputs)


from . import ops as _ops  # noqa: E402  (ops needs Tensor/Function defined first)
