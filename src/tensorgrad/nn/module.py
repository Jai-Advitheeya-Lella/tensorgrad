"""``Module`` / ``Parameter`` base classes with PyTorch-style registration."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """A tensor that is a learnable parameter of a :class:`Module`."""

    def __init__(self, data: Any, requires_grad: bool = True, name: str | None = None) -> None:
        super().__init__(data, requires_grad=requires_grad, name=name)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, dtype={self.dtype})"


class Module:
    """Base class for layers and models.

    Assigning a :class:`Parameter` or a :class:`Module` as an attribute
    registers it automatically; buffers (non-learnable state such as BatchNorm
    running statistics) are registered with :meth:`register_buffer`.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def __setattr__(self, name: str, value: Any) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError("call Module.__init__() before assigning attributes")
        for store in (self._parameters, self._modules, self._buffers):
            store.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        object.__setattr__(self, name, value)
        self._buffers[name] = value

    def add_module(self, name: str, module: "Module") -> None:
        setattr(self, name, module)

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #
    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield prefix, self
        for name, child in self._modules.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def modules(self) -> Iterator["Module"]:
        for _, m in self.named_modules():
            yield m

    def children(self) -> Iterator["Module"]:
        yield from self._modules.values()

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        for mod_name, mod in self.named_modules(prefix):
            for name, p in mod._parameters.items():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                yield (f"{mod_name}.{name}" if mod_name else name), p

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
        for mod_name, mod in self.named_modules(prefix):
            for name in mod._buffers:
                yield (f"{mod_name}.{name}" if mod_name else name), getattr(mod, name)

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def name_parameters(self) -> "Module":
        """Set ``p.name`` to the qualified parameter name (used by the graph visualiser)."""
        for name, p in self.named_parameters():
            p.name = name
        return self

    # ------------------------------------------------------------------ #
    # Modes / grads / state
    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> "OrderedDict[str, np.ndarray]":
        state: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, p in self.named_parameters():
            state[name] = p.data.copy()
        for name, b in self.named_buffers():
            state[name] = np.array(b, copy=True)
        return state

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        params = dict(self.named_parameters())
        buffers = {name: (mod, key) for mod_name, mod in self.named_modules()
                   for key in mod._buffers
                   for name in [f"{mod_name}.{key}" if mod_name else key]}
        missing = (set(params) | set(buffers)) - set(state)
        unexpected = set(state) - set(params) - set(buffers)
        if strict and (missing or unexpected):
            raise KeyError(f"state_dict mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
        for name, value in state.items():
            value = np.asarray(value)
            if name in params:
                p = params[name]
                if p.shape != value.shape:
                    raise ValueError(f"{name}: shape {value.shape} != {p.shape}")
                p.data = value.astype(p.dtype, copy=True)
            elif name in buffers:
                mod, key = buffers[name]
                buf = getattr(mod, key)
                buf[...] = value  # in place, so references held elsewhere stay valid

    # ------------------------------------------------------------------ #
    # Call / repr
    # ------------------------------------------------------------------ #
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}({self.extra_repr()}"]
        for name, child in self._modules.items():
            child_repr = repr(child).replace("\n", "\n  ")
            lines.append(f"  ({name}): {child_repr}")
        if len(lines) == 1:
            return lines[0] + ")"
        return "\n".join(lines) + "\n)"
