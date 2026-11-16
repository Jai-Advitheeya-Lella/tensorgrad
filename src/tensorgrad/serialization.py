"""Save / load state dicts as ``.npz`` archives."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np


def save(state: dict[str, np.ndarray], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **state)


def load(path: str | Path) -> "OrderedDict[str, np.ndarray]":
    with np.load(path) as f:
        return OrderedDict((k, f[k]) for k in f.files)
