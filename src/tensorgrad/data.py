"""Minimal ``Dataset`` / ``DataLoader`` with vectorised batching."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

import numpy as np


class Dataset:
    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, i: int) -> Any:  # pragma: no cover
        raise NotImplementedError


class ArrayDataset(Dataset):
    """Wraps equally sized arrays; supports fast batched indexing via ``__getitems__``.

    An optional ``transform`` is applied to each *batch* of the first array
    (e.g. normalisation or augmentation).
    """

    def __init__(self, *arrays: np.ndarray, transform: Callable[[np.ndarray], np.ndarray] | None = None):
        if not arrays:
            raise ValueError("need at least one array")
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("all arrays must have the same length")
        self.arrays, self.transform = arrays, transform

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, i: int) -> tuple[np.ndarray, ...]:
        return self.__getitems__(np.asarray([i]))

    def __getitems__(self, idx: np.ndarray) -> tuple[np.ndarray, ...]:
        out = [a[idx] for a in self.arrays]
        if self.transform is not None:
            out[0] = self.transform(out[0])
        return tuple(out)


def default_collate(items: Sequence[Any]) -> Any:
    first = items[0]
    if isinstance(first, tuple):
        return tuple(default_collate([it[k] for it in items]) for k in range(len(first)))
    return np.stack([np.asarray(it) for it in items])


class DataLoader:
    """Iterate over a dataset in (optionally shuffled) mini-batches.

    Shuffling uses a dedicated ``np.random.Generator`` seeded by ``seed`` so
    the batch order is reproducible and independent of model RNG use.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False, seed: int | None = 0,
                 collate_fn: Callable[[Sequence[Any]], Any] = default_collate) -> None:
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.drop_last = shuffle, drop_last
        self.rng = np.random.default_rng(seed)
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        batched = hasattr(self.dataset, "__getitems__")
        for start in range(0, stop, self.batch_size):
            idx = order[start : start + self.batch_size]
            if batched:
                yield self.dataset.__getitems__(idx)
            else:
                yield self.collate_fn([self.dataset[int(i)] for i in idx])
