"""Dataset downloaders (MNIST, Fashion-MNIST, Tiny Shakespeare) with no ML dependencies."""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

_MNIST_MIRRORS = {
    "mnist": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ],
    "fashion": ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"],
}
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(urls: list[str], dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tensorgrad-dataset-fetcher"})
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # try the next mirror
            last_err = e
    raise RuntimeError(f"could not download {dest.name}: {last_err}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist(root: str | Path = "data", kind: str = "mnist"
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, y_train, x_test, y_test)`` as uint8 images (N, 28, 28) and int labels.

    ``kind`` is ``"mnist"`` or ``"fashion"``. Files are cached under ``root/<kind>/``.
    """
    if kind not in _MNIST_MIRRORS:
        raise ValueError(f"unknown dataset {kind!r}")
    folder = Path(root) / kind
    arrays = {}
    for key, fname in _FILES.items():
        path = _download([m + fname for m in _MNIST_MIRRORS[kind]], folder / fname)
        arrays[key] = _read_idx(path)
    return (arrays["train_images"], arrays["train_labels"].astype(np.int64),
            arrays["test_images"], arrays["test_labels"].astype(np.int64))


def load_tiny_shakespeare(root: str | Path = "data") -> str:
    path = _download([_SHAKESPEARE_URL], Path(root) / "tinyshakespeare.txt")
    return path.read_text(encoding="utf-8")


def md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
