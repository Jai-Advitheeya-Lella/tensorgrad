"""DataLoader behaviour and small end-to-end training runs (CPU, seconds)."""

import os

import numpy as np
import pytest

import tensorgrad as tg
from tensorgrad import F, Tensor, nn, optim
from tensorgrad.data import ArrayDataset, DataLoader, Dataset
from tensorgrad.models import ConvNet


def test_loader_covers_every_index_once():
    ds = ArrayDataset(np.arange(103), np.arange(103) * 2)
    seen = np.concatenate([x for x, _ in DataLoader(ds, batch_size=10, shuffle=True, seed=3)])
    assert sorted(seen.tolist()) == list(range(103))


def test_loader_drop_last_and_len():
    ds = ArrayDataset(np.arange(103))
    dl = DataLoader(ds, batch_size=10, drop_last=True)
    batches = list(dl)
    assert len(dl) == len(batches) == 10 and all(len(b[0]) == 10 for b in batches)
    assert len(DataLoader(ds, batch_size=10)) == 11


def test_loader_is_reproducible():
    ds = ArrayDataset(np.arange(50))
    a = [b[0].tolist() for b in DataLoader(ds, 8, shuffle=True, seed=7)]
    b = [b[0].tolist() for b in DataLoader(ds, 8, shuffle=True, seed=7)]
    assert a == b


def test_loader_generic_dataset_collate():
    class Squares(Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, i):
            return np.array([i]), i * i

    xs, ys = next(iter(DataLoader(Squares(), batch_size=5)))
    assert xs.shape == (5, 1) and ys.tolist() == [0, 1, 4, 9, 16]


def _spirals(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.random(n)) * 3 * np.pi
    cls = rng.integers(0, 2, n)
    r = t * np.where(cls, 1, -1)
    x = np.stack([r * np.cos(t), r * np.sin(t)], 1) / 10 + rng.normal(0, 0.02, (n, 2))
    return x.astype(np.float32), cls


def test_mlp_learns_two_spirals():
    tg.manual_seed(0)
    x, y = _spirals()
    model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
    opt = optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        loss = F.cross_entropy(model(Tensor(x)), y)
        loss.backward()
        opt.step()
    acc = (model(Tensor(x)).data.argmax(1) == y).mean()
    assert acc > 0.97, acc


def test_convnet_overfits_small_batch():
    tg.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 1, 28, 28)).astype(np.float32)
    y = np.arange(16) % 10
    model = ConvNet(width=8, dropout=0.0)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(40):
        opt.zero_grad()
        loss = F.cross_entropy(model(Tensor(x)), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()


@pytest.mark.skipif(bool(os.environ.get("CI")), reason="downloads MNIST")
def test_mnist_download_and_shapes():
    from tensorgrad.datasets import load_mnist

    xtr, ytr, xte, yte = load_mnist("data")
    assert xtr.shape == (60000, 28, 28) and xte.shape == (10000, 28, 28)
    assert set(np.unique(ytr)) == set(range(10))
