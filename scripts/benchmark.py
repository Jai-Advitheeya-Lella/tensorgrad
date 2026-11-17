"""Time one training step (forward + backward + optimiser) in tensorgrad vs. PyTorch on CPU.

Both frameworks run the same architectures in float32 with 4 threads. The two
are interleaved trial-by-trial so background load affects both equally.

Example:
    python scripts/benchmark.py --trials 20
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as TF

import tensorgrad as tg
from tensorgrad import F, Tensor, optim
from tensorgrad.models import MLP, CharTransformer, ConvNet

torch.set_num_threads(4)


# --------------------------------------------------------------------------- #
# PyTorch twins of the tensorgrad reference models
# --------------------------------------------------------------------------- #


def torch_mlp() -> tnn.Module:
    return tnn.Sequential(tnn.Flatten(), tnn.Linear(784, 512), tnn.ReLU(), tnn.Dropout(0.1),
                          tnn.Linear(512, 256), tnn.ReLU(), tnn.Dropout(0.1), tnn.Linear(256, 10))


def torch_convnet(w: int = 32) -> tnn.Module:
    return tnn.Sequential(
        tnn.Conv2d(1, w, 3, padding=1, bias=False), tnn.BatchNorm2d(w), tnn.ReLU(),
        tnn.Conv2d(w, w, 3, padding=1, bias=False), tnn.BatchNorm2d(w), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Conv2d(w, 2 * w, 3, padding=1, bias=False), tnn.BatchNorm2d(2 * w), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Flatten(), tnn.Dropout(0.25), tnn.Linear(2 * w * 49, 128), tnn.ReLU(), tnn.Dropout(0.25),
        tnn.Linear(128, 10))


class TorchBlock(tnn.Module):
    def __init__(self, d: int, h: int, p: float) -> None:
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = tnn.LayerNorm(d), tnn.LayerNorm(d)
        self.qkv, self.proj_attn = tnn.Linear(d, 3 * d), tnn.Linear(d, d)
        self.fc, self.proj = tnn.Linear(d, 4 * d), tnn.Linear(4 * d, d)
        self.drop = tnn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = TF.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj_attn(a.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.proj(TF.gelu(self.fc(self.ln2(x)), approximate="tanh")))


class TorchCharTransformer(tnn.Module):
    def __init__(self, vocab: int, block: int, d: int, n_layers: int, h: int, p: float) -> None:
        super().__init__()
        self.tok, self.pos = tnn.Embedding(vocab, d), tnn.Embedding(block, d)
        self.blocks = tnn.Sequential(*[TorchBlock(d, h, p) for _ in range(n_layers)])
        self.ln_f, self.head = tnn.LayerNorm(d), tnn.Linear(d, vocab, bias=False)
        self.drop = tnn.Dropout(p)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok(idx) + self.pos(torch.arange(idx.shape[1])))
        return self.head(self.ln_f(self.blocks(x)))


# --------------------------------------------------------------------------- #
# Step functions
# --------------------------------------------------------------------------- #


def tg_step_fn(model: tg.nn.Module, x: np.ndarray, y: np.ndarray, as_tensor: bool) -> Callable[[], None]:
    opt = optim.Adam(model.parameters(), 1e-3)
    inp = Tensor(x) if as_tensor else x

    def step() -> None:
        loss = F.cross_entropy(model(inp), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return step


def th_step_fn(model: tnn.Module, x: np.ndarray, y: np.ndarray) -> Callable[[], None]:
    opt = torch.optim.Adam(model.parameters(), 1e-3, foreach=False)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)

    def step() -> None:
        out = model(xt)
        loss = TF.cross_entropy(out.reshape(-1, out.shape[-1]), yt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    return step


def matmul_fns(n: int) -> tuple[Callable[[], None], Callable[[], None]]:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
    pa, pb = torch.from_numpy(a).requires_grad_(), torch.from_numpy(b).requires_grad_()

    def tg_fn() -> None:
        (ta @ tb).sum().backward()

    def th_fn() -> None:
        (pa @ pb).sum().backward()

    return tg_fn, th_fn


def time_pair(fa: Callable[[], None], fb: Callable[[], None], trials: int, warmup: int = 3
              ) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        fa()
        fb()
    ta, tb = [], []
    for _ in range(trials):
        t = time.perf_counter()
        fa()
        ta.append(time.perf_counter() - t)
        t = time.perf_counter()
        fb()
        tb.append(time.perf_counter() - t)
    return ta, tb


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()
    tg.manual_seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    img128 = rng.standard_normal((128, 1, 28, 28)).astype(np.float32)
    img64 = rng.standard_normal((64, 1, 28, 28)).astype(np.float32)
    lab128, lab64 = rng.integers(0, 10, 128), rng.integers(0, 10, 64)
    tok = rng.integers(0, 65, (32, 64))
    tok_y = rng.integers(0, 65, (32, 64))

    cases = {
        "matmul 1024x1024 fwd+bwd": matmul_fns(1024),
        "MLP 784-512-256-10, batch 128": (tg_step_fn(MLP(), img128, lab128, True),
                                          th_step_fn(torch_mlp(), img128, lab128)),
        "CNN (3 conv + BN), batch 64": (tg_step_fn(ConvNet(width=32), img64, lab64, True),
                                        th_step_fn(torch_convnet(32), img64, lab64)),
        "Char transformer 4L d128, 32x64 tokens": (
            tg_step_fn(CharTransformer(65, 64, 128, 4, 4, 0.1), tok, tok_y, False),
            th_step_fn(TorchCharTransformer(65, 64, 128, 4, 4, 0.1), tok, tok_y)),
    }
    results = []
    for name, (fa, fb) in cases.items():
        ta, tb = time_pair(fa, fb, args.trials)
        ma, mb = 1e3 * float(np.median(ta)), 1e3 * float(np.median(tb))
        results.append({"case": name, "tensorgrad_ms": ma, "torch_ms": mb, "slowdown": ma / mb})
        print(f"{name:42s} tensorgrad {ma:8.2f} ms   torch {mb:8.2f} ms   x{ma / mb:5.2f}", flush=True)

    meta = {"numpy": np.__version__, "torch": torch.__version__, "threads": 4,
            "cpu": cpu_name(), "trials": args.trials}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1))


if __name__ == "__main__":
    main()
