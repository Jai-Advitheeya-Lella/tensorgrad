"""Train a character-level transformer (or LSTM) on Tiny Shakespeare with tensorgrad.

Example:
    python scripts/train_charlm.py --model transformer --steps 2500
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import time
from pathlib import Path

import numpy as np

import tensorgrad as tg
from tensorgrad import F, nn, optim
from tensorgrad.datasets import load_tiny_shakespeare
from tensorgrad.models import CharLSTM, CharTransformer


def get_batch(data: np.ndarray, block: int, bs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ix = rng.integers(0, len(data) - block - 1, bs)
    x = np.stack([data[i : i + block] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block] for i in ix])
    return x, y


def estimate_loss(model: nn.Module, data: np.ndarray, block: int, bs: int, iters: int, seed: int) -> float:
    rng = np.random.default_rng(seed)  # same batches every evaluation
    model.eval()
    losses = []
    with tg.no_grad():
        for _ in range(iters):
            x, y = get_batch(data, block, bs, rng)
            losses.append(F.cross_entropy(model(x), y).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["transformer", "lstm"], default="transformer")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256, help="LSTM hidden size")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tg.manual_seed(args.seed)
    text = load_tiny_shakespeare(args.data_dir)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    if args.model == "transformer":
        model: nn.Module = CharTransformer(len(chars), args.block_size, args.d_model, args.n_layers,
                                           args.n_heads, args.dropout)
    else:
        model = CharLSTM(len(chars), 64, args.hidden)
    decay = [p for _, p in model.named_parameters() if p.ndim == 2]
    no_decay = [p for _, p in model.named_parameters() if p.ndim != 2]
    opt = optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                       {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.99))
    sched = optim.WarmupCosineLR(opt, warmup_steps=100, total_steps=args.steps, min_ratio=0.1)
    name = f"charlm_{args.model}"
    print(f"{name}: vocab {len(chars)}, {model.num_parameters():,} params")

    rng = np.random.default_rng(args.seed)
    log: dict = {"config": vars(args), "params": model.num_parameters(), "steps": [], "evals": []}
    t0, run = time.perf_counter(), []
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, args.block_size, args.batch_size, rng)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        nn.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 25 == 0:
            log["steps"].append({"step": step, "loss": float(np.mean(run)), "lr": opt.lr,
                                 "time": time.perf_counter() - t0})
            run = []
        if step % args.eval_every == 0 or step == args.steps:
            vl = estimate_loss(model, val, args.block_size, args.batch_size, args.eval_iters, 123)
            tl = estimate_loss(model, train, args.block_size, args.batch_size, args.eval_iters, 456)
            log["evals"].append({"step": step, "train_loss": tl, "val_loss": vl,
                                 "time": time.perf_counter() - t0})
            print(f"step {step}: train {tl:.3f} val {vl:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)

    log["total_time"] = time.perf_counter() - t0
    prompt = "ROMEO:\n"
    if isinstance(model, CharTransformer):
        idx = model.generate(np.array([[stoi[c] for c in prompt]]), 500, temperature=0.8,
                             rng=np.random.default_rng(args.seed))
        sample = "".join(chars[i] for i in idx[0])
        log["sample"] = sample
        print(sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(log, indent=1))
    tg.save(model.state_dict(), Path("runs") / f"{name}.npz")
    print(f"done in {log['total_time']:.0f}s")


if __name__ == "__main__":
    main()
