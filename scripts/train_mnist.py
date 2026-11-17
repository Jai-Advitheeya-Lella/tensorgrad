"""Train an MLP or CNN on MNIST / Fashion-MNIST with tensorgrad.

Example:
    python scripts/train_mnist.py --dataset mnist --model cnn --epochs 3
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
from tensorgrad import F, Tensor, optim
from tensorgrad.data import ArrayDataset, DataLoader
from tensorgrad.datasets import load_mnist
from tensorgrad.models import MLP, ConvNet


def evaluate(model: tg.nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int = 1000) -> tuple[float, float]:
    model.eval()
    losses, correct = [], 0
    with tg.no_grad():
        for i in range(0, len(x), batch_size):
            logits = model(Tensor(x[i : i + batch_size]))
            losses.append(F.cross_entropy(logits, y[i : i + batch_size], reduction="sum").item())
            correct += int((logits.data.argmax(1) == y[i : i + batch_size]).sum())
    model.train()
    return sum(losses) / len(x), correct / len(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    ap.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="peak LR (default: 2e-3 cnn, 1e-3 mlp)")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--width", type=int, default=32, help="CNN base channel width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    tg.manual_seed(args.seed)
    xtr, ytr, xte, yte = load_mnist(args.data_dir, args.dataset)
    mean, std = xtr.mean() / 255.0, xtr.std() / 255.0
    prep = lambda a: ((a.astype(np.float32) / 255.0 - mean) / std)[:, None, :, :]  # noqa: E731
    xtr, xte = prep(xtr), prep(xte)

    model = ConvNet(width=args.width) if args.model == "cnn" else MLP()
    lr = args.lr or (2e-3 if args.model == "cnn" else 1e-3)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                       {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    loader = DataLoader(ArrayDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, seed=args.seed)
    total = args.epochs * len(loader)
    sched = optim.WarmupCosineLR(opt, warmup_steps=min(200, total // 10), total_steps=total, min_ratio=0.01)

    name = f"{args.dataset}_{args.model}"
    print(f"{name}: {model.num_parameters():,} params, {total} steps")
    log: dict = {"config": vars(args), "params": model.num_parameters(), "steps": [], "epochs": []}
    step, t0, run_loss, run_acc, n_seen = 0, time.perf_counter(), 0.0, 0.0, 0
    for epoch in range(1, args.epochs + 1):
        te = time.perf_counter()
        for xb, yb in loader:
            logits = model(Tensor(xb))
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_acc += float((logits.data.argmax(1) == yb).mean())
            n_seen += 1
            if step % args.log_every == 0:
                log["steps"].append({"step": step, "loss": run_loss / n_seen, "acc": run_acc / n_seen,
                                     "lr": opt.lr, "time": time.perf_counter() - t0})
                run_loss, run_acc, n_seen = 0.0, 0.0, 0
        test_loss, test_acc = evaluate(model, xte, yte)
        ep = {"epoch": epoch, "test_loss": test_loss, "test_acc": test_acc,
              "epoch_time": time.perf_counter() - te, "step": step}
        log["epochs"].append(ep)
        print(f"epoch {epoch}: test_loss {test_loss:.4f} test_acc {test_acc * 100:.2f}% "
              f"({ep['epoch_time']:.0f}s)", flush=True)

    log["total_time"] = time.perf_counter() - t0
    log["final_test_acc"] = log["epochs"][-1]["test_acc"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(log, indent=1))
    tg.save(model.state_dict(), Path("runs") / f"{name}.npz")
    print(f"done in {log['total_time']:.0f}s -> {out / (name + '.json')}")


if __name__ == "__main__":
    main()
