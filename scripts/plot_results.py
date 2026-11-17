"""Turn the JSON logs in results/ into the figures in assets/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def ema(x: list[float], a: float = 0.3) -> np.ndarray:
    out, m = [], None
    for v in x:
        m = v if m is None else a * v + (1 - a) * m
        out.append(m)
    return np.asarray(out)


RUNS = [("mnist_cnn", "MNIST · CNN", BLUE, "-"), ("mnist_mlp", "MNIST · MLP", ORANGE, "-"),
        ("fashion_cnn", "Fashion · CNN", BLUE, "--"), ("fashion_mlp", "Fashion · MLP", ORANGE, "--")]


def plot_mnist(res: Path, ax_loss: plt.Axes, ax_acc: plt.Axes) -> None:
    for key, label, color, ls in RUNS:
        log = load(res / f"{key}.json")
        if log is None:
            continue
        steps = log["steps"]
        n_per_epoch = log["epochs"][0]["step"]
        ep = np.array([s["step"] for s in steps]) / n_per_epoch
        ax_loss.plot(ep, ema([s["loss"] for s in steps]), color=color, ls=ls, label=label)
        e = [d["epoch"] for d in log["epochs"]]
        err = [100 * (1 - d["test_acc"]) for d in log["epochs"]]
        ax_acc.plot(e, err, color=color, ls=ls, marker="o", ms=5, label=label)
        ax_acc.annotate(f"{100 - err[-1]:.2f}%", (e[-1], err[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss (EMA)")
    ax_loss.set_title("Training loss")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_acc.set_yscale("log")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test error (%)")
    ax_acc.set_title("Test error (labels: final accuracy)")
    ax_acc.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_acc.margins(x=0.15)


def plot_charlm(res: Path, ax: plt.Axes) -> None:
    for key, label, color, dy in [("charlm_transformer", "Transformer", BLUE, -9), ("charlm_lstm", "LSTM", AQUA, 9)]:
        log = load(res / f"{key}.json")
        if log is None:
            continue
        s = [d["step"] for d in log["steps"]]
        ax.plot(s, ema([d["loss"] for d in log["steps"]], 0.2), color=color, alpha=0.35, lw=1.5)
        ev = log["evals"]
        ax.plot([d["step"] for d in ev], [d["val_loss"] for d in ev], color=color, marker="o", ms=5,
                label=f"{label} ({log['params'] / 1e3:.0f}k params) val")
        ax.annotate(f"{ev[-1]['val_loss']:.2f}", (ev[-1]["step"], ev[-1]["val_loss"]), xytext=(6, dy),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title("Tiny Shakespeare char-LM (faint: train)")
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(x=0.08)


def plot_benchmark(res: Path, out: Path) -> None:
    bench = load(res / "benchmark.json")
    if bench is None:
        return
    rows = bench["results"][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    h = 0.36
    ax.barh(y + h / 2, [r["tensorgrad_ms"] for r in rows], h, color=BLUE, label="tensorgrad (NumPy)")
    ax.barh(y - h / 2, [r["torch_ms"] for r in rows], h, color=ORANGE, label="PyTorch CPU")
    for i, r in enumerate(rows):
        ax.text(max(r["tensorgrad_ms"], r["torch_ms"]) * 1.15, i, f"{r['slowdown']:.1f}x slower",
                va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [r["case"] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("ms per training step (median, 4 threads, lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(max(r["tensorgrad_ms"], r["torch_ms"]) for r in rows) * 6)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "benchmark.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    plot_mnist(res, a, b)
    fig.tight_layout()
    fig.savefig(out / "mnist_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    plot_charlm(res, ax)
    fig.tight_layout()
    fig.savefig(out / "charlm_curves.png")
    plt.close(fig)

    plot_benchmark(res, out)

    # Hero: computation graph on top, training curves underneath.
    graph = out / "graph_mlp.png"
    if graph.exists():
        img = plt.imread(graph)
        fig = plt.figure(figsize=(14, 6.0))
        gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.25], hspace=0.12, wspace=0.28)
        ax_g = fig.add_subplot(gs[0, :])
        ax_g.imshow(img)
        ax_g.axis("off")
        plot_mnist(res, fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
        plot_charlm(res, fig.add_subplot(gs[1, 2]))
        fig.savefig(out / "hero.png", bbox_inches="tight")
        plt.close(fig)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
