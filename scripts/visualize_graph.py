"""Render tensorgrad computation graphs to PNG with Graphviz.

Produces:
  assets/graph_neuron.png  - a single tanh neuron with values and gradients
  assets/graph_mlp.png     - a 2-layer MLP + cross-entropy loss on a mini-batch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import tensorgrad as tg
from tensorgrad import F, Tensor, nn
from tensorgrad.viz import draw_graph


def neuron_graph(out_dir: Path) -> None:
    f64 = np.float64
    x1, x2 = Tensor(2.0, dtype=f64, name="x1"), Tensor(0.0, dtype=f64, name="x2")
    w1 = Tensor(-3.0, dtype=f64, requires_grad=True, name="w1")
    w2 = Tensor(1.0, dtype=f64, requires_grad=True, name="w2")
    b = Tensor(6.8813735870195432, dtype=f64, requires_grad=True, name="b")
    out = (x1 * w1 + x2 * w2 + b).tanh()
    out.name = "out"
    out.backward()
    dot = draw_graph(out, rankdir="LR", title="tanh(x1*w1 + x2*w2 + b) after backward()")
    dot.render(out_dir / "graph_neuron", format="png", cleanup=True)


def mlp_graph(out_dir: Path) -> None:
    tg.manual_seed(0)

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 3)

        def forward(self, x: Tensor) -> Tensor:
            return self.fc2(F.relu(self.fc1(x)))

    model = TinyMLP().name_parameters()
    x = Tensor(np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32), name="x (batch)")
    logits = model(x)
    loss = F.cross_entropy(logits, np.array([0, 2, 1, 1, 0]))
    loss.name = "loss"
    loss.backward()
    dot = draw_graph(loss, rankdir="LR", title="2-layer MLP + cross-entropy: forward graph with |grad| after backward()")
    dot.render(out_dir / "graph_mlp", format="png", cleanup=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    neuron_graph(out)
    mlp_graph(out)
    print(f"wrote {out}/graph_neuron.png and {out}/graph_mlp.png")


if __name__ == "__main__":
    main()
