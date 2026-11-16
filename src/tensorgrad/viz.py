"""Render the dynamic computation graph with Graphviz."""

from __future__ import annotations

from typing import Any

import numpy as np

from .tensor import Tensor, iter_graph

_PALETTE = {
    "param": "#cfe3f7",     # learnable leaf
    "input": "#eeeeee",     # constant leaf (data, masks, ...)
    "inter": "#ffffff",     # intermediate result
    "root": "#f8d9b6",      # output the graph was drawn from
    "op": "#3b6ea5",
    "edge": "#555555",
}


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _tensor_label(t: Tensor, show_values: bool) -> str:
    parts = []
    if t.name:
        parts.append(f"<b>{t.name}</b>")
    if t.size == 1 and show_values:
        parts.append(f"data {_fmt(float(t.data.reshape(-1)[0]))}")
        if t.grad is not None:
            parts.append(f"grad {_fmt(float(np.asarray(t.grad).reshape(-1)[0]))}")
    else:
        parts.append(f"{tuple(t.shape)}" if t.shape else "scalar")
        if t.grad is not None and show_values:
            parts.append(f"|grad| {_fmt(float(np.linalg.norm(t.grad)))}")
    return "<" + "<br/>".join(parts) + ">"


def draw_graph(root: Tensor, rankdir: str = "LR", show_values: bool = True,
               hide_constants: bool = False, title: str | None = None) -> Any:
    """Return a ``graphviz.Digraph`` of every op and tensor reachable from ``root``.

    Learnable leaves are blue, constants grey, the root orange; ops are the
    dark ellipses. Call ``.render(path, format="png")`` on the result.
    """
    import graphviz

    dot = graphviz.Digraph(graph_attr={"rankdir": rankdir, "bgcolor": "white", "nodesep": "0.25",
                                       "ranksep": "0.35", "fontname": "Helvetica"},
                           node_attr={"fontname": "Helvetica", "fontsize": "11"},
                           edge_attr={"color": _PALETTE["edge"], "arrowsize": "0.6"})
    if title:
        dot.attr(label=title, labelloc="t", fontsize="16")

    for t in iter_graph(root):
        is_const = t._ctx is None and not t.requires_grad
        if hide_constants and is_const:
            continue
        kind = ("root" if t is root else "param" if (t._ctx is None and t.requires_grad)
                else "input" if is_const else "inter")
        dot.node(str(id(t)), _tensor_label(t, show_values), shape="box", style="rounded,filled",
                 fillcolor=_PALETTE[kind], color="#888888")
        if t._ctx is not None:
            op_id = f"op{id(t._ctx)}"
            dot.node(op_id, t._ctx.name, shape="ellipse", style="filled", fillcolor=_PALETTE["op"],
                     fontcolor="white", color=_PALETTE["op"], fontsize="10", height="0.3")
            dot.edge(op_id, str(id(t)))
            for parent in t._ctx.inputs:
                if hide_constants and parent._ctx is None and not parent.requires_grad:
                    continue
                dot.edge(str(id(parent)), op_id)
    return dot
