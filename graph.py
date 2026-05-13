"""
graph.py — Vẽ đồ thị tính toán (Computational Graph)
Minh họa đồ thị cho: y_hat = w*x + b, L = mean(0.5*(y_hat-y)^2)
Hiển thị giá trị Forward Pass và gradient Backward Pass trên từng cạnh.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def _node(ax, x, y, label, sublabel="", color="#E1F5EE", textcolor="#085041",
          radius=0.38, fontsize=11):
    """Vẽ một node (hình tròn) với nhãn và giá trị bên trong."""
    circle = plt.Circle((x, y), radius, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y + (0.08 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel,
                ha="center", va="center", fontsize=8,
                color=textcolor, zorder=4)


def _arrow(ax, x1, y1, x2, y2, color="#888780"):
    """Vẽ mũi tên có hướng giữa hai node."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=16),
                zorder=2)


def _edge_label(ax, x, y, fwd_text, bwd_text, fwd_color="#085041", bwd_color="#993C1D"):
    """Hiển thị giá trị forward pass (trên) và gradient backward pass (dưới) trên một cạnh."""
    ax.text(x, y + 0.12, fwd_text, ha="center", va="center",
            fontsize=8, color=fwd_color,
            bbox=dict(boxstyle="round,pad=0.15", fc="#E1F5EE", ec="none"))
    ax.text(x, y - 0.12, bwd_text, ha="center", va="center",
            fontsize=8, color=bwd_color,
            bbox=dict(boxstyle="round,pad=0.15", fc="#FAECE7", ec="none"))


def draw_graph(w, b, x_sample, y_sample, dw, db, figsize=(13, 5)):
    """
    Vẽ toàn bộ đồ thị tính toán với các giá trị được chú thích.

    Tham số
    -------
    w, b        : trọng số và bias đã học được
    x_sample    : một mẫu điểm giữa kỳ (float)
    y_sample    : điểm cuối kỳ thực tế của mẫu đó (float)
    dw, db      : gradient tại mẫu đó
    """
    y_hat = w * x_sample + b
    error = y_hat - y_sample
    L = 0.5 * error ** 2

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAF8")

    # ── Node positions ────────────────────────────────────────────────
    #  x(0,2.5) w(0,1.0)  →  ×(2,1.75)  →  +(4,1.75)  →  -(6,1.75)  →  ½e²(8,1.75)  →  L(10,1.75)
    #                    b(2,0.5) ──────────────────/
    #                                   y(6,0.5) ─────/

    NODE_COLOR_INPUT = "#E1F5EE"
    NODE_COLOR_OP    = "#EEEDFE"
    NODE_COLOR_OUT   = "#FAECE7"
    TEXT_INPUT       = "#085041"
    TEXT_OP          = "#3C3489"
    TEXT_OUT         = "#712B13"

    # Input nodes
    _node(ax, 0, 2.5, "x",  f"{x_sample:.2f}", NODE_COLOR_INPUT, TEXT_INPUT)
    _node(ax, 0, 1.0, "w",  f"{w:.3f}",        NODE_COLOR_INPUT, TEXT_INPUT)
    _node(ax, 2, 0.3, "b",  f"{b:.3f}",        NODE_COLOR_INPUT, TEXT_INPUT)
    _node(ax, 6, 0.3, "y",  f"{y_sample:.2f}", NODE_COLOR_INPUT, TEXT_INPUT)

    # Operation nodes
    _node(ax, 2, 1.75, "×",  "", NODE_COLOR_OP, TEXT_OP)
    _node(ax, 4, 1.75, "+",  "", NODE_COLOR_OP, TEXT_OP)
    _node(ax, 6, 1.75, "−",  "", NODE_COLOR_OP, TEXT_OP)
    _node(ax, 8, 1.75, "½e²","", NODE_COLOR_OP, TEXT_OP, fontsize=9)

    # Output node
    _node(ax, 10, 1.75, "L", f"{L:.4f}", NODE_COLOR_OUT, TEXT_OUT)

    # ── Arrows ────────────────────────────────────────────────────────
    _arrow(ax, 0.38, 2.35, 1.62, 1.95)   # x  → ×
    _arrow(ax, 0.38, 1.10, 1.62, 1.60)   # w  → ×
    _arrow(ax, 2.38, 1.75, 3.62, 1.75)   # ×  → +
    _arrow(ax, 2.00, 0.68, 3.62, 1.60)   # b  → +
    _arrow(ax, 4.38, 1.75, 5.62, 1.75)   # +  → −
    _arrow(ax, 6.00, 0.68, 5.70, 1.42)   # y  → −
    _arrow(ax, 6.38, 1.75, 7.62, 1.75)   # −  → ½e²
    _arrow(ax, 8.38, 1.75, 9.62, 1.75)   # ½e² → L

    # ── Edge labels (forward value / backward gradient) ───────────────
    wx = w * x_sample
    _edge_label(ax, 3.0, 1.90, f"u={wx:.3f}", f"∂={error:.3f}")
    _edge_label(ax, 5.0, 1.90, f"ŷ={y_hat:.3f}", f"∂={error:.3f}")
    _edge_label(ax, 7.0, 1.90, f"e={error:.3f}", f"∂={error:.3f}")
    _edge_label(ax, 9.0, 1.90, f"L={L:.4f}", "∂=1.0")

    # ── Legend ────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=NODE_COLOR_INPUT, label="Input node"),
        mpatches.Patch(color=NODE_COLOR_OP,    label="Operation node"),
        mpatches.Patch(color=NODE_COLOR_OUT,   label="Output node"),
        mpatches.Patch(color="#E1F5EE", label=f"→ Forward: value"),
        mpatches.Patch(color="#FAECE7", label=f"← Backward: gradient"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8,
              framealpha=0.9, ncol=2)

    # ── Gradient summary box ──────────────────────────────────────────
    summary = f"∂L/∂w = {dw:.4f}   ∂L/∂b = {db:.4f}"
    ax.text(5.0, -0.25, summary, ha="center", va="center",
            fontsize=10, color="#3C3489",
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEEDFE", ec="#534AB7", lw=0.8))

    ax.set_title("Computational Graph — Forward & Backward Pass",
                 fontsize=13, fontweight="500", pad=10, color="#2C2C2A")

    plt.tight_layout()
    return fig
