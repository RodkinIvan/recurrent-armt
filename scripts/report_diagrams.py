"""Method diagram of the report: where ARMT sits in a pure-GDN model vs the Qwen3.5 attention-GDN hybrid.
Writes PNG and SVG.

    python scripts/report_diagrams.py --out reports/figures
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

STATE, MEMORY, KV = "#2e78d6", "#e8683a", "#1baf7a"           # recurrent state, associative memory, KV cache
INK, INK2, LINE, SURFACE = "#16181d", "#5a6170", "#b9bec8", "#fcfcfb"
GDN_FILL, ATT_FILL = "#e9f1fb", "#e8f6ef"

plt.rcParams.update({"font.size": 11, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "svg.fonttype": "none"})


def box(ax, x0, y0, w, h, text, fill, edge=INK2, size=10, weight="normal", lw=1.3):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fill, ec=edge, lw=lw, zorder=3))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, zorder=4,
            linespacing=1.25)


def arrow(ax, p, q, color=INK2, lw=1.6, ls="-", style="-|>", z=2, rad=0.0, scale=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=scale, color=color, lw=lw, ls=ls, zorder=z,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", capstyle="round", joinstyle="round"))


def line(ax, xs, ys, color, lw=2.2, ls="-", z=2):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def label(ax, x, y, text, ha="left", color=INK2, size=10, **kw):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, zorder=5, **kw)


def stack(ax, x0, rows, title, subtitle):
    """A model unrolled over 3 segments: rows are layers (bottom first), arrows carry what crosses the boundaries."""
    cw, ch, dx, dy, y0 = 11, 6, 17.5, 9.5, 9
    xs = [x0 + 8 + j * dx for j in range(3)]
    ys = [y0 + r * dy for r in range(len(rows))]
    top = y0 + 3 * dy + ch                 # title height of a 4-row stack, so that both panels align
    ax.text(x0, top + 9, title, fontsize=12, fontweight="bold", color=INK, va="bottom")
    ax.text(x0, top + 8.2, subtitle, fontsize=9.5, color=INK2, va="top")
    for j, x in enumerate(xs):
        arrow(ax, (x + cw / 2, y0 - 5), (x + cw / 2, y0), INK2, lw=1.3)
        label(ax, x + cw / 2, y0 - 6.5, f"segment {j + 1}", ha="center", size=9.5, color=INK)
        for r in range(len(rows) - 1):
            arrow(ax, (x + cw / 2, ys[r] + ch), (x + cw / 2, ys[r + 1]), LINE, lw=1.2)
        arrow(ax, (x + cw / 2, ys[-1] + ch), (x + cw / 2, ys[-1] + ch + 3.2), INK2, lw=1.3)
    for r, (kind, name) in enumerate(rows):
        y = ys[r]
        for j, x in enumerate(xs):
            if kind == "gdn":
                box(ax, x, y, cw, ch, name, GDN_FILL, edge=STATE, size=8.5, lw=1.5)
            else:
                box(ax, x, y, cw, ch, name, ATT_FILL, edge=KV, size=8.5, lw=1.5)
            last = j == len(xs) - 1
            x1, x2 = x + cw, (x + dx if not last else x + cw + 5)
            ls = (0, (1.5, 1.5)) if last else "-"
            if kind == "gdn":
                arrow(ax, (x1, y + ch * 0.68), (x2, y + ch * 0.68), STATE, lw=2.2, ls=ls, scale=9)
                arrow(ax, (x1, y + ch * 0.32), (x2, y + ch * 0.32), MEMORY, lw=2.2, ls=ls, scale=9)
            else:
                w = 1.6 + 2.2 * j if not last else 2.4
                arrow(ax, (x1, y + ch / 2), (x2, y + ch / 2), KV, lw=w, ls=ls, scale=9 + 2 * min(j, 1))
                if not last:
                    label(ax, (x1 + x2) / 2, y - 1.4, f"KV {j + 1}k", ha="center", size=8.5, color=KV)
    return xs, ys, cw, ch


def fig_models(out):
    """Where ARMT goes: every layer of a recurrent model; only the GDN layers of an attention-GDN hybrid."""
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set(xlim=(0, 135), ylim=(-13, 54)); ax.axis("off")

    xs, ys, cw, ch = stack(ax, 1, [("gdn", "GDN\n+ ARMT")] * 3, "(a) GDN-2 370M: every layer is recurrent",
                           "ARMT wraps all 24 layers; what crosses a boundary is fixed-size")
    ax.annotate("", (xs[0] - 2.5, ys[0]), (xs[0] - 2.5, ys[-1] + ch),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2, shrinkA=0, shrinkB=0))
    label(ax, xs[0] - 3.5, (ys[0] + ys[-1] + ch) / 2, "×8\n(24 layers)", ha="right", size=9)

    rows = [("gdn", "GDN\n+ ARMT")] * 3 + [("att", "attention\n(unchanged)")]
    xs, ys, cw, ch = stack(ax, 70, rows, "(b) Qwen3.5-0.8B hybrid: 3 GDN layers, then 1 attention layer",
                           "ARMT wraps the 18 GDN layers; the 6 attention layers keep their KV cache")
    ax.annotate("", (xs[0] - 2.5, ys[0]), (xs[0] - 2.5, ys[-1] + ch),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2, shrinkA=0, shrinkB=0))
    label(ax, xs[0] - 3.5, (ys[0] + ys[-1] + ch) / 2, "×6\n(24 layers)", ha="right", size=9)
    label(ax, xs[-1] + cw + 6, ys[-1] + ch / 2 + 0.2, "grows with\ncontext", size=8.5, color=KV)
    ax.text(70, -9.2, "In attention layers the memory tokens attend to the cache and the current segment, but are not added to the cache.",
            fontsize=9, color=INK2, va="center")

    # legend
    for x, y, color, text in [(1, -9.2, STATE, "recurrent state S (fixed size)"),
                              (31, -9.2, MEMORY, "associative memory W, z (fixed size)"),
                              (1, -11.6, KV, "KV cache: keys and values of all past tokens (grows)")]:
        line(ax, [x, x + 4], [y, y], color, lw=3)
        label(ax, x + 5, y, text, size=9, color=INK)
    fig.suptitle("Where the associative memory goes", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig0_armt_models.{ext}", dpi=160)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/figures")
    out = Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    fig_models(out)
    print("diagrams in", out)


if __name__ == "__main__":
    main()
