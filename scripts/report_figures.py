"""Figures of reports/recurrent_armt_wip.md, from outputs/<run>/log.jsonl.

    python scripts/report_figures.py --out reports/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots in fixed order (validated: scripts of the dataviz method), base model in neutral gray
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def evals(run):
    """{step: entry} of the evaluation entries (the last one when a step was evaluated twice)."""
    out = {}
    for line in open(Path("outputs") / run / "log.jsonl"):
        e = json.loads(line)
        if "val/loss" in e:
            out[e["step"]] = e
    return dict(sorted(out.items()))


def segs(entry, key):
    n = sum(1 for k in entry if k.startswith(f"{key}/seg"))
    return [entry[f"{key}/seg{i}"] for i in range(n)]


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=10, color=INK2)
    ax.plot([x], [y], "o", color=color, ms=6, mec=SURFACE, mew=1.5, zorder=5)


def fig_tie(out):
    """ARMT vs its no-memory control, per segment, long documents, step 600."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, (title, armt, ctrl, extra) in zip(axes, [
            ("RWKV-7 0.4B (annealed checkpoint)", "rwkv7_0.4b_armt", "rwkv7_0.4b_nomem", None),
            ("GDN-2 370M (not annealed, 30B tokens)", "gdn2_370m_armt", "gdn2_370m_nomem", "gdn2_370m_armt_freeze300_sd1.0")]):
        a, c = evals(armt), evals(ctrl)
        base, fa, fc = segs(a[0], "val_long"), segs(a[600], "val_long"), segs(c[600], "val_long")
        x = list(range(1, len(base) + 1))
        ax.plot(x, base, ls=(0, (2, 2)), color=GRAY, label="base model (step 0)")
        ax.plot(x, fc, color=ORANGE, marker="o", label="no memory (control)")
        ax.plot(x, fa, color=BLUE, marker="o", ms=4, label="ARMT")
        if extra:
            e = segs(evals(extra)[600], "val_long")
            ax.plot(x, e, color=AQUA, marker="s", ms=4, label="ARMT, frozen + state dropout (p=1.0)")
        gap = max(abs(p - q) for p, q in zip(fa, fc))
        ax.set(title=f"{title}\nmax |ARMT - control| per segment: {gap:.4f}", xlabel="segment (1024 tokens each)",
               ylabel="loss, long-document windows", xticks=x)
        ax.title.set_fontsize(11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    fig.suptitle("Step 600: ARMT matches its no-memory control segment by segment", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig1_armt_vs_control.png", dpi=160)


def fig_rewarm(out):
    """Validation loss over training: annealed RWKV-7 degrades, non-annealed GDN-2 improves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    panels = [
        ("RWKV-7 0.4B: annealed checkpoint", [("rwkv7_0.4b_armt", "ARMT, lr 3e-5", BLUE), ("rwkv7_0.4b_nomem", "no memory, lr 3e-5", ORANGE),
                                              ("rwkv7_0.4b_armt_freeze300", "ARMT, backbone frozen 300 steps", YELLOW),
                                              ("rwkv7_0.4b_nomem_lr3e-6", "no memory, lr 3e-6", AQUA)]),
        ("GDN-2 370M: not annealed", [("gdn2_370m_armt", "ARMT, lr 1e-4", BLUE), ("gdn2_370m_nomem", "no memory, lr 1e-4", ORANGE)]),
    ]
    for ax, (title, runs) in zip(axes, panels):
        base = None
        for run, name, color in runs:
            e = evals(run)
            steps = [s for s in e if s <= (100 if run.endswith("lr3e-6") else 600)]
            ys = [e[s]["val/loss"] for s in steps]
            base = e[0]["val/loss"]
            ax.plot(steps, ys, color=color, marker="o", ms=4, label=name)
        ax.axhline(base, color=GRAY, ls=(0, (2, 2)), lw=1.5)
        ax.text(600, base, " base", va="bottom", ha="right", color=INK2, fontsize=10)
        ax.set(title=title, xlabel="training step (262k tokens each)", ylabel="validation loss (packed windows)")
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Continued pre-training: restarting an annealed checkpoint costs loss; a mid-training one keeps improving",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_rewarm.png", dpi=160)


def fig_state_dropout(out):
    """Memory-only context (state reset at every boundary) and normal validation, state-dropout runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    runs = [("gdn2_370m_armt_freeze300_sd0.5", "state dropout p=0.5", BLUE), ("gdn2_370m_armt_freeze300_sd1.0", "state dropout p=1.0", ORANGE)]
    ax = axes[0]
    for run, name, color in runs:
        e = evals(run)
        ax.plot(list(e), [v["val_long_reset/loss"] for v in e.values()], color=color, marker="o", ms=4, label=name)
    base = evals(runs[0][0])[0]["val_long_reset/loss"]
    ax.axhline(base, color=GRAY, ls=(0, (2, 2)), lw=1.5)
    ax.text(600, base, "base model: no context across segments ", va="top", ha="right", color=INK2, fontsize=9)
    ax.set(title="Long documents, recurrent state reset at every boundary\n(only the memory carries context)",
           xlabel="training step", ylabel="loss")
    ax2 = axes[1]
    for run, name, color in runs + [("gdn2_370m_armt", "ARMT, no freeze / dropout", AQUA), ("gdn2_370m_nomem", "no memory", YELLOW)]:
        e = evals(run)
        ax2.plot(list(e), [v["val/loss"] for v in e.values()], color=color, marker="o", ms=4, label=name)
    ax2.set(title="Normal validation (recurrent state kept)", xlabel="training step", ylabel="validation loss (packed)")
    for a in axes:
        a.axvspan(0, 300, color=GRID, alpha=0.6, lw=0, zorder=0)
        lo = a.get_ylim()[0]
        a.text(150, lo, "backbone frozen, state dropout on\n", ha="center", va="bottom", fontsize=9, color=INK2)
        a.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("State dropout teaches the memory to carry context, while the backbone is frozen",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig3_state_dropout.png", dpi=160)


def fig_ablation(out):
    """How much loss the memory removes (memory off minus memory on), recurrent state intact."""
    sd5, sd1 = evals("gdn2_370m_armt_freeze300_sd0.5"), evals("gdn2_370m_armt_freeze300_sd1.0")
    rows = [  # (label, packed, long): at step 300 memory-off == the base model exactly (frozen backbone, W_mv = 0)
        ("p=0.5, step 300\n(backbone frozen)", sd5[0]["val/loss"] - sd5[300]["val/loss"], sd5[0]["val_long/loss"] - sd5[300]["val_long/loss"]),
        ("p=1.0, step 300\n(backbone frozen)", sd1[0]["val/loss"] - sd1[300]["val/loss"], sd1[0]["val_long/loss"] - sd1[300]["val_long/loss"]),
        # measured by zeroing W_mv in the final models (scripts: ablate.py runs, see the report)
        ("p=0.5, step 600\n(after unfreezing)", 0.0000, 0.0001),
        ("p=1.0, step 600\n(after unfreezing)", 0.0001, 0.0001),
        ("ARMT, no dropout\nstep 600", -0.0001, 0.0001),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    x = range(len(rows))
    w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], [r[1] for r in rows], w, color=BLUE, label="packed windows")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], [r[2] for r in rows], w, color=ORANGE, label="long-document windows")
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.annotate(f"{v:+.4f}", (bar.get_x() + bar.get_width() / 2, max(v, 0)), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(list(x), [r[0] for r in rows], fontsize=9)
    ax.set(ylabel="loss(memory off) - loss(memory on)", title="What the memory is worth with the recurrent state intact (GDN-2)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig4_memory_ablation.png", dpi=160)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/figures")
    out = Path(p.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_tie, fig_rewarm, fig_state_dropout, fig_ablation):
        f(out)
    print("figures in", out)


if __name__ == "__main__":
    main()
