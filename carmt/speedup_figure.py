"""reports/figures/fig5_carmt_speedup.png from reports/carmt_speedup.json.

    python carmt/speedup_figure.py [--json reports/carmt_speedup.json] [--out reports/figures/fig5_carmt_speedup.png]

Panel A: speedup of carmt over the PyTorch/fla implementation per workload (one series, 1x reference line).
Panel B: inference throughput vs context length, one call each, batch 1 (both in constant memory; labels: device memory).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CARMT, ORIGINAL = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 (validated pair)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="reports/carmt_speedup.json")
    p.add_argument("--out", default="reports/figures/fig5_carmt_speedup.png")
    a = p.parse_args()
    r = json.load(open(a.json))
    plt.rcParams.update({"font.size": 11, "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "axes.edgecolor": GRID, "font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.25, 1]}, facecolor=SURFACE)

    # A: speedups
    sc = [s for s in r["scenarios"] if s.get("in_figure", True)]
    labels = [s["label"] for s in sc][::-1]
    vals = [s["speedup"] for s in sc][::-1]
    ys = range(len(vals))
    ax.set_facecolor(SURFACE)
    ax.barh(ys, vals, height=0.56, color=CARMT, zorder=3)
    ax.axvline(1.0, color=TEXT2, lw=1, ls=(0, (3, 3)), zorder=4)
    ax.set_ylim(-1.0, len(vals) - 0.5)
    ax.text(1.04, -0.78, "1× = original", color=TEXT2, fontsize=9, ha="left", va="center")
    for y, v in zip(ys, vals):
        ax.text(v + 0.08, y, f"{v:.2f}×", va="center", ha="left", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_yticks(list(ys), labels, fontsize=10.5, color=TEXT)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("speedup over the PyTorch / fla implementation")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("A   Speed", loc="left", fontsize=13, fontweight="bold", color=TEXT)

    # B: throughput vs context length (the original: one call where it fits, streamed in 64k pieces beyond)
    lc = r["long_context"]
    bx.set_facecolor(SURFACE)
    xs = [row["tokens"] for row in lc]
    bx.plot(xs, [row["carmt_tok_s"] / 1e3 for row in lc], color=CARMT, lw=2, marker="o", ms=8, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=2, label="carmt (CUDA): %.2f–%.2f GB" % (min(r_["carmt_gb"] for r_ in lc), max(r_["carmt_gb"] for r_ in lc)))
    orig = [(row["tokens"], (row["original_tok_s"] or row.get("original_streamed_tok_s")) / 1e3, row["original_tok_s"] is None)
            for row in lc if row["original_tok_s"] or row.get("original_streamed_tok_s")]
    bx.plot([o[0] for o in orig], [o[1] for o in orig], color=ORIGINAL, lw=2, marker="o", ms=8, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=2,
            label="original (PyTorch / fla): %.2f–%.2f GB" % (min(r_["original_gb"] for r_ in lc if r_["original_gb"]),
                                                             max(r_["original_gb"] for r_ in lc if r_["original_gb"])))
    for x, y, streamed in orig:
        if streamed:
            bx.annotate("streamed in\n64k pieces", (x, y), textcoords="offset points", xytext=(0, 12), ha="center",
                        color=TEXT2, fontsize=9)
    row = lc[-1]
    bx.annotate(f"{row['carmt_tok_s'] / row['original_tok_s']:.1f}×", (row["tokens"], row["carmt_tok_s"] / 1e3),
                textcoords="offset points", xytext=(0, 10), ha="center", color=TEXT, fontsize=10, fontweight="bold")
    bx.set_xscale("log")
    bx.set_xticks(xs, [f"{x / 1e6:.2g}M" if x >= 1e6 else f"{x // 1024}k" for x in xs])
    bx.minorticks_off()
    bx.set_ylim(0, max(row["carmt_tok_s"] for row in lc) / 1e3 * 1.25)
    bx.set_xlabel("context length (tokens in one call, batch 1)")
    bx.set_ylabel("throughput (k tokens / s)")
    bx.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    leg = bx.legend(loc="center right", frameon=False, fontsize=10, title="device memory of the call", title_fontsize=9,
                    alignment="left")
    leg.get_title().set_color(TEXT2)
    for t in leg.get_texts():
        t.set_color(TEXT)
    bx.set_title("B   Long-context inference", loc="left", fontsize=13, fontweight="bold", color=TEXT)

    fig.suptitle(r["figure_caption"], x=0.01, ha="left", y=0.995, fontsize=10, color=TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
