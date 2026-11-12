"""Plot train / validation curves of runs (outputs/<run>/log.jsonl): total loss and per-segment losses."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if "train/loss" in r], [r for r in rows if "val/loss" in r]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="outputs/<run> directories")
    p.add_argument("--out", default="results/curves.png")
    a = p.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    for run in a.runs:
        train, val = load(Path(run) / "log.jsonl")
        name = Path(run).name
        ax[0].plot([r["step"] for r in train], [r["train/loss"] for r in train], label=name, alpha=0.8)
        for k, axis in (("val", ax[1]), ("val_long", ax[2]), ("val_long_reset", ax[3])):
            rows = [r for r in val if f"{k}/loss" in r]
            if rows:
                last = rows[-1]
                segs = sorted(int(x.split("seg")[1]) for x in last if x.startswith(f"{k}/seg"))
                axis.plot(segs, [rows[0][f"{k}/seg{s}"] for s in segs], ls=":", color="gray")
                axis.plot(segs, [last[f"{k}/seg{s}"] for s in segs], marker="o", label=f"{name} @ {last['step']}")
    ax[0].set(title="train loss", xlabel="step")
    ax[1].set(title="validation (packed): loss per segment (gray: step 0 = base model)", xlabel="segment")
    ax[2].set(title="validation (long docs): loss per segment", xlabel="segment")
    ax[3].set(title="long docs, recurrent state reset at every boundary (memory-only context)", xlabel="segment")
    for axis in ax:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(a.out)


if __name__ == "__main__":
    main()
