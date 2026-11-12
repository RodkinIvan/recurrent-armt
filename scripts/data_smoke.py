"""Real data stream: throughput, chunk statistics, the document cooldown on real chunks, validation windows."""
import argparse
import json
import time

import numpy as np

from recurrent_armt import data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="fla-hub/rwkv7-0.4B-g1a")
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--sep", type=int, default=0)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--out", default="results/data_smoke.json")
    a = p.parse_args()

    s = data.FinewebStream(a.tokenizer, a.window, a.sep, rank=0, world_size=2, seed=0, units_per_chunk=16, cooldown=64, num_proc=a.num_proc)
    t = time.time()
    windows, stats = s.build_chunk(0)
    build = time.time() - t
    units = s.chunk_units(0)
    texts = [x for u in units for x in data.read_unit(s.repo, *u)]
    docs = s.tokenizer(texts)
    w, first, last = data.pack(docs, a.window, a.sep)
    order, dropped = data.constrained_order(first, last, 64, np.random.default_rng([0, 0, 0, 1]))
    pos = np.empty(len(w), dtype=np.int64)
    pos[order] = np.arange(len(order))
    kept = np.zeros(len(w), dtype=bool)
    kept[order] = True
    min_gap = min((abs(pos[i] - pos[j]) for i in range(len(w)) if kept[i] for j in range(i + 1, min(len(w), i + 40))
                   if kept[j] and max(first[i], first[j]) <= min(last[i], last[j])), default=None)
    lengths = np.array([len(d) for d in docs])
    t = time.time()
    val = data.validation_windows(a.tokenizer, a.window, a.sep, 64, num_proc=a.num_proc)
    val_long = data.validation_windows(a.tokenizer, a.window, a.sep, 32, long_docs=True, num_proc=a.num_proc)
    res = {"chunk_build_s": round(build, 1), "tokens_per_s": round(stats["tokens"] / build), **stats,
           "docs_mean_tokens": float(lengths.mean()), "docs_median_tokens": float(np.median(lengths)),
           "frac_docs_ge_window": float((lengths >= a.window).mean()), "frac_tokens_in_docs_ge_window": float(lengths[lengths >= a.window].sum() / lengths.sum()),
           "windows_sharing_a_doc_with_next": int(sum(max(first[i], first[i + 1]) <= min(last[i], last[i + 1]) for i in range(len(w) - 1))),
           "min_stream_gap_between_windows_of_one_doc": None if min_gap is None else int(min_gap),
           "val_windows": list(val.shape), "val_long_windows": list(val_long.shape), "val_build_s": round(time.time() - t, 1)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    s.close()


if __name__ == "__main__":     # (a spawn-context pool would re-import this module)
    main()
