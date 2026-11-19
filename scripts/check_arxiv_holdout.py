"""Does the held-out arXiv file share papers with the training files? AutoMathText's arXiv can hold several versions of
one paper in different files, so a file-level hold-out does not guarantee unseen papers.

    python scripts/check_arxiv_holdout.py [--out results/arxiv_holdout_overlap.json]

Every document gets three fingerprints: its \\title{...}, its abstract (first 400 characters) and its first 2000
characters, each lower-cased with whitespace collapsed (a title / abstract under 20 characters is skipped). The script
counts held-out documents with a fingerprint that also occurs in a training file, for the whole file and for the long
documents that ``val_long_arxiv`` is built from (>= ``--min_chars``; the first ``--n_long`` of them, in file order).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_armt.data import SOURCES  # noqa: E402

TITLE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{(.{20,300}?)\}", re.S)
ABSTRACT = re.compile(r"\\begin\{abstract\}(.{20,}?)\\end\{abstract\}", re.S)


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def fingerprints(text: str) -> dict[str, str]:
    h = lambda s: hashlib.md5(norm(s).encode()).hexdigest()
    out = {"head": h(text[:2000])}
    if m := TITLE.search(text):
        out["title"] = h(m.group(1))
    if m := ABSTRACT.search(text):
        out["abstract"] = h(norm(m.group(1))[:400])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/arxiv_holdout_overlap.json")
    p.add_argument("--min_chars", type=int, default=30_000, help="~8k tokens: the long documents of val_long_arxiv")
    p.add_argument("--n_long", type=int, default=64)
    a = p.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    src = SOURCES["arxiv"]
    repo, _, revision = src.repo.partition("@")
    revision = {"~parquet": "refs/convert/parquet"}.get(revision, revision)        # HfFileSystem's alias
    # whole files, one request each (row-group range reads trip the Hub rate limit for the training jobs too)
    texts = lambda f: pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=revision or None),
                                    columns=["text"]).column("text").to_pylist()
    val = texts(src.val_file)
    print(f"held out: {len(val)} documents of {src.val_file}", flush=True)
    seen = {"head": set(), "title": set(), "abstract": set()}
    n_train = 0
    for f, _ in src.files:
        if f == src.val_file:
            continue
        for t in texts(f):
            for key, fp in fingerprints(t).items():
                seen[key].add(fp)
            n_train += 1
        print(f"read {f}: {n_train} training documents so far", flush=True)
    long_idx = [i for i, t in enumerate(val) if len(t) >= a.min_chars][:a.n_long]
    report = {"held_out_file": src.val_file, "held_out_docs": len(val), "train_docs": n_train}
    for name, idx in (("all", range(len(val))), ("long", long_idx)):
        fps = [fingerprints(val[i]) for i in idx]
        r = {"docs": len(fps)}
        for key in seen:
            have = [f for f in fps if key in f]
            r[f"{key}_present"] = len(have)
            r[f"{key}_in_train"] = sum(f[key] in seen[key] for f in have)
        r["any_in_train"] = sum(any(f[k] in seen[k] for k in f) for f in fps)
        r["any_in_train_docs"] = [i for i, f in zip(idx, fps) if any(f[k] in seen[k] for k in f)][:50]
        report[name] = r
    print(json.dumps(report, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
