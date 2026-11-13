"""Index the parquet files of the long-math sources (file -> number of row groups) into
recurrent_armt/math_sources.json, so that the stream needs no remote listing at start-up.

    python scripts/source_meta.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

OUT = Path(__file__).resolve().parents[1] / "recurrent_armt" / "math_sources.json"
SOURCES = {
    # arXiv papers (LaTeX source) with an AutoMathText math score >= 0.6, the Hub's parquet conversion
    "arxiv": dict(repo="math-ai/AutoMathText", revision="refs/convert/parquet", prefix="arxiv-0.60-to-1.00/", columns=["text"]),
    # long chain-of-thought solutions of competition problems (DeepSeek-R1 / QwQ), problem + solution
    "omr": dict(repo="nvidia/OpenMathReasoning", revision="main", prefix="data/cot-", columns=["problem", "generated_solution"]),
}


def main():
    api, fs = HfApi(), HfFileSystem()
    out = {}
    for name, s in SOURCES.items():
        files = sorted(f.path for f in api.list_repo_tree(s["repo"], repo_type="dataset", revision=s["revision"], recursive=True)
                       if f.path.startswith(s["prefix"]) and f.path.endswith(".parquet"))
        rev = "@~parquet" if s["revision"] == "refs/convert/parquet" else ""
        repo = s["repo"] + rev

        def count(f):
            with fs.open(f"datasets/{repo}/{f}") as fh:
                md = pq.ParquetFile(fh).metadata
                return md.num_row_groups, md.num_rows

        with ThreadPoolExecutor(16) as ex:
            counts = list(ex.map(count, files))
        out[name] = {"repo": repo, "columns": s["columns"], "val_file": files[-1],
                     "num_row_groups": {f: g for f, (g, _) in zip(files, counts)},
                     "num_rows": sum(r for _, r in counts)}
        print(name, len(files), "files", sum(g for g, _ in counts), "row groups", out[name]["num_rows"], "rows; held out", files[-1])
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
