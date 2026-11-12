"""Streaming pre-training windows from karpathy/fineweb-edu-100b-shuffle (nothing is downloaded up front).

Units. The corpus is 1823 parquet shards x ~52 row groups x 1024 documents; a row group is the unit read
remotely (~1.8 MB, ~1M tokens). The last shard is held out for validation.

Shuffling. All training units are put in one seeded random order and dealt to the data-parallel ranks
(unit k -> rank k % world), so ranks never share a document. A rank consumes its units in chunks of
``units_per_chunk`` (default 16: 16k documents from 16 random places of the corpus, ~2k windows of 8k
tokens). The documents of a chunk are tokenized, each one prefixed with ``doc_sep_token_id`` and packed
into windows of ``window`` tokens; the windows of the chunk are emitted in a random order.
  * Packing is chunk-local (the tail of a chunk is dropped), so two chunks never share a document.
  * Inside a chunk the order is constrained: two windows holding pieces of the same document are at least
    ``cooldown`` windows apart in the stream. With cooldown >= 2 x (windows per optimizer step per rank)
    no document has pieces in the same batch or in adjacent batches (windows that would break this at the
    very end of a chunk are dropped and counted in ``stats``).
The next chunk is built in a background thread (reading in threads, tokenizing in a process pool).

Resuming. The stream is a deterministic function of (seed, rank, world_size, the arguments).
``state_dict()`` is (chunk index, windows already taken from it); ``load_state_dict`` rebuilds that chunk
and skips ahead, so a resumed run sees exactly the windows it would have seen.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

META_FILE = Path(__file__).with_name("fineweb_edu_100b_shuffle.json")
VAL_SHARD = "shard_01822.parquet"

# ------------------------------------------------------------------------------------- reading / tokenizing
_fs = None


def read_unit(repo: str, shard: str, row_group: int, retries: int = 8) -> list[str]:
    """The texts of one row group, read remotely (with retries)."""
    global _fs
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    for attempt in range(retries):
        try:
            if _fs is None:
                _fs = HfFileSystem()
            with _fs.open(f"datasets/{repo}/{shard}", block_size=4 << 20) as f:
                return pq.ParquetFile(f).read_row_group(row_group, columns=["text"]).column("text").to_pylist()
        except Exception:                                       # network hiccups
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))


_tokenizer = None


def _init_tokenizer(name: str) -> None:
    global _tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _tokenize(texts: list[str]) -> list[np.ndarray]:
    return [np.asarray(ids, dtype=np.int32) for ids in _tokenizer(texts, add_special_tokens=False)["input_ids"]]


# "fork": the workers share the parent's pages (a "spawn" worker re-imports torch / transformers: ~1 GB each,
# which got runs killed by the OOM daemon on busy hosts). The workers only tokenize, they never touch CUDA.
mp_context = "fork" if hasattr(os, "fork") else "spawn"


class Tokenizer:
    """A process pool of tokenizers (the RWKV tokenizer is pure Python)."""

    def __init__(self, name: str, num_proc: int = 4, batch: int = 128):
        self.batch = batch
        _init_tokenizer(name)        # loads (downloads) the tokenizer once in the parent; forked workers inherit it
        ctx = mp.get_context(mp_context)
        init = (None, ()) if mp_context == "fork" else (_init_tokenizer, (name,))
        self.pool = ctx.Pool(num_proc, initializer=init[0], initargs=init[1]) if num_proc > 0 else None

    def __call__(self, texts: list[str]) -> list[np.ndarray]:
        parts = [texts[i:i + self.batch] for i in range(0, len(texts), self.batch)]
        results = self.pool.map(_tokenize, parts) if self.pool is not None else map(_tokenize, parts)
        return [ids for part in results for ids in part]

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()           # reap the workers (otherwise: "leaked semaphore" warnings)


# ---------------------------------------------------------------------------------------------- packing
def pack(docs: list[np.ndarray], window: int, sep: int | None):
    """Concatenate the documents (each prefixed with ``sep``) and cut non-overlapping windows.
    Returns windows [n, window] and, per window, the first and last document index it touches."""
    pieces = [np.concatenate([[sep], d]).astype(np.int32) if sep is not None else d for d in docs]
    lengths = np.array([len(p) for p in pieces], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    flat = np.concatenate(pieces) if pieces else np.zeros(0, np.int32)
    n = len(flat) // window
    windows = flat[:n * window].reshape(n, window)
    first = np.searchsorted(starts, np.arange(n) * window, side="right") - 1
    last = np.searchsorted(starts, np.arange(1, n + 1) * window - 1, side="right") - 1
    return windows, first, last


def constrained_order(first: np.ndarray, last: np.ndarray, cooldown: int, rng: np.random.Generator):
    """A random order of the windows such that two windows sharing a document are >= cooldown apart.
    Windows [k] hold documents first[k]..last[k]. Returns (order, number of windows dropped)."""
    queue, deferred, order = deque(rng.permutation(len(first)).tolist()), [], []
    recent, recent_docs = deque(), Counter()

    def free(k):
        return all(recent_docs[d] == 0 for d in range(first[k], last[k] + 1))

    while queue or deferred:
        pick = next((j for j, k in enumerate(deferred) if free(k)), None)
        if pick is not None:
            k = deferred.pop(pick)
        else:
            k = None
            while queue:
                cand = queue.popleft()
                if free(cand):
                    k = cand
                    break
                deferred.append(cand)
            if k is None:
                break                            # what is left conflicts with the last `cooldown` windows
        order.append(k)
        recent.append(k)
        recent_docs.update(range(first[k], last[k] + 1))
        if len(recent) > cooldown - 1:           # keep the last cooldown-1 emitted windows
            old = recent.popleft()
            recent_docs.subtract(range(first[old], last[old] + 1))
    return np.asarray(order, dtype=np.int64), len(deferred) + len(queue)


# ----------------------------------------------------------------------------------------------- stream
class FinewebStream:
    def __init__(self, tokenizer, window: int, doc_sep_token_id: int | None, rank: int = 0, world_size: int = 1,
                 seed: int = 0, units_per_chunk: int = 16, cooldown: int = 64, min_doc_tokens: int = 0,
                 num_proc: int = 4, io_threads: int = 8):
        meta = json.loads(META_FILE.read_text())
        self.repo = meta["repo"]
        self.units = [(s, g) for s, n in meta["num_row_groups"].items() if s != VAL_SHARD for g in range(n)]
        self.window, self.sep, self.rank, self.world, self.seed = window, doc_sep_token_id, rank, world_size, seed
        self.units_per_chunk, self.cooldown, self.min_doc_tokens = units_per_chunk, cooldown, min_doc_tokens
        # a HF tokenizer name (tokenized in a process pool) or any callable: list[str] -> list[np.ndarray]
        self.tokenizer = Tokenizer(tokenizer, num_proc) if isinstance(tokenizer, str) else tokenizer
        self.io = ThreadPoolExecutor(io_threads)
        self.prefetch = ThreadPoolExecutor(1)
        self.chunk, self.emitted = 0, 0
        self.stats = Counter()
        self._next = None

    # -- deterministic content
    def chunk_units(self, c: int) -> list[tuple[str, int]]:
        per_rank = len(self.units) // self.world
        per_epoch = per_rank // self.units_per_chunk
        epoch, c = divmod(c, per_epoch)
        perm = np.random.default_rng([self.seed, epoch]).permutation(len(self.units))
        mine = perm[self.rank::self.world][:per_rank]
        return [self.units[i] for i in mine[c * self.units_per_chunk:(c + 1) * self.units_per_chunk]]

    def build_chunk(self, c: int) -> tuple[np.ndarray, Counter]:
        texts = [t for unit in self.io.map(lambda u: read_unit(self.repo, *u), self.chunk_units(c)) for t in unit]
        docs = [d for d in self.tokenizer(texts) if len(d) >= self.min_doc_tokens]
        windows, first, last = pack(docs, self.window, self.sep)
        order, dropped = constrained_order(first, last, self.cooldown, np.random.default_rng([self.seed, self.rank, c, 1]))
        stats = Counter(docs=len(docs), tokens=int(sum(map(len, docs))), windows=len(order), dropped_windows=dropped)
        return windows[order], stats

    # -- iteration
    def state_dict(self) -> dict:
        return {"chunk": self.chunk, "emitted": self.emitted, "seed": self.seed, "rank": self.rank, "world_size": self.world}

    def load_state_dict(self, state: dict) -> None:
        assert (state["seed"], state["rank"], state["world_size"]) == (self.seed, self.rank, self.world), "different stream"
        self.chunk, self.emitted, self._next = state["chunk"], state["emitted"], None

    def __iter__(self):
        while True:
            current = self._next.result() if self._next is not None else self.prefetch.submit(self.build_chunk, self.chunk).result()
            self._next = self.prefetch.submit(self.build_chunk, self.chunk + 1)
            windows, stats = current
            self.stats.update(stats)
            while self.emitted < len(windows):
                self.emitted += 1
                yield windows[self.emitted - 1]
            self.chunk, self.emitted = self.chunk + 1, 0

    def batches(self, batch_size: int):
        it = iter(self)
        while True:
            yield np.stack([next(it) for _ in range(batch_size)])

    def close(self):
        if hasattr(self.tokenizer, "close"):
            self.tokenizer.close()
        self.io.shutdown(wait=False, cancel_futures=True)
        self.prefetch.shutdown(wait=False, cancel_futures=True)


def validation_windows(tokenizer_name: str, window: int, doc_sep_token_id: int | None, n: int, long_docs: bool = False,
                       num_proc: int = 4) -> np.ndarray:
    """Fixed validation windows from the held-out shard: packed like training (``long_docs=False``) or the
    first ``window`` tokens of documents that are at least that long (``long_docs=True``)."""
    meta = json.loads(META_FILE.read_text())
    tok, out, count = Tokenizer(tokenizer_name, num_proc), [], 0
    try:
        for g in range(meta["num_row_groups"][VAL_SHARD]):
            docs = tok(read_unit(meta["repo"], VAL_SHARD, g))
            if long_docs:
                sep = [] if doc_sep_token_id is None else [doc_sep_token_id]
                out += [np.concatenate([sep, d]).astype(np.int32)[:window] for d in docs if len(d) + len(sep) >= window]
            else:
                out.append(pack(docs, window, doc_sep_token_id)[0])
            count = len(out) if long_docs else sum(len(w) for w in out)
            if count >= n:
                break
    finally:
        tok.close()
    windows = np.stack(out) if long_docs else np.concatenate(out)
    return windows[:n]
