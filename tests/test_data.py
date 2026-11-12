"""The streaming pipeline: packing, the document cooldown, determinism, rank disjointness and resuming
(the remote reads and the tokenizer are replaced by deterministic fakes)."""
from __future__ import annotations

import numpy as np
import pytest

from recurrent_armt import data


def fake_read_unit(repo, shard, row_group):
    rng = np.random.default_rng([int(shard[6:11]), row_group])
    # mostly short documents, a few very long ones (several windows each)
    lengths = np.where(rng.random(64) < 0.05, rng.integers(300, 900, 64), rng.integers(5, 60, 64))
    return [f"{shard}:{row_group}:{i}:{n}" for i, n in enumerate(lengths)]


def fake_tokenizer(texts):
    out = []
    for t in texts:
        shard, rg, i, n = t.split(":")
        doc = (int(shard[6:11]) * 100 + int(rg)) * 100 + int(i)          # a unique id per document
        out.append(np.full(int(n), doc, dtype=np.int32))                 # every token of a doc carries its id
    return out


@pytest.fixture(autouse=True)
def fake_remote(monkeypatch):
    monkeypatch.setattr(data, "read_unit", fake_read_unit)


def make(rank=0, world=1, seed=0, cooldown=12):
    return data.FinewebStream(fake_tokenizer, window=128, doc_sep_token_id=-1, rank=rank, world_size=world, seed=seed,
                              units_per_chunk=3, cooldown=cooldown, io_threads=2)


def docs_of(window):
    return set(np.unique(window[window >= 0]).tolist())


def test_pack():
    docs = [np.full(n, i, dtype=np.int32) for i, n in enumerate([5, 300, 2, 2, 40, 7])]
    windows, first, last = data.pack(docs, 64, -1)
    flat = np.concatenate([np.concatenate([[-1], d]) for d in docs])
    assert np.array_equal(windows.reshape(-1), flat[:windows.size])
    for w, f, l in zip(windows, first, last):
        assert docs_of(w) <= set(range(f, l + 1)) and {f, l} & (docs_of(w) | {-1})


def test_constrained_order_cooldown():
    rng = np.random.default_rng(0)
    lengths = np.where(rng.random(400) < 0.1, rng.integers(200, 1000, 400), rng.integers(3, 50, 400))
    docs = [np.full(n, i, dtype=np.int32) for i, n in enumerate(lengths)]
    windows, first, last = data.pack(docs, 64, -1)
    order, dropped = data.constrained_order(first, last, 20, rng)
    assert len(set(order.tolist())) == len(order) == len(windows) - dropped
    assert dropped < 0.02 * len(windows)
    pos = {k: p for p, k in enumerate(order)}
    for a in range(len(windows)):
        for b in range(a + 1, len(windows)):
            if a in pos and b in pos and max(first[a], first[b]) <= min(last[a], last[b]):
                assert abs(pos[a] - pos[b]) >= 20


def take(stream, n):
    it = iter(stream)
    return [next(it) for _ in range(n)]


def test_stream_deterministic_and_cooldown_across_chunks():
    a, b = take(make(), 300), take(make(), 300)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = take(make(seed=1), 300)
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))
    for i in range(len(a)):
        for j in range(i + 1, min(len(a), i + 12)):
            assert not (docs_of(a[i]) & docs_of(a[j])), (i, j)


def test_ranks_are_disjoint():
    s0, s1 = make(rank=0, world=2), make(rank=1, world=2)
    assert not ({u for c in range(20) for u in s0.chunk_units(c)} & {u for c in range(20) for u in s1.chunk_units(c)})
    d0 = set().union(*map(docs_of, take(s0, 200)))
    d1 = set().union(*map(docs_of, take(s1, 200)))
    assert not (d0 & d1)


def test_resume_is_exact():
    ref = take(make(), 250)
    s = make()
    it = iter(s)
    first = [next(it) for _ in range(137)]
    state = s.state_dict()
    s2 = make()
    s2.load_state_dict(state)
    rest = take(s2, 250 - 137)
    assert all(np.array_equal(x, y) for x, y in zip(first + rest, ref))
