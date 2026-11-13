"""The streaming pipeline: packing, the document cooldown, determinism, rank disjointness and resuming
(the remote reads and the tokenizer are replaced by deterministic fakes)."""
from __future__ import annotations

import numpy as np
import pytest

from recurrent_armt import data


def fake_read_unit(repo, shard, row_group, columns=("text",)):
    if repo == "fake/math":       # two columns (problem, solution): long documents
        f = int(shard[1:])
        rng = np.random.default_rng([99, f, row_group])
        n = rng.integers(150, 400, 16)
        return [(f"p{(f * 37 + row_group * 13 + i) % 1000}", f"{90000 + f}:{row_group}:{i}:{n[i]}") for i in range(16)]
    rng = np.random.default_rng([int(shard[6:11]), row_group])
    # mostly short documents, a few very long ones (several windows each)
    lengths = np.where(rng.random(64) < 0.05, rng.integers(300, 900, 64), rng.integers(5, 60, 64))
    return [f"{shard}:{row_group}:{i}:{n}" for i, n in enumerate(lengths)]


def fake_tokenizer(texts):
    out = []
    for t in texts:
        if "\n\n" in t:                                                  # a math document: problem + solution
            t = t.split("\n\n")[1]
            f, rg, i, n = t.split(":")
            out.append(np.full(int(n), (int(f) * 100 + int(rg)) * 100 + int(i), dtype=np.int32))
            continue
        shard, rg, i, n = t.split(":")
        doc = (int(shard[6:11]) * 100 + int(rg)) * 100 + int(i)          # a unique id per document
        out.append(np.full(int(n), doc, dtype=np.int32))                 # every token of a doc carries its id
    return out


MATH = data.Source("math", "fake/math", tuple((f"f{k}", 3) for k in range(10)), columns=("problem", "solution"),
                   holdout_column=0, min_doc_tokens=200)


@pytest.fixture(autouse=True)
def fake_remote(monkeypatch):
    monkeypatch.setattr(data, "read_unit", fake_read_unit)
    monkeypatch.setitem(data.SOURCES, "math", MATH)


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


# ------------------------------------------------------------------------------------------------ mixes
def make_mix(rank=0, world=1, seed=0):
    return data.PretrainStream(fake_tokenizer, window=128, doc_sep_token_id=-1, rank=rank, world_size=world, seed=seed,
                               mix={"fineweb": 2, "math": 1}, cooldown=12, io_threads=2)


def test_fineweb_only_mix_is_the_fineweb_stream():
    a = take(make(), 200)
    b = take(data.PretrainStream(fake_tokenizer, window=128, doc_sep_token_id=-1, mix={"fineweb": 3}, cooldown=12,
                                 io_threads=2), 200)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert make().chunk_units(5) == [("fineweb", *u) for u in [(f, g) for _, f, g in make().chunk_units(5)]]


def test_mix_chunks_take_units_of_every_source():
    s = make_mix()
    for c in range(6):
        units = s.chunk_units(c)
        assert [u[0] for u in units] == ["fineweb", "fineweb", "math"]
    it = iter(s)
    windows = [next(it) for _ in range(150)]
    math_docs = {d for w in windows for d in docs_of(w) if d >= 9_000_000}
    assert math_docs and s.stats["tokens/math"] > 0 and s.stats["tokens/fineweb"] > 0
    for i in range(len(windows)):                     # the cooldown holds over the mixed stream too
        for j in range(i + 1, min(len(windows), i + 12)):
            assert not (docs_of(windows[i]) & docs_of(windows[j]))


def test_mix_min_doc_tokens_and_holdout():
    s = make_mix()
    windows, stats = s.build_chunk(0)
    unit = next(u for u in s.chunk_units(0) if u[0] == "math")
    lengths = [int(t.split(":")[-1]) for t in s._read(unit)]
    assert stats["docs/math"] == sum(n >= 200 for n in lengths)     # min_doc_tokens of the source
    held = trained = 0                                               # training never sees a held-out problem
    for f, g in MATH.units():
        rows = fake_read_unit("fake/math", f, g, MATH.columns)
        out = {r[0] for r in rows if data.held_out(r[0])}
        held += len(out)
        texts = s._read(("math", f, g))
        trained += len(texts)
        assert not any(t.split("\n\n")[0] in out for t in texts)
        assert MATH.texts(rows, split="val") == ["\n\n".join(r) for r in rows if r[0] in out]
    assert held > 0 and trained > 0


def test_mix_ranks_disjoint_and_resume():
    s0, s1 = make_mix(rank=0, world=2), make_mix(rank=1, world=2)
    assert not ({u for c in range(8) for u in s0.chunk_units(c)} & {u for c in range(8) for u in s1.chunk_units(c)})
    ref = take(make_mix(), 120)
    s = make_mix()
    it = iter(s)
    first = [next(it) for _ in range(77)]
    s2 = make_mix()
    s2.load_state_dict(s.state_dict())
    assert all(np.array_equal(x, y) for x, y in zip(first + take(s2, 120 - 77), ref))
