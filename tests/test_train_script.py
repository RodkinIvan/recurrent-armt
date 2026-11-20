"""train.py end to end on a tiny Mamba with fake data (the fake remote reads of test_data): ``--stop_step`` stops
early while the lr schedule runs over ``--max_steps``; ``--data_state_from`` makes a phase 2 started with
``--init_from`` continue phase 1's data instead of training on phase 1's windows again."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from helpers import CUDA, VOCAB, tiny_base
from recurrent_armt import data
from test_data import fake_read_unit, fake_tokenizer

pytestmark = pytest.mark.skipif(not CUDA, reason="train.py runs on CUDA")

WINDOW = 16                 # --segment_size 8 x --n_segments 2
BATCH = 2                   # windows per step (--batch_size 2, --grad_accum 1, one process)


def tokenize(texts):
    """The fake documents as ids of the tiny vocabulary (0 is the document separator)."""
    return [d % (VOCAB - 1) + 1 for d in fake_tokenizer(texts)]


def reference_stream():
    """The stream train.py builds from the arguments below, uninterrupted."""
    return data.PretrainStream(tokenize, WINDOW, 0, seed=0, mix={"fineweb": 3}, cooldown=2 * BATCH, io_threads=2)


@pytest.fixture
def run_train(monkeypatch, tmp_path):
    import train
    monkeypatch.setattr(data, "read_unit", fake_read_unit)
    streams = []

    class Stream(data.PretrainStream):              # the fake tokenizer; records what train.py does with the stream
        def __init__(self, tokenizer, window, sep, **kw):
            super().__init__(tokenize, window, sep, **kw)
            self.loaded, self.first_batch = None, None
            streams.append(self)

        def load_state_dict(self, state):
            self.loaded = dict(state)
            super().load_state_dict(state)

        def batches(self, batch_size):
            for i, b in enumerate(super().batches(batch_size)):
                if i == 0:
                    self.first_batch = b.copy()
                yield b

    monkeypatch.setattr(train, "PretrainStream", Stream)
    rng = np.random.default_rng(0)
    monkeypatch.setattr(train, "validation_windows",
                        lambda tokenizer, window, sep, n, **kw: rng.integers(1, VOCAB, (n, window)).astype(np.int32))
    base = tmp_path / "base"
    tiny_base("mamba").save_pretrained(base)

    def run(*args):
        argv = ["train.py", "--base_model", base, "--doc_sep_token_id", 0, "--units_per_chunk", 3, "--num_proc", 1,
                "--segment_size", 8, "--n_segments", 2, "--batch_size", BATCH, "--grad_accum", 1,
                "--num_mem_tokens", 2, "--d_mem", 16, "--no_grad_checkpointing", "--lr", 1e-3, "--warmup_steps", 2,
                "--eval_windows", 4, "--eval_long_windows", 0, "--log_every", 1, "--eval_every", 100, "--save_every", 100,
                *args]
        monkeypatch.setattr(sys, "argv", [str(x) for x in argv])
        train.main()
        return streams[-1]

    return run, tmp_path


def rows(run_dir):
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]


def test_stop_step_stops_early_on_the_full_schedule(run_train):
    from train import lr_at
    run, tmp = run_train
    run("--out_dir", tmp / "p1", "--max_steps", 8, "--stop_step", 4)
    log = rows(tmp / "p1")
    steps = [r for r in log if "train/loss" in r]
    assert [r["step"] for r in steps] == [1, 2, 3, 4]
    a = SimpleNamespace(warmup_steps=2, max_steps=8, min_lr_ratio=0.1)            # the cosine runs over 8 steps, not 4
    assert [r["lr/backbone"] for r in steps] == pytest.approx([1e-3 * lr_at(r["step"] - 1, a) for r in steps])
    assert steps[-1]["lr/backbone"] > 1e-3 * 0.8                                  # a 4-step cosine would be at 0.55
    assert [r["step"] for r in log if "val/loss" in r] == [0, 4]                  # base model; the stop step
    assert torch.load(tmp / "p1" / "last" / "trainer.pt", weights_only=False)["step"] == 4
    assert (tmp / "p1" / "final" / "config.json").exists()


def test_data_state_from_continues_phase_1s_data(run_train):
    run, tmp = run_train
    p1 = run("--out_dir", tmp / "p1", "--max_steps", 8, "--stop_step", 3)
    saved = json.loads((tmp / "p1" / "last" / "data_rank0.json").read_text())
    p2 = run("--out_dir", tmp / "p2", "--init_from", tmp / "p1" / "final", "--data_state_from", tmp / "p1" / "last",
             "--max_steps", 2)
    assert p2.loaded == saved
    batches = reference_stream().batches(BATCH)
    np.testing.assert_array_equal(p1.first_batch, next(batches))
    for _ in range(2):                                    # phase 1 took 3 batches
        next(batches)
    np.testing.assert_array_equal(p2.first_batch, next(batches))           # phase 2 starts where phase 1 stopped
    restart = run("--out_dir", tmp / "p2_restart", "--init_from", tmp / "p1" / "final", "--max_steps", 2)
    np.testing.assert_array_equal(restart.first_batch, p1.first_batch)     # without the flag: phase 1's data again
