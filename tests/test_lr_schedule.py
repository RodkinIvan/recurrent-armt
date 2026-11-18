import math
from types import SimpleNamespace

from train import lr_at

A = SimpleNamespace(warmup_steps=100, max_steps=10_000, min_lr_ratio=0.1)


def test_frozen_group_is_zero_then_warms_up():
    for restart in (False, True):
        assert lr_at(8_999, A, start=9_000, restart=restart) == 0.0
        assert lr_at(9_000, A, start=9_000, restart=restart) > 0


def test_default_joins_the_decayed_run_schedule():
    late = lr_at(9_099, A, start=9_000)                         # warmed up, but the run's cosine is at 90%
    assert math.isclose(late, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 8_999 / 9_900)), rel_tol=1e-9)
    assert late < 0.13


def test_restart_reaches_full_lr_and_decays_to_the_floor():
    assert math.isclose(lr_at(9_099, A, start=9_000, restart=True), 1.0)
    mid = lr_at(9_550, A, start=9_000, restart=True)
    assert 0.5 < mid < 0.6                                       # halfway through its own cosine: 0.1 + 0.9 / 2
    assert math.isclose(lr_at(10_000, A, start=9_000, restart=True), 0.1)


def test_restart_changes_nothing_for_a_group_trained_from_the_start():
    for step in (0, 50, 100, 5_000, 9_999):
        assert lr_at(step, A, start=0, restart=True) == lr_at(step, A, start=0)
