"""Training on GPU in bf16 autocast (the setting of train.py): the loss goes down on a fixed batch, the
memory parameters move, and gradient checkpointing gives the same gradients under autocast too."""
from __future__ import annotations

import pytest
import torch

from helpers import CUDA, KINDS, random_ids, tiny_armt

pytestmark = pytest.mark.skipif(not CUDA, reason="needs a GPU")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_overfits_a_batch_in_bf16(kind, checkpointing):
    model = tiny_armt(kind, active_memory=False).train()       # from the real init: W_mv = 0
    if checkpointing:
        model.gradient_checkpointing_enable()
    ids = random_ids(4, 16 * 4)
    w_mv0 = {k: m.W_mv.weight.detach().clone() for k, m in model.memory_layers.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    print(f"{kind} ckpt={checkpointing}: loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < 0.5 * losses[0]
    assert all(not torch.equal(w_mv0[k], m.W_mv.weight) for k, m in model.memory_layers.items())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_every_parameter_gets_gradients_under_autocast(kind, checkpointing):
    """Regression: the no-grad state probe inside autocast must not hand gradient-free cached bf16 weight
    copies to the rest of the step (the backbone would silently not train)."""
    model = tiny_armt(kind).train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    ids = random_ids(2, 16 * 3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not no_grad, no_grad


@pytest.mark.parametrize("kind", KINDS)
def test_checkpointing_same_grads_under_autocast(kind):
    model = tiny_armt(kind).train()
    ids = random_ids(2, 16 * 3 + 5)

    def grads():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        return loss.item(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}

    l0, g0 = grads()
    model.gradient_checkpointing_enable()
    l1, g1 = grads()
    worst = max((g0[n] - g1[n]).abs().max().item() / max(1e-6, g0[n].abs().max().item()) for n in g0)
    print(f"{kind}: bf16 loss {l0:.6f} vs {l1:.6f}, max rel grad diff {worst:.2e}")
    assert abs(l0 - l1) < 1e-4 and worst < 1e-2
