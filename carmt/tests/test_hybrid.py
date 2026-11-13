"""carmt on a hybrid backbone (Qwen3.5: Gated DeltaNet layers in C, attention layers in PyTorch) against
recurrent_armt.RecurrentARMTForCausalLM. The tight reference is the original with its exact recurrence
(``_gated_delta_rule_ref``, no fla kernels) in fp32 without TF32.
Run on a GPU:  python -m pytest carmt/tests/test_hybrid.py -q
"""
from __future__ import annotations

import contextlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 97
TIGHT, BF16 = 1e-4, 5e-2
G, A = "linear_attention", "full_attention"
PATTERNS = {"ends_attn": [G, A, G, G, A], "starts_attn": [A, G, G, A, G]}


def qwen_config(types, hidden=128, heads=2, head_dim=64, ff=256):
    return dict(model_type="qwen3_5_text", vocab_size=VOCAB, hidden_size=hidden, intermediate_size=ff,
                num_hidden_layers=len(types), layer_types=types, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                linear_num_key_heads=heads, linear_num_value_heads=heads, linear_key_head_dim=head_dim,
                linear_value_head_dim=head_dim, linear_conv_kernel_dim=4,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                 "mrope_section": [4, 2, 2], "mrope_interleaved": True},
                tie_word_embeddings=True)


def make(pattern="ends_attn", seed=0, active=True, **overrides):
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    torch.manual_seed(seed)
    b = load_base_model(qwen_config(PATTERNS[pattern]), dtype=torch.float32).cuda()
    with torch.no_grad():   # random, non-trivial values everywhere (the zero-centred norms too)
        for n, p in b.named_parameters():
            if p.dim() >= 2:
                p.normal_(std=0.05)
            elif "norm" in n:
                p.normal_(std=0.1)
            elif n.endswith("A_log"):
                # decay rates A in [0.01, 4] (HF draws up to 16, a log-decay of ~-20 per token: there the chunked scan's gate
                # gradient loses precision in fp32, up to ~1e-3 relative against the exact recurrence; ~1e-5 at 4)
                p.copy_(torch.empty_like(p).uniform_(0.01, 4.0).log())
    cfg = dict(num_mem_tokens=4, d_mem=8, segment_size=16, memory_dtype="float32")
    cfg.update(overrides)
    torch.manual_seed(seed + 1)
    m = RecurrentARMTForCausalLM(RecurrentARMTConfig(**cfg), base_model=b).cuda()
    if active:
        with torch.no_grad():
            for memory in m.memory_layers.values():
                memory.W_mv.weight.normal_(std=0.3)
            m.memory.normal_(std=0.5)
    return m.eval()


def ids_of(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (B, T), generator=g).cuda()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


@contextlib.contextmanager
def exact():
    """The original with its exact token-by-token GDN recurrence and fp32 matmuls (no TF32)."""
    import recurrent_armt.backbones as bb
    old_k, old_tf32 = bb._use_kernels, torch.backends.cuda.matmul.allow_tf32
    bb._use_kernels = lambda x: False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        bb._use_kernels, torch.backends.cuda.matmul.allow_tf32 = old_k, old_tf32


def runner(m, **kw):
    import carmt
    return carmt.CARMT(m, **kw)


def grads(m, fn):
    m.zero_grad(set_to_none=True)
    out = fn()
    out.loss.backward()
    return out, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("pattern", list(PATTERNS))
@pytest.mark.parametrize("T", [64, 57])
def test_logits_fp32(pattern, T):
    m = make(pattern)
    ids = ids_of(2, T)
    with exact(), torch.no_grad():
        ref = m(input_ids=ids).logits
    r = runner(m)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        out = r(input_ids=ids).logits
    assert rel(out, ref) < TIGHT, rel(out, ref)


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_loss_and_gradients_fp32(pattern):
    m = make(pattern)
    m.train()
    ids = ids_of(2, 64, seed=1)
    with exact():
        ref, gref = grads(m, lambda: m(input_ids=ids, labels=ids))
    r = runner(m)
    torch.backends.cuda.matmul.allow_tf32 = False
    out, g = grads(m, lambda: r(input_ids=ids, labels=ids))
    assert abs(out.loss.item() - ref.loss.item()) < 1e-5 * abs(ref.loss.item())
    assert set(g) == set(gref), set(g) ^ set(gref)
    bad = {n: rel(g[n], x) for n, x in gref.items() if rel(g[n], x) > 3e-4}
    assert not bad, bad


def test_logits_and_gradients_bf16():
    """bf16 against the exact fp32 model, measured against the original's own bf16 error (same recurrence, autocast)."""
    m = make()
    m.train()
    ids = ids_of(2, 64, seed=2)
    with exact():
        ref, gref = grads(m, lambda: m(input_ids=ids, labels=ids))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, gob = grads(m, lambda: m(input_ids=ids, labels=ids))
    r = runner(m)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, g = grads(m, lambda: r(input_ids=ids, labels=ids))
    assert abs(out.loss.item() - ref.loss.item()) < 1e-2 * abs(ref.loss.item())
    ec = {n: rel(g[n], x) for n, x in gref.items()}
    eo = {n: rel(gob[n], x) for n, x in gref.items()}
    bad = {n: (ec[n], eo[n]) for n in gref if ec[n] > max(3 * eo[n], BF16)}
    assert not bad, bad
    ratios = sorted(ec[n] / max(eo[n], 1e-6) for n in gref)
    assert ratios[len(ratios) // 2] < 1.5, ratios[len(ratios) // 2]
    m.eval()
    with torch.no_grad():
        with exact():
            lref = m(input_ids=ids).logits
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = r(input_ids=ids).logits
    assert rel(lo, lref) < BF16


def test_pieces_equal_whole_and_state_interop():
    """Feeding a sequence in pieces (with the carried state, KV caches included) = feeding it whole; the state is
    interchangeable with the original's."""
    m = make()
    ids = ids_of(2, 70, seed=3)
    r = runner(m)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        whole = r(input_ids=ids, return_state=True)
        st, outs = None, []
        for a, b in [(0, 5), (5, 16), (16, 40), (40, 41), (41, 70)]:
            o = r(input_ids=ids[:, a:b], state=st, return_state=True)
            st, outs = o.state, outs + [o.logits]
        assert rel(torch.cat(outs, dim=1), whole.logits) < TIGHT
        # carmt's state -> the original continues from it, and the other way round
        with exact():
            o_ref = m(input_ids=ids[:, 41:70], state=r(input_ids=ids[:, :41], return_state=True).state).logits
            st_ref = m(input_ids=ids[:, :41], return_state=True).state
        o_c = r(input_ids=ids[:, 41:70], state=st_ref).logits
        assert rel(o_ref, whole.logits[:, 41:]) < TIGHT
        assert rel(o_c, whole.logits[:, 41:]) < TIGHT


def test_generation_matches_forward():
    m = make()
    ids = ids_of(1, 40, seed=4)
    r = runner(m)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        full = r(input_ids=ids).logits
        o = r(input_ids=ids[:, :20], return_state=True)
        st, steps = o.state, [o.logits]
        for t in range(20, 40):
            o = r(input_ids=ids[:, t:t + 1], state=st, return_state=True)
            st, steps = o.state, steps + [o.logits]
    assert rel(torch.cat(steps, dim=1), full) < TIGHT


def test_state_dropout_training():
    m = make()
    m.train()
    m.rnn_state_dropout = 0.5
    ids = ids_of(4, 64, seed=5)
    torch.manual_seed(7)
    with exact():
        ref, gref = grads(m, lambda: m(input_ids=ids, labels=ids))
    r = runner(m)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(7)
    out, g = grads(m, lambda: r(input_ids=ids, labels=ids))
    assert abs(out.loss.item() - ref.loss.item()) < 1e-5 * abs(ref.loss.item())
    bad = {n: rel(g[n], x) for n, x in gref.items() if rel(g[n], x) > 3e-4}
    assert not bad, bad


def test_accelerate_training_steps_and_frozen_backbone():
    """carmt.accelerate on a hybrid: optimizer steps see the new weights; a frozen backbone gets no gradients."""
    import carmt
    m = make()
    ids = ids_of(2, 48, seed=6)
    carmt.accelerate(m)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    torch.backends.cuda.matmul.allow_tf32 = False
    for _ in range(2):
        opt.zero_grad()
        m(input_ids=ids, labels=ids).loss.backward()
        opt.step()
    with exact():
        ref = m.forward_reference(input_ids=ids, labels=ids).loss.item()
    assert abs(m(input_ids=ids, labels=ids).loss.item() - ref) < 1e-5 * abs(ref)
    for p in m.model.parameters():
        p.requires_grad_(False)
    m.zero_grad(set_to_none=True)
    m(input_ids=ids, labels=ids).loss.backward()
    assert all(p.grad is None for p in m.model.parameters())
    assert all(p.grad is not None for p in m.memory_layers.parameters() if p.requires_grad)
