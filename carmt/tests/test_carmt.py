"""Equivalence of the CUDA implementation (carmt) with recurrent_armt.RecurrentARMTForCausalLM over GDN-2.

Two references:
  * the original model as it is (fla Triton kernels; they multiply in TF32, so ~1e-3 relative noise in fp32), and
  * the original model with fla's exact PyTorch GDN-2 recurrence (``reference.exact_gdn2``): this is the tight check.
Run on a GPU:  python -m pytest carmt/tests -q
"""
from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from reference import exact_gdn2, rel  # noqa: E402

VOCAB = 97
# tight bounds (vs the exact reference, fp32) and loose ones (vs the TF32 fla kernels / bf16)
TIGHT, FLA, BF16 = 1e-4, 5e-3, 5e-2


def base_config(head_dim=64, heads=2, hidden=128, layers=2, ff=256, gate="swish"):
    return dict(model_type="gdn2_lm", vocab_size=VOCAB, hidden_size=hidden, num_hidden_layers=layers, num_heads=heads,
                head_dim=head_dim, intermediate_size=ff, output_gate=gate)


def make(seed=0, base=None, active=True, **overrides):
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    torch.manual_seed(seed)
    b = load_base_model(base or base_config(), dtype=torch.float32).cuda()
    with torch.no_grad():   # random (non-trivial) values everywhere, as after training
        for n, p in b.named_parameters():
            if p.dim() >= 2:
                p.normal_(std=0.05)
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


def runner(m, **kw):
    import carmt
    return carmt.CARMT(m, **kw)


class tf32:
    """TF32 in fp32 matmuls on / off inside the block (train.py turns it on; the bf16 build's fused scan-backward kernel
    k_post_bwd runs only then, with TF32 off the library takes its GEMM path)."""
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.on

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.old


CONFIGS = {
    "default": dict(),
    "head128": dict(base=base_config(head_dim=128, heads=2, hidden=256)),
    "sigmoid_gate": dict(base=base_config(gate="sigmoid")),
    "in_stream": dict(mem_tokens_in_stream=True),
    "mem_heads2_gating": dict(n_heads=2, gating=True),
    "partial_wrap": dict(wrap_layers=[0, 1]),
    "no_memory": dict(num_mem_tokens=0),
    "long_segment": dict(segment_size=100, num_mem_tokens=6),   # chunks of 64: partial chunks of real and memory tokens
    "three_layers": dict(base=base_config(layers=3)),
}


# ---------------------------------------------------------------------------------------------------------- forward
@pytest.mark.parametrize("name", list(CONFIGS))
def test_logits_fp32(name):
    m = make(**CONFIGS[name])
    ids = ids_of(2, 150)
    with torch.no_grad():
        with exact_gdn2():
            ref = m(input_ids=ids).logits
        fla = m(input_ids=ids).logits
        out = runner(m)(input_ids=ids).logits
    assert rel(out, ref) < TIGHT, rel(out, ref)
    assert rel(out, fla) < FLA


@pytest.mark.parametrize("name", ["default", "head128"])
def test_logits_bf16(name):
    m = make(**CONFIGS[name])
    ids = ids_of(2, 150)
    with torch.no_grad():
        with exact_gdn2():
            ref = m(input_ids=ids).logits
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fla = m(input_ids=ids).logits.float()
            out = runner(m)(input_ids=ids).logits
    # the C bf16 path is about as close to the fp32 result as the original's own bf16 autocast path (mean relative error)
    mrel = lambda a, b: ((a.double() - b.double()).abs().mean() / b.double().abs().mean()).item()
    assert rel(out, ref) < 0.15 and mrel(out, ref) < 1.3 * mrel(fla, ref) + 1e-3, (mrel(out, ref), mrel(fla, ref))


def test_loss_and_token_losses_no_grad():
    m = make()
    ids = ids_of(3, 70)
    with torch.no_grad():
        with exact_gdn2():
            ref = m(input_ids=ids, labels=ids)
        out = runner(m)(input_ids=ids, labels=ids)
    assert abs(out.loss.item() - ref.loss.item()) < 1e-5
    assert rel(out.token_losses, ref.token_losses) < TIGHT


def test_labels_mask_and_ignore_index():
    m = make()
    ids = ids_of(2, 50)
    labels = ids.clone()
    labels[0, 10:20] = -100
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[1, 30:] = False
    with torch.no_grad():
        with exact_gdn2():
            ref = m(input_ids=ids, labels=labels, labels_mask=mask)
        out = runner(m)(input_ids=ids, labels=labels, labels_mask=mask)
    assert abs(out.loss.item() - ref.loss.item()) < 1e-5
    assert rel(out.token_losses, ref.token_losses) < TIGHT


# ---------------------------------------------------------------------------------------------------------- gradients
def _grads(m, fn):
    m.zero_grad(set_to_none=True)
    out = fn()
    out.loss.backward()
    g = {n: p.grad.clone() if p.grad is not None else None for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    return out, g


@pytest.mark.parametrize("name", list(CONFIGS))
def test_gradients_fp32(name):
    m = make(**CONFIGS[name])
    ids = ids_of(2, 120)
    with exact_gdn2():
        ref, gref = _grads(m, lambda: m(input_ids=ids, labels=ids))
    _, gfla = _grads(m, lambda: m(input_ids=ids, labels=ids))
    r = runner(m)
    out, g = _grads(m, lambda: r(input_ids=ids, labels=ids))
    assert abs(out.loss.item() - ref.loss.item()) < 1e-5
    for n, x in gref.items():
        if x is None:
            assert g[n] is None or not g[n].any(), n
            continue
        assert g[n] is not None, n
        # at least 5x closer to the exact reference than the original's own (TF32) kernels, or within 3e-4
        assert rel(g[n], x) < max(3e-4, 0.2 * rel(gfla[n], x)), (n, rel(g[n], x), rel(gfla[n], x))


@pytest.mark.parametrize("name", ["default", "head128"])
@pytest.mark.parametrize("tf", [False, True])
def test_gradients_bf16(name, tf):
    m = make(**CONFIGS[name])
    ids = ids_of(2, 120)
    with exact_gdn2(), tf32(False):
        _, gref = _grads(m, lambda: m(input_ids=ids, labels=ids))
    with torch.autocast("cuda", dtype=torch.bfloat16), tf32(tf):
        _, gfla = _grads(m, lambda: m(input_ids=ids, labels=ids))
        _, g = _grads(m, lambda: runner(m)(input_ids=ids, labels=ids))
    for n, x in gref.items():
        # bf16 vs fp32 reference: within the error of the original's own bf16 path (x3), or tiny
        assert rel(g[n], x) < max(3 * rel(gfla[n], x), 2e-2), (n, rel(g[n], x), rel(gfla[n], x))


def test_gradient_accumulation_and_scale():
    """(loss / k).backward() twice accumulates like PyTorch; the flat grad buffer is scaled by the upstream grad."""
    m = make()
    ids1, ids2 = ids_of(2, 60, 1), ids_of(2, 60, 2)
    with exact_gdn2():
        m.zero_grad(set_to_none=True)
        (m(input_ids=ids1, labels=ids1).loss / 2).backward()
        (m(input_ids=ids2, labels=ids2).loss / 2).backward()
        gref = {n: p.grad.clone() for n, p in m.named_parameters()}
    r = runner(m)
    m.zero_grad(set_to_none=True)
    (r(input_ids=ids1, labels=ids1).loss / 2).backward()
    (r(input_ids=ids2, labels=ids2).loss / 2).backward()
    for n, p in m.named_parameters():
        assert rel(p.grad, gref[n]) < 1e-4, n


def test_state_dropout_training():
    """rnn_state_dropout in training mode: the same masks (same RNG draws) and the same gradients."""
    m = make()
    m.train()
    m.rnn_state_dropout = 0.5
    ids = ids_of(4, 80)
    torch.manual_seed(5)
    with exact_gdn2():
        ref, gref = _grads(m, lambda: m(input_ids=ids, labels=ids))
    torch.manual_seed(5)
    out, g = _grads(m, lambda: runner(m)(input_ids=ids, labels=ids))
    assert abs(out.loss.item() - ref.loss.item()) < 1e-5
    for n, x in gref.items():
        assert rel(g[n], x) < 3e-4, n


def test_state_reset_eval():
    m = make()
    m.state_reset = True
    ids = ids_of(2, 70)
    with torch.no_grad():
        with exact_gdn2():
            ref = m(input_ids=ids).logits
        out = runner(m)(input_ids=ids).logits
    assert rel(out, ref) < TIGHT


# ---------------------------------------------------------------------------------------------------------- state
@pytest.mark.parametrize("name", ["default", "in_stream", "long_segment"])
def test_pieces_equal_whole_and_state_matches(name):
    """forward over pieces with the state passed along == one forward; the returned state equals the original's."""
    m = make(**CONFIGS[name])
    r = runner(m)
    ids = ids_of(2, 130)
    cuts = [0, 7, 40, 41, 100, 130]
    with torch.no_grad():
        whole = r(input_ids=ids, return_state=True)
        st, parts = None, []
        for a, b in zip(cuts[:-1], cuts[1:]):
            o = r(input_ids=ids[:, a:b], state=st, return_state=True)
            st = o.state
            parts.append(o.logits)
        with exact_gdn2():
            ref = m(input_ids=ids, return_state=True)
    assert rel(torch.cat(parts, 1), whole.logits) < TIGHT
    assert st.position == whole.state.position == ref.state.position
    for (mc, rc), (mr, rr) in zip(st.layers, ref.state.layers):
        assert rel(rc["recurrent_state"], rr["recurrent_state"]) < TIGHT
        for a, b in zip(rc["conv_state"], rr["conv_state"]):
            assert rel(a, b) < TIGHT
        if mr is not None:
            assert rel(mc[0], mr[0]) < TIGHT and rel(mc[1], mr[1]) < TIGHT


def test_state_interop_with_original():
    """a state made by the C path continues in the original model and vice versa."""
    m = make()
    r = runner(m)
    ids = ids_of(2, 90)
    with torch.no_grad(), exact_gdn2():
        whole = m(input_ids=ids).logits
        a = r(input_ids=ids[:, :37], return_state=True)
        b = m(input_ids=ids[:, 37:], state=a.state)
        c = m(input_ids=ids[:, :37], return_state=True)
        d = r(input_ids=ids[:, 37:], state=c.state)
    assert rel(b.logits, whole[:, 37:]) < TIGHT
    assert rel(d.logits, whole[:, 37:]) < TIGHT


def test_generation_matches_forward():
    import carmt
    m = make()
    carmt.accelerate(m)
    prompt = ids_of(2, 13)
    with torch.no_grad():
        new, logits = m.generate(prompt, max_new_tokens=40, return_logits=True)    # crosses several segment ends
        with exact_gdn2():
            full = m.forward_reference(input_ids=torch.cat([prompt, new[:, :-1]], 1)).logits[:, prompt.shape[1] - 1:]
    assert rel(logits, full) < TIGHT


def test_accelerate_api_training_step():
    """carmt.accelerate(model): the unchanged training-loop calls produce the original's gradients."""
    import carmt
    m = make()
    ids = ids_of(2, 64)
    with exact_gdn2():
        ref, gref = _grads(m, lambda: m(input_ids=ids, labels=ids))
    carmt.accelerate(m)
    m.train()
    out, g = _grads(m, lambda: m(input_ids=ids, labels=ids))
    assert out.token_losses is not None and abs(out.loss.item() - ref.loss.item()) < 1e-5
    for n, x in gref.items():
        assert rel(g[n], x) < 3e-4, n
    assert m.memory_stats(ids)["mem/read_ratio"] > 0     # diagnostics still run (through the PyTorch path)


def test_frozen_parameters_get_no_grad():
    m = make()
    for p in m.model.parameters():
        p.requires_grad_(False)
    ids = ids_of(2, 40)
    runner(m)(input_ids=ids, labels=ids).loss.backward()
    assert all(p.grad is None for p in m.model.parameters())
    assert all(p.grad is not None for p in m.memory_layers.parameters())


# ---------------------------------------------------------------------------------------------------------- real checkpoint
CKPT = os.environ.get("CARMT_CKPT", "checkpoints/gdn2-370m-30B")


@pytest.mark.skipif(not os.path.exists(CKPT), reason="no GDN-2 checkpoint (set CARMT_CKPT)")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_pretrained_gdn2(precision):
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    base = load_base_model(CKPT, dtype=torch.float32).cuda()
    m = RecurrentARMTForCausalLM(RecurrentARMTConfig(num_mem_tokens=32, d_mem=64, segment_size=256), base_model=base).cuda().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for memory in m.memory_layers.values():
            memory.W_mv.weight.normal_(std=0.002)
    ids = torch.randint(3, 32000, (2, 1024), generator=torch.Generator().manual_seed(0)).cuda()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else torch.autocast("cuda", enabled=False)
    with torch.no_grad():
        with exact_gdn2():
            ref = m(input_ids=ids, labels=ids)
        with ctx:
            fla = m(input_ids=ids, labels=ids)
            out = runner(m)(input_ids=ids, labels=ids)
    d_c, d_f = abs(out.loss.item() - ref.loss.item()), abs(fla.loss.item() - ref.loss.item())
    print(f"\n[{precision}] loss exact-ref {ref.loss.item():.6f} carmt {out.loss.item():.6f} (|d| {d_c:.2e}) "
          f"original {fla.loss.item():.6f} (|d| {d_f:.2e})")
    assert d_c < max(2 * d_f, 1e-4)


def test_long_input_many_segments():
    """more than 4096 segments in one call (no fixed limit), and one long call == the same input in pieces"""
    m = make()
    r = runner(m)
    ids = ids_of(1, 16 * 4500 + 7)          # 4501 segments of 16 tokens
    with torch.no_grad():
        whole = r(input_ids=ids, labels=ids, output_logits="last", return_state=True)
        st, cut = None, [0, 30001, 50000, ids.shape[1]]
        for a, b in zip(cut[:-1], cut[1:]):
            o = r(input_ids=ids[:, a:b], state=st, return_state=True, output_logits="last")
            st = o.state
    assert torch.isfinite(whole.loss)
    assert rel(o.logits, whole.logits) < TIGHT
    for (mc, rc), (mw, rw) in zip(st.layers, whole.state.layers):
        assert rel(rc["recurrent_state"], rw["recurrent_state"]) < TIGHT and rel(mc[0], mw[0]) < TIGHT


# ---------------------------------------------------------------------------------------------------------- robustness
def test_nograd_forward_between_training_forward_and_backward():
    """an eval / diagnostic forward on the same model between a training forward and its backward changes nothing"""
    m = make()
    ids, other = ids_of(2, 60, 1), ids_of(3, 45, 2)
    with exact_gdn2():
        _, gref = _grads(m, lambda: m(input_ids=ids, labels=ids))
    r = runner(m)
    m.zero_grad(set_to_none=True)
    out = r(input_ids=ids, labels=ids)
    with torch.no_grad():
        r(input_ids=other, labels=other)                        # no state: uses the inference scratch
        r(input_ids=other, return_state=True)                   # with a state
    out.loss.backward()
    for n, p in m.named_parameters():
        assert rel(p.grad, gref[n]) < 3e-4, n


def test_two_runners_in_one_process():
    """two models (e.g. a model and its EMA / teacher) each keep their own packed weights and training record"""
    a, b = make(seed=0), make(seed=7)
    ra, rb = runner(a), runner(b)
    ids = ids_of(2, 50)
    with torch.no_grad():
        with exact_gdn2():
            ref_a, ref_b = a(input_ids=ids).logits, b(input_ids=ids).logits
        with torch.autocast("cuda", dtype=torch.bfloat16):      # bf16 build: packed weight copies
            la1 = ra(input_ids=ids).logits
            lb = rb(input_ids=ids).logits
            la2 = ra(input_ids=ids).logits
    assert rel(la1, la2) == 0 and rel(la1, ref_a) < BF16 and rel(lb, ref_b) < BF16 and rel(la1, lb) > 0.1
    with exact_gdn2():
        _, gref = _grads(a, lambda: a(input_ids=ids, labels=ids))
    a.zero_grad(set_to_none=True)
    out = ra(input_ids=ids, labels=ids)
    with torch.no_grad():
        rb(input_ids=ids)
    out.loss.backward()
    for n, p in a.named_parameters():
        assert rel(p.grad, gref[n]) < 3e-4, n


@pytest.mark.parametrize("opt", ["adamw_fused", "adamw_foreach", "sgd"])
def test_forward_sees_weights_after_optimizer_step(opt):
    m = make()
    r = runner(m)
    ids = ids_of(2, 40)
    params = list(m.parameters())
    o = {"adamw_fused": lambda: torch.optim.AdamW(params, lr=1e-2, fused=True),
         "adamw_foreach": lambda: torch.optim.AdamW(params, lr=1e-2, foreach=True),
         "sgd": lambda: torch.optim.SGD(params, lr=1e-1)}[opt]()
    for _ in range(2):
        with torch.no_grad():
            r(input_ids=ids)                                    # packs the current weights
        r(input_ids=ids, labels=ids).loss.backward()
        o.step()
        o.zero_grad(set_to_none=True)
        with torch.no_grad():
            with exact_gdn2():
                ref = m.forward_reference(input_ids=ids).logits if hasattr(m, "forward_reference") else m(input_ids=ids).logits
            out = r(input_ids=ids).logits
        assert rel(out, ref) < TIGHT


def test_frozen_backbone_memory_gradients():
    """with the backbone frozen the backbone's gradient work is skipped and the memory gradients are unchanged"""
    m = make()
    ids = ids_of(2, 70)
    with exact_gdn2():
        _, gref = _grads(m, lambda: m(input_ids=ids, labels=ids))
    for p in m.model.parameters():
        p.requires_grad_(False)
    m.zero_grad(set_to_none=True)
    runner(m)(input_ids=ids, labels=ids).loss.backward()
    for n, p in m.named_parameters():
        if n.startswith("model."):
            assert p.grad is None
        else:
            assert rel(p.grad, gref[n]) < 3e-4, n


def test_errors_raise_python_exceptions():
    import carmt
    r = runner(make())
    cctx = r._context("fp32")
    with pytest.raises(RuntimeError, match="backward without a recorded training forward"):
        carmt._check(cctx.lib, cctx.lib.armt_backward(cctx.handle, r._param_ptrs(), None, None, None), "backward")


def test_library_memory_is_visible_to_torch():
    m = make()
    ids = ids_of(4, 200)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    runner(m)(input_ids=ids, labels=ids).loss.backward()
    torch.cuda.synchronize()
    # the layer-input store alone: (layers + 1) x segments x B x (S + M) x D fp32
    store = 3 * 13 * 4 * 20 * 128 * 4
    assert torch.cuda.max_memory_allocated() - base > store


# ---------------------------------------------------------------------------------------------------------- bf16 at real size
@pytest.mark.skipif(not os.path.exists(CKPT), reason="no GDN-2 checkpoint (set CARMT_CKPT)")
@pytest.mark.parametrize("scan_fp32", [False, True])
@pytest.mark.parametrize("tf", [False, True])
def test_gradients_bf16_real_dimensions(scan_fp32, tf):
    """the configuration training uses: bf16, head_dim 128, 16 heads, d_model 1024, segment 1024, 32 memory tokens.
    Per-parameter relative error ||g - g_ref|| / ||g_ref|| against the fp32 original, compared with the original's own
    bf16 autocast path."""
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    torch.manual_seed(0)
    base = load_base_model(CKPT, dtype=torch.float32).cuda()
    m = RecurrentARMTForCausalLM(RecurrentARMTConfig(num_mem_tokens=32, d_mem=64, segment_size=1024), base_model=base).cuda()
    with torch.no_grad():
        for memory in m.memory_layers.values():
            memory.W_mv.weight.normal_(std=0.02)
    m.train()
    ids = torch.randint(3, 32000, (2, 3 * 1024), generator=torch.Generator().manual_seed(0)).cuda()
    with tf32(False):
        _, gref = _grads(m, lambda: m(input_ids=ids, labels=ids))                 # fp32 original
    with torch.autocast("cuda", dtype=torch.bfloat16), tf32(tf):                  # (tf32 on: as train.py)
        _, gorig = _grads(m, lambda: m(input_ids=ids, labels=ids))                # original, bf16
        _, gc = _grads(m, lambda: runner(m, scan_fp32=scan_fp32)(input_ids=ids, labels=ids))   # carmt, bf16
    rn = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()
    ec = {n: rn(gc[n], x) for n, x in gref.items()}
    eo = {n: rn(gorig[n], x) for n, x in gref.items()}
    worst = sorted(ec, key=lambda n: ec[n] / max(eo[n], 1e-3), reverse=True)[:12]
    print(f"\n[bf16 real dims, scan_fp32={scan_fp32}, tf32={tf}] median rel-norm error carmt %.2e original %.2e" % (sorted(ec.values())[len(ec) // 2],
                                                                            sorted(eo.values())[len(eo) // 2]))
    for n in worst:
        print(f"   {n:55s} carmt {ec[n]:.2e}  original {eo[n]:.2e}  |g| {gref[n].norm().item():.2e}")
    import math
    lr = sorted(math.log2(max(ec[n], 1e-6) / max(eo[n], 1e-6)) for n in gref)
    q = lambda f: 2 ** lr[int(f * (len(lr) - 1))]
    best = sorted(ec, key=lambda n: ec[n] / max(eo[n], 1e-6))[:4]
    print("   ratio carmt/original: 5%% %.2f  25%% %.2f  50%% %.2f  75%% %.2f  95%% %.2f  max %.2f" % (
        q(.05), q(.25), q(.5), q(.75), q(.95), q(1)))
    for n in best:
        print(f"   (other side) {n:42s} carmt {ec[n]:.2e}  original {eo[n]:.2e}")
    # the small, cancellation-heavy memory-write gradients are noise-dominated in any bf16 path (tails on both sides),
    # so the check is on the distribution: typically at least as accurate as the original's bf16 path, never far off
    assert q(.5) <= 1.0 and q(.95) <= 1.5, (q(.5), q(.95))
    for n in gref:
        assert ec[n] < max(10 * eo[n], 5e-2), (n, ec[n], eo[n])


def test_training_steps_do_not_repack_weights():
    """the packed weights survive a backward: micro-steps without an optimizer step do not re-pack (or re-allocate)"""
    import carmt
    m = make()
    r = runner(m)
    ids = ids_of(2, 40)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r(input_ids=ids, labels=ids).loss.backward()
        cctx = r._context("bf16")
        torch.cuda.synchronize()
        packed = ctypes_buf_ptr = None
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        r(input_ids=ids, labels=ids).loss.backward()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
    # the second micro-step re-allocates only the training stores (11) and PyTorch's own tensors, not ~90 scratch buffers
    assert after - before < 60, after - before


def test_process_exits_cleanly_with_a_model_alive_at_exit():
    """a script that keeps an accelerated model in a global until interpreter exit must exit with code 0 (the C context
    frees through PyTorch's allocator, so it has to be freed before interpreter teardown, not from __del__ during it)"""
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    code = "\n".join([
        "import sys, torch", f"sys.path[:0] = [{here!r}, {os.path.dirname(os.path.dirname(here))!r}]",
        "import carmt", "from test_carmt import make, ids_of",
        "m = make(); carmt.accelerate(m); m.train(); ids = ids_of(2, 40)",
        "with torch.autocast('cuda', dtype=torch.bfloat16): m(input_ids=ids, labels=ids).loss.backward()",
        "torch.cuda.synchronize(); print('end', flush=True)"])
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "end" in p.stdout, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    assert "Exception ignored" not in p.stderr, p.stderr[-2000:]
