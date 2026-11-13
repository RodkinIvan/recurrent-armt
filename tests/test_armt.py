"""Equivalence / invariance checks of recurrent ARMT (run on CPU in fp64 and on GPU in fp32).

1. layers-first (horizontal) == segments-first (vertical)
2. generation (one token at a time on the state) == forward
3. the state has a constant size
4. gradients flow through both the recurrent state and the associative memory
5. the speed / memory optimizations do not change the result (gradient checkpointing, chunked loss,
   CUDA kernels vs reference scans, feeding the sequence in pieces)
plus: at initialization (W_mv = 0) ARMT is exactly the base model; save / load round trip.
"""
from __future__ import annotations

import pytest
import torch

from helpers import ATOL, CUDA, DEVICE, HAS_FLA, HYBRIDS, KINDS, KTOL, grad_tol, max_diff, random_ids, tiny_armt, tiny_base, train_autocast
from recurrent_armt import ARMTState, RecurrentARMTForCausalLM, backbones
from recurrent_armt.modeling import map_tensors

OPTIONS = [False, True]          # mem_tokens_in_stream


def base_logits(base, ids):
    with torch.no_grad():
        return base(input_ids=ids).logits


def tol_vs(ref, kind=None):
    """HF returns float32 logits even for a float64 model, and HF Qwen3.5 runs its GDN in float32:
    compare at the precision of the reference."""
    return KTOL if CUDA or (ref.dtype == torch.float64 and kind not in HYBRIDS) else 1e-5


def logits_of(model, ids, vertical=False, **kw):
    model.vertical_mode = vertical
    with torch.no_grad():
        return model(input_ids=ids, **kw).logits


# ------------------------------------------------------------------------------------------- base model
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("vertical", [False, True])
def test_init_equals_base_model(kind, vertical):
    """W_mv = 0 at init -> nothing is written -> with the memory tokens off the stream, ARMT is the base
    model over the whole sequence (the recurrent state crosses segment boundaries untouched)."""
    base = tiny_base(kind)
    ids = random_ids(2, 16 * 3 + 5)
    ref = base_logits(base, ids)
    model = tiny_armt(kind, active_memory=False)
    out = logits_of(model, ids, vertical)
    assert max_diff(out, ref) < tol_vs(ref, kind), max_diff(out, ref)


# --------------------------------------------------------------------------------------- 1. V == H
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("in_stream", OPTIONS)
def test_vertical_equals_horizontal(kind, in_stream):
    model = tiny_armt(kind, mem_tokens_in_stream=in_stream)
    ids = random_ids(2, 16 * 4 + 7)
    h = logits_of(model, ids, vertical=False)
    v = logits_of(model, ids, vertical=True)
    print(f"{kind} in_stream={in_stream}: max |H - V| = {max_diff(h, v):.3e}, bitwise equal: {torch.equal(h, v)}")
    assert max_diff(h, v) <= ATOL


@pytest.mark.parametrize("kind", KINDS)
def test_memory_changes_outputs(kind):
    """Sanity: with an active memory later segments differ from the base model, earlier ones do not."""
    base = tiny_base(kind)
    model = tiny_armt(kind)
    ids = random_ids(2, 16 * 3)
    ref = base_logits(base, ids)
    out = logits_of(model, ids)
    assert max_diff(out[:, :16], ref[:, :16]) < tol_vs(ref, kind)      # first segment: memory still empty
    assert max_diff(out[:, 16:], ref[:, 16:]) > 1e-3


# ------------------------------------------------------------------- 2. generation == forward, streaming
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("in_stream", OPTIONS)
@pytest.mark.parametrize("vertical", [False, True])
def test_generation_matches_forward(kind, in_stream, vertical):
    model = tiny_armt(kind, mem_tokens_in_stream=in_stream)
    model.vertical_mode = vertical
    prompt = random_ids(2, 37)
    new, step_logits = model.generate(prompt, max_new_tokens=45, return_logits=True)   # crosses 5 segment ends
    full = logits_of(model, torch.cat([prompt, new[:, :-1]], dim=1), vertical)
    ref = full[:, prompt.shape[1] - 1:]
    print(f"{kind} in_stream={in_stream} vertical={vertical}: max |gen - fwd| = {max_diff(step_logits, ref):.3e}")
    assert max_diff(step_logits, ref) < KTOL
    assert torch.equal(step_logits.argmax(-1), ref.argmax(-1))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("cuts", [(5,), (16,), (3, 20, 21, 50), (33, 34, 35)])
def test_pieces_equal_whole(kind, cuts):
    """Feeding the sequence in pieces of any length (passing the state) == one forward."""
    model = tiny_armt(kind)
    ids = random_ids(2, 70)
    whole = logits_of(model, ids)
    parts, state, prev = [], None, 0
    for cut in list(cuts) + [ids.shape[1]]:
        with torch.no_grad():
            out = model(input_ids=ids[:, prev:cut], state=state, return_state=True)
        parts.append(out.logits)
        state, prev = out.state, cut
    assert max_diff(torch.cat(parts, dim=1), whole) < KTOL


# -------------------------------------------------------------------------------- 3. constant state size
def _signature(state):
    return [(tuple(t.shape), t.dtype) for t in state.tensors()]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("in_stream", OPTIONS)
def test_state_size_constant(kind, in_stream):
    if kind in HYBRIDS:
        pytest.skip("the KV caches of the attention layers grow: test_hybrid_state_size")
    model = tiny_armt(kind, mem_tokens_in_stream=in_stream)
    init = model.init_state(2)
    sig, nbytes = _signature(init), init.nbytes()
    assert nbytes > 0
    # after sequences of very different lengths
    for length in (1, 15, 16, 17, 100, 400):
        with torch.no_grad():
            st = model(input_ids=random_ids(2, length), return_state=True).state
        assert _signature(st) == sig and st.nbytes() == nbytes, length
        assert st.position == length % 16
    # and after every generated token
    with torch.no_grad():
        out = model(input_ids=random_ids(2, 9), return_state=True)
        st, sizes = out.state, set()
        for step in range(40):
            out = model(input_ids=out.logits[:, -1:].argmax(-1), state=st, return_state=True)
            st = out.state
            assert _signature(st) == sig
            sizes.add(st.nbytes())
    assert sizes == {nbytes}
    print(f"{kind} in_stream={in_stream}: state = {len(sig)} tensors, {nbytes} bytes for batch 2")


# ---------------------------------------------------------------------------------- 4. gradients
def _cut(state: ARMTState, rnn: bool, memory: bool) -> ARMTState:
    detach = lambda tree: map_tensors(torch.Tensor.detach, tree)
    return ARMTState([(detach(m) if memory else m, detach(r) if rnn else r) for m, r in state.layers], state.position)


def _grad_to_first_segment(model, ids, detach_rnn=False, detach_memory=False):
    """Loss on the last segment only; the sequence is fed segment by segment, optionally cutting the
    recurrent state and / or the associative memory between segments. Returns d loss / d (embeddings)."""
    S = model.config.segment_size
    x = model.backbone.embed(ids).detach().requires_grad_(True)
    state, out = None, None
    for s in range(0, ids.shape[1], S):
        if state is not None:
            state = _cut(state, detach_rnn, detach_memory)
        with train_autocast(model):
            out = model(inputs_embeds=x[:, s:s + S], labels=ids[:, s:s + S], state=state, return_state=True)
        state = out.state
    out.loss.backward()
    return x.grad


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("in_stream", OPTIONS)
def test_gradients_flow_through_state_and_memory(kind, in_stream):
    model = tiny_armt(kind, mem_tokens_in_stream=in_stream).train()
    ids = random_ids(2, 16 * 3)
    S = 16

    def first_seg_grad(**kw):
        model.zero_grad()
        return _grad_to_first_segment(model, ids, **kw)[:, :S].abs().sum().item()

    both = first_seg_grad()
    only_memory = first_seg_grad(detach_rnn=True)
    only_rnn = first_seg_grad(detach_memory=True)
    none = first_seg_grad(detach_rnn=True, detach_memory=True)
    print(f"{kind} in_stream={in_stream}: |dL/dx_seg0| both={both:.3e} memory-only={only_memory:.3e} "
          f"rnn-only={only_rnn:.3e} none={none:.3e}")
    assert both > 0 and only_memory > 0 and only_rnn > 0
    assert none == 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_all_parameters_get_gradients(kind):
    model = tiny_armt(kind).train()
    ids = random_ids(2, 16 * 3)
    with train_autocast(model):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    for name in ["memory"] + [f"memory_layers.{i}.{w}.weight" for i in model.memory_layers for w in ("W_mq", "W_mk", "W_mv", "W_mb")]:
        g = model.get_parameter(name).grad
        assert g is not None and g.abs().sum() > 0, name


@pytest.mark.parametrize("kind", KINDS)
def test_streamed_gradients_equal_whole(kind):
    """Backprop through the state passed between forward calls == backprop through one forward."""
    model = tiny_armt(kind).train()
    ids = random_ids(2, 16 * 3)
    x = model.backbone.embed(ids).detach().requires_grad_(True)
    labels_mask = torch.zeros_like(ids, dtype=torch.bool)
    labels_mask[:, 32:] = True
    with train_autocast(model):
        loss = model(inputs_embeds=x, labels=ids, labels_mask=labels_mask).loss
    loss.backward()
    whole = x.grad.clone()
    model.zero_grad()
    streamed = _grad_to_first_segment(model, ids)
    assert max_diff(whole, streamed) < grad_tol(model) * max(1.0, whole.abs().max().item())


# ------------------------------------------------------------------------------- 5. optimizations
def _loss_and_grads(model, ids, **kw):
    model.zero_grad()
    with train_autocast(model):
        out = model(input_ids=ids, labels=ids, **kw)
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _assert_same(a, b, tol):
    (la, ga), (lb, gb) = a, b
    assert max_diff(la, lb) <= tol, (la, lb)
    assert ga.keys() == gb.keys()
    worst = max(max_diff(ga[k], gb[k]) / max(1.0, ga[k].abs().max().item()) for k in ga)
    assert worst <= tol, worst
    return worst


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("vertical", [False, True])
def test_gradient_checkpointing_is_exact(kind, vertical):
    model = tiny_armt(kind).train()
    model.vertical_mode = vertical
    ids = random_ids(2, 16 * 3 + 3)
    ref = _loss_and_grads(model, ids)
    model.gradient_checkpointing_enable()
    ckpt = _loss_and_grads(model, ids)
    model.gradient_checkpointing_disable()
    print(f"{kind}: grad checkpointing max rel grad diff {_assert_same(ref, ckpt, grad_tol(model)):.3e}")


@pytest.mark.parametrize("kind", KINDS)
def test_vertical_horizontal_gradients(kind):
    model = tiny_armt(kind).train()
    ids = random_ids(2, 16 * 3 + 3)
    model.vertical_mode = False
    h = _loss_and_grads(model, ids)
    model.vertical_mode = True
    v = _loss_and_grads(model, ids)
    print(f"{kind}: V vs H max rel grad diff {_assert_same(h, v, grad_tol(model)):.3e}")


@pytest.mark.parametrize("kind", KINDS)
def test_chunked_loss_is_exact(kind):
    model = tiny_armt(kind).train()
    ids = random_ids(2, 16 * 3 + 3)
    full = _loss_and_grads(model, ids, loss_chunk_size=None)
    chunked = _loss_and_grads(model, ids, loss_chunk_size=7)
    _assert_same(full, chunked, grad_tol(model))
    with torch.no_grad(), train_autocast(model):   # and it is the plain mean token cross-entropy
        logits = model(input_ids=ids).logits
        logits = logits.float() if logits.dtype == torch.bfloat16 else logits
        ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    assert max_diff(full[0], ref) <= grad_tol(model)


@pytest.mark.skipif(not CUDA, reason="CUDA kernels")
@pytest.mark.parametrize("kind", ["mamba", "mamba2"])
def test_kernels_match_reference_scans(kind):
    model = tiny_armt(kind).train()
    ids = random_ids(2, 16 * 3 + 3)
    fast = _loss_and_grads(model, ids)
    backbones.USE_KERNELS = False
    try:
        ref = _loss_and_grads(model, ids)
    finally:
        backbones.USE_KERNELS = True
    # 5e-3: the mamba2 SSD triton kernels multiply in tf32
    print(f"{kind}: kernels vs reference max rel grad diff {_assert_same(ref, fast, 5e-3):.3e}")


# ----------------------------------------------------------------------------------------- save / load
@pytest.mark.parametrize("kind", KINDS)
def test_save_load_roundtrip(kind, tmp_path):
    model = tiny_armt(kind)
    ids = random_ids(2, 40)
    before = logits_of(model, ids)
    model.save_pretrained(tmp_path)
    loaded = RecurrentARMTForCausalLM.from_pretrained(tmp_path, dtype=before.dtype).to(DEVICE).eval()
    assert max_diff(logits_of(loaded, ids), before) == 0.0


# ------------------------------------------------------------------------------- recurrent-state dropout
@pytest.mark.parametrize("kind", KINDS)
def test_state_dropout_same_in_both_orders(kind):
    model = tiny_armt(kind).train()
    model.rnn_state_dropout = 0.5
    ids = random_ids(4, 16 * 4 + 3)
    out = {}
    for vertical in (False, True):
        model.vertical_mode = vertical
        torch.manual_seed(7)
        with torch.no_grad(), train_autocast(model):
            out[vertical] = model(input_ids=ids).logits
    assert max_diff(out[False], out[True]) <= ATOL


@pytest.mark.parametrize("kind", KINDS)
def test_state_reset_is_a_fresh_start_per_segment(kind):
    """state_reset with an empty memory (W_mv = 0) = the base model on every segment separately."""
    if kind in HYBRIDS:
        pytest.skip("the attention layers keep their KV caches: test_hybrid_state_reset_keeps_attention")
    base = tiny_base(kind)
    model = tiny_armt(kind, active_memory=False)
    model.state_reset = True
    ids = random_ids(2, 16 * 3)
    out = logits_of(model, ids)
    ref = torch.cat([base_logits(base, ids[:, s:s + 16]) for s in range(0, 48, 16)], dim=1)
    assert max_diff(out, ref) < tol_vs(ref, kind)
    model.state_reset = False
    assert max_diff(logits_of(model, ids), ref) > 1e-3        # without resets the state crosses the boundaries


@pytest.mark.parametrize("kind", KINDS)
def test_state_reset_leaves_only_the_memory_path(kind):
    model = tiny_armt(kind).train()
    model.state_reset = True
    ids = random_ids(2, 16 * 3)
    S = 16
    g_mem = _grad_to_first_segment(model, ids)[:, :S].abs().sum().item()
    model.zero_grad()
    g_none = _grad_to_first_segment(model, ids, detach_memory=True)[:, :S].abs().sum().item()
    print(f"{kind}: state reset: |dL/dx_seg0| memory {g_mem:.3e}, memory cut {g_none:.3e}")
    assert g_mem > 0
    assert g_none > 0 if kind in HYBRIDS else g_none == 0.0       # a hybrid still has its attention path


# ------------------------------------------------------------------------------- other code paths
@pytest.mark.parametrize("kind", KINDS)
def test_unwrapped_layers(kind):
    """wrap_layers with memory only in some layers: the memory tokens still pass through the unwrapped blocks."""
    backbone = tiny_armt(kind).backbone
    wrap = [1 if i % 2 == 0 and not backbone.is_attention(i) else 0 for i in range(backbone.num_layers)]
    model = tiny_armt(kind, wrap_layers=wrap)
    assert sorted(map(int, model.memory_layers.keys())) == [i for i, w in enumerate(wrap) if w]
    ids = random_ids(2, 16 * 3 + 5)
    h, v = logits_of(model, ids, vertical=False), logits_of(model, ids, vertical=True)
    assert max_diff(h, v) <= ATOL
    new, step_logits = model.generate(ids[:, :20], max_new_tokens=30, return_logits=True)
    full = logits_of(model, torch.cat([ids[:, :20], new[:, :-1]], dim=1))[:, 19:]
    assert max_diff(step_logits, full) < KTOL
    model.train()
    with train_autocast(model):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert all(m.W_mv.weight.grad is not None for m in model.memory_layers.values())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("vertical", [False, True])
def test_last_position_logits(kind, vertical):
    """output_logits="last" (segment by segment, only the last segment kept) == the full forward: logits and state."""
    model = tiny_armt(kind)
    model.vertical_mode = vertical
    ids = random_ids(2, 16 * 3 + 5)
    with torch.no_grad():
        full = model(input_ids=ids, return_state=True)
        last = model(input_ids=ids, output_logits="last", return_state=True)
    assert last.logits.shape[1] == 1 and max_diff(last.logits[:, 0], full.logits[:, -1]) <= ATOL
    assert last.state.position == full.state.position
    assert all(max_diff(a, b) <= ATOL for a, b in zip(last.state.tensors(), full.state.tensors()))


@pytest.mark.skipif(not CUDA, reason="measures GPU memory")
@pytest.mark.parametrize("kind", KINDS)
def test_last_position_logits_memory_does_not_grow(kind):
    """Inference with output_logits="last": the peak memory is the same for an 8x longer input."""
    if kind in HYBRIDS:
        pytest.skip("the KV caches of the attention layers grow")
    model = tiny_armt(kind)

    def peak(length):
        ids = random_ids(8, length)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(input_ids=ids, output_logits="last")
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - before

    peak(16 * 4)                                   # warm-up (kernel autotuning, caches)
    short, long = peak(16 * 8), peak(16 * 64)
    print(f"{kind}: peak extra memory {short / 2**20:.2f} MiB (8 segments) vs {long / 2**20:.2f} MiB (64 segments)")
    assert long <= 1.1 * short + 2**20


def test_generate_stops_per_sample_at_eos():
    model = tiny_armt("mamba")
    ids = random_ids(3, 20)
    free, _ = model.generate(ids, max_new_tokens=12, return_logits=True)
    eos = int(free[0, 2])                         # make sample 0 finish at its 3rd token
    out = model.generate(ids, max_new_tokens=12, eos_token_id=eos)
    first = (out[0] == eos).nonzero()[0].item()
    assert torch.all(out[0, first:] == eos)       # a finished sample keeps emitting EOS
    for b in range(3):                            # before finishing, every sample decodes as without EOS handling
        stop = (out[b] == eos).nonzero()
        k = stop[0].item() if len(stop) else out.shape[1]
        assert torch.equal(out[b, :k], free[b, :k])


@pytest.mark.parametrize("kind", KINDS)
def test_memory_stats(kind):
    zero = tiny_armt(kind, active_memory=False).memory_stats(random_ids(2, 48))
    assert zero["mem/read_ratio"] == 0.0 and zero["mem/W_mv_rms"] == 0.0
    active = tiny_armt(kind).memory_stats(random_ids(2, 48))
    assert active["mem/read_ratio"] > 0 and active["mem/W_mv_rms"] > 0


@pytest.mark.skipif(not HAS_FLA, reason="fla")
def test_attention_models_are_rejected():
    """A model with an fla softmax-attention layer (its state, a KV cache, grows) is refused. fla's Attention needs
    flash-attn to be constructed, so one is planted (uninitialized) into a GDN-2 block."""
    from fla.layers.attn import Attention
    from recurrent_armt.backbones import make_backbone
    model = tiny_base("gdn2")
    make_backbone(model)                                   # fine without attention
    planted = Attention.__new__(Attention)
    torch.nn.Module.__init__(planted)
    model.model.layers[0].attn = planted
    with pytest.raises(ValueError, match="attention"):
        make_backbone(model)


@pytest.mark.skipif(not HAS_FLA, reason="fla")
def test_attention_cache_is_rejected_at_runtime():
    """The runtime guard: a block that keeps an attention (KV) cache is refused even when its class is not one the
    construction-time check knows. Layer 0 of a GDN-2 model is made to store an attn_state like fla's attention does."""
    from recurrent_armt.backbones import make_backbone
    model = tiny_base("gdn2")
    backbone = make_backbone(model)                 # passes the class check
    block = model.model.layers[0]
    original = block.forward

    def forward_with_kv_cache(hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kw):
        out = original(hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, **kw)
        kv = hidden_states.unsqueeze(2)             # [B, T, 1, D]: grows with every token, like a KV cache
        past_key_values.update(attn_state=(kv, kv), layer_idx=0, offset=0)
        return out

    block.forward = forward_with_kv_cache
    x = backbone.embed(random_ids(2, 8))
    with pytest.raises(ValueError, match="attention"):
        backbone.block(0, x, {}, None)
    block.forward = original
    backbone.block(0, x, {}, None)                  # and the unmodified block is accepted


# ----------------------------------------------------------------------------------------------- hybrids
HYBRID_KINDS = [k for k in KINDS if k in HYBRIDS]


@pytest.mark.parametrize("kind", HYBRID_KINDS)
def test_hybrid_memory_only_on_recurrent_layers(kind):
    model = tiny_armt(kind)
    n = model.backbone.num_layers
    attention = [i for i in range(n) if model.backbone.is_attention(i)]
    assert attention and sorted(map(int, model.memory_layers)) == [i for i in range(n) if i not in attention]
    with pytest.raises(ValueError, match="recurrent layers only"):
        tiny_armt(kind, wrap_layers=[1] * n)


@pytest.mark.parametrize("kind", HYBRID_KINDS)
@pytest.mark.parametrize("in_stream", OPTIONS)
def test_hybrid_state_size(kind, in_stream):
    """The recurrent part of the state has a constant size; the KV cache of an attention layer holds one entry per
    real token (the memory tokens are a read-out branch) or per real and memory token (memory tokens in the stream)."""
    model = tiny_armt(kind, mem_tokens_in_stream=in_stream)
    attention = [model.backbone.is_attention(i) for i in range(model.backbone.num_layers)]
    fixed = lambda st: _signature(ARMTState([layer for layer, a in zip(st.layers, attention) if not a]))
    init = model.init_state(2)
    assert all(r is None for (_, r), a in zip(init.layers, attention) if a)
    M = model.config.num_mem_tokens
    for length in (1, 16, 17, 100):
        with torch.no_grad():
            st = model(input_ids=random_ids(2, length), return_state=True).state
        assert fixed(st) == fixed(init), length
        expected = length + (M * (length // 16) if in_stream else 0)
        for (_, kv), a in zip(st.layers, attention):
            if a:
                assert kv[0].shape[2] == kv[1].shape[2] == expected, (length, kv[0].shape)


@pytest.mark.parametrize("kind", HYBRID_KINDS)
def test_hybrid_state_reset_keeps_attention(kind):
    """state_reset zeroes the recurrent states at every boundary; the attention layers still see the whole context:
    the same as a base model whose GDN layers restart at every segment while its attention layers do not."""
    model = tiny_armt(kind, active_memory=False)
    ids = random_ids(2, 16 * 3)
    model.state_reset = True
    with torch.no_grad():
        st = model(input_ids=ids, return_state=True).state
    reset = logits_of(model, ids)
    model.state_reset = False
    assert max_diff(reset[:, :16], logits_of(model, ids)[:, :16]) <= ATOL      # the first segment is untouched
    # reference: segment by segment, zeroing the recurrent states (only) in between
    state, parts = None, []
    for s in range(0, ids.shape[1], 16):
        if state is not None:
            state = ARMTState([(m, r if model.backbone.is_attention(i) else map_tensors(torch.zeros_like, r))
                               for i, (m, r) in enumerate(state.layers)], state.position)
        with torch.no_grad():
            out = model(input_ids=ids[:, s:s + 16], state=state, return_state=True)
        parts.append(out.logits)
        state = out.state
    assert max_diff(reset, torch.cat(parts, dim=1)) <= ATOL
    assert all(st.layers[i][1][0].shape[2] == 48 for i in range(len(st.layers)) if model.backbone.is_attention(i))


@pytest.mark.skipif(not CUDA, reason="CUDA kernels")
@pytest.mark.parametrize("kind", HYBRID_KINDS)
def test_hybrid_kernels_match_reference(kind):
    """fla's chunked gated delta rule + causal_conv1d vs the step-by-step reference."""
    model = tiny_armt(kind).train()
    ids = random_ids(2, 16 * 3 + 3)
    fast = _loss_and_grads(model, ids)
    backbones.USE_KERNELS = False
    try:
        ref = _loss_and_grads(model, ids)
    finally:
        backbones.USE_KERNELS = True
    print(f"{kind}: kernels vs reference max rel grad diff {_assert_same(ref, fast, 5e-3):.3e}")
