"""Real pretrained recurrent LMs (GPU; downloads small checkpoints).

In fp32 the Triton kernels (mamba_ssm SSD, fla) multiply in TF32, so two chunkings of the same sequence
differ by ~1e-3 relative on real models. The bounds are therefore set by a control: the base model's own
cached decoding (``generate``) against its own forward on the same tokens. ARMT must agree with the base
model (at init) and with itself (generation vs forward) about as well as the base model agrees with itself.
"""
from __future__ import annotations

import os

import pytest
import torch

from helpers import CUDA, HAS_FLA, max_diff
from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
from recurrent_armt.backbones import load_base_model

pytestmark = pytest.mark.skipif(not CUDA, reason="needs a GPU")

GDN2 = "checkpoints/gdn2-370m-30B"          # scripts/convert_gdn2.py (not annealed GDN-2, 30B tokens)
CHECKPOINTS = ["state-spaces/mamba-130m-hf", "AntonV/mamba2-130m-hf"] + (["fla-hub/rwkv7-191M-world"] if HAS_FLA else []) \
    + ([GDN2] if HAS_FLA and os.path.exists(GDN2) else []) + (["Qwen/Qwen3.5-0.8B-Base"] if HAS_FLA else [])   # hybrid
N_NEW = 140


@pytest.fixture(scope="module", params=CHECKPOINTS)
def models(request):
    """`base` is a separately loaded copy: if wrapping the model in ARMT changed its weights, ARMT@init would
    no longer match it."""
    base = load_base_model(request.param, dtype=torch.float32).cuda().eval()
    wrapped = load_base_model(request.param, dtype=torch.float32)
    cfg = RecurrentARMTConfig(num_mem_tokens=16, d_mem=32, segment_size=128)
    armt = RecurrentARMTForCausalLM(cfg, base_model=wrapped).cuda().eval()
    return request.param, base, armt


def ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 20000, (2, n), generator=g).cuda()


def cached_greedy(base, x, n):
    from recurrent_armt.backbones import _fla_cache_cls
    cache = _fla_cache_cls()()
    out, seq, logits = base(input_ids=x, past_key_values=cache, use_cache=True), x, []
    for _ in range(n):
        logits.append(out.logits[:, -1].float())
        token = logits[-1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, token], dim=1)
        out = base(input_ids=token, past_key_values=cache, use_cache=True)
    return torch.stack(logits, dim=1), seq


def ce(logits, targets):
    return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten()).item()


@pytest.fixture(scope="module")
def control(models):
    """|base cached decoding - base forward| on the base model's own greedy continuation."""
    name, base, _ = models
    x = ids(128 * 2 + 30)
    with torch.no_grad():
        if name == GDN2:                     # no HF generate for our GDN-2 LM: greedy decoding with the fla cache
            step_logits, sequences = cached_greedy(base, x, N_NEW)
        else:
            out = base.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=N_NEW, min_new_tokens=N_NEW,
                                do_sample=False, output_logits=True, return_dict_in_generate=True, pad_token_id=0)
            step_logits, sequences = torch.stack(out.logits, dim=1).float(), out.sequences
        full = base(input_ids=sequences[:, :-1]).logits[:, x.shape[1] - 1:].float()
    d = max_diff(step_logits, full) / full.abs().max().item()
    print(f"{name}: control |base generate - base forward| rel {d:.2e}")
    return max(d, 1e-5)


def test_init_equals_base(models, control):
    name, base, armt = models
    x = ids(128 * 3 + 50)
    with torch.no_grad():
        ref = base(input_ids=x).logits.float()
        for vertical in (False, True):
            armt.vertical_mode = vertical
            out = armt(input_ids=x).logits.float()
            rel = max_diff(out, ref) / ref.abs().max().item()
            dce = abs(ce(out[:, :-1], x[:, 1:]) - ce(ref[:, :-1], x[:, 1:]))
            agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
            print(f"{name} vertical={vertical}: |ARMT@init - base| rel {rel:.2e} (control {control:.2e}), "
                  f"|delta CE| {dce:.1e}, argmax agree {agree:.4f}")
            assert rel < 3 * control and dce < 1e-3 and agree > 0.99


def test_active_memory_equivalences(models, control):
    name, base, armt = models
    with torch.no_grad():
        for memory in armt.memory_layers.values():
            memory.W_mv.weight.normal_(std=0.002)   # 0.02 adds ~50% random noise to every layer of a 768-d model: chaotic
    x = ids(128 * 2 + 30)
    with torch.no_grad():
        armt.vertical_mode = False
        h = armt(input_ids=x).logits
        armt.vertical_mode = True
        v = armt(input_ids=x).logits
        new, step_logits = armt.generate(x, max_new_tokens=N_NEW, return_logits=True)
        full = armt(input_ids=torch.cat([x, new[:, :-1]], dim=1)).logits[:, x.shape[1] - 1:]
    scale = h.abs().max().item()
    gen = max_diff(step_logits, full) / scale
    agree = (step_logits.argmax(-1) == full.argmax(-1)).float().mean().item()
    print(f"{name}: |H - V| = {max_diff(h, v):.3e} (bitwise {torch.equal(h, v)}), |gen - fwd| rel {gen:.2e} "
          f"(control {control:.2e}), argmax agree {agree:.4f}")
    if "Qwen3.5" in name:     # hybrid: layers-first runs each attention layer over all segments at once (other rounding)
        assert max_diff(h, v) / scale < control
    else:
        assert torch.equal(h, v)
    assert gen < 3 * control and agree > 0.99
