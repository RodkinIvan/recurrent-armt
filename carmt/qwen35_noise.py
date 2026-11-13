"""bf16 gradient error of carmt vs the original on Qwen3.5-0.8B + ARMT, against the original's own noise.

    python carmt/qwen35_noise.py [--T 2048] [--B 2]

Every variant's per-parameter gradient error (relative norm) is measured against the fp32 original (fla kernels, TF32
off). The ratio err(variant) / err(original bf16, run 1) per parameter is summarised by its median, 95th percentile and
max. The original against itself (a second identical run; backward inside autocast) is the noise floor of that ratio.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@contextlib.contextmanager
def math_sdpa():
    f, e = torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(f)
        torch.backends.cuda.enable_mem_efficient_sdp(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--B", type=int, default=2)
    a = p.parse_args()
    import carmt
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    torch.backends.cuda.matmul.allow_tf32 = False
    base = load_base_model(a.base, dtype=torch.float32).cuda()
    model = RecurrentARMTForCausalLM(RecurrentARMTConfig(num_mem_tokens=32, d_mem=64, segment_size=1024), base_model=base).cuda()
    with torch.no_grad():
        for m in model.memory_layers.values():
            m.W_mv.weight.normal_(std=0.002)
    model.train()
    runner = carmt.CARMT(model)
    ids = torch.randint(3, base.config.vocab_size, (a.B, a.T), device="cuda", generator=torch.Generator("cuda").manual_seed(0))

    def grads(fwd, bf16=True, inside=False, ctx=contextlib.nullcontext):
        model.zero_grad(set_to_none=True)
        with ctx():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                out = fwd(input_ids=ids, labels=ids)
                if inside:
                    out.loss.backward()
            if not inside:
                out.loss.backward()
        return {n: q.grad.detach().to("cpu", copy=True) for n, q in model.named_parameters() if q.grad is not None}

    ref = grads(model, bf16=False)
    runs = {
        "original bf16 (run 1)": grads(model),
        "original bf16 (run 2)": grads(model),
        "original bf16, backward inside autocast": grads(model, inside=True),
        "original bf16, math attention": grads(model, ctx=math_sdpa),
        "carmt bf16": grads(runner),
        "carmt bf16, math attention": grads(runner, ctx=math_sdpa),
    }
    err = {k: {n: rel(g[n], ref[n]) for n in ref} for k, g in runs.items()}
    base_k = "original bf16 (run 1)"
    print(f"{len(ref)} parameters; error vs the fp32 original; ratio = err / err({base_k}) per parameter")
    print(f"{'variant':42s} {'med err':>8s} {'ratio med':>9s} {'p95':>6s} {'max':>6s}  worst")
    for k, e in err.items():
        r = sorted((e[n] / max(err[base_k][n], 1e-12), n) for n in ref)
        q = lambda f: r[min(len(r) - 1, int(f * len(r)))][0]
        med_err = sorted(e.values())[len(e) // 2]
        print(f"{k:42s} {med_err:8.2e} {q(0.5):9.2f} {q(0.95):6.2f} {r[-1][0]:6.2f}  {r[-1][1]}")
    # the parameters the review flagged, every variant
    flag = [n for n in ref if n == "memory" or n.endswith("W_mb.bias") and ".1." in n or "layers.12.linear_attn.dt_bias" in n
            or "layers.0.input_layernorm" in n]
    print("\nflagged parameters, error vs fp32:")
    print(f"{'':48s}" + "".join(f"{k.replace('original', 'orig').replace(' bf16', '')[:18]:>19s}" for k in err))
    for n in flag:
        print(f"{n[-48:]:48s}" + "".join(f"{err[k][n]:19.2e}" for k in err))


if __name__ == "__main__":
    main()
