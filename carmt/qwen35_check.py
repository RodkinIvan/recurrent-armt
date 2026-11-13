"""carmt vs the PyTorch implementation on Qwen3.5-0.8B-Base + ARMT (real weights).

    python carmt/qwen35_check.py [--base Qwen/Qwen3.5-0.8B-Base] [--T 2048] [--B 2]

1. ARMT at init (W_mv = 0: the memory is a no-op) equals the base model: carmt vs HF's own forward, fp32.
2. fp32 logits with an active memory: carmt and the original (fla kernels) against the original with its exact
   token-by-token recurrence (TF32 off throughout).
3. bf16 training step: loss and every parameter's gradient against the fp32 original, next to the original's own bf16
   error (the same comparison as the GDN-2 tests).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


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
    model.eval()
    runner = carmt.CARMT(model)
    V = base.config.vocab_size
    ids = torch.randint(3, V, (a.B, a.T), device="cuda", generator=torch.Generator("cuda").manual_seed(0))

    print(f"Qwen3.5 + ARMT, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, B={a.B}, T={a.T}")
    with torch.no_grad():
        hf = base(input_ids=ids).logits
        c0 = runner(input_ids=ids).logits
        o0 = model(input_ids=ids).logits
    print(f"1. ARMT at init vs the HF base model, fp32 logits: carmt {rel(c0, hf):.2e}, original {rel(o0, hf):.2e}")
    del hf, c0, o0

    with torch.no_grad():
        for m in model.memory_layers.values():
            m.W_mv.weight.normal_(std=0.002)
        import recurrent_armt.backbones as bb
        o = model(input_ids=ids, labels=ids)
        c = runner(input_ids=ids, labels=ids)
        lo, lc = model(input_ids=ids).logits, runner(input_ids=ids).logits
        kernels, bb._use_kernels = bb._use_kernels, (lambda x: False)
        try:
            le = model(input_ids=ids).logits
        finally:
            bb._use_kernels = kernels
    print(f"2. active memory, fp32 logits vs the exact recurrence: carmt {rel(lc, le):.2e}, original (fla) {rel(lo, le):.2e}; "
          f"loss carmt {c.loss.item():.6f}, original {o.loss.item():.6f}")
    del lo, lc, le

    model.train()

    def grads(fwd, bf16):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = fwd(input_ids=ids, labels=ids)
        out.loss.backward()
        return out.loss.item(), {n: q.grad.detach().clone() for n, q in model.named_parameters() if q.grad is not None}

    l32, g32 = grads(model, False)
    lob, gob = grads(model, True)
    lcb, gcb = grads(runner, True)
    model.zero_grad(set_to_none=True)
    ec = {n: rel(gcb[n], x) for n, x in g32.items()}
    eo = {n: rel(gob[n], x) for n, x in g32.items()}
    ratios = sorted(ec[n] / max(eo[n], 1e-9) for n in g32)
    worst = sorted(g32, key=lambda n: ec[n] / max(eo[n], 1e-9), reverse=True)[:5]
    print(f"3. bf16 training step: loss carmt {lcb:.5f}, original {lob:.5f}, fp32 {l32:.5f}")
    print(f"   gradients vs fp32, {len(g32)} parameters: carmt's error / the original's bf16 error: median "
          f"{ratios[len(ratios) // 2]:.2f}, max {ratios[-1]:.2f}; median error carmt "
          f"{sorted(ec.values())[len(ec) // 2]:.2e}, original {sorted(eo.values())[len(eo) // 2]:.2e}")
    for n in worst:
        print(f"   {n:70s} carmt {ec[n]:.2e}  original {eo[n]:.2e}")


if __name__ == "__main__":
    main()
