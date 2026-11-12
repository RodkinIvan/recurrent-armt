"""Speed of the CUDA implementation (carmt) vs the original PyTorch/fla model, on a GDN-2 ARMT.

    python carmt/bench.py --base checkpoints/gdn2-370m-30B [--batch 8 --n_segments 8 --segment_size 1024]

Measures (bf16 autocast unless --fp32):
  train   forward + backward of a training window (the original with gradient checkpointing, as train.py runs it,
          and without when it fits), tokens / s
  infer   forward without grad (loss over the window), tokens / s
  decode  generation, ms / token (batch 1)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def timed(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="checkpoints/gdn2-370m-30B")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_segments", type=int, default=8)
    p.add_argument("--segment_size", type=int, default=1024)
    p.add_argument("--num_mem_tokens", type=int, default=32)
    p.add_argument("--d_mem", type=int, default=64)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--vertical", action="store_true", help="run the original segments-first (model.vertical_mode); carmt is unaffected")
    p.add_argument("--what", default="train,infer,decode")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    import carmt
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model

    torch.backends.cuda.matmul.allow_tf32 = True      # as train.py
    torch.backends.cudnn.allow_tf32 = True
    base = load_base_model(a.base, dtype=torch.float32).cuda()
    cfg = RecurrentARMTConfig(num_mem_tokens=a.num_mem_tokens, d_mem=a.d_mem, segment_size=a.segment_size)
    model = RecurrentARMTForCausalLM(cfg, base_model=base).cuda()
    model.vertical_mode = a.vertical
    with torch.no_grad():
        for m in model.memory_layers.values():
            m.W_mv.weight.normal_(std=0.002)
    runner = carmt.CARMT(model)
    T = a.n_segments * a.segment_size
    ids = torch.randint(3, base.config.vocab_size, (a.batch, T), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32)
    res = {"gpu": torch.cuda.get_device_name(), "batch": a.batch, "window": T, "segment_size": a.segment_size,
           "num_mem_tokens": a.num_mem_tokens, "precision": "fp32" if a.fp32 else "bf16",
           "original_mode": "vertical" if a.vertical else "horizontal"}
    tok = a.batch * T
    print(json.dumps(res))

    def report(key, sec, ntok=tok):
        res[key] = {"s": sec, "tok_per_s": ntok / sec}
        print(f"{key:28s} {sec * 1e3:9.1f} ms   {ntok / sec / 1e3:8.1f} k tok/s", flush=True)

    whats = a.what.split(",")
    if "train" in whats:
        model.train()

        def step(fwd):
            def f():
                with amp:
                    out = fwd(input_ids=ids, labels=ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
            return f
        torch.cuda.reset_peak_memory_stats()
        report("train/carmt", timed(step(runner), iters=a.iters))
        res["train/carmt"]["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        model.gradient_checkpointing_enable()
        torch.cuda.reset_peak_memory_stats()
        report("train/original_ckpt", timed(step(model), warmup=1, iters=max(2, a.iters // 2)))
        res["train/original_ckpt"]["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        model.gradient_checkpointing_disable()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            report("train/original_nockpt", timed(step(model), warmup=1, iters=max(2, a.iters // 2)))
            res["train/original_nockpt"]["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        except torch.OutOfMemoryError:
            print("train/original_nockpt        OOM")
            res["train/original_nockpt"] = "OOM"
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        model.eval()
    if "infer" in whats:
        with torch.no_grad(), amp:
            report("infer/carmt", timed(lambda: runner(input_ids=ids, labels=ids), iters=a.iters))
            report("infer/original", timed(lambda: model(input_ids=ids, labels=ids), warmup=1, iters=max(2, a.iters // 2)))
    if "decode" in whats:
        prompt = ids[:1, :a.segment_size - 8]
        n_new = 64          # crosses a segment end (memory tokens processed once)

        def gen(fwd):
            def f():
                st = fwd(input_ids=prompt, return_state=True, output_logits="last").state
                tokn = prompt[:, -1:]
                for _ in range(n_new):
                    o = fwd(input_ids=tokn, state=st, return_state=True, output_logits="last")
                    st, tokn = o.state, o.logits[:, -1].argmax(-1, keepdim=True)
            return f
        with torch.no_grad(), amp:
            for name, fwd in (("carmt", runner), ("original", model)):
                sec = timed(gen(fwd), warmup=1, iters=2)
                res[f"decode/{name}"] = {"ms_per_token": sec / n_new * 1e3}
                print(f"decode/{name:21s} {sec / n_new * 1e3:9.2f} ms/token (incl. prompt of {prompt.shape[1]})", flush=True)
    for k in ("train", "infer"):
        if f"{k}/carmt" in res and isinstance(res.get(f"{k}/original_ckpt", res.get(f"{k}/original")), dict):
            ref = res.get(f"{k}/original_ckpt", res.get(f"{k}/original"))
            print(f"speedup {k}: {ref['s'] / res[f'{k}/carmt']['s']:.2f}x")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
