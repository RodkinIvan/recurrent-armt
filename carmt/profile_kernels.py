"""Per-kernel GPU time of one carmt training step / inference forward (torch.profiler; CUPTI sees the C library's kernels).

    python carmt/profile_kernels.py --base checkpoints/gdn2-370m-30B [--what train|infer] [--original]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="checkpoints/gdn2-370m-30B")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_segments", type=int, default=8)
    p.add_argument("--segment_size", type=int, default=1024)
    p.add_argument("--what", default="train")
    p.add_argument("--original", action="store_true")
    p.add_argument("--rows", type=int, default=40)
    a = p.parse_args()
    import carmt
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    torch.backends.cuda.matmul.allow_tf32 = True
    base = load_base_model(a.base, dtype=torch.float32).cuda()
    model = RecurrentARMTForCausalLM(RecurrentARMTConfig(num_mem_tokens=32, d_mem=64, segment_size=a.segment_size), base_model=base).cuda()
    fwd = model if a.original else carmt.CARMT(model)
    if a.original:
        model.gradient_checkpointing_enable()
    ids = torch.randint(3, 32000, (a.batch, a.n_segments * a.segment_size), device="cuda")

    def run():
        if a.what == "train":
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd(input_ids=ids, labels=ids)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                fwd(input_ids=ids, labels=ids)
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    tot = sum(k.device_time_total for k in ka if k.device_time_total > 0 and k.key and not k.key.startswith("ProfilerStep"))
    rows = sorted([k for k in ka if k.device_time_total > 0], key=lambda k: -k.device_time_total)
    print(f"total GPU time {tot / 1e3:.1f} ms")
    for k in rows[:a.rows]:
        print(f"{k.device_time_total / 1e3:9.1f} ms {100 * k.device_time_total / tot:5.1f}%  {k.count:6d}  {k.key[:110]}")


if __name__ == "__main__":
    main()
