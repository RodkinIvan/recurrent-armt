"""GPU time per phase of a layer for one carmt training step (forward / the backward's recompute / backward), cuBLAS
included: the library's ARMT_PROF=1 phase profiler.

    python carmt/profile_phases.py [--base checkpoints/gdn2-370m-30B] [--batch 8] [--steps 3]
"""
import argparse
import os
import sys
import time

os.environ["ARMT_PROF"] = "1"            # before the library creates its contexts
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="checkpoints/gdn2-370m-30B")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_segments", type=int, default=8)
    p.add_argument("--segment_size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--what", default="train", choices=["train", "infer"])
    a = p.parse_args()
    import carmt
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    torch.backends.cuda.matmul.allow_tf32 = True
    base = load_base_model(a.base, dtype=torch.float32).cuda()
    model = RecurrentARMTForCausalLM(RecurrentARMTConfig(num_mem_tokens=32, d_mem=64, segment_size=a.segment_size),
                                     base_model=base).cuda()
    with torch.no_grad():
        for m in model.memory_layers.values():
            m.W_mv.weight.normal_(std=0.002)
    r = carmt.CARMT(model)
    T = a.n_segments * a.segment_size
    ids = torch.randint(3, base.config.vocab_size, (a.batch, T), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    model.train(a.what == "train")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.what == "train":
                r(input_ids=ids, labels=ids).loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    r(input_ids=ids, labels=ids)

    for _ in range(2):
        step()
    carmt.prof_report(reset=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    print(f"{a.what} step, batch {a.batch} x {T}: {(time.perf_counter() - t) / a.steps * 1e3:.1f} ms wall per step "
          f"(the profiler synchronizes at every library call)")
    print(carmt.prof_report(reset=True, steps=a.steps))


if __name__ == "__main__":
    main()
