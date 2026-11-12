"""carmt under DDP and DeepSpeed ZeRO-1 / ZeRO-2 (2 GPUs):

    torchrun --nproc_per_node 2 carmt/tests/dist_check.py [--ckpt checkpoints/gdn2-370m-30B]

Each rank trains on its own half of a batch. Checked:
  A. tiny model, fp32: the averaged gradients of every wrapper (DDP; DDP with gradient accumulation, with and without
     no_sync; ZeRO-1; ZeRO-2) == the exact PyTorch reference averaged over the two halves; after optimizer steps the ranks
     hold identical parameters and the C forward sees them.
  B. GDN-2 370M, bf16 (if --ckpt exists): DDP / ZeRO-2 gradients == single-process carmt gradients averaged over the halves
     (same library both sides: only the all-reduce summation order differs).
Prints one line per check; exits non-zero on the first failure.
"""
import argparse
import copy
import os
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import carmt  # noqa: E402
from reference import exact_gdn2, rel  # noqa: E402
from test_carmt import ids_of, make  # noqa: E402


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def check(name, ok, detail=""):
    flag = torch.tensor([int(bool(ok))], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    log(f"{'PASS' if flag.item() else 'FAIL'}  {name}  {detail}")
    if not flag.item():
        sys.exit(1)


def grads_of(model, fwd, batches, scale=1.0):
    model.zero_grad(set_to_none=True)
    for ids in batches:
        (fwd(input_ids=ids, labels=ids).loss * scale).backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def ranks_identical(model):
    flat = torch.cat([p.detach().reshape(-1).double() for p in model.parameters()])
    other = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(other, flat)
    return max((o - other[0]).abs().max().item() for o in other)


def worst(g, ref):
    return max((rel(g[n], x), n) for n, x in ref.items())


def part_a():
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    ref_model = make()                                     # the exact PyTorch reference (same init on every rank)
    halves = [[ids_of(2, 64, seed=10 * r + k) for k in range(2)] for r in range(world)]   # 2 micro-batches per rank
    with exact_gdn2():
        per_rank = [grads_of(ref_model, ref_model, [h[0]]) for h in halves]
        ref1 = {n: sum(g[n] for g in per_rank) / world for n in per_rank[0]}
        per_rank2 = [grads_of(ref_model, ref_model, h, scale=0.5) for h in halves]
        ref2 = {n: sum(g[n] for g in per_rank2) / world for n in per_rank2[0]}
    mine = halves[rank]

    # DDP, one step
    m = copy.deepcopy(ref_model)
    carmt.accelerate(m)
    ddp = torch.nn.parallel.DistributedDataParallel(m, device_ids=[torch.cuda.current_device()])
    g = grads_of(m, ddp, [mine[0]])
    e, n = worst(g, ref1)
    check("DDP fp32: averaged gradients == reference", e < 3e-4, f"(worst rel {e:.1e} {n})")
    # DDP, gradient accumulation (all-reduce every micro-step / no_sync on the first)
    g = grads_of(m, ddp, mine, scale=0.5)
    e, n = worst(g, ref2)
    check("DDP fp32: 2-step gradient accumulation == reference", e < 3e-4, f"(worst rel {e:.1e} {n})")
    m.zero_grad(set_to_none=True)
    with ddp.no_sync():
        (ddp(input_ids=mine[0], labels=mine[0]).loss * 0.5).backward()
    (ddp(input_ids=mine[1], labels=mine[1]).loss * 0.5).backward()
    g = {k: p.grad.detach().clone() for k, p in m.named_parameters()}
    e, n = worst(g, ref2)
    check("DDP fp32: accumulation with no_sync == reference", e < 3e-4, f"(worst rel {e:.1e} {n})")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        ddp(input_ids=mine[0], labels=mine[0]).loss.backward()
        opt.step()
    check("DDP fp32: parameters identical across ranks after 3 AdamW steps", ranks_identical(m) == 0)
    with torch.no_grad(), exact_gdn2():
        check("DDP fp32: C forward sees the updated weights",
              rel(m(input_ids=mine[0]).logits, m.forward_reference(input_ids=mine[0]).logits) < 1e-4)

    # DeepSpeed ZeRO-1 / ZeRO-2
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    for stage in (1, 2):
        m = copy.deepcopy(ref_model)
        m.zero_grad(set_to_none=True)      # deepcopy copies the reference's .grad: the engine must start clean
        carmt.accelerate(m)
        cfg = {"train_micro_batch_size_per_gpu": 2, "gradient_accumulation_steps": 1, "zero_allow_untested_optimizer": True,
               "zero_optimization": {"stage": stage, "overlap_comm": stage == 2, "contiguous_gradients": True},
               "gradient_clipping": 0.0, "steps_per_print": 10 ** 9}
        engine, *_ = deepspeed.initialize(model=m, optimizer=torch.optim.AdamW(m.parameters(), lr=1e-3), config=cfg)
        out = engine(input_ids=mine[0], labels=mine[0])
        engine.backward(out.loss)
        g = {k: safe_get_full_grad(p) for k, p in m.named_parameters()}
        e, n = worst({k: v for k, v in g.items() if v is not None}, ref1)
        check(f"ZeRO-{stage} fp32: full gradients == reference", all(v is not None for v in g.values()) and e < 3e-4,
              f"(worst rel {e:.1e} {n})")
        engine.step()
        for _ in range(2):
            out = engine(input_ids=mine[1], labels=mine[1])
            engine.backward(out.loss)
            engine.step()
        check(f"ZeRO-{stage} fp32: parameters identical across ranks after 3 steps", ranks_identical(m) == 0)
        with torch.no_grad(), exact_gdn2():
            check(f"ZeRO-{stage} fp32: C forward sees the updated weights",
                  rel(m(input_ids=mine[0]).logits, m.forward_reference(input_ids=mine[0]).logits) < 1e-4)


def part_b(ckpt):
    from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
    from recurrent_armt.backbones import load_base_model
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    base = load_base_model(ckpt, dtype=torch.float32).cuda()
    m0 = RecurrentARMTForCausalLM(RecurrentARMTConfig(num_mem_tokens=32, d_mem=64, segment_size=1024), base_model=base).cuda()
    with torch.no_grad():
        for memory in m0.memory_layers.values():
            memory.W_mv.weight.normal_(std=0.02)
    m0.train()
    gen = torch.Generator().manual_seed(1)
    halves = [torch.randint(3, 32000, (2, 2048), generator=gen).cuda() for _ in range(world)]
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    def single(model):      # single-process carmt gradients averaged over the halves
        r = carmt.CARMT(model)
        acc = None
        for ids in halves:
            with amp:
                g = grads_of(model, r, [ids])
            acc = g if acc is None else {n: acc[n] + g[n] for n in acc}
        return {n: v / world for n, v in acc.items()}

    ref = single(m0)
    rn = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()
    m = copy.deepcopy(m0)
    carmt.accelerate(m)
    ddp = torch.nn.parallel.DistributedDataParallel(m, device_ids=[torch.cuda.current_device()])
    with amp:
        g = grads_of(m, ddp, [halves[rank]])
    e = max(rn(g[n], ref[n]) for n in ref)
    check("DDP bf16 370M: gradients == single-process carmt average", e < 1e-5, f"(worst rel-norm {e:.1e})")
    import deepspeed
    from deepspeed.utils import safe_get_full_grad
    m = copy.deepcopy(m0)
    m.zero_grad(set_to_none=True)
    carmt.accelerate(m)
    cfg = {"train_micro_batch_size_per_gpu": 2, "gradient_accumulation_steps": 1, "zero_allow_untested_optimizer": True,
           "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True}, "steps_per_print": 10 ** 9,
           # DeepSpeed switches an outer torch.autocast off inside engine.forward unless its own torch_autocast is on
           # (its "bf16" section instead converts the parameters to bf16, which carmt does not take)
           # lower_precision_safe_modules = []: all gradients are all-reduced in fp32 (by default DeepSpeed reduces those
           # of nn.Linear in bf16, which adds bf16 rounding to the comparison, for any model)
           "torch_autocast": {"enabled": True, "dtype": "bfloat16", "lower_precision_safe_modules": []}}
    engine, *_ = deepspeed.initialize(model=m, optimizer=torch.optim.AdamW(m.parameters(), lr=1e-5), config=cfg)
    out = engine(input_ids=halves[rank], labels=halves[rank])
    engine.backward(out.loss)
    errs = {n: rn(safe_get_full_grad(p), ref[n]) for n, p in m.named_parameters()}
    e = max(errs.values())
    worst_n = max(errs, key=errs.get)
    check("ZeRO-2 bf16 370M: gradients == single-process carmt average", e < 1e-5, f"(worst rel-norm {e:.1e} {worst_n})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/gdn2-370m-30B")
    a = p.parse_args()
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    log(f"world {dist.get_world_size()} | {torch.cuda.get_device_name()} | NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')}")
    part_a()
    if os.path.exists(a.ckpt):
        part_b(a.ckpt)
    log("ALL DISTRIBUTED CHECKS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
