"""Pre-train recurrent ARMT on streamed fineweb-edu.

    torchrun --nproc_per_node 2 train.py --base_model fla-hub/rwkv7-0.4B-g1a --out_dir outputs/rwkv7_0.4b_armt

Every window is ``n_segments * segment_size`` tokens; the loss is the mean next-token loss over the window,
and the log also reports it per segment position (later segments can use the memory of earlier ones).
For a fresh ARMT (W_mv = 0, memory tokens off the recurrent stream) the step-0 evaluation is the base model
itself; not with --mem_tokens_in_stream or --init_from. Resuming from ``out_dir/last`` restores the model, the
optimizer, the schedule, the RNG states (CPU and CUDA: the state-dropout masks) and the exact position of every
rank in the data stream; log lines written after that checkpoint are dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PretrainedConfig

from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
from recurrent_armt.data import PretrainStream, parse_mix, validation_windows


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model
    p.add_argument("--base_model", required=True, help="pretrained recurrent LM (fla RWKV-7 / HF Mamba / Mamba2 / Qwen3.5 hybrid ...)")
    p.add_argument("--tokenizer", default=None, help="default: the base model's")
    p.add_argument("--init_from", default=None, help="an ARMT checkpoint (save_pretrained dir) to start from")
    p.add_argument("--segment_size", type=int, default=1024)
    p.add_argument("--n_segments", type=int, default=8)
    p.add_argument("--num_mem_tokens", type=int, default=32, help="0 = no memory (the plain recurrent model, a baseline)")
    p.add_argument("--d_mem", type=int, default=64)
    p.add_argument("--mem_tokens_in_stream", action="store_true")
    p.add_argument("--mem_init", default="random", choices=["random", "gdn_qk", "gdn_qkb", "gdn_qkvb"],
                   help="initialize the memory from one GDN-2 head per layer (recurrent_armt/memory_init.py; needs "
                        "--d_mem = the GDN head_dim): its query / key maps, + its erase gate, + its value path")
    p.add_argument("--mem_init_head", default="slowest", help="which GDN head: slowest (smallest A_log), fastest, or an index")
    p.add_argument("--vertical", action="store_true", help="segments-first execution (default: layers-first)")
    p.add_argument("--freeze_backbone", action="store_true", help="train only the memory parameters")
    p.add_argument("--freeze_backbone_steps", type=int, default=0,
                   help="train only the memory parameters for the first N steps; then the backbone joins with its own warmup")
    p.add_argument("--backbone_lr_restart", action="store_true",
                   help="with --freeze_backbone_steps: the unfrozen backbone warms up to the full --lr and decays over the "
                        "remaining steps (default: it warms up to where the run's cosine has decayed to)")
    p.add_argument("--rnn_state_dropout", type=float, default=0.0,
                   help="reset each sample's recurrent state at a segment boundary with this probability (training)")
    p.add_argument("--rnn_state_dropout_steps", type=int, default=None, help="apply it for the first N steps only")
    p.add_argument("--eval_state_reset", action="store_true",
                   help="also evaluate val_long with the recurrent state reset at every boundary (memory-only context)")
    p.add_argument("--no_grad_checkpointing", action="store_true")
    # optimization
    p.add_argument("--batch_size", type=int, default=8, help="windows per GPU per micro-step")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--stop_step", type=int, default=None,
                   help="stop (evaluate, save, write final/) at this step; the lr schedule still runs over --max_steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--memory_lr_mult", type=float, default=1.0, help="lr multiplier of the ARMT memory parameters")
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    # data
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--units_per_chunk", type=int, default=16, help="row groups (1024 docs each) shuffled together")
    p.add_argument("--mix", default=None, help="data mix as source:row groups per chunk, e.g. fineweb:4,arxiv:1,omr:2 "
                   "(sources in recurrent_armt.data.SOURCES; default fineweb only, --units_per_chunk)")
    p.add_argument("--cooldown", type=int, default=None, help="min distance between windows of one document "
                   "(default: 2 x windows per optimizer step per GPU = no document in the same or adjacent batches)")
    p.add_argument("--min_doc_tokens", type=int, default=0, help="drop shorter documents (e.g. to train on long documents only)")
    p.add_argument("--num_proc", type=int, default=4, help="tokenizer processes per GPU")
    p.add_argument("--doc_sep_token_id", type=int, default=None, help="default: the base model's bos (else eos) token")
    # evaluation / logging / checkpoints
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_windows", type=int, default=128)
    p.add_argument("--eval_long_windows", type=int, default=64)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_token_file", default="wandb_token", help="API key file (used when WANDB_API_KEY is not set)")
    return p.parse_args()


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return rank, world, local


def all_reduce(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(t)
    return t


def lr_at(step: int, a, start: int = 0, restart: bool = False) -> float:
    """Linear warmup over warmup_steps from `start` (0 before it: a frozen group), cosine decay to
    min_lr_ratio over the run; with ``restart``, the cosine runs from the end of that warmup to the end of the run
    instead (a group unfrozen late starts from the full lr, not from where the run's schedule has decayed to)."""
    if step < start:
        return 0.0
    warm = min(1.0, (step - start + 1) / a.warmup_steps)
    begin = start + a.warmup_steps if restart else a.warmup_steps
    progress = min(1.0, max(0, step - begin) / max(1, a.max_steps - begin))
    return warm * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def per_segment(token_losses: torch.Tensor, S: int, n: int) -> torch.Tensor:
    """[B, L-1] token losses -> [n] summed loss per segment position (position t predicts token t+1)."""
    padded = torch.nn.functional.pad(token_losses, (0, n * S - token_losses.shape[1]))
    return padded.view(token_losses.shape[0], n, S).sum(dim=(0, 2))


def segment_counts(B: int, S: int, n: int, device) -> torch.Tensor:
    counts = torch.full((n,), float(B * S), device=device)
    counts[-1] -= B                     # the last token of a window has no target
    return counts


@torch.no_grad()
def evaluate(model, windows: np.ndarray, a, rank: int, world: int, batch: int) -> dict:
    model.eval()
    S, n = a.segment_size, a.n_segments
    sums = torch.zeros(n, device="cuda")
    counts = torch.zeros(n, device="cuda")
    mine = windows[rank::world]
    for i in range(0, len(mine), batch):
        x = torch.from_numpy(mine[i:i + batch]).long().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=x)
        sums += per_segment(out.token_losses.float(), S, n)
        counts += segment_counts(x.shape[0], S, n, "cuda")
    all_reduce(sums), all_reduce(counts)
    model.train()
    seg = (sums / counts).tolist()
    return {"loss": (sums.sum() / counts.sum()).item(), "per_segment": seg}


def main():
    a = get_args()
    # the data (and its forked tokenizer workers) is set up before NCCL / CUDA start their threads
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_main = rank == 0
    log = (lambda *x: print(*x, flush=True)) if is_main else (lambda *x: None)
    if a.init_from:              # the architecture comes from the checkpoint; the CLI values must not disagree silently
        ck = PretrainedConfig.get_config_dict(a.init_from)[0]
        for k in ("segment_size", "num_mem_tokens", "d_mem", "mem_tokens_in_stream"):
            if getattr(a, k) != ck[k]:
                log(f"--init_from: using {k}={ck[k]} from the checkpoint (not {getattr(a, k)})")
                setattr(a, k, ck[k])
    window = a.segment_size * a.n_segments
    tokenizer = a.tokenizer or a.base_model

    # ---------------------------------------------------------------- data
    sep = a.doc_sep_token_id
    if sep is None:
        cfg = PretrainedConfig.get_config_dict(a.init_from or a.base_model)[0]
        cfg = cfg.get("base_model_config") or cfg.get("text_config") or cfg     # text_config: vision-language checkpoints
        sep = cfg["bos_token_id"] if cfg.get("bos_token_id") is not None else cfg.get("eos_token_id")
    per_step = a.batch_size * a.grad_accum
    cooldown = a.cooldown or 2 * per_step
    mix = parse_mix(a.mix, a.units_per_chunk)
    stream = PretrainStream(tokenizer, window, sep, rank=rank, world_size=world, seed=a.seed,
                            mix=mix, cooldown=cooldown, min_doc_tokens=a.min_doc_tokens,
                            num_proc=a.num_proc)
    val = {"val": validation_windows(tokenizer, window, sep, a.eval_windows, num_proc=a.num_proc)}
    if a.eval_long_windows:
        val["val_long"] = validation_windows(tokenizer, window, sep, a.eval_long_windows, long_docs=True, num_proc=a.num_proc)
        for source in mix:        # long held-out documents of every other source of the mix
            if source != "fineweb":
                val[f"val_long_{source}"] = validation_windows(tokenizer, window, sep, a.eval_long_windows, long_docs=True,
                                                               num_proc=a.num_proc, source=source)
    log(f"data: separator {sep}, cooldown {cooldown} windows, validation windows {', '.join(f'{k} {len(v)}' for k, v in val.items())}")

    rank, world, local = setup_distributed()
    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------- model
    if a.init_from:
        model = RecurrentARMTForCausalLM.from_pretrained(a.init_from, dtype=torch.float32)
    else:
        cfg = RecurrentARMTConfig(base_model_name=a.base_model, num_mem_tokens=a.num_mem_tokens, d_mem=a.d_mem,
                                  segment_size=a.segment_size, mem_tokens_in_stream=a.mem_tokens_in_stream)
        model = RecurrentARMTForCausalLM(cfg)
    model.vertical_mode = a.vertical
    if not a.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.cuda().train()
    if a.mem_init != "random" and not a.init_from:   # a resume loads its own memory weights over this below
        from recurrent_armt.memory_init import init_memory_from_gdn
        # calibrate on training data (this rank's first 4 windows, which it then also trains on), not on held-out
        # windows: DDP copies rank 0's parameters to the other ranks when it wraps the model, so rank 0's init is used
        calib = torch.from_numpy(stream.peek(4)).long().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            report = init_memory_from_gdn(model, a.mem_init.removeprefix("gdn_"), calib, head=a.mem_init_head)
        log(f"memory init {a.mem_init}: " + "; ".join(
            f"L{r['layer']} h{r['head']}" + "".join(f" {k} {r[k]:.3g}" for k in ("beta_mean", "gate_mean", "value_rms", "residual_rms") if k in r)
            for r in report))
    n_params = sum(p.numel() for p in model.parameters())
    n_memory = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("model."))
    log(f"model: {type(model.model).__name__} {n_params / 1e6:.1f}M params ({n_memory / 1e6:.2f}M ARMT), window {window}")

    # parameters that are ever trained (empty ones, e.g. the memory tokens with --num_mem_tokens 0, never: DDP
    # would complain about them); the backbone group starts its warmup when it is unfrozen
    trained = lambda p: p.numel() > 0
    memory_params = [p for n, p in model.named_parameters() if not n.startswith("model.") and trained(p)]
    backbone_params = [] if a.freeze_backbone else [p for n, p in model.named_parameters() if n.startswith("model.") and trained(p)]
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []
    for name, params, mult, start in (("backbone", backbone_params, 1.0, a.freeze_backbone_steps),
                                      ("memory", memory_params, a.memory_lr_mult, 0)):
        # no weight decay on vectors (norms, biases, decay / mixing coefficients)
        groups += [{"params": [p for p in params if p.dim() >= 2], "weight_decay": a.weight_decay, "lr_mult": mult, "start": start, "name": name},
                   {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0, "lr_mult": mult, "start": start, "name": name}]
    groups = [g for g in groups if g["params"]]
    optimizer = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    def set_trainable(step: int):
        """requires_grad for the groups that train at this step. DDP only reduces the parameters that require
        grad when it is built, so it is (re)built whenever this changes."""
        for g in optimizer.param_groups:
            for p in g["params"]:
                p.requires_grad_(step >= g["start"])
        return DDP(model, device_ids=[local]) if world > 1 else model

    # ---------------------------------------------------------------- resume
    step, last, old = 0, out_dir / "last", out_dir / "last.old"
    if is_main and not (last / "trainer.pt").exists() and (old / "trainer.pt").exists():
        old.rename(last)                # a crash between the two renames of save()
    dist.barrier() if dist.is_initialized() else None
    if (last / "trainer.pt").exists():
        missing, unexpected = model.load_state_dict(load_file(str(last / "model.safetensors"), device="cuda"), strict=False)
        tied = set(getattr(model, "all_tied_weights_keys", {}) or {})
        assert not unexpected and set(missing) <= tied, (missing, unexpected)     # only tied copies are not saved
        st = torch.load(last / "trainer.pt", map_location="cpu", weights_only=False)   # optimizer state moves to the params on load
        optimizer.load_state_dict(st["optimizer"])
        step = st["step"]
        stream.load_state_dict(json.loads((last / f"data_rank{rank}.json").read_text()))
        torch.set_rng_state(st["rng"][rank]["cpu"])
        if "cuda" in st["rng"][rank]:
            torch.cuda.set_rng_state(st["rng"][rank]["cuda"])
        if is_main and (out_dir / "log.jsonl").exists():      # drop what was logged after this checkpoint
            lines = (out_dir / "log.jsonl").read_text().splitlines()
            (out_dir / "log.jsonl").write_text("".join(l + "\n" for l in lines if json.loads(l)["step"] <= step))
        log(f"resumed from {last} at step {step}")
    ddp = set_trainable(step)
    frozen_until = a.freeze_backbone_steps if step < a.freeze_backbone_steps and backbone_params else None
    if frozen_until:
        checksum = sum(p.detach().double().sum().item() for p in backbone_params)
        log(f"backbone frozen until step {frozen_until} (checksum {checksum:.6f})")

    wandb = None
    if a.wandb_project and is_main:
        import re
        import wandb
        token = next((f for f in (Path(a.wandb_token_file), Path.home() / "wandb_token") if f.exists()), None)
        if not os.environ.get("WANDB_API_KEY") and token is not None:
            os.environ["WANDB_API_KEY"] = token.read_text().strip()
        # W&B ignores steps at or below a run's last logged step, so every launch is its own W&B run (a fresh id), in
        # the group <out_dir>: "<out_dir>" for a fresh start, "<out_dir>-resume<step>" for a resume (which redoes the
        # steps after its checkpoint), and a rerun into a reused out_dir does not land in the old run either.
        name = out_dir.name if step == 0 else f"{out_dir.name}-resume{step}"
        run_id = re.sub(r"[^A-Za-z0-9_-]", "-", name)[:100] + "-" + os.urandom(4).hex()
        wandb.init(project=a.wandb_project, group=out_dir.name, name=name, id=run_id, config=vars(a))
    logf = open(out_dir / "log.jsonl", "a") if is_main else None

    def record(entry):
        if is_main:
            logf.write(json.dumps(entry) + "\n")
            logf.flush()
            if wandb:
                wandb.log(entry, step=entry["step"])

    def save():
        """Atomic: everything goes to out_dir/saving; then last -> last.old, saving -> last, last.old is deleted
        (a crash in between leaves last or last.old, which startup recovers)."""
        tmp = out_dir / "saving"
        if is_main:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
        dist.barrier() if dist.is_initialized() else None
        (tmp / f"data_rank{rank}.json").write_text(json.dumps(stream.state_dict()))
        rng = [None] * world
        mine = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
        if dist.is_initialized():
            dist.all_gather_object(rng, mine)
        else:
            rng = [mine]
        if is_main:
            model.save_pretrained(tmp)
            torch.save({"optimizer": optimizer.state_dict(), "step": step, "args": vars(a), "rng": rng}, tmp / "trainer.pt")
            shutil.rmtree(old, ignore_errors=True)
            if last.exists():
                last.rename(old)
            tmp.rename(last)
            shutil.rmtree(old, ignore_errors=True)
            log(f"saved {last} at step {step}")
        dist.barrier() if dist.is_initialized() else None

    def run_eval():
        res = {k: evaluate(model, v, a, rank, world, a.batch_size) for k, v in val.items()}
        if a.eval_state_reset:           # the long-document sets again with the recurrent state reset at every boundary
            model.state_reset = True
            for k in [k for k in val if k.startswith("val_long")]:
                res[f"{k}_reset"] = evaluate(model, val[k], a, rank, world, a.batch_size)
            model.state_reset = False
        entry = {"step": step, **{f"{k}/loss": r["loss"] for k, r in res.items()},
                 **{f"{k}/seg{i}": x for k, r in res.items() for i, x in enumerate(r["per_segment"])}}
        if is_main and model.memory_layers:      # how much the memory does: |read-out| / |hidden|, size of W_mv
            model.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                entry.update(model.memory_stats(torch.from_numpy(val.get("val_long", val["val"])[:2]).long().cuda()))
            model.train()
            log(f"[memory step {step}] read/hidden {entry['mem/read_ratio']:.2e} (max {entry['mem/read_ratio_max']:.2e}), "
                f"W_mv rms {entry['mem/W_mv_rms']:.2e}")
        record(entry)
        log(f"[eval step {step}] " + " | ".join(f"{k} {r['loss']:.4f} per segment " + " ".join(f"{x:.3f}" for x in r["per_segment"])
                                                 for k, r in res.items()))

    if step == 0:
        run_eval()                     # = the base model (ARMT is the identity at init)

    # ---------------------------------------------------------------- train
    batches = stream.batches(a.batch_size)
    S, n = a.segment_size, a.n_segments
    t0, tokens_seen = time.time(), 0
    seg_sums = torch.zeros(n, device="cuda")
    seg_counts = torch.zeros(n, device="cuda")
    end = min(a.max_steps, a.stop_step or a.max_steps)
    while step < end:
        if frozen_until and step == frozen_until:
            now = sum(p.detach().double().sum().item() for p in backbone_params)
            log(f"step {step}: unfreezing the backbone (checksum {now:.6f}, unchanged: {now == checksum})")
            del ddp
            ddp, frozen_until = set_trainable(step), None
        drop_on = a.rnn_state_dropout_steps is None or step < a.rnn_state_dropout_steps
        model.rnn_state_dropout = a.rnn_state_dropout if drop_on else 0.0
        lrs = {}
        for g in optimizer.param_groups:
            g["lr"] = a.lr * g["lr_mult"] * lr_at(step, a, g["start"], restart=a.backbone_lr_restart and g["name"] == "backbone")
            lrs[g["name"]] = g["lr"]
        for micro in range(a.grad_accum):
            x = torch.from_numpy(next(batches)).long().cuda(non_blocking=True)
            sync = micro == a.grad_accum - 1 or world == 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = ddp(input_ids=x, labels=x)
                (out.loss / a.grad_accum).backward()
            seg_sums += per_segment(out.token_losses.float(), S, n)
            seg_counts += segment_counts(x.shape[0], S, n, "cuda")
            tokens_seen += x.numel() * world
        mem_grads = [p.grad.norm() for p in memory_params if p.grad is not None]     # before clipping rescales them
        mem_grad_norm = torch.stack(mem_grads).norm().item() if mem_grads else 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % a.log_every == 0:
            all_reduce(seg_sums), all_reduce(seg_counts)
            seg = (seg_sums / seg_counts).tolist()
            loss = (seg_sums.sum() / seg_counts.sum()).item()
            dt = time.time() - t0
            entry = {"step": step, "train/loss": loss, **{f"train/seg{i}": x for i, x in enumerate(seg)},
                     **{f"lr/{k}": v for k, v in lrs.items()}, "grad_norm": grad_norm.item(), "mem/grad_norm": mem_grad_norm, "tokens_per_s": tokens_seen / dt,
                     "tokens": step * per_step * world * window, "data/chunk": stream.chunk,
                     **{f"data/{k}": v for k, v in stream.stats.items()}}
            record(entry)
            log(f"step {step} loss {loss:.4f} seg0 {seg[0]:.3f} seg{n - 1} {seg[-1]:.3f} lr " + "/".join(f"{v:.1e}" for v in lrs.values()) + " "
                f"gnorm {grad_norm.item():.2f} {tokens_seen / dt / 1e3:.1f}k tok/s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            seg_sums.zero_(), seg_counts.zero_()
            t0, tokens_seen = time.time(), 0
        if step % a.eval_every == 0 or step == end:
            run_eval()
        if step % a.save_every == 0 or step == end:
            save()

    if is_main:
        model.save_pretrained(out_dir / "final")
        log("done")
    stream.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
