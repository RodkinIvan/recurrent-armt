"""State tracking: a tiny GDN-2, trained from scratch, with and without recurrent ARMT, on a group word problem.

    python train_state_tracking.py --task a5 --num_mem_tokens 4 --segment_size 2 --out_dir outputs/st_a5_armt_s2_m4
    python train_state_tracking.py --task a5 --num_mem_tokens 0 --out_dir outputs/st_a5_nomem

At every position t the model classifies y_t = x_1 ... x_t (no next-token shift). It trains on sequences of
``--train_len`` and is evaluated on longer ones (``--eval_lens``): per-position accuracy, mean over the sequence and over
its last tenth. GDN-2's own recurrence is linear in its state; ARMT adds a nonlinear one, once per segment (layer i's
memory-token outputs -> W_i -> layer i's input in the next segment), so the segment size sets how many steps of the
word problem the backbone has to compose on its own.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from recurrent_armt import GDN2Config, GDN2ForCausalLM, RecurrentARMTConfig, RecurrentARMTForCausalLM
from recurrent_armt.state_tracking import WordProblem


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="a5", help="parity / z60 / a5 / s5")
    p.add_argument("--out_dir", required=True)
    # tiny GDN-2
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--mlp", type=int, default=256)
    p.add_argument("--allow_neg_eigval", action="store_true", help="GDN-2 transition eigenvalues in [-1, 1]")
    # ARMT
    p.add_argument("--num_mem_tokens", type=int, default=4, help="0 = the plain GDN-2")
    p.add_argument("--d_mem", type=int, default=32)
    p.add_argument("--segment_size", type=int, default=2)
    # training
    p.add_argument("--train_len", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # evaluation and logging
    p.add_argument("--eval_lens", default="16,64,256,1024")
    p.add_argument("--eval_size", type=int, default=512, help="sequences per evaluated length")
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_state_reset", action="store_true",
                   help="also evaluate with GDN's state reset at every segment boundary (memory-only context)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_token_file", default="wandb_token")
    return p.parse_args(argv)


def build_model(a, vocab: int) -> RecurrentARMTForCausalLM:
    base = GDN2ForCausalLM(GDN2Config(vocab_size=vocab, hidden_size=a.hidden, num_hidden_layers=a.layers, num_heads=a.heads,
                                      head_dim=a.head_dim, intermediate_size=a.mlp, allow_neg_eigval=a.allow_neg_eigval))
    # without memory the segmentation changes nothing (the state passes exactly): one segment per sequence is fastest
    segment = a.segment_size if a.num_mem_tokens > 0 else 1 << 30
    cfg = RecurrentARMTConfig(num_mem_tokens=a.num_mem_tokens, d_mem=a.d_mem, segment_size=segment)
    return RecurrentARMTForCausalLM(cfg, base_model=base)


def token_loss_acc(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x, output_logits=True).logits
    logits = logits.float()
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
    return loss, (logits.argmax(-1) == y).float()


@torch.no_grad()
def evaluate(model, task: WordProblem, a, state_reset: bool = False) -> dict:
    model.eval()
    model.state_reset = state_reset
    tag = "reset/" if state_reset else ""
    out = {}
    for L in [int(s) for s in a.eval_lens.split(",")]:
        g = torch.Generator(device="cuda").manual_seed(1234 + L)        # the same sequences at every evaluation
        loss_sum, acc_sum, n = 0.0, torch.zeros(L, device="cuda"), 0
        for start in range(0, a.eval_size, a.eval_batch_size):
            x, y = task.sample(min(a.eval_batch_size, a.eval_size - start), L, generator=g)
            loss, acc = token_loss_acc(model, x, y)
            loss_sum += loss.sum().item()
            acc_sum += acc.sum(0)
            n += x.shape[0]
        acc = acc_sum / n
        out[f"eval/{tag}loss_L{L}"] = loss_sum / (n * L)
        out[f"eval/{tag}acc_L{L}"] = acc.mean().item()
        out[f"eval/{tag}acc_last_L{L}"] = acc[-max(1, L // 10):].mean().item()
    model.state_reset = False
    model.train()
    return out


def main(argv=None):
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = WordProblem(a.task, device="cuda")
    model = build_model(a, task.size).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    n_memory = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("model."))
    print(f"{a.task} (|G| = {task.size}), {n_params / 1e6:.2f}M parameters ({n_memory / 1e6:.3f}M of them ARMT)", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))

    def lr_at(step):
        if step < a.warmup_steps:
            return a.lr * (step + 1) / a.warmup_steps
        return a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup_steps) / max(1, a.steps - a.warmup_steps)))

    wandb = None
    if a.wandb_project:
        import wandb
        token = next((f for f in (Path(a.wandb_token_file), Path.home() / "wandb_token") if f.exists()), None)
        if not os.environ.get("WANDB_API_KEY") and token is not None:
            os.environ["WANDB_API_KEY"] = token.read_text().strip()
        run_id = re.sub(r"[^A-Za-z0-9_-]", "-", out_dir.name)[:100] + "-" + os.urandom(4).hex()
        wandb.init(project=a.wandb_project, group=out_dir.name, name=out_dir.name, id=run_id,
                   config=vars(a) | {"n_params": n_params, "n_memory_params": n_memory})
    logf = open(out_dir / "log.jsonl", "w")

    def record(entry):
        logf.write(json.dumps(entry) + "\n")
        logf.flush()
        if wandb:
            wandb.log(entry, step=entry["step"])

    model.train()
    t0, loss_acc, acc_acc, k = time.time(), 0.0, 0.0, 0
    for step in range(1, a.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step - 1)
        x, y = task.sample(a.batch_size, a.train_len)
        loss, acc = token_loss_acc(model, x, y)
        loss = loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip).item()
        optimizer.step()
        loss_acc, acc_acc, k = loss_acc + loss.item(), acc_acc + acc.mean().item(), k + 1
        if step % a.log_every == 0:
            entry = {"step": step, "loss": loss_acc / k, "acc": acc_acc / k, "lr": lr_at(step - 1), "grad_norm": grad_norm,
                     "sec_per_step": (time.time() - t0) / k}
            if a.num_mem_tokens > 0:
                entry["w_mv_rms"] = torch.stack([m.W_mv.weight.float().pow(2).mean() for m in model.memory_layers.values()]).mean().sqrt().item()
            print(" ".join(f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}" for k_, v in entry.items()), flush=True)
            record(entry)
            t0, loss_acc, acc_acc, k = time.time(), 0.0, 0.0, 0
        if step % a.eval_every == 0 or step == a.steps:
            entry = {"step": step, **evaluate(model, task, a)}
            if a.eval_state_reset and a.num_mem_tokens > 0:
                entry |= evaluate(model, task, a, state_reset=True)
            print("eval", " ".join(f"{k_}={v:.4f}" for k_, v in entry.items() if k_ != "step"), flush=True)
            record(entry)
    model.save_pretrained(out_dir / "final")
    logf.close()
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
