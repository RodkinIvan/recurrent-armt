"""Per-segment validation loss of a plain (base) model on the fineweb-edu validation windows of train.py."""
import argparse
import json

import numpy as np
import torch

import recurrent_armt  # noqa: F401  (registers the GDN-2 LM)
from recurrent_armt.backbones import load_base_model
from recurrent_armt.data import validation_windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--sep", type=int, required=True)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--segment", type=int, default=1024)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    model = load_base_model(a.model, dtype=torch.float32).cuda().eval()
    res = {}
    for name, long_docs in (("val", False), ("val_long", True)):
        w = validation_windows(a.tokenizer or a.model, a.window, a.sep, a.n, long_docs=long_docs)
        losses = []
        for i in range(0, len(w), a.batch):
            x = torch.from_numpy(w[i:i + a.batch]).long().cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x).logits.float()
            losses.append(torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), x[:, 1:], reduction="none").cpu())
        tl = torch.cat(losses)                                            # [N, window-1]
        per_seg = [tl[:, s * a.segment:(s + 1) * a.segment].mean().item() for s in range(a.window // a.segment)]
        res[name] = {"loss": tl.mean().item(), "per_segment": per_seg}
        print(name, f"{tl.mean().item():.4f}", "per segment", " ".join(f"{x:.3f}" for x in per_seg), flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
