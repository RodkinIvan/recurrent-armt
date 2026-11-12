"""Convert a lit_gpt GDN-2 checkpoint (LLM-OS-Models/gdn2-*-fineweb-edu-100b) to a GDN2ForCausalLM directory.

    python scripts/convert_gdn2.py --repo LLM-OS-Models/gdn2-370m-fineweb-edu-100b \
        --file checkpoint-6B-model-ckpt.pth --out checkpoints/gdn2-370m-30B

The checkpoint is loaded with weights_only=True (tensors and plain containers only, no code from the file).
Note the naming: milestone k (`checkpoint-kB-...`) holds 5k B tokens; `step_count` x 4.19M tokens is exact.
"""
import argparse
import shutil
import tempfile

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from recurrent_armt.gdn2 import GDN2Config, GDN2ForCausalLM, lit_gpt_to_hf

CONFIGS = {  # from the model cards
    "370m": dict(hidden_size=1024, num_hidden_layers=16, num_heads=16, head_dim=128, intermediate_size=2048),
    "1.3B": dict(hidden_size=2304, num_hidden_layers=18, num_heads=18, head_dim=128, intermediate_size=6208),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="LLM-OS-Models/gdn2-370m-fineweb-edu-100b")
    p.add_argument("--file", default="checkpoint-6B-model-ckpt.pth")
    p.add_argument("--size", default="370m", choices=list(CONFIGS))
    p.add_argument("--tokenizer", default="TinyLlama/TinyLlama_v1.1")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tmp = tempfile.mkdtemp(dir=".")         # the download (4.5 GB with the optimizer state) is deleted afterwards
    try:
        ckpt = torch.load(hf_hub_download(a.repo, a.file, local_dir=tmp), map_location="cpu", weights_only=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    state = lit_gpt_to_hf(ckpt["model"] if "model" in ckpt else ckpt)
    steps = ckpt.get("step_count")
    model = GDN2ForCausalLM(GDN2Config(**CONFIGS[a.size]))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.config.source = f"{a.repo}/{a.file}" + (f" (step {steps}, {steps * 4194304 / 1e9:.1f}B tokens)" if steps else "")
    model.save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.tokenizer).save_pretrained(a.out)
    print("saved", a.out, "|", model.config.source, "|", sum(p.numel() for p in model.parameters()) / 1e6, "M params")


if __name__ == "__main__":
    main()
