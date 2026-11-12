"""Run the unchanged train.py with every RecurrentARMTForCausalLM routed through carmt.

    python carmt/train_carmt.py <train.py arguments>
    torchrun --nproc_per_node 2 carmt/train_carmt.py <train.py arguments>

The model class is patched so that each instance (built from a config or loaded with from_pretrained) gets
carmt.accelerate() at the end of its __init__, i.e. before train.py moves it to the GPU and wraps it in DDP.
Building the libraries (if they are missing) happens once, under a file lock, before training starts.
CARMT_SCAN_FP32=1 selects the fp32-scan bf16 build (see carmt.CARMT).
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import carmt  # noqa: E402
import recurrent_armt.modeling as modeling  # noqa: E402

_init = modeling.RecurrentARMTForCausalLM.__init__


def _accelerated_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    carmt.accelerate(self, scan_fp32=os.environ.get("CARMT_SCAN_FP32", "0") == "1")


modeling.RecurrentARMTForCausalLM.__init__ = _accelerated_init

if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():       # build (or find) both libraries up front, not inside the first step
        major, minor = torch.cuda.get_device_capability()
        if not all(os.path.exists(carmt.library_path(p)) for p in ("bf16", "fp32", "bf16s32")):
            carmt.build(arch=f"sm_{major}{minor}")
    train = os.path.join(ROOT, "train.py")
    sys.argv[0] = train
    runpy.run_path(train, run_name="__main__")
