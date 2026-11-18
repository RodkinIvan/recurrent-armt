import os
import sys

sys.path.insert(0, os.path.dirname(__file__))                    # helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))   # the package

# Triton multiplies fp32 tensors in TF32 by default (Ampere+): ~1e-3 relative per matmul, and how much of it reaches the
# logits depends on the activations (greedy continuations of a random-init model are runs of one token, where it is
# several times larger than on the base model's own continuation). The GPU tests compare in real fp32 instead. Triton
# reads this when it compiles a kernel and its in-process cache does not key on it, so it has to be set before any
# kernel compiles: for the whole session. ``TRITON_F32_DEFAULT=tf32 pytest ...`` runs the tests on the TF32 path.
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
