"""Compare two kbench output dumps (relative error per tensor):  python carmt/kbench_cmp.py ref.bin new.bin"""
import struct
import sys

import numpy as np


def load(path):
    out = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(4)
            if not h:
                return out
            name = f.read(struct.unpack("i", h)[0]).decode()
            n = struct.unpack("q", f.read(8))[0]
            out[name] = np.frombuffer(f.read(4 * n), dtype=np.float32).astype(np.float64)


a, b = load(sys.argv[1]), load(sys.argv[2])
worst = 0.0
for k in a:
    e = np.linalg.norm(a[k] - b[k]) / max(np.linalg.norm(a[k]), 1e-30)
    worst = max(worst, e)
    print(f"{k:10s} rel {e:.2e}   max|a| {np.abs(a[k]).max():.3e}   finite {np.isfinite(b[k]).all()}")
print(f"worst {worst:.2e}")
