#!/bin/bash
# Runs inside the freshly created env ($PY = its python). The hosts have driver 535 (CUDA 12.2),
# so torch comes from the cu126 index (minor-version compatible; fla 0.6 needs its triton 3.5) and the CUDA extensions are
# prebuilt wheels for torch 2.9 / cu12 / cxx11abi=TRUE (no compilation on the hosts).
set -euo pipefail
$PY -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install flash-linear-attention==0.6.0 fla-core==0.6.0
REL=https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1
$PY -m pip install --no-deps "$REL/mamba_ssm-2.3.2.post1%2Bcu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
REL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0
$PY -m pip install --no-deps "$REL/causal_conv1d-1.7.0%2Bcu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
$PY -c "import importlib.metadata as m, torch, fla, mamba_ssm, causal_conv1d, transformers; print({p: m.version(p) for p in ['torch', 'triton', 'flash-linear-attention', 'mamba_ssm', 'causal_conv1d', 'transformers']}, torch.version.cuda)"
