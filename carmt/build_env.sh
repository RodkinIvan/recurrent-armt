#!/bin/bash
# Build carmt on a host whose env has no CUDA toolkit (the jobs' env installs prebuilt wheels only).
# Installs nvcc + cuBLAS headers (CUDA 12.6, matching torch's cu126 wheels) into a separate prefix and runs make.
#   bash carmt/build_env.sh [prefix=$HOME/.carmt_cuda] [arch=sm_89]
set -euo pipefail
PREFIX=${1:-$HOME/.carmt_cuda}
ARCH=${2:-sm_89}
HERE=$(cd "$(dirname "$0")" && pwd)
CONDA=${CONDA_EXE:-$(command -v conda || echo "$HOME/miniconda3/bin/conda")}
if [ ! -x "$PREFIX/bin/nvcc" ]; then
    "$CONDA" create -y -p "$PREFIX" -c nvidia/label/cuda-12.6.3 cuda-nvcc cuda-cudart-dev libcublas-dev
fi
make -C "$HERE" CUDA_HOME="$PREFIX" ARCH="$ARCH"
ls -la "$HERE"/libarmt_*.so
