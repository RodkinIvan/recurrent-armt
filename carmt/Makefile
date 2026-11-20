# Builds the three variants of the library: bf16 (training / fast inference: bf16 matmuls, bf16 scan intermediates),
# bf16s32 (bf16 matmuls, fp32 scan: the scan_fp32 option) and fp32 (fp32 everywhere: the reference checks).
# CUDA_HOME: a CUDA >= 11.8 toolkit (nvcc, cuda_runtime.h, cublas); default: the active conda env, else /usr/local/cuda.
CUDA_HOME ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX),/usr/local/cuda)
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      ?= sm_89
# the first 16 hex digits of armt.cu's sha256, built into every library (armt_source_hash): carmt/__init__.py refuses a
# library built from another armt.cu than the one next to it
SRC_HASH  := $(shell (sha256sum armt.cu 2>/dev/null || shasum -a 256 armt.cu) | cut -c1-16)
FLAGS     := -O3 -arch=$(ARCH) -std=c++17 -Xcompiler -fPIC -shared -I$(CUDA_HOME)/include -I$(CUDA_HOME)/targets/x86_64-linux/include \
             -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -Xlinker -rpath=$(CUDA_HOME)/lib -Xlinker -rpath=$(CUDA_HOME)/lib64 \
             -DARMT_SRC_HASH='"$(SRC_HASH)"'

all: libarmt_bf16.so libarmt_fp32.so libarmt_bf16s32.so

libarmt_bf16.so: armt.cu
	$(NVCC) $(FLAGS) armt.cu -lcublas -lcublasLt -o $@

libarmt_bf16s32.so: armt.cu
	$(NVCC) $(FLAGS) -DARMT_SCAN32 armt.cu -lcublas -lcublasLt -o $@

libarmt_fp32.so: armt.cu
	$(NVCC) $(FLAGS) -DARMT_FP32 armt.cu -lcublas -lcublasLt -o $@

libarmt_bf16split.so: armt.cu      # optional: hi/lo bf16 state in the forward products (see ARMT_SPLIT_STATE)
	$(NVCC) $(FLAGS) -DARMT_SPLIT_STATE armt.cu -lcublas -lcublasLt -o $@

clean:
	rm -f libarmt_*.so
