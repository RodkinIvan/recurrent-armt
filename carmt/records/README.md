# carmt records

Raw outputs behind the numbers in `carmt/README.md` and `reports/carmt_speedup.md`. Measured on 12 Nov 2026 (GDN-2) and 13 Nov 2026
(Qwen3.5 hybrid) on three hosts:
- `mbz_nlp2`: RTX 6000 Ada;
- `mbz_nlp2.07`: RTX 5880 Ada (2 GPUs);
- `mbz_nlp2.14`: RTX 5880 Ada (GPU 1; the original's fla / causal-conv1d kernels are installed there).

"Current build" = the default `libarmt_bf16.so` / `libarmt_fp32.so` of the current `armt.cu`, with the GEMM tuning, the
lighter backward recompute and the per-layer backward nodes. "Previous build" = the build of the first version of the
report, before those changes.

| file | what | build |
|---|---|---|
| `tests_rtx6000.log`, `tests_rtx5880.log` | `pytest carmt/tests -rA`: 48 passed | current |
| `dist_check_2x_rtx5880.log` | `torchrun --nproc_per_node 2 carmt/tests/dist_check.py`: DDP, ZeRO-1, ZeRO-2, run from an empty GEMM cache (every shape tuned during the run) | current |
| `bench_rtx6000_bf16.json` | `carmt/bench.py`: 1-GPU training / inference / generation, carmt and the original in one run; `peak_gb` = `torch.cuda.max_memory_allocated` (the library allocates through PyTorch) | current |
| `profile_rtx6000_carmt_train.log`, `profile_rtx6000_original_train.log` | `carmt/profile_kernels.py --what train`: per-kernel GPU time of one training step | carmt: current |
| `gemm_tune_rtx6000.log` | `ARMT_GEMM_LOG=1`, empty cache: every tuned GEMM shape of a training step (73), cuBLAS's default vs the chosen kernel | current |
| `overlap_ceiling_rtx6000.log` | two training steps on two CUDA streams at once vs back to back: the ceiling of a two-stream schedule | current |
| `long_context_rtx6000.log` | `carmt/long_context.py`: memory and throughput vs context length, batch 1; mode "last" (the original after its constant-memory fix) and mode "loss" | current; both in the same session |
| `train_ddp_rtx6000_ab_{old,new}{1,2}.log` | 30-step 2-GPU DDP runs of `carmt/train_carmt.py`, previous (old) and current (new) build alternated: 34.3k / 34.4k vs 36.2k / 36.2k tok/s, same losses | both |
| `ddp_layer_nodes_rtx6000.log` | the per-layer backward nodes in DDP: 30-step jobs with `CARMT_LAYER_NODES=0` / `1` alternated (36.1k tok/s either way), and the all-reduce time the backward does not hide (sync vs `no_sync` micro-step: 0.24 s with, 0.32 s without), with the bucket launch times | current |
| `train_ddp_rtx6000_carmt_final30.log` | = `train_ddp_rtx6000_ab_new2.log`: the DDP throughput in the reports | current |
| `train_ddp_rtx6000_{original,carmt}.{log,jsonl}` | 100-step 2-GPU DDP runs, `train.py` vs `carmt/train_carmt.py`, same arguments, data and seed: the loss parity and the original's 26.3k tok/s | carmt: an earlier build (re-packed weights every micro-step) |
| **Qwen3.5 hybrid** (`mbz_nlp2.14` GPU 1 / `mbz_nlp2.07`: RTX 5880 Ada) | | |
| `tests_hybrid_rtx5880.log` | `pytest carmt/tests/test_hybrid.py -rA`: 11 passed | current |
| `qwen35_check_rtx5880.log` | `carmt/qwen35_check.py`: Qwen3.5-0.8B-Base + ARMT vs HF, the original (fla) and the exact recurrence; bf16 gradients | current |
| `dist_check_qwen35_2x_rtx5880.log` | `torchrun --nproc_per_node 2 carmt/tests/dist_check.py --model qwen35`: DDP / ZeRO-1 / ZeRO-2, tiny hybrid fp32 and Qwen3.5-0.8B bf16 | current |
| `bench_rtx5880_qwen35_bf16.json` | `carmt/bench.py --base Qwen/Qwen3.5-0.8B-Base --batch 4`: training (carmt with and without attention checkpointing), inference, generation, carmt and the original in one run | current |
| `train_1x_rtx5880_qwen35_{carmt,original}.log` | 20-step 1-GPU `train.py` runs on Qwen3.5-0.8B-Base, `carmt/train_carmt.py` vs `train.py`, same arguments: 8.8k vs 6.6k tok/s, same losses | current |
| `phases_rtx5880_qwen35_train.log` | one training step by phase (C recurrent fwd / bwd, loss head, PyTorch attention), synchronized at each boundary | current |
| `bench_rtx6000_bf16_splitstate.json` | the optional `-DARMT_SPLIT_STATE` build (hi/lo bf16 state): ~10% slower, which is why it is off | previous, split |
| `bench_rtx6000_decode_after_fix.json` | `bench.py --what decode`, after the original's fix | previous |
| `bench_rtx6000_original_vertical.log` | `bench.py --vertical`: the original in segments-first mode is as fast as layers-first | earlier |
