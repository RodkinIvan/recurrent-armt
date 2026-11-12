# carmt records

Raw outputs behind the numbers in `carmt/README.md` and `reports/carmt_speedup.md`. Measured on 12 Nov 2026 on two hosts:
- `mbz_nlp2`: RTX 6000 Ada;
- `mbz_nlp2.07`: RTX 5880 Ada (2 GPUs).

"Final build" = the default `libarmt_bf16.so` / `libarmt_fp32.so` of the current `armt.cu`.

| file | what | build |
|---|---|---|
| `tests_rtx6000.log`, `tests_rtx5880.log` | `pytest carmt/tests -rA`: 47 passed (`tests_rtx6000.log`: against the original after its fix) | final |
| `dist_check_2x_rtx5880.log` | `torchrun --nproc_per_node 2 carmt/tests/dist_check.py`: DDP, ZeRO-1, ZeRO-2 | final |
| `bench_rtx6000_bf16.json` | `carmt/bench.py`: 1-GPU training / inference / generation; `peak_gb` = `torch.cuda.max_memory_allocated` (the library allocates through PyTorch) | final |
| `bench_rtx6000_bf16_splitstate.json` | same, with the optional `-DARMT_SPLIT_STATE` build (hi/lo bf16 state): ~10% slower, which is why it is off | split |
| `profile_rtx6000_carmt_train.log`, `profile_rtx6000_original_train.log` | `carmt/profile_kernels.py --what train`: per-kernel GPU time of one training step | final |
| `long_context_rtx6000.log` | `carmt/long_context.py`: memory and throughput vs context length, batch 1; mode "last" against the original after its constant-memory fix, mode "loss" | final |
| `bench_rtx6000_decode_after_fix.json` | `bench.py --what decode`, after the original's fix (generation goes through it) | final |
| `train_ddp_rtx6000_{original,carmt}.{log,jsonl}` | 100-step 2-GPU DDP runs, `train.py` vs `carmt/train_carmt.py`, same arguments, data and seed | carmt: the build before the review round 3 fixes (it re-packed weights every micro-step: same numbers, a few ms slower) |
| `train_ddp_rtx6000_carmt_final30.log` | 30-step 2-GPU DDP run of `carmt/train_carmt.py`: same losses, 33.1k tok/s | split build |
| `bench_rtx6000_original_vertical.log` | `bench.py --vertical`: the original in segments-first mode is as fast as layers-first (its carmt rows are from an earlier build) | earlier |
