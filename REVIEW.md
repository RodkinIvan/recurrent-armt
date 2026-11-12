# Code review: open issues

Scope: `carmt/` (the CUDA implementation), `recurrent_armt/`, `train.py`, `tests/`, `scripts/` and `jobs/`. The review is
static: nothing was built or run here. Claims were checked against `results/`, `carmt/records/`, `outputs/*/log.jsonl`
and `logs/router/*`. Ordered by severity.

---

## 1. carmt docs and records are out of date (owner: carmt; fix in progress)

- `carmt/README.md` is still at the 11:22 version:
  - "`train.py` … unchanged" and "the only change is one line" — runs now go through `train_carmt.py`.
  - The `scan_fp32` option and the `libarmt_bf16s32.so` build are not mentioned.
  - **Speed table.** It shows only the single-GPU 1.41×. The 2-GPU DDP run in `records/train_ddp_rtx6000_*.log` gives
    33.2k vs 26.3k tok/s, i.e. **1.26×**. That is the number that applies to the training jobs, and it should be in the
    table.
  - **Memory row.** It says "~12.8 GB vs ~13.6 GB". The DDP records log 17.1 GB (original) vs 19.6 GB (carmt).
  - **"About half is the model GEMMs, which run near the card's bf16 peak."** This overstates it. A training step
    does about 180 TFLOP of matmul (forward, backward recompute, backward), so ~1.5 s of GEMMs is ~120 TFLOPS, about a
    third of the RTX 6000 Ada's 364 TFLOPS dense bf16 (the owner agrees). Report the profiled GEMM time and achieved
    TFLOPS instead, naming the peak figure when giving a percentage.
- `records/bench_rtx6000_bf16.json` predates the allocator change. Its `peak_gb` of 2.98 for carmt leaves out the
  library's ~10 GB of training stores. Rerun `bench.py`.

## 2. Deferred (owner: Python side)

- **Mamba v1 `_scan_state_terms`** (`backbones.py:150-166`) still materializes several `[b, d_inner, T, n]` fp32
  tensors and runs even when `h0` is zero. Both sides agree to leave it, since Mamba v1 is not used in the runs.
