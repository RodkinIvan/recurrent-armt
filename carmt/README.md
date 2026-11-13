# carmt: ARMT + Gated DeltaNet-2 in one CUDA file

`armt.cu` is a from-scratch CUDA/C++ implementation of `recurrent_armt.RecurrentARMTForCausalLM` over a GDN-2 base model
(`recurrent_armt/gdn2.py`: fla `GatedDeltaNet2` token mixer + SwiGLU MLP), and over the hybrid Qwen3.5 (its Gated
DeltaNet layers in C, its attention layers in PyTorch: see [Hybrid: Qwen3.5](#hybrid-qwen35)). It covers:
- the forward pass with a carried fixed-size state (inference, streaming, generation);
- the forward + backward pass for training.

It depends only on the CUDA runtime and cuBLAS: no PyTorch, Triton or fla inside. It computes the same function on the same
parameters and accepts and returns the same `ARMTState`.

`train.py`, `recurrent_armt/` and `tests/` are unchanged; everything lives in this directory. Speed and correctness
summary: [reports/carmt_speedup.md](../reports/carmt_speedup.md). Raw outputs: [records/](records/README.md).

| file | |
|---|---|
| `armt.cu` | the whole model: kernels, GDN-2 chunked scan, ARMT memory, loss, forward / backward drivers, C API |
| `Makefile` | builds `libarmt_bf16.so` (default), `libarmt_bf16s32.so` (fp32 scan, `scan_fp32=True`), `libarmt_fp32.so` (exact fp32), and optionally `libarmt_bf16split.so` |
| `__init__.py` | ctypes binding: `CARMT(model)`, `accelerate(model)`, `mark_weights_changed()`, autograd glue |
| `train_carmt.py` | runs the unchanged `train.py` with carmt: `torchrun ... carmt/train_carmt.py <train.py args>` |
| `jobs/train_carmt.json` | router job: the `gdn2_370m_armt` run through `train_carmt.py` |
| `build_env.sh` | installs a CUDA 12.6 toolkit into a separate prefix (the jobs' env has none) and builds |
| `tests/test_carmt.py`, `tests/test_hybrid.py`, `tests/dist_check.py` | 48 + 11 GPU tests (GDN-2, Qwen3.5 hybrid); DDP / ZeRO-1 / ZeRO-2 checks on 2 GPUs (`--model qwen35` for the hybrid) |
| `qwen35_check.py` | Qwen3.5-0.8B-Base + ARMT: carmt vs HF / the original / the exact recurrence, fp32 and bf16 gradients |
| `bench.py`, `profile_kernels.py`, `long_context.py`, `speedup_figure.py` | speed vs the original; per-kernel GPU time; memory vs context length; the report figure |

## Use

```bash
make -C carmt CUDA_HOME=<cuda toolkit> ARCH=sm_89          # or: bash carmt/build_env.sh [prefix] [arch]
```

```python
import carmt
model = RecurrentARMTForCausalLM(cfg)          # GDN-2 backbone, as now
carmt.accelerate(model)                        # model.forward now runs armt.cu
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(input_ids=x, labels=x)         # .loss, .token_losses (as before)
out.loss.backward()                            # grads land in p.grad (DDP, ZeRO-1/2, clipping, AdamW, accumulation as usual)
model.generate(prompt, max_new_tokens=100)     # also on the C path (state in / state out)
out = model(input_ids=x, return_state=True)    # ARMTState, interchangeable with the PyTorch implementation's
```

**Training with `train.py`.** Launch `carmt/train_carmt.py` with `train.py`'s arguments; `train.py` itself is not edited.
- **How it works.** The launcher patches the model class so that every instance gets `carmt.accelerate()` before `train.py`
  moves it to the GPU and wraps it in DDP. It builds missing libraries once, under a file lock.
- **Precision.** `CARMT_SCAN_FP32=1` selects the fp32-scan build.
- **Checkpointing.** The C backward always recomputes each (layer, segment) from its stored input, whatever the flags. On a
  hybrid, the model's gradient-checkpointing flag also checkpoints the attention layers (PyTorch), as the original does.

**Other entry points.**
- Without patching: `runner = carmt.CARMT(model); runner(input_ids=..., labels=..., state=..., return_state=...)`.
- `model.forward_reference` keeps the original forward.
- `model.memory_stats` runs through the PyTorch path, because it needs the per-read hooks.

**Precision.** It follows the caller.

- **Under `torch.autocast(bfloat16)`, the default bf16 library runs:**
  - bf16 matmuls and activations, and bf16 scan intermediates (as fla's kernels use);
  - fp32 for the residual stream, the recurrent state, the decay cumsums and 64×64 inverses, and the ARMT memory (as
    autocast-disabled PyTorch).
- **`CARMT(model, scan_fp32=True)`** (or `accelerate(model, scan_fp32=True)`) runs the scan in fp32 on CUDA cores with bf16
  matmuls: slower, and in our measurements not measurably more accurate than the default.
- **Without autocast**, the fp32 library runs fp32 everywhere. TF32 is used in fp32 matmuls only if
  `torch.backends.cuda.matmul.allow_tf32` is set; that applies to the memory's fp32 matmuls in every build.

**Weights.** The bf16 weight copies are re-packed only when parameters change. Changes are detected through:
- optimizer steps, counted by a global `torch.optim` hook (fused AdamW does not bump version counters; DeepSpeed's ZeRO
  optimizer is covered too);
- version counters (loads, in-place edits);
- data pointers.

After a change none of these can see (e.g. a raw NCCL write into the parameters), call `carmt.mark_weights_changed()`.

**GEMM kernels.** On first use of each matmul shape the library times cuBLASLt's candidates against cuBLAS's default and keeps
the fastest. The choices are written to `carmt/.gemm_cache_<gpu>_<cublas version>.txt` under a file lock, so DDP ranks
and later runs read them instead of tuning again, and so compute with the same kernels. When two processes tune the same
shape at once, the first one written wins and both use it. The lock is never held while the GPU works.
- The choices are kept for good. Shapes first seen in a backward are timed while other work (e.g. DDP's all-reduce)
  may share the GPU. Delete the cache file to re-tune.
- `ARMT_GEMM_CACHE=<path prefix>` moves the file; `ARMT_GEMM_CACHE=` keeps the choices in-process only.
- `ARMT_GEMM_TUNE=0` uses cuBLAS's defaults; `ARMT_GEMM_LOG=1` prints each shape's default and chosen time.
- The memory's fp32 matmuls read the framework's parameter tensors, whose alignment depends on the framework (DeepSpeed's
  flat buffers), so they always use cuBLAS's default: tuned choices never differ between, e.g., a DDP and a ZeRO run.

**Memory and errors.**
- The library allocates through PyTorch's caching allocator, so its memory appears in `torch.cuda.memory_*` and shares
  the cache.
- The training stores are released after every backward. `runner.release_memory()` gives everything back.
- CUDA / cuBLAS errors and unsupported configurations raise `RuntimeError`; they don't exit the process. Out-of-range
  labels trigger an async device assert, as in PyTorch.

**Distributed.**
- DDP, DDP with gradient accumulation and `no_sync`, and DeepSpeed ZeRO-1 / ZeRO-2 are checked (`tests/dist_check.py`).
- The C backward runs as one autograd node per layer (top layer first), so DDP / ZeRO receive each layer's gradients as soon
  as they are final. `CARMT_LAYER_NODES=0` makes it one node for all layers above the first. In the 2-GPU job this makes
  no measurable difference (the exposed all-reduce drops from ~0.32 to ~0.24 s per optimizer step of 7.3 s:
  `records/ddp_layer_nodes_rtx6000.log`).
- Under DeepSpeed, enable its `torch_autocast` section; without it the engine silently disables an outer
  `torch.autocast`. DeepSpeed's `bf16` section can't be used: it converts the parameters to bf16.
- Set `lower_precision_safe_modules: []` to keep the gradient all-reduce in fp32.
- Not supported: ZeRO-3 / FSDP (they gather parameters through module hooks, which the C forward bypasses) and tensor
  parallelism.

**Supported.**
- **GDN-2:** short conv of size 4 without bias, `expand_v = 1`, head_dim 64 or 128, swish or sigmoid output gate.
- **ARMT:**
  - any `num_mem_tokens` (incl. 0), `d_mem`, memory `n_heads`, `gating`, `wrap_layers`, `mem_tokens_in_stream`;
  - `rnn_state_dropout` / `state_reset`, with the same RNG draws as the original;
  - `labels_mask` / `-100`;
  - any sequence length (10M+ tokens in one call) and state continuation;
  - frozen parameters: their gradient work is skipped.
- **Falls back to the PyTorch forward** for `inputs_embeds`, and for grad-enabled calls that have no labels, pass a state, or
  ask for logits / a state.

## Hybrid: Qwen3.5

`carmt.accelerate(model)` / `CARMT(model)` on a `RecurrentARMTForCausalLM` over Qwen3.5 (HF `Qwen3_5ForCausalLM`, e.g.
Qwen3.5-0.8B-Base: 18 Gated DeltaNet + 6 gated-attention layers) run the Gated DeltaNet layers, their MLPs and memories,
the embeddings and the loss in C, and the attention layers in PyTorch, through the backbone's own
`Qwen35Backbone.attention_layer` (PyTorch's flash attention for the real tokens, one masked call for all memory tokens).

- **Layer-major.** A call runs layer by layer over all its segments (`armt_hy_begin / run / head`): each run of
  consecutive recurrent layers is one C call over all segments; the hidden streams go to the attention layer between them
  as the real tokens `[B, T, D]` and the memory tokens of the segment-ending pieces `[B, nw, M, D]`. In training every run
  and the loss head are autograd nodes (`armt_hy_run_bwd`, `armt_hy_head_bwd`), the attention layers ordinary PyTorch ops in
  between, so DDP / ZeRO and the gradients work as for GDN-2.
- **The Gated DeltaNet in C** is the GDN-2 machinery with one gate and one beta per head (scalar mode of the token-prep
  kernels; the chunked scan is unchanged) and `(1 + w)` RMSNorms. The four input projections are one packed GEMM.
- **State.** `ARMTState` as the original's: the attention layers' entries are their KV caches; inference with a state,
  pieces, and generation go through the same path.
- **Memory.** With gradient checkpointing on (as `train.py` runs), the attention layers are recomputed in the backward;
  `CARMT_ATTN_CKPT=0` keeps their activations instead (faster, about twice the memory).
- **Supported:** as many value as key heads in the GDN (Qwen3.5-0.8B; the larger Qwen3.5 models have more value heads and
  fall outside), head dim 64 or 128, conv 4, `mem_tokens_in_stream=False`.

**Equivalence** (`tests/test_hybrid.py`, `qwen35_check.py`; records `records/tests_hybrid_rtx5880.log`,
`records/qwen35_check_rtx5880.log`):

| check | result |
|---|---|
| tiny Qwen3.5 (layer patterns ending / starting with attention), fp32, vs the exact recurrence | logits ≤ 1e-4, every gradient ≤ 3e-4 |
| Qwen3.5-0.8B-Base + ARMT, fp32 logits vs the exact token-by-token recurrence, 2 × 2048 tokens | carmt 2.8e-6; the original with fla's chunk kernel 1.8e-3 (HF itself is 1.8e-3 off too) |
| 0.8B bf16 training step, every gradient vs fp32 | median error 0.84× the original's own bf16 error |
| pieces == whole, state interchangeable with the original, generation == forward, state dropout | ≤ 1e-4 |
| DDP (+ accumulation, `no_sync`), ZeRO-1, ZeRO-2, tiny hybrid, fp32, vs the exact reference | ≤ 9e-5 (`records/dist_check_qwen35_2x_rtx5880.log`) |
| DDP and ZeRO-2, Qwen3.5-0.8B, bf16, vs single-process carmt averaged over the ranks' batches | ≤ 3e-7 |

Two properties of the PyTorch side show up at this precision, the same for the original:
- PyTorch's flash / memory-efficient attention backward is not deterministic: two identical bf16 steps differ by up to ~2e-2
  on the most cancellation-prone gradients (`dist_check` runs the 0.8B comparison on the deterministic math backend).
- Running `backward()` inside the `torch.autocast` block changes the attention layers' bf16 gradients by ~2e-3 relative
  (median) against running it outside, as `train.py` and DeepSpeed's engine do; carmt's C part is unaffected.

The chunked scan's fp32 gate gradients (`A_log`, `dt_bias`, `in_proj_a`) lose precision at very fast decays: with
`A = exp(A_log)` near 16 (the top of HF's init range, ~e^-20 per token) they are ~1e-3 off the exact recurrence, ~1e-5 at
A = 4. The tests draw A in [0.01, 4]; Qwen3.5-0.8B has A in [0.001, 10.6], median 0.27.

**Speed** (RTX 5880 Ada, bf16, batch 4 × 8192 tokens = 8 segments of 1024, 32 memory tokens; the original layers-first with
flash attention, as in `train.py`; `records/bench_rtx5880_qwen35_bf16.json`):

| workload | original | carmt | speedup |
|---|---|---|---|
| training step, gradient checkpointing (as `train.py`) | 5.14 s, 23.2 GB | 3.60 s, 18.3 GB | 1.43× |
| training step, `CARMT_ATTN_CKPT=0` (the original does not fit without checkpointing) | 5.14 s, 23.2 GB | 3.18 s, 35.4 GB | 1.61× |
| inference with loss | 1.26 s | 0.95 s | 1.33× |
| generation (batch 1, 1k-token prompt + 64 tokens) | 22.6 ms/token | 6.65 ms/token | 3.40× |

Where a training step goes (`records/phases_rtx5880_qwen35_train.log`): the C backward of the recurrent layers ~43%, the
248k-vocabulary loss head ~20% (about 50 TFLOP of GEMMs per step, the same in both implementations), the C forward of the
recurrent layers ~13%, the attention layers (PyTorch) and the rest ~20%.

## How it works

Per (layer, segment), the segment's tokens and the memory tokens form one sequence of `S + M` rows. All token-wise work
(memory read, norms, projections, MLP) runs once over it as a few large GEMMs:
- the 7 input projections of a GDN-2 block are packed into one `[10496 x 1024]` weight;
- `w1 | w2` into one `[4096 x 1024]` weight.

The recurrent parts run over the concatenation, which is exactly "memory tokens continue from the state after the last
real token". The state handed to the next segment is a snapshot after the real tokens, so no second block call is needed.

The GDN-2 recurrence `S_t = (I - k_t (b_t k_t)^T) Diag(e^{g_t}) S_{t-1} + k_t (w_t v_t)^T` is computed in chunks of 64.

- **Intra-chunk.** Per chunk: the cumulative decays, `L`, `M`, `A = (I + L)^-1`, and `u = A (w v)`, `W = A (b k e^gc)`.
  - The decays of the off-diagonal 16×16 sub-blocks are factored through a midpoint, `e^{gc_i - gc_r} e^{gc_r - gc_j}`.
    Both factors are ≤ 1, so nothing overflows, and the products run on tensor cores.
  - Only the four diagonal blocks use exact pairwise exponentials. `A` is inverted blockwise.
- **Inter-chunk.** A sequential pass per (sample, head, 32 state columns) on tensor cores (WMMA); the state stays fp32 in
  shared memory.
- **Chunk-parallel matmuls.** cuBLAS strided-batched GEMMs (`o = Qg S0 + M vn`, and 12 GEMMs in the backward).
- **Padding.** Chunks are zero-padded; a padded token (`k = v = b = w = 0`, `g = 0`) is an exact no-op.
- **fp32 build.** It uses CUDA-core variants of the same kernels in fp32, which the tight equivalence checks run against.

**Training.** The forward stores, per (layer, segment), the layer input and the input recurrent and memory states. It
computes the loss and its gradient with respect to the final hidden states right away (fused head + cross-entropy, logits
in 4096-row chunks).

The backward walks the layers top-down and the segments right-to-left. Each step recomputes one (layer, segment) from its
stored input, then backpropagates through it. Gradients go into one flat fp32 buffer, laid out so that every packed weight's
gradient is a single GEMM; the Python side returns views of it.

**Inference** runs segment by segment with one segment's activations plus the fixed-size state, and computes the loss per
segment. Memory therefore doesn't grow with the context length.

## Equivalence (`python -m pytest carmt/tests`, GPU; records: `records/tests_*.log`)

Two references:
- **the original model as it is.** fla's Triton kernels multiply in TF32, so the original itself differs from the exact
  result by ~1e-4 – 1e-3;
- **the original with fla's own exact PyTorch GDN-2 recurrence** swapped in (`tests/reference.py`). This is the tight check.

| check | result |
|---|---|
| logits, fp32, 9 configs (head_dim 64/128, sigmoid gate, mem in stream, 2 memory heads + gating, partial wrap, no memory, segment 100 with partial chunks, 3 layers) | ≤ 1e-4 relative to the exact reference (fla path: ~1e-3) |
| loss / every parameter gradient, fp32, same 9 configs | loss within 1e-5; each gradient ≤ max(3e-4, 1/5 of the original's own TF32 error) |
| bf16 gradients at the real dimensions (370M checkpoint, head_dim 128, segment 1024, 32 memory tokens), per-parameter relative-norm error vs fp32, both bf16 builds | median 0.82–0.84× the original's own bf16 error; 95th percentile ≤ 1.3× |
| state: pieces == whole (also > 4096 segments), returned state == original's, C state continues in the original and vice versa, generation == forward | ≤ 1e-4 |
| no-grad calls between a training forward and its backward; two models in one process; forward after fused / foreach AdamW and SGD steps; frozen backbone; errors raise; memory visible to PyTorch; no re-pack between micro-steps | pass |
| state dropout (training), `state_reset`, labels mask / -100, grad accumulation, `accelerate` API | pass |
| real GDN-2 370M checkpoint, 2×1024 tokens, 32 memory tokens | fp32: loss within 9.5e-7 of the exact reference (original: 1.2e-4) |
| DDP, ZeRO-1, ZeRO-2 on 2 GPUs (`tests/dist_check.py`, `records/dist_check_2x_rtx5880.log`) | gradients == reference (fp32: ≤ 3.3e-5; bf16 370M vs single-process carmt: ≤ 4e-7); ranks identical after steps |
| 100-step 2-GPU training run vs the PyTorch run, same data (`records/train_ddp_rtx6000_*`) | train loss within 2e-4 at every logged step (equal to 4 decimals at 9 of 10); eval loss identical |

## Speed

RTX 6000 Ada (`mbz_nlp2`). GDN-2 370M + ARMT (32 memory tokens, d_mem 64), windows of 8 × 1024 tokens, bf16 autocast. The
original runs with gradient checkpointing, as `train.py` runs it; without it, it doesn't fit in 48 GB.

| | original | carmt | speedup |
|---|---|---|---|
| **training, 2-GPU DDP: the `train.py` job** (`NCCL_P2P_DISABLE=1`, 262k tokens per step) | 26.3k tok/s | 36.2k tok/s | **1.38×** |
| training step, 1 GPU (batch 8) | 4.49 s | 3.03 s | 1.48× |
| inference forward with loss (batch 8) | 1.01 s | 0.80 s | 1.27× |
| long-context inference, 10M tokens in one call, batch 1, last-position logits | 20.8k tok/s, 0.87 GB | 88.8k tok/s, 1.09 GB | 4.27× |
| long-context loss over every token, 4.2M tokens (original streamed in 64k pieces with `state=`) | 20.3k tok/s, 3.4 GB | 84.1k tok/s, 1.3 GB (one call) | 4.14× |
| generation (batch 1, 1k-token prompt + 64 tokens) | 16.3 ms/token | 2.85 ms/token | 5.70× |
| memory of a training step: 1 GPU, batch 8 (`max_memory_allocated`) | 14.4 GB | 13.6 GB | |
| memory logged by `train.py` in the DDP job | 17.1 GB | 19.9 GB | |

**DDP gains less than one GPU.** An optimizer step of the job (7.26 s) is two micro-steps of ~3.2 s (both GPUs busy and
hot; 3.03 s alone), ~0.25 s of all-reduce the backward doesn't hide, and ~0.6 s of optimizer, clipping, data and logging.
None of that shrinks with a faster model (`records/ddp_layer_nodes_rtx6000.log`).

**Profile of a 1-GPU training step** (3.20 s GPU time, `records/profile_rtx6000_carmt_train.log`):
- **cuBLAS GEMMs: 1.64 s (51%).** This includes the scan's chunk-batched GEMMs. The model's linear layers do 176 TFLOP per
  step, so they average at least 107 TFLOPS; the RTX 6000 Ada's dense bf16 peak is 364 TFLOPS.
- **GDN-2 scan kernels: 1.23 s (38%).**
- **Other: 0.33 s (10%).**

**Tried and dropped.**
- A two-stream schedule: two whole training steps on two streams run 2–8% slower than back to back
  (`records/overlap_ceiling_rtx6000.log`). The GPU has no idle capacity for overlapping (layer, segment) steps to fill.
- A low-register conv / gate kernel: no measurable change.

**Next speedups (not done).** Fusing the scan stages; a single-token recurrent kernel for generation.
