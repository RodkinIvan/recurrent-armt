# Recurrent ARMT

ARMT (Associative Recurrent Memory Transformer) on top of **recurrent** language models instead of
transformers: RWKV-7 and other fla (flash-linear-attention) models, Mamba, Mamba2, and the recurrent layers
of attention / Gated DeltaNet hybrids (Qwen3.5).

The original ARMT (e.g. `irodkin/ARMT-Gemma-3-1B-IT-8x1024-portable`) splits the input into segments; in
every layer, the segment and `num_mem_tokens` memory tokens go through the layer, the memory tokens'
outputs are written into a per-layer associative matrix, and the next segment reads from it. Nothing else
crosses segment boundaries (attention is segment-local). With a recurrent backbone the block's recurrent
state can cross the boundary for free, so here the state handed from segment to segment is, per layer,

* the associative memory `(W, z)` (exactly the original math, see `tests/test_memory_original.py`), and
* the recurrent state of the block (RWKV-7 wkv state + token-shift states, Mamba conv + SSM states, ...).

Both are fixed-size and differentiable: gradients from later segments reach earlier ones through both.
(Hybrid backbones add the KV caches of their attention layers, which grow with the sequence; see below.)

## Layout

| file | |
|---|---|
| `recurrent_armt/memory.py` | `AssociativeMemory`: DPFP read / delta-rule write with correction (same as the original `_associate_with_mem` / `_update_mem_with_mem`) |
| `recurrent_armt/backbones.py` | one interface over recurrent LMs: `block(i, hidden, aux, state) -> (hidden, aux, state)` continuing from any state; fla models as they are, Mamba / Mamba2 / Qwen3.5 (GDN + attention) mixers re-implemented on the same weights (their HF cached path only decodes one token at a time) with CUDA kernels + pure-PyTorch reference scans |
| `recurrent_armt/modeling.py` | `RecurrentARMTConfig`, `RecurrentARMTForCausalLM` (HF `PreTrainedModel`), `ARMTState` |
| `recurrent_armt/loss.py` | next-token loss computed in checkpointed chunks (no `[B, L, vocab]` logits kept) |
| `recurrent_armt/data.py` | streaming fineweb-edu-100b-shuffle with document-aware shuffling, resumable |
| `train.py` | DDP training (torchrun), bf16 autocast, per-segment losses, exact resume |
| `tests/` | the checks listed below |
| `jobs/*.json`, `env.yml`, `env_post.sh` | router jobs / conda env for the `mbz_nlp*` machines |
| `carmt/` | an optional CUDA implementation of the GDN-2 ARMT (same parameters, same `ARMTState`): `carmt.accelerate(model)`, or `carmt/train_carmt.py <train.py args>` to run `train.py` on it. See [carmt/README.md](carmt/README.md) |

## The model

```python
from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM

cfg = RecurrentARMTConfig(base_model_name="fla-hub/rwkv7-0.4B-g1a",   # or AntonV/mamba2-370m-hf, state-spaces/mamba-370m-hf, ...
                          segment_size=1024, num_mem_tokens=32, d_mem=64)
model = RecurrentARMTForCausalLM(cfg)
out = model(input_ids=ids, labels=ids)                 # loss (logits only when asked / no labels)
out = model(input_ids=ids, return_state=True)          # .logits, .state  (ARMTState: fixed size, but for hybrids' KV caches)
out = model(input_ids=more_ids, state=out.state)       # continue the same sequence (any piece length)
new_tokens = model.generate(prompt_ids, max_new_tokens=100)   # O(1) per token on the state (hybrids: O(length), attention)
model.vertical_mode = True                             # segments-first execution; default layers-first
model.gradient_checkpointing_enable()                  # checkpoint every (layer, segment) block call
```

Per layer and segment: `hidden += read(hidden, W, z)` for the segment and the memory tokens -> the block
runs over the segment continuing from its recurrent state -> the memory tokens run through the block ->
their outputs `write` the memory. Segment boundaries are kept at absolute positions, so `forward` over a
sequence equals `forward` over its pieces with the state passed along, and generation is forward over one
token at a time (the memory tokens are processed whenever a segment is completed).

**Memory tokens and the recurrent stream** (`mem_tokens_in_stream`, default `False`). Default: the memory
tokens are a read-out branch: they start from the state after the segment's last token, write the memory,
and their own state is dropped; the next segment continues from the state after the last real token.
With `W_mv = 0` at initialization (as in the original) the model is then *exactly* the base model, so
training starts from the pretrained model's quality. `True`: the next segment continues from the state
after the memory tokens (one block call per segment instead of two, but the memory tokens perturb the
pretrained model's stream from the start).

### Hybrid backbones (Qwen3.5)

Qwen3.5 (0.8B / 2B / ...) stacks 3 Gated DeltaNet layers per full-attention layer. ARMT goes on the GDN layers
only: they become GDN-ARMT layers (associative memory + GDN state); the attention layers stay as they are.
An attention layer's "state" is the KV cache of all the tokens before (post-RoPE keys, values), so a chunk
continues it at positions `past .. past + T - 1`, and that part of the state grows with the sequence. The
memory tokens pass through attention like through any block: as a read-out branch they attend to the cache
(at the positions right after the segment) but are not added to it, so the real tokens see exactly what the
base model sees and ARMT at init is the base model. Segments-first runs an attention layer segment by segment from its cache;
layers-first runs it over all segments at once (one causal call for the real tokens, one masked call for all
the memory tokens; `Qwen35Backbone.attention_layer`). Both give the same result up to rounding, as do pieces and
generation. State dropout / `state_reset` touch the GDN states only.

    model = RecurrentARMTForCausalLM(RecurrentARMTConfig(base_model_name="Qwen/Qwen3.5-0.8B-Base", segment_size=1024))

Removed from the transformer version: attention masks, sliding window / KV cache, `past_attn_mask`,
position ids / rotary, the sink token, flash-attention / liger paths, ACT. Padding is not supported (a
recurrent state would absorb the pad tokens); batches have equal-length sequences.

## Checks (`tests/`)

Run on CPU in float64 (`pytest tests`, Mamba / Mamba2 through the reference scans) and on GPU in float32
through the router (`jobs/tests.json`; adds RWKV-7, Gated DeltaNet, the CUDA kernels and real checkpoints).

| check | test |
|---|---|
| layers-first (horizontal) == segments-first (vertical): logits and gradients | `test_vertical_equals_horizontal`, `test_vertical_horizontal_gradients` |
| generation (token by token on the state) == forward, crossing several segment ends | `test_generation_matches_forward`, `test_pretrained.py` |
| feeding a sequence in pieces == one forward (logits and gradients) | `test_pieces_equal_whole`, `test_streamed_gradients_equal_whole` |
| the state has exactly the same tensors / bytes for any length and after every generated token | `test_state_size_constant` |
| gradients reach earlier segments through the recurrent state alone, through the memory alone, and not at all when both are cut; every memory parameter gets gradients | `test_gradients_flow_through_state_and_memory`, `test_all_parameters_get_gradients` |
| trains on GPU in bf16 (with and without checkpointing) | `test_training.py` |
| optimizations do not change results: gradient checkpointing, chunked loss, CUDA kernels vs reference scans | `test_gradient_checkpointing_is_exact`, `test_checkpointing_same_grads_under_autocast`, `test_chunked_loss_is_exact`, `test_kernels_match_reference_scans` |
| at init ARMT == the base model (tiny models and real checkpoints) | `test_init_equals_base_model`, `test_pretrained.py` |
| associative memory == original ARMT math (all variants, first-segment case) | `test_memory_original.py` |
| save / load round trip | `test_save_load_roundtrip` |
| data: packing, document cooldown, determinism, disjoint ranks, exact resume | `test_data.py` |

## Data: fineweb-edu-100b-shuffle, streamed

Row groups (1024 documents, read remotely) are the unit. All training row groups are put in one seeded
order and dealt to ranks (ranks never share a document). A rank takes 16 row groups at a time (16k
documents from 16 random places of the corpus), tokenizes them in a process pool, prefixes every document
with the separator token and packs them into windows of `n_segments * segment_size` tokens. Packing is
chunk-local, so chunks share no document; inside a chunk the windows are emitted in a random order in
which two windows containing pieces of the same document are at least `cooldown` windows apart (default
`2 x windows per optimizer step per GPU`: never in the same or adjacent batches). The stream is a
deterministic function of the seed and rank, so the checkpoint stores only (chunk, position) and a resumed
run sees exactly the same windows. The last shard is held out: packed validation windows, and "long"
validation windows (the first `window` tokens of documents at least that long).

### Measured (GPU, fp32 unless noted; CPU runs the same checks in fp64 and gets ~1e-15)

| check | RWKV-7 | Gated DeltaNet | Mamba | Mamba2 |
|---|---|---|---|---|
| H vs V logits (tiny models and real checkpoints) | bitwise equal | bitwise equal | bitwise equal | bitwise equal |
| H vs V / checkpointing gradients | identical (bf16) | identical | <= 3e-7 rel | identical |
| generation vs forward, tiny model, max abs | 3e-6 | 9e-7 | 9e-7 | 7e-5 (TF32) |
| ARMT@init vs base, real checkpoint (control = base generate vs base forward) | 6.5e-4 (control 8.4e-4) | - | 9.3e-5 (3.7e-5) | 4.2e-3 (3.9e-3) |
| CUDA kernels vs reference scans, grads | - | - | 2e-6 rel | 6e-4 rel (TF32) |
| state bytes, tiny model, batch 2 (constant) | 351 KB | 394 KB | 37 KB | 90 KB |

The Mamba2 SSD and fla Triton kernels multiply in TF32 even for fp32 inputs, which is why the real-model
bounds are relative to a control (the base model's own cached decoding vs its own forward). The fp32
backward of fla's `chunk_rwkv7` needs 128 KB of shared memory (Ada GPUs: 99 KB), so RWKV-7 gradients are
tested in bf16, as it is trained.

## Running on the mbz_nlp machines

```bash
python /path/to/router/route.py run jobs/tests.json        # all checks on a GPU
python /path/to/router/route.py run jobs/data_smoke.json   # the real data stream: throughput, cooldown, statistics
python /path/to/router/route.py run jobs/train.json        # the pilot pre-training runs
```

Notes on the machines (found while setting this up):
* `NCCL_P2P_DISABLE=1` is set in the training jobs: on `mbz_nlp` GPU peer-to-peer is broken (the two GPUs
  hang in an all-reduce / DDP sees garbage); with P2P off NCCL is correct, and for 2 GPUs it costs nothing noticeable.
* `mbz_nlp` has a 200 GB per-user ZFS quota that is full (a full quota kills runs while they write a
  checkpoint, and even the traceback cannot be logged); `mbz_nlp2` has the same quota with ~35 GB free.
  `mbz_nlp2.14` GPU 0 is in a broken state (100% busy, no process, refuses contexts).
* fla 0.6 needs triton >= 3.5 (torch 2.9); the hosts' driver 535 runs the cu126 builds.
* Checkpoints are written atomically (`out_dir/saving` -> `last`, the previous one kept as `last.old` until the
  swap is done, recovered at startup); a router retry resumes exactly (data, CPU and CUDA RNG) and drops the log
  lines written after the checkpoint.
* A state created outside autocast and then used under autocast keeps its own dtypes (e.g. a float32 conv state);
  create it in the context the model runs in (`forward` / `generate` do).
* wandb: the job files pass `--wandb_project recurrent-armt`; `train.py` reads the key from `wandb_token` in the
  project dir or `~/wandb_token` on the host (placed there once, mode 600) into `WANDB_API_KEY` (rank 0 only).
  (The router's `secrets` cannot be used from macOS: its `rsync --chmod=F600` is rejected by Apple's openrsync.)
  Every launch is a new W&B run in the group `<out_dir>` (a resume is named `<out_dir>-resume<step>`): W&B drops re-logged steps. A resumed run continues
  the same wandb run (id = the output directory name).

## Pilot runs (fineweb-edu, 8192-token windows, 32 windows = 262k tokens per step, 600 steps)

ARMT settings as in the Gemma checkpoint unless noted: 1024-token segments, 32 memory tokens, d_mem 64.

| run | job file | what | val / val_long @ step 600 (step 0) |
|---|---|---|---|
| `rwkv7_0.4b_armt` | `train.json` | RWKV-7 0.4B (g1a, annealed) + ARMT, lr 3e-5, memory lr x10 | 2.4896 / 2.5351 (2.4738 / 2.5224) |
| `rwkv7_0.4b_nomem` | `train.json` | the same without memory (`--num_mem_tokens 0`): the control | 2.4890 / 2.5346 |
| `rwkv7_0.4b_nomem_lr3e-6` | `diag_lr.json` | control at lr 3e-6: no degradation (2.4740 at step 100) -> the rise above is the re-warm of an annealed checkpoint | stopped at 100 |
| `rwkv7_0.4b_armt_freeze300` | (manual, 1 GPU) | `--freeze_backbone_steps 300` | 2.4792 / 2.5254 |
| `gdn2_370m_armt` | `train.json` | GDN-2 370M, 30B-token snapshot (not annealed, `scripts/convert_gdn2.py`), lr 1e-4, memory lr x3 | 2.4182 / 2.4799 (2.4331 / 2.4936) |
| `gdn2_370m_nomem` | `train.json` | its control | 2.4180 / 2.4798 |
| `gdn2_370m_armt_seg256` | `seg256.json` | 256-token segments (32 per window) | 2.4181 / 2.4799 |
| `gdn2_370m_armt_freeze300_sd{0.5,1.0}` | `state_dropout.json` | backbone frozen and recurrent state dropout p for 300 steps | 2.4171 / 2.4783 (both) |
| `qwen35_0.8b_{armt,nomem}` | `qwen35.json` | Qwen3.5-0.8B-Base hybrid, ARMT on the 18 GDN layers / control; lr 1e-5, memory lr x30 | 2.6007 / 2.4907 vs 2.6002 / 2.4902 (2.6336 / 2.5184) |
| `gdn2_370m_math_{armt,nomem}` | `math_mix.json` | GDN-2 on the long-math mix (`--mix fineweb:4,arxiv:1,omr:2`: row groups per chunk; by tokens 24% fineweb, 18% arXiv, 58% OMR) | val_long_arxiv 1.9067 vs 1.9083, omr 1.1653 vs 1.1651 |
| `gdn2_370m_math_sd0.5_{armt,nomem}` | `math_state_dropout.json` | the same with state dropout p = 0.5 for all steps | state reset: arxiv 2.1047 vs 2.1220, omr 1.2707 vs 1.2874; state kept: tie |

Long-math and hybrid runs: see `reports/recurrent_armt_wip.md` (sections 4.5–4.7); figures from
`scripts/report_figures.py` and `scripts/report_diagrams.py`.

So far the memory is trained but unused: ARMT matches its control everywhere, and zeroing W_mv in the trained
`gdn2_370m_armt` changes the loss by <= 2e-4. With the recurrent state reset at every boundary (`val_long_reset`,
memory-only context) the base GDN-2 loses 0.12 nats on long documents; that is what state dropout makes the memory
learn to recover. Note the packed windows give little cross-segment signal (median document 626 tokens).

`--freeze_backbone_steps N` keeps the backbone frozen (no weight gradients, lr 0) for N steps; at step N it is
unfrozen and DDP is rebuilt (DDP only reduces the parameters that required grad when it was built).
`--rnn_state_dropout p` resets each sample's recurrent state at a segment boundary with probability p (training;
`--rnn_state_dropout_steps N`: only the first N steps); `--eval_state_reset` adds the `val_long_reset` evaluation.
Each evaluation also logs `mem/read_ratio` (|read-out| / |hidden| per wrapped layer), `mem/W_mv_rms`, and each
log step `mem/grad_norm`.
`python scripts/plot_runs.py outputs/<run> ...` plots the train loss and the per-segment validation losses.
