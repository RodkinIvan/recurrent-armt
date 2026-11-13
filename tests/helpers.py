"""Tiny random backbones and ARMT models for the tests."""
from __future__ import annotations

import importlib.util

import torch

from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
from recurrent_armt.backbones import load_base_model

VOCAB = 97
CUDA = torch.cuda.is_available()
HAS_FLA = CUDA and importlib.util.find_spec("fla") is not None
DEVICE = torch.device("cuda" if CUDA else "cpu")
# fp64 on CPU (exact comparisons); fp32 on GPU (the fla / mamba kernels do not take fp64)
DTYPE = torch.float32 if CUDA else torch.float64


def tiny_base_config(kind: str) -> dict:
    if kind == "mamba":
        return dict(model_type="mamba", vocab_size=VOCAB, hidden_size=32, state_size=8, num_hidden_layers=2,
                    expand=2, conv_kernel=4, time_step_rank=4, residual_in_fp32=True, tie_word_embeddings=True)
    if kind == "mamba2":
        return dict(model_type="mamba2", vocab_size=VOCAB, hidden_size=64, num_heads=8, head_dim=16, n_groups=1,
                    state_size=16, num_hidden_layers=2, expand=2, conv_kernel=4, chunk_size=16,
                    residual_in_fp32=True, tie_word_embeddings=True)
    if kind == "rwkv7":
        # head_dim 32: the fp32 backward of chunk_rwkv7 with head_dim 64 needs 128 KB of shared memory (Ada: 99 KB);
        # training runs in bf16, where head_dim 64 fits
        return dict(model_type="rwkv7", vocab_size=VOCAB, hidden_size=128, head_dim=32, num_hidden_layers=3,
                    intermediate_size=256, fuse_norm=False, fuse_cross_entropy=False)
    if kind == "gated_deltanet":
        return dict(model_type="gated_deltanet", vocab_size=VOCAB, hidden_size=128, num_heads=2, head_dim=64,
                    num_hidden_layers=2, fuse_norm=False, fuse_cross_entropy=False, fuse_swiglu=False)
    if kind == "gdn2":
        return dict(model_type="gdn2_lm", vocab_size=VOCAB, hidden_size=128, num_hidden_layers=2, num_heads=2,
                    head_dim=64, intermediate_size=256)
    if kind == "qwen3_5":
        # hybrid: GDN, attention, GDN, GDN, attention; more value than key heads (as in Qwen3.5-9B+), partial RoPE
        return dict(model_type="qwen3_5_text", vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=5, layer_types=["linear_attention", "full_attention", "linear_attention",
                                                      "linear_attention", "full_attention"],
                    num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                    linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                     "mrope_section": [4, 2, 2], "mrope_interleaved": True},
                    tie_word_embeddings=True)
    raise ValueError(kind)


HYBRIDS = {"qwen3_5"}           # recurrent + attention layers: the KV caches grow, ARMT wraps the recurrent layers only
# qwen3_5 runs its GDN through fla on GPU, the reference recurrence on CPU
KINDS = ["mamba", "mamba2"] + (["qwen3_5"] if HAS_FLA or not CUDA else []) + (["rwkv7", "gated_deltanet", "gdn2"] if HAS_FLA else [])


def tiny_base(kind: str, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = load_base_model(tiny_base_config(kind)).to(DEVICE, DTYPE)
    with torch.no_grad():   # fresh RWKV-7 zero-inits o_proj / ffn.value: nothing before them would get gradients
        for p in model.parameters():
            if p.dim() >= 2 and not p.any():
                p.normal_(std=0.02)
    return model.eval()


def tiny_armt(kind: str, seed: int = 0, active_memory: bool = True, **overrides) -> RecurrentARMTForCausalLM:
    """ARMT over a tiny random backbone. ``active_memory`` replaces the zero init of W_mv by random values
    (as after training), otherwise the memory would write nothing and its paths would be invisible."""
    base = tiny_base(kind, seed)
    cfg = dict(num_mem_tokens=4, d_mem=8, segment_size=16, memory_dtype=str(DTYPE).replace("torch.", ""))
    cfg.update(overrides)
    torch.manual_seed(seed + 1)
    model = RecurrentARMTForCausalLM(RecurrentARMTConfig(**cfg), base_model=base).to(DEVICE)
    model.test_kind = kind
    if active_memory:
        with torch.no_grad():
            for memory in model.memory_layers.values():
                memory.W_mv.weight.normal_(std=0.3)
    return model.eval()


def random_ids(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (batch, length), generator=g).to(DEVICE)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


# ATOL: the same computation in another order / grouping (CPU fp64 ~exact, GPU fp32);
# KTOL: different kernels or chunkings of the same math (e.g. chunked vs recurrent scans, HF vs ours)
ATOL = 1e-4 if CUDA else 1e-10
KTOL = 1e-3 if CUDA else 1e-8


def train_autocast(model):
    """The fp32 backward of fla's chunk_rwkv7 needs more shared memory than the Ada GPUs have (128 KB vs 99 KB):
    RWKV-7 gradients are tested in bf16 autocast (as it is trained), the other backbones in full precision."""
    import contextlib
    if CUDA and getattr(model, "test_kind", None) == "rwkv7":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def grad_tol(model) -> float:
    return 2e-2 if CUDA and getattr(model, "test_kind", None) == "rwkv7" else KTOL
