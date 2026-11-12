"""Recurrent backbones seen through one interface.

ARMT only needs a recurrent LM as  embed -> blocks -> final norm -> lm_head  where every block can
continue from a recurrent state:

    block(i, hidden, aux, state) -> (hidden, aux, state)

``hidden`` is [B, T, D] (any T >= 1), ``state`` is the fixed-size recurrent state of block ``i``
after the previous token (``None`` = start of the sequence, i.e. the zero state) and the returned
state is the one after the last token of ``hidden``. ``aux`` holds per-token tensors that a backbone
passes from block to block next to the hidden states (RWKV-7's ``v_first``). The state must be
differentiable: gradients flow from the outputs of later segments into earlier ones through it.

Supported:
  * fla (flash-linear-attention) models: RWKV-7 and the other purely recurrent fla models
    (GLA, DeltaNet, Gated DeltaNet, HGRN2, RWKV-6, ...); fla layers already continue from a state.
  * transformers Mamba (v1) and Mamba2: their cached path can only decode one token at a time, so
    the mixers are re-implemented here (same weights) to continue a whole chunk from a state.
Attention / hybrid models are rejected: their state (KV cache) grows with the sequence.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

State = Any          # a pytree (dict / tuple) of fixed-size tensors
Aux = dict[str, torch.Tensor]


class Backbone:
    """Adapter over a causal LM; holds a reference to it (the parameters stay in the LM)."""

    def __init__(self, model: nn.Module):
        self.model = model

    num_layers: int
    hidden_size: int

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def block(self, i: int, hidden: torch.Tensor, aux: Aux, state: State | None) -> tuple[torch.Tensor, Aux, State]:
        raise NotImplementedError

    def norm(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def lm_head(self) -> nn.Module:
        return self.model.lm_head


# ----------------------------------------------------------------------------------------------- fla
def _fla_cache_cls():
    try:
        from fla.models.cache_utils import Cache
    except ImportError:                 # older fla
        from fla.models.utils import Cache
    return Cache


class FLABackbone(Backbone):
    """fla ``XForCausalLM``: model.model.{embeddings, layers, norm}, model.lm_head.

    A block reads its state from ``past_key_values[layer_idx]`` and writes the new one back, so a
    one-layer cache is built around the given state for every call."""

    AUX_KEYS = {"rwkv7": ("v_first",)}       # extra per-token outputs a block returns after (hidden, attn, cache)

    def __init__(self, model: nn.Module):
        super().__init__(model)
        cfg = model.config
        if getattr(cfg, "attn", None) or self._has_attention(model):
            raise ValueError("fla models with attention layers are not supported (their state, a KV cache, grows with the sequence)")
        self.inner = model.model
        self.num_layers = len(self.inner.layers)
        self.hidden_size = cfg.hidden_size
        self.aux_keys = self.AUX_KEYS.get(cfg.model_type, ())
        self.cache_cls = _fla_cache_cls()

    @staticmethod
    def _has_attention(model) -> bool:
        try:
            from fla.layers.attn import Attention
        except ImportError:
            return False
        return any(isinstance(m, Attention) for m in model.modules())

    def embed(self, input_ids):
        return self.inner.embeddings(input_ids)

    def norm(self, hidden):
        return self.inner.norm(hidden)

    def block(self, i, hidden, aux, state):
        cache = self.cache_cls.from_legacy_cache([None] * i + [state]) if state is not None else self.cache_cls()
        out = self.inner.layers[i](hidden, attention_mask=None, past_key_values=cache, use_cache=True, **aux)
        aux = dict(zip(self.aux_keys, out[3:3 + len(self.aux_keys)]))
        new_state = {k: v for k, v in cache[i].items() if v is not None}
        if "attn_state" in new_state:      # any softmax-attention layer: its cache grows with the sequence
            raise ValueError(f"layer {i} keeps an attention (KV) cache: attention models are not supported, their state grows")
        return out[0], aux, new_state


# ---------------------------------------------------------------------------------------- mamba / mamba2
# The CUDA kernels (causal_conv1d, mamba_ssm) are used for CUDA tensors; set USE_KERNELS = False to run the
# pure-PyTorch reference scans instead (they are also the CPU path and what the kernels are tested against).
USE_KERNELS = True


def _use_kernels(x: torch.Tensor) -> bool:
    return USE_KERNELS and x.is_cuda


def _acc(x: torch.Tensor) -> torch.dtype:
    """Accumulation dtype of the reference scans (and of the recurrent states they return)."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _conv_with_state(x: torch.Tensor, conv_state: torch.Tensor | None, conv: nn.Conv1d, activation: str):
    """SiLU(causal depthwise conv) over x [B, C, T], continuing from the last K-1 inputs ``conv_state`` [B, C, K-1]."""
    assert activation in ("silu", "swish"), activation
    k = conv.kernel_size[0]
    if conv_state is None:
        conv_state = x.new_zeros(x.shape[0], x.shape[1], k - 1)
    x = torch.cat([conv_state.to(x.dtype), x], dim=-1)
    new_state = x[..., -(k - 1):].clone()          # clone: do not keep the whole segment alive in the state
    if _use_kernels(x):
        from causal_conv1d import causal_conv1d_fn
        y = causal_conv1d_fn(x, conv.weight.squeeze(1), conv.bias, activation=activation)
        return y[..., k - 1:].contiguous(), new_state     # contiguous: an offset view misaligns the scan's vector loads
    bias = None if conv.bias is None else conv.bias.to(x.dtype)
    return F.silu(F.conv1d(x, conv.weight.to(x.dtype), bias, groups=x.shape[1])), new_state


def _selective_scan_ref(u, delta, A, B, C, D, h0):
    """Mamba (v1) scan, one step at a time: h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t, y_t = C_t h_t + D u_t.
    u, delta [b, d, T]; A [d, n]; B, C [b, T, n]; D [d]; h0 [b, d, n]."""
    dA = torch.exp(delta[..., None] * A[None, :, None, :])
    dBu = (delta * u)[..., None] * B[:, None]
    h, ys = h0, []
    for t in range(u.shape[-1]):
        h = dA[:, :, t] * h + dBu[:, :, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=-1) + D[:, None] * u, h


def _scan_state_terms(A, delta, u, B, C, h0):
    """The terms of the Mamba (v1) scan that involve the state, for the zero-state CUDA scan (fp32, exact):
    y_h0[b,d,t] = sum_n C[b,t,n] exp(A[d,n] cum[b,d,t]) h0[b,d,n]                                    (initial state)
    h_T[b,d,n]  = exp(A[d,n] cum_T) h0 + sum_t exp(A[d,n] (cum_T - cum_t)) delta_t u_t B[b,t,n]       (final state)
    with cum = cumsum(delta). Both decays are <= 1, so nothing overflows."""
    with torch.autocast(delta.device.type, enabled=False):      # fp32 for real: under autocast einsum would run in bf16
        return _scan_state_terms_fp32(A.float(), delta.float(), u.float(), B.float(), C.float(), h0.float())


def _scan_state_terms_fp32(A, delta, u, B, C, h0):
    cum = delta.cumsum(-1)                                                       # [b, d, T]
    cum_T = cum[..., -1:]
    A = A[None, :, None, :]                                                      # [1, d, 1, n]
    h_T = torch.einsum("bdtn,bdt,btn->bdn", torch.exp(A * (cum_T - cum)[..., None]), delta * u, B)
    h_T = h_T + torch.exp(A[:, :, 0] * cum_T) * h0
    y_h0 = torch.einsum("bdtn,btn,bdn->bdt", torch.exp(A * cum[..., None]), C, h0)
    return y_h0, h_T


def _ssd_ref(x, dt, A, B, C, D, h0):
    """Mamba2 scan, one step at a time: h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T, y_t = h_t C_t + D x_t.
    x [b, T, h, p]; dt [b, T, h] (after softplus / limits); A, D [h]; B, C [b, T, g, n]; h0 [b, h, p, n]."""
    rep = x.shape[2] // B.shape[2]
    B, C = B.repeat_interleave(rep, dim=2), C.repeat_interleave(rep, dim=2)
    h, ys = h0, []
    for t in range(x.shape[1]):
        h = torch.exp(dt[:, t] * A)[..., None, None] * h + (dt[:, t, :, None] * x[:, t])[..., None] * B[:, t, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", h, C[:, t]))
    return torch.stack(ys, dim=1) + D[:, None] * x, h


class MambaBackbone(Backbone):
    """transformers ``MambaForCausalLM`` (v1). The CUDA scan (``selective_scan_fn``) starts from zero and
    its last state has no gradient, so the state terms are added exactly in PyTorch (checkpointed)."""

    def __init__(self, model):
        super().__init__(model)
        self.inner = model.backbone
        self.num_layers = len(self.inner.layers)
        self.hidden_size = model.config.hidden_size

    def embed(self, input_ids):
        return self.inner.embeddings(input_ids)

    def norm(self, hidden):
        return self.inner.norm_f(hidden)

    def block(self, i, hidden, aux, state):
        blk = self.inner.layers[i]
        residual = hidden.float() if blk.residual_in_fp32 else hidden
        out, state = self._mixer(blk.mixer, blk.norm(hidden.to(blk.norm.weight.dtype)), state)
        return residual + out, aux, state

    @staticmethod
    def _mixer(m, x, state):
        conv_state, h0 = state if state is not None else (None, None)
        u, gate = m.in_proj(x).transpose(1, 2).chunk(2, dim=1)                   # [b, d, T] each
        u, conv_state = _conv_with_state(u, conv_state, m.conv1d, m.activation)
        dt, Bm, Cm = torch.split(m.x_proj(u.transpose(1, 2)), [m.time_step_rank, m.ssm_state_size, m.ssm_state_size], dim=-1)
        dt = m.dt_proj.weight @ dt.transpose(1, 2)                               # [b, d, T]; the bias is added in the scan
        A = -torch.exp(m.A_log.float())                                         # [d, n]
        acc = _acc(u)
        if h0 is None:
            h0 = u.new_zeros(u.shape[0], u.shape[1], m.ssm_state_size, dtype=acc)
        delta = F.softplus(dt.to(acc) + m.dt_proj.bias.to(acc)[:, None])
        if _use_kernels(u):
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            y = selective_scan_fn(u, dt, A, Bm.transpose(1, 2), Cm.transpose(1, 2), m.D.float(), z=None,
                                  delta_bias=m.dt_proj.bias.float(), delta_softplus=True)
            y_h0, h_T = checkpoint(_scan_state_terms, A, delta, u.float(), Bm.float(), Cm.float(), h0, use_reentrant=False)
            y = y.float() + y_h0
        else:
            y, h_T = _selective_scan_ref(u.to(acc), delta, A.to(acc), Bm.to(acc), Cm.to(acc), m.D.to(acc), h0.to(acc))
        y = y.to(u.dtype) * F.silu(gate)
        return m.out_proj(y.transpose(1, 2)), (conv_state, h_T)


class Mamba2Backbone(Backbone):
    """transformers ``Mamba2ForCausalLM``; the SSD kernel takes an initial state and returns the final one
    (both differentiable), the conv continues from its last K-1 inputs."""

    def __init__(self, model):
        super().__init__(model)
        self.inner = model.backbone
        self.num_layers = len(self.inner.layers)
        self.hidden_size = model.config.hidden_size

    embed = MambaBackbone.embed
    norm = MambaBackbone.norm
    block = MambaBackbone.block

    @staticmethod
    def _mixer(m, x, state):
        conv_state, ssm_state = state if state is not None else (None, None)
        b, T, _ = x.shape
        zxbcdt = m.in_proj(x)
        gs = m.n_groups * m.ssm_state_size
        d_mlp = (zxbcdt.shape[-1] - 2 * m.intermediate_size - 2 * gs - m.num_heads) // 2
        _, _, gate, xBC, dt = zxbcdt.split([d_mlp, d_mlp, m.intermediate_size, m.conv_dim, m.num_heads], dim=-1)
        xBC, conv_state = _conv_with_state(xBC.transpose(1, 2), conv_state, m.conv1d, m.activation)
        xs, Bm, Cm = torch.split(xBC.transpose(1, 2), [m.intermediate_size, gs, gs], dim=-1)
        xs, Bm, Cm = xs.reshape(b, T, -1, m.head_dim), Bm.reshape(b, T, m.n_groups, -1), Cm.reshape(b, T, m.n_groups, -1)
        A, limit = -torch.exp(m.A_log.float()), tuple(m.time_step_limit)
        acc = _acc(xs)
        if ssm_state is None:
            ssm_state = x.new_zeros(b, m.num_heads, m.head_dim, m.ssm_state_size, dtype=acc)
        if _use_kernels(x):
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            y, ssm_state = mamba_chunk_scan_combined(
                xs, dt, A, Bm, Cm, chunk_size=m.chunk_size, D=m.D, z=None, dt_bias=m.dt_bias,
                initial_states=ssm_state, dt_softplus=True, return_final_states=True,
                **({} if limit == (0.0, float("inf")) else {"dt_limit": limit}))
        else:
            dt = F.softplus(dt.to(acc) + m.dt_bias.to(acc)).clamp(*limit)
            y, ssm_state = _ssd_ref(xs.to(acc), dt, A.to(acc), Bm.to(acc), Cm.to(acc), m.D.to(acc), ssm_state.to(acc))
        y = m.norm(y.reshape(b, T, -1).to(xs.dtype), gate)
        return m.out_proj(y), (conv_state, ssm_state)


# ------------------------------------------------------------------------------------------- loading
HF_RECURRENT = {"mamba": "MambaForCausalLM", "mamba2": "Mamba2ForCausalLM"}


def _model_class(model_type: str):
    import transformers
    if model_type in HF_RECURRENT:
        # explicit transformers classes: importing fla re-registers "mamba"/"mamba2" for the Auto classes
        return getattr(transformers, HF_RECURRENT[model_type])
    import fla  # noqa: F401  (registers the fla model types with the Auto classes)
    return transformers.AutoModelForCausalLM


def load_base_model(name_or_config: str | dict, dtype: torch.dtype | None = None) -> nn.Module:
    """A pretrained recurrent LM by name, or a randomly initialized one from a config dict
    (``dtype=None``: the current default dtype)."""
    dtype = dtype or torch.get_default_dtype()
    from transformers import AutoConfig, PretrainedConfig
    if isinstance(name_or_config, str):
        model_type = PretrainedConfig.get_config_dict(name_or_config)[0]["model_type"]
        return _model_class(model_type).from_pretrained(name_or_config, dtype=dtype)
    cfg = dict(name_or_config)
    model_type = cfg.pop("model_type")
    cls = _model_class(model_type)
    if model_type in HF_RECURRENT:
        config = cls.config_class(**cfg)
        return cls(config).to(dtype)
    config = AutoConfig.for_model(model_type, **cfg)
    return cls.from_config(config).to(dtype)


def make_backbone(model: nn.Module) -> Backbone:
    model_type = model.config.model_type
    module = type(model).__module__
    if module.startswith("transformers.") and model_type == "mamba":
        return MambaBackbone(model)
    if module.startswith("transformers.") and model_type == "mamba2":
        return Mamba2Backbone(model)
    if (module.startswith("fla.") and model_type not in HF_RECURRENT) or model_type == "gdn2_lm":
        return FLABackbone(model)          # fla models and our GDN-2 LM (fla GDN-2 layers in fla-style blocks)
    raise ValueError(f"{type(model).__name__} ({model_type}) is not a supported recurrent model")
