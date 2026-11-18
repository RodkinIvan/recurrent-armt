"""Initialize each layer's associative memory from that layer's GDN-2 time mixer, instead of at random.

GDN and ARMT are both delta-rule associative memories, so one GDN head per layer maps onto the ARMT memory of the layer:
the head's query / key maps become W_mq / W_mk, its erase gate the write strength W_mb, and its value path (value
projection, output norm, the head's slice of the output projection) the value map W_mv. The head is, by default, the one
with the slowest decay (smallest A_log): the one that keeps context longest.

    W_mq <- diag(c_q) q_proj[h] diag(n1)          c: the current-token tap of the short conv, n1: the block's RMSNorm weight
    W_mk <- diag(c_k) k_proj[h] diag(n1)
    W_mb <- mean over the head's channels of b_proj[h] diag(n1) / r          (bias 0; a scalar write strength)
    W_mv <- o_proj[:, h] diag(o_norm * gate) diag(1 / s) diag(c_v) v_proj[h] diag(n1)

What is left out: the SiLU after the conv, the decay, the token dependence of the output gate (``gate`` is its mean
over the calibration tokens, per channel) and the 1/rms of the block's input norm. The memory's features are L2-normalized DPFP maps, so the scale of W_mq / W_mk does not matter. The
gate and the values do see the raw residual stream, so their scale is calibrated on real hidden states: r is the rms of
the residual stream after the layer (what the memory tokens write), s the rms of the head's values on it, which makes
the values unit-rms as GDN's output norm does. The read is a weighted mean of values, so it then has the scale of the
head's output.

Parts: ``qk`` (W_mv stays 0: the model is still exactly the base model), ``qkb``, ``qkvb``.
"""
from __future__ import annotations

import torch

from .modeling import RecurrentARMTForCausalLM

PARTS = {"qk": ("q", "k"), "qkb": ("q", "k", "b"), "qkvb": ("q", "k", "b", "v")}


def pick_head(attn, head: str | int) -> int:
    if isinstance(head, int) or str(head).lstrip("-").isdigit():
        return int(head)
    order = attn.A_log.detach().float().argsort()            # smallest A_log = slowest decay
    if head == "slowest":
        return int(order[0])
    if head == "fastest":
        return int(order[-1])
    raise ValueError(f"head: slowest / fastest / an index, not {head!r}")


@torch.no_grad()
def residual_stream(model: RecurrentARMTForCausalLM, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """The base model's residual stream before layer 0, ..., after the last layer: L + 1 tensors [N, D] (fp32)."""
    bb = model.backbone
    hidden, aux = bb.embed(input_ids), {}
    stream = [hidden.reshape(-1, hidden.shape[-1]).float()]
    for i in range(bb.num_layers):
        hidden, aux, _ = bb.block(i, hidden, aux, None)
        stream.append(hidden.reshape(-1, hidden.shape[-1]).float())
    return stream


def _gate(attn, normed: torch.Tensor, h: int, d: int) -> torch.Tensor:
    """Mean over tokens of the output gate of head h, per channel [d] (sigmoid, or SiLU for the official GDN-2)."""
    g = attn.g_proj(normed.to(attn.g_proj[0].weight.dtype)).float()[:, h * d:(h + 1) * d]
    act = getattr(attn.o_norm, "activation", "sigmoid")
    return (torch.nn.functional.silu(g) if act in ("swish", "silu") else torch.sigmoid(g)).mean(0)


@torch.no_grad()
def init_memory_from_gdn(model: RecurrentARMTForCausalLM, parts: str, calib_ids: torch.Tensor | None = None,
                         head: str | int = "slowest") -> list[dict]:
    """Overwrite the memory parameters of every wrapped layer (see the module docstring). ``calib_ids`` [B, T] (real
    text) is needed for the parts that are calibrated (``b``, ``v``). Returns, per layer, what was done."""
    if model.config.base_model_config.get("model_type") != "gdn2_lm":
        raise ValueError("the GDN initialization needs a GDN-2 backbone (gdn2_lm)")
    use = PARTS[parts]
    layers = model.backbone.inner.layers
    stream = None
    if {"b", "v"} & set(use):
        if calib_ids is None:
            raise ValueError(f"parts={parts!r} needs calibration input ids")
        stream = residual_stream(model, calib_ids)
    report = []
    for key, memory in model.memory_layers.items():
        i = int(key)
        block = layers[i]
        attn = block.attn
        d = attn.head_k_dim
        if memory.n_heads != 1 or memory.d_mem != d or attn.head_v_dim != d:
            raise ValueError(f"layer {i}: the GDN init needs one memory head with d_mem = GDN head_dim ({d}); "
                             f"have n_heads={memory.n_heads}, d_mem={memory.d_mem}")
        h = pick_head(attn, head)
        rows = slice(h * d, (h + 1) * d)
        n1 = block.norm_1.weight.float()
        tap = lambda conv: conv.weight[rows, 0, -1].float()          # causal conv: the last tap multiplies x_t
        dtype = memory.W_mq.weight.dtype
        entry = {"layer": i, "head": h, "A_log": attn.A_log[h].item()}
        if "q" in use:
            memory.W_mq.weight.copy_((tap(attn.q_conv1d)[:, None] * attn.q_proj.weight[rows].float() * n1).to(dtype))
        if "k" in use:
            memory.W_mk.weight.copy_((tap(attn.k_conv1d)[:, None] * attn.k_proj.weight[rows].float() * n1).to(dtype))
        x = stream[i + 1] if stream is not None else None          # after the layer: what the memory tokens write
        if "b" in use:
            r = x.pow(2).mean(-1).sqrt().mean()
            w = (attn.b_proj.weight[rows].float() * n1).mean(0, keepdim=True) / r
            memory.W_mb.weight.copy_(w.expand_as(memory.W_mb.weight).to(dtype))
            memory.W_mb.bias.zero_()
            entry["beta_mean"] = torch.sigmoid(x @ w.T).mean().item()
        if "v" in use:
            v_map = tap(attn.v_conv1d)[:, None] * attn.v_proj.weight[rows].float() * n1          # [d, D]
            s = (x @ v_map.T).pow(2).mean().sqrt()
            gate = _gate(attn, block.norm_1(stream[i]), h, d)                                        # [d]
            out_map = attn.o_proj.weight[:, h * d:(h + 1) * d].float() * (attn.o_norm.weight.float() * gate)    # [D, d]
            memory.W_mv.weight.copy_((out_map @ (v_map / s)).to(dtype))
            entry["gate_mean"] = gate.mean().item()
            entry["value_rms"] = (x @ memory.W_mv.weight.float().T).pow(2).mean().sqrt().item()
            entry["residual_rms"] = x.pow(2).mean().sqrt().item()
        report.append(entry)
    return report
