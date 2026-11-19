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

A memory smaller than the head (d_mem < head_dim): the head's query / key maps are projected to d_mem with one shared
matrix P [d_mem, head_dim] with orthonormal rows, W_mq <- P (q map), W_mk <- P (k map), so that q'.k' = q^T P^T P k keeps
the part of the head's query-key similarity that lies in P's subspace. ``reduce``:
  * ``pca``: P = the top d_mem eigenvectors of E[q q^T] + E[k k^T] over the calibration tokens (q, k: the head's queries
    on the layer's input and keys on its output, as the memory reads and writes them, each scaled to unit length: the
    memory's features are length-normalized);
  * ``random``: a random orthonormal P (the baseline for ``pca``).
W_mb and W_mv do not depend on d_mem.
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


def reduction(q: torch.Tensor | None, k: torch.Tensor | None, d_mem: int, reduce: str, d: int,
              seed: int = 0) -> tuple[torch.Tensor, float]:
    """P [d_mem, d] with orthonormal rows, and the share of E[q q^T] + E[k k^T] (unit-length samples) it keeps
    (q, k: [N, d]; None, for a random P without calibration data: the share of an isotropic spectrum)."""
    unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if q is None:
        if reduce == "pca":
            raise ValueError("a PCA reduction needs calibration data")
        C = torch.eye(d)
    else:
        C = (unit(q).T @ unit(q)) / len(q) + (unit(k).T @ unit(k)) / len(k)                  # [d, d]
    if reduce == "pca":
        P = torch.linalg.eigh(C).eigenvectors[:, -d_mem:].flip(-1).T.contiguous()           # largest first
    elif reduce == "random":
        g = torch.Generator(device="cpu").manual_seed(seed)
        P = torch.linalg.qr(torch.randn(d, d_mem, generator=g))[0].T.to(C)
    else:
        raise ValueError(f"reduce: pca / random, not {reduce!r}")
    return P, (torch.trace(P @ C @ P.T) / torch.trace(C)).item()


@torch.no_grad()
def init_memory_from_gdn(model: RecurrentARMTForCausalLM, parts: str, calib_ids: torch.Tensor | None = None,
                         head: str | int = "slowest", reduce: str = "pca") -> list[dict]:
    """Overwrite the memory parameters of every wrapped layer (see the module docstring). ``calib_ids`` [B, T] (real
    text) is needed for the parts that are calibrated (``b``, ``v``) and for ``reduce="pca"`` when d_mem < head_dim.
    Returns, per layer, what was done."""
    if model.config.base_model_config.get("model_type") != "gdn2_lm":
        raise ValueError("the GDN initialization needs a GDN-2 backbone (gdn2_lm)")
    use = PARTS[parts]
    layers = model.backbone.inner.layers
    stream = None
    d_head = layers[int(next(iter(model.memory_layers)))].attn.head_k_dim
    small = next(iter(model.memory_layers.values())).d_mem < d_head
    if {"b", "v"} & set(use) or (small and reduce == "pca"):
        if calib_ids is None:
            raise ValueError(f"parts={parts!r}{' with a PCA reduction' if small else ''} needs calibration input ids")
    if calib_ids is not None and ({"b", "v"} & set(use) or small):       # (a random P: only to report what it keeps)
        stream = residual_stream(model, calib_ids)
    report = []
    for key, memory in model.memory_layers.items():
        i = int(key)
        block = layers[i]
        attn = block.attn
        d = attn.head_k_dim
        if memory.n_heads != 1 or memory.d_mem > d or attn.head_v_dim != d:
            raise ValueError(f"layer {i}: the GDN init needs one memory head with d_mem <= GDN head_dim ({d}); "
                             f"have n_heads={memory.n_heads}, d_mem={memory.d_mem}")
        h = pick_head(attn, head)
        rows = slice(h * d, (h + 1) * d)
        n1 = block.norm_1.weight.float()
        tap = lambda conv: conv.weight[rows, 0, -1].float()          # causal conv: the last tap multiplies x_t
        dtype = memory.W_mq.weight.dtype
        entry = {"layer": i, "head": h, "A_log": attn.A_log[h].item()}
        q_map = tap(attn.q_conv1d)[:, None] * attn.q_proj.weight[rows].float() * n1          # [d, D]
        k_map = tap(attn.k_conv1d)[:, None] * attn.k_proj.weight[rows].float() * n1
        P = None
        if memory.d_mem < d:            # project the head's query / key space to d_mem (see the module docstring)
            with torch.autocast(q_map.device.type, enabled=False):         # fp32 (train.py calls this under bf16 autocast)
                q = stream[i] @ q_map.T if stream is not None else None      # reads: the layer's input
                k = stream[i + 1] @ k_map.T if stream is not None else None  # writes: the memory tokens' outputs
                P, kept = reduction(q, k, memory.d_mem, reduce, d, seed=i)
            P = P.to(q_map.device)
            entry["reduce"], entry["qk_energy_kept"] = reduce, kept
        if "q" in use:
            memory.W_mq.weight.copy_((q_map if P is None else P @ q_map).to(dtype))
        if "k" in use:
            memory.W_mk.weight.copy_((k_map if P is None else P @ k_map).to(dtype))
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
