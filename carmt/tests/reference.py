"""An exact (no TF32) reference for the GDN-2 token mixer: fla's chunk / fused-recurrent kernels multiply in TF32 even
for fp32 inputs, so two correct implementations differ by ~1e-3 through them. ``exact_gdn2()`` swaps them, inside the
fla GDN-2 layer, for fla's own token-by-token PyTorch reference (``naive_recurrent_gdn2``, differentiable), with the
in-kernel q/k l2norm done the same way (x / sqrt(sum x^2 + 1e-6))."""
from __future__ import annotations

import contextlib

import torch


def _l2(x):
    return x / torch.sqrt((x.float() * x.float()).sum(-1, keepdim=True) + 1e-6).to(x.dtype)


def _naive(q, k, v, g, b, w, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kw):
    from fla.ops.gdn2.naive import naive_recurrent_gdn2
    assert cu_seqlens is None
    if use_qk_l2norm_in_kernel:
        q, k = _l2(q), _l2(k)
    o, h = naive_recurrent_gdn2(q, k, v, g, b, w, initial_state=initial_state, output_final_state=True)
    return o, (h if output_final_state else None)


@contextlib.contextmanager
def exact_gdn2():
    import fla.layers.gdn2 as layer
    saved = layer.chunk_gdn2, layer.fused_recurrent_gdn2
    layer.chunk_gdn2 = layer.fused_recurrent_gdn2 = _naive
    try:
        yield
    finally:
        layer.chunk_gdn2, layer.fused_recurrent_gdn2 = saved


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| / max |b|"""
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
