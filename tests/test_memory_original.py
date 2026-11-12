"""The associative memory computes what the original ARMT (InnerLoopAssociativeLayerWrapper of
irodkin/ARMT-Gemma-3-1B-IT-8x1024-portable) computes, including the first-segment special case."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from recurrent_armt.memory import AssociativeMemory, dpfp


class Original:
    """``_associate_with_mem`` / ``_update_mem_with_mem`` copied from the original modeling_armt.py."""

    def __init__(self, mem: AssociativeMemory):
        self.m = mem
        self.n_heads, self.use_denom, self.correction, self.gating = mem.n_heads, mem.use_denom, mem.correction, mem.gating
        self.memory_dtype = mem.dtype
        self.W_mq, self.W_mk, self.W_mv, self.W_mb = mem.W_mq, mem.W_mk, mem.W_mv, mem.W_mb
        self.phi = lambda x: dpfp(x, 3)

    def _to_heads(self, x):
        bsz, seq_len, d_model = x.shape
        x = x.reshape(bsz, seq_len, self.n_heads, d_model // self.n_heads)
        return x.permute(0, 2, 1, 3)

    def _from_heads(self, x):
        bsz, n_heads, seq_len, d_head = x.shape
        return x.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * d_head)

    def _associate_with_mem(self, hidden_states, W_mem, z):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.memory_dtype)
        q = self._to_heads(self.W_mq(hidden_states))
        mq = self.phi(q)
        mq = F.normalize(mq, dim=-1, p=2.0)
        num = torch.einsum("ihjk,ihkt->ihjt", mq, W_mem)
        if self.use_denom and z is not None:
            denom = torch.einsum("ihk,ihjk->ihj", z, mq)[..., None] + 1e-5
            hs = num / denom
        else:
            hs = num
        return self._from_heads(hs).to(original_dtype)

    def _update_mem_with_mem(self, mem_tokens, W_mem, z, first_seg):
        mem_tokens = mem_tokens.to(self.memory_dtype)
        k = self._to_heads(self.W_mk(mem_tokens))
        mk = self.phi(k)
        mk = F.normalize(mk, dim=-1, p=2.0)
        new_mv = self._to_heads(self.W_mv(mem_tokens))
        if not first_seg:
            num = torch.einsum("ihjk,ihkt->ihjt", mk, W_mem)
            if self.use_denom and z is not None:
                denom = torch.einsum("ihj,ihkj->ihk", z, mk)[..., None] + 1e-5
                prev_mv = num / denom
                if self.correction:
                    new_info_coef = (1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2)[..., None])
                    new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
                else:
                    new_info_coef = 1
            else:
                prev_mv = num
                new_info_coef = 1
        else:
            prev_mv = torch.zeros_like(new_mv, device=new_mv.device)
            new_info_coef = 1
        mv = new_mv - prev_mv
        mb = self._to_heads(torch.sigmoid(self.W_mb(mem_tokens)))
        einop = f"ihjk,ihjt,ihj{'t' if self.gating else 'x'}->ihkt"
        associations = torch.einsum(einop, mk, mv, mb)
        W_mem = W_mem + associations
        if self.use_denom and z is not None:
            z = z + (new_info_coef * mk).sum(dim=-2)
        return W_mem, z, False


@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.parametrize("use_denom,correction", [(True, True), (True, False), (False, False)])
def test_memory_matches_original(n_heads, gating, use_denom, correction):
    torch.manual_seed(0)
    B, T, M, D = 3, 11, 5, 32
    mem = AssociativeMemory(D, 16, n_heads=n_heads, correction=correction, use_denom=use_denom, gating=gating,
                            dtype=torch.float64)
    torch.nn.init.normal_(mem.W_mv.weight, std=0.3)
    orig = Original(mem)

    state = mem.init_state(B, torch.device("cpu"))
    W0 = torch.zeros(1, n_heads, mem.d_key // n_heads, D // n_heads, dtype=torch.float64)   # original: batch 1, then broadcast
    o_W, o_z, first = W0, (torch.zeros(1, n_heads, mem.d_key // n_heads, dtype=torch.float64) if use_denom else None), True
    for seg in range(4):
        h = torch.randn(B, T, D, dtype=torch.float64)
        m = torch.randn(B, M, D, dtype=torch.float64)
        ours = mem.read(h, state)
        ref = h if first else h + orig._associate_with_mem(h, o_W, o_z)          # the original skips the read at first
        assert torch.allclose(ours, ref, atol=1e-12, rtol=0), seg
        state = mem.write(m, state)
        o_W, o_z, first = orig._update_mem_with_mem(m, o_W, o_z, first)
        assert torch.allclose(state[0], o_W.expand_as(state[0]), atol=1e-12, rtol=0), seg
        if use_denom:
            assert torch.allclose(state[1], o_z.expand_as(state[1]), atol=1e-12, rtol=0), seg
