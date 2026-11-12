"""Associative memory of one ARMT layer.

The math is the one of ``InnerLoopAssociativeLayerWrapper`` (``_associate_with_mem`` /
``_update_mem_with_mem``) of the original ARMT: a DPFP feature map, a linear-attention style
read ``h + phi(q) W / (phi(q) z)`` and a delta-rule write with the "new information" correction.

The only difference is the representation of an empty memory. The original keeps ``None`` and a
``first_seg`` flag; here the state always has a fixed shape and an all-zero state means "empty":
the read of a zero ``W`` is exactly zero and the write treats rows with ``z == 0`` as the first
segment, which reproduces the ``first_seg`` branch exactly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MemoryState = tuple[torch.Tensor, torch.Tensor | None]   # (W [B, H, d_key/H, d_model/H], z [B, H, d_key/H] or None)


def dpfp(x: torch.Tensor, nu: int = 3) -> torch.Tensor:
    """Deterministic parameter-free projection (Schlag et al., 2021): R^d -> R^{2 nu d}, non-negative."""
    x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
    x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)
    x_repeat = torch.cat([x] * nu, dim=-1)
    return x_repeat * x_rolled


class AssociativeMemory(nn.Module):
    def __init__(self, d_model: int, d_mem: int, n_heads: int = 1, correction: bool = True,
                 use_denom: bool = True, gating: bool = False, nu: int = 3, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert d_mem % n_heads == 0 and d_model % n_heads == 0
        self.d_model, self.d_mem, self.n_heads, self.nu = d_model, d_mem, n_heads, nu
        self.d_key = 2 * nu * d_mem
        self.correction, self.use_denom, self.gating = correction, use_denom, gating
        self.dtype = dtype
        self.read_stats = None          # a list while diagnostics are recorded (RecurrentARMTForCausalLM.memory_stats)
        self.W_mq = nn.Linear(d_model, d_mem, bias=False, dtype=dtype)
        self.W_mk = nn.Linear(d_model, d_mem, bias=False, dtype=dtype)
        self.W_mv = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        self.W_mb = nn.Linear(d_model, d_model if gating else n_heads, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # as in the original: default nn.Linear init, W_mv = 0 (the memory starts as a no-op)
        for lin in (self.W_mq, self.W_mk, self.W_mb):
            lin.reset_parameters()
        nn.init.zeros_(self.W_mv.weight)

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        H = self.n_heads
        W = torch.zeros(batch_size, H, self.d_key // H, self.d_model // H, device=device, dtype=self.dtype)
        z = torch.zeros(batch_size, H, self.d_key // H, device=device, dtype=self.dtype) if self.use_denom else None
        return W, z

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:          # [B, T, D] -> [B, H, T, D/H]
        B, T, D = x.shape
        return x.reshape(B, T, self.n_heads, D // self.n_heads).permute(0, 2, 1, 3)

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:        # [B, H, T, D/H] -> [B, T, D]
        B, H, T, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * d)

    def _features(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(dpfp(self._to_heads(proj(x)), self.nu), dim=-1, p=2.0)

    def read(self, hidden: torch.Tensor, state: MemoryState, record: bool = True) -> torch.Tensor:
        """hidden + associative read-out of the memory (zero for an empty memory). ``record``: count this read in the
        diagnostics (only reads of a non-empty memory are counted)."""
        W, z = state
        with torch.autocast(hidden.device.type, enabled=False):
            mq = self._features(self.W_mq, hidden.to(self.dtype))
            out = torch.einsum("bhtk,bhkd->bhtd", mq, W)
            if self.use_denom:
                out = out / (torch.einsum("bhk,bhtk->bht", z, mq)[..., None] + 1e-5)
        out = self._from_heads(out).to(hidden.dtype)
        if self.read_stats is not None and record and bool(W.any()):
            self.read_stats.append((out.float().norm() / hidden.float().norm().clamp_min(1e-12)).item())
        return hidden + out

    def write(self, mem_tokens: torch.Tensor, state: MemoryState) -> MemoryState:
        """Write the outputs of the memory tokens [B, M, D] into the memory (delta rule)."""
        W, z = state
        with torch.autocast(mem_tokens.device.type, enabled=False):
            x = mem_tokens.to(self.dtype)
            mk = self._features(self.W_mk, x)                                   # [B, H, M, d_key/H]
            new_mv = self._to_heads(self.W_mv(x))                               # [B, H, M, d_model/H]
            prev_mv = torch.einsum("bhmk,bhkd->bhmd", mk, W)
            coef = 1.0
            if self.use_denom:
                denom = torch.einsum("bhk,bhmk->bhm", z, mk)[..., None] + 1e-5
                prev_mv = prev_mv / denom
                if self.correction:
                    coef = torch.clip(1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2)[..., None], 0, 1).detach()
                    empty = (z == 0).all(dim=-1)[..., None, None]               # first segment: coef = 1
                    coef = torch.where(empty, torch.ones_like(coef), coef)
            mb = self._to_heads(torch.sigmoid(self.W_mb(x)))                    # [B, H, M, d_model/H or 1]
            W = W + torch.einsum("bhmk,bhmd->bhkd", mk, (new_mv - prev_mv) * mb)
            if self.use_denom:
                z = z + (coef * mk).sum(dim=-2)
        return W, z
