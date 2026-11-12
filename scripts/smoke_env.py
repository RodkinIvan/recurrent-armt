"""Smoke test of the GPU env: fla RWKV-7 chunk kernel and mamba2 chunk scan with initial/final state gradients."""
import torch
from fla.ops.rwkv7 import chunk_rwkv7
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from causal_conv1d import causal_conv1d_fn

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
B, T, H, K = 2, 100, 4, 64
dev = "cuda"
r, k, v = (torch.randn(B, T, H, K, device=dev, requires_grad=True) for _ in range(3))
w = -torch.rand(B, T, H, K, device=dev) * 0.5
kk = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev), dim=-1)
a = torch.rand(B, T, H, K, device=dev)
h0 = torch.randn(B, H, K, K, device=dev, requires_grad=True)
o, ht = chunk_rwkv7(r=r, w=w, k=k, v=v, a=-kk, b=kk * a, initial_state=h0, output_final_state=True)
(o.sum() + ht.square().sum()).backward()
print("rwkv7 ok", o.shape, ht.shape, ht.dtype, "dh0 norm", h0.grad.norm().item())

x = torch.randn(B, T, H, 32, device=dev, requires_grad=True)
dt = torch.randn(B, T, H, device=dev)
A = -torch.rand(H, device=dev)
Bm = torch.randn(B, T, 1, 16, device=dev)
Cm = torch.randn(B, T, 1, 16, device=dev)
s0 = torch.randn(B, H, 32, 16, device=dev, requires_grad=True)
y, sT = mamba_chunk_scan_combined(x, dt, A, Bm, Cm, chunk_size=64, initial_states=s0, return_final_states=True, dt_softplus=True)
(y.sum() + sT.square().sum()).backward()
print("mamba2 ok", y.shape, sT.shape, sT.dtype, "ds0 norm", s0.grad.norm().item())
xc = torch.randn(B, 64, T, device=dev)
print("conv ok", causal_conv1d_fn(xc, torch.randn(64, 4, device=dev), None, activation="silu").shape)
