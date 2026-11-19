import pytest
import torch

from helpers import HAS_FLA, max_diff, random_ids, tiny_armt
from recurrent_armt.memory_init import init_memory_from_gdn, pick_head, reduction, residual_stream

pytestmark = pytest.mark.skipif(not HAS_FLA, reason="GDN-2 needs fla (CUDA)")


def gdn_armt(d_mem=64, **kw):
    model = tiny_armt("gdn2", active_memory=False, d_mem=d_mem, **kw)     # tiny GDN-2: head_dim 64, 2 heads, 2 layers
    with torch.no_grad():                   # distinct decays, so that "slowest" is well defined
        for i, layer in enumerate(model.backbone.inner.layers):
            layer.attn.A_log.copy_(torch.tensor([1.0, 0.2]) + i)
    return model


@torch.no_grad()
def test_last_conv_tap_multiplies_the_current_token():
    conv = gdn_armt().backbone.inner.layers[0].attn.q_conv1d
    x = torch.randn(1, 1, conv.weight.shape[0], device=conv.weight.device)
    y, _ = conv(x=x)
    assert max_diff(y, torch.nn.functional.silu(conv.weight[:, 0, -1] * x)) < 1e-5


def test_slowest_head_is_the_smallest_A_log():
    attn = gdn_armt().backbone.inner.layers[0].attn
    assert pick_head(attn, "slowest") == 1 and pick_head(attn, "fastest") == 0 and pick_head(attn, "0") == 0


@torch.no_grad()
def test_qk_init_copies_the_head_and_keeps_the_base_model():
    model = gdn_armt()
    ids = random_ids(2, 40)
    before = model(input_ids=ids).logits
    report = init_memory_from_gdn(model, "qk")
    assert [r["head"] for r in report] == [1, 1]
    block = model.backbone.inner.layers[0]
    attn, rows = block.attn, slice(64, 128)
    want = attn.q_conv1d.weight[rows, 0, -1, None] * attn.q_proj.weight[rows] * block.norm_1.weight
    assert max_diff(model.memory_layers["0"].W_mq.weight, want) < 1e-6
    assert not model.memory_layers["0"].W_mv.weight.any()
    assert max_diff(model(input_ids=ids).logits, before) == 0          # W_mv = 0: still exactly the base model


@torch.no_grad()
def test_qkvb_init_calibrates_gate_and_values():
    model = gdn_armt()
    ids = random_ids(4, 64)
    with pytest.raises(ValueError):
        init_memory_from_gdn(model, "qkvb")                                # calibration ids are required
    report = init_memory_from_gdn(model, "qkvb", ids)
    stream = residual_stream(model, ids)
    for r in report:
        memory, x = model.memory_layers[str(r["layer"])], stream[r["layer"] + 1]
        assert memory.W_mv.weight.any() and 0 < r["beta_mean"] < 1
        beta = torch.sigmoid(memory.W_mb(x.to(memory.W_mb.weight.dtype)))
        assert torch.isclose(beta.mean(), torch.tensor(r["beta_mean"], device=beta.device), atol=1e-5)
        block = model.backbone.inner.layers[r["layer"]]
        attn, d, h = block.attn, 64, r["head"]
        rows = slice(h * d, (h + 1) * d)
        v_map = attn.v_conv1d.weight[rows, 0, -1, None] * attn.v_proj.weight[rows] * block.norm_1.weight
        s = (x @ v_map.T).pow(2).mean().sqrt()                             # the values, divided by s, have rms 1
        g = attn.g_proj(block.norm_1(stream[r["layer"]]))[:, rows]
        gate = (torch.nn.functional.silu(g) if attn.o_norm.activation in ("swish", "silu") else torch.sigmoid(g)).mean(0)
        want = (attn.o_proj.weight[:, rows] * attn.o_norm.weight * gate) @ (v_map / s)
        assert max_diff(memory.W_mv.weight, want) < 1e-5
    assert model(input_ids=ids).logits.isfinite().all()


def test_wrong_sizes_and_backbones_are_rejected():
    with pytest.raises(ValueError, match="d_mem"):
        init_memory_from_gdn(tiny_armt("gdn2", active_memory=False, d_mem=128), "qk")        # larger than the head (64)
    with pytest.raises(ValueError, match="GDN-2"):
        init_memory_from_gdn(tiny_armt("gated_deltanet", active_memory=False, d_mem=64), "qk")


def test_reduction_is_orthonormal_and_pca_keeps_the_most():
    g = torch.Generator().manual_seed(0)
    scale = torch.linspace(3, 0.1, 64)                                    # an anisotropic query / key distribution
    q, k = torch.randn(2000, 64, generator=g) * scale, torch.randn(2000, 64, generator=g) * scale
    P, kept = reduction(q, k, 16, "pca", 64)
    R, kept_random = reduction(q, k, 16, "random", 64)
    for M in (P, R):
        assert M.shape == (16, 64) and max_diff(M @ M.T, torch.eye(16)) < 1e-4
    assert kept > kept_random + 0.2 and abs(kept_random - 16 / 64) < 0.1
    assert abs(reduction(q, k, 64, "pca", 64)[1] - 1) < 1e-4            # the full space keeps everything


@torch.no_grad()
@pytest.mark.parametrize("reduce", ["pca", "random"])
def test_small_memory_projects_the_head(reduce):
    model = gdn_armt(d_mem=16)
    ids = random_ids(4, 64)
    before = model(input_ids=ids).logits
    report = init_memory_from_gdn(model, "qk", ids, reduce=reduce)
    stream = residual_stream(model, ids)
    for r in report:
        block = model.backbone.inner.layers[r["layer"]]
        attn, rows = block.attn, slice(r["head"] * 64, (r["head"] + 1) * 64)
        q_map = attn.q_conv1d.weight[rows, 0, -1, None] * attn.q_proj.weight[rows] * block.norm_1.weight
        k_map = attn.k_conv1d.weight[rows, 0, -1, None] * attn.k_proj.weight[rows] * block.norm_1.weight
        P, kept = reduction(stream[r["layer"]] @ q_map.T, stream[r["layer"] + 1] @ k_map.T, 16, reduce, 64, seed=r["layer"])
        memory = model.memory_layers[str(r["layer"])]
        assert memory.W_mq.weight.shape == (16, q_map.shape[1])
        assert max_diff(memory.W_mq.weight, P.to(q_map.device) @ q_map) < 1e-5
        assert max_diff(memory.W_mk.weight, P.to(k_map.device) @ k_map) < 1e-5
        assert r["reduce"] == reduce and abs(r["qk_energy_kept"] - kept) < 1e-5
    assert max_diff(model(input_ids=ids).logits, before) == 0             # W_mv = 0: still exactly the base model


def test_small_memory_pca_needs_calibration_but_random_does_not():
    with pytest.raises(ValueError, match="calibration"):
        init_memory_from_gdn(gdn_armt(d_mem=16), "qk")
    report = init_memory_from_gdn(gdn_armt(d_mem=16), "qk", reduce="random")
    assert all(abs(r["qk_energy_kept"] - 16 / 64) < 1e-5 for r in report)   # isotropic without data: d_mem / d


@torch.no_grad()
def test_small_memory_qkvb_runs():
    model = gdn_armt(d_mem=16)
    ids = random_ids(4, 64)
    report = init_memory_from_gdn(model, "qkvb", ids)
    assert all(0 < r["qk_energy_kept"] <= 1 and r["value_rms"] > 0 for r in report)
    assert model(input_ids=ids).logits.isfinite().all()


@torch.no_grad()
def test_small_memory_init_works_under_bf16_autocast():            # as train.py calls it
    model = gdn_armt(d_mem=16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        report = init_memory_from_gdn(model, "qkvb", random_ids(4, 64))
    assert all(0 < r["qk_energy_kept"] <= 1 for r in report)
