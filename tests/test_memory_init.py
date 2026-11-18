import pytest
import torch

from helpers import HAS_FLA, max_diff, random_ids, tiny_armt
from recurrent_armt.memory_init import init_memory_from_gdn, pick_head, residual_stream

pytestmark = pytest.mark.skipif(not HAS_FLA, reason="GDN-2 needs fla (CUDA)")


def gdn_armt(**kw):
    model = tiny_armt("gdn2", active_memory=False, d_mem=64, **kw)        # tiny GDN-2: head_dim 64, 2 heads, 2 layers
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
        init_memory_from_gdn(tiny_armt("gdn2", active_memory=False, d_mem=8), "qk")
    with pytest.raises(ValueError, match="GDN-2"):
        init_memory_from_gdn(tiny_armt("gated_deltanet", active_memory=False, d_mem=64), "qk")
