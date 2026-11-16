"""A plain Gated DeltaNet-2 causal LM: fla's ``GatedDeltaNet2`` token mixer in a pre-norm block with a SwiGLU
MLP. It matches the layout of the GDN-2 checkpoints trained on FineWeb-Edu with ``lit_gpt``
(LLM-OS-Models/gdn2-*-fineweb-edu-100b), whose intermediate (not annealed) snapshots can be converted with
``scripts/convert_gdn2.py``. fla ships the GDN-2 layer and kernels but no model class.

The blocks follow fla's block interface (``block(hidden, attention_mask, past_key_values, use_cache)`` with the
recurrent state in an fla cache), so ``FLABackbone`` drives them.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class GDN2Config(PretrainedConfig):
    model_type = "gdn2_lm"

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 1024, num_hidden_layers: int = 16,
                 num_heads: int = 16, head_dim: int = 128, expand_v: float = 1.0, intermediate_size: int = 2048,
                 conv_size: int = 4, norm_eps: float = 1e-5, output_gate: str = "swish", allow_neg_eigval: bool = False,
                 tie_word_embeddings: bool = False, bos_token_id: int = 1, eos_token_id: int = 2, **kwargs):
        self.vocab_size, self.hidden_size, self.num_hidden_layers = vocab_size, hidden_size, num_hidden_layers
        self.num_heads, self.head_dim, self.expand_v = num_heads, head_dim, expand_v
        self.intermediate_size, self.conv_size, self.norm_eps = intermediate_size, conv_size, norm_eps
        self.output_gate = output_gate          # "swish" (official GDN-2) or "sigmoid" (fla's GDN-2 layer)
        # erase gate in [0, 2] instead of [0, 1]: transition eigenvalues in [-1, 1] (the checkpoints use False)
        self.allow_neg_eigval = allow_neg_eigval
        super().__init__(tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))).to(dtype)


class SwiGLU(nn.Module):
    """xformers / lit_gpt naming: w3(silu(w1 x) * w2 x)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.swiglu = SwiGLU(dim, hidden)

    def forward(self, x):
        return self.swiglu(x)


class GDN2Block(nn.Module):
    def __init__(self, config: GDN2Config, layer_idx: int):
        super().__init__()
        from fla.layers.gdn2 import GatedDeltaNet2
        self.norm_1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = GatedDeltaNet2(hidden_size=config.hidden_size, expand_v=config.expand_v, head_dim=config.head_dim,
                                   num_heads=config.num_heads, conv_size=config.conv_size, layer_idx=layer_idx,
                                   norm_eps=config.norm_eps, allow_neg_eigval=getattr(config, "allow_neg_eigval", False))
        if config.output_gate == "swish":
            # the official GDN-2 (NVlabs/GatedDeltaNet-2, used for these checkpoints) gates the output norm with
            # SiLU; fla's port uses a sigmoid. The norm's weight is the same parameter either way.
            from fla.modules import FusedRMSNormGated
            self.attn.o_norm = FusedRMSNormGated(self.attn.head_v_dim, activation="swish", eps=config.norm_eps)
        self.norm_2 = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        out, _, past_key_values = self.attn(self.norm_1(hidden_states), attention_mask=attention_mask,
                                            past_key_values=past_key_values, use_cache=use_cache)
        hidden_states = hidden_states + out
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states, None, past_key_values


class GDN2Model(nn.Module):
    def __init__(self, config: GDN2Config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GDN2Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)


class GDN2ForCausalLM(PreTrainedModel):
    config_class = GDN2Config
    base_model_prefix = "model"
    _no_split_modules = ["GDN2Block"]

    def __init__(self, config: GDN2Config):
        super().__init__(config)
        self.model = GDN2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)) and not getattr(module.weight, "_is_hf_initialized", False):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.model.embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, labels=None, past_key_values=None, use_cache=False, **kwargs):
        """Plain full-sequence forward (the reference for the ARMT adapter)."""
        h = self.model.embeddings(input_ids)
        for block in self.model.layers:
            h, _, past_key_values = block(h, past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(self.model.norm(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten())
        return CausalLMOutput(loss=loss, logits=logits)


def lit_gpt_to_hf(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a lit_gpt GDN-2 state dict to GDN2ForCausalLM's names (the block internals are identical)."""
    out = {}
    for k, v in state.items():
        k = k.replace("transformer.wte.", "model.embeddings.").replace("transformer.ln_f.", "model.norm.")
        k = k.replace("transformer.h.", "model.layers.")
        out[k] = v
    return out
