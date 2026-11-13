"""Recurrent ARMT: the Associative Recurrent Memory Transformer recipe on top of a recurrent LM.

The input is split into segments of ``segment_size`` tokens. In every (wrapped) layer and segment:
  1. associative READ: hidden += read(hidden, W, z) for the segment tokens and the memory tokens;
  2. the backbone block runs over the segment, continuing from the block's recurrent state;
  3. the memory tokens (``num_mem_tokens`` trainable embeddings appended after the segment) go
     through the block too, and their outputs WRITE the associative memory (W, z).
What a segment hands to the next one is therefore, per layer, the associative memory AND the
recurrent state of the block (in the transformer ARMT only the associative memory crosses segments).

By default (``mem_tokens_in_stream=False``) the memory tokens are a read-out branch: they start from
the state after the segment's last token, write the memory, and their own state is dropped; the next
segment continues from the state after the last real token. With W_mv = 0 at initialization the model
is then exactly the base model. With ``mem_tokens_in_stream=True`` the next segment continues from the
state after the memory tokens instead (one block call per segment instead of two).

The model can run layers-first (``vertical_mode=False``: every layer over all segments, then the next
layer; "horizontal" in the original) or segments-first (``vertical_mode=True``: every segment through
all layers). Both compute the same thing. The state (``ARMTState``) has a fixed size; ``forward`` takes
and returns it, so a sequence can be fed in pieces of any length (segment boundaries are kept at
absolute positions) and generation is just forward over one token at a time.

Hybrid backbones (recurrent + softmax-attention layers): only the recurrent layers get a memory. An
attention layer is left as it is; its state is the KV cache of the tokens before (it grows with the
sequence). The memory tokens pass through it like through a recurrent block: as a read-out branch they
attend to the cache but are not added to it, so the real tokens see exactly what the base model sees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_leaves, tree_map
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from .backbones import load_base_model, make_backbone
from .loss import lm_loss, lm_token_losses
from .memory import AssociativeMemory

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16, "float64": torch.float64}

Stream = tuple[torch.Tensor, dict]      # (hidden [B, T, D], per-token aux tensors of the backbone)


class RecurrentARMTConfig(PretrainedConfig):
    model_type = "recurrent_armt"

    def __init__(self,
                 base_model_name: str | None = None,
                 base_model_config: dict | None = None,
                 num_mem_tokens: int = 32,
                 d_mem: int = 64,
                 segment_size: int = 1024,
                 n_heads: int = 1,
                 correction: bool = True,
                 use_denom: bool = True,
                 gating: bool = False,
                 wrap_layers: list[int] | None = None,
                 memory_dtype: str = "float32",
                 freeze_mem: bool = False,
                 mem_tokens_in_stream: bool = False,
                 rnn_state_dropout: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name          # a pretrained recurrent LM (used when no base_model_config)
        self.base_model_config = base_model_config      # filled from the base model; used to rebuild it on load
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.segment_size = segment_size
        self.n_heads = n_heads
        self.correction = correction
        self.use_denom = use_denom
        self.gating = gating
        self.wrap_layers = wrap_layers
        self.memory_dtype = memory_dtype
        self.freeze_mem = freeze_mem
        self.mem_tokens_in_stream = mem_tokens_in_stream
        # training only: at every segment boundary, reset the recurrent state of each sample (all layers) to zero
        # with this probability, keeping the associative memory, so that the memory has to carry the context
        self.rnn_state_dropout = rnn_state_dropout


@dataclass
class ARMTState:
    """Everything carried from one segment to the next: for every layer (associative memory (W, z) or
    None for an unwrapped layer, recurrent state of the block), and the number of tokens already seen
    of the current, unfinished segment. Its size does not depend on the sequence length (but for the KV caches of
    the attention layers of a hybrid backbone)."""
    layers: list
    position: int = 0

    def tensors(self) -> list[torch.Tensor]:
        return [t for t in tree_leaves(self.layers) if isinstance(t, torch.Tensor)]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def map(self, fn) -> "ARMTState":
        return ARMTState(map_tensors(fn, self.layers), self.position)

    def detach(self) -> "ARMTState":
        return self.map(lambda t: t.detach())


@dataclass
class ARMTOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    token_losses: torch.Tensor | None = None      # [B, L-1], detached
    state: ARMTState | None = None


def map_tensors(fn, tree):
    """tree_map over the tensors of a pytree, leaving None (and other leaves) as they are."""
    return tree_map(lambda t: fn(t) if isinstance(t, torch.Tensor) else t, tree)


def _cat_aux(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]], dim=1) for k in a}


def _slice_aux(a: dict, start: int, end: int | None) -> dict:
    return {k: v[:, start:end] for k, v in a.items()}


class RecurrentARMTForCausalLM(PreTrainedModel):
    config_class = RecurrentARMTConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: RecurrentARMTConfig, base_model: nn.Module | None = None):
        super().__init__(config)
        if base_model is None:
            if config.base_model_config is not None:
                base_model = load_base_model(config.base_model_config)
            elif config.base_model_name is not None:
                base_model = load_base_model(config.base_model_name, dtype=torch.float32)
            else:
                raise ValueError("set base_model_name or base_model_config (or pass base_model)")
        config.base_model_config = base_model.config.to_dict()
        self.model = base_model
        self.backbone = make_backbone(base_model)

        n_layers, d_model = self.backbone.num_layers, self.backbone.hidden_size
        memory_dtype = DTYPES[config.memory_dtype]
        # by default every recurrent layer; attention layers of hybrids stay as they are (no memory)
        attention = [self.backbone.is_attention(i) for i in range(n_layers)]
        wrap = config.wrap_layers if config.wrap_layers is not None else [int(not a) for a in attention]
        assert len(wrap) == n_layers, f"wrap_layers has {len(wrap)} entries for {n_layers} layers"
        if any(w and a for w, a in zip(wrap, attention)):
            raise ValueError(f"wrap_layers {wrap}: the associative memory goes on recurrent layers only, "
                             f"attention layers are {[i for i, a in enumerate(attention) if a]}")
        self.memory = nn.Parameter(torch.empty(config.num_mem_tokens, d_model, dtype=memory_dtype))
        self.memory_layers = nn.ModuleDict({
            str(i): AssociativeMemory(d_model, config.d_mem, n_heads=config.n_heads, correction=config.correction,
                                      use_denom=config.use_denom, gating=config.gating, dtype=memory_dtype)
            for i in range(n_layers) if wrap[i] and config.num_mem_tokens > 0})
        self.vertical_mode = False
        self.rnn_state_dropout = config.rnn_state_dropout   # runtime value (train.py schedules it)
        self.state_reset = False       # reset the recurrent state at every boundary, also in eval (memory-only context)
        self.gradient_checkpointing = False
        self.post_init()
        if config.freeze_mem:
            for memory in self.memory_layers.values():
                memory.requires_grad_(False)

    # ------------------------------------------------------------------ HF plumbing
    def _init_weights(self, module):
        # only ARMT's own parameters; the base model initializes (or has loaded) its own
        if isinstance(module, AssociativeMemory):
            module.reset_parameters()
        elif module is self:
            nn.init.normal_(self.memory, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        """Checkpoint every (layer, segment) block call: only the inputs and states between them are kept."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------ state
    def _memory_of(self, i: int) -> AssociativeMemory | None:
        return self.memory_layers[str(i)] if str(i) in self.memory_layers else None

    @torch.no_grad()
    def init_state(self, batch_size: int, device: torch.device | None = None) -> ARMTState:
        """Empty memories and zero recurrent states (an empty KV cache, None, for attention layers). The shapes /
        dtypes of the recurrent states are taken from one token pushed through the blocks (call it under the autocast
        context that the model runs in)."""
        device = torch.device(device or self.memory.device)
        # The probe runs under no_grad: it must not fill autocast's weight-cast cache, or the rest of the step
        # would reuse those (gradient-free) bf16 weight copies and the backbone would get no weight gradients.
        kind = device.type
        no_cache = torch.autocast(kind, dtype=torch.get_autocast_dtype(kind), enabled=torch.is_autocast_enabled(kind),
                                  cache_enabled=False)
        with no_cache:
            hidden = self.backbone.embed(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
            aux, layers = {}, []
            for i in range(self.backbone.num_layers):
                hidden, aux, rnn_state = self.backbone.block(i, hidden, aux, None)
                memory = self._memory_of(i)
                layers.append((memory.init_state(batch_size, device) if memory is not None else None,
                               None if self.backbone.is_attention(i) else map_tensors(torch.zeros_like, rnn_state)))
        return ARMTState(layers, position=0)

    # ------------------------------------------------------------------ core
    def _segments(self, length: int, position: int) -> tuple[list[tuple[int, int, bool]], int]:
        """Split [0, length) at absolute segment boundaries, given `position` tokens already in the current
        segment. Returns (start, end, completes_segment) triples and the position after the last one."""
        S, bounds, start = self.config.segment_size, [], 0
        while start < length:
            end = min(length, start + S - position)
            done = position + end - start == S
            bounds.append((start, end, done))
            position = 0 if done else position + end - start
            start = end
        return bounds, position

    def _mem_stream(self, batch_size: int, dtype: torch.dtype, write: bool) -> Stream | None:
        if not write or self.config.num_mem_tokens == 0:
            return None
        # contiguous: unwrapped layers get the memory tokens as an ordinary tensor, not a broadcast view of the parameter
        return self.memory.to(dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous(), {}

    def _layer(self, i: int, real: Stream, mem: Stream | None, state):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._layer_impl, i, real, mem, state, use_reentrant=False)
        return self._layer_impl(i, real, mem, state)

    def _layer_impl(self, i: int, real: Stream, mem: Stream | None, state):
        """One layer over one segment: read -> block (continuing the recurrent state) -> write."""
        mem_state, rnn_state = state
        memory = self._memory_of(i)
        hidden, aux = real
        if memory is not None:
            hidden = memory.read(hidden, mem_state)
            if mem is not None:
                mem = (memory.read(mem[0], mem_state, record=False), mem[1])    # diagnostics: segment tokens only
        block = self.backbone.block
        if mem is None:                                    # unfinished segment: no memory tokens yet
            hidden, aux, rnn_state = block(i, hidden, aux, rnn_state)
        elif self.config.mem_tokens_in_stream:
            T = hidden.shape[1]
            both, both_aux, rnn_state = block(i, torch.cat([hidden, mem[0]], dim=1), _cat_aux(aux, mem[1]), rnn_state)
            hidden, aux, mem = both[:, :T], _slice_aux(both_aux, 0, T), (both[:, T:], _slice_aux(both_aux, T, None))
        else:
            hidden, aux, rnn_state = block(i, hidden, aux, rnn_state)
            mem_hidden, mem_aux, _ = block(i, mem[0], mem[1], rnn_state)    # read-out branch, its state is dropped
            mem = (mem_hidden, mem_aux)
        if memory is not None and mem is not None:
            mem_state = memory.write(mem[0], mem_state)
        return (hidden, aux), mem, (mem_state, rnn_state)

    def _reset_masks(self, n_segments: int, batch_size: int, device, position: int) -> list:
        """Per segment, None or a [B] mask of the samples whose recurrent state is reset before it (state dropout
        in training / ``state_reset``), at every segment boundary: before each segment but the first, and before
        the first if the call starts at a boundary (``position == 0``; resetting a fresh zero state is a no-op).
        Drawn once, so that the layers-first and segments-first orders apply the same resets."""
        p = 1.0 if self.state_reset else (self.rnn_state_dropout if self.training else 0.0)
        if p <= 0:
            return [None] * n_segments
        masks = [torch.rand(batch_size, device=device) < p for _ in range(n_segments)]
        return masks if position == 0 else [None] + masks[1:]

    def _reset_rnn(self, i: int, state, mask):
        """Zero the recurrent state of layer ``i`` for the samples in ``mask``; an attention layer keeps its KV cache."""
        if mask is None or self.backbone.is_attention(i):
            return state
        mem_state, rnn_state = state
        reset = lambda t: torch.where(mask.view(-1, *[1] * (t.dim() - 1)), torch.zeros_like(t), t)
        return mem_state, map_tensors(reset, rnn_state)

    def _run_vertical(self, x, state: ARMTState, length: int | None = None, last_only: bool = False):
        """Segments first: each segment goes through all layers before the next segment starts.
        ``x`` is the embedded input [B, L, D], or a function (start, end) -> the embedded slice (then ``length`` is
        L): with ``last_only`` the hidden states of all but the last segment are dropped as soon as the state has
        been carried forward, so memory does not grow with the sequence length."""
        take = x if callable(x) else (lambda start, end: x[:, start:end])
        length = x.shape[1] if length is None else length
        bounds, position = self._segments(length, state.position)
        masks = None
        layers, outputs = list(state.layers), []
        for k, (start, end, write) in enumerate(bounds):
            seg = take(start, end)
            if masks is None:
                masks = self._reset_masks(len(bounds), seg.shape[0], seg.device, state.position)
            real, mem = (seg, {}), self._mem_stream(seg.shape[0], seg.dtype, write)
            for i in range(self.backbone.num_layers):
                real, mem, layers[i] = self._layer(i, real, mem, self._reset_rnn(i, layers[i], masks[k]))
            outputs = [real[0]] if last_only else outputs + [real[0]]
        return torch.cat(outputs, dim=1), ARMTState(layers, position)

    def _run_horizontal(self, x: torch.Tensor, state: ARMTState) -> tuple[torch.Tensor, ARMTState]:
        """Layers first: each layer processes all segments before the next layer starts."""
        bounds, position = self._segments(x.shape[1], state.position)
        reals = [(x[:, start:end], {}) for start, end, _ in bounds]
        mems = [self._mem_stream(x.shape[0], x.dtype, write) for *_, write in bounds]
        masks = self._reset_masks(len(bounds), x.shape[0], x.device, state.position)
        layers = list(state.layers)
        for i in range(self.backbone.num_layers):
            if self._whole_attention(i):
                reals, mems, layers[i] = self._attention_layer(i, reals, mems, layers[i])
                continue
            for k in range(len(bounds)):
                reals[k], mems[k], layers[i] = self._layer(i, reals[k], mems[k], self._reset_rnn(i, layers[i], masks[k]))
        return torch.cat([r[0] for r in reals], dim=1), ARMTState(layers, position)

    def _whole_attention(self, i: int) -> bool:
        """Layers-first, an attention layer of a hybrid takes all segments at once (``Backbone.attention_layer``)
        when the memory tokens are a read-out branch (in the stream they would enter the KV cache between segments)."""
        return (self.backbone.is_attention(i) and not self.config.mem_tokens_in_stream
                and hasattr(self.backbone, "attention_layer"))

    def _attention_layer(self, i: int, reals: list, mems: list, state):
        """Attention layer ``i`` over all segments: the real tokens in one causal call continuing the KV cache, the
        memory tokens of all segments in one more (each reading the cache up to its segment's end). Equal to the
        segment-by-segment ``_layer`` calls up to rounding (one softmax over the whole cache instead of per segment)."""
        assert all(not r[1] for r in reals), "per-token aux streams are not supported in attention layers"
        lengths = [r[0].shape[1] for r in reals]
        ends = torch.tensor(lengths).cumsum(0).tolist()
        readouts = [(mem[0], end) for mem, end in zip(mems, ends) if mem is not None]
        args = (i, torch.cat([r[0] for r in reals], dim=1), state[1], readouts)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            hidden, kv, outs = checkpoint(self.backbone.attention_layer, *args, use_reentrant=False)
        else:
            hidden, kv, outs = self.backbone.attention_layer(*args)
        outs = iter(outs)
        mems = [None if mem is None else (next(outs), {}) for mem in mems]
        return [(h, {}) for h in hidden.split(lengths, dim=1)], mems, (state[0], kv)

    def forward(self,
                input_ids: torch.LongTensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                labels: torch.LongTensor | None = None,
                labels_mask: torch.Tensor | None = None,
                state: ARMTState | None = None,
                return_state: bool = False,
                output_logits: bool | str | None = None,
                loss_chunk_size: int | None = 4096,
                **kwargs: Any) -> ARMTOutput:
        """``state=None`` starts a new sequence. Logits are returned when there are no labels (or with
        ``output_logits=True``; ``"last"``: only for the last position, and then, without labels, the memory used does
        not grow with the sequence length); with labels the loss is computed without materializing all logits."""
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("padding is not supported: the recurrent state would absorb the pad tokens")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        if output_logits is None:
            output_logits = labels is None
        source = input_ids if input_ids is not None else inputs_embeds
        if state is None:
            state = self.init_state(source.shape[0], source.device)
        if output_logits == "last" and labels is None:
            # only the last position is needed: segments-first, embedding one segment at a time and keeping only
            # the last segment's hidden states -> memory independent of the sequence length (same result: H == V)
            take = (lambda s, e: self.backbone.embed(input_ids[:, s:e])) if inputs_embeds is None else inputs_embeds
            hidden, state = self._run_vertical(take, state, length=source.shape[1], last_only=True)
        else:
            x = self.backbone.embed(input_ids) if inputs_embeds is None else inputs_embeds
            run = self._run_vertical if self.vertical_mode else self._run_horizontal
            hidden, state = run(x, state)
        hidden = self.backbone.norm(hidden)

        logits = None
        if output_logits == "last":
            logits = self.backbone.lm_head(hidden[:, -1:])
        elif output_logits:
            logits = self.backbone.lm_head(hidden)
        loss = token_losses = None
        if labels is not None:
            token_losses = lm_token_losses(hidden, self.backbone.lm_head, labels, labels_mask, loss_chunk_size)
            loss = lm_loss(token_losses, labels, labels_mask)
            token_losses = token_losses.detach()
        return ARMTOutput(loss=loss, logits=logits, token_losses=token_losses, state=state if return_state else None)

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def memory_stats(self, input_ids: torch.LongTensor) -> dict:
        """How much the memory does: per wrapped layer, |read-out| / |hidden| averaged over the reads of the segment
        tokens that see a non-empty memory (segments after the first), and the size of W_mv (0 at init)."""
        layers = list(self.memory_layers.items())
        if not layers:
            return {}
        for _, memory in layers:
            memory.read_stats = []
        try:
            self(input_ids=input_ids, output_logits=False)
        finally:
            ratios = {k: (sum(m.read_stats) / max(1, len(m.read_stats))) for k, m in layers}
            for _, memory in layers:
                memory.read_stats = None
        w_mv = {k: m.W_mv.weight.float().pow(2).mean().sqrt().item() for k, m in layers}
        return {"mem/read_ratio": sum(ratios.values()) / len(ratios), "mem/read_ratio_max": max(ratios.values()),
                "mem/W_mv_rms": sum(w_mv.values()) / len(w_mv),
                **{f"mem/read_ratio/{k}": v for k, v in ratios.items()}}

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 1, eos_token_id: int | list[int] | None = None,
                 do_sample: bool = False, temperature: float = 1.0, top_k: int | None = None,
                 return_logits: bool = False, state: ARMTState | None = None, attention_mask=None, **kwargs):
        """Decode token by token on top of the fixed-size state (O(1) work per token; the memory tokens are
        processed whenever a segment is completed). Returns only the new tokens, like the original ARMT,
        and the logits they were chosen from if ``return_logits``. A sample that has emitted an EOS token
        keeps emitting that token; decoding stops when every sample has finished."""
        out = self(input_ids=input_ids, attention_mask=attention_mask, state=state, return_state=True, output_logits="last")
        logits, state = out.logits[:, -1], out.state
        eos = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id or [])
        eos_ids = torch.tensor(eos, device=input_ids.device, dtype=input_ids.dtype)
        finished = torch.zeros(input_ids.shape[0], 1, dtype=torch.bool, device=input_ids.device)
        new_tokens, new_logits = [], []
        for step in range(max_new_tokens):
            new_logits.append(logits)
            if do_sample:
                scores = logits.float() / temperature
                if top_k is not None:
                    kth = torch.topk(scores, top_k, dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < kth, float("-inf"))
                token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            if eos:
                token = torch.where(finished, eos_ids[0], token)
                finished |= torch.isin(token, eos_ids)
            new_tokens.append(token)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break
            out = self(input_ids=token, state=state, return_state=True, output_logits="last")
            logits, state = out.logits[:, -1], out.state
        tokens = torch.cat(new_tokens, dim=1)
        return (tokens, torch.stack(new_logits, dim=1)) if return_logits else tokens
