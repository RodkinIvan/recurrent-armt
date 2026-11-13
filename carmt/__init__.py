"""carmt: a CUDA implementation of ARMT over a Gated DeltaNet-2 backbone (``armt.cu``), usable from PyTorch.

It computes exactly what ``recurrent_armt.RecurrentARMTForCausalLM`` computes for a GDN-2 base model (same parameters,
same ``ARMTState``), in one CUDA file: cuBLAS for the matmuls, fused kernels for everything else, a chunked GDN-2
scan, and a hand-written backward that recomputes every (layer, segment) step from its stored input.

    import carmt
    carmt.accelerate(model)            # model: RecurrentARMTForCausalLM over GDN2ForCausalLM; patches model.forward
    out = model(input_ids=x, labels=x) # same call and output as before (loss, token_losses, logits, state)
    out.loss.backward()                # gradients land in the model's .grad as usual (works under DDP)

or without patching: ``runner = carmt.CARMT(model); out = runner(input_ids=..., labels=...)``.

Precision follows the caller: under ``torch.autocast("cuda", dtype=torch.bfloat16)`` the bf16 build runs (bf16 matmuls
and activations, fp32 residual stream, fp32 scan state and fp32 memory, like autocast); otherwise the fp32 build
(fp32 everything; TF32 in fp32 matmuls iff ``torch.backends.cuda.matmul.allow_tf32``).

Training (grad enabled + labels) records the forward inside the C context and runs the C backward when
``loss.backward()`` reaches it; one training forward may be pending per model at a time (forward -> backward ->
next forward, as in ``train.py``). Calls the C path does not cover (``inputs_embeds``, a state or logits with grad)
fall back to the original PyTorch forward.
"""
from __future__ import annotations

import atexit
import ctypes
import fcntl
import os
import subprocess
import sys
import weakref
from typing import Any

import torch

__all__ = ["CARMT", "accelerate", "build", "library_path", "mark_weights_changed"]

# The bf16 weight copies are re-packed when the parameters may have changed. Version counters and data pointers catch
# loads and in-place edits, but fused optimizers (e.g. AdamW(fused=True)) update parameters without bumping the version
# counters: every torch.optim step (also the one inside DeepSpeed's ZeRO optimizer) is counted through a global hook.
_WEIGHT_EPOCH = [0]


def mark_weights_changed():
    """Force a re-pack of the weight copies on the next forward (after changing parameters in a way neither an optimizer
    step nor a version counter shows, e.g. a raw NCCL broadcast into them)."""
    _WEIGHT_EPOCH[0] += 1


from torch.optim.optimizer import register_optimizer_step_post_hook  # noqa: E402

register_optimizer_step_post_hook(lambda *_: mark_weights_changed())

_HERE = os.path.dirname(os.path.abspath(__file__))
MAXL = 256
_N_GLOBAL = 4   # embeddings, final norm, lm_head, memory tokens: the parameters before the per-layer ones
_L_NAMES = ["n1", "q", "k", "v", "b", "w", "f1", "g1", "qc", "kc", "vc", "f2", "alog", "dtb", "g2", "g2b", "on", "o", "n2",
            "w1", "w2", "w3", "mq", "mk", "mv", "mb", "mbb"]
_L_TORCH = {"n1": "norm_1.weight", "q": "attn.q_proj.weight", "k": "attn.k_proj.weight", "v": "attn.v_proj.weight",
            "b": "attn.b_proj.weight", "w": "attn.w_proj.weight", "f1": "attn.f_proj.0.weight", "g1": "attn.g_proj.0.weight",
            "qc": "attn.q_conv1d.weight", "kc": "attn.k_conv1d.weight", "vc": "attn.v_conv1d.weight",
            "f2": "attn.f_proj.1.weight", "alog": "attn.A_log", "dtb": "attn.dt_bias", "g2": "attn.g_proj.1.weight",
            "g2b": "attn.g_proj.1.bias", "on": "attn.o_norm.weight", "o": "attn.o_proj.weight", "n2": "norm_2.weight",
            "w1": "mlp.swiglu.w1.weight", "w2": "mlp.swiglu.w2.weight", "w3": "mlp.swiglu.w3.weight"}
_L_MEM = {"mq": "W_mq.weight", "mk": "W_mk.weight", "mv": "W_mv.weight", "mb": "W_mb.weight", "mbb": "W_mb.bias"}


class ArmtConfig(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ("vocab", "d_model", "n_layers", "n_heads", "head_dim", "d_ff", "conv_size", "gate_swish")] + \
               [("norm_eps", ctypes.c_float)] + \
               [(n, ctypes.c_int) for n in ("num_mem", "d_mem", "mem_heads", "segment_size", "mem_in_stream", "gating", "tie_emb",
                                            "allow_tf32")] + \
               [("wrap", ctypes.c_int * MAXL)]


class ArmtForwardArgs(ctypes.Structure):
    _fields_ = [("ids", ctypes.c_void_p), ("B", ctypes.c_int), ("T", ctypes.c_int), ("labels", ctypes.c_void_p),
                ("inv_n", ctypes.c_void_p), ("token_losses", ctypes.c_void_p), ("logits", ctypes.c_void_p),
                ("logits_last", ctypes.c_int), ("position", ctypes.c_int), ("train", ctypes.c_int), ("reset", ctypes.c_void_p),
                ("state", ctypes.POINTER(ctypes.c_void_p)), ("grad", ctypes.c_void_p), ("repack", ctypes.c_int),
                ("stream", ctypes.c_void_p)]


# the library's GEMM algorithm choices (one tuning per GPU model and cuBLAS version, shared by processes and runs;
# see mm() in armt.cu). Set ARMT_GEMM_CACHE="" to keep them in-process only, ARMT_GEMM_TUNE=0 to use cuBLAS's defaults.
os.environ.setdefault("ARMT_GEMM_CACHE", os.path.join(_HERE, ".gemm_cache"))


def library_path(precision: str) -> str:
    return os.path.join(_HERE, f"libarmt_{precision}.so")


def build(arch: str | None = None, cuda_home: str | None = None) -> None:
    """Compile the two libraries (``make`` in this directory). Safe to call from several processes at once (torchrun):
    one builds under a file lock, the others wait and find the result."""
    cmd = ["make", "-C", _HERE]
    if arch:
        cmd.append(f"ARCH={arch}")
    if cuda_home:
        cmd.append(f"CUDA_HOME={cuda_home}")
    with open(os.path.join(_HERE, ".build.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            subprocess.run(cmd, check=True)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


_LIBS: dict[str, ctypes.CDLL] = {}
_ALLOC_FN = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
_FREE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


@_ALLOC_FN
def _torch_alloc(nbytes, stream):
    """The library's device memory comes from PyTorch's caching allocator (visible in torch.cuda memory stats)."""
    try:
        return torch.cuda.caching_allocator_alloc(int(nbytes), stream=int(stream or 0))
    except Exception:      # out of memory: the C side raises with the size
        return None


@_FREE_FN
def _torch_free(ptr, stream):
    torch.cuda.caching_allocator_delete(int(ptr))


def _lib(precision: str) -> ctypes.CDLL:
    if precision not in _LIBS:
        path = library_path(precision)
        if not os.path.exists(path):
            major, minor = torch.cuda.get_device_capability()
            build(arch=f"sm_{major}{minor}")
        lib = ctypes.CDLL(path)
        vp, i, l = ctypes.c_void_p, ctypes.c_int, ctypes.c_long
        lib.armt_last_error.restype = ctypes.c_char_p
        lib.armt_set_allocator.argtypes = [_ALLOC_FN, _FREE_FN]
        lib.armt_set_allocator(_torch_alloc, _torch_free)
        lib.armt_release.argtypes, lib.armt_release.restype = [vp, i], i
        lib.armt_set_grad_mask.argtypes = [vp, ctypes.c_char_p]
        lib.armt_create.argtypes, lib.armt_create.restype = [ctypes.POINTER(ArmtConfig)], vp
        lib.armt_destroy.argtypes = [vp]
        lib.armt_num_params.argtypes, lib.armt_num_params.restype = [vp], l
        lib.armt_param_numel.argtypes, lib.armt_param_numel.restype = [vp, i], l
        lib.armt_param_offset.argtypes, lib.armt_param_offset.restype = [vp, i], l
        lib.armt_grad_size.argtypes, lib.armt_grad_size.restype = [vp], l
        lib.armt_set_tf32.argtypes = [vp, i]
        lib.armt_forward.argtypes, lib.armt_forward.restype = [vp, ctypes.POINTER(vp), ctypes.POINTER(ArmtForwardArgs)], i
        lib.armt_backward.argtypes, lib.armt_backward.restype = [vp, ctypes.POINTER(vp), vp, vp, vp], i
        lib.armt_backward_range.argtypes, lib.armt_backward_range.restype = [vp, ctypes.POINTER(vp), vp, vp, vp, i, i], i
        _LIBS[precision] = lib
    return _LIBS[precision]


def _ptr(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


def _check(lib, rc, what):
    if rc != 0:
        raise RuntimeError(f"carmt {what}: {lib.armt_last_error().decode()}")


_LIVE: "weakref.WeakSet[_Context]" = weakref.WeakSet()


@atexit.register
def _destroy_all():
    """Free every C context while PyTorch is still intact. Freeing goes back into PyTorch's allocator through a ctypes
    callback, which crashes if it runs during interpreter teardown (a model kept in a module global until exit)."""
    for c in list(_LIVE):
        c.close()


class _Context:
    """One C context (one library build) for one model."""

    def __init__(self, precision: str, cfg: ArmtConfig, params: list[torch.Tensor | None]):
        self.lib = _lib(precision)
        self.precision = precision
        self.handle = self.lib.armt_create(ctypes.byref(cfg))
        _LIVE.add(self)
        if not self.handle:
            raise RuntimeError(f"carmt: {self.lib.armt_last_error().decode()}")
        n = self.lib.armt_num_params(self.handle)
        assert n == len(params), (n, len(params))
        self.numel = [self.lib.armt_param_numel(self.handle, j) for j in range(n)]
        self.offset = [self.lib.armt_param_offset(self.handle, j) for j in range(n)]
        self.grad_size = self.lib.armt_grad_size(self.handle)
        for j, p in enumerate(params):
            want = self.numel[j]
            if p is not None and want and p.numel() != want:
                raise ValueError(f"parameter {j}: {p.numel()} elements, the C layout expects {want}")
        self.pending = None        # token of the training forward awaiting its backward
        self.version = None
        self.mask = None

    def close(self):
        h, self.handle = getattr(self, "handle", None), None
        if h:
            self.lib.armt_destroy(h)

    def __del__(self, _finalizing=sys.is_finalizing):   # bound now: module globals may be None during teardown
        if not getattr(self, "handle", None) or _finalizing():   # closed, or too late to call back into PyTorch
            return
        try:
            self.close()
        except Exception:
            pass


class _Pending:
    """One recorded training forward, shared by the autograd nodes of its backward."""

    def __init__(self, runner, cctx, ids, grad, needs, keep):
        self.runner, self.cctx, self.ids, self.grad, self.needs, self.keep = runner, cctx, ids, grad, needs, keep
        self.ptrs = None

    def run(self, lo: int, hi: int, gloss):
        """C backward of layers [lo, hi) (and of the embeddings if lo == 0); hi = n_layers first, then downwards."""
        cctx = self.cctx
        if cctx.pending is not self:
            raise RuntimeError("carmt: another training forward ran between this forward and its backward")
        if self.ptrs is None:
            self.ptrs = self.runner._param_ptrs()
        g = gloss.detach().float().reshape(1).contiguous()
        rc = cctx.lib.armt_backward_range(cctx.handle, self.ptrs, self.ids.data_ptr(), g.data_ptr(),
                                          torch.cuda.current_stream().cuda_stream, lo, hi)
        if lo == 0:
            cctx.pending = None
        _check(cctx.lib, rc, "backward")
        if lo == 0:
            _check(cctx.lib, cctx.lib.armt_release(cctx.handle, 0), "release")   # the training stores go back to PyTorch's cache
            self.keep = None

    def grads(self, j0: int, params):
        out = []
        for j, p in enumerate(params, j0):
            if p is None or not self.needs[j] or self.cctx.numel[j] == 0:
                out.append(None)
            else:
                o, n = self.cctx.offset[j], self.cctx.numel[j]
                out.append(self.grad[o:o + n].view(p.shape))
        return out


class _TrainStep(torch.autograd.Function):
    """loss = mean next-token CE of the C forward. The backward is split into one autograd node per layer
    (``_LayerBackward``, chained on top of this one): each layer's gradients are handed to autograd as soon as the C
    backward has finished that layer, so DDP / DeepSpeed start reducing them while the layers below still run. This
    node's own backward does layer 0 and the embeddings and returns their and the global parameters' gradients."""

    @staticmethod
    def forward(ctx, runner: "CARMT", cctx: _Context, ids, labels, inv_n, reset, anchor, params_all, *params_own):
        B, T = ids.shape
        mask = bytes(int(p is not None and p.requires_grad) for p in params_all)
        if mask != cctx.mask:      # frozen parameters: their gradient work is skipped in C
            cctx.lib.armt_set_grad_mask(cctx.handle, mask)
            cctx.mask = mask
        grad = torch.zeros(cctx.grad_size, device=ids.device, dtype=torch.float32)
        tl = torch.zeros(B, max(T - 1, 0), device=ids.device, dtype=torch.float32)
        runner._call_forward(cctx, ids, labels=labels, inv_n=inv_n, token_losses=tl, train=True, reset=reset, grad=grad)
        # the C backward reads the reset masks and labels: keep them alive (not reused by the allocator) until it ran
        pend = _Pending(runner, cctx, ids, grad, [p is not None and p.requires_grad for p in params_all], (reset, labels, inv_n))
        cctx.pending = pend
        ctx.pend = pend
        ctx.own = params_own   # the global parameters and layer 0's
        loss = tl.sum() * inv_n[0]
        ctx.mark_non_differentiable(tl)
        return loss, tl

    @staticmethod
    def backward(ctx, gloss, _gtl):
        pend = ctx.pend
        pend.run(0, min(1, pend.runner.cfg.n_layers), gloss)
        grads = pend.grads(0, ctx.own)
        ctx.pend = ctx.own = None
        return (None, None, None, None, None, None, None, None, *grads)


class _LayerBackward(torch.autograd.Function):
    """Identity on the loss; its backward runs the C backward of layers [lo, hi) and returns their gradients."""

    @staticmethod
    def forward(ctx, pend: _Pending, lo: int, hi: int, j0: int, loss, *params):
        ctx.pend, ctx.lo, ctx.hi, ctx.j0, ctx.params = pend, lo, hi, j0, params
        return loss.clone()

    @staticmethod
    def backward(ctx, gloss):
        ctx.pend.run(ctx.lo, ctx.hi, gloss)
        grads = ctx.pend.grads(ctx.j0, ctx.params)
        ctx.pend = ctx.params = None
        return (None, None, None, None, gloss, *grads)


class CARMT:
    """Runs a ``RecurrentARMTForCausalLM`` (GDN-2 backbone) through the C/CUDA implementation, on its own parameters."""

    def __init__(self, model, scan_fp32: bool = False):
        """``scan_fp32``: under bf16 autocast run the GDN-2 scan in fp32 (library ``libarmt_bf16s32``): bf16 matmuls as
        usual, fp32 scan intermediates and CUDA-core scan kernels. Slower, numerically closer to the fp32 model."""
        from recurrent_armt.gdn2 import GDN2ForCausalLM
        self.model = model
        self.scan_fp32 = scan_fp32
        cfg = model.config
        base = model.model
        if not isinstance(base, GDN2ForCausalLM):
            raise TypeError(f"carmt supports GDN-2 backbones (GDN2ForCausalLM), not {type(base).__name__}")
        bc = base.config
        attn = base.model.layers[0].attn
        if not (attn.use_short_conv and not attn.conv_bias and not attn.allow_neg_eigval and attn.num_v_heads == attn.num_heads
                and attn.head_v_dim == attn.head_k_dim):
            raise ValueError("carmt: unsupported GDN-2 variant (needs short conv without bias, expand_v=1, no GVA, no neg eigval)")
        if cfg.num_mem_tokens > 0 and not (cfg.use_denom and cfg.correction and cfg.memory_dtype == "float32"):
            raise ValueError("carmt: the memory needs use_denom=True, correction=True, memory_dtype=float32")
        c = ArmtConfig()
        c.vocab, c.d_model, c.n_layers = bc.vocab_size, bc.hidden_size, bc.num_hidden_layers
        c.n_heads, c.head_dim, c.d_ff, c.conv_size = attn.num_heads, attn.head_k_dim, bc.intermediate_size, attn.conv_size
        c.gate_swish = int(getattr(bc, "output_gate", "swish") == "swish")
        c.norm_eps = bc.norm_eps
        c.num_mem, c.d_mem, c.mem_heads, c.segment_size = cfg.num_mem_tokens, cfg.d_mem, cfg.n_heads, cfg.segment_size
        c.mem_in_stream, c.gating = int(cfg.mem_tokens_in_stream), int(cfg.gating)
        emb, head = base.model.embeddings.weight, base.lm_head.weight
        c.tie_emb = int(emb is head or emb.data_ptr() == head.data_ptr())
        wrap = cfg.wrap_layers if cfg.wrap_layers is not None else [1] * c.n_layers
        for i in range(c.n_layers):
            c.wrap[i] = int(wrap[i])
        self.cfg = c
        self.contexts: dict[str, _Context] = {}

    # ------------------------------------------------------------------ parameters in the C order
    @property
    def params(self) -> list[torch.Tensor | None]:
        m, base = self.model, self.model.model
        ps = [base.model.embeddings.weight, base.model.norm.weight, None if self.cfg.tie_emb else base.lm_head.weight,
              m.memory if self.cfg.num_mem > 0 else None]
        for i, layer in enumerate(base.model.layers):
            mods = dict(layer.named_parameters())
            memory = m._memory_of(i)
            mm = dict(memory.named_parameters()) if memory is not None else {}
            for n in _L_NAMES:
                ps.append(mods[_L_TORCH[n]] if n in _L_TORCH else mm.get(_L_MEM[n]))
        return ps

    def _anchor(self, device):
        a = self.__dict__.setdefault("_anchors", {}).get(device)
        if a is None:
            a = self._anchors[device] = torch.zeros((), device=device, requires_grad=True)
        return a

    def _param_ptrs(self):
        ps = self.params
        for p in ps:
            if p is not None and not (p.is_cuda and p.dtype == torch.float32 and p.is_contiguous()):
                raise ValueError("carmt: parameters must be contiguous float32 CUDA tensors")
        return (ctypes.c_void_p * len(ps))(*[_ptr(p) for p in ps])

    def _context(self, precision: str) -> _Context:
        if precision == "bf16" and self.scan_fp32:
            precision = "bf16s32"
        if precision not in self.contexts:
            self.contexts[precision] = _Context(precision, self.cfg, self.params)
        cctx = self.contexts[precision]
        cctx.lib.armt_set_tf32(cctx.handle, int(torch.backends.cuda.matmul.allow_tf32))   # fp32 matmuls (memory, small scan GEMMs)
        return cctx

    @staticmethod
    def _precision() -> str:
        if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:
            return "bf16"
        return "fp32"

    def _weights_version(self):
        return _WEIGHT_EPOCH[0], tuple((p.data_ptr(), p._version) for p in self.params if p is not None)

    # ------------------------------------------------------------------ state conversion (original ARMTState <-> C layout)
    def _zero_state(self, B: int, device) -> list[torch.Tensor]:
        c, H, K = self.cfg, self.cfg.n_heads, self.cfg.head_dim
        out = []
        dk, Hm = 6 * c.d_mem, max(c.mem_heads, 1)
        for i in range(c.n_layers):
            has_mem = c.num_mem > 0 and c.wrap[i]
            out += [torch.zeros(B, H, K, K, device=device), torch.zeros(B, 3 * H * K, c.conv_size, device=device),
                    torch.zeros(B, Hm, dk // Hm, c.d_model // Hm, device=device) if has_mem else None,
                    torch.zeros(B, Hm, dk // Hm, device=device) if has_mem else None]
        return out

    def _from_armt_state(self, state) -> list[torch.Tensor | None]:
        out = []
        for mem, rnn in state.layers:
            out.append(rnn["recurrent_state"].float().contiguous().clone())
            out.append(torch.cat([t.float() for t in rnn["conv_state"]], dim=1).contiguous())
            if mem is not None:
                out += [mem[0].float().contiguous().clone(), mem[1].float().contiguous().clone()]
            else:
                out += [None, None]
        return out

    def _to_armt_state(self, st: list, position: int):
        from recurrent_armt.modeling import ARMTState
        HK = self.cfg.n_heads * self.cfg.head_dim
        layers = []
        for i in range(self.cfg.n_layers):
            S, cs, W, z = st[4 * i:4 * i + 4]
            rnn = {"recurrent_state": S, "conv_state": tuple(cs[:, j * HK:(j + 1) * HK].contiguous() for j in range(3))}
            layers.append(((W, z) if W is not None else None, rnn))
        return ARMTState(layers, position)

    # ------------------------------------------------------------------ calls
    def _call_forward(self, cctx: _Context, ids, labels=None, inv_n=None, token_losses=None, logits=None, logits_last=False,
                      position=0, train=False, reset=None, state=None, grad=None):
        B, T = ids.shape
        a = ArmtForwardArgs()
        a.ids, a.B, a.T = ids.data_ptr(), B, T
        a.labels, a.inv_n = _ptr(labels), _ptr(inv_n)
        a.token_losses, a.logits, a.logits_last = _ptr(token_losses), _ptr(logits), int(logits_last)
        a.position, a.train, a.reset = position, int(train), _ptr(reset)
        st_arr = None
        if state is not None:
            st_arr = (ctypes.c_void_p * len(state))(*[_ptr(t) for t in state])
            a.state = st_arr
        a.grad = _ptr(grad)
        version = self._weights_version()
        a.repack = int(version != cctx.version)   # optimizer steps / loads bump the version counters
        cctx.version = version
        a.stream = torch.cuda.current_stream().cuda_stream
        rc = cctx.lib.armt_forward(cctx.handle, self._param_ptrs(), ctypes.byref(a))
        _check(cctx.lib, rc, "forward")

    def _reset_masks(self, n_pieces: int, B: int, device, position: int):
        m = self.model
        p = 1.0 if m.state_reset else (m.rnn_state_dropout if m.training else 0.0)
        if p <= 0:
            return None
        masks = torch.stack([torch.rand(B, device=device) < p for _ in range(n_pieces)]).to(torch.uint8)
        if position != 0:
            masks[0] = 0
        return masks.contiguous()

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, labels_mask=None, state=None,
                return_state: bool = False, output_logits: bool | str | None = None, loss_chunk_size=None, **kwargs: Any):
        from recurrent_armt.modeling import ARMTOutput
        m = self.model
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("padding is not supported: the recurrent state would absorb the pad tokens")
        grad_on = torch.is_grad_enabled() and any(p.requires_grad for p in m.parameters())
        if output_logits is None:
            output_logits = labels is None
        unsupported = inputs_embeds is not None or not input_ids.is_cuda or (
            grad_on and (labels is None or state is not None or return_state or output_logits))
        if unsupported:
            return m.__class__.forward(m, input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                       labels=labels, labels_mask=labels_mask, state=state, return_state=return_state,
                                       output_logits=output_logits, **kwargs)
        ids = input_ids.long().contiguous()
        B, T = ids.shape
        position = state.position if state is not None else 0
        bounds, new_position = m._segments(T, position)
        reset = self._reset_masks(len(bounds), B, ids.device, position)
        inv_n = None
        if labels is not None:
            labels = labels.long()
            if labels_mask is not None:
                labels = labels.clone()
                labels[:, 1:] = labels[:, 1:].masked_fill(~labels_mask[:, :-1].bool(), -100)
            labels = labels.contiguous()
            torch._assert_async(((labels >= 0) & (labels < self.cfg.vocab) | (labels == -100)).all(), "label out of range")
            inv_n = (1.0 / (labels[:, 1:] != -100).sum().clamp(min=1).float()).reshape(1)
        cctx = self._context(self._precision())

        if grad_on:
            ps = self.params
            anchor = self._anchor(ids.device)   # keeps every node of the chain on the backward path, frozen layers too
            nl = len(_L_NAMES)
            loss, tl = _TrainStep.apply(self, cctx, ids, labels, inv_n, reset, anchor, ps, *ps[:_N_GLOBAL + nl])
            pend = cctx.pending
            # bottom to top: the backward runs the top layer first. CARMT_LAYER_NODES=0: one node for layers 1..L-1
            per_layer = os.environ.get("CARMT_LAYER_NODES", "1") != "0"
            groups = [(i, i + 1) for i in range(1, self.cfg.n_layers)] if per_layer else \
                ([(1, self.cfg.n_layers)] if self.cfg.n_layers > 1 else [])
            for lo, hi in groups:
                j0 = _N_GLOBAL + lo * nl
                loss = _LayerBackward.apply(pend, lo, hi, j0, loss, *ps[j0:_N_GLOBAL + hi * nl])
            return ARMTOutput(loss=loss, logits=None, token_losses=tl, state=None)

        with torch.no_grad():
            st = None
            if state is not None:
                st = self._from_armt_state(state)
            elif return_state:
                st = self._zero_state(B, ids.device)
            V = self.cfg.vocab
            logits = None
            if output_logits == "last":
                logits = torch.empty(B, 1, V, device=ids.device)
            elif output_logits:
                logits = torch.empty(B, T, V, device=ids.device)
            tl = torch.zeros(B, max(T - 1, 0), device=ids.device) if labels is not None else None
            self._call_forward(cctx, ids, labels=labels, inv_n=inv_n, token_losses=tl, logits=logits,
                               logits_last=output_logits == "last", position=position, reset=reset, state=st)
            loss = tl.sum() * inv_n[0] if labels is not None else None
            out_state = self._to_armt_state(st, new_position) if return_state else None
            return ARMTOutput(loss=loss, logits=logits, token_losses=tl, state=out_state)

    __call__ = forward

    def release_memory(self):
        """Give all of the library's device memory (packed weights, scratch) back to PyTorch's cache."""
        for cctx in self.contexts.values():
            _check(cctx.lib, cctx.lib.armt_release(cctx.handle, 1), "release")


def accelerate(model, scan_fp32: bool = False):
    """Route ``model.forward`` (a GDN-2 ``RecurrentARMTForCausalLM``) through the CUDA implementation. Everything else
    (parameters, optimizer, save/load, generate, DDP) stays as it is. Returns the model; ``model.carmt`` is the runner and
    ``model.forward_reference`` the original forward."""
    runner = CARMT(model, scan_fp32=scan_fp32)
    model.carmt = runner
    model.forward_reference = model.__class__.forward.__get__(model)
    model.forward = runner.forward
    orig_stats = model.memory_stats

    def memory_stats(input_ids):     # needs the per-read statistics hooks of the PyTorch path
        model.forward = model.forward_reference
        try:
            return orig_stats(input_ids)
        finally:
            model.forward = runner.forward
    model.memory_stats = memory_stats
    return model
