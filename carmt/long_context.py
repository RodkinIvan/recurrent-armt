"""Device memory and throughput vs context length (batch 1, loss only, no grad, bf16): carmt vs the original (vertical).

    python carmt/long_context.py carmt|original 65536,1048576,4194304 [loss|last]
"""
import sys, time, torch
sys.path.insert(0, ".")
import carmt
from recurrent_armt import RecurrentARMTConfig, RecurrentARMTForCausalLM
from recurrent_armt.backbones import load_base_model
which, lengths = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
mode = sys.argv[3] if len(sys.argv) > 3 else "loss"   # "loss": labels, loss over all tokens; "last": last-position logits only
torch.backends.cuda.matmul.allow_tf32 = True
base = load_base_model("checkpoints/gdn2-370m-30B", dtype=torch.float32).cuda()
m = RecurrentARMTForCausalLM(RecurrentARMTConfig(num_mem_tokens=32, d_mem=64, segment_size=1024), base_model=base).cuda().eval()
m.vertical_mode = True
fwd = carmt.CARMT(m) if which == "carmt" else m
torch.cuda.synchronize()
free0, _ = torch.cuda.mem_get_info()
for T in lengths:
    ids = torch.randint(3, 32000, (1, T), device="cuda")
    torch.cuda.synchronize(); t = time.perf_counter()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "loss":
                loss = fwd(input_ids=ids, labels=ids).loss.item()
            else:
                loss = fwd(input_ids=ids, output_logits="last").logits.float().logsumexp(-1).mean().item()
    except torch.OutOfMemoryError:
        print(f"{which:8s} {mode} T={T/1e6:5.2f}M  OOM", flush=True); break
    dt = time.perf_counter() - t
    free1, _ = torch.cuda.mem_get_info()
    print(f"{which:8s} {mode} T={T/1e6:5.2f}M  device memory used {(free0 - free1)/2**30:6.2f} GB  {T/dt/1e3:6.1f}k tok/s  loss {loss:.3f}", flush=True)
