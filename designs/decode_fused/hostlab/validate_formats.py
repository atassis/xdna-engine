"""Gate 1: our host `sym` simulator must agree with the SHIPPED packer's math.
Gate 2: reproduce the recorded rel-L2 sweep so the instrument is anchored to the KB."""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/qlab")
import sys, glob, numpy as np
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# the shipped packer, three levels up from designs/decode_fused/hostlab/
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))), "iron_operators"))
from safetensors.numpy import load_file
import wq_formats as F
from iron.operators.gemv.quant import quantize_weight, dequantize_weight

from huggingface_hub import snapshot_download
snap = _os.path.join(
    snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.safetensors", "*.json"]),
    "model.safetensors")
sd = load_file(snap)
print(f"loaded {len(sd)} tensors from {snap.split('/')[-2][:8]}")

# The recorded sweep's tensors: gate/up/down proj at layers 0, 13, 27.
names = [f"model.layers.{L}.mlp.{p}.weight" for L in (0, 13, 27)
         for p in ("gate_proj", "up_proj", "down_proj")]

print("\n--- gate 1: host sym simulator vs shipped packer (g=128, int4, f32 scale) ---")
worst = 0.0
for n in names:
    W = sd[n].astype(np.float32)
    packed = quantize_weight(W, 128, "int4")
    # emulate_kernel_scale_cast=False is the arm quantize_sym(kernel_round=False) reproduces:
    # both keep the f32 scale. Comparing it against the =True arm was a harness bug that read as
    # a 2.9e-03 disagreement for a year of nobody running this.
    ref = dequantize_weight(packed, W.shape[0], W.shape[1], 128, "int4",
                            emulate_kernel_scale_cast=False)
    mine = F.quantize_sym(W, 128, 4, scale_dtype="f32", kernel_round=False)
    worst = max(worst, float(np.abs(ref - mine).max()))
print(f"max |shipped_golden - host_sim| over 9 tensors: {worst:.3e}  "
      f"({'MATCH' if worst == 0 else 'DIFFER'})")

print("\n--- gate 2: rel-L2 vs the recorded sweep (K=1024 rows, mean over 9 tensors) ---")
print(f"{'scheme':34s} {'bits/elt':>8s} {'mean rel-L2':>12s}   recorded")
rec = {("sym", 32): 0.0997, ("sym", 64): 0.1116, ("sym", 128): 0.1230, ("sym", 256): 0.1345,
       # the recorded asymmetric sweep used a uint4 zero point; the form we ship constrains the
       # bf16 min to the grid, which lands on the same values
       ("affine", 32): 0.0822, ("affine", 64): 0.0934, ("affine", 128): 0.1042,
       ("affine", 256): 0.1150}
for scheme in ("sym", "affine"):
    for g in (32, 64, 128, 256):
        errs = []
        for n in names:
            W = sd[n].astype(np.float32)
            spec = dict(scheme=scheme, nbits=4, group=g, kernel_round=False,
                        scale_dtype="f32" if scheme == "sym" else "bf16")
            if scheme == "affine":
                spec["zero_on_grid"] = True
            errs.append(F.rel_l2(W, F.apply(W, spec)))
        b = F.bits_per_element(4, g, scheme,
                               "f32" if scheme == "sym" else "bf16", "bf16")
        r = rec[(scheme, g)]
        print(f"{scheme+' int4 g'+str(g):34s} {b:8.3f} {np.mean(errs):12.4f}   {r:.4f}")
