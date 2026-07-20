#!/usr/bin/env python3
"""Oracle for the EDSR-base arch converter (M3 ship net) + the npu-sr whole-net gate.

Exports EDSR-base xR: (1) the weights as a safetensors bag the `edsr` arch bakes from (torch keys,
`module.` stripped), (2) per-tensor .npy under artifacts/edsr/ named by the ARENA names edsr.rs emits
(for parity_edsr), (3) a small fixed LR RGB tile + its CPU EDSR SR output (gate_lr/gate_sr) for the
frontier gate. Mirrors scripts/export_espcn.py. Run with a venv that has super_image + torch + safetensors:
    <venv>/bin/python scripts/export_edsr.py [--scale 3]
"""
import argparse, numpy as np, torch
from pathlib import Path
from safetensors.torch import save_file
from super_image import EdsrModel

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=int, default=3)
A = ap.parse_args()
r = A.scale
OUT = Path("artifacts/edsr"); OUT.mkdir(parents=True, exist_ok=True)

m = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=r).eval()
sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in m.state_dict().items()}

# (1) safetensors source (contiguous f32) -- the `path:` bake source.
save_file({k: v.contiguous().float() for k, v in sd.items()}, str(OUT / "edsr_base.safetensors"))

# (2) per-tensor npy refs, named by the arena names edsr.rs emits.
def dump(arena_name, torch_key):
    np.save(OUT / f"{arena_name}.npy", sd[torch_key].numpy().astype(np.float32))

dump("sub_mean_w", "sub_mean.weight");  dump("sub_mean_b", "sub_mean.bias")
dump("head_w", "head.0.weight");        dump("head_b", "head.0.bias")
n_blocks = sum(1 for k in sd if k.startswith("body.") and k.endswith(".body.0.weight"))
for i in range(n_blocks):
    dump(f"b{i}_c0_w", f"body.{i}.body.0.weight"); dump(f"b{i}_c0_b", f"body.{i}.body.0.bias")
    dump(f"b{i}_c1_w", f"body.{i}.body.2.weight"); dump(f"b{i}_c1_b", f"body.{i}.body.2.bias")
dump("btail_w", f"body.{n_blocks}.weight"); dump("btail_b", f"body.{n_blocks}.bias")
dump("tail0_w", "tail.0.0.weight");     dump("tail0_b", "tail.0.0.bias")
dump("tail1_w", "tail.1.weight");       dump("tail1_b", "tail.1.bias")
dump("add_mean_w", "add_mean.weight");  dump("add_mean_b", "add_mean.bias")
print(f"EDSR-base x{r}: {n_blocks} residual blocks; dumped weights to {OUT}")

# (2b) emit edsr.json (the net-as-data schedule) matching the arena names + this scale.
import json
# NOTE: super_image EdsrModel.forward is head -> body(+global skip) -> tail. It does NOT apply the
# sub_mean/add_mean MeanShift layers (they exist in the state_dict but are unused), so they are NOT in
# the schedule -- including them (bias ~-114) would corrupt the [0,1] stream.
ops = [
    {"op": "conv2d", "weights": "head", "k": 3, "pad": 1, "relu": False, "cin": 3, "cout": 64},
    {"op": "save", "name": "g"},
]
for i in range(n_blocks):
    ops += [
        {"op": "save", "name": "s"},
        {"op": "conv2d", "weights": f"b{i}_c0", "k": 3, "pad": 1, "relu": True, "cin": 64, "cout": 64},
        {"op": "conv2d", "weights": f"b{i}_c1", "k": 3, "pad": 1, "relu": False, "cin": 64, "cout": 64},
        {"op": "add", "name": "s"},
    ]
ops += [
    {"op": "conv2d", "weights": "btail", "k": 3, "pad": 1, "relu": False, "cin": 64, "cout": 64},
    {"op": "add", "name": "g"},
    {"op": "conv2d", "weights": "tail0", "k": 3, "pad": 1, "relu": False, "cin": 64, "cout": 64 * r * r},
    {"op": "pixel_shuffle", "r": r},
    {"op": "conv2d", "weights": "tail1", "k": 3, "pad": 1, "relu": False, "cin": 64, "cout": 3},
]
sched = {"name": "edsr", "scale": r, "arena": "target/test-arenas/edsr.safetensors", "input": "rgb", "ops": ops}
(OUT / "edsr.json").write_text(json.dumps(sched, indent=2))
print(f"edsr.json: {len(ops)} ops")

# (3) whole-net CPU reference on a fixed LR RGB tile (deterministic).
rng = np.random.default_rng(0)
lr = rng.random((1, 3, 8, 8), dtype=np.float32)          # [1,3,H,W] RGB in [0,1]
with torch.no_grad():
    sr = m(torch.from_numpy(lr)).numpy()                  # [1,3,3H,3W] RAW (unclamped) -- the frontier
                                                          # is unclamped too; the product clamps only at RGB8.
np.save(OUT / "gate_lr.npy", lr)
np.save(OUT / "gate_sr.npy", sr)
print(f"gate: LR {lr.shape} -> SR {sr.shape}")
