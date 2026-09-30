"""Per-layer device weight streams for the 48-layer stack on the layer image (`rlayer_design m g`):
`stack_prep.py [first] [last]` writes RF_STACK_OUT/w{li}.npy, each the command's
%w: [8 x attn_in, QKV on the down form][8 x attn_out][8 x MLP, grouped].

Sliding layers take the store's attn_in and MLP blobs (byte-identical to the gated streams) and
re-pack O in the x V workers' K order (the store's is the heads order). Global layers are built
from the planar source by grun.weights' path (pair-aligned K split, O in the device's order)."""
import json
import os
import sys
import numpy as np
import rf_paths
import rlayer_design as L
import rattn2_design as A2

# Overridable so recipes/gemma4_data.sh can rebuild into a scratch root for verification without
# touching the live store; defaults are this repo's artifacts/build trees (see rf_paths.py).
STORE = os.environ.get("RF_STORE", str(rf_paths.ARTIFACTS / "store"))
OUT = os.environ.get("RF_STACK_OUT", str(rf_paths.BUILD_ROOT / "scratch/stack"))


def blob(e):
    return np.fromfile(f"{STORE}/blobs/{e['blob']}.bin", dtype=np.uint8, count=e["length"], offset=e["offset"])


def sliding(li, mat):
    import o_ref
    import weight_store as ws
    o_ref.KORDER = "workers"
    L.set_geo(False)
    a_in = L.qkv_downform(blob(mat["attn_in_stream"]))
    a_out = o_ref.stream(ws.load_o_sliding(li)).view(np.uint8).reshape(-1)
    mlp = L.mlp_grouped(blob(mat["mlp_stream"]))
    assert a_in.size == L.NC * L.wcol() and a_out.size == L.NC * L.WCOL_O
    w = np.concatenate([a_in, a_out, mlp])
    assert w.size == L.wbytes(), (w.size, L.wbytes())
    return w


def glob(li):
    import grun
    grun.G_LAYER = li
    grun.WCACHE = f"{OUT}/.g{li}.npy"
    w = grun.weights()
    os.remove(grun.WCACHE)
    L.set_geo(True)
    assert w.size == L.wbytes(), (w.size, L.wbytes())
    L.set_geo(False)
    return w


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 47
    L.ATTN, L.OPH, L.MPH, L.GLOB = True, True, True, True
    L.LAYER.oa = A2.FINISH_A = True
    m = json.load(open(f"{STORE}/manifest.json"))
    os.makedirs(OUT, exist_ok=True)
    for li in range(lo, hi + 1):
        f = f"{OUT}/w{li}.npy"
        if os.path.exists(f):
            continue
        ent = m["layers"][str(li)]
        w = glob(li) if ent["attn_type"] == "full" else sliding(li, ent["matrices"])
        np.save(f + ".tmp.npy", w)
        os.replace(f + ".tmp.npy", f)
        print(li, ent["attn_type"], w.size, flush=True)


if __name__ == "__main__":
    main()
