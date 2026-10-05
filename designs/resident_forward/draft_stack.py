"""Device weight streams for the gemma3-270m draft: `draft_stack.py` (RF_MODEL=gemma3-270m-rail, a venv with
torch). Rebuilds each layer with draft_store and lays it out as stack_prep.sliding does for the 12B:
QKV on the down form, O in the x V workers' K order, the MLP grouped. Writes RF_G3_STACK/w{li}.npy and
the head stream rhead reads (w_head.npy)."""
import os
os.environ.setdefault("RF_MODEL", "gemma3-270m-rail")
import shutil
import sys
import numpy as np
from safetensors.torch import load_file
import draft_store as DS
import o_ref
import rlayer_design as L
import rattn2_design as A2
import rf_paths

OUT = os.environ.get("RF_G3_STACK", str(rf_paths.DATA_ROOT / "artifacts/gemma3-270m/rail_stack"))


def main():
    if not DS.QAT:
        raise ValueError('set RF_G3_QAT to the source checkpoint directory')
    DS.load_head_stream(DS.OUT)
    L.ATTN, L.OPH, L.MPH = True, True, True
    L.LAYER.oa = A2.FINISH_A = True
    L.set_geo(False)
    os.makedirs(OUT, exist_ok=True)
    sd = load_file(f"{DS.QAT}/model.safetensors")
    for li in range(DS.NL):
        W = DS.load_layer(sd, li)
        attn_in, _, mlp = DS.pack_layer(W)
        o_ref.KORDER = "workers"
        a_out = o_ref.stream(W).view(np.uint8).reshape(-1)
        o_ref.KORDER = "heads"
        a_in = L.qkv_downform(attn_in.view(np.uint8).reshape(-1))
        assert a_in.size == L.NC * L.wcol() and a_out.size == L.NC * L.WCOL_O, (a_in.size, a_out.size)
        w = np.concatenate([a_in, a_out, L.mlp_grouped(mlp.view(np.uint8).reshape(-1))])
        assert w.size == L.wbytes(), (w.size, L.wbytes())
        np.save(f"{OUT}/w{li}.tmp.npy", w)
        os.replace(f"{OUT}/w{li}.tmp.npy", f"{OUT}/w{li}.npy")
        print(li, w.size, flush=True)
    shutil.copy(f"{DS.OUT}/w_head.npy", f"{OUT}/w_head.npy")


if __name__ == "__main__":
    sys.exit(main())
