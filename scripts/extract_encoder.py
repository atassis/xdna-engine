#!/usr/bin/env python3
"""Extract the FULL GigaAM-v3 encoder: all 16 blocks' weights + pre_encode
(subsampling) weights + ONNX reference tensors (per-block outputs, pre_encode
conv outputs, final 'encoded'), into artifacts/encoder/.

Runs in .venv (onnx/onnxruntime). The npu_asr package (run in .venv-iron, pyxrt,
no onnx) loads these .npy files. Foundation for the full-encoder verification.
"""
import os, json
import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort

ONNX = "models/gigaam_v3_encoder_static.onnx"
OUT = "artifacts/encoder"
NB = 16
SEED = 0


def main():
    os.makedirs(f"{OUT}/refs", exist_ok=True)
    for b in range(NB):
        os.makedirs(f"{OUT}/L{b}", exist_ok=True)
    os.makedirs(f"{OUT}/pre_encode", exist_ok=True)
    m = onnx.load(ONNX, load_external_data=True)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    by_name = {n.name: n for n in g.node}
    man = {"nblocks": NB, "blocks": {}, "pre_encode": [], "refs": {}}

    def dump(path, name, arr):
        np.save(path, arr.astype(np.float32))

    # ---- per-block weights ----
    for blk in range(NB):
        pfx = f"/layers.{blk}/"
        nb = [n for n in g.node if pfx in n.name]
        keys = []
        for n in nb:
            short = n.name.split(pfx)[1].rsplit("/", 1)[0].replace("/", ".")
            if n.op_type in ("MatMul", "Conv", "LayerNormalization"):
                for wi in [i for i in n.input if i in inits]:
                    role = "bias" if wi.endswith(".bias") else "weight"
                    dump(f"{OUT}/L{blk}/{short}.{role}.npy", f"{short}.{role}",
                         numpy_helper.to_array(inits[wi]))
                    keys.append(f"{short}.{role}")
            if n.op_type == "MatMul":
                mo = n.output[0]
                for a in nb:
                    if a.op_type == "Add" and mo in a.input:
                        bi = [i for i in a.input if i in inits and i.endswith(".bias")]
                        if bi:
                            dump(f"{OUT}/L{blk}/{short}.bias.npy", short,
                                 numpy_helper.to_array(inits[bi[0]]))
                            keys.append(f"{short}.bias")
        man["blocks"][blk] = sorted(set(keys))

    # ---- pre_encode (subsampling) weights ----
    for nm in ("pre_encode.conv.0.weight", "pre_encode.conv.0.bias",
               "pre_encode.conv.2.weight", "pre_encode.conv.2.bias"):
        dump(f"{OUT}/pre_encode/{nm}.npy", nm, numpy_helper.to_array(inits[nm]))
        man["pre_encode"].append(nm)

    # ---- reference tensors ----
    refs = {
        "block_in": by_name["/pre_encode/Transpose"].output[0],
        "pos_cos": by_name["/pos_enc/Slice"].output[0],
        "pos_sin": by_name["/pos_enc/Slice_1"].output[0],
        "pre_conv0": by_name["/pre_encode/conv/conv.0/Conv"].output[0],
        "pre_conv2": by_name["/pre_encode/conv/conv.2/Conv"].output[0],
    }
    for b in range(NB):
        refs[f"out_L{b}"] = by_name[f"/layers.{b}/norm_out/LayerNormalization"].output[0]
    refs["encoded"] = "encoded"  # graph output

    have = {o.name for o in g.output}
    for t in refs.values():
        if t not in have:
            g.output.append(helper.make_empty_tensor_value_info(t))

    rng = np.random.RandomState(SEED)
    feed = {}
    for inp in g.input:
        et = inp.type.tensor_type.elem_type
        shp = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        feed[inp.name] = (rng.standard_normal(shp).astype(np.float32) if et == 1
                          else np.array([1600], dtype=np.int64))
        if et == 1:
            np.save(f"{OUT}/refs/audio_signal.npy", feed[inp.name])

    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(list(refs.values()), feed)
    for k, arr in zip(refs.keys(), outs):
        a = np.asarray(arr)
        np.save(f"{OUT}/refs/{k}.npy", a)
        man["refs"][k] = list(a.shape)

    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=2)
    print(f"blocks={NB} weights/block={len(man['blocks'][0])} pre_encode={len(man['pre_encode'])}")
    print("ref shapes:", {k: man["refs"][k] for k in ("block_in","pre_conv0","pre_conv2","out_L0","out_L15","encoded")})


if __name__ == "__main__":
    main()
