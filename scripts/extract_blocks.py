#!/usr/bin/env python3
"""Extract weights + output refs for blocks 0..N-1 (default 3) to test stacking.

Runs in .venv. Dumps per-block weights to artifacts/stack/L{n}/ and each block's
output tensor (/layers.{n}/norm_out) to artifacts/stack/refs/, plus the block-0
input and shared pos_enc cos/sin. Lets scripts/stack_blocks.py verify the recipe
generalizes beyond block 0 and measure bf16 error accumulation across a stack.
"""
import os, json, sys
import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort

ONNX = "models/gigaam_v3_encoder_static.onnx"
OUT = "artifacts/stack"
NBLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def main():
    os.makedirs(f"{OUT}/refs", exist_ok=True)
    m = onnx.load(ONNX, load_external_data=True)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    by_name = {n.name: n for n in g.node}

    manifest = {"nblocks": NBLOCKS, "blocks": {}}

    for blk in range(NBLOCKS):
        bdir = f"{OUT}/L{blk}"; os.makedirs(bdir, exist_ok=True)
        pfx = f"/layers.{blk}/"
        nb = [n for n in g.node if pfx in n.name]
        keys = []
        for n in nb:
            short = n.name.split(pfx)[1].rsplit("/", 1)[0].replace("/", ".")
            if n.op_type in ("MatMul", "Conv", "LayerNormalization"):
                for wi in [i for i in n.input if i in inits]:
                    role = "bias" if wi.endswith(".bias") else "weight"
                    arr = numpy_helper.to_array(inits[wi]).astype(np.float32)
                    np.save(f"{bdir}/{short}.{role}.npy", arr); keys.append(f"{short}.{role}")
            if n.op_type == "MatMul":
                mout = n.output[0]
                for a in nb:
                    if a.op_type == "Add" and mout in a.input:
                        bi = [i for i in a.input if i in inits and i.endswith(".bias")]
                        if bi:
                            arr = numpy_helper.to_array(inits[bi[0]]).astype(np.float32)
                            np.save(f"{bdir}/{short}.bias.npy", arr); keys.append(f"{short}.bias")
        manifest["blocks"][blk] = sorted(set(keys))

    # reference tensors: block-0 input, shared cos/sin, each block's output
    extra = {
        "block_in": by_name["/pre_encode/Transpose"].output[0],
        "pos_cos": by_name["/pos_enc/Slice"].output[0],
        "pos_sin": by_name["/pos_enc/Slice_1"].output[0],
    }
    for blk in range(NBLOCKS):
        extra[f"out_L{blk}"] = by_name[f"/layers.{blk}/norm_out/LayerNormalization"].output[0]

    have = {o.name for o in g.output}
    for t in extra.values():
        if t not in have:
            g.output.append(helper.make_empty_tensor_value_info(t))

    feed = {}
    rng = np.random.RandomState(0)
    for inp in g.input:
        et = inp.type.tensor_type.elem_type
        shp = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        feed[inp.name] = (rng.standard_normal(shp).astype(np.float32) if et == 1
                          else np.array([1600], dtype=np.int64))

    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(list(extra.values()), feed)
    for k, arr in zip(extra.keys(), outs):
        np.save(f"{OUT}/refs/{k}.npy", np.asarray(arr))

    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
    print(f"extracted {NBLOCKS} blocks; weights/block={len(manifest['blocks'][0])}; "
          f"refs={list(extra.keys())}")


if __name__ == "__main__":
    main()
