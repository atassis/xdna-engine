#!/usr/bin/env python3
"""Phase 1 — extract the Parakeet-tdt-0.6b-v3 FastConformer encoder (24 blocks) +
pre_encode (÷8 conv2d subsample) weights + ONNX reference activations into
artifacts/parakeet/encoder/, mirroring the GigaAM layout (scripts/extract_encoder.py)
so the Rust loader (weights.rs) can consume it the same way.

Differences from GigaAM that this handles:
  - 24 blocks, d_model 1024, d_ff 4096, 8 heads x 128.
  - Linear weights are ANONYMISED in the ONNX (onnx::MatMul_*), so we key off the
    NODE name path (e.g. /layers.0/self_attn/linear_q/MatMul -> self_attn.linear_q.weight).
  - NO matmul biases (NeMo FastConformer linears are bias-free; verified).
  - rel_pos attention: dump self_attn.pos_bias_u / pos_bias_v ([8,128]) per block;
    capture the /pos_enc/Slice output (the sinusoidal rel-pos table) as a ref.
  - depthwise conv is k=9 ([C,1,9]); pre_encode is a conv2D dw-striding ÷8 stack
    (conv.0/2/3/5/6, [256,1,3,3]/[256,256,1,1]) + a final pre_encode.out MatMul [4096,1024].

Matmul weights are stored VERBATIM as ONNX [K_in, N_out] (the x@W convention the Rust
matmul expects); pointwise conv weights stay [out,in,1] (Rust squeeze_t transposes them);
all tensors fp32 on disk (bf16 happens at runtime). Run in ~/npuvox-asr-bench/.venv.

Usage: ~/npuvox-asr-bench/.venv/bin/python scripts/extract_parakeet_encoder.py
"""
import os, json, re
import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# real-copy of the cached fp32 encoder (external-data symlink deref'd into models/parakeet/)
ONNX = os.path.join(REPO, "models", "parakeet", "encoder-model.onnx")
OUT = os.path.join(REPO, "artifacts", "parakeet", "encoder")
NB = 24
T_MEL = 256           # seeded-random mel frames for the ref forward pass (->32 after ÷8)
SEED = 0


def main():
    os.makedirs(f"{OUT}/refs", exist_ok=True)
    os.makedirs(f"{OUT}/pre_encode", exist_ok=True)
    for b in range(NB):
        os.makedirs(f"{OUT}/L{b}", exist_ok=True)

    m = onnx.load(ONNX, load_external_data=True)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    by_name = {n.name: n for n in g.node}
    man = {"nblocks": NB, "d_model": 1024, "d_ff": 4096, "n_heads": 8, "head_dim": 128,
           "blocks": {}, "pre_encode": [], "refs": {}}

    def save(path, arr):
        np.save(path, np.asarray(arr).astype(np.float32))

    def short_of(nodename, pfx):
        # /layers.0/self_attn/linear_q/MatMul -> self_attn.linear_q
        return nodename.split(pfx, 1)[1].rsplit("/", 1)[0].replace("/", ".")

    # ---------- per-block weights ----------
    for blk in range(NB):
        pfx = f"/layers.{blk}/"
        keys = []
        for n in g.node:
            if pfx not in n.name:
                continue
            short = short_of(n.name, pfx)
            if n.op_type == "LayerNormalization":
                # inputs: [x, scale, bias]
                save(f"{OUT}/L{blk}/{short}.weight.npy", numpy_helper.to_array(inits[n.input[1]]))
                save(f"{OUT}/L{blk}/{short}.bias.npy",   numpy_helper.to_array(inits[n.input[2]]))
                keys += [f"{short}.weight", f"{short}.bias"]
            elif n.op_type == "MatMul":
                wi = [i for i in n.input if i in inits]
                if wi:  # linear weight (anonymised onnx::MatMul_*); bias-free
                    save(f"{OUT}/L{blk}/{short}.weight.npy", numpy_helper.to_array(inits[wi[0]]))
                    keys.append(f"{short}.weight")
            elif n.op_type == "Conv":
                # inputs: [x, weight, (bias)]
                save(f"{OUT}/L{blk}/{short}.weight.npy", numpy_helper.to_array(inits[n.input[1]]))
                keys.append(f"{short}.weight")
                if len(n.input) > 2 and n.input[2] in inits:
                    save(f"{OUT}/L{blk}/{short}.bias.npy", numpy_helper.to_array(inits[n.input[2]]))
                    keys.append(f"{short}.bias")
        # rel-pos biases (named initialisers, not attached to a weight-bearing op)
        for pb in ("pos_bias_u", "pos_bias_v"):
            key = f"layers.{blk}.self_attn.{pb}"
            save(f"{OUT}/L{blk}/self_attn.{pb}.npy", numpy_helper.to_array(inits[key]))
            keys.append(f"self_attn.{pb}")
        man["blocks"][blk] = sorted(set(keys))

    # ---------- pre_encode (÷8 conv2d dw-striding subsample) ----------
    for n in g.node:
        if "/pre_encode/" not in n.name:
            continue
        if n.op_type == "Conv":
            short = short_of(n.name, "/pre_encode/conv/")  # conv.0 / conv.2 / ...
            save(f"{OUT}/pre_encode/{short}.weight.npy", numpy_helper.to_array(inits[n.input[1]]))
            man["pre_encode"].append(f"{short}.weight")
            if len(n.input) > 2 and n.input[2] in inits:
                save(f"{OUT}/pre_encode/{short}.bias.npy", numpy_helper.to_array(inits[n.input[2]]))
                man["pre_encode"].append(f"{short}.bias")
        elif n.op_type == "MatMul":
            wi = [i for i in n.input if i in inits]
            if wi:
                save(f"{OUT}/pre_encode/out.weight.npy", numpy_helper.to_array(inits[wi[0]]))
                man["pre_encode"].append("out.weight")
    if "pre_encode.out.bias" in inits:
        save(f"{OUT}/pre_encode/out.bias.npy", numpy_helper.to_array(inits["pre_encode.out.bias"]))
        man["pre_encode"].append("out.bias")

    # ---------- reference activations (seeded forward) ----------
    # The model is >2GB so SerializeToString() overflows protobuf's 2GB cap. Instead,
    # reload the graph WITHOUT pulling external data inline, splice in the extra outputs,
    # and write a small temp .onnx beside encoder-model.onnx.data so its initializers stay
    # EXTERNAL (the temp graph is ~41MB); ORT then resolves weights from the .data file.
    mref = onnx.load(ONNX, load_external_data=False)
    gref = mref.graph
    bn = {n.name: n for n in gref.node}
    ln0 = bn["/layers.0/norm_feed_forward1/LayerNormalization"]
    refs = {
        "block_in": ln0.input[0],                        # encoder block-stack input (post pre_encode)
        "pos_enc":  bn["/pos_enc/Slice"].output[0],      # sinusoidal rel-pos table fed to linear_pos
    }
    for b in range(NB):
        refs[f"out_L{b}"] = bn[f"/layers.{b}/norm_out/LayerNormalization"].output[0]
    refs["encoded"] = gref.output[0].name                # "outputs"

    have = {o.name for o in gref.output}
    for t in refs.values():
        if t not in have:
            gref.output.append(helper.make_empty_tensor_value_info(t))

    tmp = os.path.join(os.path.dirname(ONNX), "_ref_graph.onnx")
    onnx.save(mref, tmp)  # initializers stay external -> small file referencing encoder-model.onnx.data

    rng = np.random.RandomState(SEED)
    feed = {}
    for inp in gref.input:
        et = inp.type.tensor_type.elem_type
        if et == 1:  # float audio_signal [1,128,T]
            x = rng.standard_normal((1, 128, T_MEL)).astype(np.float32)
            feed[inp.name] = x
            save(f"{OUT}/refs/audio_signal.npy", x)
        else:        # int64 length
            feed[inp.name] = np.array([T_MEL], dtype=np.int64)
            save(f"{OUT}/refs/length.npy", feed[inp.name])

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    outs = sess.run(list(refs.values()), feed)
    os.remove(tmp)
    for k, arr in zip(refs.keys(), outs):
        a = np.asarray(arr)
        save(f"{OUT}/refs/{k}.npy", a)
        man["refs"][k] = list(a.shape)

    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=2)
    print(f"OK  blocks={NB}  weights/block={len(man['blocks'][0])}  pre_encode={len(man['pre_encode'])}")
    print("block-0 keys:", man["blocks"][0])
    print("pre_encode  :", sorted(set(man["pre_encode"])))
    print("ref shapes  :", {k: man["refs"][k] for k in ("block_in", "pos_enc", "out_L0", "out_L23", "encoded")})


if __name__ == "__main__":
    main()
