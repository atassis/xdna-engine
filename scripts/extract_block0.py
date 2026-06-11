#!/usr/bin/env python3
"""Extract GigaAM-v3 block-0 weights + ONNX reference intermediates.

Runs in .venv (onnx/onnxruntime). NPU runners use .venv-iron (pyxrt, no onnx),
so everything is dumped as .npy into artifacts/ with a manifest.json.

Outputs:
  artifacts/weights/<key>.npy        all block-0 trainable tensors (fp32)
  artifacts/refs/<key>.npy           ONNX intermediate tensors on a fixed input
  artifacts/manifest.json            shapes + roles + tensor-name mapping
  artifacts/block0_dwconv_weight.npy (kept for the standalone dwconv test)
"""
import os, json
import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort

ONNX = "models/gigaam_v3_encoder_static.onnx"
OUT = "artifacts"
SEED = 0

# logical name -> ONNX node name (output[0] of that node is the tensor we capture)
REF_NODES = {
    "block_in":      "/pre_encode/Transpose",
    "ffn1_ln":       "/layers.0/norm_feed_forward1/LayerNormalization",
    "ffn1_l1":       "/layers.0/feed_forward1/linear1/Add",
    "ffn1_swish":    "/layers.0/feed_forward1/activation/Mul",
    "ffn1_l2":       "/layers.0/feed_forward1/linear2/Add",
    "after_ffn1":    "/layers.0/Add",
    "att_ln":        "/layers.0/norm_self_att/LayerNormalization",
    "pos_cos":       "/pos_enc/Slice",
    "pos_sin":       "/pos_enc/Slice_1",
    "att_reshape":   "/layers.0/self_attn/Reshape",
    "att_rope":      "/layers.0/self_attn/Add",
    "qk_in":         "/layers.0/self_attn/Transpose_1",
    "v_in":          "/layers.0/self_attn/Transpose_2",
    "scores":        "/layers.0/self_attn/MatMul",
    "q":             "/layers.0/self_attn/linear_q/Add",
    "k":             "/layers.0/self_attn/linear_k/Add",
    "v":             "/layers.0/self_attn/linear_v/Add",
    "attn_probs":    "/layers.0/self_attn/Softmax",
    "attn_ctx":      "/layers.0/self_attn/MatMul_1",
    "attn_out":      "/layers.0/self_attn/linear_out/Add",
    "after_mhsa":    "/layers.0/Add_1",
    "conv_ln":       "/layers.0/norm_conv/LayerNormalization",
    "conv_pw1":      "/layers.0/conv/pointwise_conv1/Conv",
    "conv_glu":      "/layers.0/conv/Mul",
    "conv_dw":       "/layers.0/conv/depthwise_conv/Conv",
    "conv_bn":       "/layers.0/conv/batch_norm/LayerNormalization",
    "conv_swish":    "/layers.0/conv/activation/Mul",
    "conv_pw2":      "/layers.0/conv/pointwise_conv2/Conv",
    "after_conv":    "/layers.0/Add_2",
    "ffn2_l2":       "/layers.0/feed_forward2/linear2/Add",
    "after_ffn2":    "/layers.0/Add_3",
    "block_out":     "/layers.0/norm_out/LayerNormalization",
}


def main():
    for d in (OUT, f"{OUT}/weights", f"{OUT}/refs"):
        os.makedirs(d, exist_ok=True)
    m = onnx.load(ONNX, load_external_data=True)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    by_name = {n.name: n for n in g.node}
    n0 = [n for n in g.node if "/layers.0/" in n.name]

    manifest = {"weights": {}, "refs": {}, "input": {}}

    # ---- weights: walk block-0 nodes, dump each weight/bias initializer ----
    def dump_w(key, init_name):
        arr = numpy_helper.to_array(inits[init_name]).astype(np.float32)
        np.save(f"{OUT}/weights/{key}.npy", arr)
        manifest["weights"][key] = {"src": init_name, "shape": list(arr.shape)}
        return arr

    for n in n0:
        short = n.name.split("/layers.0/")[1].rsplit("/", 1)[0].replace("/", ".")
        if n.op_type in ("MatMul", "Conv", "LayerNormalization"):
            wins = [i for i in n.input if i in inits]
            for wi in wins:
                role = "bias" if wi.endswith(".bias") else "weight"
                dump_w(f"{short}.{role}", wi)
        if n.op_type == "MatMul":
            # bias is the Add node that consumes this MatMul's output
            mout = n.output[0]
            for a in n0:
                if a.op_type == "Add" and mout in a.input:
                    bi = [i for i in a.input if i in inits and i.endswith(".bias")]
                    if bi:
                        dump_w(f"{short}.bias", bi[0])

    # keep the standalone dwconv weight file path stable
    np.save(f"{OUT}/block0_dwconv_weight.npy",
            numpy_helper.to_array(inits["layers.0.conv.depthwise_conv.weight"])
            .reshape(768, 5).astype(np.float32))

    # ---- references: expose intermediates, run ORT on a fixed input ----
    ref_tensor_names = {}
    for key, node_name in REF_NODES.items():
        if node_name not in by_name:
            print(f"  WARN: node {node_name} not found, skipping {key}")
            continue
        ref_tensor_names[key] = by_name[node_name].output[0]

    existing_out = {o.name for o in g.output}
    for key, tname in ref_tensor_names.items():
        if tname not in existing_out:
            g.output.append(helper.make_empty_tensor_value_info(tname))

    rng = np.random.RandomState(SEED)
    feed = {}
    for inp in g.input:
        nm = inp.name
        et = inp.type.tensor_type.elem_type
        shp = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if et == 1:  # float32 -> audio_signal mel features
            x = rng.standard_normal(shp).astype(np.float32)
            feed[nm] = x
            np.save(f"{OUT}/refs/encoder_input.npy", x)
            manifest["input"][nm] = {"shape": shp, "dtype": "float32", "seed": SEED}
        elif et == 7:  # int64 -> valid length (full time dim = 1600)
            feed[nm] = np.array([1600], dtype=np.int64)
            manifest["input"][nm] = {"shape": shp, "dtype": "int64", "value": 1600}

    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    wanted = list(ref_tensor_names.values())
    outs = sess.run(wanted, feed)
    for key, arr in zip(ref_tensor_names.keys(), outs):
        a = np.asarray(arr)
        np.save(f"{OUT}/refs/{key}.npy", a)
        manifest["refs"][key] = {"tensor": ref_tensor_names[key], "shape": list(a.shape)}

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"weights: {len(manifest['weights'])}  refs: {len(manifest['refs'])}")
    print("ref shapes:", {k: v["shape"] for k, v in manifest["refs"].items()})


if __name__ == "__main__":
    main()
