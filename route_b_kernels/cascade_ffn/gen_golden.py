#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""IRON-free numpy/torch golden for the cascade FFN correctness gate.

Reproduces EXACTLY the device-faithful bf16 golden that
`route_b_kernels/decode_fused/gen_ffn.py` emits, but without importing IRON
(the IRON env is version-broken). The golden MATH is pure numpy/torch/ml_dtypes,
so it is reproduced standalone here to unblock the cascade-FFN correctness gate
(Task 4 gate: rel-L2(device out, buffers/out.bin) <= 0.08).

Decode token (M=1) FFN: x -> ln_final -> fc1 -> +bias -> GELU(tanh) -> fc2 -> +bias.
No decode-block residual +x (matches the cascade STRUCTURE.md B.4: fc2 + b_fc2 only).

Buffers written (raw bf16 bytes, 2 bytes/elem) into <out>/buffers/:
    x    (768)         input
    Wfc1 (3072*768)    folded fc1 matrix [FF, D] reshaped flat
    bfc1 (3072)        folded fc1 bias
    Wfc2 (768*3072)    fc2 matrix [D, FF] reshaped flat
    bfc2 (768)         fc2 bias
    out  (768)         o2, the golden output
"""
import argparse
import hashlib
import json
import os

import numpy as np
import ml_dtypes
import torch

BF16 = ml_dtypes.bfloat16
D = 768
FF = 3072


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy"))


def ln(x_f32):
    t = torch.from_numpy(x_f32.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def gelu_tanh(h_f32):
    t = torch.from_numpy(h_f32.astype(np.float32))
    return torch.nn.functional.gelu(t, approximate="tanh").numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="artifacts/whisper-small/whisper_decoder")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", default="artifacts/cascade_ffn/iron_baseline")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L = args.layer

    Wfc1 = npy(args.weights, L, "fc1.weight").astype(np.float32)  # [768, 3072]
    b_fc1 = npy(args.weights, L, "fc1.bias").astype(np.float32)   # [3072]
    Wfc2 = npy(args.weights, L, "fc2.weight").astype(np.float32)  # [3072, 768]
    b_fc2 = npy(args.weights, L, "fc2.bias").astype(np.float32)   # [768]
    gf = npy(args.weights, L, "ln_final.weight").astype(np.float32)  # [768]
    bf = npy(args.weights, L, "ln_final.bias").astype(np.float32)    # [768]
    assert Wfc1.shape == (D, FF) and Wfc2.shape == (FF, D)

    # fold gamma_f into fc1; beta_f into bias'. IRON gemv matrix is [M, K] = W^T.
    mat_fc1 = bf16((gf[:, None] * Wfc1).T.copy())     # [FF, D]
    bias_fc1 = bf16(bf @ Wfc1 + b_fc1)                # [FF]
    mat_fc2 = bf16(Wfc2.T.copy())                     # [D, FF]
    b_fc2_bf = bf16(b_fc2)                            # [D]

    # --- device-faithful golden (bf16 at every stage) ---
    rng = np.random.default_rng(args.seed)
    x = bf16(rng.standard_normal(D).astype(np.float32))
    n_hw = bf16(ln(x.astype(np.float32)))
    h1 = bf16(mat_fc1.astype(np.float32) @ n_hw.astype(np.float32))
    h2 = bf16(h1.astype(np.float32) + bias_fc1.astype(np.float32))
    h3 = bf16(gelu_tanh(h2.astype(np.float32)))
    o1 = bf16(mat_fc2.astype(np.float32) @ h3.astype(np.float32))
    o2 = bf16(o1.astype(np.float32) + b_fc2_bf.astype(np.float32))

    def wbuf(name, vals):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    wbuf("x", x)
    wbuf("Wfc1", mat_fc1.reshape(-1))
    wbuf("bfc1", bias_fc1)
    wbuf("Wfc2", mat_fc2.reshape(-1))
    wbuf("bfc2", b_fc2_bf)
    wbuf("out", o2)

    out_bytes = np.asarray(o2, dtype=BF16).tobytes()
    out_md5 = hashlib.md5(out_bytes).hexdigest()
    meta = {
        "seed": args.seed,
        "dims": {"D": D, "FF": FF, "layer": L},
        "weights_dir": args.weights,
        "buffers": {
            "x": {"elems": D, "bytes": D * 2},
            "Wfc1": {"elems": FF * D, "bytes": FF * D * 2, "shape": [FF, D]},
            "bfc1": {"elems": FF, "bytes": FF * 2},
            "Wfc2": {"elems": D * FF, "bytes": D * FF * 2, "shape": [D, FF]},
            "bfc2": {"elems": D, "bytes": D * 2},
            "out": {"elems": D, "bytes": D * 2},
        },
        "out_md5": out_md5,
        "gate": "rel-L2(device out, buffers/out.bin) <= 0.08",
    }
    with open(os.path.join(args.out, "golden_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote buffers + golden_meta.json to {args.out}")
    print(f"out.bin md5 = {out_md5}")
    print(f"o2[:8] = {o2[:8].astype(np.float32)}")


if __name__ == "__main__":
    main()
