#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 step-4/5: Whisper FFN (GELU MLP) sub-block as a fused full ELF, on real whisper-small weights.

Decode token (M=1) FFN: x -> ln_final -> fc1 -> +bias -> GELU(tanh) -> fc2 -> +bias.
    x_norm = LayerNorm(x)                              # IRON layer_norm (non-affine)
    h      = (γf ⊙ x_norm + βf) @ Wfc1 + b_fc1         # affine folded into fc1; bias ON-DEVICE (feeds GELU)
    h      = gelu_tanh(h)                              # IRON gelu (approximate="tanh")
    out    = h @ Wfc2 + b_fc2

All bias adds are on-device `elementwise_add` (fc1 bias feeds the GELU nonlinearity, so it cannot be
deferred to host). γf folds into Wfc1 (W''=diag(γf)·W); βf into bias' = βf@Wfc1 + b_fc1. fc2 has no
preceding norm (no fold). Gate (generic fused_elf_probe): rel-L2(device out, buffers/out.bin) <= 0.08,
golden = the same bf16 dataflow the device runs.

Run inside IRON env (aiebu-asm on PATH). See gen_ln_qkv.py for invocation.
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes
import torch

from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemv.op import GEMV
from iron.operators.layer_norm.op import LayerNorm
from iron.operators.gelu.op import GELU
from iron.operators.elementwise_add.op import ElementwiseAdd

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
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
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

    # fold γf into fc1; βf into bias'. IRON gemv matrix is [M, K] = Wᵀ.
    mat_fc1 = bf16((gf[:, None] * Wfc1).T.copy())     # [FF, D]
    bias_fc1 = bf16(bf @ Wfc1 + b_fc1)                # [FF]
    mat_fc2 = bf16(Wfc2.T.copy())                     # [D, FF]
    b_fc2_bf = bf16(b_fc2)                            # [D]

    ctx = AIEContext()
    ln_op = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    g1 = GEMV(M=FF, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=FF // 8, context=ctx)
    add1 = ElementwiseAdd(size=FF, tile_size=FF // 8, num_aie_columns=8, context=ctx)
    gelu = GELU(size=FF, num_aie_columns=8, num_channels=1, tile_size=FF // 8, context=ctx)
    g2 = GEMV(M=D, K=FF, num_aie_columns=8, tile_size_input=4, tile_size_output=D // 8, context=ctx)
    add2 = ElementwiseAdd(size=D, tile_size=D // 8, num_aie_columns=8, context=ctx)

    runlist = [
        (ln_op, "x", "x_norm"),
        (g1, "Wfc1", "x_norm", "h"),
        (add1, "h", "bfc1", "h"),       # on-device bias (in-place); feeds GELU
        (gelu, "h", "h"),
        (g2, "Wfc2", "h", "out"),
        (add2, "out", "bfc2", "out"),
    ]
    fused = FusedMLIROperator("ffn", runlist, input_args=["x"], output_args=["out"], context=ctx)
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    names = ("x", "out", "Wfc1", "bfc1", "Wfc2", "bfc2", "x_norm", "h")
    lay = {n: fused.get_layout_for_buffer(n) for n in names}
    print("buffer_sizes (in,out,scratch) =", fused.buffer_sizes)
    for n, v in lay.items():
        print(f"  {n}: type={v[0]} off={int(v[1])} len={int(v[2])}")

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
    with open(os.path.join(args.out, "ffn.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "ffn.elf",
        "kernel_name": "main:sequence",
        "input_size": int(in_sz),
        "output_size": int(out_sz),
        "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"],
        "weights": ["Wfc1", "bfc1", "Wfc2", "bfc2"],
        "output": "out",
        "dims": {"D": D, "FF": FF, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote ELF ({len(elf_bytes)}B) + buffers + meta.json to {args.out}")


if __name__ == "__main__":
    main()
