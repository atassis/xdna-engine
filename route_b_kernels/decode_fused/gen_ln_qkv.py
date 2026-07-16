#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 step-4/5: a REAL Whisper sub-block as a fused full ELF — LayerNorm(+affine) → QKV projection.

Decode token (M=1) self-attention front half on real whisper-small layer weights:
    x_norm   = LayerNorm(x)                       # IRON layer_norm op (non-affine, torch semantics)
    qkv      = (gamma ⊙ x_norm + beta) @ W_qkv + b # affine folded into the GEMV: W'' = diag(gamma)·W,
             = x_norm @ W'' + bias'                #   bias' = beta@W + b  (added host-side after read)

IRON's layer_norm is non-affine, so Whisper's affine LN folds into the projection exactly (the same
host pre-norm fold as [[decode-norm-gemv]]). IRON gemv computes matrix[M,K] @ vec[K] = out[M], so the
device matrix is W''ᵀ with M=2304 (q|k|v concat), K=768.

On-device gate (generic `fused_elf_probe`): rel-L2(device qkv_nobias, buffers/qkv.bin) <= 0.08, where
the golden = matrix_bf16 @ bf16(LN(x)). The affine-fold correctness (golden + bias' ≈ true Whisper qkv)
is asserted here at generation time.

Run inside IRON env (aiebu-asm on PATH):
    cd ~/repositories/ns/atassis/xdna-engine-workspace/amd/IRON && source ironenv/bin/activate
    python <this> --weights <wt>/artifacts/whisper-small/whisper_decoder --layer 0 --out <wt>/artifacts/fused_ln_qkv
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

BF16 = ml_dtypes.bfloat16
D = 768
QKV = 2304  # q|k|v = 3*768


def bf16(a):
    return np.asarray(a).astype(BF16)


def npy(wdir, layer, name):
    return np.load(os.path.join(wdir, f"L{layer}", f"{name}.npy"))


def layernorm_torch(x_f32):
    """Match IRON layer_norm == torch.nn.functional.layer_norm (eps 1e-5, population var, non-affine)."""
    t = torch.from_numpy(x_f32.astype(np.float32)).reshape(1, -1)
    return torch.nn.functional.layer_norm(t, normalized_shape=(t.shape[-1],)).numpy().reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="whisper_decoder weights dir (has L0..L11)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "buffers"), exist_ok=True)
    L = args.layer

    # --- real weights (stored [K_in=768, N_out=768], x@W convention) ---
    Wq, Wk, Wv = (npy(args.weights, L, n) for n in ("q.weight", "k.weight", "v.weight"))
    bq, bk, bv = (npy(args.weights, L, n) for n in ("q.bias", "k.bias", "v.bias"))
    gamma = npy(args.weights, L, "ln_self.weight").astype(np.float32)  # [768]
    beta = npy(args.weights, L, "ln_self.bias").astype(np.float32)     # [768]
    W_qkv = np.concatenate([Wq, Wk, Wv], axis=1).astype(np.float32)    # [768, 2304]
    b_qkv = np.concatenate([bq, bk, bv]).astype(np.float32)            # [2304]
    assert W_qkv.shape == (D, QKV) and b_qkv.shape == (QKV,)

    # --- fold affine into the GEMV ---
    # qkv[j] = sum_i (gamma[i]*n[i]) W[i,j] + bias'[j];  matrix[j,i] = gamma[i]*W[i,j] = (gamma[:,None]*W).T
    matrix = (gamma[:, None] * W_qkv).T.copy()      # [2304, 768]  (M=QKV, K=D)
    bias_p = beta @ W_qkv + b_qkv                   # [2304]
    matrix_bf = bf16(matrix)

    # --- build the fused op ---
    ctx = AIEContext()
    ln = LayerNorm(size=D, num_aie_columns=1, num_channels=1, tile_size=D, context=ctx)
    gemv = GEMV(M=QKV, K=D, num_aie_columns=8, tile_size_input=4, tile_size_output=QKV // 8, context=ctx)
    runlist = [
        (ln, "x", "x_norm"),
        (gemv, "Wqkv", "x_norm", "qkv"),
    ]
    fused = FusedMLIROperator("ln_qkv", runlist, input_args=["x"], output_args=["qkv"], context=ctx)
    fused.compile()

    elf_bytes = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes
    lay = {n: fused.get_layout_for_buffer(n) for n in ("x", "qkv", "Wqkv", "x_norm")}
    print("buffer_sizes (in,out,scratch) =", fused.buffer_sizes)
    for n, v in lay.items():
        print(f"  {n}: type={v[0]} off={int(v[1])} len={int(v[2])}")

    # --- inputs + device golden ---
    rng = np.random.default_rng(args.seed)
    x = bf16(rng.standard_normal(D).astype(np.float32) * 1.0)
    n_hw = bf16(layernorm_torch(x.astype(np.float32)))            # what the LN op outputs (bf16)
    qkv_dev = bf16(matrix_bf.astype(np.float32) @ n_hw.astype(np.float32))  # device GEMV (no bias)

    # --- affine-fold correctness check (generation-time, f32) ---
    n_ref = layernorm_torch(x.astype(np.float32))
    qkv_true = (gamma * n_ref + beta) @ W_qkv + b_qkv             # true Whisper qkv
    qkv_via_fold = qkv_dev.astype(np.float32) + bias_p            # device + host bias'
    rel = np.linalg.norm(qkv_via_fold - qkv_true) / (np.linalg.norm(qkv_true) + 1e-9)
    print(f"\n[fold check] rel-L2(device+bias', true Whisper qkv) = {rel:.5f}  (bf16-expected ~1e-2)")
    assert rel < 0.05, f"affine fold wrong: rel {rel}"

    # --- write buffers (bf16) + bias' (f32) + meta ---
    def wbuf(name, vals, dtype=BF16):
        with open(os.path.join(args.out, "buffers", f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=dtype).tobytes())

    wbuf("x", x)
    wbuf("Wqkv", matrix_bf.reshape(-1))
    wbuf("qkv", qkv_dev)                 # device golden (no bias)
    wbuf("bias_p", bias_p, dtype=np.float32)  # host epilogue bias' (f32)
    with open(os.path.join(args.out, "ln_qkv.elf"), "wb") as f:
        f.write(elf_bytes)

    meta = {
        "elf": "ln_qkv.elf",
        "kernel_name": "main:sequence",
        "input_size": int(in_sz),
        "output_size": int(out_sz),
        "scratch_size": int(scratch_sz),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"],
        "weights": ["Wqkv"],
        "output": "qkv",
        "dims": {"D": D, "QKV": QKV, "layer": L},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote ELF ({len(elf_bytes)}B) + buffers + meta.json to {args.out}")


if __name__ == "__main__":
    main()
