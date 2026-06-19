#!/usr/bin/env python3
"""Standalone proj_out (lm-head) GEMV ELF — the e2e/NPU wide-dispatch path.

Builds ONE constant resident ELF that computes the per-token logits projection on the NPU in a SINGLE
dispatch: `logits[VOCAB_PAD] = proj_out_w · ln_post(hidden)`. This replaces the 17-chunk ctx2 path
(`NPU_DECODE_PROJOUT_CTX2`, latency-negative — 17 host round-trips) with one GEMV dispatch.

WHY a GEMV (not the whole_array ctx2 GEMM): the whole_array form tiles the output `[M,N]` into 256-row
blocks, so its M-row-block DMA stride is `256·N` which blows the AIE shim DMA's 2^20 stride limit at
N>~3840 (this is the root cause of NA=3072). The GEMV lays proj_out out as VOCAB-as-M -> the output is a
contiguous `[VOCAB_PAD]` VECTOR (no 2D output strides -> no stride wall), and it builds at large M. As a
STANDALONE ELF its arena is just the weight + tiny I/O (~80 MB), avoiding the 410 MB decode-scratch wall
the inline attempt hit. See `internal notes` (ADDENDUM).

The LN affine-normalize stays on host (cheap). The `β·W` bias is folded INTO the GEMV via K-augmentation
(`K_aug = D + VS`): an extra weight column holds `bias'[m] = (β·W)[m]` (and `−1e30` for the pad rows so
padding never wins argmax), the input gets a constant `1` at index D — so the GEMV emits the COMPLETE logits
`norm·(γ⊙W) + β·W` on-device. This is the prerequisite for on-NPU argmax (step-2: argmax must run over biased
logits). The ELF weight is `mat[VOCAB_PAD, K_aug]` = `[ (γ[:,None]·proj_out_w).T | bias' | 0… ]`.

  logits = (norm·γ + β)·W = norm·(γ⊙W) + (β·W)·1   (the `·1` term is the K-aug column)

Usage (mirror build_deepc_decode.sh env): python gen_projout.py --weights <dir> --out <dir>
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 — MUST precede iron imports (new-mlir-aie port shim)
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemv.op import GEMV

BF16 = ml_dtypes.bfloat16
D = 768
VS = 64  # GEMV kernel_vector_size — K must be a multiple; K-aug adds exactly one VS block for the bias
K_AUG = D + VS  # 832 (%64 ✓): cols 0:D = γ⊙W, col D = bias', cols D+1:K_AUG = 0; input has a 1 at index D
VOCAB, VOCAB_PAD = 51865, 52224  # 52224 = 8*6528 (%8 GEMV M cols), 6528 = 408*16 (per-col tile %16)


def bf16(a):
    return np.asarray(a, np.float32).astype(BF16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--argmax", action="store_true",
                    help="fuse per-column partial argmax (step-2): ELF also outputs [cols] local idx + vals")
    a = ap.parse_args()
    w = a.weights
    os.makedirs(os.path.join(a.out, "buffers"), exist_ok=True)

    # weight fold (K-aug): mat[VOCAB_PAD, K_AUG] = [ (γ⊙W).T | bias' | 0… ].
    #   cols 0:D   = (γ_post ⊙ proj_out_w).T   (the γ-folded projection, pad rows = 0)
    #   col  D     = bias'[m] = (β_post·proj_out_w)[m] for m<VOCAB; -1e30 for pad rows (never win argmax)
    #   cols D+1:K = 0   (rest of the VS block)
    g_post = np.load(os.path.join(w, "ln_post.weight.npy")).astype(np.float32)          # [D]
    b_post = np.load(os.path.join(w, "ln_post.bias.npy")).astype(np.float32)            # [D]
    Wproj = np.load(os.path.join(w, "proj_out.weight.npy")).astype(np.float32)           # [D, VOCAB]
    mat_proj = (g_post[:, None] * Wproj).T.copy()                                        # [VOCAB, D]
    bias_proj = (b_post @ Wproj).astype(np.float32)                                      # [VOCAB]
    mat_pad = np.zeros((VOCAB_PAD, K_AUG), np.float32)
    mat_pad[0:VOCAB, 0:D] = mat_proj
    mat_pad[0:VOCAB, D] = bias_proj
    mat_pad[VOCAB:VOCAB_PAD, D] = -1e30                                                  # pad rows: huge -bias

    COLS = 8
    ctx = AIEContext()
    op_proj_out = GEMV(M=VOCAB_PAD, K=K_AUG, num_aie_columns=COLS, tile_size_input=4,
                       tile_size_output=VOCAB_PAD // COLS, context=ctx)
    rl = [(op_proj_out, "Wproj", "x", "logits")]
    bufsz = {"x": K_AUG * 2, "logits": VOCAB_PAD * 2}
    # step-2: fuse per-column partial argmax. The ELF ALSO outputs amax_idx[COLS] (i32, local index per
    # column) + amax_val[COLS] (f32, max value) — the host does the trivial COLS-way reduce → token id, and
    # reads the full `logits` only at step 0 (pick_lang carve-out). logits stays a final output for that.
    out_args = ["logits"]
    if a.argmax:
        from argmax_op import Argmax  # local op (route_b_kernels/decode_fused) — no shared-IRON edit
        op_argmax = Argmax(N=VOCAB_PAD, cols=COLS, context=ctx)
        rl.append((op_argmax, "logits", "amax"))
        bufsz["amax"] = COLS * 8  # COLS × [val:f32 | idx:i32] = COLS×8 bytes (bf16-typed: COLS×4 elems)
        out_args = ["amax", "logits"]

    fused = FusedMLIROperator("projout", rl, input_args=["x"], output_args=out_args,
                              buffer_sizes=bufsz, context=ctx)
    fused.compile()
    elf = load_elf(fused).view(np.uint8).tobytes()
    in_sz, out_sz, scr = fused.buffer_sizes
    lay = {n: fused.get_layout_for_buffer(n) for n in (["x", "logits", "Wproj"] + (["amax"] if a.argmax else []))}

    bdir = os.path.join(a.out, "buffers")

    def wb(n, v):
        open(os.path.join(bdir, f"{n}.bin"), "wb").write(np.asarray(v, BF16).tobytes())

    wb("Wproj", bf16(mat_pad).reshape(-1))
    # x placeholder = the K-aug input tail [0…0, 1@D, 0…0]: the engine writes nrm into [0:D] per token, and
    # the constant 1 at index D (+ zeros) persists (so GEMV adds bias'·1). Written here as the initial buffer.
    x0 = np.zeros(K_AUG, np.float32)
    x0[D] = 1.0
    wb("x", bf16(x0))

    open(os.path.join(a.out, "projout.elf"), "wb").write(elf)
    meta = {
        "elf": "projout.elf", "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])} for n, v in lay.items()},
        "inputs": ["x"], "weights": ["Wproj"], "output": "logits",
        "vocab": VOCAB, "vocab_pad": VOCAB_PAD, "D": D, "k_aug": K_AUG, "vs": VS,
        "argmax": bool(a.argmax), "cols": COLS,
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"wrote proj_out GEMV ELF ({len(elf)}B, scratch {scr/1e6:.1f}MB, weight {mat_pad.nbytes//2/1e6:.0f}MB bf16) to {a.out}")


if __name__ == "__main__":
    main()
