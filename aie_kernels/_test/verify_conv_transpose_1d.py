#!/usr/bin/env python3
"""Device rel-L2 verify for the conv-transpose-1d brick. Gate 3e-2. Run under the device lock.

ONE input buffer, ONE output buffer, both 64 floats -- the only configuration measured to transfer
intact on this harness:
  * 3 separate inputs is rejected outright (a core tile has 2 input / 2 output DMA channels).
  * With 2 inputs, in0 arrives as ZEROS and in1 arrives with a single element replicated
    (probe_ct1d_inputs.py). So the multi-input oneshot path does not deliver.
  * An 8-float (32 B) buffer arrives as zeros even when it is the only input.
So x, w and bias are packed into one 64-float buffer and sliced in the shim. verify_rmsnorm packs
gamma|beta the same way for the same DMA reason; this just goes one step further.

stride 4 exercises the misalignment every codec rate ([8,8,4,2]) has.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "conv-transpose-1d").resolve()

C_IN, C_OUT, K, T, STRIDE = 1, 2, 8, 4, 4
OUT_LEN = (T - 1) * STRIDE + K                 # 20
X_OFF, W_OFF = 0, C_IN * T                     # 0, 4
B_OFF = W_OFF + C_IN * C_OUT * K               # 20
PACK = 64
GATE = 3e-2

spec = importlib.util.spec_from_file_location("ct1d_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
x = rng.standard_normal((C_IN, T)).astype(np.float32)
w = (rng.standard_normal((C_IN, C_OUT, K)).astype(np.float32) * 0.5)
b = (rng.standard_normal(C_OUT).astype(np.float32) * 0.1)
ref = golden.conv_transpose_1d_ref(x, w, b, STRIDE)

packed = np.zeros(PACK, np.float32)
packed[X_OFF:X_OFF + x.size] = x.ravel()
packed[W_OFF:W_OFF + w.size] = w.ravel()
packed[B_OFF:B_OFF + b.size] = b

_cb = int(time.time() * 1000) % 10**9
res = bricklib.verify_oneshot(
    name="conv_transpose_1d",
    brick_cc=BRICK / "conv_transpose_1d.cc",
    shim_body=(
        f"// cachebust {_cb}\n"
        'extern "C" void ct1d_verify(float *p, float *out) {\n'
        f"  conv_transpose_1d_f32(p + {X_OFF}, p + {W_OFF}, p + {B_OFF}, out,\n"
        f"                        {C_IN}, {C_OUT}, {K}, {T}, {STRIDE}, {OUT_LEN});\n"
        "}\n"
    ),
    symbol="ct1d_verify",
    inputs=[(packed, np.float32)],
    out_numel=PACK,
    out_shape=(PACK,),
    unpack=lambda d: np.asarray(d)[:C_OUT * OUT_LEN].reshape(C_OUT, OUT_LEN),
    golden=ref,
    gate=GATE,
    out_dt=np.float32,
)
got = np.asarray(res["got"], np.float32)
print(f"  device vs ref rel-L2: {res['rel_l2']:.3e}")
print(f"  max abs err         : {float(np.max(np.abs(got - ref))):.3e}")
assert res["ok"], f"conv_transpose_1d device gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")

# ADDITIVE: conv_transpose_channel_core_vec, the phase-decomposed vectorised PER-CHANNEL core (see
# conv_transpose_1d.cc's own header for why the scatter is deleted rather than aligned). Gated via
# bricklib.verify_streamed -- window_driver._conv_transpose_chunk's own streamed-[c_out,c_in*k+1]-
# rows-past-a-resident-[c_in,t]-activation convention, i.e. what the real driver actually
# dispatches -- rather than extending the oneshot harness above, because this file's own header
# records that non-64-float oneshot buffers have arrived as zeros on this device
# (probe_ct1d_inputs.py); verify_streamed sidesteps that rather than fighting it.
#
# Output is PHASE-MAJOR [stride, t] per channel (see the kernel's own comment) -- unpack reshapes
# and transposes it back to the interleaved [t*stride] shape the same golden already returns, a
# pure host reshape, matching the kernel's claim that no on-device store is ever strided.
#
# T=32 (not window_driver.UPSAMPLE_T=16): the vector core's t % 16 == 0 precondition is satisfied
# by both, but T=16 is exactly one N=16 chunk, which never exercises the "aligned pair, non-zero
# shuffle" load branch (read_base >= 0 but not N-aligned) -- only the direct-aligned-load and the
# zero-splice branches. T=32 (conv_1d's own CV_T default) gives a second chunk and hits all three.
#
# Cases cover the codec's two crop conventions and its three distinct strides (stage 1 and 2 share
# stride 8, so testing it once covers both) plus the quantizer's stride=2, k=2, crop_right=0.
CT_VEC_T = 32
CT_VEC_CASES = [  # (c_in, c_out, stride, k, crop_right, label)
    (2, 3, 8, 16, 8, "decoder stride=8 (stage 1/2), k=2*stride, crop_right=stride"),
    (2, 3, 4, 8, 4, "decoder stride=4 (stage 3), k=2*stride, crop_right=stride"),
    (2, 3, 2, 4, 2, "decoder stride=2 (stage 4), k=2*stride, crop_right=stride"),
    (2, 3, 2, 2, 0, "quantizer stride=2, k=stride, crop_right=0"),
]

ok_vec = True
for c_in, c_out, stride, k, crop_right, label in CT_VEC_CASES:
    t = CT_VEC_T
    assert t % 16 == 0, f"{label}: conv_transpose_channel_core_vec requires t % 16 == 0, got t={t}"
    xw = rng.standard_normal((c_in, t)).astype(np.float32)
    ww = (rng.standard_normal((c_in, c_out, k)).astype(np.float32) * 0.5)
    bw = (rng.standard_normal(c_out).astype(np.float32) * 0.1)
    refw = golden.conv_transpose_1d_ref(xw, ww, bw, stride, crop_right=crop_right)
    assert refw.shape == (c_out, t * stride), f"{label}: golden shape {refw.shape} != {(c_out, t * stride)}"

    tile_w = c_in * k
    tiles = np.zeros((c_out, tile_w + 1), np.float32)
    for co in range(c_out):
        tiles[co, :tile_w] = ww[:, co, :].reshape(-1)   # same layout _conv_transpose_chunk streams
        tiles[co, tile_w] = bw[co]
    resident = xw.reshape(-1).astype(np.float32)

    _cb = int(time.time() * 1000) % 10**9
    sym = f"ct1d_vec_verify_s{stride}_k{k}"
    shim = bricklib.GEN / f"ct1d_vec_s{stride}_k{k}_shim.cc"
    shim.write_text(
        f"// AUTO-GENERATED verify shim for the conv-transpose-1d VECTORISED brick, {label}. "
        f"cb {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{BRICK / "conv_transpose_1d.cc"}"\n'
        f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
        f"  route_b_bricks::conv_transpose_channel_core_vec<16>(resident, tile, tile[{tile_w}], "
        f"out,\n"
        f"                                                      {c_in}, {k}, {t}, {stride});\n"
        "}\n"
    )

    resv = bricklib.verify_streamed(
        name=f"conv_transpose_1d_vec_s{stride}_k{k}",
        shim=shim,
        symbol=sym,
        in_tiles=tiles,
        out_tile_numel=stride * t,
        resident=resident,
        unpack=lambda d, c_out=c_out, stride=stride, t=t: (
            np.asarray(d).reshape(c_out, stride, t).transpose(0, 2, 1).reshape(c_out, t * stride)),
        golden=refw,
        gate=GATE,
        in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
    )
    gotw = np.asarray(resv["got"], np.float32)
    print(f"  VEC {label}: rel-L2 {resv['rel_l2']:.3e}  "
          f"max abs err {float(np.max(np.abs(gotw - refw))):.3e}  {resv['status']}")
    ok_vec = ok_vec and resv["ok"]

assert ok_vec, "conv_transpose_1d VECTOR device gate failed"
print("PASS (vector)")
