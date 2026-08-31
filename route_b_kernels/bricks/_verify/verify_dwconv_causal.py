#!/usr/bin/env python3
"""Device rel-L2 verify: is route_b_kernels/dwconv1d/dwconv1d.cc REUSABLE for the S2 codec
quantizer's ConvNeXt depthwise conv, or does it need a new brick? Gate 3e-2. Run under the device
lock.

ANSWER (settled device-free, see the reasoning below): REUSABLE WITH PARAMETERIZATION. No new
kernel file. This gates the EXISTING, untouched route_b_kernels/dwconv1d/dwconv1d.cc (read-only --
Parakeet's shipped kernel; nothing here edits it) against scripts/codec_quantizer_ref.py's own
dwconv_1d_causal, which is the real oracle-matching reference (not a reimplementation of it).

THE TWO AXES THAT LOOKED LIKE THEY MIGHT BLOCK REUSE, AND WHY THEY DON'T:

1. PADDING CONVENTION (the one that matters -- this is the silent-wrong-answer axis). dwconv1d.cc's
   shipped Parakeet instantiations use 'same' (centered) padding: P=(K-1)/2. S2's ConvNeXt dwconv is
   CAUSAL: shift = K-1-j, i.e. P=K-1, all padding on the left. These are genuinely different ops --
   host-checked below (not asserted from reading the code): building the SAME 'same'-padding formula
   dwconv1d_same_scalar implements with P=(K-1)/2 vs P=K-1 and comparing both to
   codec_quantizer_ref.dwconv_1d_causal on random data (C=5,T=20,K=7,seed=0):
     P=K-1     (causal)  rel-L2 vs dwconv_1d_causal = 3.1e-08  (bit-identical modulo fp order)
     P=(K-1)/2 (centered) rel-L2 vs dwconv_1d_causal = 9.2e-01  (unrelated -- the shift bug)
   So P is not a fixed property of the kernel, it is a TEMPLATE PARAMETER
   (dwconv1d.cc:69 `template <int T, int K, int P, bool BIAS>`, dwconv1d.cc:114 for the scalar
   form) that the Parakeet call sites (dwconv1d.cc:296-322) simply always instantiate at (K-1)/2.
   Setting P=K-1 at a NEW call site is exactly the same class of reuse as mha_decode's HD=64->128 --
   a template arg, not a kernel change. This script instantiates dwconv1d_same_scalar<T,7,6,true>
   (P=6=K-1) directly via #include, proving it on a REAL device run, not just the host algebra above.

2. K (kernel size). dwconv1d.cc ships K=9 (Parakeet FastConformer) and K=5 (GigaAM) instantiations.
   S2's real GGUF weight (c.quantizer.upsample.{0,1}.1.dwconv.conv.weight, confirmed via
   scripts/gguf_extract.py against the real s2-pro-q6_k.gguf) is K=7, C=1024. Two of the four
   internal FIR forms in dwconv1d.cc hardcode K=9 via `static_assert(K == 9, ...)`
   (dwconv1d_shift at dwconv1d.cc:152, dwconv1d_shift_f32o at dwconv1d.cc:214) -- those two are NOT
   directly reusable for K=7 without zero-padding 2 extra taps. But dwconv1d_same_scalar
   (dwconv1d.cc:114-129) and dwconv1d_tmajor_f32o (dwconv1d.cc:257-282) both take K as a genuinely
   free template parameter with no such assert -- K=7 compiles and runs with NO source change. This
   script uses dwconv1d_same_scalar for exactly that reason (see "kernel choice" below).

FUSED ACTIVATION: dwconv1d_same_scalar never had a SiLU epilogue branch at all (unlike
dwconv1d_shift, which has one gated behind -DDWCONV_SILU, default off). S2's dwconv_1d_causal is
bias-add only, no activation (GELU happens later, after LayerNorm+pwconv1) -- so the plain scalar
path needs no flag to stay plain; there's no macro to get wrong.

KERNEL CHOICE: dwconv1d_same_scalar over dwconv1d_shift. Reasons, not just "it happened to be free
on K": dwconv1d_shift also statically requires T % 16 == 0 (dwconv1d.cc:153), but S2's T is
data-dependent (T=110, T=220 below, from real quantizer_stage_up output shapes -- see docstring
example in codec_quantizer_ref.py's module doc: codes[10,55] -> latent[1024,220]) and is not
generally a multiple of 16. dwconv1d_same_scalar has neither constraint: no static_assert on K, no
static_assert on T, and (bonus) no local padded-copy buffer at all -- it bounds-checks directly into
`in[]`, so there is no alignas/spill surface to get wrong (see dwconv1d.cc:114-129). It is also
already the documented "correct on the current toolchain" path (dwconv1d.cc:107-113) and the same
"scalar first, vectorize later" phase route_b_kernels/bricks/conv-1d/conv_1d.cc explicitly commits
to for the analogous dense-causal brick ("SCALAR ... This is the correctness phase"). The
vectorized dwconv1d_shift COULD also be reused (P=K-1=8 with 2 leading zero-padded taps to reach
K=9, T padded up to a multiple of 16) -- that is a real, available perf lever, left unexplored here
by design: it is a speed question, and which FIR form (sliding_mul vs shuffle_down_fill vs scalar)
is fastest is an OPEN, unmeasured question per the 2026-07-30 dwconv miscompile-retraction note,
not something to decide inside a reuse-verify script.

COMPILE FLAG dwconv1d.cc NEEDS (found by actually compiling the generated shim with
route_b_kernels/bricks/_verify/compile_check.sh before wiring this into bricklib -- it failed
first): dwconv1d.cc:34 `#include "../aie_kernel_utils.h"` is a relative include resolved against
dwconv1d.cc's OWN directory (route_b_kernels/dwconv1d/), not the shim's. That header is not tracked
in route_b_kernels at all -- it is upstream mlir-aie's aie_kernels/aie_kernel_utils.h (a handful of
AIE_LOOP_UNROLL_FULL-style macros), and normally resolves only because scripts/sync_kernels.sh
copies dwconv1d.cc to <mlir-aie>/aie_kernels/aie2p/dwconv1d.cc first, sibling to the header. A
#include'ing shim outside that sandbox therefore needs one extra flag:
`-I<toolchain-instance>/src/aie_kernels/aie2p` (instance root = same one bricklib._aie_api_include()
derives from the live `aie` package), so `"../aie_kernel_utils.h"` resolves to
`<instance>/src/aie_kernels/aie_kernel_utils.h`. Confirmed COMPILES with this flag added (both T=110
and T=220), FAILS (fatal error: file not found) without it. This is a harness-plumbing fact about
dwconv1d.cc's existing include, not a new dependency this reuse introduces.

L1: this per-op segment is NOT C-resident (unlike the C=1024-whole-tile trap the task brief warns
about generically). Each core handles ONE channel's [T] row at a time, exactly like the existing
Parakeet dwconv1d.py dataflow (of_in/of_w/of_out all streamed per channel, C only controls the
`range_(cpc)` loop trip count, dwconv1d.py:50-58). At the larger S2 case (T=220, K=7): in_tile
220*2B=440B, w_tile 8*2B=16B, out_tile 440B, all objectFifo depth 2 -> ~1.8 KB/core, far under the
64 KB budget, with no dependence on C=1024 at all.

TEST STRUCTURE: dwconv is truly depthwise (no cross-channel mixing), so each channel's (in, w) pair
is independent -- there is no "resident operand shared across channels" in the real dataflow (unlike
conv-1d's dense conv, where every output channel legitimately shares one input activation). To fit
bricklib's generic streamed-tile-past-one-resident-operand shape (bricklib.verify_streamed) without
re-deriving a hand-rolled design, this script broadcasts ONE representative input row x0 as the
resident operand and streams REAL per-channel (tap, bias) tiles sampled across all 1024 real S2
channels as the tile operand -- i.e. it holds x fixed across channels and varies only w, which is a
valid way to gate "does the shift-and-accumulate FIR correctly realize the causal formula for
arbitrary real taps" (the thing in question) without re-proving that varying BOTH in and w per
channel plumbs correctly on device -- that per-channel-varying-both-operands dataflow is unchanged
from dwconv1d.py's own proven, WER-gated design and is not what P/K changed.
"""
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
DWCONV_CC = (ROOT / "route_b_kernels" / "dwconv1d" / "dwconv1d.cc").resolve()
assert DWCONV_CC.exists(), f"expected the shipped Parakeet dwconv1d.cc at {DWCONV_CC}"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import bricklib  # noqa: E402
import codec_quantizer_ref as cq  # noqa: E402  -- the real oracle-matching golden, not reimplemented
import gguf_extract as gx  # noqa: E402


def _aie_kernels_aie2p_include():
    """dwconv1d.cc:34 `#include "../aie_kernel_utils.h"` resolves relative to dwconv1d.cc's own
    directory, so a shim that #include's it from elsewhere (bricklib.GEN) needs this extra -I.
    Same instance-derivation trick as bricklib._aie_api_include() (that one resolves aie_api;
    this one resolves the sibling aie_kernels/aie2p dir the header actually lives next to)."""
    import aie
    inst = Path(aie.__file__).resolve().parent.parent.parent
    cand = inst / "src" / "aie_kernels" / "aie2p"
    if (cand / ".." / "aie_kernel_utils.h").resolve().exists():
        return [f"-I{cand}"]
    return []


GATE = 3e-2
_bf16 = ml_dtypes.bfloat16
K = 7                       # real K, read from the GGUF (c.quantizer.upsample.*.1.dwconv.conv.weight)
P_CAUSAL = K - 1            # the reparameterization: 'same' P=(K-1)/2 -> causal P=K-1
KW = K + 1                  # taps[0..K-1] + bias[K], no padding needed (dwconv1d_same_scalar is free-K)
GGUF = cq.DEFAULT_GGUF

# Real quantizer_stage_up output shapes (codec_quantizer_ref.py module docstring: codes[10,55] ->
# latent[1024,220], i.e. T=110 after upsample.0 (x2), T=220 after upsample.1 (x2 again)). Testing
# each stage's real tensor at its real T is more honest than one arbitrary T.
CASES = [(0, 110), (1, 220)]

# Sample across all 1024 real channels rather than a handful, since the property under test (does
# P=K-1 reproduce dwconv_1d_causal) is a per-channel scalar-tap computation with no cross-channel
# term -- a stride sample exercises many distinct real tap sets cheaply.
SEL = np.arange(0, 1024, 32)  # 32 channels

rng = np.random.default_rng(0)


def run_case(stage_idx, T):
    w_all = gx.load(GGUF, f"c.quantizer.upsample.{stage_idx}.1.dwconv.conv.weight").astype(np.float32)
    b_all = gx.load(GGUF, f"c.quantizer.upsample.{stage_idx}.1.dwconv.conv.bias").astype(np.float32)
    assert w_all.shape == (1024, 1, K), w_all.shape
    assert b_all.shape == (1024,), b_all.shape

    w_sel = w_all[SEL]          # [NCH,1,K]
    b_sel = b_all[SEL]          # [NCH]
    nch = len(SEL)

    x0 = (rng.standard_normal(T).astype(np.float32) * 0.5)
    x_bcast = np.tile(x0, (nch, 1))                     # [NCH,T], same row broadcast per channel

    ref = cq.dwconv_1d_causal(x_bcast, w_sel, b_sel)     # [NCH,T] -- the real oracle-matching golden

    # weight tiles: [NCH, KW] = [tap0..tap{K-1}, bias]
    w_tiles = np.zeros((nch, KW), np.float32)
    w_tiles[:, :K] = w_sel[:, 0, :]
    w_tiles[:, K] = b_sel

    _cb = int(time.time() * 1000) % 10 ** 9
    shim = bricklib.GEN / f"dwconv_causal_s{stage_idx}_t{T}_shim.cc"
    shim.write_text(
        f"// AUTO-GENERATED verify shim for dwconv-causal reuse of dwconv1d.cc (READ-ONLY, "
        f"#include'd not edited). cachebust {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{DWCONV_CC}"\n'
        # Positional wrapper: bricklib's resident-3-arg convention calls kern(tile_in, resident,
        # tile_out); the real kernel's own order is (in, w, out). Reorder here, same trick
        # bricks/_verify/verify_conv_1d.py's shim uses for conv_1d_causal_core.
        f'extern "C" void dwconv_causal_verify(bfloat16 *w_tile, bfloat16 *x0, bfloat16 *out) {{\n'
        f"  dwconv1d_same_scalar<{T}, {K}, {P_CAUSAL}, true>(x0, w_tile, out);\n"
        "}\n"
    )

    res = bricklib.verify_streamed(
        name=f"dwconv_causal_s{stage_idx}_t{T}",
        shim=shim,
        symbol="dwconv_causal_verify",
        in_tiles=w_tiles.astype(_bf16),
        out_tile_numel=T,
        resident=x0.astype(_bf16),
        unpack=lambda d: np.asarray(d).reshape(nch, T),
        golden=ref,
        gate=GATE,
        in_dt=_bf16, out_dt=_bf16, resident_dt=_bf16,
        compile_flags=_aie_kernels_aie2p_include(),
    )
    got = np.asarray(res["got"], np.float32)
    print(f"  stage upsample.{stage_idx} T={T} K={K} P={P_CAUSAL} (causal) NCH={nch}: "
          f"rel-L2 {res['rel_l2']:.3e}  max abs err {float(np.max(np.abs(got - ref))):.3e}  "
          f"{res['status']}")
    return res["ok"]


def main():
    ok = True
    for stage_idx, T in CASES:
        ok = run_case(stage_idx, T) and ok
    assert ok, "dwconv_causal (dwconv1d.cc reparameterized: K=7, P=K-1) device gate failed"
    print("PASS")


if __name__ == "__main__":
    main()
