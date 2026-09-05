#!/usr/bin/env python3
"""Does accum<accfloat,N>::to_vector<bfloat16>() consult crRnd -- GENERALIZED past
the one file the 2026-07-28 note scoped its result to (gemm_bf16xbfp16.cc, N=64)?

Same call shape (accfloat accumulator narrowed with .template to_vector<bfloat16>())
recurs across 25+ kernels. This tests THREE more real, shipped sites, varying N and
kernel family, by patching each site's OWN rounding-mode call (floor / conv_even /
no call at all) and diffing device output bit-for-bit -- same 3-arm design the
2026-08-30 to_fixed audit used (verify_tofixed_rounding_ab.py).

Sites:
  residual_add    N=16  RESADD_BF16 arm   -- elementwise add, save/restore idiom
  ln_affine_cast  N=16  default (LN_BF16_WRITE=0) -- LN+affine, save/restore idiom
  gemm_bfp16_ebs8 N=64  mac_8x8_8x8T MMUL -- sibling GEMM family, plain set() (no
                        restore currently -- shipped state, see wt-rounding-contract
                        0d7b9fb which fixes exactly this file's save/restore but NOT
                        the rounding-consultation question asked here)

Each site's own conv_even/floor call is PATCHED in a `gen/`-only copy of the real
file (text substitution on the exact statement, nothing upstream touched). "none"
strips the call entirely so the narrow runs under whatever the ambient register is.

Anti-vacuity: for each site, host-computes the exact f32 value entering the narrow
and counts, via the bf16-truncation low-16-bits split at 0x8000, how many of the
64 output elements are rounding-sensitive (truncate-toward-zero disagrees with
round-nearest-even) -- reported before trusting any "bit-identical" verdict.

RESULT (2026-08-30 re-run, device f37308d2b719, xrt-smi Power Mode: Default):
ALL THREE SITES BREAK. crRnd is live at every site tested, so the 2026-07-28
single-file "inert" result generalizes to nothing.

  site              sensitive   floor-vs-conv_even   ULP        verdict
  residual_add      20/64       16/64                max 1      BREAKS
  ln_affine_cast    33/64       30/64                max 1      BREAKS
  gemm_bfp16_ebs8   34/64       57/64                max 240    BREAKS

run2run = 0.00e+00 on all nine arms; conv_even-vs-none matches floor-vs-conv_even
at every site, i.e. the ambient default is floor.

This SUPERSEDES the first run of this script, which reported ln_affine_cast and
gemm_bfp16_ebs8 INERT and residual_add at 1/64. That was a comparator defect, not a
device result -- see bits() below. Nothing about the device or the arms changed;
the same instance, pin and inputs re-run to the corrected numbers above.

Two things the first run left open are now closed by the corrected numbers. The
"1/64 vs 20/64 predicted" anomaly, flagged as an open mechanism needing an isolated
single-op repro, was the truncation: the real 16/64 sits just under the 20/64 host
proxy, so no emulated vector<float> intermediate needs to be invoked. And
gemm_bfp16_ebs8's max 240 ULP is far past a tie-breaking nudge -- the block-float
quantization itself is biased, not just the final narrow.

Run:  ./run.sh verify_crrnd_scope_ab.py
"""
import re
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bricklib  # noqa: E402

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)
BF16 = ml_dtypes.bfloat16
AIE_KERNELS = Path(__file__).parents[2] / "aie_kernels"
BRICKS = Path(__file__).parents[1]

MODES = ("floor", "conv_even", "none")


def bf16_sensitivity(y_f32):
    """Fraction of elements where a bf16 narrow of y_f32 is rounding-mode-sensitive:
    truncate-toward-zero (mask low 16 bits) disagrees with round-nearest-even."""
    bits = y_f32.astype(np.float32).view(np.uint32)
    trunc = bits & np.uint32(0xFFFF0000)
    bias = ((bits >> 16) & 1) + 0x7FFF
    rne = (bits.astype(np.uint64) + bias).astype(np.uint32) & np.uint32(0xFFFF0000)
    differ = trunc != rne
    return trunc, rne, differ


SWAP_RE = re.compile(
    r"const auto saved_rounding =\s*::?aie::swap_rounding\(::?aie::rounding_mode::conv_even\);")
RESTORE_RE = re.compile(r"::?aie::set_rounding\(saved_rounding\);")


def patch_swap_restore(src, mode, name):
    """Sites using the save/restore idiom (residual_add, ln_affine_cast).

    residual_add.cc carries the IDENTICAL swap/restore pair in three #if/#elif/#else
    wire-format branches (only one is preprocessor-active per build), so this replaces
    ALL occurrences rather than just the first -- a count=1 first-match replace silently
    patches a DEAD branch (caught in dry-run: RESADD_B_BF16's copy, not RESADD_BF16's).
    Patching dead branches too is harmless; they're discarded by the active -D anyway.
    """
    if mode == "conv_even":
        return src
    n = len(SWAP_RE.findall(src))
    if n == 0:
        raise SystemExit(f"{name}: swap_rounding(conv_even) pattern not found -- "
                          "source shape changed, refusing to report a bogus A/B")
    if mode == "floor":
        return SWAP_RE.sub(
            lambda m: m.group(0).replace("conv_even", "floor"), src, count=0)
    if mode == "none":
        src = SWAP_RE.sub("", src, count=0)
        src = RESTORE_RE.sub("", src, count=0)
        return src
    raise ValueError(mode)


SET_RE = re.compile(r"aie::set_rounding\(aie::rounding_mode::conv_even\);")


def patch_plain_set(src, mode, name):
    """Sites that just set() with no restore (gemm_bfp16_ebs8, shipped state)."""
    if mode == "conv_even":
        return src
    if not SET_RE.search(src):
        raise SystemExit(f"{name}: set_rounding(conv_even) pattern not found")
    if mode == "floor":
        return SET_RE.sub("aie::set_rounding(aie::rounding_mode::floor);", src, count=1)
    if mode == "none":
        return SET_RE.sub("", src, count=1)
    raise ValueError(mode)


def variant_file(name, mode, real_path, patch_fn):
    src = patch_fn(real_path.read_text(), mode, name)
    p = GEN / f"{name}_{mode}.cc"
    p.write_text(src)
    return str(p)


# ---------------------------------------------------------------------------
# Site 1: residual_add (RESADD_BF16 arm), N=16
# ---------------------------------------------------------------------------
def run_residual_add(rng):
    real = AIE_KERNELS / "residual_add.cc"
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)
    scale = 0.5
    a_bf = a.astype(BF16)
    b_bf = b.astype(BF16)
    # Exact f32 value entering the narrow (a widened bf16 + scale * b widened bf16;
    # bf16->f32 widen is exact, matches the kernel's own arithmetic).
    y = a_bf.astype(np.float32) + scale * b_bf.astype(np.float32)
    trunc, rne, differ = bf16_sensitivity(y)
    frac = differ.mean()
    print(f"[residual_add  N=16] rounding-sensitive elements: {differ.sum()}/64 ({frac:.0%})")

    golden = y.astype(BF16).astype(np.float64)
    results = {}
    for mode in MODES:
        cc = variant_file("residual_add", mode, real, patch_swap_restore)
        shim = (
            f'extern "C" void residual_ab_verify_{mode}(bfloat16*a,bfloat16*b,bfloat16*out){{'
            f'residual_add_row<16>(a,b,out,{scale}f,64);}}'
        )
        r = bricklib.verify_oneshot(
            f"residual_add-{mode}", cc, shim, f"residual_ab_verify_{mode}",
            inputs=[(a_bf, BF16), (b_bf, BF16)], out_numel=64, out_shape=(64,),
            unpack=lambda flat: flat, golden=golden, gate=5e-2,
            compile_flags=["-DRESADD_BF16"], out_dt=BF16)
        results[mode] = r
    return frac, results


# ---------------------------------------------------------------------------
# Site 2: ln_affine_cast (default, LN_BF16_WRITE=0 arm), N=16
# ---------------------------------------------------------------------------
def run_ln_affine_cast(rng):
    real = AIE_KERNELS / "ln_affine_cast.cc"
    cols = 64
    x = rng.standard_normal(cols).astype(np.float32) * 2.0
    gamma = (rng.standard_normal(cols).astype(np.float32) * 0.1 + 1.0)
    beta = rng.standard_normal(cols).astype(np.float32) * 0.1
    gb = np.concatenate([gamma, beta])

    mean = x.astype(np.float64).mean()
    var = ((x.astype(np.float64) - mean) ** 2).mean()
    inv_std = 1.0 / np.sqrt(var + 1e-5)
    y = (x.astype(np.float64) - mean) * inv_std * gamma.astype(np.float64) + beta.astype(np.float64)
    y_f32 = y.astype(np.float32)
    trunc, rne, differ = bf16_sensitivity(y_f32)
    frac = differ.mean()
    print(f"[ln_affine_cast N=16] rounding-sensitive elements: {differ.sum()}/64 ({frac:.0%})")

    golden = y_f32.astype(BF16).astype(np.float64)
    results = {}
    for mode in MODES:
        cc = variant_file("ln_affine_cast", mode, real, patch_swap_restore)
        shim = (
            f'extern "C" void ln_ab_verify_{mode}(float*input,float*gb,bfloat16*output){{'
            f'ln_affine_cast_row<16>(input,gb,output,{cols});}}'
        )
        r = bricklib.verify_oneshot(
            f"ln_affine_cast-{mode}", cc, shim, f"ln_ab_verify_{mode}",
            inputs=[(x, np.float32), (gb, np.float32)], out_numel=cols, out_shape=(cols,),
            unpack=lambda flat: flat, golden=golden, gate=5e-2, out_dt=BF16)
        results[mode] = r
    return frac, results


# ---------------------------------------------------------------------------
# Site 3: gemm_bfp16_ebs8 (mac_8x8_8x8T MMUL family, single 8x8x8 tile), N=64
# ---------------------------------------------------------------------------
def run_gemm_bfp16_ebs8(rng):
    real = BRICKS / "gemm-bfp16-ebs8" / "gemm_bfp16_ebs8.cc"
    A = rng.standard_normal((8, 8)).astype(np.float32)
    B = rng.standard_normal((8, 8)).astype(np.float32) * (1.0 / np.sqrt(8))
    A_bf = A.astype(BF16)
    B_bf = B.astype(BF16)
    # Approximate golden: bf16-widened exact matmul (NOT a bit-exact model of the
    # bfp16ebs8 block-float quantization the kernel actually performs -- that step
    # shares the same crRnd call, so this proxy is for the anti-vacuity estimate and
    # the loose correctness gate only, not for attributing a device diff to the
    # OUTPUT narrow specifically vs the A/B-quantization narrow. See report caveat.
    acc_f32 = (A_bf.astype(np.float64) @ B_bf.astype(np.float64)).astype(np.float32)
    trunc, rne, differ = bf16_sensitivity(acc_f32.reshape(-1))
    frac = differ.mean()
    print(f"[gemm_bfp16_ebs8 N=64] rounding-sensitive elements: {differ.sum()}/64 ({frac:.0%})")

    golden = acc_f32.reshape(-1).astype(BF16).astype(np.float64)
    results = {}
    for mode in MODES:
        cc = variant_file("gemm_bfp16_ebs8", mode, real, patch_plain_set)
        shim = (
            f'extern "C" void gemm_bfp16_ebs8_verify_{mode}(bfloat16*pA,bfloat16*pB,bfloat16*pC){{'
            f'gemm_bfp16_ebs8_tile<8,8,8>(pA,pB,pC);}}'
        )
        r = bricklib.verify_oneshot(
            f"gemm-bfp16-ebs8-{mode}", cc, shim, f"gemm_bfp16_ebs8_verify_{mode}",
            inputs=[(A_bf.reshape(-1), BF16), (B_bf.reshape(-1), BF16)],
            out_numel=64, out_shape=(64,),
            unpack=lambda flat: flat, golden=golden, gate=1.0e-1, out_dt=BF16)
        results[mode] = r
    return frac, results


def bits(r):
    """Device output as bf16 bit patterns.

    bricklib returns `got` as float64 VALUES, not bit patterns. The 2026-08-30 run
    compared arms with .astype(np.int64), which truncates toward zero: every site here
    emits standard-normal-scale outputs, so a 1-ULP bf16 difference only registered if
    it happened to cross an integer boundary. That reported ln_affine_cast INERT while
    the same log showed its floor arm at rel_L2 4.100e-03 against a golden its conv_even
    arm matched at 0. bf16 -> float64 -> bf16 is exact, so the view below is the real
    bit pattern.
    """
    return r["got"].astype(BF16).view(np.uint16)


def verdict(site, results):
    f, c, n = (bits(results[m]) for m in ("floor", "conv_even", "none"))
    d_fc = int(np.sum(f != c))
    d_cn = int(np.sum(c != n))
    total = f.size
    print(f"  floor vs conv_even: {d_fc}/{total} differ   "
          f"conv_even vs none: {d_cn}/{total} differ")
    # ULP spread, so a diff cannot be waved off as one denormal straggler.
    for label, a, b in (("floor/conv_even", f, c), ("conv_even/none", c, n)):
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        if d.any():
            print(f"    {label}: max {d.max()} ULP, mean over differing "
                  f"{d[d > 0].mean():.2f} ULP")
    generalizes = (d_fc == 0) and (d_cn == 0)
    print(f"  -> {'GENERALIZES (inert here too)' if generalizes else 'BREAKS -- register IS live at this site'}")
    return generalizes


def main():
    rng = np.random.default_rng(20260830)
    sites = [
        ("residual_add N=16", run_residual_add),
        ("ln_affine_cast N=16", run_ln_affine_cast),
        ("gemm_bfp16_ebs8 N=64", run_gemm_bfp16_ebs8),
    ]
    all_generalize = True
    for label, fn in sites:
        print(f"\n===== {label} =====")
        frac, results = fn(rng)
        if frac < 0.2:
            print(f"  WARNING: only {frac:.0%} of elements are rounding-sensitive -- "
                  "a bit-identical result here would be weak evidence.")
        ok = verdict(label, results)
        all_generalize = all_generalize and ok

    print("\n===== SUMMARY =====")
    print("ALL SITES GENERALIZE (register inert everywhere tested)" if all_generalize
          else "GENERALIZATION BREAKS on at least one site -- see per-site verdicts above")
    return 0 if all_generalize else 1


if __name__ == "__main__":
    sys.exit(main())
