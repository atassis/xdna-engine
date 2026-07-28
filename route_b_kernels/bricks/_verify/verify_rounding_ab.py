#!/usr/bin/env python3
"""A/B the gemm-bf16xbfp16 output narrow's rounding mode, on device.

WHY THIS EXISTS, and why the normal harness could not answer it.

`gemm_bf16xbfp16_core` ends each output tile with `acc.to_vector<bfloat16>()`.
That accfloat -> bf16 narrow obeys the core's GLOBAL rounding register, and
aie_api never sets it; the documented default is floor. So the brick was biased
toward negative infinity by up to one ulp on every element.

The existing verifier (`verify_bfp16.py::do_gemm_bf16xbfp16`) CANNOT see that,
because its shim sets `aie::set_rounding(conv_even)` for its own on-chip B
quantization -- and the rounding mode is a sticky core register, so that setting
was still in effect when the kernel narrowed its output. The harness was
silently supplying the very thing the kernel was missing. A brick that passes
only because its test shim configures the core for it is not verified.

RESULT (2026-07-28, on device): all four arms are BIT-IDENTICAL at
rel-L2 6.777424096e-03. Forcing floor explicitly changes nothing against forcing
conv_even, so this narrow does NOT consult the rounding register at all -- even
though `mov crrnd, #0xc` is verifiably emitted into the object. Truncation and
round-to-nearest differ on ~half of random mantissas, so across 4096 outputs
that rules out "the data happened not to care".

So the premise the sweep flagged is FALSE FOR THIS SITE: there was no floor bias
to fix. The kernel keeps the explicit set_rounding as a one-instruction
declaration of intent, not as a bug fix.

Four arms, identical except where the register is set:

  A  masked_rne      shim sets conv_even, kernel does NOT  (what the old
                     harness measured)
  B  unmasked_floor  shim sets nothing,   kernel does NOT  (shipped, unmasked)
  C  unmasked_rne    shim sets nothing,   kernel DOES      (the fix alone)
  D  forced_floor    shim sets FLOOR,     kernel does NOT  (the CONTROL)

D is the arm that decides it. B vs C prices the fix; A vs C shows the masking;
D vs C tells you whether the register is live at all -- and it is not.

Run:  ./run.sh verify_rounding_ab.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bricklib  # noqa: E402
from verify_bfp16 import BF16, pack_B_nk_blocks  # noqa: E402
from verify_f2 import golden_mod, tile_pack, tile_unpack  # noqa: E402

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)
BRICK_CC = Path(__file__).parents[1] / "gemm-bf16xbfp16" / "gemm_bf16xbfp16.cc"

M, K, N = 64, 64, 64


def variant_cc(name, source_text):
    """Materialise a .cc variant under gen/ (gitignored) and return its path."""
    p = GEN / f"gemm_bf16xbfp16_{name}.cc"
    p.write_text(source_text)
    return str(p)


# The commit that ADDED set_rounding to the brick. The floor baseline must come from its
# PARENT, not from HEAD.
#
# This bit me: the first run of this script used `HEAD`, which by then already contained the
# fix, so all three arms silently ran the SAME kernel and returned bit-identical rel-L2. That
# reads as "the fix changes nothing" when it actually means "the experiment varied nothing" --
# the same class of error as the stale-object A/B that once inverted a verdict here. Resolve the
# baseline by content, and assert it differs from the working tree before trusting any number.
FIX_COMMIT = "eb57ccc"


def head_version():
    """The PRE-FIX kernel, straight out of git -- not a hand-edit, and not HEAD."""
    repo = Path(__file__).parents[3]
    rel = BRICK_CC.relative_to(repo)
    src = subprocess.run(
        ["git", "-C", str(repo), "show", f"{FIX_COMMIT}^:{rel.as_posix()}"],
        capture_output=True, text=True, check=True).stdout
    if "set_rounding" in src:
        raise SystemExit(
            f"baseline resolved from {FIX_COMMIT}^ still contains set_rounding -- "
            "it is not the pre-fix kernel, refusing to report a bogus A/B")
    if "set_rounding" not in BRICK_CC.read_text():
        raise SystemExit(
            "working-tree kernel has no set_rounding -- the 'fixed' arm is not fixed")
    return src


def shim_src(cc_path, set_rounding_in_shim):
    """The verify shim. `set_rounding_in_shim` is the ONLY thing that varies
    between arms: None = set nothing, or a rounding_mode name to set."""
    setr = (f'aie::set_rounding(aie::rounding_mode::{set_rounding_in_shim});'
            if set_rounding_in_shim else '')
    return (
        'extern "C" void gemm_bf16xbfp16_verify(const bfloat16*A,const bfloat16*Bbf16,bfloat16*C){'
        f'constexpr unsigned M={M},K={K},N={N},KB=K/8,NB=N/8,NBLK=KB*NB;'
        'using BV=aie::block_vector<bfp16ebs8,64>;'
        f'{setr}'
        'alignas(64) static uint8_t Bq_raw[NBLK*BV::memory_bytes()];'
        'bfp16ebs8*Bq=reinterpret_cast<bfp16ebs8*>(Bq_raw);'
        'aie::block_vector_output_buffer_stream<bfp16ebs8,64> bout(Bq);'
        'for(unsigned blk=0;blk<NBLK;++blk){'
        'aie::vector<bfloat16,64> v=aie::load_v<64>(Bbf16+blk*64);'
        'aie::accum<accfloat,64> acc(v);bout.push(::to_v64bfp16ebs8(acc));}'
        'gemm_bf16xbfp16_core<8,8,8,M/8,K/8,N/8>(A,Bq,C);}')


def run_arm(label, cc_path, set_rounding_in_shim, a_bf, b_bf, ref):
    res = bricklib.verify_oneshot(
        f"gemm-bf16xbfp16-{label}", cc_path,
        shim_src(cc_path, set_rounding_in_shim), "gemm_bf16xbfp16_verify",
        inputs=[(tile_pack(a_bf, 8, 8), BF16), (pack_B_nk_blocks(b_bf, K, N), BF16)],
        out_numel=M * N, out_shape=(M, N),
        unpack=lambda flat: tile_unpack(np.asarray(flat, np.float32), M, N, 8, 8),
        golden=ref, gate=3e-2, out_dt=BF16)
    return res


def main():
    g, _ = golden_mod("gemm-bf16xbfp16", "gemm_bf16xbfp16.cc")
    rng = np.random.default_rng(43)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = (rng.standard_normal((K, N)).astype(np.float32) * (1.0 / np.sqrt(K)))
    ref = np.asarray(g.gemm_bf16xbfp16(a, b), np.float32)
    a_bf = g.to_bf16(a).astype(BF16)
    b_bf = g.to_bf16(b).astype(BF16)

    cc_floor = variant_cc("floor", head_version())          # committed kernel
    cc_rne = variant_cc("rne", BRICK_CC.read_text())        # working-tree kernel

    # Arm D is the CONTROL, and it is the one that decides whether any of this matters.
    # It sets `floor` EXPLICITLY from the shim against the pre-fix kernel. If D differs from
    # C, the register is live and the hardware's reset value simply is not the floor the
    # aie_api docs claim. If D equals C, then this narrow never consults crrnd at all and the
    # whole premise -- ours and #3442's framing of it -- needs revisiting.
    arms = [
        ("A masked_rne    ", cc_floor, "conv_even"),
        ("B unmasked_floor", cc_floor, None),
        ("C unmasked_rne  ", cc_rne, None),
        ("D forced_floor  ", cc_floor, "floor"),
    ]
    out = {}
    for label, cc, setr in arms:
        r = run_arm(label.strip().replace(" ", ""), cc, setr, a_bf, b_bf, ref)
        out[label.strip()] = r
        print(f"{label}  rel-L2 = {r['rel_l2']:.9e}   {'PASS' if r['ok'] else 'FAIL'}")

    print()
    b = out["B unmasked_floor"]["rel_l2"]
    c = out["C unmasked_rne"]["rel_l2"]
    d = out["D forced_floor"]["rel_l2"]
    print(f"CONTROL  D(explicit floor) vs C(rne): {d:.9e} vs {c:.9e}")
    if abs(d - c) < 1e-12:
        print("  -> IDENTICAL. Explicitly forcing floor changes NOTHING, so this narrow")
        print("     does not depend on the rounding register at all. The premise is wrong:")
        print("     the fix is inert here, and #3442's separate A/B-conversion site is what")
        print("     the rounding argument actually rests on.")
    else:
        print("  -> DIFFER. The register IS live, so the hardware reset value is already")
        print("     conv_even, not the floor the aie_api docs claim. The fix is correct and")
        print("     defensive but buys nothing at the default.")
    print(f"FIX      B(unset) vs C(rne):         {b:.9e} vs {c:.9e}"
          f"  -> {'no change' if abs(b - c) < 1e-12 else 'CHANGED'}")


if __name__ == "__main__":
    main()
