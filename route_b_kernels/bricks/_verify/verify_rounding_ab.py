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

Three arms, all identical except where the register is set:

  A  masked_floor    shim SETS conv_even, kernel does NOT   (what the old
                     harness measured -- expected to pass, and to be
                     indistinguishable from C)
  B  unmasked_floor  shim does NOT set,   kernel does NOT   (the shipped
                     behaviour with no shim to mask it -- expected to show
                     the floor bias)
  C  unmasked_rne    shim does NOT set,   kernel DOES       (the fix, standing
                     on its own)

B vs C is the measurement that actually prices the fix. A vs C is the evidence
that the old harness was masking.

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


def head_version():
    """The committed (pre-fix) kernel, straight out of git -- not a hand-edit."""
    repo = Path(__file__).parents[3]
    rel = BRICK_CC.relative_to(repo)
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{rel.as_posix()}"],
        capture_output=True, text=True, check=True).stdout


def shim_src(cc_path, set_rounding_in_shim):
    """The verify shim. `set_rounding_in_shim` is the ONLY thing that varies
    between the masked and unmasked arms."""
    setr = 'aie::set_rounding(aie::rounding_mode::conv_even);' if set_rounding_in_shim else ''
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

    arms = [
        ("A masked_floor  ", cc_floor, True),
        ("B unmasked_floor", cc_floor, False),
        ("C unmasked_rne  ", cc_rne, False),
    ]
    out = {}
    for label, cc, setr in arms:
        r = run_arm(label.strip().split()[0] + label.strip().split()[1], cc, setr, a_bf, b_bf, ref)
        out[label] = r
        print(f"{label}  rel-L2 = {r['rel_l2']:.6e}   {'PASS' if r['ok'] else 'FAIL'}")

    print()
    a_l2 = out["A masked_floor  "]["rel_l2"]
    b_l2 = out["B unmasked_floor"]["rel_l2"]
    c_l2 = out["C unmasked_rne  "]["rel_l2"]
    print(f"masking effect (A vs C): {a_l2:.6e} vs {c_l2:.6e}"
          f"  -> harness {'WAS masking' if abs(a_l2 - c_l2) < 1e-9 else 'differs'}")
    print(f"the fix        (B vs C): {b_l2:.6e} vs {c_l2:.6e}"
          f"  -> {b_l2 / c_l2:.2f}x better" if c_l2 else "")


if __name__ == "__main__":
    main()
