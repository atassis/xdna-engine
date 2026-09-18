#!/usr/bin/env python3
"""DEVICE gate for the bf16 arm of the codec's conv kernels (Q2, the FORMAT lever).

NON-DEVICE authored (session nondev-trace); NOT run here -- hand to the device agent.
Run under the device lock, PYTHONPATH at whichever instance toolchain_up.sh currently resolves to:

    python3 verify_conv_bf16_device_gate.py [--gate 3e-2]

WHAT THIS GATES, per op the residual unit / upsample stage actually use, at REAL S2-Pro GGUF
weights and REAL checkpoint shapes (c_in=96, k=7, t=32 for the dilated+1x1 conv; c_in=192, c_out=96,
k=4, stride=2, t=16 for stage-4 conv-transpose) -- never synthetic tensors:

  1. conv_1d_causal_core_bf16_scalar / _vec  (bricks/conv-1d/conv_1d_bf16.cc), dilations 1/3/9 (the
     residual unit's dilated conv) and k=1 (its 1x1 conv).
  2. conv_transpose_channel_core_bf16 (codec_block/conv_transpose_channel_bf16.cc), stage-4 shape.

Each gate reports rel-L2 against TWO references (see bricks/conv-1d/golden_bf16.py's header for
why both matter):
  - vs the f32 TRUTH golden       -> what bf16 format costs in accuracy (the number the residual
                                      unit's own 3e-2 gate cares about).
  - vs the bf16 MODEL golden      -> device correctness, isolated from bf16's own unavoidable
                                      quantization loss (a device/layout bug shows up here even if
                                      the format-accuracy number above still passes).

WHAT THIS DOES NOT GATE. No fused, on-device, multi-op residual unit or upsample stage: composing
these bricks into one hardware context needs either the resident-stream carrying the intermediate
(architecture work, not a kernel-authoring change) or reusing residual_unit.cc's static-buffer
two-phase pattern, which conflicts with this task's "no static L1 state" constraint -- see this
change's own report for the tradeoff. A device-free HOST SIMULATION of the FULL decoder chain
(head -> 4 stages -> tail) with bf16 quantization injected at every conv/conv-transpose boundary is
in host_bf16_codec_sim.py instead, run separately (no device needed) against the SAME
~/.cache/s2-oracle/dump80/codec_audio.bin oracle this script's per-op numbers feed into.

Rounding: conv_1d_bf16.cc's vector core calls `aie::set_rounding(conv_even)` before its
accfloat->bfloat16 narrow, defensively (see that file's header -- measured INERT for the identical
conversion shape elsewhere in this tree, not re-verified here). --rounding-ablation adds a second
build of the vector core with that call compiled out (-DCONV1D_BF16_NO_SET_ROUNDING), so this run
can settle empirically, for THIS kernel, whether it matters -- same A/B shape as the fc1 rounding
probe in kb/log/bfp16-emulation-inherits-floor-rounding.md.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import ml_dtypes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "aie_kernels" / "_test"))
sys.path.insert(0, str(ROOT / "scripts"))

import bricklib  # noqa: E402
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402

BF16 = ml_dtypes.bfloat16
GGUF = Path(codec_paths.gguf())

CONV1D_BRICK = (ROOT / "aie_kernels" / "conv-1d").resolve()
CONV1D_BF16_CC = CONV1D_BRICK / "conv_1d_bf16.cc"
CT_BF16_CC = (HERE / "conv_transpose_channel_bf16.cc").resolve()

C_IN, C_OUT, K, T = 96, 96, 7, 32
CT_C_IN, CT_C_OUT, CT_K, CT_STRIDE, CT_T = 192, 96, 4, 2, 16
CT_OUT_LEN = CT_T * CT_STRIDE

# .block.2/.3/.4 of decoder stage 4: the three residual units, dilations 1/3/9. Same tensors
# verify_conv_1d.py gates the f32 arm against.
DILATED_UNITS = [("c.decoder.model.4.block.2.block.1.conv", 1),
                 ("c.decoder.model.4.block.3.block.1.conv", 3),
                 ("c.decoder.model.4.block.4.block.1.conv", 9)]
ONEBYONE = ("c.decoder.model.4.block.2.block.3.conv", 1, 1)   # (tensor, k, dilation)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f32_golden = _load_module(CONV1D_BRICK / "golden.py", "conv1d_golden_f32")
bf16_golden = _load_module(CONV1D_BRICK / "golden_bf16.py", "conv1d_golden_bf16")
ct_f32_golden = _load_module(ROOT / "aie_kernels" / "conv-transpose-1d" / "golden.py",
                             "ct_golden_f32")
ct_bf16_golden = _load_module(HERE / "golden_conv_transpose_bf16.py", "ct_golden_bf16")


def report(op_name, res, ref_f32, ref_bf16):
    got = np.asarray(res["got"], np.float64)
    rl2_truth = bf16_golden.rel_l2(got, ref_f32)
    rl2_model = bf16_golden.rel_l2(got, ref_bf16)
    print(f"  {op_name:32s} vs f32-truth rel-L2 {rl2_truth:.3e}  vs bf16-model rel-L2 {rl2_model:.3e}  "
         f"nz={res['nonzero']:.2e} run2run={res['run2run']:.2e}  {res['status']}")
    return rl2_truth


def gate_dilated_conv(core_kind, gate, rounding_flag):
    """core_kind: 'scalar' or 'vec'. Returns list of (name, rl2_vs_truth, ok)."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((C_IN, T)).astype(np.float32) * 0.5)
    resident_bf16 = np.ascontiguousarray(x.reshape(-1)).astype(BF16)
    results = []

    def _build_and_run(tag, tensor, k, dilation, symbol_kind):
        w = gx.load(str(GGUF), f"{tensor}.weight").astype(np.float32)
        bias = gx.load(str(GGUF), f"{tensor}.bias").astype(np.float32).reshape(-1)
        assert w.shape == (C_OUT, C_IN, k), f"{tensor}: weight shape {w.shape}"

        ref_f32 = f32_golden.conv_1d_causal_ref(x, w, bias, dilation)
        ref_bf16 = bf16_golden.conv_1d_causal_bf16_model(x, w, bias, dilation)

        tile_w = C_IN * k
        # weight+bias packed as tile_w+1 columns; at bf16 (2 B/elem) an ODD column count is an
        # odd number of 2-byte units, which fails aiecc's dma_bd 4-byte-alignment check
        # ("transfer length must be multiple of 4") -- found running this gate (tile_w+1=673 for
        # the dilated conv, 97 for the 1x1 conv). tile_w itself is always even here (C_IN is
        # even), so +1 is always the odd case; pad one extra dead column so the DMA transfer
        # length is a whole number of 4-byte units. The kernel never reads past index tile_w
        # (bias), so the pad column is inert.
        pad_w = tile_w + 1 if (tile_w + 1) % 2 == 0 else tile_w + 2
        tiles = np.zeros((C_OUT, pad_w), np.float32)
        for co in range(C_OUT):
            tiles[co, :tile_w] = w[co].reshape(-1)
            tiles[co, tile_w] = bias[co]
        tiles_bf16 = tiles.astype(BF16)

        _cb = int(time.time() * 1000) % 10**9
        shim = bricklib.GEN / f"conv1d_bf16_{symbol_kind}_{tag}_shim.cc"
        symbol = f"conv1d_bf16_{symbol_kind}_verify_{tag}"
        extra_define = "-DCONV1D_BF16_NO_SET_ROUNDING" if rounding_flag == "off" else ""
        if symbol_kind == "scalar":
            call = (f"route_b_bricks::conv_1d_causal_core_bf16_scalar<{C_IN}, {k}, {T}, "
                   f"{dilation}>(resident, wtile, (float)wtile[{tile_w}], out);")
        else:
            call = (f"route_b_bricks::conv_1d_causal_core_bf16_vec<32, {C_IN}, {k}, {T}, "
                   f"{dilation}>(resident, wtile, (float)wtile[{tile_w}], out);")
        shim.write_text(
            f"// AUTO-GENERATED verify shim, conv-1d bf16 {symbol_kind}, {tag}. cb {_cb}\n"
            "#include <stdint.h>\n"
            f'#include "{CONV1D_BF16_CC}"\n'
            f'extern "C" void {symbol}(bfloat16 *wtile, bfloat16 *resident, bfloat16 *out) {{\n'
            f"  {call}\n"
            "}\n"
        )
        res = bricklib.verify_streamed(
            name=f"conv1d_bf16_{symbol_kind}_{tag}",
            shim=shim, symbol=symbol,
            in_tiles=tiles_bf16, out_tile_numel=T, resident=resident_bf16,
            unpack=lambda d: np.asarray(d).reshape(C_OUT, T).astype(np.float32),
            golden=ref_f32, gate=gate,
            in_dt=BF16, out_dt=BF16, resident_dt=BF16,
            compile_flags=[extra_define] if extra_define else None,
        )
        rl2 = report(f"{symbol_kind} {tag}", res, ref_f32, ref_bf16)
        results.append((tag, rl2, res["ok"]))

    for tensor, dilation in DILATED_UNITS:
        _build_and_run(f"dil{dilation}", tensor, K, dilation, core_kind)
    tensor, k1, d1 = ONEBYONE
    _build_and_run("1x1", tensor, k1, d1, core_kind)
    return results


def gate_conv_transpose(gate):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((CT_C_IN, CT_T)).astype(np.float32) * 0.5)
    resident_bf16 = np.ascontiguousarray(x.reshape(-1)).astype(BF16)

    w = gx.load(str(GGUF), "c.decoder.model.4.block.1.conv.weight").astype(np.float32)
    bias = gx.load(str(GGUF), "c.decoder.model.4.block.1.conv.bias").astype(np.float32).reshape(-1)
    assert w.shape == (CT_C_IN, CT_C_OUT, CT_K), f"weight shape {w.shape}"

    ref_f32 = ct_f32_golden.conv_transpose_1d_ref(x, w, bias, CT_STRIDE, crop_right=CT_STRIDE)
    ref_bf16 = ct_bf16_golden.conv_transpose_1d_bf16_model(x, w, bias, CT_STRIDE, crop_right=CT_STRIDE)

    tile_w = CT_C_IN * CT_K
    # same bf16 4-byte dma_bd alignment pad as _build_and_run above (tile_w+1=769 here, odd).
    pad_w = tile_w + 1 if (tile_w + 1) % 2 == 0 else tile_w + 2
    tiles = np.zeros((CT_C_OUT, pad_w), np.float32)
    for co in range(CT_C_OUT):
        tiles[co, :tile_w] = w[:, co, :].reshape(-1)
        tiles[co, tile_w] = bias[co]
    tiles_bf16 = tiles.astype(BF16)

    _cb = int(time.time() * 1000) % 10**9
    shim = bricklib.GEN / "convT_bf16_stage4_shim.cc"
    symbol = "convT_bf16_verify_stage4"
    shim.write_text(
        f"// AUTO-GENERATED verify shim, conv-transpose bf16, stage4. cb {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{CT_BF16_CC}"\n'
        f'extern "C" void {symbol}(bfloat16 *wtile, bfloat16 *resident, bfloat16 *out) {{\n'
        f"  route_b_bricks::conv_transpose_channel_core_bf16<{CT_C_IN}, {CT_K}, {CT_T}, "
        f"{CT_STRIDE}>(resident, wtile, (float)wtile[{tile_w}], out);\n"
        "}\n"
    )
    res = bricklib.verify_streamed(
        name="convT_bf16_stage4",
        shim=shim, symbol=symbol,
        in_tiles=tiles_bf16, out_tile_numel=CT_OUT_LEN, resident=resident_bf16,
        unpack=lambda d: np.asarray(d).reshape(CT_C_OUT, CT_OUT_LEN).astype(np.float32),
        golden=ref_f32, gate=gate,
        in_dt=BF16, out_dt=BF16, resident_dt=BF16,
    )
    rl2 = report("conv-transpose stage4", res, ref_f32, ref_bf16)
    return [("stage4", rl2, res["ok"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", type=float, default=3e-2)
    ap.add_argument("--rounding-ablation", action="store_true",
                    help="also build the vector core with set_rounding(conv_even) compiled out, "
                         "to settle empirically whether it matters for this narrow")
    args = ap.parse_args()

    all_ok = True
    print("=== dilated + 1x1 conv-1d, bf16 SCALAR core (reference, no vector-loop hazard) ===")
    for _, rl2, ok in gate_dilated_conv("scalar", args.gate, "on"):
        all_ok = all_ok and ok

    print("\n=== dilated + 1x1 conv-1d, bf16 VECTOR core (native aie::mac, the format lever) ===")
    vec_results = gate_dilated_conv("vec", args.gate, "on")
    for _, rl2, ok in vec_results:
        all_ok = all_ok and ok

    if args.rounding_ablation:
        print("\n=== rounding A/B: vector core with set_rounding(conv_even) COMPILED OUT ===")
        off_results = gate_dilated_conv("vec", args.gate, "off")
        print("\n  rounding A/B deltas (on - off), rel-L2 vs f32 truth:")
        for (tag, rl2_on, _), (_, rl2_off, _) in zip(vec_results, off_results):
            print(f"    {tag:10s} on={rl2_on:.6e} off={rl2_off:.6e} delta={rl2_on - rl2_off:+.3e}")

    print("\n=== conv-transpose, bf16 SCALAR core (stage-4 shape) ===")
    for _, rl2, ok in gate_conv_transpose(args.gate):
        all_ok = all_ok and ok

    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'} (gate {args.gate:.1e})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
