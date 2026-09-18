#!/usr/bin/env python3
"""Export the four S2 AR-half bricks that already have proven kernels but no standalone
artifact -- rmsnorm, qk-norm, rope-interleaved, swiglu -- as `final.xclbin`+`insts.bin`+
`meta.json` triples at real S2 AR shapes, mirroring `codec_block/export_codec_artifacts.py`'s
artifact set (same three files per design, same `manifest.json` aggregation) so `rust/npu-s2`
can load them unchanged.

SCOPE: NOT `gemm-bfp16-ebs8` / `lm-head-argmax`. Both carry a large resident weight (wqkv ~31.5MB,
the LM head an embedding table) and whether AR weights are re-uploaded per dispatch (npu-s2's
current `S2Design::dispatch` convention) or registered once (npu-whisper's `register_weight`
convention) is an open owner decision this script does not make.

SHAPES come from `scripts/s2_ar_ref.py::read_ar_hparams` run against the real GGUF (not
hand-copied numbers) -- every dimension below cites the hp field and/or `docs/s2-ar-graph-map.md`
line it matches, and `_assert_hparams` fails loud if a future GGUF changes an assumption this file
bakes in (e.g. slow/fast sharing one embedding_length).

WHY SIX DIRECTORIES FROM FOUR KERNELS. `qk-norm` and `rope-interleaved` both normalize/rotate Q and
K, and Q/K have different row counts per decode step (head_count=32 vs head_count_kv=8) and (for
qk-norm) different gamma vectors (q_norm vs k_norm) -- two shapes, so two compiled xclbins, exactly
like `export_codec_artifacts.py` emitting many `stageN_resM_*` directories off one `conv-1d.cc`.
rmsnorm and swiglu need only one shape each (see their sections below).

ROPE'S DTYPE. `rope_interleaved.cc`'s real kernel (`rope_interleaved_prologue`) is bf16 Q/K
in/out, resident f32 cossin -- proven as-is by `_verify/verify_rope_interleaved.py`. `rust/npu-s2`
hardcodes f32-only (`S2Design::prepare`'s `is_f32` check, `F32_BYTES=4` sizing) for every buffer.
This script's `ar_rope` shim (see AR_ROPE section) wraps the proven kernel with an f32 boundary so
the exported artifact satisfies npu-s2's contract -- this is NEW code, not a byte-identical reuse
of the device-proven shim, and is flagged as such in the delivery report; device-gate it before
trusting it.

    python3 export_ar_artifacts.py <out_dir> [--only SUBSTRING]
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent          # this worktree
WS = REPO.parent


def _toolchain_env():
    """Same recipe as `codec_block/export_codec_artifacts.py::_toolchain_env` -- instance from
    THIS worktree's toolchain.lock, venv from wherever a `.venv-iron` exists, no NPU lock (this
    process never dispatches)."""
    inst = subprocess.check_output(
        [str(REPO / "scripts" / "toolchain_up.sh")], text=True).strip()
    if not inst:
        raise RuntimeError("toolchain_up.sh returned no instance dir")
    venv = os.environ.get("BRICK_VENV")
    if not venv:
        for cand in [REPO / ".venv-iron"] + sorted(WS.glob("*/.venv-iron")):
            if (cand / "bin" / "python").exists():
                venv = str(cand)
                break
    if not venv:
        raise RuntimeError("no .venv-iron found; set BRICK_VENV")
    peano = next((p for p in Path(venv, "lib").glob("python*/site-packages/llvm-aie")), None)
    if peano is None:
        raise RuntimeError(f"no llvm-aie under {venv}/lib/python*/site-packages/")
    os.environ["PATH"] = f"{venv}/bin:{venv}/cc-shim:{os.environ.get('PATH', '')}"
    os.environ["AIECC_PATH"] = f"{inst}/bin/aiecc"
    os.environ["PEANO_INSTALL_DIR"] = str(peano)
    os.environ.setdefault("XRT_INC_DIR", "/usr/include")
    os.environ.setdefault("XRT_LIB_DIR", "/usr/lib")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, f"{inst}/python")
    return inst, venv


INST, VENV = _toolchain_env()
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "aie_kernels" / "_test"))

import aie.iron as iron                          # noqa: E402
from aie.iron.device import NPU2                 # noqa: E402

iron.set_current_device(NPU2())                  # see module docstring: must precede any compile()

import bricklib                                   # noqa: E402
import codec_paths                                 # noqa: E402
import s2_ar_ref                                    # noqa: E402

BRICKS = REPO / "aie_kernels"


def _load_hparams():
    """Real AR hyperparameters, read from the shipped GGUF via `s2_ar_ref.read_ar_hparams` --
    not hand-copied numbers. `--report-types` on `scripts/s2_ar_ref.py` prints the same values
    this reads (measured 2026-09-02: block_count=36 embedding_length=2560 head_count=32/8
    attention_qk_norm=True; fast: block_count=4 head_count=32/8 head_dim=128 qk_norm=False)."""
    gg = s2_ar_ref.open_gguf(codec_paths.gguf())
    return s2_ar_ref.read_ar_hparams(gg)


def _assert_hparams(hp):
    """Every shape this script bakes into a design assumes slow and fast agree on
    embedding_length/feed_forward_length/head_dim/head_count/head_count_kv/rope_freq_base (so ONE
    rmsnorm/swiglu/rope design serves both stacks) and that qk-norm is slow-only. Fails loud if a
    future GGUF changes any of that instead of silently exporting a design for the wrong shape.
    (docs/s2-ar-graph-map.md's Slow/Fast table, section 1, records these as equal today.)
    """
    assert hp.embedding_length == hp.fast_embedding_length == 2560, (
        hp.embedding_length, hp.fast_embedding_length)
    assert hp.feed_forward_length == hp.fast_feed_forward_length == 9728, (
        hp.feed_forward_length, hp.fast_feed_forward_length)
    assert hp.head_dim == hp.fast_head_dim == 128, (hp.head_dim, hp.fast_head_dim)
    assert hp.head_count == hp.fast_head_count == 32, (hp.head_count, hp.fast_head_count)
    assert hp.head_count_kv == hp.fast_head_count_kv == 8, (hp.head_count_kv, hp.fast_head_count_kv)
    assert hp.rope_freq_base == hp.fast_rope_freq_base == 1e6, (
        hp.rope_freq_base, hp.fast_rope_freq_base)
    assert hp.attention_qk_norm is True, "qk-norm brick is only used where attention_qk_norm=True"
    assert hp.fast_attention_qk_norm is False, "fast stack has no qk-norm (confirms slow-only)"
    # rmsnorm_f32_nobeta/qk_norm_f32_gamma both hardcode eps=1e-6f; verified bit-identical to the
    # GGUF's own rms_norm_eps (np.float32(1e-6) == np.float32(hp.rms_norm_eps), checked 2026-09-02:
    # both are 9.999999974752427e-07) so the existing wrappers need no custom-eps shim.
    assert np.float32(hp.rms_norm_eps) == np.float32(1e-6), hp.rms_norm_eps
    assert np.float32(hp.fast_rms_norm_eps) == np.float32(1e-6), hp.fast_rms_norm_eps


# ---- design specs ---------------------------------------------------------------------------
# Each entry builds ONE compiled xclbin via bricklib._build_streamed -- the same resident-stream
# `kern(tile_in[, resident], tile_out)` ABI every device-verified brick in this tree already uses
# (aie_kernels/_test/bricklib.py). `shim` is generated C++ text, not a path -- written
# to <out_dir>/<name>/shim.cc so it is inspectable exactly like export_codec_artifacts.py's designs.

def _rmsnorm_spec(hp):
    """rmsnorm brick -> attention_norm/ffn_norm/norm.weight/fast_norm.weight, all cols=
    embedding_length (s2_ar_ref.py:322 embedding_length=2560; docs/s2-ar-graph-map.md lines 92,
    103, 109, 129 -- every rmsnorm call site in the AR graph is this same [*, 2560] shape). One
    row per call: decode step processes ONE token's residual stream at a time (the "M=1" decode
    granularity docs/s2-ar-graph-map.md section 6 uses for the decode-attention sizing row). No
    beta: the op graph (section 2/3) never adds a bias after rmsnorm -- gamma-only, matching
    rmsnorm_f32_nobeta.
    """
    cols = hp.embedding_length  # 2560
    shim = (
        '// AUTO-GENERATED export shim for the rmsnorm brick at S2 AR shape (cols=embedding_length).\n'
        '#include <stdint.h>\n'
        f'#include "{BRICKS / "rmsnorm" / "rmsnorm.cc"}"\n'
        'extern "C" void ar_rmsnorm(float *x, float *gamma, float *out) {\n'
        f'  rmsnorm_f32_nobeta(x, gamma, out, {cols});\n'
        '}\n'
    )
    return dict(design_name="rmsnorm", op="rmsnorm", symbol="ar_rmsnorm", shim=shim,
               n_tiles=1, in_tile=cols, out_tile=cols, resident_len=cols,
               in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
               compile_flags=[], resident_depth=2, stack_size=None,
               op_params=dict(kind="rmsnorm", cols=cols, eps=1e-6))


def _qk_norm_spec(hp, role):
    """qk-norm brick -> per-head RMSNorm on Q or K pre-attention, slow-only (attention_qk_norm=
    True; fast_attention_qk_norm=False, confirmed by _assert_hparams). cols=head_dim=128 (docs/
    s2-ar-graph-map.md line 59, "head_dim | 128 (from q_norm/k_norm shape, NOT embedding_length/
    head_count=80)"). rows = head_count (Q, 32) or head_count_kv (K, 8) -- one decode step's Q/K
    tensor is [rows, head_dim] after the reshape at op-graph step 8 (s2-ar-graph-map.md line 94).
    gamma differs between Q (q_norm) and K (k_norm) -- see qk_norm.cc's own header ("gamma is a
    single [cols] learned-scale vector... broadcast over every token/head row"), so this is
    inherently two shapes/two resident vectors, not two calls of one design.
    """
    cols = hp.head_dim  # 128
    rows = hp.head_count if role == "q" else hp.head_count_kv  # 32 or 8
    shim = (
        f'// AUTO-GENERATED export shim for the qk-norm brick ({role}), S2 AR shape cols=head_dim.\n'
        '#include <stdint.h>\n'
        f'#include "{BRICKS / "qk-norm" / "qk_norm.cc"}"\n'
        'extern "C" void ar_qk_norm(float *x, float *gamma, float *out) {\n'
        f'  qk_norm_f32_gamma(x, gamma, out, 1, {cols});\n'
        '}\n'
    )
    return dict(design_name=f"qk_norm_{role}", op="qk_norm", symbol="ar_qk_norm", shim=shim,
               n_tiles=rows, in_tile=cols, out_tile=cols, resident_len=cols,
               in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
               compile_flags=[], resident_depth=2, stack_size=None,
               op_params=dict(kind="qk_norm", role=role, cols=cols, rows=rows, eps=1e-6))


def _rope_spec(hp, role):
    """rope-interleaved brick -> RoPE(Q) / RoPE(K) pre-attention, ADJACENT-PAIR convention (see
    rope_interleaved.cc header). D=ROT=head_dim=128 -- full rotary, no partial-rotary tail:
    s2_ar_ref.py's rope_interleaved() asserts `n_dims == head_dim` at every call site in this
    model. rows = head_count (Q, 32) / head_count_kv (K, 8), same op-graph step (10) as qk-norm's
    role split. ROPE_M=1 (one row per kernel call, matching qk-norm's per-row streaming shape,
    NOT the whole [rows,D] block in one call) because every head-row at a decode step shares the
    SAME position (rope-interleaved/golden.py: "every head at a token shares that token's
    position") -- so the resident cossin table is exactly ONE [ROT]-wide row, acquired once and
    reused for all `rows` streamed tiles, not a [rows,ROT] table. This also keeps the design's L1
    footprint at ~2.5KB instead of ~64KB+ (a whole-block ROPE_M=rows design does not fit a 64KB
    core tile at rows=32: in+out fifos alone are 2*(2*32*128*4)=64KB before the resident or the
    kernel's own registers).

    AR_ROPE F32 WRAPPER (see module docstring): rope_interleaved_prologue is bf16 Q/K in/out; this
    shim widens/narrows at the boundary so the exported artifact is f32, matching npu-s2's only
    supported ABI. It reuses qk_out's own memory as the bf16 scratch (2 bytes/elem fits inside
    qk_out's 4 bytes/elem) instead of a new local array, specifically BECAUSE rope_interleaved.cc's
    own header flags local-scratch-buffer misalignment as a proven silent-corruption hazard on
    this toolchain ("NO LOCAL SCRATCH BUFFER" note) -- reusing an existing DMA-owned f32 buffer
    sidesteps that alignment class entirely, at the cost of a strict-aliasing violation (float* and
    bfloat16* over the same bytes) mitigated with -fno-strict-aliasing. The narrow step (f32->bf16
    into qk_out's memory) reads from the SEPARATE qk_in buffer, so ordering is unconstrained; the
    widen step (bf16->f32 in place) must run index DESCENDING: writing qk_out[i] touches bytes
    [4i,4i+4), and by induction every not-yet-read tmp[k] (k<i) lives at bytes [2k,2k+2) with
    2k+2 <= 4i (since k <= i-1), strictly below the write -- so no write ever clobbers an unread
    bf16 value. This is NEW code with no device precedent; see the delivery report.
    """
    D = hp.head_dim  # 128
    rows = hp.head_count if role == "q" else hp.head_count_kv  # 32 or 8
    shim = (
        f'// AUTO-GENERATED export shim for rope-interleaved ({role}), S2 AR shape D=head_dim, M=1\n'
        '// (one row per call; every head at a decode step shares one position -- see docstring).\n'
        '#include <stdint.h>\n'
        f'#include "{BRICKS / "rope-interleaved" / "rope_interleaved.cc"}"\n'
        'extern "C" void ar_rope(float *qk_in, float *cossin, float *qk_out) {\n'
        f'  bfloat16 *tmp = reinterpret_cast<bfloat16 *>(qk_out);\n'
        f'  for (unsigned i = 0; i < {D}u; ++i) tmp[i] = (bfloat16)qk_in[i];\n'
        '  rope_interleaved_prologue(tmp, cossin);\n'
        f'  for (unsigned i = {D}u; i-- > 0; ) qk_out[i] = (float)tmp[i];\n'
        '}\n'
    )
    return dict(design_name=f"rope_interleaved_{role}", op="rope_interleaved", symbol="ar_rope",
               shim=shim, n_tiles=rows, in_tile=D, out_tile=D, resident_len=D,
               in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
               compile_flags=[f"-DROPE_D={D}", f"-DROPE_ROT={D}", "-DROPE_M=1",
                              "-fno-strict-aliasing"],
               resident_depth=2, stack_size=None,
               op_params=dict(kind="rope_interleaved", role=role, D=D, rot=D, rows=rows,
                              rope_freq_base=hp.rope_freq_base))


def _swiglu_spec(hp):
    """swiglu brick -> FFN gate*up, total width = feed_forward_length (s2_ar_ref.py:323
    feed_forward_length=9728, shared by slow/fast; docs/s2-ar-graph-map.md line 145 "SwiGLU
    (silu(gate)*up) | bricks/swiglu | REUSE, exact match"). One decode step's FFN intermediate
    (M=1, same granularity as rmsnorm) is CHUNKED along the 9728-wide axis into `n_tiles` tiles,
    each tile's kernel call taking that chunk's gate‖up concatenated (the exact packing
    verify_f1.py's do_swiglu already establishes: swiglu_f32_f32(x, x+chunk, o)) -- no resident
    operand.

    Chunking is forced by two hardware ceilings measured empirically against this shape (a single
    n_tiles=1/in_tile=2*9728=19456 design was tried first and rejected by both):
      1. one aie.dma_bd's max transfer is 16383 32-bit words (aiecc: "buffer descriptor length
         (19456 32-bit words) exceeds the maximum of 16383" on the n_tiles=1 attempt) -- in_tile
         (=2*chunk) must stay under that.
      2. depth-2 in+out objectFifos must fit a 64KB core tile: bytes = 2*(2*chunk*4) [in] +
         2*(chunk*4) [out] = 24*chunk; targeting <=32KB (half the tile, leaving headroom for the
         kernel's own registers/stack, the same conservative margin the rope/qk-norm designs get
         "for free" from their much smaller per-tile shape) caps chunk at 1365.
    9728 = 2^9*19; the largest divisor of 9728 that is a multiple of 16 (swiglu.cc's
    static_assert(size%16==0)) and <=1365 is 1216 (9728/8), giving n_tiles=8, in_tile=2432 (well
    under the 16383-word ceiling), L1 usage 24*1216=29184B (~28.5KB, well under 32KB).
    """
    cols = hp.feed_forward_length  # 9728
    n_tiles = 8
    chunk = cols // n_tiles  # 1216; see chunking derivation above
    assert chunk * n_tiles == cols and chunk % 16 == 0 and 2 * chunk <= 16383
    shim = (
        '// AUTO-GENERATED export shim for the swiglu brick, S2 AR shape: feed_forward_length\n'
        f'// chunked into {n_tiles} tiles of {chunk} elements each (see _swiglu_spec docstring).\n'
        '#include <stdint.h>\n'
        f'#include "{BRICKS / "swiglu" / "swiglu.cc"}"\n'
        'extern "C" void ar_swiglu(float *x, float *out) {\n'
        f'  swiglu_f32_f32(x, x + {chunk}, out);\n'
        '}\n'
    )
    return dict(design_name="swiglu", op="swiglu", symbol="ar_swiglu", shim=shim,
               n_tiles=n_tiles, in_tile=2 * chunk, out_tile=chunk, resident_len=0,
               in_dt=np.float32, out_dt=np.float32, resident_dt=None,
               compile_flags=[f"-DSWIGLU_M=1", f"-DSWIGLU_N={chunk}"],
               resident_depth=2, stack_size=None,
               op_params=dict(kind="swiglu", cols=cols, n_tiles=n_tiles, chunk=chunk))


def build_specs(hp):
    return [
        _rmsnorm_spec(hp),
        _qk_norm_spec(hp, "q"),
        _qk_norm_spec(hp, "k"),
        _rope_spec(hp, "q"),
        _rope_spec(hp, "k"),
        _swiglu_spec(hp),
    ]


# ---- compile + write artifacts, mirroring export_codec_artifacts.py's export_one/_up_to_date ----

_DTYPE_BYTES = {"float32": 4, "bfloat16": 2}


def _read_toolchain_pin():
    text = (REPO / "toolchain.lock").read_text()
    out = {}
    for var in ("MLIR_AIE_FORK_COMMIT", "PEANO_FORK_COMMIT"):
        m = re.search(rf'^{var}=(\S+)', text, re.M)
        out[var] = m.group(1) if m else None
    return out


def _up_to_date(meta_path, xclbin_path, inst_path, symbol, flags, shim_sha):
    if not (meta_path.exists() and xclbin_path.exists() and inst_path.exists()):
        return False
    try:
        prev = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (prev.get("shim_sha256") == shim_sha and prev.get("symbol") == symbol
            and prev.get("compile_flags") == flags)


def export_one(spec, out_dir):
    ddir = out_dir / spec["design_name"]
    ddir.mkdir(parents=True, exist_ok=True)
    xclbin_path, inst_path, meta_path = ddir / "final.xclbin", ddir / "insts.bin", ddir / "meta.json"
    shim_sha = hashlib.sha256(spec["shim"].encode()).hexdigest()

    if _up_to_date(meta_path, xclbin_path, inst_path, spec["symbol"], spec["compile_flags"], shim_sha):
        meta = json.loads(meta_path.read_text())
        return dict(spec, build_s=0.0, skipped=True, xclbin_bytes=xclbin_path.stat().st_size,
                    out_bytes=meta["buffer_bytes"]["out"])

    shim_path = ddir / "shim.cc"
    shim_path.write_text(spec["shim"])

    design = bricklib._build_streamed(
        spec["symbol"], shim_path, spec["n_tiles"], spec["in_tile"], spec["out_tile"],
        spec["resident_len"], spec["compile_flags"], spec["in_dt"], spec["out_dt"],
        spec["resident_dt"], resident_depth=spec["resident_depth"], stack_size=spec["stack_size"])

    t0 = time.time()
    design.compile(xclbin_path=xclbin_path, inst_path=inst_path)
    build_s = time.time() - t0

    n_tiles, in_tile, out_tile = spec["n_tiles"], spec["in_tile"], spec["out_tile"]
    resident_len = spec["resident_len"]
    in_bytes = n_tiles * in_tile * _DTYPE_BYTES["float32"]
    out_bytes = n_tiles * out_tile * _DTYPE_BYTES["float32"]
    resident_bytes = resident_len * _DTYPE_BYTES["float32"]
    insts_bytes = inst_path.stat().st_size
    xclbin_bytes = xclbin_path.stat().st_size

    meta = dict(
        design_name=spec["design_name"], symbol=spec["symbol"], op=spec["op"],
        group="ar", stage=None, dispatch_tag=spec["design_name"],
        # NOT named "op_params": npu-s2's S2Meta.op_params is a #[serde(tag = "kind")] enum with
        # exactly three variants (Snake/Conv/ConvTranspose, the codec chain's own op kinds) --
        # an internally-tagged enum with an unrecognized "kind" (ours: rmsnorm/qk_norm/
        # rope_interleaved/swiglu) is a hard deserialize error, not a graceful skip, even though
        # the field is Option<_>. "ar_params" is not a field S2Meta declares, so it lands in its
        # #[serde(flatten)] extra: serde_json::Map catch-all instead -- preserved, not validated.
        ar_params=spec["op_params"],
        n_tiles=n_tiles, in_tile=in_tile, out_numel=out_tile, resident_len=resident_len,
        resident_depth=spec["resident_depth"], compile_flags=spec["compile_flags"],
        dtypes={"in": "float32", "out": "float32",
               "resident": ("float32" if resident_len else None)},
        buffer_bytes={"in": in_bytes, "out": out_bytes, "resident": resident_bytes},
        insts_bytes=insts_bytes, insts_words=insts_bytes // 4, xclbin_bytes=xclbin_bytes,
        shim_sha256=shim_sha,
        xclbin_sha256=hashlib.sha256(xclbin_path.read_bytes()).hexdigest(),
        build_seconds=build_s,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return dict(spec, build_s=build_s, skipped=False, xclbin_bytes=xclbin_bytes, out_bytes=out_bytes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--only", default=None,
                    help="substring filter against design_name/op; matches are compiled, "
                         "everything else is still enumerated (for manifest.json) but skipped")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hp = _load_hparams()
    _assert_hparams(hp)
    specs = build_specs(hp)
    print(f"enumerated {len(specs)} AR designs from {len(set(s['op'] for s in specs))} bricks "
          f"(embedding_length={hp.embedding_length} feed_forward_length={hp.feed_forward_length} "
          f"head_dim={hp.head_dim} head_count={hp.head_count}/{hp.head_count_kv})", flush=True)

    rows = []
    for spec in specs:
        haystack = f"{spec['design_name']} {spec['op']}"
        if args.only and args.only not in haystack:
            continue
        rows.append(export_one(spec, args.out_dir))
        r = rows[-1]
        tag = "SKIP(up-to-date)" if r["skipped"] else f"{r['build_s']:6.1f}s"
        print(f"  [{tag:>16s}] {r['design_name']:24s} {r['n_tiles']:4d}x{r['in_tile']:<6d} "
              f"out={r['out_bytes']:7d}B xclbin={r['xclbin_bytes']:8d}B", flush=True)

    manifest_designs = []
    for md in sorted(args.out_dir.glob("*/meta.json")):
        try:
            m = json.loads(md.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        manifest_designs.append(dict(
            name=m["design_name"], dir=md.parent.name, op=m["op"], stage=m["stage"],
            group=m["group"], xclbin_bytes=m["xclbin_bytes"], build_seconds=m["build_seconds"]))
    manifest = dict(toolchain=_read_toolchain_pin(), designs=manifest_designs)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{'design':24s} {'tiles':>11s} {'out B':>9s} {'xclbin B':>10s} {'build s':>8s}")
    for r in rows:
        print(f"{r['design_name']:24s} {r['n_tiles']:4d}x{r['in_tile']:<6d} {r['out_bytes']:9d} "
              f"{r['xclbin_bytes']:10d} {r['build_s']:8.2f}")
    total_s = sum(r["build_s"] for r in rows)
    print(f"\n{len(rows)}/{len(specs)} designs exported to {args.out_dir}, "
          f"{total_s:.1f}s total build time")


if __name__ == "__main__":
    main()
