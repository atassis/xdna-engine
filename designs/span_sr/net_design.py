"""SPAN x2 on the array: one core per conv, rows streamed front to back in ONE design.

Stage order, row layouts and skip depths come from net_layout. Weights enter once: one shim
channel per net_layout.weight_groups group, split to the cores in a MemTile, resident after.
conv_cat's four sources meet in a MemTile join as deep as the longest skip (net_layout.skip_depth);
each source core broadcasts its row to the next core and to the join. `upto` stops the chain after
that stage and drains its output, which is how the chain is brought up front to back.
"""
import hashlib
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import AnyMemTile, Tile
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

import net_layout as NL
from block_design import KDIR, _aie_api_include, _golden

CAT_DIR = KDIR.parent / "conv2d-1x1-cat"

# stack reservation per kind; aiecc measures each core and fails naming the bytes it needs
STACK = {"conv1": 2048, "silu16": 3584, "silu_i16": 3584, "silu_x": 2560, "silu": 2560,
         "gate": 3584, "plain": 2048, "cat": 2048, "up": 2048, "gate_half": 3584}
# objectFIFO depth of a stage's output (both ends, see the shared-pool trap). Was 3 (zero producer
# slack against b1c2's 3-row window forced b1c1/b1c2 to alternate, computes summed: TRACE_RESULTS.md
# "b1c1<->b1c2 alternation-sum, RE-CONFIRMED"); raised to 4 2026-09-27, device-confirmed on the
# full net at W=32: same-process trace, b1c1 586.7->482.8 cyc/px, b1c2 587.0->483.8, b3c3
# 574.5->483.8 (~16-18%), LOCK_STALL 54.4/45.1/49.2% -> 44.5/33.3/39.0%. Needs DATA_SIZES/
# SKIP_COND_DEPTHS below to fit L1 at W=32 (see TRACE_RESULTS.md's byte-budget section).
DEPTH = {"b1c1": 4}
# aiecc's own measured static-data size for b1c1's silu16 LUT table (checkDataSizeRequirements),
# reserved explicitly so DEPTH["b1c1"]=4's extra output buffer slot doesn't starve it (automatic
# placement left only ~1.5 KB where the LUT needs 4160 B once buffers grow).
DATA_SIZES = {"b1c1": 4160}
# conv_1's own consumer depth into b1c1 (normally MAIN_DEPTH), dropped by one slot to free 2304 B
# on b1c1's tile (0,3) -- the other half of DEPTH["b1c1"]=4's L1 cost. conv_1's own compute is
# only ~40 cyc/px (measured), so alternating conv_1<->b1c1 at zero slack gives a ~360 cyc/px
# ceiling on that link alone, comfortably under the ~480 cyc/px pace this net now runs at --
# device-confirmed not to become the new bottleneck (same trace run as above).
SKIP_CONS_DEPTHS = {"conv_1": 3}
MAIN_DEPTH = 4   # a 3-row window plus the row being written
PROD_DEPTH = 2   # producer side of a fifo that leaves the core by DMA (skip broadcasts)
GATE_SPLIT_LO = 32   # channels on split_gate's "lo" core; NL.split_gate_params' default (32/16)


def _call(kind, p, w, lay):
    c = f"{w}, check, {p['pre']}, {p['shift']}, 0, {w}"
    u, s = "(const uint8_t *)", "(const int16_t *)"
    h = NL.half(w)
    return {
        "conv1": lambda: f"conv3x3_u8i8({u}l0, {u}l1, {u}l2, p, o, {c});",
        "silu16": lambda: f"conv3x3_i8_lut16(l0, l1, l2, p, (int16_t *)o, {c});",
        "silu_i16": lambda: f"conv3x3_i16i8_lut({s}l0, {s}l1, {s}l2, p, o, {c});",
        "silu_x": lambda: f"conv3x3_i8_lut(l0, l1, l2, p, o, {c});",
        "silu": lambda: f"conv3x3_i8_lut(l0, l1, l2, p, o, {c});",
        "gate": lambda: (f"conv3x3_i8_gate(l0, l1, l2, l1 + {lay.x_in}, p, o, {c}, {p['ga']}, "
                         f"{p['gb']}, {p['gs1']}, {p['gc']}, {p['gs2']});"),
        "plain": lambda: f"conv3x3_i8(l0, l1, l2, p, o, {c});",
        "up": lambda: f"conv3x3_i8u8(l0, l1, l2, p, (uint8_t *)o, {c});",
        "cat": lambda: (f"conv1x1_cat_i8(l0, l0 + {h}, l0 + {2 * h}, l0 + {3 * h}, p, o, {w}, "
                        f"{p['pre']}, {p['shift']}, 0, {w});"),
    }[kind]()


def _shim(path, sym, kind, p, w, bracket=False):
    """bracket=True wraps the whole call (kernel + skip forward) in event0()/event1() so a
    traced core's per-row compute vs. gap is readable off the trace, mirroring
    codec_block/trace_conv_dispatch.py's bracketing convention. Off by default: production
    shims stay byte-identical."""
    lay = NL.layout(kind, w)
    fwd = ""
    if lay.x_in is not None and lay.x_out is not None:
        fwd = (f"  for (int i = 0; i < {NL.half(w)}; i += 64)\n"
               f"    aie::store_v(o + {lay.x_out} + i, aie::load_v<64>(l1 + {lay.x_in} + i));\n")
    if kind == "cat":
        src, args = CAT_DIR / "conv1x1_cat.cc", "int8_t *l0, int8_t *p, int8_t *o"
    else:
        src = KDIR / "conv3x3_u8.cc"
        args = "int8_t *l0, int8_t *l1, int8_t *l2, int8_t *p, int8_t *o, int32_t check"
    ev0, ev1 = ("  event0();\n", "  event1();\n") if bracket else ("", "")
    path.write_text(f'#include <stdint.h>\n#include "{src}"\n'
                    f'extern "C" void {sym}({args}) {{\n{ev0}  {_call(kind, p, w, lay)}\n{fwd}{ev1}}}\n')
    return path


def weights_blob(NP, names, split_gate=None, lo_channels=GATE_SPLIT_LO):
    """NL.weights_blob, but honoring split_gate the way build() lays weights out on the wire: a
    split stage contributes its lo half then its hi half (NL.split_gate_params) instead of one
    full blob -- build()'s weight groups are keyed on core_names, not `names`."""
    split_gate = frozenset(split_gate or ())
    parts = []
    for n in names:
        if n in split_gate:
            npl, nph = NL.split_gate_params(NP[n], lo_channels=lo_channels)
            parts += [npl["blob"], nph["blob"]]
        else:
            parts.append(NP[n]["blob"])
    return np.concatenate(parts).astype(np.int8)


def _shim_gate_half(path, sym, p, w, xoff, bracket=False):
    """Half-COUT gate core (BALANCE.md option (a)): same conv3x3_i8_gate call as the unsplit gate,
    CONV3X3_COUT at the call site's compile flags, x read at its own channel half's offset.
    bracket=True: same event0()/event1() convention as _shim (off by default, byte-identical)."""
    args = "int8_t *l0, int8_t *l1, int8_t *l2, int8_t *p, int8_t *o, int32_t check"
    call = (f"conv3x3_i8_gate(l0, l1, l2, l1 + {xoff}, p, o, {w}, check, {p['pre']}, {p['shift']}, "
           f"0, {w}, {p['ga']}, {p['gb']}, {p['gs1']}, {p['gc']}, {p['gs2']});")
    ev0, ev1 = ("  event0();\n", "  event1();\n") if bracket else ("", "")
    path.write_text(f'#include <stdint.h>\n#include "{KDIR / "conv3x3_u8.cc"}"\n'
                    f'extern "C" void {sym}({args}) {{\n{ev0}  {call}\n{ev1}}}\n')
    return path


# kinds split_gate accepts, generalizing BALANCE.md's gate-only mechanism (Phase 2 M1, spec
# 2026-09-28-span-frame-width-design.md S5-iii) to b1c1/b1c2 (silu16/silu_i16): same conv3x3_core
# COUT-block-pair split, same net_layout.split_gate_params blob slice (already kind-agnostic).
SPLIT_KINDS = {"gate", "silu16", "silu_i16"}


def _split_layout(kind, w, lo_channels):
    """Byte split of a stage's own COUT-indexed output segment at `lo_channels`, plus the LOCAL
    offset in the hi half's own buffer where an x-forward copy lands (None: no forward on this
    kind). gate's own segment is out_bytes (int8, h); silu16's is the int16 SiLU result (2h,
    x rides after at x_out=2h); silu_i16's is its int8 conv result (h, x rides after at
    x_out=h) -- both silu kinds forward x on the hi half only, same as the unsplit x_out offset,
    now local to hi's own (smaller) buffer."""
    h = NL.half(w)
    lo, hi = lo_channels, NL.C - lo_channels
    if kind == "gate":
        return h * lo // NL.C, h * hi // NL.C, None
    if kind == "silu16":
        lo_b, hi_b = 2 * h * lo // NL.C, 2 * h * hi // NL.C
        return lo_b, hi_b + h, hi_b
    if kind == "silu_i16":
        lo_b, hi_b = h * lo // NL.C, h * hi // NL.C
        return lo_b, hi_b + h, hi_b
    raise ValueError(f"split_gate unsupported for kind={kind}")


def _shim_split_half(path, sym, kind, p, w, xoff=None, fwd_offset=None, bracket=False):
    """Half-COUT core for any SPLIT_KINDS member. xoff: gate's own extra x-arg offset (gate
    only, None otherwise). fwd_offset: local offset in this half's own output buffer for the
    x-forward copy (silu16/silu_i16's hi half only; None elsewhere). Same event0()/event1()
    convention as _shim_gate_half."""
    lay = NL.layout(kind, w)
    c = f"{w}, check, {p['pre']}, {p['shift']}, 0, {w}"
    s = "(const int16_t *)"
    call = {
        "silu16": lambda: f"conv3x3_i8_lut16(l0, l1, l2, p, (int16_t *)o, {c});",
        "silu_i16": lambda: f"conv3x3_i16i8_lut({s}l0, {s}l1, {s}l2, p, o, {c});",
        "gate": lambda: (f"conv3x3_i8_gate(l0, l1, l2, l1 + {xoff}, p, o, {c}, {p['ga']}, "
                         f"{p['gb']}, {p['gs1']}, {p['gc']}, {p['gs2']});"),
    }[kind]()
    fwd = ""
    if fwd_offset is not None:
        fwd = (f"  for (int i = 0; i < {NL.half(w)}; i += 64)\n"
               f"    aie::store_v(o + {fwd_offset} + i, aie::load_v<64>(l1 + {lay.x_in} + i));\n")
    args = "int8_t *l0, int8_t *l1, int8_t *l2, int8_t *p, int8_t *o, int32_t check"
    ev0, ev1 = ("  event0();\n", "  event1();\n") if bracket else ("", "")
    path.write_text(f'#include <stdint.h>\n#include "{KDIR / "conv3x3_u8.cc"}"\n'
                    f'extern "C" void {sym}({args}) {{\n{ev0}  {call}\n{fwd}{ev1}}}\n')
    return path


def _flags_gate_half(incs, cout):
    f = ["-DCONV3X3_CIN=48", f"-DCONV3X3_COUT={cout}"]
    for i, inc in enumerate(incs):
        f.append(f'-DCONV3X3_LUT{"" if i == 0 else "2"}_INC="{inc}"')
    return f


def _flags(kind, incs):
    if kind == "cat":
        return ["-DCONV1X1_NSRC=4", "-DCONV1X1_CSRC=48", "-DCONV1X1_COUT=48"]
    f = [f"-DCONV3X3_CIN={8 if kind == 'conv1' else 48}",
         f"-DCONV3X3_COUT={16 if kind == 'up' else 48}"]
    for i, inc in enumerate(incs):
        f.append(f'-DCONV3X3_LUT{"" if i == 0 else "2"}_INC="{inc}"')
    return f


def build(w, h, NP, gen, upto="up", depths=None, stacks=None, tag="spannet",
         trace_stages=None, trace_config=None, coretile_events=None, egress_shim_col=1,
         main_depth=None, skip_slack=None, prod_depth=None, cat_cons_depth=None,
         data_sizes=None, split_gate=None, skip_cons_depths=None, join_tile=None,
         b1_int8=False):
    """NP: span_int.Span.net_params(). Returns an iron.jit callable (x, wts, y): x from
    net_layout.conv1_rows (as int8), wts from net_layout.weights_blob, y gets `upto`'s rows.

    trace_stages/trace_config: same two-half mechanism as bricklib._build_streamed_traced --
    trace_config (a TraceConfig, REQUIRED if trace_stages is given) is passed to iron.jit so it
    injects a `trace_size` compile kwarg into `design`; that call bakes Program.enable_trace on
    exactly the Workers named in trace_stages, bracketed with event0()/event1() in their shim.
    None (default) leaves every shim byte-identical to production.

    main_depth: override for MAIN_DEPTH (default None -> the module constant, 4). `windowed()`
    holds a 3-row sliding window and releases 1/iteration, so depth=4 leaves exactly 1 free
    slot for the upstream producer -- diagnostic for the per-row lockstep hypothesis in
    TRACE_RESULTS.md. Does not touch PROD_DEPTH (skip-broadcast producer depth) or the
    per-stage DEPTH overrides (e.g. b1c1=3), which still take precedence via `depths=`.

    skip_slack: override for net_layout.SKIP_SLACK (default None -> the module constant, 2),
    used only to size the join ring (`skip_depth(src) = rows_ahead(src) + skip_slack`) -- the
    skip-ring latency-throttle hypothesis in TRACE_RESULTS.md. Does not touch NL.SKIP_SLACK
    itself (net_layout stays byte-identical), so the module default is unaffected.

    prod_depth: override for PROD_DEPTH (default None -> the module constant, 2), the CORE-side
    producer depth of a skip source's output ObjectFifo (conv_1/conv_2/b1c3/b6c1). That fifo has
    TWO consumers -- the next main-path core (depth=main_depth) and the join's MemTile ring
    (depth=skip_depths[src]) -- and a broadcast producer can only run as far ahead as its
    slowest consumer allows; prod_depth is the producer's own buffering against that.

    cat_cons_depth: override for conv_cat's own input depth (default None -> 2), the `rowwise()`
    worker's `f_cat.cons(depth=...)` handle that reads the join ring. Independent of skip_slack
    (which sizes the ring itself, not conv_cat's read-ahead into it).

    data_sizes: per-stage override for Worker(data_size=...), the explicit static-data (constant
    array/LUT table) reservation aiecc's own error suggests when buffer allocation leaves too
    little room for a core's LUT: "aiecc: core main_core_0_3 needs space for 4160 bytes of static
    data ... but it may fit if you reserve it explicitly." Merged over DATA_SIZES (default {},
    currently {"b1c1": 4160} -- required for DEPTH["b1c1"]=4 to fit L1 at W=32 when b1c1 runs
    silu16's two hi/lo tables). DATA_SIZES itself is skipped when b1_int8 (silu_x's one int8
    table needs far less; a 4160 B reservation sized for the two-table int16 LUT would only take
    back the L1 int8 is meant to free -- INT8_BLOCK1.md).

    split_gate: stage names (kind in SPLIT_KINDS: gate, silu16, silu_i16) to split onto two cores
    by output channel
    (BALANCE.md option (a), GATE_SPLIT_LO/NL.C-GATE_SPLIT_LO channels).

    skip_cons_depths: per-SOURCE override (keyed by the skip source's own name, e.g. "conv_1") for
    the next main-path core's consumer depth on that source's broadcast fifo -- normally hardcoded
    to `main_depth` for every `prev in skips` hop. Lets a skip source's own main-path hop run at a
    shallower depth than the network-wide main_depth (e.g. to free L1 on the DOWNSTREAM consumer's
    tile), at the cost of that hop alternating with its source if the source's own compute is
    small enough not to become the new ceiling -- check with a trace, do not assume. Merged over
    SKIP_CONS_DEPTHS (default {}, currently {"conv_1": 3} -- the other half of DEPTH["b1c1"]=4's
    L1 fix).

    join_tile: Tile override for the join's own MemTile (default None -> AnyMemTile, the
    unconstrained placer choice that landed on the SAME MemTile as weight group 3 -- see
    TRACE_RESULTS.md's Phase 1a MemTile(4,1) section). Phase 1e's targeted experiment: pin the
    join off that shared tile onto a dedicated one.

    b1_int8: b1c1/b1c2 run blocks 2-6's own int8 kinds (silu_x/silu) instead of silu16/silu_i16
    (INT8_BLOCK1.md). NP must have been produced by span_int.Span(b1_mode="int8")."""
    main_depth = MAIN_DEPTH if main_depth is None else main_depth
    skip_slack = NL.SKIP_SLACK if skip_slack is None else skip_slack
    prod_depth = PROD_DEPTH if prod_depth is None else prod_depth
    cat_cons_depth = 2 if cat_cons_depth is None else cat_cons_depth
    data_sizes = {**({} if b1_int8 else DATA_SIZES), **(data_sizes or {})}
    split_gate = frozenset(split_gate or ())
    skip_cons_depths = {**SKIP_CONS_DEPTHS, **(skip_cons_depths or {})}
    assert w % 16 == 0, "conv3x3_u8.cc needs width % 16 == 0 (and the x copy, whole 64-byte vectors)"
    gen = Path(gen)
    gen.mkdir(parents=True, exist_ok=True)
    g = _golden()
    names = NL.stage_names(upto)
    kind = NL.stages(b1_int8)
    for n in split_gate:
        assert kind[n] in SPLIT_KINDS, f"split_gate unsupported for kind={kind[n]} ({n})"
    lay = {n: NL.layout(kind[n], w) for n in names}
    stacks = {**STACK, **(stacks or {})}
    depth = {n: main_depth for n in names} | DEPTH | (depths or {})
    joined = "conv_cat" in names
    skips = {s for s, _ in NL.CAT_SOURCES} if joined else set()

    trace_stages = set(trace_stages or [])
    texts = [(KDIR / "conv3x3_u8.cc").read_text(), (CAT_DIR / "conv1x1_cat.cc").read_text()]
    spec, core_names, half_of, split_bytes, split_cout = {}, [], {}, {}, {}
    for n in names:
        if n in split_gate:
            npl, nph = NL.split_gate_params(NP[n], lo_channels=GATE_SPLIT_LO)
            lo_bytes, hi_bytes, fwd_off = _split_layout(kind[n], w, GATE_SPLIT_LO)
            split_bytes[n] = (lo_bytes, hi_bytes)
            is_gate = kind[n] == "gate"
            xo_lo = lay[n].x_in if is_gate else None
            xo_hi = (lay[n].x_in + lo_bytes) if is_gate else None
            halves = ((f"{n}_lo", npl, xo_lo, None, GATE_SPLIT_LO),
                     (f"{n}_hi", nph, xo_hi, fwd_off, NL.C - GATE_SPLIT_LO))
            for suffix, npi, xo, fo, cout in halves:
                sym = f"{tag}_{suffix}_w{w}"
                incs = [g.lut_inc(t, gen / f"{sym}_t{i}.inc") for i, t in enumerate(npi.get("tables", []))]
                shim = _shim_split_half(gen / f"{sym}.cc", sym, kind[n], npi, w, xoff=xo,
                                        fwd_offset=fo, bracket=suffix in trace_stages)
                spec[suffix] = (sym, shim, incs, npi)
                texts += [shim.read_text()] + [Path(i).read_text() for i in incs]
                core_names.append(suffix)
                half_of[suffix] = n
                split_cout[suffix] = cout
        else:
            sym = f"{tag}_{n}_w{w}"
            incs = [g.lut_inc(t, gen / f"{sym}_t{i}.inc") for i, t in enumerate(NP[n].get("tables", []))]
            shim = _shim(gen / f"{sym}.cc", sym, kind[n], NP[n], w, bracket=n in trace_stages)
            spec[n] = (sym, shim, incs, NP[n])
            texts += [shim.read_text()] + [Path(i).read_text() for i in incs]
            core_names.append(n)
    groups = NL.weight_groups(core_names)
    skip_depths = {s: NL.rows_ahead(s) + skip_slack for s in sorted(skips)}
    digest = hashlib.sha256("".join(texts).encode() + repr(
        (w, h, upto, sorted(stacks.items()), sorted(depth.items()), groups, skip_depths,
         main_depth, prod_depth, cat_cons_depth, sorted(data_sizes.items()),
         sorted(skip_cons_depths.items()), sorted(trace_stages), tuple(coretile_events or ()),
         egress_shim_col, sorted(split_gate),
         (join_tile.col, join_tile.row) if join_tile is not None else None, b1_int8,
         )).encode()).hexdigest()[:12]
    base = _aie_api_include() + [f"-DSPAN_NET_DIGEST={digest}"]
    plen = {cn: spec[cn][3]["blob"].size for cn in core_names}
    wtotal = sum(plen.values())
    x_row, y_row = lay["conv_1"].in_bytes, lay[names[-1]].out_bytes

    def ty(k):
        return np.ndarray[(k,), np.dtype[np.int8]]

    def windowed(fi, fp, fo, k):
        ep = fp.acquire(1)
        e = fi.acquire(2)
        o = fo.acquire(1)
        k(e[0], e[0], e[1], ep, o, 0)
        fo.release(1)
        for _ in range_(h - 2):
            e = fi.acquire(3)
            o = fo.acquire(1)
            k(e[0], e[1], e[2], ep, o, 1)
            fi.release(1)
            fo.release(1)
        e = fi.acquire(2)
        o = fo.acquire(1)
        k(e[0], e[1], e[1], ep, o, 2)
        fi.release(2)
        fo.release(1)
        fp.release(1)

    def padded(fi, fp, fo, k):      # conv_1: h+2 host rows, every output a middle row
        ep = fp.acquire(1)
        for _ in range_(h):
            e = fi.acquire(3)
            o = fo.acquire(1)
            k(e[0], e[1], e[2], ep, o, 1)
            fi.release(1)
            fo.release(1)
        fi.acquire(2)
        fi.release(2)
        fp.release(1)

    def rowwise(fi, fp, fo, k):     # conv_cat: one joined element per row
        ep = fp.acquire(1)
        for _ in range_(h):
            e = fi.acquire(1)
            o = fo.acquire(1)
            k(e, ep, o)
            fi.release(1)
            fo.release(1)
        fp.release(1)

    def design(x: In, wts: In, y: Out, *, trace_size: CompileTime[int] = 0):
        kern = {}
        for cn in core_names:
            sym, shim, incs, npi = spec[cn]
            if cn in half_of:
                n = half_of[cn]
                out_bytes = split_bytes[n][0 if cn.endswith("_lo") else 1]
                args = [ty(lay[n].in_bytes)] * 3 + [ty(plen[cn]), ty(out_bytes), np.int32]
                kern[cn] = ExternalFunction(sym, source_file=str(shim), arg_types=args,
                                            compile_flags=base + _flags_gate_half(incs, split_cout[cn]))
                continue
            n = cn
            args = ([ty(lay[n].in_bytes), ty(plen[n]), ty(lay[n].out_bytes)] if kind[n] == "cat"
                    else [ty(lay[n].in_bytes)] * 3 + [ty(plen[n]), ty(lay[n].out_bytes), np.int32])
            kern[n] = ExternalFunction(sym, source_file=str(shim), arg_types=args,
                                       compile_flags=base + _flags(kind[n], incs))
        p_fifo, w_fifos = {}, []
        for gi, grp in enumerate(groups):
            wf = ObjectFifo(ty(sum(plen[n] for n in grp)), name=f"w{gi}", depth=1)
            offs = np.cumsum([0] + [plen[n] for n in grp[:-1]]).tolist()
            subs = wf.cons().split(offs, obj_types=[ty(plen[n]) for n in grp],
                                   depths=[1] * len(grp), names=[f"p_{n}" for n in grp])
            p_fifo.update(zip(grp, subs))
            w_fifos.append(wf)
        f_in = ObjectFifo(ty(x_row), name="x_in", depth=main_depth)
        out, f_cat = {}, None
        if joined:
            # a join's inputs and output share one MemTile pool, sized by the output fifo, so
            # it must hold the deepest skip
            f_cat = ObjectFifo(ty(lay["conv_cat"].in_bytes), name="cat_in",
                               depth=max(skip_depths.values()), iterate_bds=True)
            subs = f_cat.prod().join(
                [o * NL.half(w) for _, o in NL.CAT_SOURCES],
                obj_types=[ty(lay[s].out_bytes) for s, _ in NL.CAT_SOURCES],
                depths=[skip_depths[s] for s, _ in NL.CAT_SOURCES],
                names=[f"{s}_skip" for s, _ in NL.CAT_SOURCES],
                tile=join_tile if join_tile is not None else AnyMemTile)
            out.update({s: f for (s, _), f in zip(NL.CAT_SOURCES, subs)})
        for n in names:
            if n not in out:
                out[n] = ObjectFifo(ty(lay[n].out_bytes), name=f"{n}_out", depth=depth[n])
        workers, worker_by_name = [], {}
        for i, n in enumerate(names):
            if n in split_gate:
                prev = names[i - 1]
                fi_depth = main_depth if prev in skips else depth[prev]
                lo_bytes, hi_bytes = split_bytes[n]
                lo, hi = out[n].prod().join(
                    [0, lo_bytes], obj_types=[ty(lo_bytes), ty(hi_bytes)], depths=[depth[n]] * 2,
                    names=[f"{n}_lo_j", f"{n}_hi_j"])
                for suffix, sub in ((f"{n}_lo", lo), (f"{n}_hi", hi)):
                    wk = Worker(windowed, fn_args=[out[prev].cons(fi_depth),
                               p_fifo[suffix].cons(), sub.prod(), kern[suffix]],
                               stack_size=stacks["gate_half"], data_size=data_sizes.get(suffix))
                    workers.append(wk)
                    worker_by_name[suffix] = wk
                continue
            if kind[n] == "cat":
                fi, body = f_cat.cons(depth=cat_cons_depth), rowwise
            elif i == 0:
                fi, body = f_in.cons(main_depth), padded
            else:
                prev = names[i - 1]
                fi = out[prev].cons(skip_cons_depths.get(prev, main_depth) if prev in skips
                                   else depth[prev])
                body = windowed
            fo = out[n].prod(depth=prod_depth) if n in skips else out[n].prod()
            wk = Worker(body, fn_args=[fi, p_fifo[n].cons(), fo, kern[n]],
                       stack_size=stacks[kind[n]], data_size=data_sizes.get(n))
            workers.append(wk)
            worker_by_name[n] = wk

        w_taps, off = [], 0
        for grp in groups:
            size = sum(plen[n] for n in grp)
            w_taps.append(TensorAccessPattern((1, wtotal), off, [1, 1, 1, size], [0, 0, 0, 1]))
            off += size
        x_tap = TensorTiler2D.group_tiler((h + 2, x_row), (1, x_row), (h + 2, 1))[0]
        y_tap = TensorTiler2D.group_tiler((h, y_row), (1, y_row), (h, 1))[0]

        def sequence(x_, w_, y_, hx, hy, *hw):
            for hwi, tap in zip(hw, w_taps):
                hwi.fill(w_, tap)
            hx.fill(x_, x_tap)
            hy.drain(y_, y_tap, wait=True)

        rt = Runtime(sequence, [ty((h + 2) * x_row), ty(wtotal), ty(h * y_row), f_in.prod(),
                                out[names[-1]].cons()] + [wf.prod() for wf in w_fifos])
        prog = Program(iron.get_current_device(), rt, workers=workers)
        if trace_size:
            traced = [worker_by_name[n] for n in trace_stages]
            prog.enable_trace(trace_size=trace_size, workers=traced,
                              coretile_events=coretile_events, egress_shim_col=egress_shim_col)
        return prog.resolve_program()

    design.__name__ = design.__qualname__ = f"{tag}_{digest}"
    return iron.jit(design, use_cache=True, trace_config=trace_config)
