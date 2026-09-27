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
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

import net_layout as NL
from block_design import KDIR, _aie_api_include, _golden

CAT_DIR = KDIR.parent / "conv2d-1x1-cat"

# stack reservation per kind; aiecc measures each core and fails naming the bytes it needs
STACK = {"conv1": 2048, "silu16": 3584, "silu_i16": 3584, "silu_x": 2560, "silu": 2560,
         "gate": 3584, "plain": 2048, "cat": 2048, "up": 2048}
# objectFIFO depth of a stage's output (both ends, see the shared-pool trap). b1c1's rows are
# three halves wide: four of them do not fit in L1 beside its weights and its input window.
DEPTH = {"b1c1": 3}
MAIN_DEPTH = 4   # a 3-row window plus the row being written
PROD_DEPTH = 2   # producer side of a fifo that leaves the core by DMA (skip broadcasts)


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


def _flags(kind, incs):
    if kind == "cat":
        return ["-DCONV1X1_NSRC=4", "-DCONV1X1_CSRC=48", "-DCONV1X1_COUT=48"]
    f = [f"-DCONV3X3_CIN={8 if kind == 'conv1' else 48}",
         f"-DCONV3X3_COUT={16 if kind == 'up' else 48}"]
    for i, inc in enumerate(incs):
        f.append(f'-DCONV3X3_LUT{"" if i == 0 else "2"}_INC="{inc}"')
    return f


def build(w, h, NP, gen, upto="up", depths=None, stacks=None, tag="spannet",
         trace_stages=None, trace_config=None, coretile_events=None, egress_shim_col=1):
    """NP: span_int.Span.net_params(). Returns an iron.jit callable (x, wts, y): x from
    net_layout.conv1_rows (as int8), wts from net_layout.weights_blob, y gets `upto`'s rows.

    trace_stages/trace_config: same two-half mechanism as bricklib._build_streamed_traced --
    trace_config (a TraceConfig, REQUIRED if trace_stages is given) is passed to iron.jit so it
    injects a `trace_size` compile kwarg into `design`; that call bakes Program.enable_trace on
    exactly the Workers named in trace_stages, bracketed with event0()/event1() in their shim.
    None (default) leaves every shim byte-identical to production."""
    assert w % 16 == 0, "conv3x3_u8.cc needs width % 16 == 0 (and the x copy, whole 64-byte vectors)"
    gen = Path(gen)
    gen.mkdir(parents=True, exist_ok=True)
    g = _golden()
    names = NL.stage_names(upto)
    kind = dict(NL.STAGES)
    lay = {n: NL.layout(kind[n], w) for n in names}
    stacks = {**STACK, **(stacks or {})}
    depth = {n: MAIN_DEPTH for n in names} | DEPTH | (depths or {})
    joined = "conv_cat" in names
    skips = {s for s, _ in NL.CAT_SOURCES} if joined else set()

    trace_stages = set(trace_stages or [])
    texts = [(KDIR / "conv3x3_u8.cc").read_text(), (CAT_DIR / "conv1x1_cat.cc").read_text()]
    spec = {}
    for n in names:
        sym = f"{tag}_{n}_w{w}"
        incs = [g.lut_inc(t, gen / f"{sym}_t{i}.inc") for i, t in enumerate(NP[n].get("tables", []))]
        shim = _shim(gen / f"{sym}.cc", sym, kind[n], NP[n], w, bracket=n in trace_stages)
        spec[n] = (sym, shim, incs)
        texts += [shim.read_text()] + [Path(i).read_text() for i in incs]
    groups = NL.weight_groups(names)
    skip_depths = {s: NL.skip_depth(s) for s in sorted(skips)}
    digest = hashlib.sha256("".join(texts).encode() + repr(
        (w, h, upto, sorted(stacks.items()), sorted(depth.items()), groups, skip_depths,
         MAIN_DEPTH, PROD_DEPTH, sorted(trace_stages), tuple(coretile_events or ()),
         egress_shim_col)).encode()).hexdigest()[:12]
    base = _aie_api_include() + [f"-DSPAN_NET_DIGEST={digest}"]
    plen = {n: NP[n]["blob"].size for n in names}
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
        for n in names:
            sym, shim, incs = spec[n]
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
        f_in = ObjectFifo(ty(x_row), name="x_in", depth=MAIN_DEPTH)
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
                names=[f"{s}_skip" for s, _ in NL.CAT_SOURCES])
            out.update({s: f for (s, _), f in zip(NL.CAT_SOURCES, subs)})
        for n in names:
            if n not in out:
                out[n] = ObjectFifo(ty(lay[n].out_bytes), name=f"{n}_out", depth=depth[n])
        workers = []
        for i, n in enumerate(names):
            if kind[n] == "cat":
                fi, body = f_cat.cons(depth=2), rowwise
            elif i == 0:
                fi, body = f_in.cons(MAIN_DEPTH), padded
            else:
                prev = names[i - 1]
                fi = out[prev].cons(MAIN_DEPTH if prev in skips else depth[prev])
                body = windowed
            fo = out[n].prod(depth=PROD_DEPTH) if n in skips else out[n].prod()
            workers.append(Worker(body, fn_args=[fi, p_fifo[n].cons(), fo, kern[n]],
                                  stack_size=stacks[kind[n]]))

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
            traced = [workers[names.index(n)] for n in trace_stages]
            prog.enable_trace(trace_size=trace_size, workers=traced,
                              coretile_events=coretile_events, egress_shim_col=egress_shim_col)
        return prog.resolve_program()

    design.__name__ = design.__qualname__ = f"{tag}_{digest}"
    return iron.jit(design, use_cache=True, trace_config=trace_config)
