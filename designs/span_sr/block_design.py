"""One SPAN attention block as one IRON design: c1 (conv+SiLU) -> c2 (conv+SiLU) -> c3 (conv+gate).

Rows are channel-blocked [C/8][width+16][8] int8 (conv2d-3x3-u8's layout). Between the block's
cores every row element is [conv output | x]: each core copies the block input x from its window's
centre row into its output's second half, so the gate core reads x in-band and needs only two DMA
inputs (activations, weights). Weights enter once through a depth-1 fifo and stay resident.
"""
import hashlib
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D

KDIR = Path(__file__).resolve().parents[2] / "aie_kernels" / "conv2d-3x3-u8"
C = 48


def row_bytes(width, ch=C):
    return (width + 16) * ch


def _golden():
    import importlib.util
    spec = importlib.util.spec_from_file_location("c3g", KDIR / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _aie_api_include():
    """-I for aie_api when the active toolchain instance does not expose it implicitly."""
    import aie
    for inst in Path(aie.__file__).resolve().parents:
        for cand in (inst / "include", inst / "src" / "third_party" / "aie_api" / "include"):
            if (cand / "aie_api" / "aie.hpp").exists():
                return [f"-I{cand}"]
    return []


def _shim(gen, sym, core, width, p):
    """Per-core C++ entry: (l0, l1, l2, params, out, check)."""
    half = row_bytes(width)
    if core == "c3":
        call = (f"conv3x3_i8_gate(l0, l1, l2, l1 + {half}, p, o, {width}, check, {p['pre']}, "
                f"{p['shift']}, 0, {width}, {p['ga']}, {p['gb']}, {p['gs1']}, {p['gc']}, "
                f"{p['gs2']});")
        fwd = ""
    else:
        xoff = 0 if core == "c1" else half      # c1's input IS x; c2's carries it second
        call = (f"conv3x3_i8_lut(l0, l1, l2, p, o, {width}, check, {p['pre']}, {p['shift']}, "
                f"0, {width});")
        fwd = (f"  for (int i = 0; i < {half}; i += 64)\n"
               f"    aie::store_v(o + {half} + i, aie::load_v<64>(l1 + {xoff} + i));\n")
    path = gen / f"{sym}.cc"
    path.write_text(
        f'#include <stdint.h>\n#include "{KDIR / "conv3x3_u8.cc"}"\n'
        f'extern "C" void {sym}(int8_t *l0, int8_t *l1, int8_t *l2, int8_t *p, int8_t *o,'
        f' int32_t check) {{\n  {call}\n{fwd}}}\n')
    return path


def build(width, height, P, gen, stacks=None, tag="spanblk"):
    """P: span_int.Span.core_params(i). Returns an iron.jit callable (x, p1, p2, p3, y)."""
    gen = Path(gen)
    gen.mkdir(parents=True, exist_ok=True)
    stacks = stacks or {"c1": 2560, "c2": 2560, "c3": 3584}
    half = row_bytes(width)
    # the JIT cache keys on the design name and the .o cache on the shim text: both must see
    # the brick source, which the shim only #includes
    src = [(KDIR / "conv3x3_u8.cc").read_text()]
    kern_specs = {}
    for core in ("c1", "c2", "c3"):
        sym = f"{tag}_{core}_w{width}"
        inc = _golden().lut_inc(P[core]["table"], gen / f"{sym}.inc")
        shim = _shim(gen, sym, core, width, P[core])
        kern_specs[core] = (sym, shim, inc)
        src += [shim.read_text(), Path(inc).read_text()]
    digest = hashlib.sha256("".join(src).encode() + repr((width, height, stacks)).encode()
                            ).hexdigest()[:12]
    base_flags = _aie_api_include() + ["-DCONV3X3_CIN=48", "-DCONV3X3_COUT=48",
                                       f"-DSPAN_BLOCK_DIGEST={digest}"]
    in_w = {"c1": half, "c2": 2 * half, "c3": 2 * half}
    out_w = {"c1": 2 * half, "c2": 2 * half, "c3": half}
    plen = {k: P[k]["blob"].size for k in P}

    def design(x: In, p1: In, p2: In, p3: In, y: Out):
        ty = lambda n: np.ndarray[(n,), np.dtype[np.int8]]
        kern = {}
        for core, (sym, shim, inc) in kern_specs.items():
            kern[core] = ExternalFunction(
                sym, source_file=str(shim),
                arg_types=[ty(in_w[core])] * 3 + [ty(plen[core]), ty(out_w[core]), np.int32],
                compile_flags=base_flags + [f'-DCONV3X3_LUT_INC="{inc}"'])
        f_in = ObjectFifo(ty(half), name="x_in")
        # depth 4 on both ends: between neighbouring cores the fifo lowers to shared memory with
        # the PRODUCER's depth, and a 3-row window over 2 buffers never acquires (device hang)
        f_12 = ObjectFifo(ty(2 * half), name="c1_c2", depth=4)
        f_23 = ObjectFifo(ty(2 * half), name="c2_c3", depth=4)
        f_out = ObjectFifo(ty(half), name="y_out")
        f_p = {k: ObjectFifo(ty(plen[k]), name=f"p_{k}", depth=1) for k in ("c1", "c2", "c3")}

        def conv_rows(fi, fp, fo, k):
            ep = fp.acquire(1)
            e = fi.acquire(2)                       # top row: rows 0, 1
            o = fo.acquire(1)
            k(e[0], e[0], e[1], ep, o, 0)
            fo.release(1)
            for _ in range_(height - 2):             # rows 1 .. height-2
                e = fi.acquire(3)
                o = fo.acquire(1)
                k(e[0], e[1], e[2], ep, o, 1)
                fi.release(1)
                fo.release(1)
            e = fi.acquire(2)                       # bottom row
            o = fo.acquire(1)
            k(e[0], e[1], e[1], ep, o, 2)
            fi.release(2)
            fo.release(1)
            fp.release(1)

        workers = [
            Worker(conv_rows, fn_args=[f_in.cons(4), f_p["c1"].cons(), f_12.prod(), kern["c1"]],
                   stack_size=stacks["c1"]),
            Worker(conv_rows, fn_args=[f_12.cons(4), f_p["c2"].cons(), f_23.prod(), kern["c2"]],
                   stack_size=stacks["c2"]),
            Worker(conv_rows, fn_args=[f_23.cons(4), f_p["c3"].cons(), f_out.prod(), kern["c3"]],
                   stack_size=stacks["c3"]),
        ]
        row_tap = TensorTiler2D.group_tiler((height, half), (1, half), (height, 1))[0]
        p_tap = {k: TensorTiler2D.group_tiler((1, plen[k]), (1, plen[k]), (1, 1))[0] for k in plen}

        def sequence(x_, p1_, p2_, p3_, y_, hx, h1, h2, h3, hy):
            h1.fill(p1_, p_tap["c1"])
            h2.fill(p2_, p_tap["c2"])
            h3.fill(p3_, p_tap["c3"])
            hx.fill(x_, row_tap)
            hy.drain(y_, row_tap, wait=True)

        rt = Runtime(sequence, [ty(height * half), ty(plen["c1"]), ty(plen["c2"]), ty(plen["c3"]),
                                ty(height * half), f_in.prod(), f_p["c1"].prod(), f_p["c2"].prod(),
                                f_p["c3"].prod(), f_out.cons()])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

    design.__name__ = design.__qualname__ = f"{tag}_{digest}"
    return iron.jit(design, use_cache=True)
