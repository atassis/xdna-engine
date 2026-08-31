#!/usr/bin/env python3
"""Shared rail for brick device-verify (brick-wave1-device-verify).

Generalizes the proven rmsnorm rail: a thin pure-buffer verify-shim (scalars baked via
-D) #include's the brick .cc; an @iron.jit design streams rows of a [m, in_cols] input
through the kernel on aie2p (optionally with ONE packed resident const buffer, to stay
within a core tile's 2-input DMA channels); rel-L2 vs the numpy golden gates it.
Run-twice self-check guards the CLFLUSH read race.
"""
import hashlib
import re
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D

# The JIT cache keyed only what it was told about, so editing a kernel .cc did not
# invalidate it and a stale .o was linked -- hence use_cache=False everywhere here,
# at ~150 ms of recompilation per design() call against ~0.19 ms cached. Set
# BRICK_JIT_CACHE=1 to enable it against a toolchain that records what a build
# actually consumed; do NOT enable it otherwise, and never without re-running the
# perturb check (edit a kernel, confirm the gate moves).
import os as _os

_JIT_CACHE = _os.environ.get("BRICK_JIT_CACHE") == "1"

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)


def _shim_digest(shim):
    """Hash the shim and every local header it pulls in, transitively.

    This is ONE PART of the cache key, not the whole of it -- see `_design_key`, which also
    carries shapes, dtypes and flags. The two guard different failures and neither subsumes
    the other: `_design_key` alone serves one shape's output BO to every later shape, while
    this alone lets two different shapes with identical source collide.

    The original note on why the source half is needed:

    A design name that CHANGES whenever the build inputs change.

    `extra` carries build inputs that are NOT in the shim text or the compile flags -- currently the
    resident objectFIFO depth, which changes the generated design and therefore must change the key.
    Anything added to the design that is neither source nor a compile flag belongs here, or the
    cache silently serves an artifact built with different structure.

    THE CACHE KEY IS THIS NAME, NOT THE SHIM TEXT. That is the opposite of what this harness
    documented for weeks, and it silently invalidated an unknown number of runs. Measured
    head-to-head: two builds in one process, same brick, same shim body apart from a comment, one
    bound to the symbol `sin_verify` (reused from earlier runs) and one to `sin_verify_<nonce>` --
    the reused name returned a stale xclbin at rel-L2 2.052e+01 with output in [-110.9, +124.96],
    the fresh name returned rel-L2 1.275e-04 in [-1, 1]. Passing `use_cache=False` does not disable
    that layer; only a different name does.

    The cost of getting this wrong is not a failed run, it is a PASSING one: a verify whose symbol
    never changes keeps re-reporting whatever the first build of that name produced, so edits to
    the kernel look like no-ops and a brick can be recorded green on code that no longer exists.
    The per-shim `// cachebust <ms>` markers were written to prevent exactly this and could not,
    since they only ever changed the text.

    So key the name on the real inputs: the shim, every local header it pulls in (transitively --
    the brick .cc lives behind one), and the compile flags. Identical inputs still reuse the
    artifact, which is the property that made caching worth having.
    """
    h = hashlib.sha256()
    seen, queue = set(), [Path(shim)]
    while queue:
        f = queue.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        try:
            src = f.read_text()
        except OSError:
            continue
        h.update(src.encode())
        for inc in re.findall(r'#include\s+"([^"]+)"', src):
            queue.append((f.parent / inc).resolve())
    return h.hexdigest()[:16]


def _aie_api_include():
    """Resolve the aie_api include dir and return it as a -I flag, or nothing if the
    active toolchain instance already exposes the headers.

    Every brick .cc opens with `#include <aie_api/aie.hpp>`, but whether that resolves is
    a property of WHICH instance is active: instances built with an `include/` tree carry
    it implicitly, while a leaner one only symlinks `src/third_party/aie_api` and the
    Peano invocation never sees it (fatal: 'aie_api/aie.hpp' file not found). Deriving the
    path from the live `aie` package keeps the harness instance-independent rather than
    tying it to whichever instance happened to be pinned when it was written.
    """
    import aie

    inst = Path(aie.__file__).resolve().parent.parent.parent  # <instance>/python/aie/__init__.py
    for cand in (inst / "include", inst / "src" / "third_party" / "aie_api" / "include"):
        if (cand / "aie_api" / "aie.hpp").exists():
            return [f"-I{cand}"]
    return []


_AIE_API_INC = _aie_api_include()


def _npty(shape, dt):
    return np.ndarray[shape, np.dtype[dt]]


def _include_closure_digest(source, compile_flags):
    """Hash the CONTENT of everything `source` transitively #includes by quoted path.

    ExternalFunction._content_digest hashes the source file's own text plus the include
    DIRECTORIES' mtimes -- and a directory mtime does not move when a file inside it is edited in
    place. So editing an #included table (rope_lut_tables.inc, a *_tables.inc, a shared header) is
    invisible to the JIT cache key and the previous xclbin is served. That cost two sessions once:
    a fix "did not work" because the build never happened.

    compile_flags IS in the digest, so returning this as a -D makes those edits visible without
    touching upstream. Quoted includes only -- system/angle-bracket headers come from the toolchain
    and are already pinned by toolchain.lock.
    """
    inc_dirs = [f[2:] for f in compile_flags if f.startswith("-I")]
    seen, pending, h = set(), [Path(source)], hashlib.sha256()
    while pending:
        cur = pending.pop()
        try:
            key = cur.resolve()
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        try:
            text = key.read_text()
        except OSError:
            continue
        h.update(str(key).encode())
        h.update(text.encode())
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.M):
            cand = [key.parent / inc] + [Path(d) / inc for d in inc_dirs]
            for c in cand:
                if c.exists():
                    pending.append(c)
                    break
    return h.hexdigest()[:16]


def _design_key(symbol, *parts):
    """JIT cache key for a built design.

    The design's __name__ IS the cache key, so it must carry everything that changes the PROGRAM --
    buffer sizes, dtypes, compile flags -- not just the kernel symbol. Keying on the symbol alone
    silently serves the FIRST build to every later shape: a sweep reusing one symbol gets the first
    shape's output BO for all of them, so exactly that many elements come back correct and the rest
    of the host buffer reads as zero. That reads as a kernel returning partial rows, not as a cache
    hit, so it can absorb a lot of debugging before anyone suspects the key.
    """
    blob = "|".join(str(x) for x in parts)
    return f"design_{symbol}_{hashlib.sha1(blob.encode()).hexdigest()[:10]}"


def _build_streamed(symbol, shim, n_tiles, in_tile, out_tile, resident_len, compile_flags,
                    in_dt, out_dt, resident_dt, resident_depth=2, stack_size=None):
    """Stream `n_tiles` fixed-size operand tiles past ONE resident operand.

    This is the general resident-stream `[tile, D]` delivery: the core acquires the
    resident buffer ONCE for the whole stream, then consumes one `in_tile`-element tile
    and produces one `out_tile`-element tile per iteration. L1 therefore has to hold one
    TILE of the streamed operand rather than the whole thing, which is the entire
    difference between a gate that fits a 64 KB core tile and one that does not.

    The kernel is invoked as `kern(tile_in, resident, tile_out)`. If resident_len == 0 the
    resident fifo is omitted and it is `kern(tile_in, tile_out)`.

    `resident_depth` is the objectFIFO depth of the RESIDENT operand only. It defaults to 2, the
    IRON default, so every brick gated before this parameter existed is unaffected. Pass 1 when the
    resident is large: the core acquires it ONCE before the loop and releases it after (see `core`
    below), so exactly one buffer is ever live and the second is pure waste -- and it is waste of
    the scarcest resource, since a core tile has 64 KB total. Measured: gather-rows' 1024x8 f32
    codebook is 32 KB, which at depth 2 asks for 64 KB on that fifo alone and aiecc fails with
    "'aie.tile' op Basic sequential allocation also failed"; at depth 1 the same design fits.
    Depth 2 remains the default because a STREAMED operand genuinely wants double buffering (that
    is what overlaps the next tile's DMA with this tile's compute); a resident one never does.

    Row-wise verification (`_build_rowwise`) is the special case where one tile is one row
    of a matrix; `verify_streamed` is the tiled-operand case. Both are the same design.

    `stack_size` is the core's stack reservation in bytes; None takes the AIE dialect default of
    0x400. The generated linker script places the stack immediately below the objectFIFO buffers
    with zero clearance, so a kernel whose frame exceeds the reservation overwrites them -- which
    presents as missing output tiles, not as a crash. Measured frames on straight-line vector code
    reach 0x700. amd/IRON's own gemm and mha operators pass 0xD00 for the same reason.
    """
    compile_flags = _AIE_API_INC + list(compile_flags or [])
    # Make edits to #included files visible to the JIT cache key (see _include_closure_digest).
    compile_flags = compile_flags + [
        f"-DBRICK_INCLUDE_DIGEST={_include_closure_digest(shim, compile_flags)}"]
    has_resident = resident_len > 0
    _check_symbol_arity(symbol, shim, 3 if has_resident else 2)

    if has_resident:
        def design(inp: In, cst: In, out: Out):
            in_row = _npty((in_tile,), in_dt)
            out_row = _npty((out_tile,), out_dt)
            cst_ty = _npty((resident_len,), resident_dt)
            in_full = _npty((n_tiles * in_tile,), in_dt)
            out_full = _npty((n_tiles * out_tile,), out_dt)
            kern = ExternalFunction(symbol, source_file=str(shim),
                                    arg_types=[in_row, cst_ty, out_row],
                                    compile_flags=compile_flags)
            inf = ObjectFifo(in_row, name="inf")
            cf = ObjectFifo(cst_ty, name="cf", depth=resident_depth)
            of = ObjectFifo(out_row, name="of")

            def core(inf, cf, of, kern):
                ec = cf.acquire(1)
                for _ in range_(n_tiles):
                    ei = inf.acquire(1)
                    eo = of.acquire(1)
                    kern(ei, ec, eo)
                    of.release(1)
                    inf.release(1)
                cf.release(1)

            worker = Worker(core, fn_args=[inf.cons(), cf.cons(), of.prod(), kern],
                            stack_size=stack_size)
            in_tap = TensorTiler2D.group_tiler((n_tiles, in_tile), (1, in_tile), (n_tiles, 1))[0]
            out_tap = TensorTiler2D.group_tiler((n_tiles, out_tile), (1, out_tile), (n_tiles, 1))[0]
            cst_tap = TensorTiler2D.group_tiler((1, resident_len), (1, resident_len), (1, 1))[0]
            def sequence(a, c, o, in_h, cst_h, out_h):
                in_h.fill(a, in_tap)
                cst_h.fill(c, cst_tap)
                out_h.drain(o, out_tap, wait=True)

            rt = Runtime(
                sequence,
                [in_full, cst_ty, out_full, inf.prod(), cf.prod(), of.cons()],
            )
            return Program(
                iron.get_current_device(), rt, workers=[worker]
            ).resolve_program()
        design.__name__ = design.__qualname__ = _design_key(
            symbol, n_tiles, in_tile, out_tile, resident_len, in_dt, out_dt, resident_dt,
            compile_flags, resident_depth, stack_size, _shim_digest(shim))
        return iron.jit(design, use_cache=_JIT_CACHE)

    def design(inp: In, out: Out):
        in_row = _npty((in_tile,), in_dt)
        out_row = _npty((out_tile,), out_dt)
        in_full = _npty((n_tiles * in_tile,), in_dt)
        out_full = _npty((n_tiles * out_tile,), out_dt)
        kern = ExternalFunction(symbol, source_file=str(shim),
                                arg_types=[in_row, out_row],
                                compile_flags=compile_flags)
        inf = ObjectFifo(in_row, name="inf")
        of = ObjectFifo(out_row, name="of")

        def core(inf, of, kern):
            for _ in range_(n_tiles):
                ei = inf.acquire(1)
                eo = of.acquire(1)
                kern(ei, eo)
                of.release(1)
                inf.release(1)

        worker = Worker(core, fn_args=[inf.cons(), of.prod(), kern], stack_size=stack_size)
        in_tap = TensorTiler2D.group_tiler((n_tiles, in_tile), (1, in_tile), (n_tiles, 1))[0]
        out_tap = TensorTiler2D.group_tiler((n_tiles, out_tile), (1, out_tile), (n_tiles, 1))[0]
        def sequence(a, o, in_h, out_h):
            in_h.fill(a, in_tap)
            out_h.drain(o, out_tap, wait=True)

        rt = Runtime(sequence, [in_full, out_full, inf.prod(), of.cons()])
        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()
    design.__name__ = design.__qualname__ = _design_key(
        symbol, n_tiles, in_tile, out_tile, resident_len, in_dt, out_dt, resident_dt,
        compile_flags, stack_size, _shim_digest(shim))
    return iron.jit(design, use_cache=_JIT_CACHE)


def _build_rowwise(symbol, shim, m, in_cols, out_cols, const_len, compile_flags,
                   in_dt, out_dt, const_dt, stack_size=None):
    """Rows-of-a-matrix view of `_build_streamed`: one tile is one row.

    Kept as its own name because that IS what the row-wise bricks mean, and their call
    sites read better for it. The generated design is identical.
    """
    return _build_streamed(symbol, shim, m, in_cols, out_cols, const_len, compile_flags,
                           in_dt, out_dt, const_dt, stack_size=stack_size)


def _find_kernel_params(symbol, shim_path):
    """Return the parameter-list text of `symbol`, searching the shim and its local includes.

    Returns None if the symbol cannot be located (parse limits, macros, templates) -- callers
    treat that as "cannot check", never as "wrong".
    """
    shim_path = Path(shim_path)
    seen, queue = set(), [shim_path]
    while queue:
        f = queue.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        try:
            src = f.read_text()
        except OSError:
            continue
        m = re.search(rf'\b{re.escape(symbol)}\s*\(([^)]*)\)\s*\{{', src, re.S)
        if m:
            return m.group(1)
        for inc in re.findall(r'#include\s+"([^"]+)"', src):
            queue.append((f.parent / inc).resolve())
    return None


def _check_symbol_arity(symbol, shim, n_buffers):
    """Guard the exact bug that cost multiple sessions on gemm-int8xint4-dequant.

    A shim can define one function while the harness BINDS a different symbol -- typically
    the brick .cc's own exported wrapper, pulled in by the `#include`. If that wrapper takes
    more pointers than the harness can deliver (a core tile has only 2 input DMA channels),
    the extra parameters silently bind to whatever is in the argument registers: an operand
    lands on the zero-initialised OUTPUT buffer and the real output pointer dangles. Nothing
    errors -- no link failure, no verifier complaint -- the op just returns zeros.

    So: whatever symbol is bound must accept exactly the buffers we pass (inputs + output).
    """
    params = _find_kernel_params(symbol, shim)
    if params is None:
        return  # cannot locate it; do not guess
    depth, count, stripped = 0, 1, params.strip()
    if not stripped:
        count = 0
    else:
        for ch in params:
            if ch in "<([":
                depth += 1
            elif ch in ">)]":
                depth -= 1
            elif ch == "," and depth == 0:
                count += 1
    if count != n_buffers:
        raise ValueError(
            f"kernel symbol '{symbol}' takes {count} parameters but the harness delivers "
            f"{n_buffers} buffers ({n_buffers - 1} in + 1 out).\n"
            f"  shim: {shim}\n  params: ({params.strip()})\n"
            "An arity mismatch is SILENT on device -- surplus parameters bind to garbage "
            "registers, so an operand can alias the output buffer and the real output "
            "pointer dangles, yielding an all-zero result that looks like a kernel bug. "
            "Bind the shim's own function, or deliver the operands it actually wants."
        )


def _build_oneshot(symbol, shim, in_numels, out_numel, in_dts, out_dt, compile_flags,
                   stack_size=None):
    """Whole-buffer inputs, ONE kernel call, one output. For GEMM/GEMV-style bricks.
    In/Out params are signature markers for iron.jit's tensor-arg count; data flows
    through rt.sequence."""
    compile_flags = _AIE_API_INC + list(compile_flags or [])
    # Make edits to #included files visible to the JIT cache key (see _include_closure_digest).
    compile_flags = compile_flags + [
        f"-DBRICK_INCLUDE_DIGEST={_include_closure_digest(shim, compile_flags)}"]
    nin = len(in_numels)
    _check_symbol_arity(symbol, shim, nin + 1)

    def build_program():
        in_tys = [_npty((in_numels[i],), in_dts[i]) for i in range(nin)]
        out_ty = _npty((out_numel,), out_dt)
        kern = ExternalFunction(symbol, source_file=str(shim),
                                arg_types=list(in_tys) + [out_ty],
                                compile_flags=compile_flags)
        in_fifos = [ObjectFifo(in_tys[i], name=f"in{i}") for i in range(nin)]
        of = ObjectFifo(out_ty, name="of")

        def core(*args):
            *cons, ofc, k = args
            elems = [c.acquire(1) for c in cons]
            eo = ofc.acquire(1)
            k(*elems, eo)
            ofc.release(1)
            for c in cons:
                c.release(1)

        worker = Worker(core, fn_args=[f.cons() for f in in_fifos] + [of.prod(), kern],
                        stack_size=stack_size)
        seq_tys = list(in_tys) + [out_ty]

        def sequence(*params):
            *ins, o, in_prods, out_h = params
            for i in range(nin):
                in_prods[i].fill(ins[i])
            out_h.drain(o, wait=True)

        rt = Runtime(
            sequence,
            [*seq_tys, [f.prod() for f in in_fifos], of.cons()],
        )
        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    if nin == 1:
        def design(a: In, out: Out):
            return build_program()
    elif nin == 2:
        def design(a: In, b: In, out: Out):
            return build_program()
    elif nin == 3:
        def design(a: In, b: In, c: In, out: Out):
            return build_program()
    elif nin == 4:
        def design(a: In, b: In, c: In, d: In, out: Out):
            return build_program()
    else:
        raise ValueError(f"_build_oneshot supports 1-4 inputs, got {nin}")

    # `stack_size` belongs in the key for the same reason shapes do: it changes the program but
    # appears in neither the source nor the flags, so a sweep over it would otherwise compare one
    # artifact against itself and report that the stack does not matter.
    design.__name__ = design.__qualname__ = _design_key(
        symbol, in_numels, out_numel, in_dts, out_dt, compile_flags, stack_size,
        _shim_digest(shim))
    return iron.jit(design, use_cache=_JIT_CACHE)


def verify_oneshot(name, brick_cc, shim_body, symbol, inputs, out_numel, out_shape,
                   unpack, golden, gate, compile_flags=None, out_dt=np.int32,
                   stack_size=None):
    """inputs: list of (packed_1d_np, dtype). unpack(dev_flat)->got array (native shape).
    golden: reference array (native shape). Returns result dict."""
    compile_flags = list(compile_flags or [])
    shim = GEN / f"{name}_shim.cc"
    shim.write_text(
        f'// AUTO-GENERATED verify shim for the {name} brick.\n'
        f'#include <stdint.h>\n#include "{brick_cc}"\n{shim_body}\n'
    )
    in_numels = [int(np.asarray(a).size) for a, _ in inputs]
    in_dts = [dt for _, dt in inputs]
    design = _build_oneshot(symbol, shim, in_numels, out_numel, in_dts, out_dt, compile_flags,
                            stack_size=stack_size)

    def run_once():
        in_ts = [iron.tensor(np.ascontiguousarray(np.asarray(a).reshape(-1)), dtype=dt,
                             device="npu") for a, dt in inputs]
        out_t = iron.zeros((out_numel,), dtype=out_dt, device="npu")
        design(*in_ts, out_t)
        return out_t.numpy().reshape(-1).copy()

    dev1 = run_once()
    dev2 = run_once()
    got = np.asarray(unpack(dev1), np.float64)
    exp = np.asarray(golden, np.float64)
    num = np.linalg.norm((got - exp).ravel())
    den = np.linalg.norm(exp.ravel())
    rl2 = float(num / den) if den else float(num)
    determ = float(np.linalg.norm((dev1.astype(np.float64) - dev2.astype(np.float64)).ravel()))
    nz = float(np.abs(dev1).sum())
    ok = (nz > 0.0) and (rl2 <= gate)
    status = "PASS" if ok else ("FAIL-ZERO" if nz == 0.0 else "FAIL")
    print(f"[{name:22s}] rel_l2={rl2:.3e} gate={gate:.1e} nz={nz:.2e} "
          f"run2run={determ:.2e} -> {status}")
    # `got` is returned so a probe can inspect WHAT is wrong, not just how wrong.
    return dict(name=name, rel_l2=rl2, gate=gate, nonzero=nz, run2run=determ,
                status=status, ok=ok, got=got)


def verify_streamed(name, shim, symbol, in_tiles, out_tile_numel, resident,
                    unpack, golden, gate, in_dt, out_dt, resident_dt=None,
                    compile_flags=None, resident_depth=2, stack_size=None):
    """Gate a brick whose operands are too big to stage whole into L1.

    `verify_oneshot` moves every operand into L1 in one piece, so it can only gate shapes
    whose entire working set fits a 64 KB core tile (half that at objectFIFO depth 2).
    That ceiling is not a kernel property and it is not negotiable by picking better
    shapes -- a tall-K decode weight is megabytes. Here the big operand arrives as a
    stream of equal tiles and the small one stays resident, so L1 holds one tile.

    in_tiles:  (n_tiles, in_tile) -- tile i is the slice of the streamed operand that
               the kernel needs for output tile i, already in the kernel's own layout.
    resident:  1-D operand acquired once for the whole stream, or None.
    unpack:    (n_tiles, out_tile_numel) device result -> `got` in the golden's shape.

    Returns the same result dict as `verify_oneshot`, including the run-twice self-check
    that guards the CLFLUSH read race.
    """
    in_tiles = np.ascontiguousarray(np.asarray(in_tiles))
    n_tiles, in_tile = in_tiles.shape
    resident_len = 0 if resident is None else int(np.asarray(resident).size)
    design = _build_streamed(symbol, shim, n_tiles, in_tile, out_tile_numel, resident_len,
                             compile_flags, in_dt, out_dt, resident_dt, resident_depth,
                             stack_size=stack_size)

    def run_once():
        in_t = iron.tensor(np.ascontiguousarray(in_tiles.reshape(-1)), dtype=in_dt,
                           device="npu")
        out_t = iron.zeros((n_tiles * out_tile_numel,), dtype=out_dt, device="npu")
        if resident_len:
            r_t = iron.tensor(np.ascontiguousarray(np.asarray(resident).reshape(-1)),
                              dtype=resident_dt, device="npu")
            design(in_t, r_t, out_t)
        else:
            design(in_t, out_t)
        return out_t.numpy().reshape(n_tiles, out_tile_numel).copy()

    dev1 = run_once()
    dev2 = run_once()
    got = np.asarray(unpack(dev1), np.float64)
    exp = np.asarray(golden, np.float64)
    num = np.linalg.norm((got - exp).ravel())
    den = np.linalg.norm(exp.ravel())
    rl2 = float(num / den) if den else float(num)
    determ = float(np.linalg.norm((dev1.astype(np.float64) - dev2.astype(np.float64)).ravel()))
    nz = float(np.abs(dev1.astype(np.float64)).sum())
    ok = (nz > 0.0) and (rl2 <= gate)
    status = "PASS" if ok else ("FAIL-ZERO" if nz == 0.0 else "FAIL")
    print(f"[{name:34s}] rel_l2={rl2:.3e} gate={gate:.1e} nz={nz:.2e} "
          f"run2run={determ:.2e} tiles={n_tiles}x{in_tile} -> {status}", flush=True)
    return dict(name=name, rel_l2=rl2, gate=gate, nonzero=nz, run2run=determ,
                status=status, ok=ok, got=got)


def verify_rowwise(name, brick_cc, shim_body, symbol, m, in_cols, out_cols,
                   x, expected, gate, const=None, compile_flags=None,
                   in_dt=np.float32, out_dt=np.float32, const_dt=np.float32,
                   stack_size=None):
    """Generate shim, build+run the design twice on device, gate rel-L2.

    x: (m, in_cols) input.  const: 1-D packed const or None.  expected: (m, out_cols).
    Returns a result dict.
    """
    compile_flags = list(compile_flags or [])
    shim = GEN / f"{name}_shim.cc"
    shim.write_text(
        f'// AUTO-GENERATED verify shim for the {name} brick.\n'
        f'#include <stdint.h>\n#include "{brick_cc}"\n{shim_body}\n'
    )
    const_len = 0 if const is None else int(np.asarray(const).size)
    design = _build_rowwise(symbol, shim, m, in_cols, out_cols, const_len,
                            compile_flags, in_dt, out_dt, const_dt,
                            stack_size=stack_size)

    def run_once():
        x_t = iron.tensor(np.ascontiguousarray(x.reshape(-1)), dtype=in_dt, device="npu")
        out_t = iron.zeros((m * out_cols,), dtype=out_dt, device="npu")
        if const_len:
            c_t = iron.tensor(np.ascontiguousarray(np.asarray(const).reshape(-1)),
                              dtype=const_dt, device="npu")
            design(x_t, c_t, out_t)
        else:
            design(x_t, out_t)
        return out_t.numpy().reshape(m, out_cols).astype(np.float64).copy()

    dev1 = run_once()
    dev2 = run_once()
    exp = np.asarray(expected, np.float64)
    num = np.linalg.norm((dev1 - exp).ravel())
    den = np.linalg.norm(exp.ravel())
    rl2 = float(num / den) if den else float(num)
    determ = float(np.linalg.norm((dev1 - dev2).ravel()))
    nz = float(np.abs(dev1).sum())
    ok = (nz > 0.0) and (rl2 <= gate)
    status = "PASS" if ok else ("FAIL-ZERO" if nz == 0.0 else "FAIL")
    print(f"[{name:22s}] rel_l2={rl2:.3e} gate={gate:.1e} nz={nz:.2e} "
          f"run2run={determ:.2e} -> {status}")
    # `got` is returned so a probe can inspect WHAT is wrong, not just how wrong.
    # rowwise has no unpack step: dev1 IS the (m, out_cols) device result.
    return dict(name=name, rel_l2=rl2, gate=gate, nonzero=nz, run2run=determ,
                status=status, ok=ok, got=dev1)
