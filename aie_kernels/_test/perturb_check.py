#!/usr/bin/env python3
"""Perturb check for bricklib's JIT-cache content-addressing.

CONTEXT (2026-09 task: retire `codec_block/window_driver.py`'s `_CB`). bricklib.py's own
comment on `_JIT_CACHE` used to say: "The JIT cache keyed only what it was told about, so
editing a kernel .cc did not invalidate it and a stale .o was linked -- hence use_cache=False
everywhere here... Set BRICK_JIT_CACHE=1 to enable it against a toolchain that records what a
build actually consumed; do NOT enable it otherwise, and never without re-running the perturb
check (edit a kernel, confirm the gate moves)." This IS that check.

The failure mode this guards is not a failed build, it is a PASSING one that is silently wrong:
`_shim_digest`'s own docstring records a reused design name once returning a stale xclbin at
rel-L2 2.052e+01 (output in [-110.9, +124.96]) while a fresh name on the same shim returned
1.275e-04. A cache that does not notice a kernel changed reproduces exactly that, quietly.

TWO HALVES, run separately:

  PART A (this script, device-free, run automatically below): build via
  `CallableDesign.compile(xclbin_path=, inst_path=)` -- the explicit-path form used by
  `export_codec_artifacts.py`, which per its own docstring's `iron.set_current_device(NPU2())`
  precedes it, WRITES ARTIFACTS DIRECTLY AND BYPASSES THE ON-DISK JIT CACHE (see
  `CompilableDesign.compile`'s own docstring: "The on-disk cache is bypassed in this mode"). So
  this half tests something the cache flag does not gate at all: is the BUILD ITSELF
  deterministic and content-sensitive -- same kernel source -> byte-identical xclbin, different
  kernel source -> different xclbin. That is a NECESSARY precondition for any cache built on top
  of it to be correct, and it needs no device.

  PART B (`--timing-worker`, invoked as separate subprocesses below): the actual on-disk JIT
  cache (`~/.npu/cache` by default, overridden here to an isolated dir so this does not collide
  with another live agent's cache), exercised through the no-explicit-path `.compile()` form,
  timed cold (empty cache dir) vs warm (same dir, second process) vs `BRICK_JIT_CACHE=0` (always
  rebuild, the pre-2026-09 default) -- three SEPARATE processes, because a warm-cache win that
  only shows up via in-process Python object reuse (`bricklib._DESIGNS`, `CallableDesign.
  _kernel_cache`) is not the thing being tested here; the persistent, cross-process artifact
  cache is.

Neither half touches /dev/accel0: `iron.set_current_device(NPU2())` runs before any compile()
call, short-circuiting `get_current_device()`'s runtime probe (see `hostruntime._CURRENT_DEVICE`
fast path) -- the same device-free pattern `export_codec_artifacts.py` documents and relies on.

Usage:
    python3 perturb_check.py              # runs Part A, then orchestrates Part B
    python3 perturb_check.py --timing-worker <cache_dir> <use_cache 0|1>   # Part B subprocess
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def _bind_device():
    import aie.iron as iron
    from aie.iron.device import NPU2
    iron.set_current_device(NPU2())


def _write_kernel(path, symbol, coeff):
    path.write_text(
        "#include <aie_api/aie.hpp>\n"
        f'extern "C" void {symbol}(float *in, float *out) {{\n'
        "  event0();\n"
        "  ::aie::vector<float,16> v = ::aie::load_v<16>(in);\n"
        f"  v = ::aie::mul(v, ::aie::broadcast<float,16>({coeff}f));\n"
        "  ::aie::store_v(out, v);\n"
        "  event1();\n"
        "}\n"
    )


def _build_design(bricklib, kernel_path, shim_path, symbol, coeff):
    import numpy as np
    _write_kernel(kernel_path, symbol, coeff)
    shim_path.write_text(f'#include <stdint.h>\n#include "{kernel_path}"\n')
    return bricklib._build_streamed(
        symbol, shim_path, 1, 16, 16, 0, None,
        np.float32, np.float32, None,
    )


def part_a(out_dir):
    """Explicit-path perturb check: build determinism + content sensitivity, cache bypassed.

    Gates on `insts.bin` (the DMA/dispatch program) and `<symbol>.o` (Peano's own compiled
    kernel object, in `kernel_dir`) for the "unperturbed rebuild is IDENTICAL" claim -- both
    confirmed reliably byte-reproducible across repeated runs. `main.pdi` (bootgen's packed
    Programmable Device Image) is NOT: it matched on the first run of this check and did not on
    a later one, with no kernel change involved -- bootgen embeds its own build-time provenance
    into the PDI container, same failure class as xclbinutil's UUID/timestamp fields (found
    first, chasing those: a `"XclBinUUID":"<hex>"` JSON field, a raw 16-byte binary UUID in the
    AXLF header duplicated at a second offset, and a 4-byte build timestamp -- none a property
    of the compiled program). `main.pdi` is still checked for the "perturbed rebuild DIFFERS"
    claim, where an occasional false negative from timestamp noise cutting the other way is not
    a correctness risk -- only `insts.bin`/`<symbol>.o` gate the "identical" side, where
    container noise would falsely FAIL a check that should pass.
    """
    _bind_device()
    sys.path.insert(0, str(HERE))
    import bricklib

    out_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = out_dir / "kernel.cc"
    shim_path = out_dir / "shim.cc"
    symbol = "perturb_check_op"

    def build(tag, coeff):
        design = _build_design(bricklib, kernel_path, shim_path, symbol, coeff)
        xclbin = out_dir / f"{tag}.xclbin"
        inst = out_dir / f"{tag}.bin"
        t0 = time.time()
        design.compile(xclbin_path=xclbin, inst_path=inst)
        dt = time.time() - t0
        pdi = design.get_pdi_path()
        kernel_o = xclbin.parent / f"{xclbin.stem}.prj" / f"{symbol}.o"
        return inst.read_bytes(), pdi.read_bytes(), kernel_o.read_bytes(), dt

    ib_a1, pdi_a1, ko_a1, dt_a1 = build("a1", "3.0")
    ib_a2, pdi_a2, ko_a2, dt_a2 = build("a2", "3.0")   # unperturbed rebuild
    ib_b, pdi_b, ko_b, dt_b = build("b", "5.0")        # perturbed rebuild

    print(f"[part A] a1 (coeff=3.0): insts {len(ib_a1)}B, pdi {len(pdi_a1)}B, "
          f"kernel.o {len(ko_a1)}B, {dt_a1:.2f}s")
    print(f"[part A] a2 (coeff=3.0, rebuild): insts {len(ib_a2)}B, pdi {len(pdi_a2)}B, "
          f"kernel.o {len(ko_a2)}B, {dt_a2:.2f}s")
    print(f"[part A] b  (coeff=5.0, perturbed): insts {len(ib_b)}B, pdi {len(pdi_b)}B, "
          f"kernel.o {len(ko_b)}B, {dt_b:.2f}s")

    insts_unpert_identical = (ib_a1 == ib_a2)
    ko_unpert_identical = (ko_a1 == ko_a2)
    pdi_unpert_identical = (pdi_a1 == pdi_a2)          # informational only, see docstring
    pdi_pert_differs = (pdi_a1 != pdi_b)
    ko_pert_differs = (ko_a1 != ko_b)

    print(f"[part A] insts.bin identical, unperturbed rebuild:  {insts_unpert_identical}  (GATED)")
    print(f"[part A] {symbol}.o identical, unperturbed rebuild: {ko_unpert_identical}  (GATED)")
    print(f"[part A] main.pdi  identical, unperturbed rebuild:  {pdi_unpert_identical}  (informational -- bootgen embeds its own build provenance)")
    print(f"[part A] main.pdi  DIFFERENT, perturbed rebuild:    {pdi_pert_differs}")
    print(f"[part A] {symbol}.o DIFFERENT, perturbed rebuild:   {ko_pert_differs}  (GATED)")

    ok = insts_unpert_identical and ko_unpert_identical and ko_pert_differs
    print(f"[part A] VERDICT: {'PASS' if ok else 'FAIL'} -- "
          f"{'build is deterministic and content-sensitive' if ok else 'INVESTIGATE: build is not reliably content-sensitive/deterministic'}")
    return ok


def timing_worker(cache_dir, use_cache_flag):
    """Part B subprocess body: one design, one no-explicit-path .compile(), report elapsed."""
    os.environ["NPU_CACHE_HOME"] = str(cache_dir)
    os.environ["BRICK_JIT_CACHE"] = use_cache_flag
    _bind_device()
    sys.path.insert(0, str(HERE))
    import bricklib
    work = Path(cache_dir).parent / "timing_work"
    work.mkdir(parents=True, exist_ok=True)
    kernel_path = work / "kernel.cc"
    shim_path = work / "shim.cc"
    design = _build_design(bricklib, kernel_path, shim_path, "perturb_timing_op", "3.0")
    t0 = time.time()
    design.compile()   # no explicit paths: consults/populates NPU_CACHE_HOME
    dt = time.time() - t0
    print(f"{dt:.4f}")


def part_b(out_dir):
    """Cold vs warm vs always-rebuild, three separate subprocesses, isolated cache dirs.

    `always-rebuild` (BRICK_JIT_CACHE=0) gets its OWN fresh cache_dir, not the `cold`/`warm`
    pair's -- `compile()`'s `kernel_dir = NPU_CACHE_HOME / cache_hash` is computed from content
    alone, independent of `use_cache`, and `compile_external_kernel` "skips any .o that is
    already present" in that directory regardless of the JIT-cache flag (see bricklib.py's
    comment on the manifest-validated nested cache). Reusing the warm pair's cache_dir for the
    always-rebuild run measured 0.2s instead of ~2.4s: not the JIT cache (correctly bypassed by
    the flag) but the SAME directory's already-compiled kernel .o being picked up underneath it.
    A useful finding in its own right (a second, nested cache layer exists below `use_cache`),
    but it made the "always rebuild" baseline dishonest, so it gets an isolated directory here.
    """
    cold_dir = out_dir / "npu_cache_home_cold"
    rebuild_dir = out_dir / "npu_cache_home_rebuild"
    for d in (cold_dir, rebuild_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    def run_worker(cache_dir, use_cache_flag):
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--timing-worker", str(cache_dir), use_cache_flag],
            capture_output=True, text=True, cwd=str(HERE),
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr, file=sys.stderr)
            raise RuntimeError(f"timing worker failed (use_cache={use_cache_flag})")
        return float(r.stdout.strip().splitlines()[-1])

    t_cold = run_worker(cold_dir, "1")      # empty dir -- first process, forced miss
    t_warm = run_worker(cold_dir, "1")      # SAME dir, same kernel content -- should hit
    t_norebuild_baseline = run_worker(rebuild_dir, "0")   # own, never-touched dir

    print(f"[part B] cold   (BRICK_JIT_CACHE=1, empty cache dir):        {t_cold:.3f}s")
    print(f"[part B] warm   (BRICK_JIT_CACHE=1, same dir, 2nd process):  {t_warm:.3f}s")
    print(f"[part B] always-rebuild (BRICK_JIT_CACHE=0, own empty dir): {t_norebuild_baseline:.3f}s")
    if t_warm > 0:
        print(f"[part B] warm speedup vs cold: {t_cold / t_warm:.1f}x")
    shutil.rmtree(cold_dir, ignore_errors=True)
    shutil.rmtree(rebuild_dir, ignore_errors=True)
    return t_cold, t_warm, t_norebuild_baseline


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--timing-worker":
        timing_worker(sys.argv[2], sys.argv[3])
        sys.exit(0)

    scratch = Path(
        os.environ.get("PERTURB_CHECK_SCRATCH")
        or (Path.home() / ".cache" / "perturb_check_scratch")
    )
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    ok_a = part_a(scratch / "part_a")
    print()
    part_b(scratch / "part_b")

    shutil.rmtree(scratch, ignore_errors=True)
    sys.exit(0 if ok_a else 1)
