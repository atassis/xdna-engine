#!/usr/bin/env python3
"""CPU-only regression for the shipped rel-pos MHA rowtiled-stream generator.

Runs the real canonical generator
(designs/relpos_mha/relpos_rowtiled_stream_iron.py) with the three maintained
bucket arguments -- `scripts/relpos_prebuild.sh` BUCKETS
`100:25:bucket_100 152:38:bucket_152 172:43:single` at TQ=8, HEADS=8, ROWS=1,
SPLITP=1, TSKIP=0, TACTIVE=BUILT_T -- under scripts/iron_env.sh with
TOOLCHAIN_RESOLVE_ONLY=1 (resolve the pinned instance, never provision). The
device comes from the explicit `-d npu2` on the command line: there is no
autodetection, no aiecc and no kernel compilation here, only MLIR emission.

Three contracts, all read out of the emitted MLIR rather than out of an exit code:

  * import surface -- every name the generator takes from `aie.helpers.taplib`
    exists in the resolved instance. The frozen pin exports Affine/PerCall/
    TensorAccessPattern only, so a retired tiler name kills all three buckets
    before any compiler runs, which is a build failure and not a design failure.
  * topology and geometry -- one `aie.core` per head; every once-per-head shim tap
    sits at a uniformly spaced offset (h * head_len); the STREAM-A k/V replay tap
    stays one shared 4-D descriptor per head with strides `0, 0, DK, 1`; and the
    biggest baked buffer grows with the bucket's BUILT_T, so a bucket that silently
    emitted another bucket's geometry fails here.
  * emission -- the resolved module parses and verifies with the pinned aie.ir
    bindings and declares the device the command line asked for, and a malformed
    module is rejected (the check above is a real parser, not a no-op).
"""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENERATOR = REPO / "designs/relpos_mha/relpos_rowtiled_stream_iron.py"

# Parakeet n_heads on 8 parallel cores, and the generator's DK (head dim).
HEADS, DK = 8, 128

# `source` the accepted environment (it self-derives REPO and exports PYTHONPATH
# at the pinned instance), then exec the generator with the real CLI arguments.
_LAUNCH = 'set -e; . "$1"; shift; exec python3 "$@"'

# (name, BUILT_T, KB) straight off scripts/relpos_prebuild.sh's BUCKETS table.
BUCKETS = [
    ("bucket_100", 100, 25),
    ("bucket_152", 152, 38),
    ("single", 172, 43),
]

DMA_BD = re.compile(
    r"aie\.dma_bd\(%\S+ : memref<(?P<elems>\d+)x(?P<dtype>\w+)> "
    r"offset = (?P<offset>\d+) len = (?P<len>\d+) "
    r"sizes = \[(?P<sizes>[^\]]*)\] strides = \[(?P<strides>[^\]]*)\]\)"
)


def cli(built_t, kb):
    """The prebuild's Make invocation for one bucket, as one argv list."""
    return [
        "-d", "npu2", "-T", str(built_t), "--tq", "8", "--kb", str(kb),
        "--tactive", str(built_t), "--heads", str(HEADS), "--splitp", "--rows", "1",
    ]


def generate(args):
    """Run the generator once under the pinned env; return its MLIR text."""
    env = dict(
        os.environ,
        TOOLCHAIN_RESOLVE_ONLY="1",
        PYTHONDONTWRITEBYTECODE="1",
        AIE_DEVICE="npu2",
        OMP_NUM_THREADS="1",
    )
    result = subprocess.run(
        ["bash", "-c", _LAUNCH, "_", str(REPO / "scripts/iron_env.sh"),
         str(GENERATOR), *args],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{GENERATOR.name} {' '.join(args)} exited {result.returncode}\n"
            f"{result.stderr[-3000:]}"
        )
    return result.stdout


def verify(mlir):
    """Parse + verify with the pinned aie.ir bindings (same process, no compiler)."""
    import aie.dialects.aie  # noqa: F401  (registers the aie dialect)
    from aie.ir import Context, Module

    return Module.parse(mlir, Context()).operation.verify()


def tap_groups(mlir):
    """[(offset, len)] per distinct (dtype, sizes, strides) descriptor pattern, in order."""
    groups = {}
    for m in DMA_BD.finditer(mlir):
        groups.setdefault((m["dtype"], m["sizes"], m["strides"]), []).append(
            (int(m["offset"]), int(m["len"]))
        )
    return groups


class RelposGeneratorTest(unittest.TestCase):
    """The shipped resident-MHA design still emits all three buckets on npu2."""

    @classmethod
    def setUpClass(cls):
        # Disk-backed fixture space (TMPDIR), removed in tearDownClass.
        cls.tmp = tempfile.mkdtemp(prefix="relpos-generator-")
        cls.mlir = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def emit(self, name, built_t, kb):
        """Emit one bucket once; keep the MLIR plus a disk-backed fixture to tear down."""
        if name not in self.mlir:
            mlir = generate(cli(built_t, kb))
            self.assertTrue(mlir.startswith("module"),
                            f"{name}: stdout is not MLIR: {mlir[:200]}")
            fixture_dir = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))
            (fixture_dir / "aie.mlir").write_text(mlir)
            self.mlir[name] = mlir
        return self.mlir[name]

    def test_generator_imports_the_pinned_taplib_surface(self):
        # The cheapest premise: the pin has no TensorTiler2D, so any name the
        # generator asks taplib for has to be one the pin actually exports.
        import aie.helpers.taplib as taplib

        self.assertFalse(hasattr(taplib, "TensorTiler2D"),
                         "the pinned taplib unexpectedly re-exported TensorTiler2D")
        imported = re.findall(r"^from aie\.helpers\.taplib import (.+)$",
                              GENERATOR.read_text(), re.MULTILINE)
        self.assertEqual(len(imported), 1, "expected exactly one taplib import line")
        for name in imported[0].replace(" ", "").split(","):
            self.assertTrue(hasattr(taplib, name),
                            f"pinned taplib does not export {name} (imported at {imported[0]!r})")

    def test_all_three_buckets_emit_and_verify(self):
        for name, built_t, kb in BUCKETS:
            with self.subTest(bucket=name):
                mlir = self.emit(name, built_t, kb)
                self.assertIn("aie.device(npu2)", mlir)
                self.assertNotIn("aie.device(npu1)", mlir)
                self.assertTrue(verify(mlir), f"{name}: emitted module failed to verify")

    def test_eight_heads_are_eight_parallel_cores(self):
        for name, built_t, kb in BUCKETS:
            with self.subTest(bucket=name):
                groups = tap_groups(self.emit(name, built_t, kb))
                self.assertEqual(self.mlir[name].count("aie.core("), HEADS,
                                 f"{name}: one core per head expected")
                # The per-head SCATTER: every tap pattern used once per head sits at
                # HEADS distinct, uniformly spaced, increasing offsets (h * head_len).
                per_head = [key for key, taps in groups.items() if len(taps) == HEADS]
                self.assertTrue(per_head, f"{name}: no per-head tap groups found")
                for key in per_head:
                    offsets = [offset for offset, _ in groups[key]]
                    step = offsets[1] - offsets[0]
                    self.assertGreater(step, 0, f"{name}: {key[0]} taps not increasing")
                    self.assertEqual(
                        offsets, [offsets[0] + step * h for h in range(HEADS)],
                        f"{name}: {key[0]} head offsets are not h*head_len",
                    )

    def test_kpv_stream_tap_shape_is_unchanged(self):
        """STREAM-A: ONE 4-D k/V replay tap per head, strides 0,0,DK,1, heads agreeing."""
        for name, built_t, kb in BUCKETS:
            with self.subTest(bucket=name):
                kpv = [m for m in DMA_BD.finditer(self.emit(name, built_t, kb))
                       if m["strides"] == f"0, 0, {DK}, 1"]
                self.assertEqual(len(kpv), HEADS, f"{name}: one k/V replay tap per head expected")
                shapes = {m["sizes"] for m in kpv}
                self.assertEqual(len(shapes), 1, f"{name}: heads disagree on the k/V shape")
                self.assertEqual(len(next(iter(shapes)).split(",")), 4,
                                 f"{name}: k/V tap is not the 4-D replay pattern")

    def test_baked_geometry_grows_with_built_t(self):
        biggest = {}
        for name, built_t, kb in BUCKETS:
            elems = (int(m["elems"]) for m in DMA_BD.finditer(self.emit(name, built_t, kb)))
            biggest[name] = max(elems)
        self.assertLess(biggest["bucket_100"], biggest["bucket_152"], biggest)
        self.assertLess(biggest["bucket_152"], biggest["single"], biggest)

    def test_verify_is_a_real_parser(self):
        """Negative guard: malformed modules must be rejected, not silently accepted."""
        mlir = self.emit(*BUCKETS[0])
        self.assertTrue(verify(mlir))
        with self.assertRaises(Exception):
            verify(mlir[: len(mlir) // 2])
        with self.assertRaises(Exception):
            verify("module { this is not mlir }")
        with self.assertRaises(Exception):
            verify(mlir.replace("aie.dma_bd(", "aie.not_a_real_op("))


if __name__ == "__main__":
    unittest.main()
