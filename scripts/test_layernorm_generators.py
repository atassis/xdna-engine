#!/usr/bin/env python3
"""CPU-only regression for the two shipped ctxLN device generators.

Runs the real canonical generators (designs/ctx_ln/ctx_ln_iron.py and
designs/ctx_ln/affine_cast_iron.py) with the production arguments used by
Makefile.ctxln / Makefile.affinecast, under scripts/iron_env.sh with
TOOLCHAIN_RESOLVE_ONLY=1 (resolve the pinned instance, never provision). The
device comes from the explicit `-d npu2` on the command line: there is no
autodetection, no aiecc and no kernel compilation here, only MLIR emission.

Two contracts, both read out of the emitted MLIR rather than out of an exit code:

  * geometry -- the retired `aie.helpers.taplib.TensorTiler2D.simple_tiler` gave
    each of the 8 cores one contiguous partition of rows_per_core rows, so the 8
    descriptor walks are disjoint and cover rows*cols exactly once. The direct
    replacement (`aie.helpers.taplib.TensorAccessPattern`, the owner the working
    whole_array/dwconv generators already use) has to reproduce those numbers
    address for address, so the test asserts the emitted `aie.dma_bd` offset/len
    of every row walk, in emission order.
  * emission -- the resolved module parses and verifies with the pinned aie.ir
    bindings, and declares the device the command line asked for.
"""

import os
import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CTX_LN = REPO / "designs/ctx_ln/ctx_ln_iron.py"
AFFINE_CAST = REPO / "designs/ctx_ln/affine_cast_iron.py"

ROWS, COLS, CORES = 512, 1024, 8
ROWS_PER_CORE = ROWS // CORES
ELEMENTS = ROWS * COLS
PARTITION = ROWS_PER_CORE * COLS

# `source` the accepted environment (it self-derives REPO and exports PYTHONPATH
# at the pinned instance), then exec the generator with the real CLI arguments.
_LAUNCH = 'set -e; . "$1"; shift; exec python3 "$@"'

DMA_BD = re.compile(
    r"aie\.dma_bd\(%\S+ : memref<(?P<elems>\d+)x(?P<dtype>\w+)> "
    r"offset = (?P<offset>\d+) len = (?P<len>\d+) "
    r"sizes = \[(?P<sizes>[^\]]*)\] strides = \[(?P<strides>[^\]]*)\]\)"
)


def generate(generator, args):
    """Run one canonical generator under the pinned env; return its MLIR text."""
    env = dict(
        os.environ,
        TOOLCHAIN_RESOLVE_ONLY="1",
        PYTHONDONTWRITEBYTECODE="1",
        AIE_DEVICE="npu2",
    )
    result = subprocess.run(
        ["bash", "-c", _LAUNCH, "_", str(REPO / "scripts/iron_env.sh"),
         str(generator), *args],
        capture_output=True, text=True, env=env, timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{generator.name} {' '.join(args)} exited {result.returncode}\n"
            f"{result.stderr[-2000:]}"
        )
    return result.stdout


def row_walks(mlir, dtype):
    """The (offset, len, sizes, strides) of every row walk of `dtype`, in order."""
    return [
        (int(m["offset"]), int(m["len"]), m["sizes"], m["strides"])
        for m in DMA_BD.finditer(mlir)
        if int(m["len"]) == PARTITION and int(m["elems"]) == ELEMENTS and m["dtype"] == dtype
    ]


def verify(mlir):
    """Parse + verify with the pinned aie.ir bindings (same process, no compiler)."""
    import aie.dialects.aie  # noqa: F401  (registers the aie dialect)
    from aie.ir import Context, Module

    return Module.parse(mlir, Context()).operation.verify()


class LayernormGeneratorTest(unittest.TestCase):
    """The shipped ctxLN/affine-cast designs still emit at 512x1024 on npu2."""

    @classmethod
    def setUpClass(cls):
        cls.ctx_ln = generate(CTX_LN, ["-d", "npu2", "-r", "512", "-c", "1024", "-t", "0"])
        cls.affine_cast = generate(
            AFFINE_CAST, ["-d", "npu2", "-r", "512", "-c", "1024", "-t", "0", "--dtype", "f32"]
        )

    def assert_row_partitions(self, walks, what):
        """`walks` is a whole number of the 8-walk row-partition pattern (in, then out)."""
        self.assertTrue(walks, f"{what}: no row walks emitted")
        self.assertEqual(len(walks) % CORES, 0, f"{what}: not a whole number of 8-core patterns")
        expected = [(i * PARTITION, PARTITION) for i in range(CORES)]
        for start in range(0, len(walks), CORES):
            group = walks[start : start + CORES]
            self.assertEqual(
                [(offset, length) for offset, length, _, _ in group], expected,
                f"{what}: row walks are not the 8 contiguous row partitions in order",
            )
            for offset, _, sizes, strides in group:
                self.assertEqual(sizes, "1, 1, 64, 1024", f"{what}: unexpected tile sizes")
                self.assertEqual(strides, "0, 0, 1024, 1", f"{what}: unexpected row strides")
            # Disjoint and complete, straight from the descriptors.
            cursor = 0
            for offset, length in sorted((o, n) for o, n, _, _ in group):
                self.assertEqual(offset, cursor, f"{what}: gap or overlap at {cursor}")
                cursor = offset + length
            self.assertEqual(cursor, ELEMENTS, f"{what}: does not cover rows*cols")

    def test_ctx_ln_row_partitions_cover_the_matrix(self):
        # ctxLN is f32 in / f32 out: one 8-walk pattern per direction.
        self.assert_row_partitions(row_walks(self.ctx_ln, "f32"), "ctx_ln f32")
        self.assertEqual(len(row_walks(self.ctx_ln, "f32")), 2 * CORES,
                         "ctx_ln: expected one 8-walk pattern for in and one for out")

    def test_affine_cast_row_partitions_cover_the_matrix(self):
        self.assert_row_partitions(row_walks(self.affine_cast, "f32"), "affine_cast f32 in")
        self.assert_row_partitions(row_walks(self.affine_cast, "bf16"), "affine_cast bf16 out")
        self.assertEqual(len(row_walks(self.affine_cast, "f32")), CORES)
        self.assertEqual(len(row_walks(self.affine_cast, "bf16")), CORES)

    def test_affine_cast_keeps_the_gamma_beta_broadcast(self):
        # One full [gamma|beta] fill per core -- the tiler port must not touch it.
        broadcasts = [
            (int(m["offset"]), int(m["len"]))
            for m in DMA_BD.finditer(self.affine_cast)
            if int(m["len"]) == 2 * COLS
        ]
        self.assertEqual(broadcasts, [(0, 2 * COLS)] * CORES)

    def test_emitted_modules_declare_the_requested_device(self):
        for name, mlir in (("ctx_ln", self.ctx_ln), ("affine_cast", self.affine_cast)):
            with self.subTest(name):
                self.assertIn("aie.device(npu2)", mlir)
                self.assertNotIn("aie.device(npu1)", mlir)

    def test_emitted_modules_verify(self):
        for name, mlir in (("ctx_ln", self.ctx_ln), ("affine_cast", self.affine_cast)):
            with self.subTest(name):
                self.assertTrue(verify(mlir))
        # A guard that the check above is a real parser and not a no-op.
        with self.assertRaises(Exception):
            verify(self.ctx_ln[: len(self.ctx_ln) // 2])


if __name__ == "__main__":
    unittest.main()
