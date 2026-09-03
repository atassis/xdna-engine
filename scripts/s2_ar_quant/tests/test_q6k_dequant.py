#!/usr/bin/env python3
"""Validate q6k_dequant.dequantize_q6k against upstream ggml, not a re-derived
reference: compiles gen_q6k_golden.c against the real s2.cpp/ggml/src/ggml-quants.c
and compares this port's output to what that C function actually produces.

Needs a C compiler and a checkout of s2.cpp/ggml (set S2CPP_GGML_DIR, or this file
finds it at the conventional workspace sibling path ../../../../../s2.cpp/ggml);
skips rather than failing when neither is available. stdlib unittest only --
no pytest/numpy-extra dependency beyond the numpy this whole tree already needs.

    python3 test_q6k_dequant.py -v
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from q6k_dequant import BLOCK_BYTES, dequantize_q6k  # noqa: E402

DEFAULT_GGML_DIR = HERE.parents[3] / "s2.cpp" / "ggml"  # tests/ -> s2_ar_quant/ -> scripts/ -> wt-tts-s2-mvp/ -> workspace root


def _ggml_dir():
    d = Path(os.environ.get("S2CPP_GGML_DIR", DEFAULT_GGML_DIR))
    return d if (d / "src" / "ggml-quants.c").is_file() else None


class TestQ6KDequant(unittest.TestCase):
    @unittest.skipIf(shutil.which("cc") is None, "no C compiler on PATH")
    def test_against_upstream_ggml(self):
        ggml_dir = _ggml_dir()
        if ggml_dir is None:
            self.skipTest(f"s2.cpp/ggml not found (checked {DEFAULT_GGML_DIR}); "
                           "set S2CPP_GGML_DIR to override")

        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "gen_q6k_golden"
            subprocess.run(
                ["cc", "-O2", "-DGGML_COMMON_DECL_C",
                 "-I", str(ggml_dir / "src"), "-I", str(ggml_dir / "include"),
                 "-o", str(exe), str(HERE / "gen_q6k_golden.c"),
                 str(ggml_dir / "src" / "ggml-quants.c"), "-lm"],
                check=True, capture_output=True, text=True,
            )
            subprocess.run([str(exe), td], check=True, capture_output=True, text=True)

            raw = (Path(td) / "blocks.bin").read_bytes()
            golden = np.frombuffer((Path(td) / "golden.bin").read_bytes(), dtype=np.float32)

        nb = len(raw) // BLOCK_BYTES
        self.assertEqual(len(raw), nb * BLOCK_BYTES)
        got = dequantize_q6k(raw, nb * 256)

        self.assertEqual(got.shape, golden.shape)
        # Bit-exact: both do the same three f32 multiplies (d * scale * q) in the
        # same order, so there is no accumulation-order slack to tolerate.
        np.testing.assert_array_equal(got, golden)

    def test_single_block_smoke(self):
        """An all-zero block (quants and scales both 0, d=1.0) must dequantize to
        all-zero -- exercises the struct layout independent of the C golden path."""
        raw = bytes(BLOCK_BYTES - 2) + struct.pack("<e", 1.0)
        out = dequantize_q6k(raw, 256)
        self.assertEqual(out.shape, (256,))
        self.assertTrue(np.all(out == 0.0))


if __name__ == "__main__":
    unittest.main()
