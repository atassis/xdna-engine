"""CPU checks for the actual FC2 exact-RNE add source."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "aie_kernels/acc-add/acc_add_ieee_rne.cc"


class ExactRneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="fc2-rne-")
        library = Path(cls.tmp.name) / "add.so"
        subprocess.run(["c++", "-std=c++20", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
                        "-Wno-psabi", str(SOURCE), "-o", str(library)], check=True)
        cls.add = ctypes.CDLL(str(library)).acc_add_row_ieee_rne
        cls.add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def exact(self, a, b):
        a = np.ascontiguousarray(a, dtype=np.float32).ravel()
        b = np.ascontiguousarray(b, dtype=np.float32).ravel()
        self.assertEqual(a.size % 16, 0)
        out = np.empty_like(a)
        self.add(a.ctypes.data, b.ctypes.data, out.ctypes.data, out.size)
        return out

    def test_matches_numpy_for_boundaries_and_random_finite(self):
        bits = np.array([0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x3f800000,
                         0x3f800001, 0x7f7fffff, 0xff7fffff] * 16, dtype=np.uint32)
        rng = np.random.default_rng(0x524e45)
        random = rng.integers(0, 1 << 32, 16 * 4096, dtype=np.uint32)
        random[(random & 0x7f800000) == 0x7f800000] ^= np.uint32(0x00800000)
        for a_bits, b_bits in ((bits, bits[::-1]), (random, random[::-1])):
            a, b = a_bits.view(np.float32), b_bits.view(np.float32)
            with np.errstate(all="ignore"):
                expected = np.add(a, b, dtype=np.float32)
            actual = self.exact(a, b)
            valid = ~np.isnan(expected)
            self.assertTrue(np.array_equal(actual.view(np.uint32)[valid], expected.view(np.uint32)[valid]))

    def test_ordered_zero_seed_witness(self):
        partials = [np.full(16, value, np.float32) for value in (1e20, 1.0, -1e20, 3.0)]
        acc = np.zeros(16, np.float32)
        for partial in partials:
            acc = self.exact(acc, partial)
        self.assertTrue(np.all(acc == np.float32(3.0)))

    def test_signed_zero_and_canonical_nan(self):
        a = np.array([0, 0x80000000, 0x7f800001, 0x7f800000] * 4, np.uint32).view(np.float32)
        b = np.array([0x80000000, 0x80000000, 0x3f800000, 0xff800000] * 4, np.uint32).view(np.float32)
        got = self.exact(a, b).view(np.uint32)
        self.assertEqual(got[0], 0)
        self.assertEqual(got[1], 0x80000000)
        self.assertTrue(np.all(got[2::4] == 0x7fc00000))
        self.assertTrue(np.all(got[3::4] == 0x7fc00000))


if __name__ == "__main__":
    unittest.main()
