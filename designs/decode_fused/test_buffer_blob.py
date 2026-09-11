#!/usr/bin/env python3
"""Pin buffer_blob.write_blob: every byte reads back, and zero runs cost no blocks.

The content assertions are the load-bearing half -- a cache blob is not always zero
(gen_decode.py seeds a random past segment into kc/vc when P>0), and several probes read
every meta["weights"] blob by name. The sparsity assertions skip on a filesystem that does
not deallocate, since that is a property of the target fs, not of this code.

    python3 test_buffer_blob.py -v
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_blob import write_blob  # noqa: E402

MIB = 1 << 20


def disk_bytes(path):
    return os.stat(path).st_blocks * 512


class WriteBlob(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        probe = os.path.join(self.dir, "probe")
        write_blob(probe, bytes(8 * MIB))
        self.sparse_fs = disk_bytes(probe) == 0

    def roundtrip(self, name, data):
        path = write_blob(os.path.join(self.dir, name), data)
        self.assertEqual(os.path.getsize(path), len(data))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), data)
        return path

    def test_all_zero_blob_reads_back_and_costs_nothing(self):
        path = self.roundtrip("kc.bin", np.zeros(32 * MIB, np.uint16).tobytes())
        if not self.sparse_fs:
            self.skipTest("filesystem does not deallocate zero runs")
        self.assertEqual(disk_bytes(path), 0)

    def test_seeded_past_survives_and_only_the_zero_tail_is_freed(self):
        past = np.random.default_rng(0).integers(1, 255, 4 * MIB, dtype=np.uint8).tobytes()
        path = self.roundtrip("seeded.bin", past + bytes(60 * MIB))
        if not self.sparse_fs:
            self.skipTest("filesystem does not deallocate zero runs")
        self.assertLess(disk_bytes(path), 8 * MIB)

    def test_dense_blob_keeps_every_block(self):
        dense = np.random.default_rng(1).integers(1, 255, 8 * MIB, dtype=np.uint8).tobytes()
        path = self.roundtrip("W_head.bin", dense)
        self.assertGreaterEqual(disk_bytes(path), 8 * MIB)

    def test_nonzero_byte_in_a_short_trailing_chunk_survives(self):
        self.roundtrip("tail.bin", bytes(MIB) + b"\x07" + bytes(3))

    def test_empty_blob(self):
        self.roundtrip("empty.bin", b"")

    def test_interior_nonzero_run_between_holes(self):
        self.roundtrip("interior.bin", bytes(4 * MIB) + b"\x01" * MIB + bytes(4 * MIB))


if __name__ == "__main__":
    unittest.main()
