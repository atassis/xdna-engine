#!/usr/bin/env python3
"""Pin buffer_blob.write_blob: every byte reads back, zero runs cost no blocks, and
arms packing identical weights share one inode without sharing a rebuild.

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

    def test_rewrite_breaks_a_hardlink_instead_of_writing_through(self):
        """The invariant dedup_artifacts.sh rests on: arms sharing an inode stay independent."""
        arm_a = write_blob(os.path.join(self.dir, "arm_a.bin"), b"\x11" * MIB)
        arm_b = os.path.join(self.dir, "arm_b.bin")
        os.link(arm_a, arm_b)
        self.assertEqual(os.stat(arm_a).st_ino, os.stat(arm_b).st_ino)

        write_blob(arm_a, b"\x22" * MIB)

        self.assertNotEqual(os.stat(arm_a).st_ino, os.stat(arm_b).st_ino)
        with open(arm_b, "rb") as f:
            self.assertEqual(f.read(), b"\x11" * MIB)

    def test_no_tmp_file_is_left_behind(self):
        self.roundtrip("leftover.bin", b"\x05" * MIB)
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [])


class BlobPool(unittest.TestCase):
    """The pool is what keeps a ladder of arms from costing 8-12 GB each."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.pool = os.path.join(self.dir, "blobs")

    def arm(self, name):
        d = os.path.join(self.dir, name, "buffers")
        os.makedirs(d, exist_ok=True)
        return d

    def blob(self, arm, name, data):
        return write_blob(os.path.join(self.arm(arm), f"{name}.bin"), data)

    def pool_entries(self):
        return sorted(os.listdir(self.pool)) if os.path.isdir(self.pool) else []

    def test_two_arms_packing_the_same_weights_share_one_inode(self):
        w = b"\xab" * (2 * MIB)
        a = self.blob("decode_s6912", "W_embed", w)
        b = self.blob("decode_s262144", "W_embed", w)
        self.assertEqual(os.stat(a).st_ino, os.stat(b).st_ino)
        self.assertEqual(len(self.pool_entries()), 1)
        for p in (a, b):
            with open(p, "rb") as f:
                self.assertEqual(f.read(), w)

    def test_differing_weights_do_not_share(self):
        a = self.blob("bf16", "Wqkv", b"\x01" * MIB)
        b = self.blob("int8", "Wqkv", b"\x02" * MIB)
        self.assertNotEqual(os.stat(a).st_ino, os.stat(b).st_ino)
        self.assertEqual(len(self.pool_entries()), 2)

    def test_rebuilding_one_arm_leaves_its_sharers_alone(self):
        w = b"\x11" * MIB
        a = self.blob("arm_a", "W_head", w)
        b = self.blob("arm_b", "W_head", w)
        self.assertEqual(os.stat(a).st_ino, os.stat(b).st_ino)

        self.blob("arm_a", "W_head", b"\x22" * MIB)

        self.assertNotEqual(os.stat(a).st_ino, os.stat(b).st_ino)
        with open(b, "rb") as f:
            self.assertEqual(f.read(), w)

    def test_rebuilding_with_identical_bytes_is_idempotent(self):
        w = b"\x33" * MIB
        a = self.blob("arm_a", "Wd", w)
        b = self.blob("arm_b", "Wd", w)
        self.blob("arm_a", "Wd", w)
        self.assertEqual(os.stat(a).st_ino, os.stat(b).st_ino)
        self.assertEqual(len(self.pool_entries()), 1)

    def test_pooling_preserves_holes(self):
        path = self.blob("arm_a", "kc", bytes(32 * MIB))
        probe = write_blob(os.path.join(self.dir, "probe"), bytes(8 * MIB))
        if disk_bytes(probe) != 0:
            self.skipTest("filesystem does not deallocate zero runs")
        self.assertEqual(disk_bytes(path), 0)

    def test_disabled_by_env(self):
        prior = os.environ.get("XDNA_BLOB_POOL")
        self.addCleanup(lambda: os.environ.__setitem__("XDNA_BLOB_POOL", prior)
                        if prior is not None else os.environ.pop("XDNA_BLOB_POOL", None))
        os.environ["XDNA_BLOB_POOL"] = "0"
        w = b"\x44" * MIB
        a = self.blob("arm_a", "W_embed", w)
        b = self.blob("arm_b", "W_embed", w)
        self.assertNotEqual(os.stat(a).st_ino, os.stat(b).st_ino)
        self.assertEqual(self.pool_entries(), [])

    def test_a_path_outside_buffers_is_not_pooled(self):
        loose = write_blob(os.path.join(self.dir, "loose.bin"), b"\x55" * MIB)
        self.assertEqual(os.stat(loose).st_nlink, 1)
        self.assertEqual(self.pool_entries(), [])

    def test_no_temporaries_are_left_behind(self):
        self.blob("arm_a", "W_embed", b"\x66" * MIB)
        self.blob("arm_b", "W_embed", b"\x66" * MIB)
        for arm in ("arm_a", "arm_b"):
            leftovers = [f for f in os.listdir(self.arm(arm)) if f.endswith((".tmp", ".lnk"))]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
