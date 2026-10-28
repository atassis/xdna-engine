import unittest
from types import SimpleNamespace

import kvring
import rforward


class KVRingTotalCountTests(unittest.TestCase):
    def test_parameter_abi_cannot_be_mistaken_for_legacy_extra_counts(self):
        self.assertEqual(getattr(kvring, "PARAMETER_ABI", None), "zero-base-total-v1")
        self.assertEqual(kvring.COUNTS, ("kvw_a_rows", "kvw_b_rows", "kvr_a_blocks", "kvr_b_blocks"))
        self.assertTrue(set(kvring.PARAMS).isdisjoint({"kvw_an", "kvw_bn", "kvr_an", "kvr_bn"}))

    def test_descriptors_use_truthful_extents_and_total_count_units(self):
        write = kvring.write_bd(0, 0, 0)
        self.assertEqual(write[1:5], (0, [1, 64, 4, 64], [0, 64, 4096, 1], 256))
        self.assertEqual(write[5:], ("kvw_a", "kvw_a_rows"))
        read = kvring.read_bd(0, 0, 0, 1024)
        self.assertEqual(read[1:5], (0, [1, 16, 64, 256], [0, 262144, 256, 1], 16384))
        self.assertEqual(read[5:], ("kvr_a", "kvr_a_blocks"))
        text = kvring.bd_text("%kv", write)
        self.assertIn("len = 0 sizes = [1, 64, 4, 64]", text)
        self.assertIn("length_unit = 256 : i32", text)
        self.assertNotIn("length_granule", text)

    def test_parameters_are_total_span_lengths(self):
        # Cross row and ring boundaries; each individual span remains nonempty.
        p = kvring.params(62, 6, 14 * 64, 3, 1024)
        self.assertEqual([p[n] for n in kvring.COUNTS], [2, 4, 2, 1])
        self.assertTrue(all(p[n] >= 1 for n in kvring.COUNTS))
        # No extra-count convention: a one-granule span is represented by one.
        p = kvring.params(0, 2, 0, 2, 1024)
        self.assertEqual([p[n] for n in kvring.COUNTS], [1, 1, 1, 1])

    def test_ordered_addresses_across_row_and_ring_splits(self):
        def write_addresses(start, rows):
            return [kvring.slot_elem((start + r) % 128) + s * 4096 + k
                    for r in range(rows) for s in range(4) for k in range(64)]

        def read_addresses(start_block, blocks):
            return [((start_block + b) % 16) * kvring.BLOCK_E + k
                    for b in range(blocks) for k in range(kvring.SLAB)]

        for start, count in ((0, 2), (62, 6), (127, 2), (125, 4)):
            p = kvring.params(start, count, 0, 2, 128)
            spans = kvring.write_spans(start, count, 128)
            actual = []
            for i, (slot, n) in enumerate(spans):
                self.assertEqual(p[kvring.WRITE[2 * i + 1]], n)
                actual.extend(kvring.slot_elem((slot + r) % 128) + s * 4096 + k
                              for r in range(n) for s in range(4) for k in range(64))
            self.assertEqual(actual, write_addresses(start, count))

        for start_block, count in ((0, 2), (14, 2), (15, 2), (12, 4)):
            p = kvring.params(0, 16, start_block * 64, count, 1024)
            spans = kvring.read_spans(start_block * 64, count, 1024)
            actual = []
            for i, (block, n) in enumerate(spans):
                self.assertEqual(p[kvring.READ[2 * i + 1]], n)
                actual.extend(((block + b) % 16) * kvring.BLOCK_E + k
                              for b in range(n) for k in range(kvring.SLAB))
            self.assertEqual(actual, read_addresses(start_block, count))

    def test_host_fit_rejects_empty_negative_and_overflowing_spans(self):
        desc = kvring.describe(1024, 8, 1024 * kvring.BLOCK_E)
        good = {"kvw_a": 0, "kvw_a_rows": 2, "kvw_b": 2 * kvring.BLOCK_E,
                "kvw_b_rows": 2, "kvr_a": 0, "kvr_a_blocks": 2,
                "kvr_b": 4 * kvring.BLOCK_E, "kvr_b_blocks": 2}
        self.assertEqual(kvring.fits(desc, good), [])
        for key, bad_value in (("kvw_a_rows", 0), ("kvw_a_rows", -1),
                               ("kvw_a", desc["region_bytes"]),
                               ("kvr_a", 2), ("kvr_b_blocks", 1024)):
            bad = good | {key: bad_value}
            self.assertTrue(kvring.fits(desc, bad), (key, bad_value))

    def test_retarget_scales_length_unit_for_i32_and_i64_scratch_elements(self):
        source = ("aie.dma_bd(%kvw : memref<16384xbf16> offset = 0 len = 0 "
                  "sizes = [1, 64, 4, 64] strides = [0, 64, 4096, 1]) "
                  "{offset_parameter = @kvw_a, length_parameter = @kvw_a_rows, "
                  "length_unit = 256 : i32}")
        for unit, write_unit, read_unit in ((4, 128, 8192), (8, 64, 4096)):
            plan = SimpleNamespace(XROWS=1, xbuf_l=1, OB=1, RS=0, RG=0, WS=0, WG=0,
                                  wbase={}, hbase=0, kvbase={0: 0}, S_UNIT=unit,
                                  S_TY=f"i{unit * 8}", KF=1 << 20, SF=1 << 20,
                                  XF=1, OF=1)
            out = rforward.retarget(source, plan, 0, False, "layer")
            self.assertIn(f"length_unit = {write_unit} : i32", out)
            self.assertIn(f"len = 0 sizes = [1, 64, 4, {64 // (unit // 2)}]", out)
            self.assertIn(f"strides = [0, {64 // (unit // 2)}, {4096 // (unit // 2)}, 1]", out)
            read = source.replace("%kvw", "%kvr").replace("16384xbf16", "4194304xbf16")
            read = read.replace("[1, 64, 4, 64]", "[1, 16, 64, 256]")
            read = read.replace("[0, 64, 4096, 1]", "[0, 262144, 256, 1]")
            read = read.replace("length_unit = 256", "length_unit = 16384")
            read = read.replace("kvw_a_rows", "kvr_a_blocks").replace("kvw_a", "kvr_a")
            out = rforward.retarget(read, plan, 0, False, "layer")
            self.assertIn(f"length_unit = {read_unit} : i32", out)


if __name__ == "__main__":
    unittest.main()
