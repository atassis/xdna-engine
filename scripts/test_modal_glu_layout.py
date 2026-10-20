#!/usr/bin/env python3
"""Check the row-major mm.cc blocked-C contract used by the GLU epilogue."""

import unittest


def blocked_coordinates(rows, cols, r, t):
    return [
        (row_base + row, col_base + col)
        for row_base in range(0, rows, r)
        for col_base in range(0, cols, t)
        for row in range(r)
        for col in range(t)
    ]


def half_block_pairs(rows, cols, r, t):
    order = blocked_coordinates(rows, cols, r, t)
    offsets = {coordinate: i for i, coordinate in enumerate(order)}
    half = r * cols // 2
    return [
        offsets[row, col] + half < len(order)
        and order[offsets[row, col] + half] == (row, col + cols // 2)
        for row in range(rows)
        for col in range(cols // 2)
    ]


class ModalGluLayoutTest(unittest.TestCase):
    def test_rectangular_native_and_square_fast_pair_the_same_logical_columns(self):
        # Native uses 4x8 C sub-tiles, fast BF16 uses 8x8. Square is not a precondition.
        for r, t in ((4, 8), (8, 8), (4, 4)):
            for cols in (32, 128, 768, 3072):
                with self.subTest(r=r, t=t, cols=cols):
                    pairs = half_block_pairs(32, cols, r, t)
                    self.assertEqual(len(pairs), 32 * cols // 2)
                    self.assertTrue(all(pairs))

    def test_misaligned_value_gate_split_is_a_known_bad_layout(self):
        # Both full tiles fit, but the 12-column split cuts through an 8-column sub-tile.
        pairs = half_block_pairs(32, 24, 4, 8)
        self.assertTrue(pairs)
        self.assertFalse(all(pairs))
        self.assertNotEqual(24 % (2 * 8), 0)

    def test_pairing_stays_inside_its_own_row_block(self):
        order = blocked_coordinates(32, 32, 4, 8)
        half = 4 * 32 // 2
        for base in range(0, len(order), 4 * 32):
            for offset in range(half):
                row, col = order[base + offset]
                self.assertEqual(order[base + offset + half], (row, col + 16))


if __name__ == "__main__":
    unittest.main()
