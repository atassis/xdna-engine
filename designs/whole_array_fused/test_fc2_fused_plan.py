"""CPU contract tests for the same-core four-partial FC2 schedule."""
import unittest

import numpy as np

from designs.whole_array_fused.fc2_fused_plan import (
    FC2FusedConfig,
    fused_fc2_reference,
    plan_accesses,
)


class FC2FusedPlanTests(unittest.TestCase):
    def test_rejects_nonqualified_geometry(self):
        for kwargs in (
            {"m": 32}, {"k": 64}, {"n": 64}, {"rows": 128}, {"partial_k": 2048},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    FC2FusedConfig(**kwargs)

    def test_complete_panel_weight_output_walk_is_discriminating(self):
        config = FC2FusedConfig()
        accesses = plan_accesses(config)
        self.assertEqual(len(accesses), config.rows // config.m * config.cols // config.n * 4)
        self.assertEqual({x.partial for x in accesses}, {0, 1, 2, 3})
        self.assertEqual({x.k_tiles for x in accesses}, {32})
        self.assertEqual({x.output_offset for x in accesses},
                         {row * config.cols + col
                          for row in range(0, config.rows, config.m)
                          for col in range(0, config.cols, config.n)})
        self.assertEqual({x.a_panel_offset for x in accesses},
                         {i * config.rows * config.partial_k for i in range(4)})
        self.assertEqual({x.w_panel_offset for x in accesses},
                         {i * config.partial_k * config.cols for i in range(4)})

        # These sentinels fail if A panels, W2 slices, or the final C tile order swap.
        a = np.zeros((4, config.rows, config.partial_k), np.float32)
        w = np.zeros((4, config.partial_k, config.cols), np.float32)
        for partial in range(4):
            a[partial, :, 0] = partial + 1
            w[partial, 0, :] = (partial + 1) * 10
        result = fused_fc2_reference(a, w, config)
        self.assertTrue(np.all(result == np.float32(300.0)))
        swapped = fused_fc2_reference(a[[1, 0, 2, 3]], w, config)
        self.assertFalse(np.array_equal(swapped, result))

    def test_ordered_zero_seed_rejects_tree_and_continuous_substitute(self):
        config = FC2FusedConfig()
        partials = np.empty((4, config.rows, config.cols), np.float32)
        partials[0].fill(np.float32(1e20))
        partials[1].fill(np.float32(1.0))
        partials[2].fill(np.float32(-1e20))
        partials[3].fill(np.float32(3.0))
        got = fused_fc2_reference(None, None, config, partials=partials)
        self.assertTrue(np.all(got == np.float32(3.0)))
        tree = ((partials[0] + partials[2]) + (partials[1] + partials[3])).astype(np.float32)
        self.assertFalse(np.array_equal(tree, got))
        with self.assertRaises(ValueError):
            fused_fc2_reference(None, None, config, partials=partials[:3])

    def test_positive_zero_seed_is_observable_for_negative_zero_partial(self):
        config = FC2FusedConfig()
        partials = np.full((4, config.rows, config.cols), np.float32(-0.0))
        seeded = fused_fc2_reference(None, None, config, partials=partials)
        # The known-bad omitted seed starts with p0 and retains its negative sign.
        omitted_seed = partials[0].copy()
        for partial in partials[1:]:
            omitted_seed = np.add(omitted_seed, partial, dtype=np.float32)
        self.assertTrue(np.all(seeded.view(np.uint32) == 0))
        self.assertTrue(np.all(omitted_seed.view(np.uint32) == 0x80000000))


if __name__ == "__main__":
    unittest.main()
