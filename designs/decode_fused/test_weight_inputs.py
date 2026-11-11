# SPDX-License-Identifier: Apache-2.0
import unittest
from weight_inputs import WeightInputs


class WeightInputTests(unittest.TestCase):
    def test_records_graph_order_scaling_and_explicit_padding(self):
        r = WeightInputs(True)
        r.rows('q', 'q_proj', [4, 2], first=2, count=2)
        r.rows('k', 'k_proj', [2, 2])
        r.concat('qkv', ['q', 'k'])
        r.pad('qkv', 4)
        r.rows('n', 'q_norm', [2], scale=0.125)
        r.rows('head', 'embedding', [3, 2])
        result = r.finish({'qkv': 20, 'n': 4, 'head': 12}, 'embedding', [3, 2], {'model_type': 'fixture'})
        self.assertEqual(result['buffers']['qkv'], [
            dict(op='rows', tensor='q_proj', shape=[4, 2], first=2, count=2),
            dict(op='rows', tensor='k_proj', shape=[2, 2], first=0, count=2),
            dict(op='zero', bytes=4)])
        self.assertEqual(result['buffers']['n'][0]['scale'], 0.125)
        self.assertNotIn('q', result['buffers'])

    def test_incomplete_or_unsupported_graph_does_not_claim_a_contract(self):
        r = WeightInputs(True)
        r.rows('w', 'weight', [2, 2])
        self.assertIsNone(r.finish({'w': 8, 'unrecorded': 2}, 'weight', [2, 2], {'model_type': 'fixture'}))
        self.assertIsNone(WeightInputs(False).finish({'w': 8}, 'weight', [2, 2], {'model_type': 'fixture'}))
        with self.assertRaisesRegex(ValueError, 'byte extent'):
            r.finish({'w': 10}, 'weight', [2, 2], {'model_type': 'fixture'})


if __name__ == '__main__':
    unittest.main()
