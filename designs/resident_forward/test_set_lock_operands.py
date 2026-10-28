import re
import unittest

from m1_design import M
import rforward


class SetLockOperandTests(unittest.TestCase):
    def test_constants_preserve_values_and_use_unique_ssa_operands(self):
        m = M()
        existing = m.tmp("value")
        values = [2, 0, 1, 63, 2]
        for value in values:
            m.set_lock("%lock", value, 6)
        constants = [re.fullmatch(r"      (%\w+) = arith.constant (-?\d+) : i32", line)
                     for line in m.lines[::2]]
        self.assertTrue(all(constants))
        names = [match.group(1) for match in constants]
        self.assertEqual([int(match.group(2)) for match in constants], values)
        self.assertEqual(len(set(names)), len(values))
        self.assertNotIn(existing, names)
        self.assertEqual(m.lines[1::2],
                         [f"      aiex.set_lock(%lock, {name})" for name in names])

    def test_forward_splicing_renames_constants_and_their_uses(self):
        m = M()
        m("aie.runtime_sequence @p() {")
        m.set_lock("%lock", 2, 6)
        m.set_lock("%lock", 0, 6)
        m("}")
        names = []
        for layer in ("L0", "L1"):
            lines = rforward.splice(m.lines, None, 0, False, "layer", layer)
            constants = re.findall(r"(%\w+) = arith.constant (\d+) : i32", "\n".join(lines))
            self.assertEqual([value for _, value in constants], ["2", "0"])
            for name, _ in constants:
                self.assertIn(f"      aiex.set_lock(%lock, {name})", lines)
                self.assertTrue(name.endswith("_" + layer))
                names.append(name)
        self.assertEqual(len(set(names)), 4)


if __name__ == "__main__":
    unittest.main()
