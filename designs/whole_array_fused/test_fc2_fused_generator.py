"""Generator-connected FC2 fusion contract tests.

These inspect the MLIR emitted by the real entry point.  Negative IR mutations
are structural schedule controls, not claims about AIE numerical execution.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "designs/whole_array_fused/whole_array_modal_iron.py"
BASE = ["--dev", "npu2", "-M", "256", "-K", "1024", "-N", "1024", "-m", "16", "-k", "32", "-n", "128", "--n-aie-cols", "8", "--b-col-maj", "0", "--no-silu", "--emulate-bf16-mmul-with-bfp16", "1", "--k-loop-rtp", "--fc2-fused-ieee-rne"]


def emitted_mlir(extra=()):
    if not os.environ.get("MLIR_AIE_INSTANCE"):
        raise unittest.SkipTest("requires the pinned MLIR-AIE generator environment")
    env = os.environ.copy()
    env.update({"WA_C_DEPTH": "1", "WA_AB_DEPTH": "2", "WA_L3L2_DEPTH": "2"})
    return subprocess.run([sys.executable, str(GENERATOR), *BASE, *extra], env=env,
                          text=True, capture_output=True, check=True).stdout


def dma_walk(text, elements, sizes, strides):
    """Ordered coordinate walk from actual BD offsets, independently of generator code."""
    pattern = (rf"memref<{elements}x(?:bf16|f32)> offset = (\d+) len = \d+ "
               rf"sizes = \[{', '.join(map(str, sizes))}\] strides = \[{', '.join(map(str, strides))}\]")
    walks = []
    for offset in map(int, re.findall(pattern, text)):
        # Coordinate-dependent sentinels expose panel/K, row, and column swaps.
        values = []
        for i0 in range(sizes[0]):
            for i1 in range(sizes[1]):
                for i2 in range(sizes[2]):
                    for i3 in range(sizes[3]):
                        address = offset + i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3]
                        values.append((address, 1000000007 * i0 + 1000003 * i1 + 1009 * i2 + i3))
        walks.append(values)
    return walks


def validate_fc2_ir(text):
    a = dma_walk(text, 1048576, [4, 32, 16, 32], [262144, 32, 1024, 1])
    w = dma_walk(text, 4194304, [4, 32, 32, 128], [1048576, 32768, 1024, 1])
    c = dma_walk(text, 262144, [1, 2, 64, 128], [0, 65536, 1024, 1])
    if not (len(a) == 16 and len(w) == 32 and len(c) == 16):
        raise AssertionError("incomplete ordered A/W/C DMA task sequence")
    # Exact first/last coordinate walks, not offset sets. A row tiles advance
    # 16 rows; B/C column tiles advance 128 columns in producer order.
    expected_a_rows = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
    if [x[0][0] for x in a] != [row * 16 * 1024 for row in expected_a_rows]:
        raise AssertionError("A row walk differs from the FC2 row-major broadcast schedule")
    expected_w = [col * 128 for col in range(8) for _ in range(2)] * 2
    if [x[0][0] for x in w] != expected_w:
        raise AssertionError("W2 column walk differs from row-major [4096,1024]")
    expected_c = [col * 128 for col in range(8)] + [128 * 1024 + col * 128 for col in range(8)]
    if [x[0][0] for x in c] != expected_c:
        raise AssertionError("C walk differs from row-major [256,1024]")
    validate_worker_schedules(text)
    return a, w, c


def worker_bodies(text):
    # Each core ends with this unique top-level terminator; inner scf blocks are
    # more deeply indented. The test covers all 32 emitted core bodies.
    return re.findall(r"aie\.core\([^\n]+\) \{(.*?)^    \} \{stack_size", text,
                      flags=re.MULTILINE | re.DOTALL)


def matching_block(text, open_brace):
    """Return the text in one MLIR brace block; fail rather than cross a core."""
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:index]
    raise AssertionError("unterminated MLIR block")


def validate_worker_schedules(text):
    workers = worker_bodies(text)
    if len(workers) != 32:
        raise AssertionError("expected all 32 FC2 worker bodies")
    for worker_number, body in enumerate(workers):
        # No operation may intervene between acquiring C and seeding it with +0.
        seed = re.search(r"(?P<c>%\d+) = aie\.objectfifo\.acquire @C_[^(]+\(Produce, 1\)[^\n]*\n\s*func\.call @zero_f32\((?P=c)\)", body)
        if not seed:
            raise AssertionError(f"worker {worker_number}: C acquire/+0 seed is missing, delayed, or retargeted")
        c = seed.group("c")
        partial_names = re.findall(r"func\.call @zero_f32\((%fc2_partial_\d+_\d+)\)", body)
        if len(partial_names) != 4 or len(set(partial_names)) != 1:
            raise AssertionError(f"worker {worker_number}: expected four resets of one partial tile")
        partial = partial_names[0]
        reset_positions = [m.start() for m in re.finditer(rf"func\.call @zero_f32\({re.escape(partial)}\)", body)]
        if seed.end() > reset_positions[0]:
            raise AssertionError(f"worker {worker_number}: C seed is not before first partial reset")
        release = re.search(r"aie\.objectfifo\.release @C_[^(]+\(Produce, 1\)", body)
        if not release or release.start() < reset_positions[-1] or len(re.findall(r"aie\.objectfifo\.release @C_[^(]+\(Produce, 1\)", body)) != 1:
            raise AssertionError(f"worker {worker_number}: C release is early or not unique")
        expected_left = [c, None, c, None]
        expected_out = [None, c, None, c]
        previous_out = None
        for panel, start in enumerate(reset_positions):
            end = reset_positions[panel + 1] if panel < 3 else release.start()
            group = body[start:end]
            # Resolve the actual SSA loop bounds, then inspect only its matched
            # brace block. A matmul elsewhere in this reset group is not a K32
            # reduction and must not satisfy the schedule contract.
            constants = {name: int(value) for name, value in re.findall(
                r"(%c\d+[^\s]*) = arith\.constant (\d+) : index", group)}
            kloop = re.search(r"scf\.for %\w+ = (?P<lower>%\w+) to (?P<upper>%\w+) step (?P<step>%\w+) \{", group)
            if not kloop or any(name not in constants for name in kloop.group("lower", "upper", "step")):
                raise AssertionError(f"worker {worker_number} panel {panel}: unresolved K loop")
            if tuple(constants[name] for name in kloop.group("lower", "upper", "step")) != (0, 32, 1):
                raise AssertionError(f"worker {worker_number} panel {panel}: K loop is not 0..32 step 1")
            loop_body = matching_block(group, kloop.end() - 1)
            acquired_a = re.search(r"(?P<a>%\d+) = aie\.objectfifo\.acquire @A_[^(]+\(Consume, 1\)", loop_body)
            acquired_b = re.search(r"(?P<b>%\d+) = aie\.objectfifo\.acquire @B_[^(]+\(Consume, 1\)", loop_body)
            matmul = (acquired_a and acquired_b and re.search(
                rf"@matmul_bf16_f32\({re.escape(acquired_a.group('a'))}, {re.escape(acquired_b.group('b'))}, {re.escape(partial)}\)", loop_body))
            material = re.search(rf"@mm_modal_epilogue_f32_f32\({re.escape(partial)}, {re.escape(partial)},", group)
            if not matmul:
                raise AssertionError(f"worker {worker_number} panel {panel}: no acquired A/B matmul inside K loop")
            add = re.search(r"@acc_add_row_ieee_rne\((%[^,]+), (%[^,]+), (%[^,]+), %c2048", group)
            if not material or not add or material.start() > add.start():
                raise AssertionError(f"worker {worker_number} panel {panel}: materialization/add order changed")
            left, add_partial, out = add.groups()
            if add_partial != partial or len({left, add_partial, out}) != 3:
                raise AssertionError(f"worker {worker_number} panel {panel}: exact-add partial/alias ABI changed")
            if panel == 0:
                alternate = out
                if left != c:
                    raise AssertionError(f"worker {worker_number}: first add is not seeded C + p0")
            else:
                if left != previous_out or out != (c if panel % 2 else alternate):
                    raise AssertionError(f"worker {worker_number} panel {panel}: exact-add order changed")
            previous_out = out
        if previous_out != c:
            raise AssertionError(f"worker {worker_number}: final exact add does not return to C")


class FC2FusedGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mlir = emitted_mlir()

    def test_actual_entry_point_rejects_modal_and_layout_contract_changes(self):
        for extra in (("--mode-resadd2a", "--rtp-mode-resadd2a"), ("--a-panel-width", "512"),
                      ("--b-col-maj", "1"), ("--c-panel-width", "1024"), ("--mode-resadd",)):
            with self.subTest(extra=extra):
                run = subprocess.run([sys.executable, str(GENERATOR), *BASE, *extra],
                                     env=os.environ.copy(), text=True, capture_output=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("fc2 fused IEEE-RNE rejects incompatible", run.stderr)

    def test_actual_dma_walk_and_worker_schedule(self):
        a, w, c = validate_fc2_ir(self.mlir)
        # Panel/K/row/column-dependent sentinels differ at every coordinate;
        # compare complete ordered walk fingerprints rather than uniform values.
        self.assertNotEqual(a[0], a[1])
        self.assertNotEqual(w[0], w[2])
        self.assertNotEqual(c[0], c[1])

    def test_known_bad_ir_mutations_fail_the_actual_contract_checker(self):
        reset = "func.call @zero_f32(%fc2_partial_0_0)"
        seed = "func.call @zero_f32(%64)"
        moved_reset = self.mlir.replace(reset, "", 1)
        material = "func.call @mm_modal_epilogue_f32_f32(%fc2_partial_0_0, %fc2_partial_0_0, %rtp_0_0)"
        # Deliberately move, rather than delete, the reset behind the reduction.
        moved_reset = moved_reset.replace(material, reset + "\n          " + material, 1)
        delayed_seed = self.mlir.replace(seed, "", 1).replace(reset, reset + "\n          " + seed, 1)
        matmul_line = re.search(r"func\.call @matmul_bf16_f32\([^\n]+", self.mlir).group(0)
        outside_matmul = self.mlir.replace("func.call @matmul_bf16_f32", "func.call @missing_matmul", 1)
        outside_matmul = outside_matmul.replace(material, matmul_line + "\n          " + material, 1)
        early_release = "aie.objectfifo.release @C_L1L2_0_0(Produce, 1)"
        mutations = {
            "W2-stride": self.mlir.replace("[1048576, 32768, 1024, 1]", "[1048576, 32768, 1, 1024]", 1),
            "C-offset": self.mlir.replace("offset = 128 len = 16384", "offset = 0 len = 16384", 1),
            "native-add": self.mlir.replace("func.call @acc_add_row_ieee_rne", "func.call @native_float_add", 1),
            "retargeted-partial-reset": self.mlir.replace(reset, "func.call @zero_f32(%64)", 1),
            "moved-partial-reset": moved_reset,
            "K31-bound": self.mlir.replace("%c32 = arith.constant 32 : index", "%c32 = arith.constant 31 : index", 1),
            "K-lower-1": self.mlir.replace("%c0_71 = arith.constant 0 : index", "%c0_71 = arith.constant 1 : index", 1),
            "K-step-2": self.mlir.replace("%c1_72 = arith.constant 1 : index", "%c1_72 = arith.constant 2 : index", 1),
            "delayed-C-seed": delayed_seed,
            "missing-matmul": self.mlir.replace("func.call @matmul_bf16_f32", "func.call @missing_matmul", 1),
            "matmul-outside-K-loop": outside_matmul,
            "missing-materialization": self.mlir.replace("func.call @mm_modal_epilogue_f32_f32", "func.call @missing_epilogue", 1),
            "early-C-release": self.mlir.replace(reset, early_release + "\n          " + reset, 1),
            # Reorder the second add to consume C rather than alternate.
            "reordered-add": self.mlir.replace("@acc_add_row_ieee_rne(%fc2_alternate_0_0,", "@acc_add_row_ieee_rne(%64,", 1),
            # Alias exact-add output with its partial input.
            "aliased-add": self.mlir.replace("%fc2_alternate_0_0, %c2048_i32)", "%fc2_partial_0_0, %c2048_i32)", 1),
        }
        for name, bad in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(bad, self.mlir, "mutation did not target emitted MLIR")
                with self.assertRaises(AssertionError):
                    validate_fc2_ir(bad)


if __name__ == "__main__":
    unittest.main()
