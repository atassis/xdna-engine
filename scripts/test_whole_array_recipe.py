#!/usr/bin/env python3
import hashlib
import json
import tempfile
import pathlib
import subprocess
import sys
import unittest
import tomllib


SCRIPT = pathlib.Path(__file__).with_name("kernel_families") / "whole_array_recipe.py"


class WholeArrayRecipeTest(unittest.TestCase):
    def run_recipe(self, stem):
        return subprocess.run(
            [sys.executable, str(SCRIPT), stem],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_resolves_served_modal_and_espcn_stems(self):
        modal = self.run_recipe("512x1024x4096_64x32x128_8c_modalsilukrtp")
        self.assertEqual(modal.returncode, 0, modal.stderr)
        self.assertIn("Makefile.modal", modal.stdout)
        self.assertIn("build/final_512x1024x4096_64x32x128_8c_modalsilukrtp.xclbin", modal.stdout)

        espcn = self.run_recipe("512x576x256_32x32x32_8c")
        self.assertEqual(espcn.returncode, 0, espcn.stderr)
        self.assertIn("K=576", espcn.stdout)
        self.assertIn("N=256", espcn.stdout)

        panel = self.run_recipe("512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024")
        self.assertEqual(panel.returncode, 0, panel.stderr)
        self.assertIn("dtype_out=bf16", panel.stdout)
        self.assertIn("c_panel_width=1024", panel.stdout)

    def test_parakeet_declares_dispatch_streams(self):
        repo = SCRIPT.parents[2]
        scenario = tomllib.loads((repo / "scenarios/asr.toml").read_text())
        required = scenario["kernels"]["whole_array"]["required"]
        for suffix in ("", "krtp"):
            for n in (1024, 2048):
                stem = f"512x1024x{n}_64x32x128_8c_modalid{suffix}"
                self.assertIn(stem, required)
                result = self.run_recipe(stem)
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("512x4096x1024_64x32x128_8c_modalidkrtpapanel1024", required)

    def test_resolves_parakeet_panel_input_stream(self):
        stem = "512x4096x1024_64x32x128_8c_modalidkrtpapanel1024"
        result = self.run_recipe(stem)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("k_loop_rtp=1", result.stdout)
        self.assertIn("a_panel_width=1024", result.stdout)
        self.assertIn("no_silu=1", result.stdout)
        self.assertIn(f"build/final_{stem}.xclbin", result.stdout)

    def test_larger_m_native_and_invalid_row_grid(self):
        for mode in ("silu", "id"):
            stem = f"1024x800x3072_32x32x32_8c_modal{mode}nat"
            result = self.run_recipe(stem)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("M=1024", result.stdout)
            self.assertNotIn("emulate_bfloat16_mmul_with_bfp16=1", result.stdout)
            self.assertIn(f"final_{stem}.xclbin", result.stdout)
        for rows in (0, 513, 1025):
            result = self.run_recipe(f"{rows}x800x3072_32x32x32_8c_modalidnat")
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("M", result.stderr)

    def test_native_contract_reads_emitted_shapes_and_hashes_actual_artifacts(self):
        for n, variant in [(3072, "modalidnat"), (768, "modalidnat"), (3072, "modalsilunat")]:
            stem = f"1024x800x{n}_32x32x32_8c_{variant}"
            with tempfile.TemporaryDirectory() as raw:
                directory = pathlib.Path(raw)
                mlir = directory / f"aie_{stem}.mlir"
                xclbin = directory / f"final_{stem}.xclbin"
                insts = directory / f"insts_{stem}.txt"
                xclbin.write_bytes(b"fixture-xclbin")
                insts.write_bytes(b"00000001\n")
                def record():
                    return subprocess.run([sys.executable, str(SCRIPT), "--record-native-contract", stem, raw],
                                          text=True, capture_output=True, check=False)
                mlir.write_text(f"aie.runtime_sequence(%a: memref<409600xbf16>, %b: memref<{800*n}xbf16>, %c: memref<{512*n}xf32>) {{}}")
                bad = record()
                self.assertEqual(bad.returncode, 1)
                self.assertIn("operand shapes", bad.stderr)
                receipt = directory / f"contract_{stem}.json"
                self.assertFalse(receipt.exists(), "512-row source must not certify a 1024-row filename")
                mlir.write_text(f"aie.runtime_sequence(%a: memref<819200xbf16>, %b: memref<{800*n}xbf16>, %c: memref<{1024*n}xf32>) {{}}")
                good = record()
                self.assertEqual(good.returncode, 0, good.stderr)
                contract = json.loads(receipt.read_text())
                self.assertEqual((contract["rows"], contract["k"], contract["n"]), (1024, 800, n))
                self.assertEqual(contract["xclbin_sha256"], hashlib.sha256(xclbin.read_bytes()).hexdigest())
                self.assertEqual(contract["insts_sha256"], hashlib.sha256(insts.read_bytes()).hexdigest())

    def test_rejects_unknown_variant(self):
        result = self.run_recipe("512x800x3072_64x32x96_8c_not_a_recipe")
        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing", result.stderr)


if __name__ == "__main__":
    unittest.main()
