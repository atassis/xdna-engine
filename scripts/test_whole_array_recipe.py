#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import unittest


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

    def test_rejects_unknown_variant(self):
        result = self.run_recipe("512x800x3072_64x32x96_8c_not_a_recipe")
        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing", result.stderr)


if __name__ == "__main__":
    unittest.main()
