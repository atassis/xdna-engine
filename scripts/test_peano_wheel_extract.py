import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class PeanoWheelExtractionTests(unittest.TestCase):
    def test_executable_modes_survive_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "seed.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                item = zipfile.ZipInfo("llvm-aie/bin/clang")
                item.create_system = 3
                item.external_attr = (stat.S_IFREG | 0o755) << 16
                archive.writestr(item, b"fixture")
            helper = Path(__file__).with_name("extract_peano_wheel.py")
            result = subprocess.run([sys.executable, str(helper), str(wheel), str(root / "out")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "out/llvm-aie/bin/clang").stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
