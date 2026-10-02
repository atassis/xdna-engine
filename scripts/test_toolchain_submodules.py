import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "lib" / "init_source_aiebu.sh"


class AiebuBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = os.environ.copy()
        self.env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="protocol.file.allow",
                        GIT_CONFIG_VALUE_0="always")

    def git(self, repo, *args):
        return subprocess.check_output(
            ["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", *args],
            env=self.env, text=True, stderr=subprocess.STDOUT,
        ).strip()

    def make_repo(self, name):
        repo = self.root / name
        repo.mkdir()
        self.git(repo, "init", "-q")
        (repo / "CMakeLists.txt").write_text("fixture\n")
        self.git(repo, "add", "CMakeLists.txt")
        self.git(repo, "commit", "-qm", "fixture")
        return repo

    def initialize(self, source):
        self.assertTrue(HELPER.is_file(), "pinned-source initializer is missing")
        return subprocess.run(
            ["bash", "-c", 'source "$1"; init_source_aiebu "$2"', "fixture", str(HELPER), str(source)],
            env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_pin_without_aiebu_is_supported(self):
        source = self.make_repo("source")
        result = self.initialize(source)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse((source / "third_party" / "aiebu").exists())

    def test_cold_clone_initializes_the_pinned_recursive_dependency(self):
        leaf = self.make_repo("leaf")
        aiebu = self.make_repo("aiebu")
        self.git(aiebu, "submodule", "add", "-q", str(leaf), "third_party/ELFIO")
        self.git(aiebu, "commit", "-qm", "fixture dependency")
        parent = self.make_repo("parent")
        self.git(parent, "submodule", "add", "-q", str(aiebu), "third_party/aiebu")
        self.git(parent, "commit", "-qm", "fixture dependency")
        cold = self.root / "cold"
        self.git(parent, "clone", "-q", str(parent), str(cold))
        nested = cold / "third_party/aiebu/third_party/ELFIO"
        self.assertFalse((nested / "CMakeLists.txt").exists())
        for _ in range(2):
            result = self.initialize(cold)
            self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue((nested / "CMakeLists.txt").is_file())
        self.assertEqual(self.git(cold / "third_party/aiebu", "rev-parse", "HEAD"),
                         self.git(aiebu, "rev-parse", "HEAD"))
        self.assertEqual(self.git(nested, "rev-parse", "HEAD"), self.git(leaf, "rev-parse", "HEAD"))

    def test_cold_toolchain_calls_the_initializer_before_cmake(self):
        source = (REPO / "scripts" / "toolchain_up.sh").read_text()
        call = 'init_source_aiebu "$SRC"'
        self.assertIn(call, source)
        self.assertLess(source.index(call), source.index("cmake -G Ninja"))


if __name__ == "__main__":
    unittest.main()
