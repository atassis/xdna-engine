import hashlib
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "lib" / "init_source_aiebu.sh"


class ReadOnlyToolchainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        (self.repo / "scripts/lib").mkdir(parents=True)
        shutil.copy2(REPO / "scripts/toolchain_up.sh", self.repo / "scripts/toolchain_up.sh")
        shutil.copy2(REPO / "scripts/lib/data_root.sh", self.repo / "scripts/lib/data_root.sh")
        self.trace = self.root / "unexpected-setup"
        for helper in ("lib/init_source_aiebu.sh", "lib/fork_python_binding.sh",
                       "fast_build_env.sh", "toolchain_gc.sh"):
            (self.repo / "scripts" / helper).write_text(f'echo setup >> "{self.trace}"\nexit 99\n')
        source = self.root / "pinned-source"
        source.mkdir()
        subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "Fixture"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "fixture@invalid"], check=True)
        (source / "fixture.txt").write_text("pinned\n")
        subprocess.run(["git", "-C", str(source), "add", "fixture.txt"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "pinned fixture"], check=True)
        self.pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        self.lock_pin = self.pin[:12]
        self.lock = f"MLIR_AIE_FORK_COMMIT={self.lock_pin} # fixture\nMLIR_DISTRO_DIR=distro\nIRON_SOURCE_COMMIT=one\n"
        (self.repo / "toolchain.lock").write_text(self.lock)
        distro = self.repo / "distro/bin"
        distro.mkdir(parents=True)
        (distro / "mlir-tblgen").touch()
        semantic = f"MLIR_AIE_FORK_COMMIT={self.lock_pin}\nMLIR_DISTRO_DIR=distro\n"
        self.key = hashlib.sha256(semantic.encode()).hexdigest()[:12]
        self.home = self.root / "instances"
        self.instance = self.home / self.key
        (self.instance / "python/aie/iron").mkdir(parents=True)
        (self.instance / "python/aie/iron/program.py").write_text(
            "class Program:\n    def resolve_program(self, device_name): pass\n")
        subprocess.run(["git", "-C", str(source), "worktree", "add", "--detach",
                        str(self.instance / "src"), self.pin], check=True, stdout=subprocess.DEVNULL)
        (self.repo / ".venv-iron/bin").mkdir(parents=True)
        (self.repo / ".venv-iron/bin/python").symlink_to(shutil.which("python3"))
        self.tools = ("aiecc", "aie-opt", "aie-translate", "aiebu-asm", "bootgen")
        (self.instance / "bin").mkdir()
        for tool in self.tools:
            p = self.instance / "bin" / tool
            p.write_text(f'#!/bin/sh\necho tool >> "{self.trace}"\nexit 99\n')
            p.chmod(0o755)
        include = self.instance / "build/include/aie_api"
        include.mkdir(parents=True)
        (include / "aie.hpp").touch()
        self.env = dict(os.environ, TOOLCHAIN_RESOLVE_ONLY="1", TOOLCHAIN_HOME=str(self.home),
                        XDNA_DATA=str(self.root / "data"), XDNA_CACHE=str(self.root / "cache"))

    def set_lock_pin(self, pin):
        lock = f"MLIR_AIE_FORK_COMMIT={pin} # fixture\nMLIR_DISTRO_DIR=distro\nIRON_SOURCE_COMMIT=one\n"
        (self.repo / "toolchain.lock").write_text(lock)
        semantic = f"MLIR_AIE_FORK_COMMIT={pin}\nMLIR_DISTRO_DIR=distro\n"
        key = hashlib.sha256(semantic.encode()).hexdigest()[:12]
        target = self.home / key
        if target != self.instance:
            self.instance.rename(target)
            self.instance = target
        self.lock = lock
        self.key = key

    def snapshot(self):
        return {str(p.relative_to(self.root)): (
                    hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
                    p.stat().st_mtime_ns)
                for p in self.root.rglob("*")}

    def resolve(self):
        before = self.snapshot()
        result = subprocess.run(["bash", str(self.repo / "scripts/toolchain_up.sh")],
                                env=self.env, text=True, capture_output=True)
        self.assertEqual(before, self.snapshot(), result.stderr)
        self.assertFalse(self.trace.exists())
        self.assertFalse((self.root / "cache").exists())
        return result

    def test_ready_instance_resolves_without_setup_or_writes(self):
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.instance))

    def test_missing_tool_is_not_backfilled(self):
        for tool in self.tools:
            with self.subTest(tool=tool):
                p = self.instance / "bin" / tool
                saved = p.read_bytes()
                p.unlink()
                result = self.resolve()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("read-only", result.stderr)
                p.write_bytes(saved)
                p.chmod(0o755)

    def test_missing_python_or_headers_are_not_backfilled(self):
        for name in ("python/aie/iron/program.py", "build/include/aie_api/aie.hpp"):
            with self.subTest(name=name):
                p = self.instance / name
                saved = p.read_bytes()
                p.unlink()
                result = self.resolve()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("read-only", result.stderr)
                p.write_bytes(saved)

    def test_source_commit_mismatch_is_rejected(self):
        source = self.instance / "src"
        (source / "fixture.txt").write_text("different\n")
        subprocess.run(["git", "-C", str(source), "commit", "-am", "wrong source"], check=True,
                       stdout=subprocess.DEVNULL)
        result = self.resolve()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("source", result.stderr.lower())
        self.assertIn(self.pin, result.stderr)

    def test_missing_source_repository_is_rejected(self):
        shutil.copytree(self.root / "pinned-source/.git", self.instance / ".git")
        (self.instance / "src/.git").unlink()
        result = self.resolve()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("source", result.stderr.lower())

    def test_ambient_git_dir_cannot_mask_wrong_source_commit(self):
        source = self.instance / "src"
        (source / "fixture.txt").write_text("different\n")
        subprocess.run(["git", "-C", str(source), "commit", "-am", "wrong source"], check=True,
                       stdout=subprocess.DEVNULL)
        self.env["GIT_DIR"] = str(self.root / "pinned-source/.git")
        result = self.resolve()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("source", result.stderr.lower())

    def test_invalid_pins_are_rejected(self):
        for pin in ("", self.pin[:6], self.pin.upper(), "1234567x"):
            with self.subTest(pin=pin):
                self.set_lock_pin(pin)
                result = self.resolve()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("pin", result.stderr.lower())

    def test_legacy_instance_is_not_adopted(self):
        legacy = self.home / hashlib.sha256(self.lock.encode()).hexdigest()[:12]
        self.instance.rename(legacy)
        result = self.resolve()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.instance.exists())
        self.assertIn("read-only", result.stderr)

    def test_iron_only_change_reuses_the_same_compiler_instance(self):
        (self.repo / "toolchain.lock").write_text(self.lock.replace("IRON_SOURCE_COMMIT=one", "IRON_SOURCE_COMMIT=two"))
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.instance))


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
