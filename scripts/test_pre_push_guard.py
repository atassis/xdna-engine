from pathlib import Path
import subprocess
import tempfile
import unittest


HOOK = Path(__file__).resolve().parents[1] / "hooks" / "pre-push"
ZERO = "0" * 40


class PushPathGuardTests(unittest.TestCase):
    def check_path(self, path, root_commit):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(repo), "-c", "user.name=Gate fixture",
                     "-c", "user.email=gate@invalid", *args], text=True
                ).strip()
            git("init", "-q")
            if not root_commit:
                (repo / "README").write_text("fixture\n")
                git("add", "README")
                git("commit", "-qm", "fixture")
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture\n")
            git("add", path)
            git("commit", "-qm", "fixture")
            sha = git("rev-parse", "HEAD")
            return subprocess.run(
                ["bash", str(HOOK)], cwd=repo,
                input=f"refs/heads/main {sha} refs/heads/main {ZERO}\n",
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )

    def test_public_wrapper_is_allowed_in_root_and_later_commits(self):
        for root in (True, False):
            with self.subTest(root=root):
                result = self.check_path("rust/npu-sr/src/lib.rs", root)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_private_path_is_rejected_in_root_and_later_commits(self):
        for root in (True, False):
            with self.subTest(root=root):
                result = self.check_path(Path("internal") / "fixture.md", root)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("forbidden private path", result.stdout)

    def test_unpublished_kernel_is_rejected_in_root_and_later_commits(self):
        for root in (True, False):
            with self.subTest(root=root):
                result = self.check_path("designs/fsr1/kernel.cc", root)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("forbidden private path", result.stdout)


if __name__ == "__main__":
    unittest.main()
