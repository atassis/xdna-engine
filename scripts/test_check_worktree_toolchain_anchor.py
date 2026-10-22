#!/usr/bin/env python3
"""CPU-only filesystem regressions for toolchain carrier retirement checks."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(os.environ.get('ANCHOR_SCRIPT', Path(__file__).with_name('check_worktree_toolchain_anchor.sh')))


class ToolchainAnchorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='toolchain-anchor-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.git('init', '-q', str(self.repo))
        self.git('-C', str(self.repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '-q', '--allow-empty', '-m', 'fixture')
        self.carrier = self.root / 'carrier'
        self.git('-C', str(self.repo), 'worktree', 'add', '-q', '--detach', str(self.carrier))
        self.admin = Path(self.git('-C', str(self.carrier), 'rev-parse', '--absolute-git-dir').stdout.strip())
        self.instances = self.root / 'instances'
        self.source = self.instances / 'sample' / 'src'
        self.source.parent.mkdir(parents=True)
        self.git('-C', str(self.repo), 'worktree', 'add', '-q', '--detach', str(self.source))

    def git(self, *args):
        return subprocess.run(['git', *args], check=True, text=True, capture_output=True)

    def check(self):
        return subprocess.run(['bash', str(SCRIPT), str(self.repo), str(self.carrier), str(self.instances)],
                              text=True, capture_output=True)

    def gitfile(self, relative, target):
        path = self.source / relative / '.git'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('gitdir: ' + str(target) + '\n')
        return path

    def test_unrelated_source_is_safe(self):
        self.assertEqual(self.check().returncode, 0)

    def test_outer_administration_anchor_refused(self):
        self.gitfile('.', self.admin / 'modules/compiler/worktrees/source')
        self.assertEqual(self.check().returncode, 1)

    def test_relative_outer_administration_anchor_refused(self):
        self.gitfile('.', os.path.relpath(self.admin / 'modules/compiler', self.source))
        self.assertEqual(self.check().returncode, 1)

    def test_outer_in_tree_anchor_refused(self):
        self.gitfile('.', self.carrier / 'compiler/.git/worktrees/source')
        self.assertEqual(self.check().returncode, 1)

    def test_nested_administration_anchor_refused(self):
        self.gitfile('third_party/aiebu', self.admin / 'modules/compiler/worktrees/source/modules/aiebu')
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('third_party/aiebu/.git', result.stderr)

    def test_deep_relative_administration_anchor_refused(self):
        directory = self.source / 'third_party/aiebu/src/cpp/ELFIO'
        target = self.admin / 'modules/compiler/worktrees/source/modules/aiebu/modules/ELFIO'
        self.gitfile('third_party/aiebu/src/cpp/ELFIO', os.path.relpath(target, directory))
        self.assertEqual(self.check().returncode, 1)

    def test_absolute_source_symlink_refused(self):
        directory = self.source / 'third_party'
        directory.mkdir()
        (directory / 'aie_api').symlink_to(self.carrier / 'compiler/third_party/aie_api')
        self.assertEqual(self.check().returncode, 1)

    def test_relative_source_symlink_refused(self):
        directory = self.source / 'third_party'
        directory.mkdir()
        target = self.carrier / 'compiler/third_party/aie_api'
        (directory / 'aie_api').symlink_to(os.path.relpath(target, directory))
        self.assertEqual(self.check().returncode, 1)

    def test_source_link_to_entire_carrier_refused(self):
        (self.source / 'linked-source').symlink_to(self.carrier)
        self.assertEqual(self.check().returncode, 1)

    def test_source_symlink_loop_is_not_safe(self):
        (self.source / 'loop').symlink_to('loop')
        self.assertEqual(self.check().returncode, 2)

    def test_sibling_prefix_is_not_anchor(self):
        self.gitfile('third_party/aiebu', Path(str(self.carrier) + '-other') / '.git/worktrees/source')
        self.assertEqual(self.check().returncode, 0)

    def test_malformed_nested_gitfile_is_not_safe(self):
        path = self.gitfile('third_party/aiebu', self.repo / '.git')
        path.write_text('not a gitdir pointer\n')
        self.assertEqual(self.check().returncode, 2)


if __name__ == '__main__':
    unittest.main()
