#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import copy
import errno
import os
import select
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pack_cache
from pack_cache import ContentStore


class PackCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def source(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return str(path)

    def test_put_over_budget_preserves_unleased_paths_until_explicit_prune(self):
        reader = ContentStore(str(self.root / "cache"), max_bytes=8)
        reader.put("a" * 64, self.source("a", b"aaaa"))
        reader.put("b" * 64, self.source("b", b"bbbb"))
        a, b = reader.get("a" * 64), reader.get("b" * 64)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        identities = [(os.stat(path).st_dev, os.stat(path).st_ino) for path in (a, b)]
        lock = Path(reader.root) / ".lock"
        lock_identity = (lock.stat().st_dev, lock.stat().st_ino)
        os.utime(a, ns=(1, 1))
        os.utime(b, ns=(2, 2))

        writer = ContentStore(reader.root, max_bytes=8)
        c = writer.put("c" * 64, self.source("c", b"cccc"))
        self.assertTrue(Path(a).is_file(), "ordinary put invalidated an unleased reader path")
        self.assertEqual([Path(path).read_bytes() for path in (a, b, c)],
                         [b"aaaa", b"bbbb", b"cccc"])
        self.assertEqual([(os.stat(path).st_dev, os.stat(path).st_ino) for path in (a, b)],
                         identities)
        self.assertEqual(sum(p.stat().st_size for p in Path(writer.objects).glob("*/*.bin")), 12)

        writer.prune()
        self.assertFalse(Path(a).exists())
        self.assertEqual([Path(path).read_bytes() for path in (b, c)], [b"bbbb", b"cccc"])
        self.assertEqual(sum(p.stat().st_size for p in Path(writer.objects).glob("*/*.bin")), 8)
        self.assertEqual((lock.stat().st_dev, lock.stat().st_ino), lock_identity)

    def test_environment_budget_does_not_enable_automatic_pruning(self):
        with patch.dict(os.environ, {"PACK_CACHE_MAX_BYTES": "4"}):
            store = ContentStore(str(self.root / "cache"))
            a = store.put("a" * 64, self.source("a", b"aaaa"))
            os.utime(a, ns=(1, 1))
            b = store.put("b" * 64, self.source("b", b"bbbb"))
            self.assertTrue(Path(a).is_file(), "a finite environment budget enabled eviction")
            self.assertEqual(Path(a).read_bytes(), b"aaaa")
            self.assertEqual(Path(b).read_bytes(), b"bbbb")
            self.assertEqual(store.max_bytes, 4)

    def test_explicit_prune_evicts_oldest_to_respect_budget(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=8)
        a = store.put("a" * 64, self.source("a", b"aaaa"))
        os.utime(a, ns=(1, 1))
        store.put("b" * 64, self.source("b", b"bbbb"))
        store.put("c" * 64, self.source("c", b"cccc"))
        store.prune()
        self.assertIsNone(store.get("a" * 64))
        self.assertIsNotNone(store.get("b" * 64))
        self.assertIsNotNone(store.get("c" * 64))
        self.assertEqual(sum(p.stat().st_size for p in Path(store.objects).glob("*/*.bin")), 8)

    def test_hit_refreshes_eviction_order(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=8)
        a = store.put("a" * 64, self.source("a", b"aaaa"))
        b = store.put("b" * 64, self.source("b", b"bbbb"))
        os.utime(a, ns=(1, 1))
        os.utime(b, ns=(2, 2))
        self.assertIsNotNone(store.get("a" * 64))
        store.put("c" * 64, self.source("c", b"cccc"))
        store.prune()
        self.assertIsNotNone(store.get("a" * 64))
        self.assertIsNone(store.get("b" * 64))

    def test_oversized_object_does_not_flush_warm_cache(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=4)
        store.put("a" * 64, self.source("a", b"aaaa"))
        source = self.source("large", b"12345")
        self.assertEqual(store.put("b" * 64, source), source)
        self.assertIsNotNone(store.get("a" * 64))
        self.assertIsNone(store.get("b" * 64))

    def test_reader_pins_object_until_eviction_can_run(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=4)
        store.put("a" * 64, self.source("a", b"aaaa"))
        evict = ContentStore(store.root, max_bytes=0)
        started = threading.Event()

        def prune():
            started.set()
            evict.prune()

        with ThreadPoolExecutor(max_workers=1) as pool:
            with store.open("a" * 64) as cached:
                future = pool.submit(prune)
                self.assertTrue(started.wait(2))
                with self.assertRaises(TimeoutError):
                    future.result(timeout=0.05)
                self.assertEqual(cached.read(), b"aaaa")
            future.result(timeout=2)
        self.assertIsNone(store.get("a" * 64))


    def test_default_store_describes_both_decode_and_prefill(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(pack_cache.default_cache_dir(),
                             str(pack_cache._data_root.XDNA_CACHE / "packed-weights"))

    def test_prune_ignores_files_outside_object_contract(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=4)
        obj = store.put("a" * 64, self.source("a", b"aaaa"))
        note = Path(obj).parent / "notes.bin"
        note.write_bytes(b"evidence")
        outside = self.root / "outside"
        outside.mkdir()
        external = outside / ("b" * 64 + ".bin")
        external.write_bytes(b"outside")
        (Path(store.objects) / "bb").symlink_to(outside, target_is_directory=True)
        store.max_bytes = 0
        store.prune()
        self.assertEqual(note.read_bytes(), b"evidence")
        self.assertEqual(external.read_bytes(), b"outside")

    def test_eviction_leaves_materialized_hardlink_intact(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=4)
        store.put("a" * 64, self.source("a", b"aaaa"))
        destination = str(self.root / "artifact.bin")
        self.assertTrue(store.materialize("a" * 64, destination))
        cached = store.get("a" * 64)
        assert cached is not None
        self.assertEqual(os.stat(cached).st_ino, os.stat(destination).st_ino)
        store.put("b" * 64, self.source("b", b"bbbb"))
        store.prune()
        self.assertIsNone(store.get("a" * 64))
        self.assertEqual(Path(destination).read_bytes(), b"aaaa")

    def test_open_miss_does_not_create_an_object(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=4)
        with store.open("a" * 64) as cached:
            self.assertIsNone(cached)
        self.assertFalse(Path(store.objects).exists())

    def test_budget_environment_is_checked(self):
        with patch.dict(os.environ, {"PACK_CACHE_MAX_BYTES": "4"}):
            self.assertEqual(ContentStore(str(self.root / "cache")).max_bytes, 4)
        with patch.dict(os.environ, {"PACK_CACHE_MAX_BYTES": "-1"}):
            with self.assertRaises(ValueError):
                ContentStore(str(self.root / "cache"))

    def test_directory_override_precedence(self):
        with patch.dict(os.environ, {"PACK_CACHE_DIR": "shared", "PREFILL_PACK_CACHE_DIR": "other"}):
            self.assertEqual(pack_cache.default_cache_dir(), "shared")
        with patch.dict(os.environ, {"PREFILL_PACK_CACHE_DIR": "other"}, clear=True):
            self.assertEqual(pack_cache.default_cache_dir(), "other")

    def test_failed_materialization_preserves_destination_and_cache(self):
        store = ContentStore(str(self.root / "cache"))
        key = "a" * 64
        cached = store.put(key, self.source("input", b"new bytes"))
        destination = self.root / "artifact.bin"
        destination.write_bytes(b"previous bytes")
        previous_inode = destination.stat().st_ino

        def broken_copy(source, target):
            Path(target).write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "fixture: destination full")

        with patch.object(pack_cache.os, "link", side_effect=OSError(errno.EXDEV, "fixture")), \
                patch.object(pack_cache.shutil, "copy2", side_effect=broken_copy):
            with self.assertRaises(OSError):
                store.materialize(key, str(destination))
        self.assertEqual(destination.read_bytes(), b"previous bytes")
        self.assertEqual(destination.stat().st_ino, previous_inode)
        self.assertEqual(Path(cached).read_bytes(), b"new bytes")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["artifact.bin", "cache", "input"])

    def test_failed_replace_preserves_destination(self):
        store = ContentStore(str(self.root / "cache"))
        key = "a" * 64
        store.put(key, self.source("input", b"new bytes"))
        destination = self.root / "artifact.bin"
        destination.write_bytes(b"previous bytes")
        with patch.object(pack_cache.os, "replace", side_effect=OSError(errno.EIO, "fixture")):
            with self.assertRaises(OSError):
                store.materialize(key, str(destination))
        self.assertEqual(destination.read_bytes(), b"previous bytes")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["artifact.bin", "cache", "input"])

    def test_copy_fallback_publishes_complete_bytes(self):
        store = ContentStore(str(self.root / "cache"))
        key = "a" * 64
        cached = store.put(key, self.source("input", b"new bytes"))
        destination = self.root / "artifact.bin"
        destination.write_bytes(b"previous bytes")
        with patch.object(pack_cache.os, "link", side_effect=OSError(errno.EXDEV, "fixture")):
            self.assertTrue(store.materialize(key, str(destination)))
        self.assertEqual(destination.read_bytes(), b"new bytes")
        self.assertNotEqual(destination.stat().st_ino, Path(cached).stat().st_ino)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["artifact.bin", "cache", "input"])

    def test_invalid_keys_are_rejected_before_path_use(self):
        store = ContentStore(str(self.root / "cache"))
        source = self.source("input", b"new bytes")
        for key in ("", "../escape", "/absolute", "A" * 64, "a" * 63, None):
            with self.subTest(key=key), self.assertRaises(ValueError):
                store.put(key, source)

    def test_every_repack_content_input_changes_key(self):
        entry = {"buf": "W", "N": 16, "K": 32, "tile_k": 8, "tile_n": 4,
                 "group_size": 16, "weight_dtype": "int4", "cols": 4,
                 "scale_dtype": "f32", "mmul": [4, 4, 8]}
        digests = ["source", "packer", "generator"]
        key = pack_cache.content_key(entry, *digests)
        changes = {"N": 32, "K": 64, "tile_k": 16, "tile_n": 8,
                   "group_size": 32, "weight_dtype": "int8", "cols": 8,
                   "scale_dtype": "bf16", "mmul": [4, 8, 8]}
        for field, value in changes.items():
            changed = copy.deepcopy(entry)
            changed[field] = value
            with self.subTest(field=field):
                self.assertNotEqual(key, pack_cache.content_key(changed, *digests))
        for index in range(len(digests)):
            changed = list(digests)
            changed[index] += " changed"
            self.assertNotEqual(key, pack_cache.content_key(entry, *changed))
        entry.update(buf="other", src_dir="elsewhere", src_file="alias.npy")
        self.assertEqual(key, pack_cache.content_key(entry, *digests))

    def test_two_process_readers_hold_stable_lock_against_pruning(self):
        store = ContentStore(str(self.root / "cache"), max_bytes=4)
        store.put("a" * 64, self.source("input", b"aaaa"))
        lock = Path(store.root) / ".lock"
        identity = (lock.stat().st_dev, lock.stat().st_ino)
        module_path = str(Path(pack_cache.__file__).parent)
        prefix = "import sys; sys.path.insert(0, sys.argv[1]); from pack_cache import ContentStore\n"
        reader = prefix + ("with ContentStore(sys.argv[2], max_bytes=4).open('a'*64) as cached:\n"
                           " print('pinned', flush=True)\n"
                           " sys.stdin.readline()\n"
                           " print(cached.read().decode(), flush=True)\n")
        pruner = prefix + ("print('waiting', flush=True)\n"
                           "ContentStore(sys.argv[2], max_bytes=0).prune()\n"
                           "print('pruned', flush=True)\n")
        children = []

        def start(code):
            child = subprocess.Popen([sys.executable, "-u", "-c", code, module_path, store.root],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            children.append(child)
            self.addCleanup(child.stdin.close)
            self.addCleanup(child.stdout.close)
            self.addCleanup(child.stderr.close)
            return child

        def read_line(child):
            ready, _, _ = select.select([child.stdout], [], [], 2)
            self.assertTrue(ready, "child did not reach the expected synchronization point")
            return child.stdout.readline().strip()

        try:
            first, second = start(reader), start(reader)
            self.assertEqual(read_line(first), "pinned")
            self.assertEqual(read_line(second), "pinned")
            pruning = start(pruner)
            self.assertEqual(read_line(pruning), "waiting")
            with self.assertRaises(subprocess.TimeoutExpired):
                pruning.wait(timeout=0.1)
            first.stdin.write("release\n")
            first.stdin.flush()
            self.assertEqual(read_line(first), "aaaa")
            self.assertEqual(first.wait(timeout=2), 0)
            with self.assertRaises(subprocess.TimeoutExpired):
                pruning.wait(timeout=0.1)
            second.stdin.write("release\n")
            second.stdin.flush()
            self.assertEqual(read_line(second), "aaaa")
            self.assertEqual(second.wait(timeout=2), 0)
            self.assertEqual(pruning.wait(timeout=2), 0)
            self.assertEqual(read_line(pruning), "pruned")
            self.assertIsNone(store.get("a" * 64))
            self.assertEqual((lock.stat().st_dev, lock.stat().st_ino), identity)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
