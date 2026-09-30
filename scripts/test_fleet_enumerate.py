# SPDX-License-Identifier: Apache-2.0
"""fleet_enumerate: dedup kernel compiles (from kcc_log_launcher JSONL) and operator devices
(from aie.mlir text) across a fleet of designs."""
import json
import textwrap
from pathlib import Path

from fleet_enumerate import compile_seconds, device_keys, summarize

HERE = Path(__file__).resolve().parent


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_summarize_dedups_by_normalized_source_and_args(tmp_path):
    # Two designs, three compiles total, two identical after normalization: same
    # source_sha256, args equal once the -include-pch operand and any /work/ -I path are
    # normalized away (both vary per tmpdir even for the same kernel).
    a = tmp_path / "designA" / "kcc.jsonl"
    a.parent.mkdir()
    _write_jsonl(a, [
        {"source": "/w1/work/k.cc", "source_sha256": "hh", "compiler": "clang++",
         "args": ["-c", "-I/w1/work/inc", "-include-pch", "/w1/work/x.pch", "-DOFFSET=1"]},
    ])
    b = tmp_path / "designB" / "kcc.jsonl"
    b.parent.mkdir()
    _write_jsonl(b, [
        {"source": "/w2/work/k.cc", "source_sha256": "hh", "compiler": "clang++",
         "args": ["-c", "-I/w2/work/inc", "-include-pch", "/w2/work/x.pch", "-DOFFSET=1"]},
        {"source": "/w2/work/other.cc", "source_sha256": "zz", "compiler": "clang++",
         "args": ["-c", "-DOFFSET=2"]},
    ])

    result = summarize([a, b])

    assert result["compiles"] == 3
    assert result["unique"] == 2
    assert result["per_source"] == {str(a): {"compiles": 1, "unique": 1},
                                     str(b): {"compiles": 2, "unique": 2}}


def test_summarize_on_two_identical_logs_halves(tmp_path):
    # Gate named in the plan: run the dedup counter on two copies of the same design and
    # confirm unique == total / 2.
    log = tmp_path / "one" / "kcc.jsonl"
    log.parent.mkdir()
    _write_jsonl(log, [
        {"source": "/w/work/k1.cc", "source_sha256": "aa", "compiler": "clang++", "args": ["-c"]},
        {"source": "/w/work/k2.cc", "source_sha256": "bb", "compiler": "clang++", "args": ["-c"]},
    ])
    copy = tmp_path / "one_copy" / "kcc.jsonl"
    copy.parent.mkdir()
    copy.write_text(log.read_text())

    result = summarize([log, copy])

    assert result["compiles"] == 4
    assert result["unique"] == 2


def test_compile_seconds_sums_total_and_first_occurrence_of_each_unique_key(tmp_path):
    log = tmp_path / "one" / "kcc.jsonl"
    log.parent.mkdir()
    _write_jsonl(log, [
        {"source": "/w/work/k.cc", "source_sha256": "aa", "compiler": "c", "args": ["-c"],
         "start": 0.0, "end": 2.0},
        {"source": "/w2/work/k.cc", "source_sha256": "aa", "compiler": "c", "args": ["-c"],
         "start": 5.0, "end": 8.0},  # same key as above (dup), different duration
        {"source": "/w/work/k2.cc", "source_sha256": "bb", "compiler": "c", "args": ["-c"],
         "start": 0.0, "end": 1.5},
    ])

    result = compile_seconds([log])

    assert result["total_s"] == 2.0 + 3.0 + 1.5
    assert result["unique_s"] == 2.0 + 1.5   # only the first occurrence of the "aa" key counts


def test_device_keys_splits_top_level_device_blocks_and_strips_locs_and_names(tmp_path):
    mlir = tmp_path / "aie.mlir"
    mlir.write_text(textwrap.dedent('''\
        module {
          aie.device(npu2) @seq_a {
            %t = aie.tile(0, 0) loc("file.py":10:2)
            aie.core(%t) { }
          } loc("file.py":9:0)
          aie.device(npu2) @seq_b {
            %t = aie.tile(0, 0) loc("other.py":3:1)
            aie.core(%t) { }
          } loc("file.py":20:0)
        }
    '''))

    keys = device_keys(mlir)

    assert [name for name, _ in keys] == ["seq_a", "seq_b"]
    # Identical device bodies modulo loc(...) and the @name hash to the same key.
    assert keys[0][1] == keys[1][1]
    assert len(keys[0][1]) == 64  # sha256 hex digest
