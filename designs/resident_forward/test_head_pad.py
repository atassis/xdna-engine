import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path


HERE = Path(__file__).parent


def generate(case):
    sys.path.insert(0, str(HERE))
    sys.modules["pyxrt"] = None
    import rlayer_design as L

    if case in ("recipe", "split"):
        words = shlex.split((HERE / "recipes/rf48C.sh").read_text().replace("\\\n", " "), comments=True)
        flags = words[words.index("rf48C") + 1].split()
        flags[flags.index("fwd=0-47+h")] = "fwd=0-1"
        flags.insert(flags.index("1"), "fnt=3:64")
        flags.insert(flags.index("1"), "emit=g1s128" if case == "split" else "emit=p1,p2,p3,f1,f3")
        flags.append("3")
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"full_attention_layers": []}))
            import rforward
            rforward.STORE = str(manifest)
            return L.build_text(flags)
    flags = {"odd": ["m", "1", "3"], "even": ["m", "2"], "head": ["1"]}[case]
    return L.build_text(flags)


def sequences(text):
    out = {}
    name = None
    for line in text.splitlines():
        match = re.match(r"    aie.runtime_sequence @(\w+)\(", line)
        if match:
            name = match[1]
            out[name] = []
        elif name is not None and line == "    }":
            name = None
        elif name is not None:
            out[name].append(line)
    return out


def tasks(lines):
    out = {}
    current = None
    for i, line in enumerate(lines):
        match = re.match(r"      %(\w+) = aiex.dma_configure_task\(%(\w+), (MM2S|S2MM), (\d+)", line)
        if match:
            current = match[1]
            out[current] = {"tile": match[2], "direction": match[3], "channel": int(match[4]),
                            "configure": i, "bds": [], "token": False}
        elif current is not None:
            bd = re.search(r"bd_id = (\d+) : i32", line)
            if bd:
                out[current]["bds"].append(int(bd[1]))
            if line.startswith("      }"):
                out[current]["token"] = "issue_token = true" in line
                current = None
        event = re.match(r"      aiex.dma_(start|await|free)_task\(%(\w+)\)", line)
        if event:
            out[event[2]].setdefault(event[1], []).append(i)
    return out


class HeadPadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = {}
        env = {key: value for key, value in os.environ.items() if not key.startswith(("RF_", "RSEG_"))}
        env.update(RF_FAST="1", RF_ATTN_H="1", PYTHONDONTWRITEBYTECODE="1")
        for case in ("recipe", "odd", "even", "head", "split"):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--generate", case],
                                    cwd=HERE, env=env, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise AssertionError(result.stderr)
            cls.generated[case] = result.stdout

    def check_completion(self, case, name, suffixes):
        text = self.generated[case]
        lines = sequences(text)[name]
        configured = tasks(lines)
        pending = {}
        for line in lines:
            event = re.match(r"      aiex.dma_(start|await)_task\(%(\w+)\)", line)
            if not event:
                continue
            op, task_name = event.groups()
            task = configured[task_name]
            key = (task["tile"], task["direction"], task["channel"])
            queue = pending.setdefault(key, deque())
            if op == "start" and task["token"]:
                queue.append(task_name)
            elif op == "await":
                self.assertTrue(queue, f"{task_name} awaits an unissued channel token")
                completed = queue.popleft()
                if task_name.startswith("hp"):
                    self.assertEqual(completed, task_name, "pad await consumes an earlier task's token")
        expected = {f"hp{c}{suffix}" for suffix in suffixes for c in range(8)}
        self.assertEqual({task for task in configured if re.fullmatch(r"hp\d+(?:_L\d+)?", task)}, expected)
        for pad in sorted(expected):
            with self.subTest(case=case, sequence=name, pad=pad):
                task = configured[pad]
                self.assertEqual((task["direction"], task["channel"], task["bds"]), ("MM2S", 4, [16]))
                self.check_lifetime(task)
                reusers = [other for other in configured.values()
                           if other["tile"] == task["tile"] and 16 in other["bds"]
                           and other["configure"] > task["configure"]]
                self.assertTrue(reusers, f"fixture does not exercise {pad}'s BD reuse")
                self.assertLess(task["free"][0], min(other["configure"] for other in reusers))
                col = task["tile"].split("_")[1]
                self.assertRegex(text, rf"aie.packet_flow\(\d+\) \{{\s+aie.packet_source<%mem_{col}, TileControl : 0>\s+aie.packet_dest<%shim_{col}, South : 0>\s+\}} \{{ctrl_pkt_flow = true, keep_pkt_header = true\}}")

    def check_lifetime(self, task):
        self.assertTrue(task["token"], "pad cannot be awaited without a completion token")
        for op in ("start", "await", "free"):
            self.assertEqual(len(task.get(op, [])), 1, f"pad needs exactly one {op}")
        self.assertLess(task["start"][0], task["await"][0])
        self.assertLess(task["await"][0], task["free"][0])

    def test_lifetime_check_rejects_missing_token_or_await_and_early_free(self):
        task = tasks(sequences(self.generated["recipe"])["p1"])["hp0"]
        for changes in ({"token": False}, {"await": []}, {"free": [task["await"][0] - 1]}):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                self.check_lifetime(task | changes)

    def test_completion_tokens_are_matched_and_drained(self):
        for case in ("recipe", "odd", "even", "split"):
            for name, lines in sequences(self.generated[case]).items():
                with self.subTest(case=case, sequence=name):
                    configured = tasks(lines)
                    pending = {}
                    for line in lines:
                        event = re.match(r"      aiex.dma_(start|await)_task\(%(\w+)\)", line)
                        if not event:
                            continue
                        op, task_name = event.groups()
                        task = configured[task_name]
                        channel = (task["tile"], task["direction"], task["channel"])
                        queue = pending.setdefault(channel, deque())
                        if op == "start" and task["token"]:
                            queue.append(task_name)
                        elif op == "await":
                            self.assertTrue(queue, task_name)
                            self.assertEqual(queue.popleft(), task_name)
                    self.assertFalse(any(pending.values()), pending)

    def test_python_syntax(self):
        for name in ("rlayer_design.py", "test_head_pad.py"):
            compile((HERE / name).read_text(), name, "exec")

    def test_decode_pad_completes_before_descriptor_reuse(self):
        self.check_completion("recipe", "p1", ("",))

    def test_odd_prefill_pad_completes_before_descriptor_reuse(self):
        self.check_completion("recipe", "p3", ("",))

    def test_splice_preserves_pad_completion(self):
        for name in ("f1", "f3"):
            with self.subTest(sequence=name):
                self.check_completion("recipe", name, ("_L0", "_L1"))

    def test_unsegmented_odd_sequences_route_completion(self):
        for name in ("p1", "p3"):
            with self.subTest(sequence=name):
                self.check_completion("odd", name, ("",))

    def test_rope_receive_completes_at_existing_barrier_before_reuse(self):
        for case, names in (("recipe", ("p1", "p2", "p3", "f1", "f3")),
                            ("odd", ("p1", "p3")), ("even", ("p2",))):
            for name in names:
                lines = sequences(self.generated[case])[name]
                configured = tasks(lines)
                receives = {task_name: task for task_name, task in configured.items()
                            if re.fullmatch(r"rin\d+(?:_L\d+)?", task_name)}
                self.assertEqual(len(receives), 16 if name.startswith("f") else 8)
                for task_name, task in receives.items():
                    with self.subTest(case=case, sequence=name, task=task_name):
                        self.check_lifetime(task)
                        self.assertEqual((task["direction"], task["channel"], task["bds"]),
                                         ("S2MM", 1, [33]))
                        suffix = task_name.partition("_L")[2]
                        barrier = configured["kws7" + ("_L" + suffix if suffix else "")]
                        self.assertLess(barrier["await"][0], task["await"][0])
                        reusers = [other for other in configured.values()
                                   if other["tile"] == task["tile"] and 33 in other["bds"]
                                   and other["configure"] > task["configure"]]
                        self.assertTrue(reusers)
                        self.assertLess(task["await"][0], min(other["configure"] for other in reusers))

    def test_even_sequences_have_no_pad_and_only_receive_completion_awaits(self):
        for case in ("recipe", "even"):
            configured = tasks(sequences(self.generated[case])["p2"])
            self.assertFalse(any(name.startswith("hp") for name in configured))
            awaited = {name for name, task in configured.items()
                       if task["tile"].startswith("mem_") and task.get("await")}
            self.assertEqual(awaited, {f"rin{c}" for c in range(8)} | {f"tk_{c}" for c in range(8)}
                             | {f"oob{c}" for c in range(8)})

    def test_attention_readout_completion_is_explicit_at_existing_barrier(self):
        for name in ("p1", "p2", "p3", "f1", "f3"):
            configured = tasks(sequences(self.generated["recipe"])[name])
            observers = {task_name: task for task_name, task in configured.items()
                         if re.fullmatch(r"tk_\d+(?:_L\d+)?", task_name)}
            self.assertEqual(len(observers), 16 if name.startswith("f") else 8)
            for task_name, task in observers.items():
                with self.subTest(sequence=name, task=task_name):
                    self.check_lifetime(task)
                    self.assertEqual(task["bds"], [41])
                    suffix = task_name.partition("_L")[2]
                    sink = "so_7" + ("_L" + suffix if suffix else "")
                    self.assertLess(configured[sink]["await"][0], task["await"][0])

    def test_output_projection_releases_completion_credit_before_next_phase(self):
        for name in ("p1", "p2", "p3", "f1", "f3"):
            configured = tasks(sequences(self.generated["recipe"])[name])
            consumers = {task_name: task for task_name, task in configured.items()
                         if re.fullmatch(r"oob\d+(?:_L\d+)?", task_name)}
            self.assertEqual(len(consumers), 16 if name.startswith("f") else 8)
            for task_name, task in consumers.items():
                with self.subTest(sequence=name, task=task_name):
                    self.check_lifetime(task)
                    suffix = task_name.partition("_L")[2]
                    sink = "oso7" + ("_L" + suffix if suffix else "")
                    self.assertLess(configured[sink]["await"][0], task["await"][0])

    def test_split_export_and_merge_complete_before_descriptor_reuse(self):
        configured = tasks(sequences(self.generated["split"])["g1s128"])
        for prefix, count, barrier in (("ctout_", 8, "so_7"), ("mgx", 2, "mgw1")):
            for c in range(count):
                with self.subTest(task=f"{prefix}{c}"):
                    task = configured[f"{prefix}{c}"]
                    self.check_lifetime(task)
                    self.assertEqual(task["bds"], [41])
                    self.assertLess(configured[barrier]["await"][0], task["await"][0])
                    reusers = [other for other in configured.values()
                               if other["tile"] == task["tile"] and 41 in other["bds"]
                               and other["configure"] > task["configure"]]
                    self.assertTrue(reusers)
                    self.assertLess(task["await"][0], min(other["configure"] for other in reusers))

    def test_split_merge_receive_completes_before_credit_reinitialization(self):
        configured = tasks(sequences(self.generated["split"])["g1s128"])
        for c in range(2):
            with self.subTest(column=c):
                task = configured[f"mgl{c}"]
                self.check_lifetime(task)
                self.assertEqual((task["direction"], task["channel"], task["bds"]),
                                 ("S2MM", 4, [2]))
                self.assertLess(configured["mgw1"]["await"][0], task["await"][0])

    def test_split_replay_completion_is_explicit_at_existing_barrier(self):
        configured = tasks(sequences(self.generated["split"])["g1s128"])
        for c in range(8):
            with self.subTest(column=c):
                task = configured[f"rtk_{c}"]
                self.check_lifetime(task)
                self.assertEqual((task["direction"], task["channel"], task["bds"]),
                                 ("MM2S", 5, [41]))
                self.assertLess(configured["rso_7"]["await"][0], task["await"][0])

    def test_head_only_pad_does_not_issue_an_unrouted_token(self):
        configured = tasks(sequences(self.generated["head"])["p1"])
        pads = [task for name, task in configured.items() if name.startswith("hp")]
        self.assertEqual(len(pads), 8)
        self.assertTrue(all(not task["token"] and not task.get("await") for task in pads))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--generate":
        sys.stdout.write(generate(sys.argv[2]))
    else:
        unittest.main()
