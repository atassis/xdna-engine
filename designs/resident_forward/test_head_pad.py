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

    if case == "recipe":
        words = shlex.split((HERE / "recipes/rf48C.sh").read_text().replace("\\\n", " "), comments=True)
        flags = words[words.index("rf48C") + 1].split()
        flags[flags.index("fwd=0-47+h")] = "fwd=0-1"
        flags.insert(flags.index("1"), "fnt=3:64")
        flags.insert(flags.index("1"), "emit=p1,p2,p3,f1,f3")
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
        for case in ("recipe", "odd", "even", "head"):
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

    def test_even_sequences_have_no_pad_or_memtile_await(self):
        for case in ("recipe", "even"):
            configured = tasks(sequences(self.generated[case])["p2"])
            self.assertFalse(any(name.startswith("hp") for name in configured))
            self.assertFalse(any(task.get("await") for task in configured.values() if task["tile"].startswith("mem_")))

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
