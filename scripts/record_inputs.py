#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a Python build step, record every input it read, and later check whether any changed.

  record_inputs.py record --manifest M.json -- script.py [args...]
  record_inputs.py check  --manifest M.json   # exit 0: every recorded input unchanged
  record_inputs.py key    --manifest M.json   # the content key of the recorded inputs

Recorded: env vars read (os.environ[...] / .get / in), files opened for reading (audit hook,
imports included), loaded module files outside the stdlib, argv, the interpreter version.
os.environ.copy() does not record: a child process's environment is declared by its runner.
RECORD_INPUTS_SABOTAGE=env|files disables one class, for the tests that prove each class matters.
"""
import hashlib
import json
import os
import runpy
import sys
import sysconfig


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(manifest, script, args):
    sabotage = os.environ.get("RECORD_INPUTS_SABOTAGE", "")
    env_reads, opened_r, opened_w = {}, set(), set()
    stdlib = os.path.realpath(sysconfig.get_paths()["stdlib"])
    environ_cls = type(os.environ)
    orig_getitem, orig_copy = environ_cls.__getitem__, environ_cls.copy
    copying = [False]

    def getitem(self, key):
        try:
            value = orig_getitem(self, key)
        except KeyError:
            if not copying[0]:
                env_reads[key] = None
            raise
        if not copying[0]:
            env_reads[key] = value
        return value

    def copy(self):
        copying[0] = True
        try:
            return orig_copy(self)
        finally:
            copying[0] = False

    def hook(event, a):
        if event != "open" or not isinstance(a[0], (str, bytes)):
            return
        path = os.path.realpath(os.fsdecode(a[0]))
        mode, flags = a[1], a[2]
        writing = (mode is not None and any(c in mode for c in "wax+")) or (
            mode is None and flags & (os.O_WRONLY | os.O_RDWR))
        (opened_w if writing else opened_r).add(path)

    if sabotage != "env":
        environ_cls.__getitem__, environ_cls.copy = getitem, copy
    sys.addaudithook(hook)
    sys.argv = [script, *args]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    code = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        environ_cls.__getitem__, environ_cls.copy = orig_getitem, orig_copy
    if code != 0:
        return code
    modules = {os.path.realpath(m.__file__) for m in list(sys.modules.values())
               if getattr(m, "__file__", None)}
    files = {}
    if sabotage != "files":
        for p in (opened_r | modules) - opened_w:
            in_stdlib = p.startswith(stdlib + os.sep) and "site-packages" not in p
            if in_stdlib or not os.path.isfile(p):
                continue
            files[p] = _sha(p)
    doc = {"argv": [os.path.abspath(script), *args], "python": sys.version,
           "env": env_reads, "files": files}
    with open(manifest, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return 0


def _changes(doc):
    out = []
    if doc["python"] != sys.version:
        out.append("python version")
    for k, v in doc["env"].items():
        if os.environ.get(k) != v:
            out.append(f"env {k}")
    for p, h in doc["files"].items():
        if not os.path.isfile(p) or _sha(p) != h:
            out.append(f"file {p}")
    return out


def main():
    # Hand-rolled instead of argparse: a REMAINDER positional after a plain
    # positional greedily swallows a later required option (argparse#*),
    # e.g. "record --manifest M -- script.py" loses --manifest to REMAINDER.
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("record", "check", "key"):
        print(__doc__, file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if "--manifest" not in rest:
        print("record_inputs.py: error: the following arguments are required: --manifest",
              file=sys.stderr)
        return 2
    i = rest.index("--manifest")
    manifest = rest[i + 1]
    tail = rest[:i] + rest[i + 2:]
    if cmd == "record":
        tail = tail[1:] if tail[:1] == ["--"] else tail
        return _record(manifest, tail[0], tail[1:])
    doc = json.load(open(manifest))
    if cmd == "key":
        print(hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest())
        return 0
    changed = _changes(doc)
    for c in changed:
        print(c)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
