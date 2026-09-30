#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild from a manifest with ONLY its recorded reads visible (read-only) and its writable
roots fresh and empty, then compare the output root with the recorded run's.

Replaces bwrap (one --ro-bind pair of argv per recorded path; bwrap 0.13.0 aborts at 9000
args, well under a real recipe's ~9.6k ops) with `unshare --user --map-root-user --mount` plus
direct mount(2) calls in ONE process -- no per-path argv or fork, so the path count is
unbounded. Binding a whole directory in one mount is deliberately never done for a recorded
path: each recorded file/dir is bound individually, so unread siblings stay invisible.

usage: replay.py <manifest.json> <recorded-out-dir> <scratch>
exit 0 = same identity.py identity (device bytes; drops meta.json provenance); anything else =
the manifest is not sufficient. A byte diff excluding meta.json is printed as extra info.
"""
import ctypes, ctypes.util, glob, json, os, pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import identity  # noqa: E402
MS_RDONLY, MS_BIND, MS_REMOUNT = 1, 4096, 32
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# Kept minimal and explicit (task furniture, not build inputs): a shim-recorded build never logs
# reads of these special files, so they are not in any manifest and must be supplied here.
DEV_NODES = ("null", "zero", "urandom", "random")


def _mount(src, tgt, fstype, flags):
    r = _libc.mount(str(src).encode(), str(tgt).encode(),
                    fstype.encode() if fstype else None, ctypes.c_ulong(flags), None)
    if r != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"mount {fstype or 'bind'} {src} -> {tgt}: {os.strerror(e)}")


def _bind_file(root, p):
    """Bind one recorded FILE read-only. Never a directory: that would expose every sibling,
    read or not (the Peano-dir case -- see module docstring)."""
    tgt = root / p.lstrip("/")
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.touch()
    _mount(p, tgt, None, MS_BIND)
    _mount(p, tgt, None, MS_BIND | MS_REMOUNT | MS_RDONLY)


def _bind_whole(root, p, writable=False):
    """Bind a whole real path (furniture only -- not a recorded build input). `writable` is for
    a real path the recorded run itself mutates in place (toolchain_up.sh's idempotent backfill
    symlinks into the shared instance) -- read-only would make replay diverge from what the
    real run actually did."""
    tgt = root / p.lstrip("/")
    (tgt.mkdir(parents=True, exist_ok=True) if os.path.isdir(p)
     else (tgt.parent.mkdir(parents=True, exist_ok=True), tgt.touch()))
    _mount(p, tgt, None, MS_BIND)
    if not writable:
        _mount(p, tgt, None, MS_BIND | MS_REMOUNT | MS_RDONLY)


def _recover_pyc_sources(reads):
    """CPython validates a cached .pyc against its SOURCE's mtime (a stat, unhooked on success)
    without necessarily opening the source at all, so a fresh-cache import records only the
    .pyc read -- the .py itself is missing from "reads" even though CPython's own loader
    requires it to be present. Recover it from any recorded __pycache__ entry."""
    extra = set()
    for p in reads:
        d, name = os.path.split(p)
        if os.path.basename(d) != "__pycache__" or not name.endswith(".pyc"):
            continue
        mod = name.split(".")[0]
        src = os.path.join(os.path.dirname(d), mod + ".py")
        if os.path.exists(src):
            extra.add(src)
    return extra


def build_plan(manifest_path, scratch):
    m = json.load(open(manifest_path))
    files = set(m["reads"]) | _recover_pyc_sources(m["reads"])
    dirs = set(m["dirs"])
    # Furniture (replay_base.txt), never a build input: bound whole, same as before.
    base = {l.strip() for l in (HERE / "replay_base.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")}
    base -= files | dirs
    files = sorted((p for p in files if os.path.exists(p)), key=len)
    dirs = sorted((p for p in dirs if os.path.exists(p)), key=len)
    furniture = [[p, False] for p in sorted(base, key=len) if os.path.exists(p)]
    # build_llm_decode.sh/build_prefill.sh/gate_llm.sh etc probe `[ -x "$VENV_IRON/bin/python" ]`
    # (VENV_IRON defaults to "$REPO/.venv-iron") before running anything -- a stat(2)/access(2)
    # check rec_preload.c's open-based hooks never see, so no manifest records it either way.
    repo = m["env"].get("REPO")
    if repo:
        # build_llm_decode.sh's own fallback ("$REPO/.venv-iron" -> "$WS/xdna-engine/.venv-iron",
        # WS = dirname(REPO)) is the same kind of probe: bind whichever one exists, as furniture.
        for venv_python in (os.path.join(repo, ".venv-iron", "bin", "python"),
                            os.path.join(os.path.dirname(repo), "xdna-engine", ".venv-iron",
                                        "bin", "python")):
            if os.path.exists(venv_python):
                furniture.append([venv_python, False])
    # toolchain_up.sh's `[ -e "$MLIR_DISTRO_ABS/bin/mlir-tblgen" ]` probe is unhooked-on-success
    # like the others, AND the instance's own python/aie/dialects/*.py are themselves symlinks
    # INTO this tree (vendored generated MLIR Python sources) -- a furnitured instance dir alone
    # still resolves those to a target nothing bound. Furnish the whole mlir-distro/<wheel> tree.
    xdna_cache = m["env"].get("XDNA_CACHE")
    if xdna_cache:
        # Both the alias path (XDNA_CACHE may itself be a symlink, e.g. a worktree's shared
        # cache) and its realpath: an embedded symlink's stored target is an absolute string
        # fixed at creation time, so it may name either form independent of which one the
        # recipe's own env happens to use.
        seen = set()
        for p in glob.glob(os.path.join(xdna_cache, "mlir-distro", "*")):
            if not os.path.isdir(p):
                continue
            for variant in (p, os.path.realpath(p)):
                if variant not in seen:
                    seen.add(variant)
                    furniture.append([variant, False])
    # The whole built toolchain instance: content-addressed by toolchain.lock's LOCKHASH
    # (buildstore.py's resolve_mlir_aie_instance), so its tree is furniture, not per-file
    # inputs -- its every internal existence probe is otherwise the same unhooked-probe class
    # as mlir-tblgen above, and there are too many of them (aie-translate, vendored symlinks,
    # backfill markers) to track one by one. WRITABLE: the real run itself backfills idempotent
    # symlinks into this shared instance in place (toolchain_up.sh's _link_vendored_tools);
    # read-only would make replay diverge from what the real run actually did.
    instance = m["env"].get("MLIR_AIE_INSTANCE")
    if instance and os.path.isdir(instance):
        furniture.append([instance, True])
    symlinks = {"/lib64": "usr/lib", "/lib": "usr/lib", "/bin": "usr/bin", "/sbin": "usr/bin"}
    symlinks.update(m["links"])
    # ld.so resolves a DT_NEEDED soname through ld.so.cache to a *symlink* path (e.g.
    # libreadline.so.8 -> libreadline.so.8.3), a lookup rec_preload.c's hooks never see; derive
    # the alias from ld.so.cache for any target this manifest did record.
    reads = set(m["reads"])
    for line in subprocess.run(["ldconfig", "-p"], capture_output=True,
                               text=True).stdout.splitlines()[1:]:
        if "=>" not in line:
            continue
        alias, path = line.split("=>")
        alias, path = alias.split("(")[0].strip(), path.strip()
        real = os.path.realpath(path)
        if path != real and real in reads:
            symlinks[path] = real
    # A PATH-searched name (bash's "awk" -> gawk) or a venv shim (".venv-iron/bin/python" ->
    # /usr/bin/python3.14) is chased by the KERNEL inside execve/openat -- rec_preload.c's hooks
    # see only the resolved target, never the symlink hop, so the given name is missing from
    # "reads" even though its target was recorded. Recover it: any symlink sibling, in a
    # directory a recorded read already lives in, whose target resolves to a recorded read.
    for d in {os.path.dirname(p) for p in reads}:
        try:
            entries = os.scandir(d)
        except OSError:
            continue
        with entries:
            for e in entries:
                full = e.path
                if full not in reads and full not in symlinks and os.path.islink(full) \
                        and os.path.realpath(full) in reads:
                    symlinks[full] = os.readlink(full)
    return {"files": files, "dirs": dirs, "furniture": furniture, "rw": list(m["writable"]),
            "cwd": m["cwd"], "env": m["env"], "symlinks": symlinks, "argv": m["argv"],
            "scratch": str(scratch)}


def apply_and_exec(plan):
    """Runs as the mapped-root user inside its own user+mount namespace (see main())."""
    scratch = pathlib.Path(plan["scratch"])
    root = scratch / "root"
    root.mkdir(parents=True, exist_ok=True)
    _mount("tmpfs", root, "tmpfs", 0)

    # A recorded "dirs" entry (its listing was read) gets a plain tmpfs directory, never a bind
    # of the real one -- same reason as _bind_file. Left writable, not locked ro: a non-recursive
    # self-bind would shadow the individual file mounts already placed inside it. The
    # anti-vacuous invariant only needs the listing to hide unread siblings, not read-only.
    for p in plan["dirs"]:
        (root / p.lstrip("/")).mkdir(parents=True, exist_ok=True)
    for p in plan["files"]:
        _bind_file(root, p)
    for p in plan["rw"]:
        real = scratch / "rw" / p.lstrip("/")
        real.mkdir(parents=True, exist_ok=True)
        (root / p.lstrip("/")).mkdir(parents=True, exist_ok=True)
    (root / "proc").mkdir(parents=True, exist_ok=True)
    (root / "dev").mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    (root / plan["cwd"].lstrip("/")).mkdir(parents=True, exist_ok=True)
    for given, target in plan["symlinks"].items():
        link = root / given.lstrip("/")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            continue
        os.symlink(target, link)

    for p, writable in plan["furniture"]:
        _bind_whole(root, p, writable=writable)
    for p in plan["rw"]:
        real = scratch / "rw" / p.lstrip("/")
        _mount(real, root / p.lstrip("/"), None, MS_BIND)

    _mount("proc", root / "proc", "proc", 0)
    for name in DEV_NODES:
        src = pathlib.Path("/dev") / name
        if src.exists():
            (root / "dev" / name).touch()
            _mount(src, root / "dev" / name, None, MS_BIND)
    _mount("tmpfs", root / "tmp", "tmpfs", 0)

    os.chroot(root)
    os.chdir(plan["cwd"])
    os.execvpe(plan["argv"][0], plan["argv"], plan["env"])


def main(argv):
    manifest, ref, scratch = argv[1], pathlib.Path(argv[2]), pathlib.Path(argv[3])
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    plan = build_plan(manifest, scratch)
    planfile = scratch / "plan.json"
    planfile.write_text(json.dumps(plan))
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--pid", "--ipc",
                        "--uts", "--fork", "--",
                        sys.executable, str(HERE / "replay.py"), "_apply", str(planfile)])
    if r.returncode != 0:
        return r.returncode
    out_root = scratch / "rw" / str(ref).lstrip("/")
    if not out_root.is_dir():
        print(f"replay: writable output root missing: {out_root}", file=sys.stderr)
        return 1
    # Compare by IDENTITY, not bytes: meta.json's generator git provenance (commit, dirty flag)
    # is not reproducible inside the sandbox, and identity.py already excludes exactly those
    # keys as not describing device bytes. A byte diff of everything else is kept as info.
    ref_id, out_id = identity.of(ref), identity.of(out_root)
    if ref_id == out_id:
        return 0
    print(f"replay: identity differs ({ref_id} != {out_id})", file=sys.stderr)
    diff = subprocess.run(["diff", "-r", "-x", "meta.json", str(ref), str(out_root)],
                          capture_output=True, text=True)
    if diff.stdout:
        print(diff.stdout, file=sys.stderr, end="")
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_apply":
        plan = json.loads(pathlib.Path(sys.argv[2]).read_text())
        try:
            apply_and_exec(plan)
        except OSError as e:
            print(f"replay: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(main(sys.argv))
