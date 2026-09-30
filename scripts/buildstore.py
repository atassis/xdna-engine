#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI over scripts/buildstore/{record,store,identity}.py.

build <recipe>...|--all [--recipes TSV] [--out-root DIR] [--no-hit] [--report FILE]: take each
recipe's lock, check stored actions newest-first for a hit, else run it under the recorder and
store the result. TSV format matches run_s0.sh: tab-separated, `#` comments, NORECIPE skipped.
--report writes {recipe: {identity, tree, status, seconds, peak_rss_kb}}; with --report, one
recipe's failure is a FAILED line, not an abort (only --all uses this in practice). --verify is
the sufficiency gate: on a HIT, cold-rebuild into scratch, outside the store, and require the
same identity.of() (owner 2026-09-30; supersedes the plan's bwrap/unshare replay as the gate).

gate-record/gate-status: device gate results keyed by artifact identity + driver + firmware.
repin-report A.json B.json: compare two `build --report` outputs by identity.
replay MANIFEST REF_OUT SCRATCH: DIAGNOSTIC only, not a gate -- see cmd_replay.
"""
import argparse, datetime, glob, hashlib, json, os, pathlib, shutil, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "buildstore"))
import identity, record, store  # noqa: E402

WS = HERE.parent.parent


def parse_recipes(path):
    recipes = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _, cmd = line.partition("\t")
        recipes[name] = cmd
    return recipes


def _elapsed_seconds(s):
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def resolve_xdna_cache(repo, env):
    """XDNA_CACHE, when not already in the hermetic env: cache_env.sh's own default resolves it
    via a worktree's .git (git-common-dir, to inherit the main checkout's cache), which a replay
    sandbox never has. Resolve it here, once, outside any sandbox, and bake it into env -- same
    reasoning as iron_pin_verified."""
    if "XDNA_CACHE" in env:
        return
    cache_env = pathlib.Path(repo) / "scripts" / "cache_env.sh"
    if not cache_env.is_file():
        return
    r = subprocess.run(["bash", "-c", f'. "{cache_env}"; echo "$XDNA_CACHE"'],
                       cwd=repo, env=env, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        env["XDNA_CACHE"] = r.stdout.strip()


def resolve_mlir_aie_instance(repo, env):
    """toolchain_up.sh's cached-instance branch is gated entirely by [ -e ]/[ -L ] probes
    (aie-translate, vendored-tool symlinks, ...) that succeed against real files a shim's
    probe hooks never log (they only log ENOENT). A replay sandbox then takes the wrong,
    mutating branch. The instance is content-addressed by toolchain.lock's LOCKHASH (already
    deterministic, same reasoning as decision 3's cache roots), so it is resolved once here and
    replay.py furnishes the WHOLE tree instead of tracking it probe-by-probe."""
    if "MLIR_AIE_INSTANCE" in env:
        return
    script = pathlib.Path(repo) / "scripts" / "toolchain_up.sh"
    if not script.is_file():
        return
    r = subprocess.run(["bash", str(script)], cwd=repo, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"toolchain_up.sh failed for {repo}: {r.stderr.strip()}")
    lines = r.stdout.strip().splitlines()
    if lines:
        env["MLIR_AIE_INSTANCE"] = lines[-1]


def iron_pin_verified(repo, env):
    """Run amd_paths.sh's iron_require_pin ONCE, outside any replay sandbox (which never has
    .git -- see the function's own comment), and return the sha it verified. None for a repo
    with no amd_paths.sh (the test fixtures) or no IRON pin to check."""
    amd_paths = pathlib.Path(repo) / "scripts" / "amd_paths.sh"
    if not amd_paths.is_file():
        return None
    script = (f'. "{amd_paths}"; iron_require_pin || exit 1; '
              r"""sed -n 's/^IRON_FORK_COMMIT=\([0-9a-f]\{7,\}\).*/\1/p' toolchain.lock | head -1""")
    r = subprocess.run(["bash", "-c", script], cwd=repo, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"iron pin check failed for {repo}: {r.stderr.strip()}")
    lines = r.stdout.strip().splitlines()
    return lines[-1] if lines else None


def parse_time_v(path):
    """(wall seconds, peak RSS kB) from `/usr/bin/time -v` output; either may be None."""
    wall = peak = None
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            wall = _elapsed_seconds(line.rsplit(": ", 1)[1])
        elif line.startswith("Maximum resident set size"):
            peak = int(line.rsplit(": ", 1)[1])
    return wall, peak


def _recipe_env(repo, s, cache_key):
    """Hermetic env + its cache roots for one recipe run. `cache_key` names the caches dir --
    a real build and its --verify cold rebuild use different keys so verify never reads a
    cache the real build warmed."""
    env = record.hermetic_env(os.environ, record.allowlist())
    env["REPO"] = str(repo)
    resolve_xdna_cache(repo, env)
    resolve_mlir_aie_instance(repo, env)
    want = iron_pin_verified(repo, env)
    if want:
        env["IRON_PIN_VERIFIED"] = want
    npu_cache = pathlib.Path(s.root) / "caches" / cache_key / "npu_cache"
    ccache_dir = pathlib.Path(s.root) / "caches" / cache_key / "ccache"
    env["XDNA_BLOB_POOL"] = "0"
    env["NPU_CACHE_HOME"] = str(npu_cache)
    env["CCACHE_DIR"] = str(ccache_dir)
    cache_roots = [npu_cache, ccache_dir]
    if env.get("XDNA_CACHE"):
        cache_roots.append(pathlib.Path(env["XDNA_CACHE"]) / "kobj")
    return env, cache_roots


def _verify_one(name, cmd, repo, s, expect_identity):
    """The P1 sufficiency gate (owner 2026-09-30): cold-rebuild `cmd` into a fresh scratch dir,
    outside the store, and compare its identity to a HIT's. This replaces the plan's original
    bwrap/unshare replay as the GATE -- that mechanism (`replay` subcommand) stays a diagnostic;
    it currently stops at CPython venv bootstrap (sysconfig installed_base)."""
    scratch = pathlib.Path(s.root) / "verify" / name
    shutil.rmtree(scratch, ignore_errors=True)
    out, work = scratch / "out", scratch / "work"
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    env, cache_roots = _recipe_env(repo, s, f"{name}-verify")
    env["OUT"] = str(out)
    env["KEEP_WORK"] = str(work)
    for root in cache_roots[:2]:
        pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    argv = ["bash", "-c", f"cd {repo} && {cmd}"]
    record.run(argv, cwd=repo, env=env, out_roots=[out], work_roots=[work], cache_roots=cache_roots)
    got = identity.of(out)
    shutil.rmtree(scratch, ignore_errors=True)
    if got != expect_identity:
        raise RuntimeError(f"verify: {name} cold identity {got} != {expect_identity}")


def _build_one(name, cmd, repo, s, out_root, no_hit, verify):
    """Build (or hit) one recipe. Returns {status: HIT|BUILT, identity, tree, seconds,
    peak_rss_kb}. Raises on failure -- the caller (cmd_build) decides whether that aborts the
    whole run (explicit recipe list) or is recorded as one FAILED line (--all --report)."""
    t0 = time.time()
    with s.lock(name):
        argv = ["bash", "-c", f"cd {repo} && {cmd}"]
        env, cache_roots = _recipe_env(repo, s, name)

        staging = pathlib.Path(s.root) / "staging" / name
        out = staging / "out"
        work = staging / "work"
        env["OUT"] = str(out)
        env["KEEP_WORK"] = str(work)

        if not no_hit:
            hit = next((a for a in s.actions(name)
                       if record.check(a["manifest"], env=env, argv=argv) == []), None)
            if hit is not None:
                if verify:
                    _verify_one(name, cmd, repo, s, hit["identity"])
                s.materialize(hit["tree"], out_root / name)
                return {"status": "HIT", "identity": hit["identity"], "tree": hit["tree"],
                        "seconds": time.time() - t0, "peak_rss_kb": None}

        shutil.rmtree(staging, ignore_errors=True)
        out.mkdir(parents=True)
        work.mkdir(parents=True)
        for root in cache_roots[:2]:  # npu_cache, ccache_dir; XDNA_CACHE/kobj is real, not ours to create
            pathlib.Path(root).mkdir(parents=True, exist_ok=True)
        time_file = staging / "time.txt"
        manifest = record.run(argv, cwd=repo, env=env, out_roots=[out], work_roots=[work],
                              cache_roots=cache_roots, time_file=time_file)
        wall, peak_kb = parse_time_v(time_file)

        tree = s.put_tree(out)
        ident = identity.of(out)
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        s.put_action(name, key, {"manifest": manifest, "tree": tree, "identity": ident,
                                 "timings": {"wall": wall, "peak_rss_kb": peak_kb}})
        s.materialize(tree, out_root / name)
        shutil.rmtree(staging, ignore_errors=True)
        return {"status": "BUILT", "identity": ident, "tree": tree,
                "seconds": time.time() - t0, "peak_rss_kb": peak_kb}


def cmd_build(args):
    cas = pathlib.Path(os.environ.get("BUILDSTORE_CAS", "/mnt/data/xdna/cas"))
    repo = pathlib.Path(os.environ.get("BUILDSTORE_REPO", str(WS / "xdna-engine")))
    recipes_path = args.recipes or (HERE / "s0_recipes.tsv")
    recipes = parse_recipes(recipes_path)
    s = store.Store(cas)
    statcache = cas / "statcache.json"
    record.load_statcache(str(statcache))
    out_root = pathlib.Path(args.out_root)
    names = [n for n, c in recipes.items() if c != "NORECIPE"] if args.all else args.recipe

    report = {}
    for name in names:
        cmd = recipes.get(name)
        if cmd is None or cmd == "NORECIPE":
            print(f"SKIP {name} NORECIPE")
            continue
        try:
            r = _build_one(name, cmd, repo, s, out_root, args.no_hit, args.verify)
        except Exception as e:  # noqa: BLE001 -- --all must not abort on one recipe's failure
            if not args.report:
                raise
            print(f"FAILED {name} {e}")
            report[name] = {"status": "FAILED", "identity": None, "tree": None,
                            "seconds": None, "peak_rss_kb": None}
            continue
        print(f"{r['status']} {name} {r['identity']} {r['seconds']:.2f}"
              + (f" {r['peak_rss_kb']}" if r["status"] == "BUILT" else ""))
        report[name] = r

    record.save_statcache(str(statcache))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, sort_keys=True, indent=2))


def cmd_replay(args):
    """DIAGNOSTIC, not a gate (owner 2026-09-30): the unshare/mount manifest replay of Task 3,
    exposed here for convenience. `build --verify`'s cold-rebuild identity match is the actual
    sufficiency gate. Replay currently stops at CPython venv bootstrap (sysconfig
    installed_base) -- a failing exit here is a known gap, not a build regression."""
    return subprocess.run([sys.executable, str(HERE / "buildstore" / "replay.py"),
                           args.manifest, args.ref_out, args.scratch]).returncode


def cmd_repin_report(args):
    """One line per recipe comparing two `build --report` JSONs by IDENTITY: IDENTICAL (its
    gates, keyed by identity, already apply -- nothing to carry over by hand), CHANGED <n
    files>, or FAILED. Ends with a device-work summary. Always exits 0: a repin report is
    information, never a gate."""
    a = json.loads(pathlib.Path(args.report_a).read_text())
    b = json.loads(pathlib.Path(args.report_b).read_text())
    cas = pathlib.Path(os.environ.get("BUILDSTORE_CAS", "/mnt/data/xdna/cas"))
    s = store.Store(cas)
    device_work = []
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name), b.get(name)
        if not ra or not rb or ra["status"] == "FAILED" or rb["status"] == "FAILED":
            print(f"FAILED {name}")
            continue
        if ra["identity"] == rb["identity"]:
            print(f"IDENTICAL {name}")
            continue
        ta, tb = s.tree(ra["tree"]), s.tree(rb["tree"])
        diff = sorted(p for p in set(ta) | set(tb) if ta.get(p) != tb.get(p))
        print(f"CHANGED {name} {len(diff)} files differ: {', '.join(diff)}")
        device_work.append(name)
    print(f"device work: {', '.join(device_work)}")


def _driver_srcversion():
    path = os.environ.get("BUILDSTORE_DRIVER_FILE", "/sys/module/amdxdna/srcversion")
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None


def _firmware_version():
    override = os.environ.get("BUILDSTORE_FW_FILE")
    paths = [override] if override else sorted(
        glob.glob("/sys/bus/pci/drivers/amdxdna/0000:*/fw_version"))
    for p in paths:
        try:
            return pathlib.Path(p).read_text().strip()
        except OSError:
            continue
    return None


def cmd_gate_record(args):
    """Record a device gate result keyed by artifact IDENTITY, not path or pin -- never fails
    a gate: a broken artifact dir or an unreadable driver/fw file is a WARNING, not an error."""
    try:
        cas = pathlib.Path(os.environ.get("BUILDSTORE_CAS", "/mnt/data/xdna/cas"))
        ident = identity.of(args.artifact_dir)
        rec = {
            "result": "pass" if int(args.rc) == 0 else "fail",
            "json": pathlib.Path(args.json_file).read_text()
                if args.json_file and os.path.isfile(args.json_file) else None,
            "driver": _driver_srcversion(),
            "firmware": _firmware_version(),
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        path = cas / "gates" / ident / f"{args.gate}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(rec, sort_keys=True))
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 -- recording must never fail a gate
        print(f"gate-record: WARNING: {e}", file=sys.stderr)


def cmd_gate_status(args):
    """VALID when identity + driver + firmware all match the current system, STALE <what
    moved> otherwise, NONE when this identity has no recorded gate at all."""
    cas = pathlib.Path(os.environ.get("BUILDSTORE_CAS", "/mnt/data/xdna/cas"))
    ident = identity.of(args.artifact_dir)
    gdir = cas / "gates" / ident
    files = sorted(gdir.glob("*.json")) if gdir.is_dir() else []
    if not files:
        print("NONE")
        return
    driver_now, fw_now = _driver_srcversion(), _firmware_version()
    for f in files:
        rec = json.loads(f.read_text())
        stale = [n for n, now in (("driver", driver_now), ("firmware", fw_now))
                 if rec.get(n) != now]
        print(f"STALE {f.stem} {' '.join(stale)}" if stale else f"VALID {f.stem}")


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("recipe", nargs="*")
    b.add_argument("--recipes", type=pathlib.Path, default=None)
    b.add_argument("--out-root", required=True)
    b.add_argument("--no-hit", action="store_true")
    b.add_argument("--all", action="store_true")
    b.add_argument("--report", type=pathlib.Path, default=None)
    b.add_argument("--verify", action="store_true",
                   help="on a HIT, cold-rebuild into scratch and compare identity (the P1 "
                        "sufficiency gate; see _verify_one)")
    rl = sub.add_parser("replay", help="diagnostic only, not a gate -- see replay.py")
    rl.add_argument("manifest")
    rl.add_argument("ref_out")
    rl.add_argument("scratch")
    gr = sub.add_parser("gate-record")
    gr.add_argument("artifact_dir")
    gr.add_argument("gate")
    gr.add_argument("rc")
    gr.add_argument("json_file", nargs="?")
    gs = sub.add_parser("gate-status")
    gs.add_argument("artifact_dir")
    rp = sub.add_parser("repin-report")
    rp.add_argument("report_a")
    rp.add_argument("report_b")
    args = p.parse_args(argv)
    if args.cmd == "build":
        if not args.all and not args.recipe:
            p.error("build: give a recipe name or --all")
        cmd_build(args)
    elif args.cmd == "gate-record":
        cmd_gate_record(args)
    elif args.cmd == "gate-status":
        cmd_gate_status(args)
    elif args.cmd == "repin-report":
        cmd_repin_report(args)
    elif args.cmd == "replay":
        sys.exit(cmd_replay(args))


if __name__ == "__main__":
    main()
