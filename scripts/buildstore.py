#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI over scripts/buildstore/{record,store,identity}.py.

build <recipe>... [--recipes TSV] [--out-root DIR] [--no-hit]: take the recipe's lock, check
stored actions newest-first for a hit, else run the recipe under the recorder and store the
result. TSV format matches run_s0.sh: tab-separated, `#` comments, NORECIPE skipped.
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


def cmd_build(args):
    cas = pathlib.Path(os.environ.get("BUILDSTORE_CAS", "/mnt/data/xdna/cas"))
    repo = pathlib.Path(os.environ.get("BUILDSTORE_REPO", str(WS / "xdna-engine")))
    recipes_path = args.recipes or (HERE / "s0_recipes.tsv")
    recipes = parse_recipes(recipes_path)
    s = store.Store(cas)
    statcache = cas / "statcache.json"
    record.load_statcache(str(statcache))
    out_root = pathlib.Path(args.out_root)

    for name in args.recipe:
        cmd = recipes.get(name)
        if cmd is None or cmd == "NORECIPE":
            print(f"SKIP {name} NORECIPE")
            continue
        t0 = time.time()
        with s.lock(name):
            argv = ["bash", "-c", f"cd {repo} && {cmd}"]
            env = record.hermetic_env(os.environ, record.allowlist())
            env["REPO"] = str(repo)
            resolve_xdna_cache(repo, env)
            resolve_mlir_aie_instance(repo, env)
            want = iron_pin_verified(repo, env)
            if want:
                env["IRON_PIN_VERIFIED"] = want

            staging = cas / "staging" / name
            out = staging / "out"
            work = staging / "work"
            npu_cache = cas / "caches" / name / "npu_cache"
            ccache_dir = cas / "caches" / name / "ccache"
            env["OUT"] = str(out)
            env["KEEP_WORK"] = str(work)
            env["XDNA_BLOB_POOL"] = "0"
            env["NPU_CACHE_HOME"] = str(npu_cache)
            env["CCACHE_DIR"] = str(ccache_dir)
            cache_roots = [npu_cache, ccache_dir]
            if env.get("XDNA_CACHE"):
                cache_roots.append(pathlib.Path(env["XDNA_CACHE"]) / "kobj")

            if not args.no_hit:
                hit = next((a for a in s.actions(name)
                           if record.check(a["manifest"], env=env, argv=argv) == []), None)
                if hit is not None:
                    s.materialize(hit["tree"], out_root / name)
                    print(f"HIT {name} {hit['identity']} {time.time() - t0:.2f}")
                    continue

            shutil.rmtree(staging, ignore_errors=True)
            out.mkdir(parents=True)
            work.mkdir(parents=True)
            npu_cache.mkdir(parents=True, exist_ok=True)
            ccache_dir.mkdir(parents=True, exist_ok=True)
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
            print(f"BUILT {name} {ident} {time.time() - t0:.2f} {peak_kb}")

    record.save_statcache(str(statcache))


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
    b.add_argument("recipe", nargs="+")
    b.add_argument("--recipes", type=pathlib.Path, default=None)
    b.add_argument("--out-root", required=True)
    b.add_argument("--no-hit", action="store_true")
    gr = sub.add_parser("gate-record")
    gr.add_argument("artifact_dir")
    gr.add_argument("gate")
    gr.add_argument("rc")
    gr.add_argument("json_file", nargs="?")
    gs = sub.add_parser("gate-status")
    gs.add_argument("artifact_dir")
    args = p.parse_args(argv)
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "gate-record":
        cmd_gate_record(args)
    elif args.cmd == "gate-status":
        cmd_gate_status(args)


if __name__ == "__main__":
    main()
