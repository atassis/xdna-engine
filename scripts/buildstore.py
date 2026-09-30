#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI over scripts/buildstore/{record,store,identity}.py.

build <recipe>... [--recipes TSV] [--out-root DIR] [--no-hit]: take the recipe's lock, check
stored actions newest-first for a hit, else run the recipe under the recorder and store the
result. TSV format matches run_s0.sh: tab-separated, `#` comments, NORECIPE skipped.
"""
import argparse, hashlib, json, os, pathlib, shutil, sys, time

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


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("recipe", nargs="+")
    b.add_argument("--recipes", type=pathlib.Path, default=None)
    b.add_argument("--out-root", required=True)
    b.add_argument("--no-hit", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "build":
        cmd_build(args)


if __name__ == "__main__":
    main()
