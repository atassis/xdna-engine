# SPDX-License-Identifier: Apache-2.0
"""Run a build command under rec_preload.so with a declared environment, and check a manifest.

Log kinds from rec_preload.c: R read, W write, D dir listed, A absent, M mapped, L symlink.
BUILDSTORE_SABOTAGE=reads|dirs|absent|env drops one class, for the tests that prove each matters.
"""
import hashlib, json, os, pathlib, shutil, subprocess, tempfile

HERE = pathlib.Path(__file__).resolve().parent
EXCLUDED = ("/proc/", "/sys/", "/dev/", "/run/", "/tmp/")
_statcache = {}


def preload_so():
    return subprocess.run([str(HERE / "build_preload.sh")], check=True, capture_output=True,
                          text=True).stdout.strip()


def hermetic_env(environ, names):
    return {n: environ[n] for n in names if n in environ}


def allowlist():
    return [l.strip() for l in (HERE / "hermetic_env.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def sha_file(path):
    st = os.stat(path)
    sig = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    hit = _statcache.get(path)
    if hit and hit[:5] == sig:
        return hit[5]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _statcache[path] = sig + [h.hexdigest()]
    return h.hexdigest()


def sha_dir(path):
    return hashlib.sha256("\n".join(sorted(os.listdir(path))).encode()).hexdigest()


def load_statcache(path):
    try:
        _statcache.update(json.load(open(path)))
    except (OSError, ValueError):
        pass


def save_statcache(path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_statcache, f)
    os.replace(tmp, path)


def _under(p, roots):
    return any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)


def manifest_from_logs(rec_dir, argv, env, skip_roots):
    ev = {k: set() for k in "RWDAM"}
    links = {}
    for f in pathlib.Path(rec_dir).iterdir():
        for line in f.read_text(errors="replace").splitlines():
            kind, _, rest = line.partition(" ")
            if kind == "L":
                given, _, target = rest.partition("\t")
                links[given] = target
            elif kind in ev:
                ev[kind].add(rest)
    skip = [r for r in skip_roots if r] + [e.rstrip("/") for e in EXCLUDED]
    written = ev["W"]
    # Git metadata is read for provenance (gen_llm_decode.py's rev-parse/status), which identity.py
    # excludes; as a key input it would turn every commit into a fleet-wide miss.
    keep = lambda p: p not in written and not _under(p, skip) and "/.git/" not in p + "/"
    sab = os.environ.get("BUILDSTORE_SABOTAGE", "")
    reads = {p: sha_file(p) for p in sorted((ev["R"] | ev["M"]))
             if keep(p) and os.path.isfile(p)} if sab != "reads" else {}
    dirs = {p: sha_dir(p) for p in sorted(ev["D"])
            if keep(p) and os.path.isdir(p)} if sab != "dirs" else {}
    absent = sorted(p for p in ev["A"] if keep(p) and not os.path.lexists(p)) \
        if sab != "absent" else []
    links = {g: t for g, t in links.items() if t in reads}
    return {"argv": argv, "env": env if sab != "env" else {}, "reads": reads, "dirs": dirs,
            "absent": absent, "links": links}


def run(argv, cwd, env, out_roots, work_roots, cache_roots, time_file=None):
    """Run argv under the recorder; return its manifest. Raises CalledProcessError on failure."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("BUILDSTORE_SCRATCH")) as rec:
        full = dict(env, LD_PRELOAD=preload_so(), REC_DIR=rec)
        cmd = (["/usr/bin/time", "-v", "-o", str(time_file)] if time_file else []) + list(argv)
        subprocess.run(cmd, cwd=cwd, env=full, check=True)
        writable = [str(r) for r in (*out_roots, *work_roots, *cache_roots)]
        m = manifest_from_logs(rec, list(argv), env, writable + [rec])
        m.update(cwd=str(cwd), writable=writable)
        return m


def check(m, env=None, argv=None):
    """Every recorded input that changed, as '<class> <name>'. Empty = the recorded run stands."""
    out = ["argv"] if argv is not None and list(argv) != m["argv"] else []
    if env is not None:
        out += [f"env {k}" for k in sorted(set(m["env"]) | set(env))
                if m["env"].get(k) != env.get(k)]
    for g, t in m["links"].items():
        if os.path.realpath(g) != t:
            out.append(f"link {g}")
    for p, h in m["reads"].items():
        if not os.path.isfile(p) or sha_file(p) != h:
            out.append(f"file {p}")
    for p, h in m["dirs"].items():
        if not os.path.isdir(p) or sha_dir(p) != h:
            out.append(f"dir {p}")
    out += [f"appeared {p}" for p in m["absent"] if os.path.lexists(p)]
    return out
