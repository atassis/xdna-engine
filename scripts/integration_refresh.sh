#!/usr/bin/env python3
"""integration_refresh -- rebuild per-fork `integration` branches from the manifest.

Reads internal notes. For each repo it drops carries whose PR
has MERGED (queried live via gh -- necessary because upstream squash-merges, so
git's patch-id auto-drop does NOT fire), then cherry-picks the survivors onto fresh
upstream.

  plan  (default)  -- dry run: fetch upstream, show per-carry KEEP/DROP + the resulting
                      cherry-pick list. Mutates nothing.
  apply            -- LOCAL rebuild: create/reset <integration_branch> at upstream and
                      cherry-pick the survivors. Stops on first conflict. NEVER pushes.

  --repo NAME      -- limit to one repo (mlir-aie|IRON|mlir-air). Default: all carried.
  --repo tracked   -- just report tracked_only PR states (llvm-aie pin watch).

Push is always a separate manual step (git push <fork_remote> <integration_branch>),
so a bad rebuild can never reach a fork by accident.
"""
import argparse, json, os, subprocess, sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "internal notes")

def run(cmd, cwd, check=True, capture=True):
    r = subprocess.run(cmd, cwd=cwd, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}\n{r.stdout or ''}")
    return (r.stdout or "").strip(), r.returncode

def pr_state(repo, pr):
    out, rc = run(["gh", "pr", "view", str(pr), "--repo", repo, "--json", "state"], ROOT, check=False)
    if rc != 0:
        return "UNKNOWN"
    try:
        return json.loads(out)["state"]        # OPEN | MERGED | CLOSED
    except Exception:
        return "UNKNOWN"

def commits_by_subject(cwd, base, branch, subjects, exclude=None):
    """SHAs on branch-not-in-base whose subject contains one of `subjects`, oldest-first."""
    out, _ = run(["git", "log", "--reverse", "--no-merges", "--format=%H\t%s", f"{base}..{branch}"], cwd)
    picked = []
    for line in filter(None, out.splitlines()):
        sha, subj = line.split("\t", 1)
        if exclude and any(e in subj for e in exclude):
            continue
        if any(s in subj for s in subjects):
            picked.append((sha, subj))
    return picked

def commits_range(cwd, base, branch):
    """All SHAs on branch not in base, oldest-first (for a full PR stack). Skips merges."""
    out, _ = run(["git", "log", "--reverse", "--no-merges", "--format=%H\t%s", f"{base}..{branch}"], cwd)
    return [(l.split("\t",1)[0], l.split("\t",1)[1]) for l in filter(None, out.splitlines())]

def resolve_dir(repo_cfg):
    if repo_cfg.get("assemble_dir"):
        return repo_cfg["assemble_dir"]
    return os.path.join(ROOT, repo_cfg["checkout"])

def plan_repo(name, cfg, apply=False):
    cwd = resolve_dir(name if False else cfg)
    up_r, up_b = cfg["upstream_remote"], cfg["upstream_branch"]
    print(f"\n=== {name}  ({cwd})")
    print(f"    upstream {up_r}/{up_b}  ->  {cfg['integration_branch']}")
    run(["git", "fetch", up_r, up_b], cwd, check=False)
    base = f"{up_r}/{up_b}"
    base_sha, _ = run(["git", "rev-parse", "--short", base], cwd)
    # Optional: assemble on an already-validated local base branch that ALREADY contains
    # the `local` carries (e.g. sync/mlir-aie-latest = pinned base + build-speed patches).
    # Then we skip the local carries and only stack the PR-carries on top. Advancing that
    # base onto newer upstream (and reconciling) is a separate, deliberate refresh.
    assemble_base = cfg.get("assemble_base")
    wt_base = assemble_base if assemble_base else base
    print(f"    fresh upstream base: {base_sha}" +
          (f"   |   ASSEMBLE ON: {assemble_base} (carries locals already)" if assemble_base else ""))

    picks = []          # (sha, label, source_cwd)
    for c in cfg.get("carries", []):
        cid, kind = c["id"], c["kind"]
        if kind == "local" and assemble_base:
            print(f"    KEEP  [local]     {cid}: already present in {assemble_base} (not re-picked)")
            continue
        if kind == "local":
            got = commits_by_subject(cwd, base, c["source_branch"], c["subjects"],
                                     c.get("exclude_subjects"))
            print(f"    KEEP  [local]     {cid}: {len(got)} commit(s) from {c['source_branch']}")
            for sha, subj in got:
                picks.append((sha, f"{cid}: {subj[:60]}", cwd))
            miss = len(c["subjects"]) - len({s for _, s in got for x in c['subjects'] if x in s})
        elif kind in ("our-pr", "third-pty"):
            st = pr_state(c["pr_repo"] if "pr_repo" in c else _default_repo(name), c["pr"])
            drop = st == "MERGED" or (kind == "third-pty" and st == "CLOSED")
            tag = "DROP" if drop else "KEEP"
            note = "(merged upstream)" if st == "MERGED" else f"({st})"
            print(f"    {tag}  [{kind}] {cid}: PR #{c['pr']} {note}")
            if drop:
                continue
            if "source_branch" in c:                       # our branch present locally
                got = commits_range(cwd, base, c["source_branch"])
                for sha, subj in got:
                    picks.append((sha, f"{cid}: {subj[:60]}", cwd))
            else:                                          # third-party: fetch PR head
                head, _ = run(["gh","pr","view",str(c["pr"]),"--repo",c["pr_repo"],
                               "--json","headRefOid","-q",".headRefOid"], ROOT, check=False)
                run(["git","fetch",cfg["upstream_remote"],head], cwd, check=False)
                got = commits_range(cwd, base, head) if head else []
                for sha, subj in got:
                    picks.append((sha, f"{cid}: {subj[:60]}", cwd))
                if not got:
                    print(f"          !! could not resolve PR #{c['pr']} head commits -- carry manually")

    print(f"    -> cherry-pick plan ({len(picks)} commits):")
    for sha, label, _ in picks:
        print(f"         {sha[:12]}  {label}")

    if not apply:
        return
    ib = cfg["integration_branch"]
    # Assemble in an ISOLATED worktree -- never disturb the primary checkout, which may
    # hold other sessions' uncommitted work (shared-checkout hazard).
    # XDNA_CACHE (in-workspace build cache); fall back to the workspace root's .cache derived from this file.
    _xdna_cache = os.environ.get("XDNA_CACHE") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".cache")
    wt = os.path.join(_xdna_cache, "integration-wt", name)
    run(["git", "worktree", "remove", "--force", wt], cwd, check=False)
    run(["git", "worktree", "add", "-B", ib, wt, wt_base], cwd, check=False, capture=False)
    print(f"    APPLYING in worktree {wt}: cherry-pick {len(picks)} commits ...")
    for sha, label, _ in picks:
        _, rc = run(["git", "cherry-pick", sha], wt, check=False)
        if rc != 0:
            print(f"    !! CONFLICT on {sha[:12]} ({label}).")
            print(f"       Resolve in {wt}, `git cherry-pick --continue`, then re-run apply, OR abort:")
            print(f"       (cd {wt} && git cherry-pick --abort)")
            raise SystemExit(2)
    head, _ = run(["git", "rev-parse", "--short", "HEAD"], wt)
    print(f"    OK: {ib} rebuilt on {base_sha} in {wt} (HEAD {head}). NOT pushed.")
    print(f"       review: (cd {wt} && git log --oneline {base}..{ib})")
    print(f"       push:   git -C {wt} push {cfg['fork_remote']} {ib}   # after device-gate")
    if cfg.get("device_gate"):
        print(f"    NEXT: device-gate this build, then bump {cfg.get('pins')} in toolchain.lock before trusting it.")

def _default_repo(name):
    return {"mlir-aie":"Xilinx/mlir-aie","IRON":"amd/IRON","mlir-air":"Xilinx/mlir-air"}[name]

def report_tracked(m):
    to = m.get("tracked_only", {}) or {}
    print("\n=== tracked_only (no carried stack -- pin-watch)")
    for t in to.get("prs", []):
        print(f"    {t['repo']} #{t['pr']}: {pr_state(t['repo'], t['pr'])}  -- {t.get('note','')}")
    if to.get("loop_closer"):
        print(f"    loop-closer: {to['loop_closer']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="plan", choices=["plan", "apply"])
    ap.add_argument("--repo", default="all")
    args = ap.parse_args()
    m = yaml.safe_load(open(MANIFEST))
    if args.repo == "tracked":
        report_tracked(m); return
    repos = m["repos"]
    targets = repos if args.repo == "all" else {args.repo: repos[args.repo]}
    for name, cfg in targets.items():
        plan_repo(name, cfg, apply=(args.mode == "apply"))
    if args.repo == "all":
        report_tracked(m)
    print("\nDone. (apply rebuilds locally only -- push each fork manually when device-gated.)")

if __name__ == "__main__":
    main()
