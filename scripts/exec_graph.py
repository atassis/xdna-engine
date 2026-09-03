#!/usr/bin/env python3
"""Emit the execution graph as DATA, for every model the tree can build.

Why this exists. `docs/reference/npu-dataflow-map.md` is a METHOD with no instance and nothing
generating it, so every session re-derives the graph by reading dispatch code and settles
disagreements by argument. One such re-derivation (op adjacency, read off a per-op timing list)
became load-bearing for a design invariant on 2026-09-03.

The contract, and the reason this is JSON and not a diagram:

  * Every field carries an `ev` (evidence) tag. `disk` and `config` are READ from the tree.
    `declared` is a table in this file that cites the code it mirrors -- it is the weakest tier and
    is the one a runtime trace must overrule. `derived` is computed from the others.
  * A picture is a VIEW. `--mermaid` renders one FROM this data so it cannot disagree with it.
    Never hand-maintain the picture.
  * `hw_context` is the load-bearing column. The fusion cost model is a claim about context
    ALTERNATIONS along execution order, so a graph without it cannot answer the question it exists
    for -- and it is exactly the column a diagram would omit.

Honest limit: the op ORDER and unit assignment are `declared` here, not measured. That is a second
source of truth and it WILL drift -- which is the failure this file otherwise argues against. The
mitigation is `--check`: once a runtime trace exists (ENC_PEROP_TIMING emitting order + context),
it becomes the source and this table is diffed against it rather than trusted.

Usage:
  scripts/exec_graph.py                 # JSON to stdout
  scripts/exec_graph.py --mermaid       # rendered view
  scripts/exec_graph.py --summary       # human table
  scripts/exec_graph.py --check         # non-zero if any model's declared path is unbuildable
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEMM = REPO / "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
MHA_DIR = REPO / "artifacts/encoder_mha"
ENGINE_TOML = Path.home() / ".config/npu/engine.toml"

# Whisper's modal GEMM augments K by 32 for the on-chip bias row (768->800, 1280->1312). That is
# NOT universal: parakeet's kernels are built at K = d_model exactly (1024). Matching on the
# augmented value alone reported parakeet as having no kernels while its whole set was on disk, so
# try both conventions and record which one matched rather than assuming either.
K_AUG_BIAS = 32

# The declared schedule below describes the WHISPER encoder block only. Parakeet and GigaAM are
# FastConformers -- different block structure (conv module, depthwise conv, different norms), so
# applying this schedule to them would fabricate a graph. Models outside this set get config +
# kernel inventory and an explicit `schedule: not_declared`, never an invented op list.
WHISPER_FAMILY = ("whisper-small", "whisper-turbo")

# ---------------------------------------------------------------------------------------------
# DECLARED: the encoder op sequence. Mirrors rust/npu-whisper/src/encoder.rs (block()) and
# rust/npu-asr/src/ctx2.rs. Weakest evidence tier -- cite the site, and let --check overrule it.
# `ctx` is the hardware context: ops sharing one string pay NO transition between them
# (program changes inside a context measured -1.0 us; context changes 1253 us).
# ---------------------------------------------------------------------------------------------
ENCODER_OPS = [
    # name,        unit,   ctx,     n_kind,     gate env,             default_on, cite
    ("conv_stem",  "host", None,    None,       "NPU_ENC_CONV_NPU",   False, "encoder.rs conv stem"),
    ("ln",         "host", None,    None,       "NPU_LN_NPU",         False, "encoder.rs layernorm"),
    ("qkv_proj",   "npu",  "ctx2",  "d_model",  None,                 True,  "ctx2.rs modal GEMM"),
    ("mha",        "npu",  "mha",   None,       "NPU_ENC_MHA_NPU",    True,  "encoder.rs mha_npu (default flipped 2026-09-03)"),
    ("out_proj",   "npu",  "ctx2",  "d_model",  None,                 True,  "ctx2.rs modal GEMM"),
    ("fc1",        "npu",  "ctx2",  "ffn",      None,                 True,  "ctx2.rs modal GEMM"),
    ("gelu",       "host", None,    None,       "NPU_ENC_GELU_FUSED", False, "encoder.rs gelu / fused epilogue"),
    ("fc2",        "npu",  "ctx2",  "d_model",  None,                 True,  "ctx2.rs modal GEMM"),
    ("residual",   "host", None,    None,       None,                 True,  "encoder.rs residual add"),
]

# Known hazards, keyed to the KB note that owns each. Annotation only -- never a pass/fail on its own.
HAZARDS = [
    {"match": {"tile": "64x32x96"},
     "severity": "broken",
     "note": "ctx2 GEMM hangs with ERT_CMD_STATE_TIMEOUT at this tile; 64x32x32 / 32x32x96 / 32x32x32 all pass",
     "kb": "the-whisper-small-hang-is-an-m-n-interaction-not-one-parameter"},
    {"match": {"op": "mha"},
     "severity": "note",
     "note": "own hw_context: cannot co-reside with the modal GEMM (both saturate core DMA 2/2, MemTile 6/6)",
     "kb": "mha-cannot-co-reside-with-the-modal-gemm-the-ports-are-gone"},
]


def sh(*a):
    try:
        return subprocess.run(a, cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def kernel_inventory():
    """Every built whole_array GEMM, parsed from its own filename. ev=disk."""
    rx = re.compile(r"^final_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)c_(\w+)\.xclbin$")
    out = []
    for f in sorted(GEMM.glob("final_*.xclbin")) if GEMM.is_dir() else []:
        m = rx.match(f.name)
        if not m:
            continue
        M, K, N, tm, tk, tn, cols, mode = m.groups()
        st = f.stat()
        out.append({"M": int(M), "K_aug": int(K), "N": int(N),
                    "tile": f"{tm}x{tk}x{tn}", "cols": int(cols), "mode": mode,
                    "file": f.name, "bytes": st.st_size, "mtime": int(st.st_mtime)})
    return out


def mha_inventory():
    rx = re.compile(r"^StaticMHA_h(\d+)_s(\d+)_d(\d+)_kv\d+(_causal0)?_npu2\.xclbin$")
    out = []
    for f in sorted(MHA_DIR.glob("StaticMHA_*.xclbin")) if MHA_DIR.is_dir() else []:
        m = rx.match(f.name)
        if not m:
            continue
        h, s, d, legacy = m.groups()
        out.append({"heads": int(h), "seq": int(s), "head_dim": int(d),
                    "legacy_causal0_name": bool(legacy), "file": f.name,
                    "bytes": f.stat().st_size, "mtime": int(f.stat().st_mtime)})
    return out


def scenarios():
    """Model shapes, read from scenarios/*.toml. ev=config."""
    try:
        import tomllib
    except ImportError:
        return []
    out = []
    reg = ENGINE_TOML.read_text() if ENGINE_TOML.is_file() else ""
    for p in sorted((REPO / "scenarios").glob("*.toml")):
        try:
            d = tomllib.loads(p.read_text())
        except Exception:
            continue
        sc, mo = d.get("scenario", {}), d.get("model", {})
        if sc.get("kind") != "asr" or not mo.get("hidden"):
            continue
        name = sc.get("name", p.stem)
        out.append({"name": name, "scenario": p.name,
                    "d_model": mo.get("hidden"), "ffn": mo.get("ff"),
                    "n_heads": mo.get("n_heads"), "head_dim": mo.get("head_dim"),
                    "n_layers": mo.get("n_layers"), "max_seq": mo.get("max_seq"),
                    "K_candidates": [(mo.get("hidden") or 0) + K_AUG_BIAS, mo.get("hidden") or 0],
                    # Key on the scenario PATH, not the display name: engine.toml registers
                    # `scenario = "scenarios/foo.toml"` and its model `name` need not match the
                    # scenario's own. Matching the name reported parakeet unregistered while it
                    # was loaded and serving.
                    "registered_in_engine_toml": p.name in reg})
    return out


def build_model(sc, gemms, mhas):
    # Pick whichever K convention actually has kernels on disk; prefer the bias-augmented one.
    kaug, mine = sc["K_candidates"][0], []
    for cand in sc["K_candidates"]:
        hit = [k for k in gemms if k["K_aug"] == cand]
        if hit:
            kaug, mine = cand, hit
            break
    sc = {**sc, "K_aug": kaug, "k_convention": "d_model+32" if kaug != sc["K_candidates"][1] else "d_model"}
    if sc["name"] not in WHISPER_FAMILY:
        # No declared schedule for this architecture. Report what IS evidence -- config and the
        # kernels on disk -- and say plainly that the op graph is unknown. A fabricated schedule
        # here would be the exact drift this emitter exists to prevent.
        return {
            "name": sc["name"], "scenario": sc["scenario"],
            "config": {k: sc[k] for k in ("d_model", "ffn", "n_heads", "head_dim", "n_layers", "max_seq", "K_aug")},
            "k_convention": sc["k_convention"],
            "registered_in_engine_toml": sc["registered_in_engine_toml"],
            "schedule": "not_declared",
            "health": {"status": "unknown", "missing": [],
                       "hazards": [{"severity": "note",
                                    "note": "no declared op schedule: not a Whisper-family block "
                                            "(FastConformer). Kernel inventory shown; op graph unknown.",
                                    "kb": "execution-graph-must-be-generated-and-gated"}]},
            "ops": [],
            "derived": {"gemm_tiles_built": sorted({k["tile"] for k in mine}),
                        "gemm_shapes_built": sorted({f'{k["K_aug"]}x{k["N"]}' for k in mine}),
                        "ev": "derived"},
        }
    n_for = {"d_model": sc["d_model"], "ffn": sc["ffn"]}
    tiles = sorted({k["tile"] for k in mine})
    ops, missing = [], []
    for order, (name, unit, ctx, nkind, env, dflt, cite) in enumerate(ENCODER_OPS):
        op = {"order": order, "op": name, "unit": unit, "hw_context": ctx,
              "gate": {"env": env, "default_on": dflt, "ev": "declared"},
              "ev": "declared", "cite": cite}
        if unit == "npu" and nkind:
            N = n_for[nkind]
            cands = [k for k in mine if k["N"] == N]
            op["shape"] = {"M": 512, "K_aug": kaug, "N": N, "ev": "derived"}
            op["artifacts"] = sorted({k["file"] for k in cands})
            op["tiles_available"] = sorted({k["tile"] for k in cands})
            op["artifact_present"] = bool(cands)
            if not cands:
                missing.append(f"{name}: no GEMM at K_aug={kaug} N={N}")
        if name == "mha":
            cands = [m for m in mhas if m["heads"] == sc["n_heads"]]
            op["artifacts"] = [m["file"] for m in cands]
            op["artifact_present"] = bool(cands)
            if not cands:
                missing.append(f"mha: no StaticMHA at h={sc['n_heads']}")
        ops.append(op)

    # DERIVED: context alternations along execution order -- the term the fusion model turns on.
    seq = [o["hw_context"] for o in ops if o["unit"] == "npu"]
    alts = sum(1 for a, b in zip(seq, seq[1:]) if a != b)

    haz = []
    for h in HAZARDS:
        if "tile" in h["match"] and h["match"]["tile"] in tiles:
            haz.append({**h, "applies_to": f"tile {h['match']['tile']} present for this model"})
        if "op" in h["match"] and any(o["op"] == h["match"]["op"] and o["unit"] == "npu" for o in ops):
            haz.append({**h, "applies_to": f"op {h['match']['op']}"})

    status = "unbuildable" if missing else ("hazard" if any(x["severity"] == "broken" for x in haz) else "ok")
    return {
        "name": sc["name"], "scenario": sc["scenario"],
        "config": {k: sc[k] for k in ("d_model", "ffn", "n_heads", "head_dim", "n_layers", "max_seq", "K_aug")},
        "k_convention": sc["k_convention"],
        "registered_in_engine_toml": sc["registered_in_engine_toml"],
        "health": {"status": status, "missing": missing, "hazards": haz},
        "ops": ops,
        "derived": {
            "npu_ops": sum(1 for o in ops if o["unit"] == "npu"),
            "host_ops": sum(1 for o in ops if o["unit"] == "host"),
            "hw_contexts": sorted({o["hw_context"] for o in ops if o["hw_context"]}),
            "context_alternations_per_block": alts,
            "context_alternations_per_clip": alts * (sc["n_layers"] or 0),
            "gemm_tiles_built": tiles,
            "ev": "derived",
        },
    }


def graph():
    gemms, mhas = kernel_inventory(), mha_inventory()
    models = [build_model(s, gemms, mhas) for s in scenarios()]
    return {
        "schema": "xdna-exec-graph/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_commit": sh("git", "rev-parse", "--short=12", "HEAD"),
            "git_dirty": bool(sh("git", "status", "--porcelain")),
            "mlir_aie_pin": (re.search(r"MLIR_AIE_FORK_COMMIT=(\S+)", (REPO / "toolchain.lock").read_text()).group(1)[:12]
                             if (REPO / "toolchain.lock").is_file() else None),
            "ev": "derived",
        },
        "evidence_tiers": {
            "disk": "read from built artifacts on disk",
            "config": "read from scenarios/*.toml and engine.toml",
            "declared": "table in scripts/exec_graph.py citing the code it mirrors -- WEAKEST, overruled by a runtime trace",
            "derived": "computed from the above",
        },
        "models": models,
        "kernel_inventory": {"gemm": gemms, "encoder_mha": mhas, "ev": "disk"},
    }


def mermaid(g):
    L = ["```mermaid", "flowchart LR"]
    for mi, m in enumerate(g["models"]):
        st = m["health"]["status"]
        L.append(f'  subgraph M{mi}["{m["name"]} — {st} — {m["config"]["n_layers"]}L d={m["config"]["d_model"]}"]')
        L.append("    direction LR")
        prev = None
        for o in m["ops"]:
            nid = f"M{mi}_{o['order']}"
            if o["unit"] == "host":
                label = f'{o["op"]}<br/>host'
                shape = f'{nid}(["{label}"])'
            else:
                bad = o.get("artifact_present") is False
                label = f'{o["op"]}<br/>npu · {o["hw_context"]}' + ("<br/>MISSING" if bad else "")
                shape = f'{nid}["{label}"]'
            L.append("    " + shape)
            if prev:
                a = next(x for x in m["ops"] if f"M{mi}_{x['order']}" == prev)
                cross = a["hw_context"] != o["hw_context"] and a["unit"] == "npu" and o["unit"] == "npu"
                L.append(f"    {prev} {'==>' if cross else '-->'} {nid}")
            prev = nid
        L.append("  end")
    L.append("```")
    L.append("")
    L.append("`==>` marks a hardware-context alternation (0.79-3.55 ms each); `-->` is free inside a context.")
    return "\n".join(L)


def summary(g):
    out = [f"exec-graph  commit={g['provenance']['git_commit']}  pin={g['provenance']['mlir_aie_pin']}"
           f"{'  [DIRTY]' if g['provenance']['git_dirty'] else ''}", ""]
    for m in g["models"]:
        d, h = m["derived"], m["health"]
        if m.get("schedule") == "not_declared":
            out.append(f"{m['name']:<20} {h['status']:<12} layers={m['config']['n_layers']:<3} "
                       f"K_aug={m['config']['K_aug']:<5} schedule=not_declared "
                       f"reg={'y' if m['registered_in_engine_toml'] else 'n'}")
            out.append(f"{'':<20} tiles: {', '.join(d['gemm_tiles_built']) or '(none at this K_aug)'}")
            out.append("")
            continue
        out.append(f"{m['name']:<16} {h['status']:<12} layers={m['config']['n_layers']:<3} "
                   f"K_aug={m['config']['K_aug']:<5} npu={d['npu_ops']} host={d['host_ops']} "
                   f"ctx={','.join(d['hw_contexts']) or '-'} alt/blk={d['context_alternations_per_block']} "
                   f"alt/clip={d['context_alternations_per_clip']} "
                   f"reg={'y' if m['registered_in_engine_toml'] else 'n'}")
        out.append(f"{'':<16} tiles: {', '.join(d['gemm_tiles_built']) or '(none built)'}")
        for x in h["missing"]:
            out.append(f"{'':<16} MISSING  {x}")
        for x in h["hazards"]:
            out.append(f"{'':<16} {x['severity'].upper():<8} {x['note']}  [[{x['kb']}]]")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mermaid", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 if any model's declared path is unbuildable")
    a = ap.parse_args()
    g = graph()
    if a.mermaid:
        print(mermaid(g))
    elif a.summary:
        print(summary(g))
    else:
        print(json.dumps(g, indent=2))
    if a.check:
        # Only models with a declared schedule can be judged unbuildable; "unknown" is not a failure.
        bad = [m["name"] for m in g["models"] if m["health"]["status"] == "unbuildable"]
        if bad:
            print(f"exec-graph: unbuildable: {', '.join(bad)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
