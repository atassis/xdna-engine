#!/usr/bin/env python3
"""List every `artifacts/...`-relative path a SERVED scenario references.

Used by install.sh to build the served-only artifacts tree (link only what's
loaded, not the whole store). Reads engine.toml's `[[model]] scenario = "..."`
entries -- same source install.sh's own preflight already walks -- then, for
each referenced scenario TOML, pulls every string field that names an
`artifacts/`-relative path out of the field set `rust/npu-models/src/config.rs`
declares on `Artifacts`, `DiarizationCfg`, `TtsCfg` and `MultimodalCfg`.

A path already covered by a shorter path already in the set (e.g. `decode`'s
own dir and `weights = "<decode>/buffers"`) is dropped: it resolves through
the ancestor's symlink once that is staged, and creating a second link
*inside* it would write into the artifact's SOURCE directory instead of the
staged tree.

Usage: collect_served_artifacts.py <engine.toml> <root>
`root` is the base engine.toml's `scenario = "scenarios/..."` paths are relative
to (the staged root, or the repo checkout when run pre-stage) -- NOT the
scenarios/ dir itself, matching how install.sh's own preflight resolves them.
Prints one artifacts/-relative path per line, sorted, deduplicated.
"""
import sys
import tomllib
from pathlib import Path

ARTIFACT_FIELDS = ("weights", "tokenizer", "onnx_ref", "decode", "prefill",
                    "tokenizer_dir", "resident", "nli_head", "checkpoint")
DIARIZATION_FIELDS = ("manifest",)
TTS_FIELDS = ("slow_ar", "fast_ar", "codec")
MULTIMODAL_FIELDS = ("tower_checkpoint",)


def served_scenarios(engine_config: Path) -> list[str]:
    cfg = tomllib.loads(engine_config.read_text())
    return [m["scenario"] for m in cfg.get("model", []) if "scenario" in m]


def scenario_artifact_paths(scen: dict) -> list[str]:
    out = []
    for section, fields in (
        ("artifacts", ARTIFACT_FIELDS),
        ("diarization", DIARIZATION_FIELDS),
        ("tts", TTS_FIELDS),
        ("multimodal", MULTIMODAL_FIELDS),
    ):
        block = scen.get(section, {})
        for field in fields:
            v = block.get(field, "")
            if isinstance(v, str) and v.startswith("artifacts/"):
                out.append(v)
    return out


def drop_covered(paths: set[str]) -> list[str]:
    """Drop any path that sits inside another path already in the set."""
    ordered = sorted(paths, key=len)
    kept: list[str] = []
    for p in ordered:
        if any(p == k or p.startswith(k + "/") for k in kept):
            continue
        kept.append(p)
    return sorted(kept)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    engine_config, root = Path(sys.argv[1]), Path(sys.argv[2])

    all_paths: set[str] = set()
    for scen_rel in served_scenarios(engine_config):
        scen_path = Path(scen_rel)
        if not scen_path.is_absolute():
            scen_path = root / scen_path
        if not scen_path.is_file():
            print(f"# WARN: served scenario not found: {scen_path}", file=sys.stderr)
            continue
        scen = tomllib.loads(scen_path.read_text())
        all_paths.update(scenario_artifact_paths(scen))

    for p in drop_covered(all_paths):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
