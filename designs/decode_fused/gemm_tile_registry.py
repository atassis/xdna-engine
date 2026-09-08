#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measured GEMM tile choices, keyed by shape -- and a HARD FAILURE for any shape not in it.

The prefill generators used to carry `TILE_M = TILE_K = TILE_N = 64` and `num_aie_columns = 8` as
module constants, applied to every GEMM in the model: qkv (K=1024, N=2048/1024), o (K=2048,
N=1024), gate/up (K=1024, N=3072), down (K=3072, N=1024), scores (K=128, N=2048), ctx (K=2048,
N=128). One triple, six shapes, and the triple was DERIVED (it divides, it fits in L1) rather than
measured. That `ctx` already needed `tile_n=16` -- because `128 % (64*8)` fails -- is the standing
proof that one size does not fit: the constant had already been overridden once by the arithmetic,
and never by a measurement.

Tile choice is a first-order knob on both axes this rail cares about:

  * TIME. It sets the shape of the mmul the core issues, the L1 footprint, the number of k-tiles
    the reduction loop runs, and how many BDs a C transfer costs.
  * NUMERICS. `iron/operators/gemm/design.py` makes the L1 C buffer the OUTPUT dtype unless
    `prio_accuracy` is set, so the partial sum is rounded to bf16 once per k-tile -- `K/tile_k`
    times. Halving tile_k doubles the number of narrowings. That is why `prio_accuracy` is part of
    the KEY here and not a footnote: the winning tile under an f32 accumulator is not necessarily
    the winning tile under a bf16 one.

So the tile is data, measured per shape, and this file is where the measurement lands.

=== Raise on a miss, always ===

`lookup()` raises `UnsweptGemmShape` for a shape with no entry. It does NOT fall back to 64/64/64,
and adding such a fallback would delete the entire point of the file -- a registry that quietly
defaults is the hardcoded triple with extra steps, minus the honesty of having it written in one
place. The raise names the sweep command that fills the hole.

A deliberately UNMEASURED entry is still allowed, and is not the same thing: `sweep_gemm_tiles.py
--seed` writes one with `source: "seed"` (what the generators used before this file existed) or
`"assumed"` (a human's choice, pending a sweep). Both are visible in the JSON and in
`registry_status()`; a `source: "sweep"` entry is the only one carrying device numbers.

=== The key ===

`(M, K, N, dtype, emulate, prio_accuracy)`. `M` is the token batch, `K`/`N` the projection's own
dims. `dtype` is `in>out` (`bf16>bf16` today). `emulate` is
`GEMM(emulate_bf16_mmul_with_bfp16=...)`: it picks the microkernel's (r,s,t) and therefore which
tiles are legal at all, so two arms with different `emulate` are not comparable and must not share
an entry.

`b_col_maj` is NOT part of the key -- every prefill projection reads its weight in decode's stored
`[Nout, K]` order, and the one exception (`ctx`, which reads the V cache as `[K=S, N=HD]`) is
already distinguished by its shape. It is RECORDED per entry anyway and `lookup()` raises if a
caller asks with a different one, because the B streaming pattern differs between the two
(`design.py` picks `dims_to_stream` on it) and a silently-shared measurement would be a lie.
"""
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from llm_decode_spec import gemm_tiling_rejection

DEFAULT_REGISTRY = Path(__file__).with_name("gemm_tiles.json")
SCHEMA = 1

#: `source` values, weakest evidence first. Only "sweep" carries device numbers.
SOURCES = ("seed", "assumed", "sweep")


class UnsweptGemmShape(LookupError):
    """No registry entry for this shape. Run the sweep; do not guess a tile."""


class IllegalRegistryEntry(ValueError):
    """An entry exists but its tiling is not legal for the shape it is filed under."""


@dataclass(frozen=True)
class TileChoice:
    tile_m: int
    tile_k: int
    tile_n: int
    cols: int
    source: str
    b_col_maj: bool | None = None
    measured: dict | None = None
    label: str | None = None

    @property
    def gemm_kwargs(self) -> dict:
        """The GEMM(...) keyword arguments this choice sets. `b_col_maj` stays the caller's."""
        return dict(tile_m=self.tile_m, tile_k=self.tile_k, tile_n=self.tile_n,
                    num_aie_columns=self.cols)

    def __str__(self):
        return (f"{self.tile_m}x{self.tile_k}x{self.tile_n}@{self.cols}cols "
                f"[{self.source}{'' if not self.label else ' ' + self.label}]")


def key_of(M: int, K: int, N: int, *, dtype: str = "bf16>bf16", emulate: bool = True,
           prio_accuracy: bool = False) -> str:
    """The canonical registry key. Every field that changes which tile wins is in it."""
    return (f"m{M}_k{K}_n{N}_{dtype.replace('>', '2')}"
            f"_bfp{int(bool(emulate))}_acc{int(bool(prio_accuracy))}")


def _parse_overrides(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"GEMM_TILES_OVERRIDE is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("GEMM_TILES_OVERRIDE must be a JSON object keyed by registry key or "
                         "op label, e.g. '{\"scores\": {\"tile_n\": 32}}'")
    return parsed


class Registry:
    """The on-disk tile registry, loaded once and consulted per GEMM."""

    def __init__(self, entries: dict, path: Path | None = None, overrides: dict | None = None):
        self.entries = entries
        self.path = path
        self.overrides = overrides if overrides is not None else \
            _parse_overrides(os.environ.get("GEMM_TILES_OVERRIDE"))

    # ---- I/O ----
    @classmethod
    def load(cls, path=None, overrides: dict | None = None) -> "Registry":
        path = Path(path or os.environ.get("GEMM_TILES_JSON") or DEFAULT_REGISTRY)
        if not path.is_file():
            raise FileNotFoundError(
                f"no GEMM tile registry at {path}. It is version-controlled next to the "
                f"generators; set GEMM_TILES_JSON to point elsewhere, or seed one with "
                f"`sweep_gemm_tiles.py --seed-current`.")
        doc = json.loads(path.read_text())
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema {doc.get('schema')!r}, this module speaks {SCHEMA}")
        return cls(doc.get("entries", {}), path=path, overrides=overrides)

    def save(self, path=None) -> Path:
        path = Path(path or self.path or DEFAULT_REGISTRY)
        doc = {
            "schema": SCHEMA,
            "note": ("Measured GEMM tile choices. See gemm_tile_registry.py. A lookup for a shape "
                     "absent here RAISES; nothing falls back to a default triple."),
            "entries": dict(sorted(self.entries.items())),
        }
        path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        return path

    # ---- the lookup that must not guess ----
    def lookup(self, M: int, K: int, N: int, *, dtype: str = "bf16>bf16", emulate: bool = True,
               prio_accuracy: bool = False, b_col_maj: bool | None = None,
               label: str | None = None) -> TileChoice:
        """The winning tile for this shape, or `UnsweptGemmShape`.

        `label` is the op's name in the graph ("scores", "down", ...). It is used for the error
        message and as an override handle; it is NOT part of the key, because two ops with the
        same shape have the same answer.
        """
        key = key_of(M, K, N, dtype=dtype, emulate=emulate, prio_accuracy=prio_accuracy)
        ent = self.entries.get(key)
        ovr = self.overrides.get(key) or (self.overrides.get(label) if label else None)
        if ent is None and ovr is None:
            raise UnsweptGemmShape(self._miss_message(key, M, K, N, dtype, emulate,
                                                      prio_accuracy, label))
        merged = dict(ent or {})
        if ovr is not None:
            unknown = set(ovr) - {"tile_m", "tile_k", "tile_n", "cols"}
            if unknown:
                raise ValueError(f"GEMM_TILES_OVERRIDE[{key if key in self.overrides else label}]: "
                                 f"unknown field(s) {sorted(unknown)}; only tile_m/tile_k/tile_n/"
                                 f"cols can be overridden")
            missing = {"tile_m", "tile_k", "tile_n", "cols"} - set(merged) - set(ovr)
            if missing:
                raise ValueError(f"GEMM_TILES_OVERRIDE for an UNSWEPT shape must give the whole "
                                 f"tiling; {sorted(missing)} missing for {key}")
            merged.update(ovr)
            merged["source"] = "override"
            merged["measured"] = None

        choice = TileChoice(tile_m=merged["tile_m"], tile_k=merged["tile_k"],
                            tile_n=merged["tile_n"], cols=merged["cols"],
                            source=merged.get("source", "seed"),
                            b_col_maj=merged.get("b_col_maj"),
                            measured=merged.get("measured"), label=label)

        rej = gemm_tiling_rejection(M, K, N, choice.tile_m, choice.tile_k, choice.tile_n,
                                    choice.cols, bfp16=emulate, prio_accuracy=prio_accuracy)
        if rej is not None:
            raise IllegalRegistryEntry(
                f"{key}{'' if not label else ' (' + label + ')'}: the recorded tiling "
                f"{choice} is not legal for M={M} K={K} N={N} -- {rej.detail}. A registry entry "
                f"is checked against the same rules the sweep filters on, so this is a bad "
                f"hand-edit or a shape that moved under a stale entry.")
        if b_col_maj is not None and choice.b_col_maj is not None \
                and bool(b_col_maj) != bool(choice.b_col_maj):
            raise IllegalRegistryEntry(
                f"{key}{'' if not label else ' (' + label + ')'}: recorded with "
                f"b_col_maj={choice.b_col_maj}, asked for b_col_maj={b_col_maj}. B's "
                f"dims_to_stream differs between the two (gemm/design.py), so the measurement "
                f"does not carry across; sweep this shape in the orientation you build.")
        return choice

    def _miss_message(self, key, M, K, N, dtype, emulate, prio_accuracy, label) -> str:
        where = f" for {label}" if label else ""
        return (
            f"no GEMM tile entry{where}: M={M} K={K} N={N} dtype={dtype} "
            f"emulate={emulate} prio_accuracy={prio_accuracy} (key {key}).\n"
            f"Nothing is guessed here on purpose. Either sweep it and record the winner:\n"
            f"    bash scripts/sweep_gemm_tiles.sh --shape {M}x{K}x{N}"
            f"{'' if not label else ' --label ' + label}"
            f"{'' if emulate else ' --no-emulate'}"
            f"{' --prio-accuracy' if prio_accuracy else ''}\n"
            f"    bash scripts/time_gemm_tiles.sh <sweep-dir>/manifest.json    # on the device\n"
            f"or, if you deliberately want an UNMEASURED entry, say so explicitly:\n"
            f"    designs/decode_fused/sweep_gemm_tiles.py --seed {M}x{K}x{N}"
            f" --tile 64,64,64 --cols 8 --source assumed\n"
            f"Registry: {self.path or DEFAULT_REGISTRY}")

    # ---- writing ----
    def record(self, M: int, K: int, N: int, tile_m: int, tile_k: int, tile_n: int, cols: int, *,
               dtype: str = "bf16>bf16", emulate: bool = True, prio_accuracy: bool = False,
               source: str = "sweep", b_col_maj: bool | None = None, measured: dict | None = None,
               labels=None, candidates=None) -> str:
        if source not in SOURCES:
            raise ValueError(f"source={source!r} not one of {SOURCES}")
        rej = gemm_tiling_rejection(M, K, N, tile_m, tile_k, tile_n, cols, bfp16=emulate,
                                    prio_accuracy=prio_accuracy)
        if rej is not None:
            raise IllegalRegistryEntry(f"refusing to record an illegal tiling: {rej.detail}")
        key = key_of(M, K, N, dtype=dtype, emulate=emulate, prio_accuracy=prio_accuracy)
        self.entries[key] = {
            "M": M, "K": K, "N": N, "dtype": dtype,
            "emulate": bool(emulate), "prio_accuracy": bool(prio_accuracy),
            "tile_m": tile_m, "tile_k": tile_k, "tile_n": tile_n, "cols": cols,
            "b_col_maj": None if b_col_maj is None else bool(b_col_maj),
            "source": source,
            "labels": sorted(set(labels or [])),
            "measured": measured,
            "candidates_considered": candidates,
        }
        return key

    def status(self) -> dict:
        """How much of the registry is actually measured -- one line for a build to print."""
        by = {}
        for e in self.entries.values():
            by[e.get("source", "seed")] = by.get(e.get("source", "seed"), 0) + 1
        return by


_CACHED: Registry | None = None


def registry(path=None) -> Registry:
    """Process-wide registry, so a 28-layer build reads the JSON once."""
    global _CACHED
    if _CACHED is None or path is not None:
        _CACHED = Registry.load(path)
    return _CACHED


def tiles_for(M: int, K: int, N: int, **kw) -> TileChoice:
    """Convenience wrapper over `registry().lookup()`."""
    return registry().lookup(M, K, N, **kw)


def as_dict(choice: TileChoice) -> dict:
    return asdict(choice)
