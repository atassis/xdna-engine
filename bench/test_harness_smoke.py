"""CPU-only rot guard for the benchmark harnesses. No NPU, no models, no network, no build.

`bench/compare.py` -- the harness that produced a shipped head-to-head result -- silently stopped
running for weeks and was only caught by accident, because nothing exercises these modules except a
human doing a comparison. Four independent rots had accumulated: it spawned a binary deleted in the CLI
migration, started a version-incompatible service, constructed an energy meter that threw before taking
a measurement, and stopped a renamed systemd unit so it never freed the device it claimed to free.

Most of that class is statically checkable in seconds. The rules below are deliberately chosen to be
checkable IN-REPO, with nothing built and nothing installed:

  * a fresh clone must pass -- a guard that false-FAILs on a clean checkout is worse than the rot it
    guards, because it trains you to ignore it;
  * so we assert things the repo itself determines (imports, pure functions, parser behaviour, and
    binary names cross-checked against Cargo's own `[[bin]]` declarations), NOT things only a
    provisioned machine determines (is the binary built, is the unit installed, is FLM the right
    version). Those stay a manual/device concern.

The device-dependent half -- does a backend actually serve -- cannot be smoke-tested cheaply and is out
of scope. But a harness that cannot resolve its own entry points should never reach a device session.
"""

import importlib
import json
import re
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"

# Every non-test module in bench/, discovered rather than listed: a NEW harness module is covered the
# moment it lands, which is the failure mode being guarded (nobody remembers to add it here).
BENCH_MODULES = sorted(
    p.stem for p in BENCH.glob("*.py")
    if not p.stem.startswith("test_") and p.stem != "__init__"
)


def test_bench_modules_were_discovered():
    # Guards the guard: if the glob silently matched nothing, every parametrized test below would
    # vacuously "pass" and this file would be worthless.
    assert len(BENCH_MODULES) >= 4, BENCH_MODULES


def _is_first_party(module_name):
    """Is a failed import OUR code moving, or a third-party package simply not installed?

    The distinction is the whole point of this guard. `bench/bench.py` needs `onnx_asr`, an optional
    pypi dep -- absent on a fresh clone and absent in CI, and its absence says nothing about rot. But
    if `bench.backends` stops resolving, that IS rot and must fail loudly. So: a module is first-party
    when its top-level name corresponds to a file or package inside the repo.
    """
    top = module_name.split(".")[0]
    return (ROOT / f"{top}.py").exists() or (ROOT / top).is_dir()


@pytest.mark.parametrize("mod", BENCH_MODULES)
def test_module_imports(mod):
    """Import-time rot: a moved first-party dependency or a syntax error anywhere in bench/.

    Deliberately does NOT fail on a missing optional third-party package -- a guard that goes red for
    an unrelated reason gets ignored, and an ignored guard is how the original rot survived for weeks.
    """
    try:
        importlib.import_module(f"bench.{mod}")
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if _is_first_party(missing):
            raise AssertionError(
                f"bench.{mod} cannot import first-party module {missing!r} -- this is rot, not a "
                f"missing optional dependency."
            ) from e
        pytest.skip(f"bench.{mod} needs optional third-party {missing!r} (not installed)")


# --- pure functions on fixtures ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,want",
    [
        ("Hello, World!", "hello world"),
        ("  multiple   spaces\t\n", "multiple spaces"),
        ("Punctuation... removed?!", "punctuation removed"),
        ("", ""),
        (None, ""),                      # normalize() is documented to tolerate None
        ("MiXeD CaSe", "mixed case"),
    ],
)
def test_normalize(raw, want):
    from bench.compare import normalize
    assert normalize(raw) == want


@pytest.mark.parametrize(
    "ref,hyp,rate,nwords",
    [
        ("a b c d", "a b c d", 0.0, 4),          # identical
        ("a b c d", "a b c x", 0.25, 4),         # one substitution
        ("a b c d", "a b c", 0.25, 4),           # one deletion
        ("a b c", "a b c d", 1 / 3, 3),          # one insertion
        ("", "", 0.0, 0),                        # empty ref, empty hyp -> 0.0 by contract
        ("", "spurious", 1.0, 0),                # empty ref, non-empty hyp -> 1.0 by contract
    ],
)
def test_wer(ref, hyp, rate, nwords):
    from bench.compare import wer
    got_rate, got_n = wer(ref, hyp)
    assert got_rate == pytest.approx(rate)
    assert got_n == nwords


def test_wer_is_symmetric_in_cost_not_in_rate():
    # Edit DISTANCE is symmetric; the RATE is normalized by the reference length, so swapping the
    # arguments must change the rate. Pins the normalization, which is the easy thing to get wrong.
    from bench.compare import wer
    fwd, _ = wer("a b c", "a b c d")
    rev, _ = wer("a b c d", "a b c")
    assert fwd != rev


def test_ols_recovers_a_known_line():
    from bench.llm_decode_sweep import ols
    xs = [0, 1, 2, 3, 4]
    ys = [3 + 2 * x for x in xs]          # y = 3 + 2x, exact
    fit = ols(xs, ys)
    assert fit["intercept"] == pytest.approx(3.0)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n"] == 5


def test_ols_flat_series_is_r2_one_not_nan():
    # ss_tot == 0 would be a divide-by-zero; the implementation special-cases it. A NaN here would
    # silently poison any downstream bandwidth-bound fit.
    from bench.llm_decode_sweep import ols
    fit = ols([0, 1, 2], [5, 5, 5])
    assert fit["slope"] == pytest.approx(0.0)
    assert fit["r2"] == 1.0


# --- streaming-chunk parser (recorded-SSE fixture) --------------------------------------------


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


def _sse(*deltas):
    """Build SSE wire lines the way an OpenAI-compatible server emits them."""
    out = []
    for d in deltas:
        out.append(b"data: " + json.dumps({"choices": [{"delta": d}]}).encode())
    out.append(b"data: [DONE]")
    return out


def _run_stream(monkeypatch, lines):
    from bench import llm_decode_sweep
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeStream(lines))
    return llm_decode_sweep.stream(1234, "m", "p", 8)


def test_stream_counts_content_tokens(monkeypatch):
    ttft, itl, n = _run_stream(monkeypatch, _sse(
        {"role": "assistant"},                 # role-only preamble: NOT a token
        {"content": "a"}, {"content": "b"}, {"content": "c"},
    ))
    assert n == 3
    assert len(itl) == 2                       # n-1 inter-token gaps
    assert ttft >= 0


def test_stream_counts_reasoning_content(monkeypatch):
    """The rot that made thinking models report zero tokens.

    qwen3 thinking models stream chain-of-thought as `reasoning_content`, not `content`. That is the
    same decode cost per token, so it must count -- otherwise those models look like they emit nothing
    and the whole sweep silently reports zero.
    """
    _, _, n = _run_stream(monkeypatch, _sse(
        {"role": "assistant"},
        {"reasoning_content": "think"}, {"reasoning_content": "more"}, {"content": "answer"},
    ))
    assert n == 3


def test_stream_ignores_role_only_preamble(monkeypatch):
    _, _, n = _run_stream(monkeypatch, _sse({"role": "assistant"}, {"content": "x"}))
    assert n == 1


def test_stream_raises_when_no_tokens(monkeypatch):
    # Failing loud beats reporting a zero-token result as if it were a measurement.
    with pytest.raises(RuntimeError):
        _run_stream(monkeypatch, _sse({"role": "assistant"}))


def test_stream_survives_malformed_chunk(monkeypatch):
    from bench import llm_decode_sweep
    lines = [b"data: {not json", b"data: " + json.dumps(
        {"choices": [{"delta": {"content": "ok"}}]}).encode(), b"data: [DONE]"]
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeStream(lines))
    _, _, n = llm_decode_sweep.stream(1234, "m", "p", 8)
    assert n == 1


# --- entry points: the rot that actually shipped -----------------------------------------------


def _declared_cargo_bins():
    """Every `[[bin]] name = "..."` across the rust workspace."""
    names = set()
    for toml in ROOT.glob("rust/*/Cargo.toml"):
        text = toml.read_text()
        for block in text.split("[[bin]]")[1:]:
            m = re.search(r'^\s*name\s*=\s*"([^"]+)"', block, re.M)
            if m:
                names.add(m.group(1))
    return names


def _spawned_binary_names():
    """Binary names the bench harnesses execute, scraped from their source.

    Covers the three shapes the harnesses actually use:
      ROOT / "rust" / "target" / "release" / "npu"   (Path join)
      "release/npu serve"                            (pkill -f pattern)
      "~/.local/bin/npu"                             (installed fallback)
    """
    pats = [
        re.compile(r'"release"\s*/\s*"([A-Za-z0-9_-]+)"'),
        re.compile(r'release/([A-Za-z0-9_-]+)'),
        re.compile(r'~/\.local/bin/([A-Za-z0-9_-]+)'),
    ]
    found = set()
    for py in BENCH.glob("*.py"):
        if py.stem.startswith("test_"):
            continue
        src = py.read_text()
        for p in pats:
            found.update(p.findall(src))
    return found


def test_scraper_finds_the_entry_point():
    # Guards the guard again: if the scrape regexes stop matching (someone rewrites how the path is
    # built), the cross-check below would pass vacuously and the original rot could return unnoticed.
    assert _spawned_binary_names(), "entry-point scrape matched nothing -- update the patterns"


def test_spawned_binaries_are_declared_cargo_bins():
    """THE rot that shipped: compare.py spawned `rust/target/release/engine_serve`, a binary deleted
    in the CLI migration (the entry point is now `npu serve --config`).

    Cross-checking against Cargo's own `[[bin]]` declarations catches a rename statically -- no cargo
    build, no installed binary, so this holds on a fresh clone. `engine_serve` is not a declared bin,
    so this test would have failed the day the migration landed.
    """
    declared = _declared_cargo_bins()
    assert declared, "no [[bin]] declarations found -- workspace layout changed?"
    spawned = _spawned_binary_names()
    unknown = sorted(spawned - declared)
    assert not unknown, (
        f"bench harness spawns binaries with no [[bin]] declaration: {unknown}. "
        f"Declared: {sorted(declared)}"
    )


def test_default_scenario_config_exists():
    """compare.py's default `--scenario` must resolve, or a plain `python -m bench.compare` cannot
    start regardless of what is built."""
    import bench.compare as c
    src = Path(c.__file__).read_text()
    m = re.search(r'"--scenario",\s*default="([^"]+)"', src)
    assert m, "could not find the --scenario default -- did the CLI change?"
    assert (ROOT / m.group(1)).exists(), f"default scenario missing: {m.group(1)}"


def test_corpus_manifest_exists():
    assert (BENCH / "corpus.toml").exists()
