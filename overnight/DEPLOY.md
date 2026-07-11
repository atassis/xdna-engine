# DEPLOY.md -- from-scratch runbook for the xdna2 NPU ASR engine

Canonical, ordered, copy-pasteable runbook to take a CLEAN clone of this repo on a
fresh CachyOS/Arch box to a working NPU ASR service (Parakeet-TDT-0.6b-v3), serving
character-identical to the shipped baseline.

Every step is tagged `[CPU]` (no NPU needed -- builds, exports, kernel compiles) or
`[DEVICE]` (touches the single-tenant NPU -- install + the character-identity gate).
"Produces", "success signal", and rough "time" are listed per step so you can tell a
step worked before moving on.

READ THIS FIRST: the chain is now nearly hands-off. One step (0.b) still needs a manual
command that no script performs today, and two dependency-fetch steps (2, 3) lean on
assets that can rotate out of upstream indexes. GAP #1 (models/parakeet/), GAP #3 (silent
Peano miss) and GAP #5 (WER clips) are CLOSED -- see the GAP LIST and the Recreatability
debts table at the end BEFORE you start.

---

## What this reproduces

The engine runs the entire Parakeet FastConformer encoder (24 blocks, d_model 1024) on
ONE resident whole_array matmul xclbin on the AMD XDNA2 NPU, with the TDT decoder_joint
and the mel preprocessor as small ONNX graphs on CPU. The Rust `npu serve` binary exposes
an OpenAI-style `/v1/audio/transcriptions` endpoint on :11434. GigaAM-v3 (RU) is an
optional second ASR scenario.

The reproduction is decoupled from any build tree: the AIE toolchain is built into a
content-addressed instance under `~/.cache/xdna2-build/`, and the model weight arena +
xclbins are regenerated from pinned upstream sources.

---

## Prerequisites

### Hardware / OS
- **AMD Ryzen AI (XDNA2 / NPU2)** laptop APU with the NPU enabled in firmware.
- **CachyOS / Arch Linux** (the toolchain scripts assume Arch conventions:
  gcc-13/g++-13 shims onto system gcc, `/usr/include` + `/usr/lib` for XRT).
- **XRT + amdxdna driver** installed at the system level, with the **python binding
  `pyxrt` built for the venv's python (3.14)** visible in system site-packages
  (`/usr/lib/python3.14/site-packages/pyxrt.*.so`). `.venv-iron` is created with
  `--system-site-packages` specifically to see this. Verify:
  `python3 -c "import pyxrt"` must succeed. If it does not, install/rebuild XRT's
  python bindings for py3.14 -- nothing in this repo provisions XRT itself.

### Toolchain binaries on PATH (all present on the reference box)
`uv`, `gh` (authenticated: `gh auth status`), `cmake`, `ninja`, `cargo`/rustc,
`gcc`/`g++`, `python3` (3.14 + 3.12 available to `uv`), `ffmpeg` + `ffprobe`, `curl`,
`git`. Optional but strongly recommended for build speed: `ccache`, `lld`
(auto-used by `fast_build_env.sh`; no-ops if absent).

### Network
Required for the fetch steps: GitHub (fork remote + `gh release download`),
Hugging Face hub (model repos + FLEURS clips), PyPI + the PyTorch CPU index.
Once caches are warm the run can go offline (see the offline notes per step).

### Disk
Budget ~15-20 GB: ~1.8 GB toolchain wheels, ~2.5 GB Parakeet fp32 encoder + `.data`,
~1 GB GigaAM (optional), the MLIR distro, the built toolchain instance, and Rust target.

---

## Step 0 -- Clone + place the pinned mlir-aie fork commit  [CPU]

`mlir-aie` is a git submodule (`.gitmodules` url = Xilinx/mlir-aie, `ignore=all`) but the
build needs it on the **fork integration branch** `atassis/mlir-aie:xdna2-asr`, pinned by
`toolchain.lock:MLIR_AIE_FORK_COMMIT`. A plain `git submodule update --init` would fetch
the WRONG (upstream default) branch. `setup_route_b.sh` (Step 1) has the correct
fetch-by-SHA + fork-branch checkout logic, so you normally do NOT need to touch the
submodule by hand -- but you DO need a `mlir-aie/.git` to exist first.

### 0.a  Clone
```bash
git clone <this-repo-url> xdna-engine
cd xdna-engine
git checkout chore/adopt-upstream-softmax-kwargs   # the validated branch
```

### 0.b  Bootstrap the submodule checkout (MANUAL -- see GAP #1)
`setup_route_b.sh` only bootstraps a checkout when `mlir-aie/.git` is ABSENT, via
`git submodule update --init` (which needs the submodule URL to serve the pinned fork
commit -- it does not, the commit lives on the fork). The robust from-scratch sequence,
which lands the exact pinned commit regardless, is:
```bash
git submodule update --init --depth 1 mlir-aie || mkdir -p mlir-aie   # get an empty/base checkout
git -C mlir-aie remote add fork https://github.com/atassis/mlir-aie 2>/dev/null || true
# read the pin from toolchain.lock:
FORK_SHA=$(sed -n 's/^MLIR_AIE_FORK_COMMIT=\([0-9a-f]*\).*/\1/p' toolchain.lock)
git -C mlir-aie fetch fork xdna2-asr
git -C mlir-aie checkout -B xdna2-asr "$FORK_SHA"
```
- **Produces:** `mlir-aie/` checked out on `xdna2-asr` at the pinned commit.
- **Success signal:** `git -C mlir-aie rev-parse HEAD` equals `MLIR_AIE_FORK_COMMIT`.
- **Time:** 2-5 min (fork fetch).
- **Note:** Step 1 re-runs the remote-add + fetch-by-SHA + checkout idempotently, so if
  0.a produced a `mlir-aie/.git` at all, you can let Step 1 do the fork checkout. The
  manual sequence here is the safe belt-and-suspenders when 0.a's submodule fetch cannot
  serve the fork commit.

---

## Step 1 -- Route B toolchain env: `.venv-iron` + AIE wheels + fork checkout  [CPU]

```bash
scripts/setup_route_b.sh
```
- **Produces:** `.venv-iron` (py3.14, `--system-site-packages`), the `mlir_aie` +
  `nanobind` wheels installed, Peano (`llvm-aie`) binaries copied into site-packages,
  gcc-13/g++-13 shims, the `mlir-aie` fork-branch checkout re-ensured, and the route_b
  kernels synced into the mlir-aie sandbox.
- **Success signal:** prints `mlir-aie on xdna2-asr @ <sha>` and
  `Route B env ready.`; `.venv-iron/bin/python -c "import aie"` works and
  `.venv-iron/lib/python3.14/site-packages/llvm-aie/bin/clang` exists.
- **Time:** 3-10 min warm cache; longer if wheels must be fetched.
- **Wheelhouse prerequisite (see GAP #2):** the pinned `mlir_aie==0.0.1.2026033104+e4f35d6`
  wheel is not reliably fetchable from the network. The script's resolution order is:
  (1) offline install from the **uv archive cache**; (2) `vendor/wheelhouse/` (gitignored;
  auto-rebuilt by `scripts/build_wheelhouse.sh` by repacking the wheel FROM the uv cache);
  (3) network find-links `latest-wheels-4`. On a truly cold machine where the uv cache is
  empty AND `vendor/wheelhouse/` is absent, tiers (1) and (2) both fail and you are relying
  on tier (3) -- see GAP #2 for how to pre-warm.
- **Peano prerequisite (see GAP #3, CLOSED):** `llvm-aie` (Peano) ships only a cp310 wheel
  that will NOT install into the py3.14 venv, so the script provides it by copying the
  unpacked tree straight out of `~/.cache/uv/archive-v0`. The tree-copy still only WARNS on
  a cache miss, but the script now ends with a TERMINAL GUARD: if
  `.venv-iron/lib/python3.14/site-packages/llvm-aie/bin/clang` is still absent (or if the
  `mlir_aie` wheel left `import aie` broken), it prints exactly what is missing + the
  pre-warm command and `exit 1`s -- so the gap fails LOUD here, not confusingly at Step 3/4.
  The warm-cache path is unaffected (both checks pass, guard is a no-op). Pre-warm per GAP #3.

---

## Step 2 -- Fetch the pinned MLIR core distro  [CPU, network]

```bash
scripts/fetch_mlir_distro.sh
```
- **Produces:** the prebuilt LLVM/MLIR framework wheel (`MLIR_DISTRO_WHEEL` in
  `toolchain.lock`) unpacked into
  `~/.cache/xdna2-build/mlir-distro/<ver>/mlir` (content-addressed).
- **Success signal:** prints the resolved `.../mlir` dir; `.../mlir/bin/mlir-tblgen` exists.
- **Time:** 2-5 min (downloads via `gh release download` from Xilinx/mlir-aie mlir-distro).
- **Needs:** authenticated `gh`. (See GAP #4: this pin is a dated release asset.)

---

## Step 3 -- Build the aiecc fork toolchain instance  [CPU]

```bash
scripts/toolchain_up.sh    # prints the instance dir on stdout
```
- **Produces:** a content-addressed toolchain instance under
  `~/.cache/xdna2-build/instances/<lockhash>/` -- fork `aiecc` + `aie-opt` + IRON python
  bindings built from a clean git-worktree of the pinned fork commit, with the vendored
  tools + `aie_api`/`aie_kernels` include symlinks wired in.
- **Success signal:** prints the instance path; on a warm re-run it early-returns the same
  path (self-healing the symlinks).
- **Time:** COLD build tens of minutes (LLVM/MLIR-scale; ccache + lld cut re-builds to
  minutes). Do NOT interrupt.
- **Depends on:** Steps 1 + 2 (venv python, Peano, MLIR distro).

---

## Step 4 -- CPU smoke gate  [CPU]

```bash
scripts/toolchain_smoke.sh
```
- **Produces:** a throwaway whole_array xclbin built end-to-end through the fork toolchain
  (restored/non-destructive to any existing device-validated xclbin).
- **Success signal:** prints `SMOKE PASS (CPU): logical_tile=... -> placed aie.tile=... , xclbin built`.
- **Time:** 1-3 min.
- **This is the gate:** if this fails, do NOT proceed to kernel builds -- the toolchain
  (placement model / bindings / Peano) is broken.

---

## Step 5 -- Model-export venv  [CPU, network]

```bash
scripts/setup_export_venv.sh
```
- **Produces:** `.venv-export` (py3.12) with CPU-only torch + onnx + onnxruntime + onnx-asr
  (`scripts/requirements-export.txt`, `--index-strategy unsafe-best-match`).
- **Success signal:** prints `Export venv ready at .venv-export.`;
  `.venv-export/bin/python -c "import onnx_asr, torch, onnxruntime"` works.
- **Time:** 3-8 min.
- **Why separate from `.venv-iron`:** export deps must not pollute the py3.14 toolchain env,
  and `onnx-asr` bundles the mel preprocessor (`nemo128.onnx`) that Step 7 copies.

---

## Step 6 -- Fetch model sources + assemble serving artifacts  [CPU, network]

```bash
scripts/fetch_models.sh                 # set FETCH_EXTRA=0 to skip the 11 non-ASR repos
```
- **Produces:** the HF hub cache populated with `istupakov/parakeet-tdt-0.6b-v3-onnx` +
  `istupakov/gigaam-v3-onnx` (and, unless `FETCH_EXTRA=0`, 11 extra repos for other-arch
  parity tasks -- NOT needed for ASR serving), the serve-ready
  `artifacts/parakeet/{preprocessor.onnx (nemo128 128-mel), decoder_joint.onnx, vocab.txt}`,
  and (GAP #1 fix) the fp32 encoder source `models/parakeet/{encoder-model.onnx,
  encoder-model.onnx.data, decoder_joint-model.onnx, vocab.txt}` that Step 7 consumes,
  `cp -L`-dereferenced from the HF snapshot's symlink farm under their ORIGINAL upstream names.
- **Success signal:** prints `[parakeet] assembled .../artifacts/parakeet : ...`,
  `[parakeet-models] materialized .../models/parakeet : ...`, and `Done.`
- **Time:** 5-20 min depending on `FETCH_EXTRA` (the parakeet repo alone is ~2.5 GB); the
  `models/parakeet/` materialize adds a local ~2.4 GB `cp` from the cache.
- **Does NOT produce:** `artifacts/parakeet/encoder/` (the weight arena -- Step 7). The HF
  cache is near-empty on a clean box, so this is real downloads (set `HF_HUB_OFFLINE=1` only
  if already cached).

---

## Step 7 -- Build the Parakeet encoder weight arena  [CPU]

### 7.b  Materialize `models/parakeet/encoder-model.onnx` (AUTOMATED -- GAP #1 CLOSED)
`extract_parakeet_encoder.py` reads `models/parakeet/encoder-model.onnx` (+ its external
`.onnx.data`). This directory is now produced by `prep_parakeet_models()` in Step 6's
`fetch_models.sh` (it `cp -L`-derefs `encoder-model.onnx`, `encoder-model.onnx.data`,
`decoder_joint-model.onnx`, `vocab.txt` from the HF snapshot into `models/parakeet/` under
their original names; the `.onnx` references the `.data` by relative filename, so both land
side by side). No manual step is required. If you skipped Step 6 or the snapshot was not
cached, re-run `scripts/fetch_models.sh` -- it is idempotent.
- **Produces:** `models/parakeet/encoder-model.onnx` (+ `.onnx.data`, ~2.4 GB).
- **Success signal:** both files present; total ~2.5 GB (Step 6 prints the `materialized` line).

### 7.a  Extract the arena
```bash
.venv-export/bin/python scripts/extract_parakeet_encoder.py
```
- **Produces:** `artifacts/parakeet/encoder/` -- per-block `.npy` weights (24 blocks),
  `pre_encode/`, `refs/`, and `manifest.json`.
- **Success signal:** prints `OK blocks=24 weights/block=... pre_encode=...` and the ref
  shapes line.
- **Time:** 2-5 min (loads a >2 GB ONNX + one seeded CPU forward pass).

---

## Step 8 -- WER / character-identity clips  [NO ACTION -- from VCS]

GAP #5 was a FALSE ALARM: the clips the install test (Step 11) and the 4-clip A/B gate
(Step 12) read are COMMITTED to git, so a clean clone already has them -- there is no fetch
step and no network needed here.
```bash
git ls-files artifacts/wer_clips/   # 24 tracked files, verify present in your clone
```
- **Already present from the clone:** `artifacts/wer_clips/{en_01..en_04,ru_01..ru_13}.wav`
  (17 wavs) + `refs.json` + `SOURCE.md` + the baseline JSONs
  (`parakeet_eval_results.json`, `int8_eval_results.json`, `whisper_*`, `*_oracle.json`).
- **Note:** `scripts/fetch_wer_clips.py` still exists to REGENERATE the clips from FLEURS
  (CC-BY-4.0, 16 kHz mono via ffmpeg) if you ever need to refresh them, but the deploy chain
  does not need to run it -- the tracked copies are the shipped-baseline reference.

---

## Step 9 -- Build the resident Parakeet encoder xclbins  [CPU]

```bash
scripts/build_parakeet_kernels.sh       # AIECC_JOBS=<n> to parallelize; NPU_NATIVE path built too
```
- **Produces, under
  `mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/`:**
  `final_512x1024x{1024,2048,4096}_64x32x128_8c.xclbin` (FAST BFP16, the default resident
  kernel) + matching `insts_*.txt`, and the same for the NATIVE `32x32x32` tile
  (`NPU_NATIVE=1` path). The engine (`rust/npu-parakeet/src/npu.rs`) loads these from this
  exact relative dir with `WorkingDirectory=$REPO`.
- **Success signal:** prints `Built Parakeet resident xclbins ...`; the six `final_*.xclbin`
  + `insts_*.txt` exist.
- **Time:** 10-40 min (six aiecc builds; warm ccache much faster).
- **Depends on:** Step 4 green (sources `iron_env.sh`, re-syncs kernels).

### 9.opt  Decode GEMV kernels (OPTIONAL -- not on the Parakeet serve path)
`scripts/build_decode_kernels.sh` builds a thin-M GEMV xclbin used by decode probes/spikes,
NOT by the Parakeet ASR serve (the TDT `decoder_joint` runs as ONNX on CPU). Skip for a
plain ASR deploy; build only if you are exercising the on-NPU decode experiments.

---

## Step 10 (OPTIONAL) -- GigaAM-v3 RU encoder  [CPU]

Only if you want the GigaAM (RU) scenario (`scenarios/asr-gigaam.toml`). Needs
`models/gigaam_v3_encoder_static.onnx` first:
```bash
.venv-export/bin/python scripts/export_gigaam_encoder.py     # -> models/gigaam_v3_encoder_static.onnx
.venv-export/bin/python scripts/quantize_encoder_static.py   # -> models/quant/gigaam_v3_encoder_int8_static.onnx
```
- **Success signal:** each prints its output path + size.
- **Time:** 5-15 min. **Depends on:** Step 6 (gigaam HF repo) + Step 8 (RU calib clips).

---

## Step 11 -- Install + start the service  [DEVICE]

Attended -- takes the single-tenant NPU. Stop any other NPU user first
(`fuser` the device / stop `flm-asr.service`).
```bash
scripts/install.sh
```
- **Produces:** the `npu` + `npu-weights` release binaries in `~/.local/bin`, the
  onnxruntime lib in `~/.local/lib/npu-asr`, `~/.config/npu/engine.toml`, and the
  `npu-serve.service` systemd --user unit (WorkingDirectory=$REPO), started on :11434.
- **Success signal:** prints `npu-serve active on :11434`.
- **Time:** 5-15 min (cold Rust release build) + seconds to start.

---

## Step 12 -- Verify + character-identity A/B gate  [DEVICE]

```bash
scripts/test_install.sh                                 # single-clip smoke through the service
```
- **Success signal:** `PASS - transcription returned: "..."`.

Then the 4-clip character-identity gate (this session's device gate) -- transcribe
`en_01, en_02, en_03, ru_02` through the service and confirm output is character-identical
to the shipped baseline:
```bash
for c in en_01 en_02 en_03 ru_02; do
  echo "== $c =="
  curl -fsS -F "file=@artifacts/wer_clips/$c.wav" \
    http://127.0.0.1:11434/v1/audio/transcriptions
  echo
done
```
- **Success signal:** each clip returns non-empty `text`; the four transcripts match the
  shipped-baseline reference character-for-character.
- **Time:** < 1 min.

---

## Recreatability debts table

Every gitignored runtime dependency, its producing step, and whether that step is fully
automated from a clean clone.

| Gitignored dep | Produced by | Automated? |
|---|---|---|
| `.venv-iron` (py3.14 AIE venv) | Step 1 `setup_route_b.sh` | AUTO -- the `mlir_aie` wheel + Peano tree still lean on a warm uv cache / wheelhouse / rotating network index (GAP #2 OPEN), but a miss now HARD-FAILS loudly with the pre-warm command (GAP #3 CLOSED) instead of silently absenting Peano. |
| `mlir-aie/` submodule on fork branch | Step 0.b (manual) / re-ensured by Step 1 | PARTIAL -- Step 1 does fork remote-add + fetch-by-SHA + checkout, but only if a `mlir-aie/.git` already exists; the initial bootstrap from a bare clone needs the manual 0.b (GAP #1). |
| `vendor/wheelhouse/mlir_aie-*.whl` | Step 1 -> `build_wheelhouse.sh` | PARTIAL -- rebuildable ONLY from a warm uv archive cache; empty cache => hard error, falls back to a rotating network index (GAP #2). |
| MLIR core distro (`~/.cache/xdna2-build/mlir-distro/...`) | Step 2 `fetch_mlir_distro.sh` | AUTO -- needs authenticated `gh`; pin is a dated release asset (GAP #4). |
| Toolchain instance (`~/.cache/xdna2-build/instances/<hash>`) | Step 3 `toolchain_up.sh` | AUTO (cold = tens of min). |
| `.venv-export` (py3.12 export venv) | Step 5 `setup_export_venv.sh` | AUTO (network: PyPI + torch-cpu index). |
| HF model cache (parakeet, gigaam) | Step 6 `fetch_models.sh` | AUTO (network: HF hub). |
| `artifacts/parakeet/{preprocessor,decoder_joint,vocab}` | Step 6 `fetch_models.sh` (`prep_parakeet_artifacts`) | AUTO. |
| `models/parakeet/encoder-model.onnx(.data)` | Step 6 `fetch_models.sh` (`prep_parakeet_models`) | AUTO -- `cp -L`-derefs from the HF snapshot under original names (GAP #1 CLOSED). |
| `artifacts/parakeet/encoder/` (weight arena) | Step 7.a `extract_parakeet_encoder.py` | AUTO -- input now auto-produced by Step 6. |
| `artifacts/wer_clips/*.wav` + `refs.json` + baseline JSONs | TRACKED IN VCS (committed) | AUTO -- present from the clone, no fetch step (GAP #5 was a false alarm; `fetch_wer_clips.py` only regenerates). |
| `models/gigaam_v3_encoder_static.onnx` + `models/quant/*` | Step 10 (optional) | AUTO (optional scenario). |
| Parakeet resident xclbins + insts (under `mlir-aie/.../whole_array/build/`) | Step 9 `build_parakeet_kernels.sh` | AUTO (needs Step 4 green). |
| `rust/target/` release binaries + onnxruntime lib | Step 11 `install.sh` | AUTO (fetches onnxruntime during build). |

---

## GAP LIST (ordered by severity)

### GAP #1 -- `models/parakeet/encoder-model.onnx` has NO automated producer  [HIGH -- CLOSED]
- **Was:** Step 7 `extract_parakeet_encoder.py` hard-reads
  `models/parakeet/encoder-model.onnx` (+ external `.onnx.data`), but `fetch_models.sh`
  populated only the HF cache and `artifacts/parakeet/{preprocessor,decoder_joint,vocab}`
  -- it never created `models/parakeet/`. On a clean clone Step 7 failed with a missing-file
  error, so the encoder arena (the core NPU weights) was never built.
- **Fix (DONE):** added `prep_parakeet_models()` to `fetch_models.sh`, called right after
  `prep_parakeet_artifacts`. It `cp -L`-derefs `encoder-model.onnx`, `encoder-model.onnx.data`,
  `decoder_joint-model.onnx`, `vocab.txt` from the HF snapshot into `models/parakeet/` under
  their original upstream names (so the `.onnx`'s relative `.data` reference resolves).
  Idempotent (`cp -Lf` overwrites in place). Verified: the dereferenced file sizes match the
  on-box `models/parakeet/` byte-for-byte (encoder 41770866, decoder_joint 72520893, vocab
  93939) and the filenames are exactly what `extract_parakeet_encoder.py` reads.
- **Still note (0.b):** the initial bare-clone submodule bootstrap remains manual -- `git
  submodule update --init` targets the Xilinx URL, which does not carry the pinned fork
  commit; only the fork remote does. Step 1's explicit fetch-by-SHA handles it once a
  `mlir-aie/.git` exists, but the first bootstrap still needs the manual 0.b sequence.

### GAP #2 -- pinned `mlir_aie` wheel depends on a warm uv cache OR a rotating index  [HIGH -- OPEN, owned by `proper-install-consumable`]
- **Owner note:** durably hosting the ~290 MB `mlir_aie` wheel (release asset / LFS / object
  store) is the `proper-install-consumable` task's job. On THIS machine the warm uv cache +
  content-addressed toolchain cache make the clone build work; a truly-bare machine needs the
  wheel vendored. GAP #3's guard now makes an absence fail loud instead of silent.
- **What breaks:** Step 1 installs `mlir_aie==0.0.1.2026033104+e4f35d6`. Its resolution
  order is (1) offline uv archive cache, (2) `vendor/wheelhouse/` (gitignored; rebuilt by
  `build_wheelhouse.sh` -- which itself ONLY repacks from the uv archive cache and hard-errors
  on an empty cache), (3) network find-links `latest-wheels-4`. On a brand-new machine the
  uv cache is empty and `vendor/` is gitignored/absent, so tiers 1 and 2 are dead and you
  depend entirely on tier 3 -- but nightly `latest-wheels-4` assets rotate out, and this
  specific dated version may 404. Result: `.venv-iron` has no `aie` module and everything
  downstream fails.
- **Suggested fix:** commit the ~290 MB `vendor/wheelhouse/mlir_aie-*.whl` to a release
  asset or LFS (or document a one-time `uv pip install mlir_aie==<ver> --find-links
  <index>` pre-warm step while the version is still on the index), and record the exact
  provisioning command in `toolchain.lock` next to the pin. Do NOT rely on the nightly
  index staying populated.

### GAP #3 -- Peano (`llvm-aie`) silently missing on empty cache  [HIGH -- CLOSED]
- **Was:** the cp310 Peano wheel cannot install into py3.14, so Step 1 copies the unpacked
  `llvm-aie` tree out of `~/.cache/uv/archive-v0`. If that cache entry was absent the script
  only printed `WARNING: ... Peano binaries unavailable` and continued (set -e safe) -- Peano
  was then silently missing and Step 3/Step 4 failed confusingly two steps later.
- **Fix (DONE):** added a TERMINAL GUARD to `setup_route_b.sh` after all Peano-provisioning
  tiers. If `have_peano` (`.venv-iron/lib/python3.14/site-packages/llvm-aie/bin/clang`) is
  still false, it prints exactly what is missing + the pre-warm command
  (`uv pip install --python 3.14 "$PEANO_PIN" --find-links .../nightly`) and `exit 1`s. A
  matching guard hard-fails if the `mlir_aie` wheel left `import aie` broken. The warm-cache
  path is untouched -- both checks pass and the guard is a silent no-op (verified against the
  current `.venv-iron`: peano clang present, `import aie` OK). This does NOT itself perform a
  network fetch (the pre-warm is still a manual/documented step -- GAP #2's supply-chain
  concern), it just converts a silent absence into a loud, actionable early failure.

### GAP #4 -- MLIR distro + Peano pins are dated release/nightly assets  [MEDIUM -- OPEN, owned by `proper-install-consumable`]
- **Owner note:** mirroring the pinned MLIR-distro wheel and the Peano/`llvm-aie` dated
  nightly assets to a project-owned durable location is the `proper-install-consumable`
  task's job. THIS machine works off the warm uv cache + content-addressed toolchain cache;
  a truly-bare machine needs these two assets vendored.
- **What breaks:** Step 2 downloads `MLIR_DISTRO_WHEEL=mlir-23.0.0.2026060107+068c6c5c` and
  Peano `21.0.0.2026062301+cb664e8c` from upstream release/nightly tags. Dated nightly
  assets are pruned upstream over time; once pruned, `gh release download` / find-links
  return nothing and there is no local fallback for the MLIR distro (unlike the mlir_aie
  wheelhouse). Also requires `gh` authenticated.
- **Suggested fix:** mirror both pinned assets to a project-owned durable location (release
  asset / object store) and point the fetch scripts there with the upstream index as a
  fallback, not the primary.

### GAP #5 -- WER clips "never fetched by the chain"  [MEDIUM -- CLOSED, was a false alarm]
- **Verdict:** NOT a gap. `git ls-files artifacts/wer_clips/` returns 24 TRACKED files -- all
  17 wavs (`en_01..en_04`, `ru_01..ru_13`), `refs.json`, `SOURCE.md`, and the baseline JSONs
  (`parakeet_eval_results.json`, `int8_eval_results.json`, `whisper_npu_wer_{npu,onnx}.json`,
  `whisper_small_oracle.json`). A clean clone ALREADY has them; the Step 11/12 gates and the
  GigaAM calibration read straight from the checkout. No fetch step, no network needed.
- **`fetch_wer_clips.py`** remains only as a REGENERATOR (re-pull FLEURS, CC-BY-4.0, 16 kHz
  mono via ffmpeg) if the reference set ever needs refreshing -- it is intentionally not in
  the deploy chain, because the committed copies ARE the shipped-baseline reference.

### GAP #6 -- systemd unit hardcodes `WorkingDirectory=$REPO`; xclbins are relative  [LOW]
- **What breaks nothing today, but is fragile:** `install.sh` writes
  `WorkingDirectory=$REPO` and `npu-parakeet` loads xclbins from the RELATIVE
  `mlir-aie/programming_examples/.../whole_array/build/`. If the repo is later moved, or
  the xclbins are pruned by the aggressive `**/build/` gitignore + a `cargo clean`-style
  sweep, the service starts but fails to find kernels at request time. The dependency of a
  DEVICE service on gitignored build-tree artifacts is implicit.
- **Suggested fix:** stage the resident xclbins + insts into a stable `artifacts/`-side dir
  the engine also searches, so the running service does not depend on the disposable
  mlir-aie build sandbox.

---

## Bottom line

The chain is now ~95% automated. GAP #1 (`models/parakeet/`, the hard blocker) is CLOSED --
`fetch_models.sh` materializes it. GAP #3 is CLOSED -- a missing Peano/wheel now fails loud
and early with the pre-warm command instead of confusingly at Step 3/4. GAP #5 was a false
alarm -- the WER clips are committed to VCS. What remains: the manual submodule bootstrap
(0.b), and the two supply-chain durability debts GAP #2 (~290 MB `mlir_aie` wheel) and GAP
#4 (MLIR-distro + Peano dated assets), both OPEN and owned by the `proper-install-consumable`
task. On THIS machine the warm uv cache + content-addressed toolchain cache make the clone
build work end-to-end; a truly-bare machine still needs those assets vendored.
