# Qwen3 Embedding verification and delivery candidate

The causal adapter supports the pinned embedding checkpoints below through the existing engine API.
The full **36-layer 4B BF16 baseline** has passed bounded NPU numerical, ranking and independent-state
checks with both M1 sequential priming and repaired M256 token-block prefill. The smaller 0.6B M1
candidate also passed real CLI/HTTP and lifecycle checks; those served-path checks remain outstanding
for 4B. These results do not establish superiority over BGE or production/index compatibility.
BGE defaults are unchanged. Qwen3 generation artifacts are not embedding weights.

The selectable 4B delivery candidate is `scenarios/qwen3-embedding-4b.toml`. It uses full-dimensional
BF16 decode/prefill artifacts and relocatable artifact paths. It is not included in the default model
configuration. The candidate pins clean IRON source containing the QKV scheduling and LUT SiLU fixes.
Installation remains gated on full-model artifacts compatible with the selected toolchain and real 4B
CLI/HTTP/lifecycle verification. Retained development images are not release-safe merely because
their numerical results passed; do not bypass toolchain or IRON provenance checks. The candidate's
current-toolchain decode builds, but paired prefill is blocked on migrating IRON GEMM to the current
tap/FIFO APIs. The full recipe is therefore not yet an accepted end-to-end build.

## Pinned model contract

`scripts/qwen_embedding_pins.json` binds embedding checkpoint, configuration and tokenizer files to
upstream revisions and content hashes. The runner checks every allowlisted file before loading and
presents only those files to Transformers. LFS files use upstream SHA256; small files use upstream
Git blob IDs. Evidence also records content SHA256 for all files. Do not modify local snapshot files
while a run is reading them.

| Model ID | Upstream revision | Full dimension | Required file bytes |
|---|---|---:|---:|
| `qwen3-embedding-0.6b` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 1024 | 1,203,021,431 |
| `qwen3-embedding-4b` | `5cf2132abc99cad020ac570b19d031efec650f2b` | 2560 | 8,055,054,326 |

Sources: [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B),
[official usage](https://github.com/QwenLM/Qwen3-Embedding).
The selected fast-tokenizer files do not need separate vocab/merges downloads.
No downloader, automatic provisioning, cloud call, or runtime fallback is implemented.

Reference preparation:

- Documents: unchanged text; queries: `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}`.
- Fast tokenizer, `add_special_tokens=True`, no generation chat template, left padding and attention mask.
- Initial tooling policy: at most **512 tokens including instructions**; reject overflow, no truncation.
  This is not a measured NPU limit or a promise of the model card's advertised 32K context.
- Empty input strings, empty batches, unsupported IDs and zero-token encodings reject.
- Transformers `AutoModel` backbone output includes final RMSNorm. Pool the final valid token, not
  CLS, mean or the final padded position; convert that vector to float32 and L2-normalize.
- Full output dimension only. Reduced-dimensional support is not implemented.
- CPU only, explicit float32 (default) or bfloat16, eval/inference mode, eager attention, KV cache off,
  two PyTorch threads. Use an existing CPU environment with PyTorch and Transformers>=4.57.6.

The semantic profile records source files, instruction policy, tokenizer behavior, pooling,
normalization, dimensions and effective length limit. Runtime dtype is recorded separately. The
same profile is necessary but not sufficient for precision/runtime compatibility: target/reference
qualification is still required. Do not mix BGE/Qwen or incompatible model/profile versions in one index.

## Offline tests (no model download/device access)

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/test_qwen_embedding.py
python scripts/test_evaluate_embeddings.py
python scripts/test_qwen_embedding_specs.py
```

The small randomly initialized Transformers Qwen3 backbone in tests checks harness mechanics and
batch independence, not learned embedding quality, tokenizer parity with a real checkpoint or NPU parity.

## Reproduce a reference evaluation

Provision only the files listed in the pin manifest, at the exact official revision, after approving
bandwidth/storage costs. `SNAPSHOT` must name that local directory; loading is local-only.
`RUN` must be a new local evidence directory outside any production index. Never put private text
or evidence in tracked public files. Set `CPU_PYTHON` to the prepared CPU interpreter.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir "$RUN"
"$CPU_PYTHON" - "$RUN" <<'PY'
import json, pathlib, sys
fixture = json.loads(pathlib.Path('tests/data/qwen_embedding_synthetic.json').read_text())
for role, key in [('document', 'documents'), ('query', 'queries')]:
    with (pathlib.Path(sys.argv[1]) / (role + '-input.json')).open('x') as f:
        json.dump([item['text'] for item in fixture[key]], f, ensure_ascii=False)
PY
for role in document query; do
  "$CPU_PYTHON" scripts/qwen_embedding.py --model qwen3-embedding-0.6b \
    --snapshot "$SNAPSHOT" --input-type "$role" --dtype float32 \
    --input "$RUN/$role-input.json" --output "$RUN/$role-reference.json"
done
"$CPU_PYTHON" scripts/evaluate_embeddings.py \
  --dataset tests/data/qwen_embedding_synthetic.json \
  --documents-ref "$RUN/document-reference.json" --queries-ref "$RUN/query-reference.json" \
  --output "$RUN/quality.json"
```

Output records include ordered token IDs/masks, normalized vectors, semantic profile, CPU/runtime
identity, input-text digest, code digests, repeat maximum error, verified-load wall time, first-encode
wall time and repeated warm-encode wall time. The first encode is after loading and preprocessing;
it is not an end-to-end cold-service measurement. No peak-memory estimate is reported as a measurement.

The evaluator can additionally accept `--documents-target` and `--queries-target` in the same record
format. It rejects mismatched ordered texts or semantic profiles, reports vector cosine/absolute/
relative-L2 errors and both rankings. Default reports remain descriptive and unqualified. Explicit
numerical qualification requires both `--qualification-policy` and `--target-receipt`, plus both
target roles. The format-1 policy binds frozen bounds and complete dataset/reference/input/source/
producer/artifact identities; ordered tokens and loaded semantic/runtime profiles must match before
bounds are checked. Freeze justified limits before target acceptance scoring, not after observing a
failure. The evaluator does not independently establish actual device execution or an internal
schedule setting; retain the corresponding guarded execution evidence and actual process settings.
An explicit `deadline_unix_s: null` has no overall policy expiry; receipt timestamps must still be
finite, ordered and no earlier than the policy start. Execution authorization and per-command runtime
limits belong to the runner, not this numerical policy.

## Evaluation boundaries

The frozen synthetic development set has22 documents and14 queries, with explicit EN→EN, RU→RU,
RU→EN and EN→RU candidate scopes. It covers identifiers, paraphrases, intent negatives, negation,
unsupported limits, current/historical procedure, deployed/source distinction, measured/hypothesized
cost, residency distinctions and multi-source recall. Scope restricts language candidates explicitly;
a language filter is an evaluation condition, **not a security boundary**.

English candidate pools have16 documents and Russian pools6. Recall@10 on the latter is necessarily
saturated; per-case ranks/MRR and hard negatives are more informative. These tiny pools are not
population-level quality estimates. Labels are synthetic authored relevance judgments, not human
benchmark gold or claims about a real engine. Reference and target use the same candidates and labels.

Reports include Recall@5/10 (fraction of annotated relevant sources), MRR@10 (misses count zero),
language groups, each complete ranking, and hard negatives ahead of the first relevant result.
The later private sample needs independently read source sections, acceptable alternatives, source
hashes/statuses and an off-index label manifest. Neither that sample nor actual model quality has
been evaluated by the synthetic tests.

## Target import preparation

The fused-Qwen model-spec table now distinguishes `qwen3-embedding-0.6b` and
`qwen3-embedding-4b` from generation models, including their actual vocabularies151669/151665.
`scripts/dump_llm_weights.py` accepts these IDs only with a verified pinned local
`--checkpoint-dir`; it rejects repository overrides and implicit downloads. Only allowlisted weight
shards are read. Export metadata and resume identity include the pinned checkpoint source.
A small synthetic checkpoint exercises export, successful resume and changed-source rejection.
The pinned 0.6B backbone has been fully exported (310 tensors) and compiled offline at 28 layers.
Embedding checkpoints use unprefixed backbone names (`embed_tokens.weight`, `layers.*`), unlike
causal-LM checkpoints with `model.*`. The BF16 vocabulary head pads its physical row count without
changing the actual vocabulary or embedding lookup values. The host accepts padded raw BF16
`W_head` only when declared physical rows, precision, layout and file size agree; token lookup still
uses the logical vocabulary. Host-only embedding sidecars remain exact logical-sized BF16 tables,
not device-head packing. Both artifact writers carry the pinned source receipt, and shared-arena
prefill rejects a different decode source.

Offline build checks cover a 512-capacity decode artifact and a separate paired decode/prefill
artifact at physical capacity 2048 with 256 **token rows**, not 256 independent documents.
The latter reuses existing `seed`/`assumed` GEMM registry entries, not measured performance winners.
The initial 512-total-token input policy is unchanged. A 32-row prefill violates the 512-capacity
decode's 64-row KV block; the legal 64-row shape is not in the registry. No guard was disabled or
registry entry invented to force those configurations through.

ELF digests, declared weight sizes, shared weight offsets, selected embedding rows, vocabulary
padding and final-norm representation have been checked offline. Those checks alone are **not
forward-pass comparisons**. The compiled `xf` is a scratch-arena buffer; runtime acceptance requires
synchronized hidden readback. The full 4B decode/prefill pair subsequently passed the bounded
qualification described below; a fresh build or a different source/toolchain needs its own checks.

The source adapter reuses existing Qwen prefill/decode and
`DecodeStep::final_hidden()`, rather than introducing a separate embedding-only forward graph.
This retains unused vocabulary-head computation on decode steps. It is a correctness-first path;
its cost must be measured, not presented as embedding-only execution.

## Selectable BF16 artifact recipe

The existing model-artifact dispatcher recognizes `qwen3-embedding-4b` with
`scenarios/qwen3-embedding-4b.toml`. Its stages are pinned local weight export, 36-layer S512/T64
decode, shared M256 prefill, and tokenizer packaging. The FP32 export is transient build data, not
a served dependency. The prefill tile choices are explicit baseline data, not measured performance
winners. Publication checks reject source/tokenizer drift, wrong geometry or precision, noncausal
prefill, missing shared weights and references outside the paired outputs.

The pinned checkpoint must already exist at `$INPUTS/qwen3-embedding-4b`; this recipe does not
automatically download it. From the repository root, inspect the selected recipe without building:

```sh
MODEL_EXPORT_PY="$CPU_PYTHON" MODEL_IRON_PY="$IRON_PYTHON" \
  "$CPU_PYTHON" scripts/model_artifact_dispatch.py \
  --config "$CONFIG" --repo "$PWD" --artifacts-root "$ARTIFACTS" \
  --model-input-root "$INPUTS" --build-dir "$BUILD" --plan
```

`CONFIG` is an engine configuration containing the explicit model entry shown below. Set the two
Python paths to prepared export/IRON environments and the data paths to the intended disk-backed
roots. After resource/execution admission, `--build` uses the same staged recipe. A completed build
still needs numerical and real-interface qualification; it does not install or register the model.

## Source adapter and API (not deployed)

An explicit `[embeddings] backbone = "qwen3-embedding-0.6b"` selects the causal adapter; the scenario
also requires `pooling = "last"`, `normalize = true`, no `batch_rows`, a `[model] hidden = 1024`,
and `artifacts.decode`/`artifacts.tokenizer_dir` paths. `artifacts.prefill` is optional. The registry
checks the embedding source receipt, pinned tokenizer/config bytes, and artifact/config geometry
before opening the device. Generation checkpoints are rejected. There is no new default scenario or
automatic artifact build. The existing BGE/ESM scenarios retain their previous selection behavior.

Each array item starts with explicitly cleared KV state, independent of generation's KV-reuse
environment setting. The complete batch is tokenized and length-validated before dispatch. Prefill
primes all but the final token; that token runs through decode, then the final-normed hidden is
read back and L2-normalized. Normalization accumulates in f64 and returns f32, rejecting zero,
nonfinite and wrong-sized hidden states. Scratch readback now synchronizes scratch, rather than
assuming the output-arena synchronization covers it. Clearing caches and synchronizing the whole
scratch BO have unmeasured costs; this is not an embedding-optimized implementation.

The source `/v1/embeddings` route accepts `input_type = "query" | "document"` (omitted means
`document`) and optional `dimensions`. Qwen accepts only its full dimension; invalid role/type,
zero dimensions and reduced dimensions reject. Query instructions count toward the token limit.
Options travel through the existing actor/Embedder path, with response shape and item order unchanged.
Legacy requests use the existing batch route. BGE/ESM reject explicit query preparation or dimension
options rather than silently ignoring them; an explicit document role alone preserves their behavior.
The source CLI accepts the same options, forwarding them through the existing control socket:

```sh
npu embed "how does the cache reset?" --model qwen3-embedding-0.6b --input-type query --dimensions 1024
npu embed "Each independent document clears KV state." --model qwen3-embedding-0.6b --input-type document
```

Neither option is inserted when omitted, so existing CLI document requests remain unchanged. The CLI
rejects invalid roles and non-positive/non-integer dimensions; the model rejects unsupported dimensions.
These examples describe source behavior, not an installed model or a new default.

CPU-only Rust tests exercise adapter mechanics, invalid outputs, reset/synchronization ordering and
mock HTTP routing. An explicitly selected local-fixture test checks the real pinned tokenizer against
the recorded CPU-reference document/query token IDs and checks the offline paired artifacts' preflight.
Neither test executes an NPU forward. The shared adapter handles 1024/2560-dimensional output.
The 4B runtime metadata SHA256 pins are derived from bytes verified against the pinned upstream
Git blob IDs. Its retained full-depth BF16 pair passed runtime admission and bounded target
qualification. Adding metadata pins alone still does not qualify a new artifact.

## Profile and runtime discovery

After a successful Qwen model load, the existing `GET /v1/models` response includes an optional
`embedding` object for that model, with `identity_scope = "last-successful-load"`:

- `profile`: pinned checkpoint/revision/file hashes, fixed query/document preparation, special-token
  policy, effective token limit, pooling, normalization, full dimension and profile format version.
  Its schema matches the CPU reference. The padding policy remains left padding; serial execution
  does not add padding rows. `normalize = "l2-float32"` names the f32 normalized output, while the
  runtime records the actual accumulator precision.
- `runtime`: backend, serial batching, hidden/output dtypes, normalization accumulator, declared
  weight quantization, decode/prefill artifact hash prefixes and toolchain fingerprints. Artifact
  identifiers use the existing telemetry format, the first 12 hex digits of SHA256, not full-file
  cryptographic validation. No local artifact paths are exposed.
- `runtime.qualification = "unqualified"` and `runtime.index_compatibility = "not-qualified"`:
  loading does not establish target/reference numerical or index compatibility.

Metadata is captured from the successful loaded instance and its parsed artifact declarations, not
reread from mutable files on discovery. Actual loaded decode/prefill source receipts must match one
another and the preflight receipt. A cold model has no embedding metadata until its first successful
load. After unload, any retained metadata describes that last successful instance, not the files a
future load will use; read the existing model `state` and snapshot `age_s` alongside it. A failed
reload clears the metadata rather than advertising the previous identity as current. BGE/ESM do not
acquire invented profiles. The standard `/v1/embeddings` response shape is unchanged.

Compare complete semantic profiles before reusing an index; then separately qualify the runtime's
precision and target errors. A matching profile or artifact hash prefix alone is not qualification.

## Bounded qualification and remaining delivery

The explicit M1 BF16 candidate passed full bounded numerical/repeat checks, the real Model API,
source-built CLI/HTTP validation and cold/unload/failed-reload recovery. Bounded retrieval showed
no regression against the same Qwen original-FP32 CPU oracle; that is not a BGE comparison or a
population-level quality/adoption result. The default batched-prefill path still has a retained
numerical acceptance failure. A controlled replay distinguishes that path/handoff from sequential
priming, but does not identify an individual operator or prove an intrinsic precision floor.

M1 uses the existing `NPU_LLM_PREFILL_BATCHED=0` switch on the process loading the model, not merely
on a client connecting to an already-running service. It is an explicitly selected experimental
schedule, not an automatic fallback or a new default. Keep it isolated: this process-wide switch
also controls other models using the same prefill primitive. Discovery's `serial-unpadded` describes
independent input batching, not attestation of this internal schedule; qualify the actual process
settings separately. Runtime/index qualification labels intentionally remain unchanged.

The retained 4B BF16 M1 and repaired M256 runs passed numerical acceptance against independent
original-weight FP32 references on 34 documents and 29 queries. Minimum vector cosine was
0.9993694 for M1 and 0.9968951 for M256, with exact repeats. Both preserved all relevant-result ranks
and hard-negative lists, including the reference's two existing hard-negative errors. The sample
mixes synthetic development and historical cases; it is not an independent population-level quality
benchmark or evidence that Qwen is better than BGE. M256 denotes token rows, not independent documents.

The baseline still performs unused vocabulary-head computation, host token lookup and host L2
normalization. The embedding-only fused endpoint is separate unfinished optimization work, not a
requirement for selectable BF16 delivery. INT8 and mixed INT4 weight-only paths need separate
quantized-weight references, paired-kernel correctness, retrieval and measured cost checks; neither
is qualified for this model. A generic quantization option is not a supported Qwen embedding artifact.

4B CLI/HTTP/lifecycle, BGE-relative retrieval, matched cold/warm and indexing/query costs, reproducible
release builds and adoption remain open. No 0.6B-to-4B numerical extrapolation or unmeasured speedup
is claimed. No Qwen embedding model is registered in the live service by these source changes.
Do not assume an installed service implements them or that incompatible vectors can share an index.

After the delivery gates pass, an explicitly selected engine configuration can register the scenario
without changing the default embedding model:

```toml
[[model]]
name = "qwen3-embedding-4b"
scenario = "scenarios/qwen3-embedding-4b.toml"
```

The matching request is:

```sh
npu embed "how does independent cache reset work?" --model qwen3-embedding-4b --input-type query --dimensions 2560
```

Registration, supported installation and service cutover are separate operations. Keep existing BGE
models, search defaults and indexes unchanged until their own acceptance and adoption decisions.
