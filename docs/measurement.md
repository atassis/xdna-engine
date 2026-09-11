# Per-request measurement

Every generation this engine serves is measured, and the measurement is part of the answer. This
document is the contract: what is recorded, where each number comes from, what it is safe to
compare against, and the on-disk format of a run log.

The design constraint that shapes all of it: on this box a latency is not a property of the model.
The AIE core clock is a DPM ladder, the power mode is per-boot and often unpinned, and the machine
drifts over hours. Two honest runs of identical code can differ by double digits. So a number
without its conditions is not a measurement, and the commonest apparent regression is a power mode
that moved.

---

## 1. What is recorded

One record per **decoded token**, not per emitted text frame. Those are not the same thing: a token
that completes no UTF-8 codepoint emits no text at all (a multi-byte character split across two BPE
tokens produces text only on the second), and a stop-sequence flush emits text with no token behind
it. A consumer counting text frames counts the wrong thing.

Each record carries:

| Field | Meaning |
| --- | --- |
| `seq`, `tok_id` | Position in the completion, and the token sampled. `tok_id` is null for a stop-flush record. |
| `text` | The token's own detokenized text. Empty when it completes no codepoint. |
| `emit` | What actually reached the client at this step. Not always `text`: a stop matcher holds text back until it knows the sequence did not start, then releases it in one piece. It is not stored under `x_npu` -- it IS the chunk's own `delta.content` (`text` for a completion), which is what makes a log line a replayable frame rather than a copy of one. |
| `t_ms`, `dt_ms` | Time since the start of the generation, and since the previous record. |
| `ph.step_ms` | The device dispatch that produced this token's logits, plus the backend's host glue around it. |
| `ph.sample_ms` | Logit sampling: penalties, top-k/top-p, the draw. |
| `ph.detok_ms` | Incremental detokenization and stop matching. |
| `residual_ms` | `dt_ms` minus the phases above. |
| `dev.dispatches`, `dev.transitions` | Device dispatches and hardware-context transitions this token cost. Present only when `NPU_DISPATCH_LOG` is set. |

Two attribution rules are worth stating outright, because both are easy to get wrong and neither is
visible in the output once it is wrong.

**`step_ms` belongs to the token whose logits it produced, not the iteration it ran in.** The loop
samples token N from logits already in hand, then dispatches to obtain the logits for N+1. The first
completion token's logits come from priming, so its `step_ms` is 0 and that cost sits in the prefill
record. Charging it to the token would count prefill twice.

**`dt_ms` for `seq 0` is measured from the end of prefill**, so it is a first-token latency wearing
an inter-token name. Percentiles are therefore computed over `seq >= 1` only. Folding it in lets a
long prompt masquerade as a decode stall.

Prefill is recorded as one object, not per token. It runs as a handful of batched dispatches over
many positions; a per-token timing there would be an average dressed up as a measurement.

## 2. The residual is a reported quantity

Phase shares are computed against the measured decode window, never against the sum of the phases.
If the named phases do not add up, the difference appears as `unattributed` rather than being spread
across whichever phases happen to be instrumented.

This matters more than it sounds. A breakdown whose parts are *defined* to total 100% cannot tell
you that the cost is somewhere nothing is looking, which is the single most useful thing a
breakdown can say. A large residual is a finding, not a rounding error.

## 3. Cost, and why this is not behind a flag

Measuring costs three `Instant::now()` calls and one small struct per token, against a decode step
measured in tens of milliseconds. Measured 2026-09-09 on qwen3-0.6b, 64 tokens, seed 42, temp 0: the
instrumented binary produces byte-identical output to the one without it, and their wall clocks
overlap (5251-6221 ms against 5286-6140 over two rounds; each run reloads the model cold, which is
most of that spread). The instrumentation cost is under run-to-run noise. It is therefore unconditional. An instrument you have to
remember to switch on is never armed for the run that turns out to matter.

Not everything is that cheap, and the difference decides each surface:

| Surface | Default | Why |
| --- | --- | --- |
| Recording | Always | Under run-to-run noise, measured above. |
| `timings` + `x_npu` on a non-streaming response | Always | A few hundred bytes, once. |
| CLI footer, on stderr | Always | Free to compute; stderr keeps a piped stdout clean. |
| Per-token `x_npu` on an SSE stream | Opt-in | 173 B/token measured (354 with it, 182 without); a 512-token completion carries ~89 KB of it against ~2 KB of text. |
| Trailing `npu.run.summary` frame | Opt-in | A strict OpenAI client is entitled to be surprised by an unknown `object`. |

The two opt-in rows are gated on bandwidth and client compatibility, never on the cost of measuring.

One counter is deliberately *not* sampled per token. NPU package power
(`/sys/class/accel/accel0/device/hwmon/*/power1_input`) costs ~785 us to read on this box -- 3.7% of
a decode step, enough to visibly perturb the interval it would be describing. It is read exactly
twice per generation, at the ends, and reported as two samples. Do not turn that pair into a
joules-per-token figure; it cannot carry one. A reading of 0 is reported as absent, because a driver
that does not populate the counter and a device drawing no power are not the same claim.

## 4. The verdict

Each summary names the layer the run was actually spending its time in -- `queue`, `load`,
`tokenize`, `prefill`, `device`, `sampling`, `detokenize`, or `unattributed` -- with that layer's
share of the total and the lever that acts on it.

This exists because a table of numbers does not answer the question people actually have. `queue`
and `load` in particular are not properties of the model at all: the NPU is single-tenant and one
thread owns it, so a second request waits out the first one's entire generation, and a cold request
pays for a model load inside its own latency. Both used to be invisible inside time-to-first-token
with no name attached.

## 5. Conditions

Every report carries what the run happened under: engine version, power mode (`xrt-smi`'s
`devices[0].platforms[0].status.power_mode`, probed once per process off the request path), kernel
release, and whether the model was already resident when the request arrived.

Residency is snapshotted *before* the model is resolved rather than inferred afterwards from how
long the load took. A cold first token is a different measurement from a warm one and must not be
averaged with it.

An unknown power mode is reported as unknown. Comparing two runs taken under different modes is not
a comparison, and `npu stats --diff` says so before it prints any timing delta.

## 6. Run logs

A run log is JSONL: one JSON object per line, each tagged by an `object` field, the way OpenAI
already discriminates its own stream. The per-token lines **are** the OpenAI stream chunks, with one
extra namespaced key. So the same bytes serve as a wire capture, a debugging record, and something a
client can be replayed against, and there is one serializer rather than three that can disagree.

```
{"object":"npu.run.header",  ...}   conditions the run happened under
{"object":"chat.completion.chunk", ..., "x_npu":{...}}   one per decoded token
{"object":"npu.prefill",     ...}   prompt side
{"object":"npu.run.summary", ...}   rolled-up numbers, verdict, conditions
```

A per-token line:

```json
{"id":"chatcmpl-3f2a","object":"chat.completion.chunk","created":1757400000,
 "model":"qwen3-0.6b",
 "choices":[{"index":0,"delta":{"content":" dispatch"},"finish_reason":null}],
 "x_npu":{
   "det":  {"seq":10,"tok_id":6108,"text":" dispatch"},
   "time": {"t_ms":636.6,"dt_ms":21.1,
            "ph":{"step_ms":20.2,"sample_ms":0.62,"detok_ms":0.09},
            "residual_ms":0.19},
   "dev":  {"dispatches":1,"transitions":0}}}
```

Four properties of that shape are load-bearing:

- **`det` versus `time`.** What is reproducible across runs lives in `det`; what is not lives in
  `time`. A determinism check is `jq '.x_npu.det'` on two logs and a diff -- no schema-aware filter
  list to keep in sync with the writer, which is the part that always rots.
- **`dt_ms` is stored, not derived.** Redundant against `t_ms`, and it stops three consumers from
  computing inter-token latency three slightly different ways.
- **`dev` is absent, not zeroed, when nothing counted.** An absent key reads as "not measured"; a
  zero reads as "measured, and it was none".
- **`dispatches` is the join key.** It is what lets a shim-BD dump or an AIE trace be attached to a
  specific token rather than to the run as a whole.

Line order is not significant -- the reader dispatches on `object`. The prefill line appears after
the token lines because its record only reaches the writing layer with the terminal item. A
truncated log parses: a run killed mid-generation is exactly when a log is worth the most, and it
reports as truncated rather than as a run with no tokens. An unknown `object` is skipped rather than
rejected, so a log written by a later version stays readable.

Write one by redirecting the CLI's own output -- `npu generate "..." --output json > run.jsonl` --
or set `NPU_TELEMETRY_LOG=<dir>` on the service to get one per request. There is deliberately no
"log to this file" flag: the shell already redirects, tees and pipes, and such a flag would be a
second output destination that can disagree with the first. The env var is the exception, because a
daemon has no per-request stdout to redirect, and it serves the case the per-request opt-in cannot:
the run you did not know you would need to explain, which you only find out about afterwards.

## 7. Reading them back

```
npu generate "..." --stats               # full breakdown after the answer, on stderr
npu generate "..." --output json         # NDJSON on stdout, one line per token, live
npu generate "..." --output json --no-stream   # one chat.completion object instead
npu stats run.jsonl                      # render a log
npu stats a.jsonl --diff b.jsonl         # compare two runs
npu replay run.jsonl [--realtime] [--frames]
npu top                                  # live: who is resident, serving, and for how long
```

`--output json` follows the stream flag, the way `/v1/chat/completions` does, so the streaming form
IS the run-log format: `> run.jsonl` produces a file the two readers below accept, `| tee run.jsonl`
keeps one while you watch, and `| jq 'select(.object=="chat.completion.chunk") | .x_npu.time.dt_ms'`
prints inter-token latencies as they happen. Lines are flushed individually, because a
block-buffered pipe would otherwise hold the whole run and emit it at the end. In JSON mode the
completion text is not echoed separately -- it is inside each chunk's `delta.content`.

`npu stats --diff` reports **token divergence first**. A timing difference between two runs that
produced different tokens is not a regression, it is a different computation, and printing the
milliseconds first invites reading it as one. It then flags a power-mode difference before any
delta, for the reason in section 5.

`npu replay` needs no config, no model and no device: a run log holds the frames that were served,
so replaying is reading them back. It emits the recorded bytes rather than re-rendering them, so
what it produces is what the client saw and not what this version's serializer would produce today.

## 8. Compatibility

The measurement objects sit alongside `usage`, which is unchanged. Two of them are deliberately
other people's schemas:

- **`timings`** is llama.cpp's object, field for field (`prompt_n`, `prompt_ms`,
  `prompt_per_token_ms`, `prompt_per_second`, and the four `predicted_*` equivalents).
- **`ollama`**, in the run-log summary, is Ollama's duration set in nanoseconds (`total_duration`,
  `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`).

Only fields whose meaning genuinely matches are emitted. A compatibility field that means something
slightly different from what its name implies elsewhere is worse than an absent one, because nothing
downstream can tell. Anything without a true counterpart lives in `x_npu` instead.
