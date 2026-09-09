//! How a run reaches a wire or a file: the JSONL run log, and the two compat objects that make
//! existing clients render our numbers without knowing anything about us.
//!
//! The JSONL log is not a second format. Every line is a JSON object with an `object` field, the
//! way OpenAI already discriminates its own stream, and the per-token lines ARE the OpenAI stream
//! chunks with one extra namespaced key. So the same bytes are a wire capture, a debugging log,
//! and something a client can be replayed against -- and there is one serializer, not three that
//! can disagree.
//!
//! Layout of a file:
//! ```text
//! {"object":"npu.run.header",  ...}   conditions the run happened under
//! {"object":"npu.prefill",     ...}   prompt side; not per-token, and not pretending to be
//! {"object":"chat.completion.chunk", ..., "x_npu":{...}}   one per DECODED TOKEN
//! {"object":"npu.run.summary", ...}   rolled-up numbers and the reconciliation
//! ```
//!
//! Inside `x_npu`, `det` holds what is reproducible across runs and `time` holds what is not. A
//! determinism check is then `jq '.x_npu.det'` on both files and a diff -- no schema-aware filter
//! list to keep in sync with this struct, which is the thing that always rots.

use serde_json::{json, Map, Value};

use crate::pipeline::FinishReason;
use crate::telemetry::{Bound, GenerationReport, PrefillRecord, RunConditions, StepPhases, StepRecord, Summary};

/// Identity every line in one file shares.
#[derive(Debug, Clone)]
pub struct RunMeta {
    /// The completion id, so a log line and a served response can be matched up later.
    pub id: String,
    pub created: i64,
    pub model: String,
    /// Chat shape (`delta`) versus completion shape (`text`). A replay has to produce the shape the
    /// client asked for, not the one we prefer.
    pub chat: bool,
}

fn us_f(us: u64) -> f64 {
    us as f64 / 1e3
}

pub fn header_line(c: &RunConditions, m: &RunMeta) -> Value {
    json!({
        "object": "npu.run.header",
        "id": m.id,
        "created": m.created,
        "model": m.model,
        "engine_version": c.engine_version,
        "power_mode": c.power_mode,
        "resident": c.resident,
        "kernel": c.kernel,
        "started_unix": c.started_unix,
    })
}

pub fn prefill_line(p: &PrefillRecord, m: &RunMeta) -> Value {
    json!({
        "object": "npu.prefill",
        "id": m.id,
        "tokens": p.tokens,
        "batched": p.batched,
        "stepwise": p.stepwise,
        "ms": us_f(p.us),
        "dispatches": p.dispatches,
    })
}

fn phases_json(p: &StepPhases) -> Value {
    json!({ "step_ms": us_f(p.step_us), "sample_ms": us_f(p.sample_us), "detok_ms": us_f(p.detok_us) })
}

/// One decoded token, as an OpenAI stream chunk carrying its own measurement.
///
/// `finish_reason` is null here exactly as in a live stream: the terminal frame is the summary
/// line, so a replay emits these verbatim and then closes the stream itself.
pub fn chunk_line(r: &StepRecord, m: &RunMeta) -> Value {
    let mut x = Map::new();
    x.insert("det".into(), json!({ "seq": r.seq, "tok_id": r.token, "text": r.text }));
    x.insert("time".into(), json!({
        "t_ms": us_f(r.t_us),
        "dt_ms": us_f(r.dt_us),
        "ph": phases_json(&r.phases),
        "residual_ms": us_f(r.residual_us()),
    }));
    // Omitted rather than null-filled when the backend counts nothing: an absent key reads as "not
    // measured" to every consumer, where a zero would read as "measured, and it was none".
    if r.dispatches.is_some() || r.transitions.is_some() {
        x.insert("dev".into(), json!({ "dispatches": r.dispatches, "transitions": r.transitions }));
    }
    let choice = if m.chat {
        json!({ "index": 0, "delta": { "content": r.emit }, "finish_reason": Value::Null })
    } else {
        json!({ "index": 0, "text": r.emit, "finish_reason": Value::Null })
    };
    json!({
        "id": m.id,
        "object": if m.chat { "chat.completion.chunk" } else { "text_completion" },
        "created": m.created,
        "model": m.model,
        "choices": [choice],
        "x_npu": Value::Object(x),
    })
}

/// llama.cpp's `timings` object, field for field.
///
/// Emitted because it costs one function and every tool that already reads llama-server reads it.
/// Only fields whose meaning genuinely matches are here -- a compat field that means something
/// slightly different is worse than an absent one, because nothing downstream can tell.
pub fn timings_object(s: &Summary) -> Value {
    let per = |ms: f64, n: u32| if n == 0 { 0.0 } else { ms / n as f64 };
    let rate = |ms: f64, n: u32| if ms == 0.0 { 0.0 } else { n as f64 * 1e3 / ms };
    let (pms, dms) = (us_f(s.prefill_us), us_f(s.decode_us));
    json!({
        "prompt_n": s.prompt_tokens,
        "prompt_ms": pms,
        "prompt_per_token_ms": per(pms, s.prompt_tokens),
        "prompt_per_second": rate(pms, s.prompt_tokens),
        "predicted_n": s.completion_tokens,
        "predicted_ms": dms,
        "predicted_per_token_ms": per(dms, s.completion_tokens),
        "predicted_per_second": s.tok_per_s,
    })
}

/// Ollama's duration fields, in nanoseconds as Ollama reports them.
///
/// This is what makes an Open WebUI stats widget light up against this engine with no client-side
/// work at all. `load_duration` is our residency work and `prompt_eval_*` our prefill, which are
/// the same quantities under different names; anything without a true counterpart is left out.
pub fn ollama_object(s: &Summary) -> Value {
    let ns = |us: u64| us * 1_000;
    json!({
        "total_duration": ns(s.total_us),
        "load_duration": ns(s.load_us),
        "prompt_eval_count": s.prompt_tokens,
        "prompt_eval_duration": ns(s.prefill_us),
        "eval_count": s.completion_tokens,
        "eval_duration": ns(s.decode_us),
    })
}

/// The rolled-up numbers, plus the reconciliation, the verdict, and the conditions the serving
/// thread could not see when it wrote the header.
///
/// Takes the whole report rather than a `Summary` someone computed elsewhere: a summary and a
/// report that do not describe the same run is a bug nothing downstream could detect.
pub fn summary_line(r: &GenerationReport, m: &RunMeta, reason: FinishReason) -> Value {
    let s = &r.summarize();
    json!({
        "object": "npu.run.summary",
        "id": m.id,
        "model": m.model,
        "finish_reason": reason.as_str(),
        "usage": {
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "total_tokens": s.prompt_tokens + s.completion_tokens,
        },
        "timings": timings_object(s),
        "ollama": ollama_object(s),
        "x_npu": npu_object(s),
        "conditions": {
            "engine_version": r.conditions.engine_version,
            "power_mode": r.conditions.power_mode,
            "resident": r.conditions.resident,
            "kernel": r.conditions.kernel,
            // Two samples at the ends of the generation, never an integral -- see the field docs.
            "npu_power_uw": { "start": r.npu_power_start_uw, "end": r.npu_power_end_uw },
        },
    })
}

/// Our own namespaced view: the things no other engine reports, and the ones it would be dishonest
/// to fold into a compat field.
pub fn npu_object(s: &Summary) -> Value {
    json!({
        "tok_per_s": s.tok_per_s,
        "prompt_tok_per_s": s.prompt_tok_per_s,
        "ttft_ms": us_f(s.ttft_us),
        "itl_ms": {
            "mean": us_f(s.itl_mean_us),
            "p50": us_f(s.itl_p50_us),
            "p95": us_f(s.itl_p95_us),
            "p99": us_f(s.itl_p99_us),
            "max": us_f(s.itl_max_us),
        },
        "spans_ms": {
            "queue": us_f(s.queue_us),
            "load": us_f(s.load_us),
            "tokenize": us_f(s.tokenize_us),
            "prefill": us_f(s.prefill_us),
            "decode": us_f(s.decode_us),
            "total": us_f(s.total_us),
        },
        "decode_phases_ms": phases_json(&s.phases),
        "decode_residual_ms": us_f(s.residual_us),
        "dispatches": s.dispatches,
        "transitions": s.transitions,
        "bound": s.bound.as_str(),
        "bound_share": s.bound_share,
        "lever": s.bound.lever(),
    })
}

/// The non-streaming response body: OpenAI's completion object with the measurements attached.
///
/// One definition, used by the HTTP route and by the CLI's `--output json`. Two hand-rolled copies
/// of the same object is how the two surfaces end up reporting different numbers for one run.
pub fn completion_object(text: &str, reason: FinishReason, r: &GenerationReport, m: &RunMeta) -> Value {
    let s = r.summarize();
    let choice = if m.chat {
        json!({ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": reason.as_str() })
    } else {
        json!({ "index": 0, "text": text, "finish_reason": reason.as_str() })
    };
    json!({
        "id": m.id,
        "object": if m.chat { "chat.completion" } else { "text_completion" },
        "created": m.created,
        "model": m.model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "total_tokens": s.prompt_tokens + s.completion_tokens,
        },
        "timings": timings_object(&s),
        "x_npu": npu_object(&s),
    })
}

/// One parsed run log.
#[derive(Debug, Default, Clone)]
pub struct Run {
    pub conditions: RunConditions,
    pub id: String,
    pub model: String,
    pub chat: bool,
    pub prefill: PrefillRecord,
    pub steps: Vec<StepRecord>,
    /// Present unless the run was cut short before its summary was written -- which is itself worth
    /// seeing, so a truncated file parses rather than failing.
    pub summary: Option<Summary>,
    pub finish_reason: Option<String>,
    /// Chunk lines the parser could read as OpenAI frames, verbatim, for replay. Kept as written
    /// rather than re-rendered from `steps`, so a replay reproduces the recorded bytes even if this
    /// module's rendering changes later.
    pub frames: Vec<String>,
}

fn f64_ms_to_us(v: Option<&Value>) -> u64 {
    v.and_then(Value::as_f64).map(|ms| (ms * 1e3).round() as u64).unwrap_or(0)
}

fn opt_u32(v: Option<&Value>) -> Option<u32> {
    v.and_then(|x| if x.is_null() { None } else { x.as_u64() }).map(|n| n as u32)
}

/// Read a run log back. Unknown `object` values are skipped rather than rejected: a log written by
/// a later version must still be readable by this one, or the format stops being a debugging tool
/// the first time it grows a field.
pub fn parse_run(text: &str) -> Result<Run, String> {
    let mut run = Run::default();
    let mut seen = false;
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        let obj = v.get("object").and_then(Value::as_str).unwrap_or("");
        seen = true;
        match obj {
            "npu.run.header" => {
                run.id = v["id"].as_str().unwrap_or_default().to_string();
                run.model = v["model"].as_str().unwrap_or_default().to_string();
                run.conditions = RunConditions {
                    engine_version: v["engine_version"].as_str().unwrap_or_default().to_string(),
                    model: run.model.clone(),
                    power_mode: v["power_mode"].as_str().map(str::to_string),
                    resident: v["resident"].as_bool(),
                    kernel: v["kernel"].as_str().map(str::to_string),
                    started_unix: v["started_unix"].as_i64().unwrap_or(0),
                };
            }
            "npu.prefill" => {
                run.prefill = PrefillRecord {
                    tokens: v["tokens"].as_u64().unwrap_or(0) as u32,
                    batched: v["batched"].as_u64().unwrap_or(0) as u32,
                    stepwise: v["stepwise"].as_u64().unwrap_or(0) as u32,
                    us: f64_ms_to_us(v.get("ms")),
                    dispatches: opt_u32(v.get("dispatches")),
                };
            }
            "chat.completion.chunk" | "text_completion" => {
                run.chat = obj == "chat.completion.chunk";
                run.frames.push(line.to_string());
                let ch = &v["choices"][0];
                let emit = if run.chat { &ch["delta"]["content"] } else { &ch["text"] };
                let x = &v["x_npu"];
                let ph = &x["time"]["ph"];
                run.steps.push(StepRecord {
                    seq: x["det"]["seq"].as_u64().unwrap_or(0) as u32,
                    token: opt_u32(x["det"].get("tok_id")),
                    text: x["det"]["text"].as_str().unwrap_or_default().to_string(),
                    emit: emit.as_str().unwrap_or_default().to_string(),
                    t_us: f64_ms_to_us(x["time"].get("t_ms")),
                    dt_us: f64_ms_to_us(x["time"].get("dt_ms")),
                    phases: StepPhases {
                        step_us: f64_ms_to_us(ph.get("step_ms")),
                        sample_us: f64_ms_to_us(ph.get("sample_ms")),
                        detok_us: f64_ms_to_us(ph.get("detok_ms")),
                    },
                    dispatches: opt_u32(x["dev"].get("dispatches")),
                    transitions: opt_u32(x["dev"].get("transitions")),
                });
            }
            "npu.run.summary" => {
                run.finish_reason = v["finish_reason"].as_str().map(str::to_string);
                let n = &v["x_npu"];
                let sp = &n["spans_ms"];
                let itl = &n["itl_ms"];
                let ph = &n["decode_phases_ms"];
                run.summary = Some(Summary {
                    prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                    completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
                    ttft_us: f64_ms_to_us(n.get("ttft_ms")),
                    queue_us: f64_ms_to_us(sp.get("queue")),
                    load_us: f64_ms_to_us(sp.get("load")),
                    tokenize_us: f64_ms_to_us(sp.get("tokenize")),
                    prefill_us: f64_ms_to_us(sp.get("prefill")),
                    decode_us: f64_ms_to_us(sp.get("decode")),
                    total_us: f64_ms_to_us(sp.get("total")),
                    tok_per_s: n["tok_per_s"].as_f64().unwrap_or(0.0),
                    prompt_tok_per_s: n["prompt_tok_per_s"].as_f64(),
                    itl_mean_us: f64_ms_to_us(itl.get("mean")),
                    itl_p50_us: f64_ms_to_us(itl.get("p50")),
                    itl_p95_us: f64_ms_to_us(itl.get("p95")),
                    itl_p99_us: f64_ms_to_us(itl.get("p99")),
                    itl_max_us: f64_ms_to_us(itl.get("max")),
                    phases: StepPhases {
                        step_us: f64_ms_to_us(ph.get("step_ms")),
                        sample_us: f64_ms_to_us(ph.get("sample_ms")),
                        detok_us: f64_ms_to_us(ph.get("detok_ms")),
                    },
                    residual_us: f64_ms_to_us(n.get("decode_residual_ms")),
                    dispatches: opt_u32(n.get("dispatches")),
                    transitions: opt_u32(n.get("transitions")),
                    bound: match n["bound"].as_str().unwrap_or("") {
                        "queue" => Bound::Queue,
                        "load" => Bound::Load,
                        "tokenize" => Bound::Tokenize,
                        "prefill" => Bound::Prefill,
                        "device" => Bound::Device,
                        "sampling" => Bound::Sampling,
                        "detokenize" => Bound::Detokenize,
                        _ => Bound::Unattributed,
                    },
                    bound_share: n["bound_share"].as_f64().unwrap_or(0.0),
                });
            }
            _ => {}
        }
    }
    if !seen {
        return Err("empty run log".to_string());
    }
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::GenerateUsage;
    use crate::telemetry::GenerationReport;

    fn meta() -> RunMeta {
        RunMeta { id: "chatcmpl-test".into(), created: 1_700_000_000, model: "qwen3-0.6b".into(), chat: true }
    }

    fn a_report() -> GenerationReport {
        let steps = (0..4u32)
            .map(|i| StepRecord {
                seq: i,
                token: Some(1000 + i),
                text: format!("t{i}"),
                emit: format!("t{i}"),
                t_us: 5_000 + 20_000 * i as u64,
                dt_us: if i == 0 { 5_000 } else { 20_000 },
                phases: StepPhases { step_us: if i == 0 { 0 } else { 18_000 }, sample_us: 300, detok_us: 90 },
                dispatches: Some(if i == 0 { 0 } else { 1 }),
                transitions: Some(0),
            })
            .collect();
        GenerationReport {
            conditions: RunConditions {
                engine_version: "0.1.0".into(),
                model: "qwen3-0.6b".into(),
                power_mode: Some("turbo".into()),
                resident: Some(true),
                kernel: Some("7.2.0".into()),
                started_unix: 1_700_000_000,
            },
            queue_us: 1_200,
            load_us: 0,
            tokenize_us: 400,
            prefill: PrefillRecord { tokens: 12, batched: 0, stepwise: 12, us: 4_000, dispatches: Some(12) },
            steps,
            generate_us: 66_000,
            usage: GenerateUsage { prompt_tokens: 12, completion_tokens: 4 },
            npu_power_start_uw: None,
            npu_power_end_uw: None,
        }
    }

    fn render(r: &GenerationReport) -> String {
        let m = meta();
        let mut out = vec![header_line(&r.conditions, &m).to_string(), prefill_line(&r.prefill, &m).to_string()];
        out.extend(r.steps.iter().map(|st| chunk_line(st, &m).to_string()));
        out.push(summary_line(r, &m, FinishReason::Stop).to_string());
        out.join("\n")
    }

    #[test]
    fn a_token_line_is_a_valid_openai_chunk() {
        // The whole premise: an OpenAI client reading this line sees a normal chunk and ignores
        // the extra key. If that stops being true the format has become a private one.
        let v = chunk_line(&a_report().steps[1], &meta());
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["content"], "t1");
        assert!(v["choices"][0]["finish_reason"].is_null());
        assert_eq!(v["x_npu"]["det"]["tok_id"], 1001);
    }

    #[test]
    fn a_completion_run_uses_the_completion_shape() {
        let m = RunMeta { chat: false, ..meta() };
        let v = chunk_line(&a_report().steps[1], &m);
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "t1");
        assert!(v["choices"][0].get("delta").is_none());
    }

    #[test]
    fn unmeasured_device_counters_leave_the_key_out_entirely() {
        let mut st = a_report().steps[1].clone();
        st.dispatches = None;
        st.transitions = None;
        let v = chunk_line(&st, &meta());
        assert!(v["x_npu"].get("dev").is_none(), "absent, not null and not zero");
    }

    #[test]
    fn the_completion_object_is_openai_shaped_with_measurements_attached() {
        // What `npu generate --output json` prints and what the non-streaming HTTP route returns
        // are this one object. A client that knows only OpenAI reads the first four keys and skips
        // the rest.
        let r = a_report();
        let v = completion_object("hello", FinishReason::Stop, &r, &meta());
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 16);
        assert_eq!(v["timings"]["predicted_n"], 4);
        assert_eq!(v["x_npu"]["bound"], "device");
        let m = RunMeta { chat: false, ..meta() };
        let v = completion_object("hello", FinishReason::Length, &r, &m);
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "hello");
        assert!(v["choices"][0].get("message").is_none());
    }

    #[test]
    fn a_run_round_trips_through_the_log() {
        let r = a_report();
        let back = parse_run(&render(&r)).unwrap();
        assert_eq!(back.id, "chatcmpl-test");
        assert_eq!(back.conditions.power_mode.as_deref(), Some("turbo"));
        assert_eq!(back.prefill, r.prefill);
        assert_eq!(back.steps, r.steps, "records survive the round trip unchanged");
        assert_eq!(back.finish_reason.as_deref(), Some("stop"));
        let s = back.summary.unwrap();
        assert_eq!(s.completion_tokens, 4);
        assert_eq!(s.bound, Bound::Device);
        assert_eq!(s.itl_p99_us, r.summarize().itl_p99_us);
    }

    #[test]
    fn replaying_the_frames_reproduces_the_text() {
        let r = a_report();
        let back = parse_run(&render(&r)).unwrap();
        let text: String = back.steps.iter().map(|s| s.emit.as_str()).collect();
        assert_eq!(text, "t0t1t2t3");
        assert_eq!(back.frames.len(), 4, "one replayable frame per token");
    }

    #[test]
    fn a_truncated_log_still_parses_up_to_the_cut() {
        // A run killed mid-generation is exactly when the log matters most.
        let full = render(&a_report());
        let cut: Vec<&str> = full.lines().take(4).collect();
        let back = parse_run(&cut.join("\n")).unwrap();
        assert_eq!(back.steps.len(), 2);
        assert!(back.summary.is_none());
    }

    #[test]
    fn an_unknown_line_type_is_skipped_not_rejected() {
        let mut s = render(&a_report());
        s.push_str("\n{\"object\":\"npu.future.thing\",\"whatever\":1}");
        assert_eq!(parse_run(&s).unwrap().steps.len(), 4);
    }

    #[test]
    fn the_compat_objects_agree_with_the_summary_they_came_from() {
        // Two names for the same measurement must not drift: llama.cpp reports milliseconds,
        // Ollama nanoseconds, and both have to be the number we measured.
        let s = a_report().summarize();
        let t = timings_object(&s);
        let o = ollama_object(&s);
        assert_eq!(t["predicted_n"], 4);
        assert_eq!(o["eval_count"], 4);
        assert_eq!(
            (t["predicted_ms"].as_f64().unwrap() * 1e6).round() as u64,
            o["eval_duration"].as_u64().unwrap(),
            "eval duration is one quantity in two units"
        );
        assert_eq!(o["prompt_eval_duration"].as_u64().unwrap(), s.prefill_us * 1_000);
    }
}
