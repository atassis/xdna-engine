//! The Ollama API surface, so a client can DISCOVER what a model can do.
//!
//! Why a second surface at all, when `/v1/*` already serves the same models: OpenAI's `/v1/models`
//! carries `id`, `object`, `created`, `owned_by` and nothing else. It has no field for a context
//! length and none for a capability, which is why every OpenAI-compatible client makes the user
//! tick "supports tools" by hand. Ollama's `/api/show` carries both -- a `capabilities` array and
//! `model_info["<arch>.context_length"]` -- and clients read them.
//!
//! We already listen on 11434, Ollama's own port, so a client pointed at this engine as an Ollama
//! server configures itself. That is the whole reason this module exists; it adds no capability to
//! the engine, it just stops the engine lying by omission about the ones it has.
//!
//! THE WIRE IS NOT OPENAI'S, in three ways that each cost a bug if assumed:
//!   * streaming is NDJSON -- one bare JSON object per line, no `data:` prefix and no `[DONE]`;
//!   * `message.tool_calls[].function.arguments` is an OBJECT here and a JSON STRING on `/v1`;
//!   * the terminal frame is a normal object with `done: true`, not a separate event.

use std::collections::HashMap;
use std::sync::Mutex;

use npu_engine::llm::{ChatTemplate, ToolSyntax};
use serde_json::{json, Value};

use crate::registry::{LoadState, ModelStatus};

/// What a generate model can do, resolved WITHOUT loading it.
///
/// Loading to answer a metadata query would evict whatever is resident, and a client polls this
/// per model on every page load. Everything here comes from files the scenario already names.
#[derive(Debug, Clone, Default)]
pub struct GenerateCapabilities {
    /// `dims.S` from the decode artifact's `meta.json` -- the KV window the ELF was BUILT at, which
    /// is the real bound a client needs, not a number from a config the artifact may disagree with.
    pub context_length: Option<u64>,
    /// The chat template renders tool calls in a syntax we can scan for. Same probe the generator
    /// uses, so this cannot claim a capability the request path would then refuse.
    pub tools: bool,
    /// The template reads `enable_thinking` -- the reasoning-model kwarg.
    pub thinking: bool,
    /// `model_type` from the checkpoint's `config.json` (`qwen3`, `gemma3`, ...). Ollama keys
    /// `model_info` by architecture, so this is not cosmetic.
    pub arch: String,
}

/// Probed capabilities per model name. A probe reads two small files and renders a template twice;
/// cheap, but a client hits `/api/show` once per model per page load and none of it can change
/// without a restart.
static CACHE: Mutex<Option<HashMap<String, GenerateCapabilities>>> = Mutex::new(None);

/// Resolve a generate model's capabilities from its scenario. `None` for a non-generate model.
pub fn capabilities(root: &std::path::Path, scenario: &str, name: &str) -> Option<GenerateCapabilities> {
    if let Some(hit) = CACHE.lock().ok()?.as_ref().and_then(|m| m.get(name)).cloned() {
        return Some(hit);
    }
    let probed = probe(root, scenario)?;
    if let Ok(mut g) = CACHE.lock() {
        g.get_or_insert_with(HashMap::new).insert(name.to_string(), probed.clone());
    }
    Some(probed)
}

fn probe(root: &std::path::Path, scenario: &str) -> Option<GenerateCapabilities> {
    let sc = npu_engine::config::ScenarioConfig::load(&root.join(scenario)).ok()?;
    if sc.scenario.kind != "generate" {
        return None;
    }
    let mut out = GenerateCapabilities::default();

    // The window the ELF was built at. `dims.S` is the artifact's own record of it; a scenario
    // cannot override it and a config that disagreed would be the wrong number to report.
    let decode_meta = root.join(&sc.artifacts.decode).join("meta.json");
    if let Some(v) = read_json(&decode_meta) {
        out.context_length = v.get("dims").and_then(|d| d.get("S")).and_then(Value::as_u64);
    }

    let tok_dir = root.join(&sc.artifacts.tokenizer_dir);
    if let Some(v) = read_json(&tok_dir.join("config.json")) {
        out.arch = v.get("model_type").and_then(Value::as_str).unwrap_or("llama").to_string();
    }
    if let Some(v) = read_json(&tok_dir.join("tokenizer_config.json")) {
        if let Some(src) = v.get("chat_template").and_then(Value::as_str) {
            out.thinking = src.contains("enable_thinking");
            // The SAME probe the generator gates on. Advertising `tools` off a keyword match would
            // let a client offer a capability the request path then answers 400 for.
            out.tools = ToolSyntax::probe(&ChatTemplate::new(src.to_string())).is_some();
        }
    }
    if out.arch.is_empty() {
        out.arch = "llama".to_string();
    }
    Some(out)
}

fn read_json(p: &std::path::Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
}

/// `GET /api/version`. A client probes this to decide it is talking to an Ollama server at all, so
/// it must answer before anything else here is reachable.
pub fn version_json() -> String {
    json!({ "version": format!("{}-xdna-engine", env!("CARGO_PKG_VERSION")) }).to_string()
}

/// `GET /api/tags`: every model, the way `ollama list` shows them.
///
/// `size` is the model's live device bytes, which is 0 until it loads -- honest, and not the file
/// size a real Ollama would report. `digest` is not a content hash and is not claimed to be one.
pub fn tags_json(status: &[ModelStatus], caps: &dyn Fn(&str) -> Option<GenerateCapabilities>) -> String {
    let models: Vec<Value> = status
        .iter()
        .map(|s| {
            let c = caps(&s.name);
            json!({
                "name": s.name,
                "model": s.name,
                "modified_at": "1970-01-01T00:00:00Z",
                "size": s.bo_bytes,
                "digest": "",
                "details": details_obj(s, c.as_ref()),
            })
        })
        .collect();
    json!({ "models": models }).to_string()
}

fn details_obj(s: &ModelStatus, c: Option<&GenerateCapabilities>) -> Value {
    let family = c.map(|c| c.arch.as_str()).unwrap_or_else(|| s.capability.map(|c| c.0).unwrap_or("unknown"));
    json!({
        "parent_model": "",
        "format": "xclbin",
        "family": family,
        "families": [family],
        "parameter_size": "",
        "quantization_level": "",
    })
}

/// `POST /api/show`: the endpoint a client reads capabilities and context length out of.
///
/// A model with no chat template reports `completion` alone -- it can still generate, it just
/// cannot be told about tools. Reporting `tools` there would make a client offer them and get a 400.
pub fn show_json(name: &str, status: &[ModelStatus], c: Option<&GenerateCapabilities>) -> Option<String> {
    let s = status.iter().find(|s| s.name == name)?;
    let mut capabilities = vec!["completion"];
    if let Some(c) = c {
        if c.tools {
            capabilities.push("tools");
        }
        if c.thinking {
            capabilities.push("thinking");
        }
    }
    let mut model_info = serde_json::Map::new();
    if let Some(c) = c {
        model_info.insert("general.architecture".into(), json!(c.arch));
        if let Some(n) = c.context_length {
            // Ollama keys this by ARCHITECTURE, and a client looks it up under the architecture it
            // read from the same object -- so the two must be derived from one source, as they are.
            model_info.insert(format!("{}.context_length", c.arch), json!(n));
        }
    }
    Some(
        json!({
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": details_obj(s, c),
            "model_info": Value::Object(model_info),
            "capabilities": capabilities,
        })
        .to_string(),
    )
}

/// One NDJSON streaming frame carrying decoded text.
pub fn chat_chunk(model: &str, created: &str, text: &str) -> String {
    json!({
        "model": model,
        "created_at": created,
        "message": { "role": "assistant", "content": text },
        "done": false,
    })
    .to_string()
}

/// One NDJSON frame carrying a tool call.
///
/// `arguments` is an OBJECT here. On `/v1` the same call goes out as a JSON STRING, because that is
/// what each API specifies; a client that parsed the wrong one gets a type error, not a wrong
/// answer, which is the only reason this asymmetry is survivable.
pub fn chat_tool_call_chunk(model: &str, created: &str, call: &npu_engine::ToolCall) -> String {
    json!({
        "model": model,
        "created_at": created,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{ "function": { "name": call.name, "arguments": call.arguments } }],
        },
        "done": false,
    })
    .to_string()
}

/// The terminal frame. A normal object with `done: true` -- there is no `[DONE]` sentinel here.
pub fn chat_done(
    model: &str,
    created: &str,
    reason: npu_engine::FinishReason,
    report: &npu_engine::GenerationReport,
) -> String {
    let s = report.summarize();
    json!({
        "model": model,
        "created_at": created,
        "message": { "role": "assistant", "content": "" },
        "done": true,
        // Ollama's own vocabulary: it says "stop" or "length", and `tool_calls` has no spelling
        // here, so a tool call ends as `stop` and the client reads message.tool_calls instead.
        "done_reason": match reason {
            npu_engine::FinishReason::Length => "length",
            _ => "stop",
        },
        "total_duration": (report.generate_us as u64).saturating_mul(1_000),
        "load_duration": 0,
        "prompt_eval_count": s.prompt_tokens,
        "prompt_eval_duration": (report.prefill.us as u64).saturating_mul(1_000),
        "eval_count": s.completion_tokens,
        "eval_duration": (report.generate_us as u64).saturating_mul(1_000),
    })
    .to_string()
}

/// The buffered (non-streaming) body: one object, `done: true`, carrying the whole message.
pub fn chat_buffered(
    model: &str,
    created: &str,
    text: &str,
    calls: &[npu_engine::ToolCall],
    reason: npu_engine::FinishReason,
    report: &npu_engine::GenerationReport,
) -> String {
    let mut v: Value = serde_json::from_str(&chat_done(model, created, reason, report)).unwrap();
    v["message"]["content"] = json!(text);
    if !calls.is_empty() {
        v["message"]["tool_calls"] = Value::Array(
            calls
                .iter()
                .map(|c| json!({ "function": { "name": c.name, "arguments": c.arguments } }))
                .collect(),
        );
    }
    v.to_string()
}

/// Whether a model is servable through this surface at all. Ollama has no vocabulary for an ASR or
/// embedding model, and listing one would put it in a client's chat picker.
pub fn is_chat_model(s: &ModelStatus) -> bool {
    s.capability.map(|c| c.0) == Some("generate") && s.state != LoadState::Failed
}
