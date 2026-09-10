//! Jinja2 chat-template rendering (minijinja) for `Prompt::Chat`. Kept as DATA on [`ChatTemplate`]
//! (the Jinja source string), never inlined in the decode loop, so a second model is a
//! `ChatTemplate::new(source)` call, not a rewrite.
//!
//! HF chat templates lean on Python `str` METHODS (`content.strip('\n')`, `.split(...)`, ...) that
//! vanilla Jinja/minijinja expose only as filters (`content|trim`), not as object methods. Rather
//! than pull in `minijinja-contrib` (unavailable in this offline build) for its `pycompat` shim,
//! [`pycompat_method`] implements exactly the handful Qwen3's template calls, via minijinja's own
//! `unknown_method_callback` hook -- the same extension point `pycompat` uses.

use minijinja::value::Value;
use minijinja::{context, Environment, Error, ErrorKind, State};

use crate::api::EngineError;
use crate::pipeline::ChatMessage;

/// A model's chat template: the raw Jinja source from `tokenizer_config.json`'s `chat_template`.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    source: String,
}

impl ChatTemplate {
    pub fn new(source: String) -> Self {
        ChatTemplate { source }
    }

    /// Render `messages`, appending the assistant-turn opener when `add_generation_prompt`.
    /// No tools, no `enable_thinking`.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String, EngineError> {
        self.render_with(messages, add_generation_prompt, None)
    }

    /// `render` plus `enable_thinking`, the template kwarg the reasoning-model families read.
    /// `None` leaves the variable UNDEFINED rather than passing a default: Qwen3's template tests
    /// `enable_thinking is defined and enable_thinking is false`, so a defaulted `true` and an
    /// absent variable are the same prompt, and only `false` is a real instruction.
    pub fn render_with(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String, EngineError> {
        self.render_full(messages, add_generation_prompt, enable_thinking, &[])
    }

    /// `render_with` plus the declared `tools`. The narrower entry points delegate here.
    ///
    /// `tools` was hardcoded to an empty list until 2026-09-10, which made every chat template's
    /// tool half unreachable. It is the CLIENT's JSON passed through untouched -- key order
    /// included, because the rendered block is part of the cached prefix and reordering it moves
    /// the prompt (measured: 74 of 159 tokens shared when one schema's keys were re-sorted).
    pub fn render_full(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
        tools: &[serde_json::Value],
    ) -> Result<String, EngineError> {
        let mut env = env();
        env.add_template("chat", &self.source)
            .map_err(|e| EngineError::Load(format!("chat template parse: {e}")))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| EngineError::Load(format!("chat template lookup: {e}")))?;

        let msgs: Value = messages.iter().map(message_value).collect();
        let tools = Value::from_serialize(tools);
        let ctx = match enable_thinking {
            Some(t) => context! {
                messages => msgs,
                add_generation_prompt => add_generation_prompt,
                tools => tools,
                enable_thinking => t,
            },
            None => context! {
                messages => msgs,
                add_generation_prompt => add_generation_prompt,
                tools => tools,
            },
        };
        tmpl.render(ctx).map_err(|e| EngineError::Load(format!("chat template render: {e}")))
    }

    /// Render an arbitrary context against this template. The conversation path goes through
    /// [`render_with`](Self::render_with), which is the only caller that knows the variable names a
    /// chat template expects; this exists for the tool-syntax probe and for exercising one filter in
    /// isolation.
    pub(crate) fn render_probe(&self, ctx: minijinja::value::Value) -> Result<String, EngineError> {
        let mut env = env();
        env.add_template("chat", &self.source)
            .map_err(|e| EngineError::Load(format!("chat template parse: {e}")))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| EngineError::Load(format!("chat template lookup: {e}")))?;
        tmpl.render(ctx).map_err(|e| EngineError::Load(format!("chat template render: {e}")))
    }
}

/// One message as the template sees it.
///
/// `tool_calls` and `tool_call_id` are emitted only when they carry something. That is not tidiness:
/// templates test `{%- if message.tool_calls %}`, and an empty list is falsy in Jinja but a `None`
/// tool_call_id serialised as null is NOT -- emitting the keys unconditionally makes our render
/// diverge from `transformers` on exactly the turns tool calling depends on.
fn message_value(m: &ChatMessage) -> Value {
    let mut fields = vec![
        ("role", Value::from(m.role.clone())),
        ("content", Value::from(m.content.clone())),
    ];
    if !m.tool_calls.is_empty() {
        let calls: Vec<serde_json::Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                serde_json::json!({
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                })
            })
            .collect();
        fields.push(("tool_calls", Value::from_serialize(&calls)));
    }
    if let Some(id) = &m.tool_call_id {
        fields.push(("tool_call_id", Value::from(id.clone())));
    }
    Value::from_iter(fields)
}

/// The one environment every render uses. Built per call because `add_template` borrows the source.
fn env<'a>() -> Environment<'a> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(pycompat_method);
    // `transformers` renders `tojson` as `json.dumps(..., ensure_ascii=False)` -- a space after
    // every `:` and `,`. minijinja's builtin is `serde_json::to_string`, which emits neither, so the
    // SAME template produces a different prompt here than in transformers, vLLM, llama.cpp or
    // Ollama. Measured 2026-09-10 on one Qwen3 tool schema: 60 tokens against HF's 82, diverging at
    // token 2. Neither side is wrong alone -- compact JSON is JSON, and the defect lives only in
    // their disagreement, which is why nothing caught it until `tools` stopped being hardcoded `[]`.
    env.add_filter("tojson", |v: Value| -> Result<String, Error> {
        let json: serde_json::Value = serde_json::to_value(&v).map_err(|e| {
            Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(e)
        })?;
        Ok(json_dumps_py(&json))
    });
    env
}

/// `json.dumps(obj, ensure_ascii=False)`: `", "` between items, `": "` after a key, insertion order
/// preserved (both `serde_json` and `minijinja` need their `preserve_order` feature for that, and
/// both have it -- a schema re-sorted on either hop moves the prompt).
///
/// Written out rather than configured: `serde_json`'s `Formatter` cannot express this separator pair
/// without a custom impl either way, and non-ASCII already passes through as UTF-8, which is exactly
/// what `ensure_ascii=False` means.
fn json_dumps_py(v: &serde_json::Value) -> String {
    let mut s = String::new();
    write_dumps(v, &mut s);
    s
}

fn write_dumps(v: &serde_json::Value, s: &mut String) {
    use std::fmt::Write;
    match v {
        serde_json::Value::Object(m) => {
            s.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let _ = write!(s, "{}: ", serde_json::Value::String(k.clone()));
                write_dumps(val, s);
            }
            s.push('}');
        }
        serde_json::Value::Array(a) => {
            s.push('[');
            for (i, val) in a.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                write_dumps(val, s);
            }
            s.push(']');
        }
        // Scalars: `serde_json`'s own Display is already `json.dumps`-compatible, escaping included.
        other => {
            let _ = write!(s, "{other}");
        }
    }
}

/// Python `str` methods called as OBJECT methods (`x.strip('\n')`) by HF chat templates. Anything
/// else falls through to minijinja's normal "unknown method" error.
fn pycompat_method(_state: &State, value: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let s = match value.as_str() {
        Some(s) => s,
        None => return Err(Error::from(ErrorKind::UnknownMethod)),
    };
    match method {
        "startswith" => Ok(Value::from(s.starts_with(arg_str(args, 0, method)?.as_str()))),
        "endswith" => Ok(Value::from(s.ends_with(arg_str(args, 0, method)?.as_str()))),
        "split" => {
            let sep = arg_str(args, 0, method)?;
            Ok(s.split(sep.as_str()).map(Value::from).collect())
        }
        "strip" => Ok(Value::from(py_strip(s, args, true, true)?)),
        "lstrip" => Ok(Value::from(py_strip(s, args, true, false)?)),
        "rstrip" => Ok(Value::from(py_strip(s, args, false, true)?)),
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

fn arg_str(args: &[Value], i: usize, method: &str) -> Result<String, Error> {
    args.get(i)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, format!("{method}() needs a string argument")))
}

/// Python `str.strip([chars])` semantics: `chars` (default whitespace) is a SET of characters to
/// trim from either end, not a substring -- `"a\n\nb".rstrip("\n")` removes both trailing `\n`s.
/// Qwen3's template only ever passes a single-character set (`'\n'`), so this distinction from a
/// substring-strip never actually bites here, but it is the correct semantics to implement.
fn py_strip(s: &str, args: &[Value], left: bool, right: bool) -> Result<String, Error> {
    let chars: Option<String> = match args.first() {
        Some(v) if !v.is_undefined() && !v.is_none() => Some(
            v.as_str()
                .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "strip() arg must be a string"))?
                .to_string(),
        ),
        _ => None,
    };
    let is_trim_char = |c: char| match &chars {
        Some(set) => set.contains(c),
        None => c.is_whitespace(),
    };
    let mut out = s;
    if left {
        out = out.trim_start_matches(is_trim_char);
    }
    if right {
        out = out.trim_end_matches(is_trim_char);
    }
    Ok(out.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// minijinja's `tojson` is `serde_json::to_string` -- COMPACT. `transformers` renders the same
    /// filter as `json.dumps(..., ensure_ascii=False)`, which puts a space after every `:` and `,`.
    /// Every HF chat template that renders a tool calls `tojson`, for the schema and again for the
    /// call's arguments, so without the override the model sees a tools block no other server
    /// produces: measured 60 tokens against HF's 82 on one schema, diverging at token 2.
    ///
    /// The key order in the expected string is the INSERTION order, not the alphabetical one. It
    /// holds only because `serde_json` and `minijinja` both carry `preserve_order`; drop either and
    /// this reads `{"a": [1, 2], "b": 1}`.
    #[test]
    fn tojson_matches_python_json_dumps_separators_and_order() {
        let tmpl = ChatTemplate::new("{{ x | tojson }}".to_string());
        let out = tmpl
            .render_probe(minijinja::context! { x => serde_json::json!({"b": 1, "a": [1, 2]}) })
            .unwrap();
        assert_eq!(out, r#"{"b": 1, "a": [1, 2]}"#);
    }

    /// Non-ASCII passes through as UTF-8 rather than as `\uXXXX`, which is what `ensure_ascii=False`
    /// means and what every HF template is rendered with.
    #[test]
    fn tojson_does_not_escape_non_ascii() {
        let tmpl = ChatTemplate::new("{{ x | tojson }}".to_string());
        let out = tmpl
            .render_probe(minijinja::context! { x => serde_json::json!({"city": "Köln"}) })
            .unwrap();
        assert_eq!(out, r#"{"city": "Köln"}"#);
    }

    fn weather_tool() -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        })
    }

    /// The real Qwen3 template's `{% if tools %}` branch, unreachable while `tools` was hardcoded
    /// to an empty list. The expected substring carries `json.dumps` spacing AND insertion key
    /// order, so this is also the regression test for both `preserve_order` features.
    #[test]
    fn real_qwen3_template_renders_the_tools_block() {
        let path = qwen3_tokenizer_config_path();
        if !path.exists() {
            eprintln!("SKIP: {} missing", path.display());
            return;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let tmpl = ChatTemplate::new(cfg["chat_template"].as_str().unwrap().to_string());
        let tools = [weather_tool()];
        let out = tmpl
            .render_full(&[msg("user", "Weather in Paris?")], true, None, &tools)
            .unwrap();

        assert!(out.contains("# Tools"), "tools branch not taken:\n{out}");
        assert!(
            out.contains(r#"{"type": "function", "function": {"name": "get_weather", "#),
            "tool schema is not json.dumps-shaped:\n{out}"
        );
        assert!(out.contains("<tool_call>"), "call-format instructions missing:\n{out}");
    }

    /// An assistant turn that CALLED a tool, and the tool turn answering it. Both are message
    /// shapes the template branches on and neither was representable before 2026-09-10.
    #[test]
    fn real_qwen3_template_renders_a_tool_call_turn_and_its_result() {
        let path = qwen3_tokenizer_config_path();
        if !path.exists() {
            eprintln!("SKIP: {} missing", path.display());
            return;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let tmpl = ChatTemplate::new(cfg["chat_template"].as_str().unwrap().to_string());
        let tools = [weather_tool()];
        let convo = [
            msg("user", "Weather in Paris?"),
            ChatMessage::new("assistant", "").with_tool_calls(vec![crate::pipeline::ToolCall {
                id: "call_0".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({ "city": "Paris" }),
            }]),
            ChatMessage::new("tool", r#"{"temp_c": 14}"#).with_tool_call_id("call_0"),
        ];
        let out = tmpl.render_full(&convo, true, None, &tools).unwrap();

        assert!(
            out.contains(r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#),
            "call turn not rendered as the template documents:\n{out}"
        );
        assert!(
            out.contains("<tool_response>\n{\"temp_c\": 14}\n</tool_response>"),
            "tool result not wrapped:\n{out}"
        );
    }

    /// A message with no tool calls must not emit the key at all. Templates test
    /// `{%- if message.tool_calls %}`, and a present-but-empty list changes nothing in Jinja but a
    /// present-and-null `tool_call_id` is truthy -- so emitting unconditionally would diverge from
    /// `transformers` on ordinary turns, not just tool ones.
    #[test]
    fn a_plain_turn_emits_neither_tool_key() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{{ 'HAS' if m.tool_calls is defined else 'NONE' }}\
             {{ 'ID' if m.tool_call_id is defined else 'NOID' }}{% endfor %}"
                .to_string(),
        );
        let out = tmpl.render(&[msg("user", "hi")], false).unwrap();
        assert_eq!(out, "NONENOID");
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::new(role, content)
    }

    #[test]
    fn plain_template_renders_loop_and_concatenation() {
        let tmpl = ChatTemplate::new(
            "{%- for m in messages -%}[{{ m.role }}:{{ m.content }}]{%- endfor -%}\
             {%- if add_generation_prompt -%}<gen>{%- endif -%}"
                .to_string(),
        );
        let out = tmpl.render(&[msg("user", "hi"), msg("assistant", "yo")], true).unwrap();
        assert_eq!(out, "[user:hi][assistant:yo]<gen>");
    }

    #[test]
    fn pycompat_strip_split_startswith_match_python() {
        // Exercises exactly the methods Qwen3's template calls, standalone.
        let tmpl = ChatTemplate::new(
            "{%- set c = messages[0].content -%}\
             {{- c.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') -}}\
             |{{- c.split('</think>')[-1].lstrip('\\n') -}}\
             |{%- if c.startswith('<think>') -%}yes{%- else -%}no{%- endif -%}\
             |{%- if c.endswith('nope') -%}yes{%- else -%}no{%- endif -%}"
                .to_string(),
        );
        let out = tmpl.render(&[msg("assistant", "<think>\nreasoning\n</think>\n\nfinal")], false).unwrap();
        assert_eq!(out, "reasoning|final|yes|no");
    }

    #[test]
    fn namespace_and_reversed_slice_are_supported() {
        // `{% set ns.x = ... %}` inside a `messages[::-1]` loop -- the exact shape Qwen3 uses to
        // find the last user-turn index.
        let tmpl = ChatTemplate::new(
            "{%- set ns = namespace(last=-1) -%}\
             {%- for m in messages[::-1] -%}\
                 {%- set idx = (messages|length - 1) - loop.index0 -%}\
                 {%- if ns.last == -1 and m.role == 'user' -%}{%- set ns.last = idx -%}{%- endif -%}\
             {%- endfor -%}\
             {{- ns.last -}}"
                .to_string(),
        );
        let out = tmpl.render(&[msg("user", "a"), msg("assistant", "b"), msg("user", "c")], false).unwrap();
        assert_eq!(out, "2");
    }

    /// Real Qwen3-0.6B `chat_template`, byte-for-byte the string in `tokenizer_config.json`,
    /// against a Python `jinja2` oracle (`trim_blocks=True, lstrip_blocks=True`, matching
    /// `transformers.apply_chat_template`'s environment). Covers: system+user with a generation
    /// prompt; a mid-conversation assistant turn whose `<think>...</think>` block is stripped
    /// WITHOUT reinjection (not the last turn); a bare raw user turn with no generation prompt.
    #[test]
    fn real_qwen3_template_matches_python_jinja2_oracle() {
        let path = qwen3_tokenizer_config_path();
        if !path.exists() {
            eprintln!("SKIP: {} missing -- `huggingface-cli download Qwen/Qwen3-0.6B` recreates it", path.display());
            return;
        }
        let txt = std::fs::read_to_string(&path).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(&txt).unwrap();
        let src = cfg["chat_template"].as_str().unwrap().to_string();
        let tmpl = ChatTemplate::new(src);

        let case1 = [msg("system", "You are a helpful assistant."), msg("user", "What is 2+2?")];
        let out1 = tmpl.render(&case1, true).unwrap();
        assert_eq!(
            out1,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );

        let case2 = [
            msg("user", "Hi"),
            msg("assistant", "<think>\nThinking a bit\n</think>\n\nHello there!"),
            msg("user", "How are you?"),
        ];
        let out2 = tmpl.render(&case2, true).unwrap();
        assert_eq!(
            out2,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello there!<|im_end|>\n\
             <|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
        );

        let case3 = [msg("user", "Bare raw")];
        let out3 = tmpl.render(&case3, false).unwrap();
        assert_eq!(out3, "<|im_start|>user\nBare raw<|im_end|>\n");
    }

    /// The same tokenizer dir `scenarios/generate-qwen3-0.6b.toml` names, so the test and the
    /// shipped scenario cannot point at different snapshots of the template they both depend on.
    fn qwen3_tokenizer_config_path() -> PathBuf {
        if let Ok(dir) = std::env::var("QWEN3_TOKENIZER_DIR") {
            return PathBuf::from(dir).join("tokenizer_config.json");
        }
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap()
            .join("artifacts/qwen3-0.6b/tokenizer/tokenizer_config.json")
    }

    /// `enable_thinking` is the one template kwarg that changes what the model DOES, and leaving it
    /// unset is not neutral: Qwen3 then reasons, and at the 256-token default it spends the whole
    /// budget in `<think>` and never emits an answer. Measured over HTTP 2026-09-05 on
    /// "What is 2+2?": 256 completion tokens, `finish_reason: length`, no `</think>` at all.
    ///
    /// The template's own test is `enable_thinking is defined and enable_thinking is false`, so
    /// this also pins the asymmetry: `None` and `Some(true)` MUST render identically, and only
    /// `Some(false)` may differ. A defaulted `true` would have looked like it worked.
    #[test]
    fn enable_thinking_false_is_the_only_value_that_changes_the_qwen3_prompt() {
        let path = qwen3_tokenizer_config_path();
        if !path.exists() {
            eprintln!("SKIP: {} missing -- `huggingface-cli download Qwen/Qwen3-0.6B` recreates it", path.display());
            return;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let tmpl = ChatTemplate::new(cfg["chat_template"].as_str().unwrap().to_string());
        let msgs = [msg("user", "What is 2+2?")];

        let unset = tmpl.render_with(&msgs, true, None).unwrap();
        let on = tmpl.render_with(&msgs, true, Some(true)).unwrap();
        let off = tmpl.render_with(&msgs, true, Some(false)).unwrap();

        assert_eq!(unset, on, "an undefined kwarg and an explicit true are the same prompt");
        assert_ne!(unset, off, "enable_thinking=false must change the prompt");
        assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                "thinking-off pre-fills an empty think block, got {off:?}");
        assert!(!on.contains("<think>"), "thinking-on leaves the block to the model, got {on:?}");
    }
}
