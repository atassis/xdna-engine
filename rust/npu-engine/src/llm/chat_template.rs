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
    /// `tools` is always an empty list -- tool-calling is out of scope for this milestone, so the
    /// template's `{% if tools %}` branch is never taken.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String, EngineError> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(pycompat_method);
        env.add_template("chat", &self.source)
            .map_err(|e| EngineError::Load(format!("chat template parse: {e}")))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| EngineError::Load(format!("chat template lookup: {e}")))?;

        let msgs: Value = messages
            .iter()
            .map(|m| Value::from_iter([("role", Value::from(m.role.clone())), ("content", Value::from(m.content.clone()))]))
            .collect();
        let ctx = context! {
            messages => msgs,
            add_generation_prompt => add_generation_prompt,
            tools => Value::from(Vec::<Value>::new()),
        };
        tmpl.render(ctx).map_err(|e| EngineError::Load(format!("chat template render: {e}")))
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

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.to_string(), content: content.to_string() }
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

    fn qwen3_tokenizer_config_path() -> PathBuf {
        if let Ok(dir) = std::env::var("QWEN3_TOKENIZER_DIR") {
            return PathBuf::from(dir).join("tokenizer_config.json");
        }
        PathBuf::from(
            "/mnt/data/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/\
             c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json",
        )
    }
}
