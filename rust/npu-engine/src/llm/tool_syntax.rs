//! Recover a model's tool-call syntax from its own chat template, by rendering a call with known
//! sentinels and reading the delimiters back out.
//!
//! The alternative is vLLM's model: a `--tool-call-parser hermes|llama3_json|mistral|...` registry
//! with a hand-written parser per family. That cannot satisfy "a model we have not seen works",
//! which is the requirement here. llama.cpp derives the format from the template instead; we can go
//! a step further, because the renderer already holds the model's real Jinja and can therefore
//! interrogate it rather than pattern-match its output.
//!
//! What comes back is deliberately small -- an open literal, a close literal, and how `arguments`
//! is encoded. That is the whole of what a parser needs for the delimiter-wrapped-JSON family, which
//! covers Qwen, Hermes and most of what ships. A template outside it (Mistral's `[TOOL_CALLS]`
//! control token, GPT-OSS harmony channels) probes to `None`, and the request layer keeps answering
//! `tools` with a 400 for that model. An honest 400 beats a guessed syntax: a wrong guess does not
//! fail, it fabricates a call the model never made.

use crate::llm::chat_template::ChatTemplate;
use crate::pipeline::{ChatMessage, ToolCall};

/// How one model writes a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSyntax {
    /// Literal that opens a call. Qwen3: `"<tool_call>\n"`.
    pub open: String,
    /// Literal that closes it. Qwen3: `"\n</tool_call>"`.
    pub close: String,
    /// Whether the template renders `arguments` as a JSON *string* rather than a nested object.
    /// Both forms exist -- Qwen3's own template branches on `tool_call.arguments is string` -- and
    /// the parser has to undo whichever one this model uses.
    pub args_are_string: bool,
}

/// Two independent sentinel sets. Deliberately unlike each other in length and characters, and
/// containing no JSON metacharacter: a delimiter that survives both is a delimiter, not an artefact
/// of one payload.
const SENTINELS: [(&str, &str, &str); 2] = [
    ("Zqf_name_A", "Zqf_key_A", "Zqf_val_A"),
    ("Wbnprobe_name_BB", "Wbnprobe_key_BB", "Wbnprobe_val_BB"),
];

impl ToolSyntax {
    /// `None` when this template cannot render a tool call, or renders one whose delimiters depend
    /// on the payload. Both are "we do not support tools on this model", which is a supported answer.
    pub fn probe(tmpl: &ChatTemplate) -> Option<ToolSyntax> {
        let a = probe_once(tmpl, SENTINELS[0])?;
        let b = probe_once(tmpl, SENTINELS[1])?;
        // Delimiters that move with the payload are not delimiters.
        if a != b {
            return None;
        }
        Some(a)
    }
}

fn probe_once(tmpl: &ChatTemplate, (name, key, val): (&str, &str, &str)) -> Option<ToolSyntax> {
    let tools = [serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": "probe",
            "parameters": { "type": "object", "properties": {} }
        }
    })];
    let user = ChatMessage::new("user", "probe");

    // `prefix` is what the model is handed before it writes anything; everything after it in
    // `called` is output the model itself would have had to produce. Thinking is switched OFF for
    // both so a reasoning template does not inject a `<think>` block into one and not the other.
    let prefix = tmpl.render_full(std::slice::from_ref(&user), true, Some(false), &tools).ok()?;
    let called = tmpl
        .render_full(
            &[
                user,
                ChatMessage::new("assistant", "").with_tool_calls(vec![ToolCall {
                    id: "probe".into(),
                    name: name.into(),
                    arguments: serde_json::json!({ key: val }),
                }]),
            ],
            false,
            Some(false),
            &tools,
        )
        .ok()?;

    let shared = common_prefix_len(&prefix, &called);
    let tail = called.get(shared..)?;
    // The LAST occurrence: the tools block near the top of the prompt also names the function, and
    // that copy is before `shared` only when the template puts tools in the system turn. Searching
    // from the end lands on the call itself either way.
    let name_at = tail.rfind(name)?;
    let obj_start = tail.get(..name_at)?.rfind('{')?;
    let obj_end = balanced_object_end(tail, obj_start)?;

    let open = tail.get(..obj_start)?.to_string();
    let close = close_literal(tail.get(obj_end..)?);
    // A template that renders the call with no delimiter on either side leaves nothing to scan for,
    // and a parser with an empty `open` would report a tool call in every completion.
    if open.trim().is_empty() && close.trim().is_empty() {
        return None;
    }
    let payload = tail.get(obj_start..obj_end)?;
    Some(ToolSyntax { open, close, args_are_string: args_rendered_as_string(payload, key) })
}

/// Bytes shared by both strings, truncated to a char boundary so the result is always sliceable.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let n = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes())
        .take_while(|(x, y)| x == y)
        .count();
    let mut n = n;
    while n > 0 && !a.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// Byte index one past the `}` that closes the object starting at `start`.
///
/// String- and escape-aware. A brace counter that is not gets `{"a": "}"}`, `{"a": "\\"}` and a
/// nested object wrong -- and the arguments payload is exactly where a `}` inside a string shows up.
fn balanced_object_end(s: &str, start: usize) -> Option<usize> {
    let b = s.as_bytes();
    if b.get(start) != Some(&b'{') {
        return None;
    }
    let (mut depth, mut in_str, mut escaped) = (0usize, false, false);
    for (i, &c) in b.iter().enumerate().skip(start) {
        if in_str {
            match c {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// The closing delimiter, cut before the turn terminator the TEMPLATE appends.
///
/// The model emits its own EOS; `<|im_end|>\n` and friends belong to the template, not to the call
/// syntax, and scanning for them in a completion would never match.
fn close_literal(rest: &str) -> String {
    match rest.find("<|") {
        Some(i) => rest[..i].to_string(),
        None => rest.to_string(),
    }
}

/// Whether this template stringified `arguments` instead of nesting it. With a nested object the
/// key appears as a bare `"key"`; stringified, its quotes are escaped (`\"key\"`).
fn args_rendered_as_string(payload: &str, key: &str) -> bool {
    !payload.contains(&format!("\"{key}\"")) && payload.contains(&format!("\\\"{key}\\\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn real_qwen3_template() -> Option<ChatTemplate> {
        let path = qwen3_tokenizer_config_path();
        if !path.exists() {
            eprintln!("SKIP: {} missing", path.display());
            return None;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        Some(ChatTemplate::new(cfg["chat_template"].as_str().unwrap().to_string()))
    }

    fn qwen3_tokenizer_config_path() -> PathBuf {
        if let Ok(dir) = std::env::var("QWEN3_TOKENIZER_DIR") {
            return PathBuf::from(dir).join("tokenizer_config.json");
        }
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("artifacts/qwen3-0.6b/tokenizer/tokenizer_config.json")
    }

    #[test]
    fn probe_recovers_qwen3_delimiters() {
        let Some(tmpl) = real_qwen3_template() else { return };
        let syn = ToolSyntax::probe(&tmpl).expect("Qwen3 renders tool calls");
        assert_eq!(syn.open, "<tool_call>\n");
        assert_eq!(syn.close, "\n</tool_call>");
        assert!(!syn.args_are_string, "Qwen3 nests arguments; it does not stringify them");
    }

    /// The probe must not depend on WHICH sentinel it used. A template whose delimiters vary with
    /// the payload is not a delimiter grammar, and half-supporting it fabricates calls.
    #[test]
    fn probe_is_stable_across_sentinels() {
        let Some(tmpl) = real_qwen3_template() else { return };
        let a = probe_once(&tmpl, SENTINELS[0]).unwrap();
        let b = probe_once(&tmpl, SENTINELS[1]).unwrap();
        assert_eq!(a, b);
    }

    /// A template with no tool branch is tool-INCAPABLE, and saying so is the honest answer: the
    /// request layer keeps returning 400 for `tools` on that model rather than guessing a syntax.
    #[test]
    fn probe_fails_closed_on_a_template_with_no_tool_branch() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}".to_string(),
        );
        assert!(ToolSyntax::probe(&tmpl).is_none());
    }

    /// A template that renders the call as a bare JSON object with nothing around it. `open` empty
    /// would make every completion look like a tool call, so this fails closed too.
    #[test]
    fn probe_fails_closed_when_there_are_no_delimiters() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             {{ {'name': c.function.name, 'arguments': c.function.arguments} | tojson }}\
             {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}"
                .to_string(),
        );
        assert!(ToolSyntax::probe(&tmpl).is_none());
    }

    /// A stringified-arguments template, which Qwen3's own `{%- if tool_call.arguments is string %}`
    /// branch shows is a real shape. The flag is what tells the parser to JSON-decode twice.
    #[test]
    fn probe_detects_stringified_arguments() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             [CALL]{{ {'name': c.function.name, 'arguments': (c.function.arguments | tojson)} | tojson }}[/CALL]\
             {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}"
                .to_string(),
        );
        let syn = ToolSyntax::probe(&tmpl).expect("delimited");
        assert_eq!(syn.open, "[CALL]");
        assert_eq!(syn.close, "[/CALL]");
        assert!(syn.args_are_string, "arguments came back as a JSON string, not an object");
    }

    /// A naive brace counter gets every one of these wrong, and the arguments payload is exactly
    /// where a `}` inside a string turns up.
    #[test]
    fn balanced_object_end_is_string_and_escape_aware() {
        let cases = [
            (r#"{"a": "}"}"#, 10),
            (r#"{"a": {"b": 1}}"#, 15),
            (r#"{"a": "\""}"#, 11),
            (r#"{"a": "\\"}"#, 11),
        ];
        for (s, want) in cases {
            assert_eq!(balanced_object_end(s, 0), Some(want), "input {s}");
            assert_eq!(&s[..want], s, "consumed the whole object for {s}");
        }
        assert_eq!(balanced_object_end(r#"{"a": 1"#, 0), None, "unterminated");
        assert_eq!(balanced_object_end("nope", 0), None, "not an object");
    }

    /// The trailing turn terminator belongs to the TEMPLATE, not to the call syntax: the model
    /// emits its own EOS, so scanning a completion for `<|im_end|>` would never match.
    #[test]
    fn close_literal_stops_at_the_turn_terminator() {
        assert_eq!(close_literal("\n</tool_call><|im_end|>\n"), "\n</tool_call>");
        assert_eq!(close_literal("\n</tool_call>"), "\n</tool_call>");
    }

    #[test]
    fn common_prefix_len_never_splits_a_char() {
        assert_eq!(common_prefix_len("aöb", "aöc"), 3);
        assert_eq!(common_prefix_len("aöb", "axb"), 1);
        assert_eq!(common_prefix_len("", "x"), 0);
    }
}
