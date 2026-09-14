//! Recover a model's tool-call syntax from its own chat template, by rendering a call with known
//! sentinels and reading the delimiters back out.
//!
//! The alternative is vLLM's model: a `--tool-call-parser hermes|llama3_json|mistral|...` registry
//! with a hand-written parser per family. That cannot satisfy "a model we have not seen works",
//! which is the requirement here. llama.cpp derives the format from the template instead; we can go
//! a step further, because the renderer already holds the model's real Jinja and can therefore
//! interrogate it rather than pattern-match its output.
//!
//! Two payload shapes are recognised, and which one a template uses is itself probed rather than
//! configured: the name INSIDE a JSON object (Qwen, Hermes) and the name OUTSIDE a brace-delimited
//! key-value body (Gemma-4). A template outside both -- Mistral's `[TOOL_CALLS]` control token,
//! GPT-OSS harmony channels -- probes to `None`, and the request layer answers `tools` with a 400
//! for that model. An honest 400 beats a guessed syntax: a wrong guess does not fail, it fabricates
//! a call the model never made.
//!
//! Every outcome carries a [`ProbeReason`] and a SAMPLE of what the template actually rendered, so
//! an unrecognised format is read off the refusal rather than reconstructed from its symptoms.

use crate::llm::chat_template::ChatTemplate;
use crate::llm::tool_parse::decode_call;
use crate::pipeline::{ChatMessage, ToolCall};

/// How one model writes a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSyntax {
    /// Literal that opens a call. Qwen3: `"<tool_call>\n"`. Gemma-4: `"<|tool_call>call:"`.
    pub open: String,
    /// Literal that closes it. Qwen3: `"\n</tool_call>"`. Gemma-4: `"<tool_call|>"`.
    pub close: String,
    /// How the span between them encodes the name and the arguments.
    pub payload: PayloadFormat,
}

/// What sits between [`ToolSyntax::open`] and [`ToolSyntax::close`].
///
/// The delimiters and the body are probed independently because they vary independently: the
/// scanner only needs the two literals, and only the decoder needs to know the body is not JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadFormat {
    /// `{"name": ..., "arguments": ...}` -- one JSON object carrying both.
    Json {
        /// The template renders `arguments` as a JSON *string* rather than a nested object. Both
        /// forms exist -- Qwen3's own template branches on `tool_call.arguments is string` -- and
        /// the decoder has to undo whichever one this model uses.
        args_are_string: bool,
    },
    /// `name{key:value,...}`: the name before the body, bare keys, and strings wrapped in `quote`
    /// rather than `"`. Gemma-4 renders `call:get_weather{city:<|"|>Paris<|"|>}`.
    NamedDsl {
        /// What wraps a string value. A sentinel token in Gemma-4's vocabulary, not a quote mark,
        /// and the format gives it no escape -- so a value containing it is unrepresentable on the
        /// way out and unparseable on the way back.
        quote: String,
    },
}

impl PayloadFormat {
    /// Machine-readable family name, for a metadata surface.
    pub fn name(&self) -> &'static str {
        match self {
            PayloadFormat::Json { .. } => "delimited-json",
            PayloadFormat::NamedDsl { .. } => "named-dsl",
        }
    }
}

/// The probe's verdict, with enough of the render attached to diagnose a `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolProbe {
    /// `None` means this model is tool-incapable as far as this server is concerned.
    pub syntax: Option<ToolSyntax>,
    pub reason: ProbeReason,
    /// What the template rendered for a call whose name and arguments were known, truncated to
    /// [`SAMPLE_LIMIT`]. Empty only when the template failed to render at all.
    pub sample: String,
}

/// Why [`ToolSyntax::probe`] answered as it did. Each variant names a distinct place the recovery
/// stopped, because "unsupported" alone sends the next reader back to the template with no clue
/// which half of it to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeReason {
    /// A syntax was recovered and it parsed its own render back.
    Recovered,
    /// The template raised while rendering. Nothing was learned about its tool support.
    RenderFailed,
    /// The rendered call does not contain the function name: the template has no tool branch.
    NoToolBranch,
    /// The name is there, but no `{...}` body sits on either side of it.
    NoArgumentBody,
    /// A body was found in a shape neither payload format describes.
    UnknownPayload,
    /// Nothing delimits the call, so every completion would look like one.
    NoDelimiters,
    /// The recovered syntax does not parse the render it was recovered from.
    SelfCheckFailed,
    /// The delimiters moved when the payload changed, so they are not delimiters.
    SentinelsDisagree,
}

impl ProbeReason {
    /// Machine-readable, for a metadata surface. Kebab-case to match the capability vocabulary.
    pub fn as_str(self) -> &'static str {
        match self {
            ProbeReason::Recovered => "recovered",
            ProbeReason::RenderFailed => "render-failed",
            ProbeReason::NoToolBranch => "no-tool-branch",
            ProbeReason::NoArgumentBody => "no-argument-body",
            ProbeReason::UnknownPayload => "unknown-payload",
            ProbeReason::NoDelimiters => "no-delimiters",
            ProbeReason::SelfCheckFailed => "self-check-failed",
            ProbeReason::SentinelsDisagree => "sentinels-disagree",
        }
    }

    /// What a reader has to go and look at.
    pub fn explain(self) -> &'static str {
        match self {
            ProbeReason::Recovered => "the template's tool-call syntax was recovered",
            ProbeReason::RenderFailed => "the chat template raised while rendering a tool call",
            ProbeReason::NoToolBranch => "the chat template has no tool-call branch",
            ProbeReason::NoArgumentBody => "the rendered call has no {...} argument body",
            ProbeReason::UnknownPayload => "the rendered call's body matches no format this server can parse",
            ProbeReason::NoDelimiters => "the rendered call carries no delimiters to scan for",
            ProbeReason::SelfCheckFailed => "the recovered syntax could not parse the call it was recovered from",
            ProbeReason::SentinelsDisagree => "the rendered delimiters change with the payload",
        }
    }
}

/// Two independent sentinel sets. Deliberately unlike each other in length and characters, and
/// containing no JSON metacharacter: a delimiter that survives both is a delimiter, not an artefact
/// of one payload.
const SENTINELS: [(&str, &str, &str); 2] = [
    ("Zqf_name_A", "Zqf_key_A", "Zqf_val_A"),
    ("Wbnprobe_name_BB", "Wbnprobe_key_BB", "Wbnprobe_val_BB"),
];

/// Assistant content for the ordinary-turn baseline. Starts with a letter on purpose: a baseline
/// whose first differing byte matches the call's eats that byte off the front of `open`.
const TEXT_BASELINE: &str = "Zqfplain";

/// How much of a render a [`ToolProbe`] carries. Long enough to show a delimiter and the head of a
/// body, short enough to sit in an HTTP error.
const SAMPLE_LIMIT: usize = 240;

impl ToolSyntax {
    /// `None` when this template cannot render a tool call, or renders one in a shape this server
    /// cannot parse. Use [`probe_report`](Self::probe_report) when the reason matters.
    pub fn probe(tmpl: &ChatTemplate) -> Option<ToolSyntax> {
        Self::probe_report(tmpl).syntax
    }

    /// The same recovery, with the verdict and a sample of the render attached.
    pub fn probe_report(tmpl: &ChatTemplate) -> ToolProbe {
        let a = probe_once(tmpl, SENTINELS[0]);
        let b = probe_once(tmpl, SENTINELS[1]);
        match (&a.syntax, &b.syntax) {
            (Some(x), Some(y)) if x == y => a,
            // Delimiters that move with the payload are not delimiters.
            (Some(_), Some(_)) => ToolProbe {
                syntax: None,
                reason: ProbeReason::SentinelsDisagree,
                sample: a.sample,
            },
            // One set failed: its reason is the specific one, so report that rather than a generic
            // disagreement.
            (None, _) => a,
            (_, None) => b,
        }
    }
}

fn probe_once(tmpl: &ChatTemplate, (name, key, val): (&str, &str, &str)) -> ToolProbe {
    let tools = [serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": "probe",
            "parameters": { "type": "object", "properties": {} }
        }
    })];
    let user = ChatMessage::new("user", "probe");
    let expect = serde_json::json!({ key: val });

    // Thinking is switched OFF in every render so a reasoning template does not inject a `<think>`
    // block into one and not another.
    let render = |msgs: &[ChatMessage], generation_prompt: bool| {
        tmpl.render_full(msgs, generation_prompt, Some(false), &tools).ok()
    };

    let called = render(
        &[
            user.clone(),
            ChatMessage::new("assistant", "").with_tool_calls(vec![ToolCall {
                id: "probe".into(),
                name: name.into(),
                arguments: expect.clone(),
            }]),
        ],
        false,
    );
    let Some(called) = called else {
        return ToolProbe { syntax: None, reason: ProbeReason::RenderFailed, sample: String::new() };
    };

    // Two baselines, and the EARLIER divergence wins, because `open` must start where the MODEL's
    // output starts. The generation-prompt baseline alone is one byte short on Gemma-4: its
    // assistant opener ends `<channel|>` and its call opens `<|tool_call>`, so the shared `<` is
    // consumed and the scanner would hunt a delimiter the model never writes. An ordinary-text
    // assistant turn diverges at the right place because the template branches there.
    let baselines = [
        render(&[user.clone(), ChatMessage::new("assistant", TEXT_BASELINE)], false),
        render(std::slice::from_ref(&user), true),
    ];
    let shared = baselines
        .iter()
        .flatten()
        .map(|b| common_prefix_len(b, &called))
        .min();
    let Some(shared) = shared else {
        return ToolProbe {
            syntax: None,
            reason: ProbeReason::RenderFailed,
            sample: sample_of(&called),
        };
    };
    let tail = called.get(shared..).unwrap_or("");
    let sample = sample_of(tail);
    let fail = |reason| ToolProbe { syntax: None, reason, sample: sample.clone() };

    // The LAST occurrence: the tools block near the top of the prompt also names the function, and
    // that copy is before `shared` only when the template puts tools in the system turn. Searching
    // from the end lands on the call itself either way.
    let Some(name_at) = tail.rfind(name) else { return fail(ProbeReason::NoToolBranch) };

    // Which side of the name the argument body sits on IS the family: a JSON payload wraps the
    // name, a named-DSL payload follows it.
    let wrapping = tail[..name_at]
        .rfind('{')
        .and_then(|i| balanced_object_end(tail, i).map(|e| (i, e)))
        .filter(|&(_, e)| e > name_at);

    let (body_start, body_end, payload) = match wrapping {
        Some((start, end)) => (
            start,
            end,
            PayloadFormat::Json { args_are_string: args_rendered_as_string(&tail[start..end], key) },
        ),
        None => {
            let after = name_at + name.len();
            if tail.as_bytes().get(after) != Some(&b'{') {
                return fail(ProbeReason::NoArgumentBody);
            }
            let Some(quote) = string_quote(&tail[after..], val) else {
                return fail(ProbeReason::UnknownPayload);
            };
            let Some(end) = balanced_dsl_end(tail, after, &quote) else {
                return fail(ProbeReason::UnknownPayload);
            };
            (name_at, end, PayloadFormat::NamedDsl { quote })
        }
    };

    let open = tail[..body_start].to_string();
    let close = close_literal(tail.get(body_end..).unwrap_or(""));
    // A call with nothing on one side of it cannot be scanned for: an empty `open` reports a tool
    // call in every completion, an empty `close` never ends one.
    if open.trim().is_empty() || close.trim().is_empty() {
        return fail(ProbeReason::NoDelimiters);
    }

    let syntax = ToolSyntax { open, close, payload };
    // The recovery must parse the render it came FROM. A syntax that is self-consistently wrong --
    // right delimiters, misread body -- satisfies every assertion written from the same reading of
    // the template, and this is the one check that does not share that reading.
    match decode_call(&syntax, &tail[body_start..body_end], 0) {
        Some(c) if c.name == name && c.arguments == expect => {
            ToolProbe { syntax: Some(syntax), reason: ProbeReason::Recovered, sample }
        }
        _ => fail(ProbeReason::SelfCheckFailed),
    }
}

/// Bytes shared by both strings, truncated to a char boundary so the result is always sliceable.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes())
        .take_while(|(x, y)| x == y)
        .count();
    while n > 0 && !a.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// Byte index one past the `}` that closes the JSON object starting at `start`.
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

/// [`balanced_object_end`] for a body whose strings are wrapped in `quote` and carry no escape.
/// A `}` inside a quoted value must not close the body, which is the only thing depth counting
/// needs to know here.
fn balanced_dsl_end(s: &str, start: usize, quote: &str) -> Option<usize> {
    let (b, q) = (s.as_bytes(), quote.as_bytes());
    if b.get(start) != Some(&b'{') || q.is_empty() {
        return None;
    }
    let (mut i, mut depth) = (start, 0usize);
    while i < b.len() {
        if b[i..].starts_with(q) {
            let rel = find_bytes(&b[i + q.len()..], q)?;
            i += q.len() + rel + q.len();
            continue;
        }
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// What this template wraps a string value in, read off a value the probe knows is a string.
///
/// Symmetric by construction: the same literal has to close the span it opened, or nothing can tell
/// where a value ends. `"` satisfies that, and so does Gemma-4's `<|"|>`. A structural character
/// disqualifies a candidate -- those belong to the body, not to its strings.
fn string_quote(body: &str, val: &str) -> Option<String> {
    let at = body.find(val)?;
    let (before, after) = (&body[..at], &body[at + val.len()..]);
    const MAX_QUOTE: usize = 8;
    (1..=MAX_QUOTE.min(before.len()))
        .rev()
        .filter(|&n| before.is_char_boundary(before.len() - n))
        .map(|n| &before[before.len() - n..])
        .find(|q| after.starts_with(*q) && !q.contains([',', ':', '{', '}', '[', ']']))
        .map(str::to_string)
}

/// The closing delimiter, cut before the turn terminator the TEMPLATE appends.
///
/// The model emits its own EOS; `<|im_end|>\n` and Gemma-4's `<|tool_response>` belong to the
/// template, not to the call syntax, and scanning a completion for them would never match. The cut
/// point is the first `<|` PAST the start, so a close literal that itself opens with `<|` survives.
fn close_literal(rest: &str) -> String {
    match rest.get(1..).and_then(|r| r.find("<|")) {
        Some(i) => rest[..i + 1].to_string(),
        None => rest.to_string(),
    }
}

/// Whether this template stringified `arguments` instead of nesting it. With a nested object the
/// key appears as a bare `"key"`; stringified, its quotes are escaped (`\"key\"`).
fn args_rendered_as_string(payload: &str, key: &str) -> bool {
    !payload.contains(&format!("\"{key}\"")) && payload.contains(&format!("\\\"{key}\\\""))
}

/// A readable excerpt of a render. Escaped, because the interesting bytes in a delimiter are the
/// newlines and the sample has to survive a JSON error body and a terminal intact.
pub(crate) fn sample_of(s: &str) -> String {
    let mut n = SAMPLE_LIMIT.min(s.len());
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    let mut out: String = s[..n].escape_debug().to_string();
    if n < s.len() {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A template straight out of a model directory, with the special tokens it substitutes.
    fn template_from(dir_env: &str, fallback: &str) -> Option<ChatTemplate> {
        let path = match std::env::var(dir_env) {
            Ok(dir) => PathBuf::from(dir).join("tokenizer_config.json"),
            Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join(fallback),
        };
        if !path.exists() {
            eprintln!("SKIP: {} missing", path.display());
            return None;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let tok = |k: &str| {
            cfg.get(k).and_then(|v| {
                v.as_str().map(str::to_string).or_else(|| {
                    v.get("content").and_then(|c| c.as_str()).map(str::to_string)
                })
            })
        };
        Some(
            ChatTemplate::new(cfg["chat_template"].as_str().unwrap().to_string())
                .with_special_tokens(tok("bos_token"), tok("eos_token")),
        )
    }

    fn real_qwen3_template() -> Option<ChatTemplate> {
        template_from("QWEN3_TOKENIZER_DIR", "artifacts/qwen3-0.6b/tokenizer/tokenizer_config.json")
    }

    fn real_gemma4_template() -> Option<ChatTemplate> {
        template_from("GEMMA4_TOKENIZER_DIR", "artifacts/gemma4-12b/tokenizer/tokenizer_config.json")
    }

    #[test]
    fn probe_recovers_qwen3_delimiters() {
        let Some(tmpl) = real_qwen3_template() else { return };
        let syn = ToolSyntax::probe(&tmpl).expect("Qwen3 renders tool calls");
        assert_eq!(syn.open, "<tool_call>\n");
        assert_eq!(syn.close, "\n</tool_call>");
        assert_eq!(
            syn.payload,
            PayloadFormat::Json { args_are_string: false },
            "Qwen3 nests arguments in a JSON body; it does not stringify them"
        );
    }

    /// Gemma-4 is the case the JSON-only recovery could not see: the name sits BEFORE the body, the
    /// body is not JSON, and strings are wrapped in a vocabulary token rather than a quote mark.
    #[test]
    fn probe_recovers_gemma4_named_dsl() {
        let Some(tmpl) = real_gemma4_template() else { return };
        let report = ToolSyntax::probe_report(&tmpl);
        let syn = report.syntax.as_ref().unwrap_or_else(|| {
            panic!("Gemma-4 renders tool calls; probe said {}: {}", report.reason.as_str(), report.sample)
        });
        assert_eq!(syn.open, "<|tool_call>call:");
        assert_eq!(syn.close, "<tool_call|>");
        assert_eq!(syn.payload, PayloadFormat::NamedDsl { quote: "<|\"|>".into() });
    }

    /// The probe must not depend on WHICH sentinel it used. A template whose delimiters vary with
    /// the payload is not a delimiter grammar, and half-supporting it fabricates calls.
    #[test]
    fn probe_is_stable_across_sentinels() {
        for tmpl in [real_qwen3_template(), real_gemma4_template()].into_iter().flatten() {
            let a = probe_once(&tmpl, SENTINELS[0]);
            let b = probe_once(&tmpl, SENTINELS[1]);
            assert_eq!(a.syntax, b.syntax);
            assert_eq!(a.reason, ProbeReason::Recovered, "{}", a.sample);
        }
    }

    /// A template with no tool branch is tool-INCAPABLE, and saying so is the honest answer: the
    /// request layer keeps returning 400 for `tools` on that model rather than guessing a syntax.
    #[test]
    fn probe_fails_closed_on_a_template_with_no_tool_branch() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}".to_string(),
        );
        let report = ToolSyntax::probe_report(&tmpl);
        assert!(report.syntax.is_none());
        assert_eq!(report.reason, ProbeReason::NoToolBranch);
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
        let report = ToolSyntax::probe_report(&tmpl);
        assert!(report.syntax.is_none());
        assert_eq!(report.reason, ProbeReason::NoDelimiters);
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
        assert_eq!(syn.payload, PayloadFormat::Json { args_are_string: true });
    }

    /// Every refusal has to carry what the template actually wrote. Without it the next reader has
    /// a 400 and a model name, and has to re-derive the format by hand to learn anything.
    #[test]
    fn a_refusal_carries_the_render_that_caused_it() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             CALL {{ c.function.name }} WITH {{ c.function.arguments | tojson }} END\
             {% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}"
                .to_string(),
        );
        let report = ToolSyntax::probe_report(&tmpl);
        assert!(report.syntax.is_none(), "a space-separated call is not a delimiter grammar");
        assert!(
            report.sample.contains("Zqf_name_A"),
            "the sample must show the unrecognised render, got {:?}",
            report.sample
        );
        assert_ne!(report.reason.explain(), ProbeReason::Recovered.explain());
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

    /// The same rule for a body whose strings are wrapped in a multi-byte literal instead of `"`.
    #[test]
    fn balanced_dsl_end_ignores_braces_inside_a_quoted_span() {
        let q = "<|\"|>";
        let s = "{a:<|\"|>}}}<|\"|>,b:{c:1}}TAIL";
        let end = balanced_dsl_end(s, 0, q).expect("balanced");
        assert_eq!(&s[..end], "{a:<|\"|>}}}<|\"|>,b:{c:1}}");
        assert_eq!(balanced_dsl_end("{a:<|\"|>oops", 0, q), None, "unterminated string");
        assert_eq!(balanced_dsl_end("{a:1", 0, q), None, "unterminated body");
    }

    #[test]
    fn string_quote_reads_the_wrapper_off_a_known_string_value() {
        assert_eq!(string_quote("{k:<|\"|>V<|\"|>}", "V").as_deref(), Some("<|\"|>"));
        assert_eq!(string_quote(r#"{"k": "V"}"#, "V").as_deref(), Some("\""));
        // Asymmetric wrappers are not wrappers: nothing could say where a value ends.
        assert_eq!(string_quote("{k:[V]}", "V"), None);
    }

    /// The trailing turn terminator belongs to the TEMPLATE, not to the call syntax: the model
    /// emits its own EOS, so scanning a completion for `<|im_end|>` would never match.
    #[test]
    fn close_literal_stops_at_the_turn_terminator() {
        assert_eq!(close_literal("\n</tool_call><|im_end|>\n"), "\n</tool_call>");
        assert_eq!(close_literal("\n</tool_call>"), "\n</tool_call>");
        // Gemma-4 opens the response slot right after the call.
        assert_eq!(close_literal("<tool_call|><|tool_response>"), "<tool_call|>");
        // A close literal that itself starts `<|` keeps its own opener.
        assert_eq!(close_literal("<|/tool_call|><|im_end|>"), "<|/tool_call|>");
    }

    #[test]
    fn common_prefix_len_never_splits_a_char() {
        assert_eq!(common_prefix_len("aöb", "aöc"), 3);
        assert_eq!(common_prefix_len("aöb", "axb"), 1);
        assert_eq!(common_prefix_len("", "x"), 0);
    }

    #[test]
    fn sample_of_truncates_on_a_char_boundary_and_says_so() {
        let long = "ö".repeat(SAMPLE_LIMIT);
        let s = sample_of(&long);
        assert!(s.ends_with("..."), "{s}");
        assert_eq!(sample_of("a\nb"), "a\\nb", "control characters must stay readable");
    }
}
