//! Turn a model's raw completion into content plus tool calls, driven by the [`ToolSyntax`] probed
//! from its template.
//!
//! Two entry points over one scanner: [`parse_completion`] for a buffered response, and
//! [`StreamingToolParser`] for SSE. The streaming one is why this lives in the engine rather than
//! in the wire layer -- it has to HOLD BACK a tail that might be the start of a delimiter. A parser
//! that forwards text as it lands leaks `<tool_` to the client as assistant content and then cannot
//! retract it.
//!
//! Garbage is CONTENT. A call that opens and never closes, or whose body decodes to nothing usable,
//! comes back as text. Inventing a call from a truncated payload is the one outcome worse than not
//! supporting tools at all: the client would execute it. Every such span is also recorded in
//! `rejects`, so a format this server reads wrongly shows up as itself rather than as stray text in
//! an answer.

use crate::llm::tool_syntax::{sample_of, PayloadFormat, ToolSyntax};
use crate::pipeline::ToolCall;

/// How many rejected spans one generation keeps. A model that misformats a call misformats every
/// call in the completion, so the first few carry the whole diagnosis.
const MAX_REJECTS: usize = 4;

/// A completion split into what the user sees and what the client must execute.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedCompletion {
    pub content: String,
    pub calls: Vec<ToolCall>,
    /// Spans that sat between the delimiters and did not decode, truncated. Also in `content`:
    /// this is a diagnostic copy, not a diversion.
    pub rejects: Vec<String>,
}

/// Split a finished completion. Text outside the delimiters is content, in order.
pub fn parse_completion(syn: &ToolSyntax, text: &str) -> ParsedCompletion {
    let mut out = ParsedCompletion::default();
    let mut rest = text;
    while let Some(open_at) = rest.find(&syn.open) {
        let after_open = open_at + syn.open.len();
        let Some(close_rel) = rest[after_open..].find(&syn.close) else {
            // Opened and never closed: the model was cut off mid-call. All of it is content.
            push_reject(&mut out.rejects, &rest[open_at..]);
            break;
        };
        let payload = &rest[after_open..after_open + close_rel];
        match decode_call(syn, payload, out.calls.len()) {
            Some(call) => {
                out.content.push_str(&rest[..open_at]);
                out.calls.push(call);
            }
            // Well-delimited but not a call. Keep the whole span as content rather than dropping
            // text the model meant to say.
            None => {
                push_reject(&mut out.rejects, payload);
                out.content.push_str(&rest[..after_open + close_rel + syn.close.len()]);
            }
        }
        rest = &rest[after_open + close_rel + syn.close.len()..];
    }
    out.content.push_str(rest);
    out
}

fn push_reject(rejects: &mut Vec<String>, span: &str) {
    if rejects.len() < MAX_REJECTS {
        rejects.push(sample_of(span));
    }
}

/// One delimited span -> a call, in whichever shape this template writes them. `index` becomes the
/// id: OpenAI requires one on every call and no model emits it.
pub(crate) fn decode_call(syn: &ToolSyntax, payload: &str, index: usize) -> Option<ToolCall> {
    let (name, arguments) = match &syn.payload {
        PayloadFormat::Json { args_are_string } => decode_json_body(payload, *args_are_string)?,
        PayloadFormat::NamedDsl { quote } => decode_dsl_body(payload, quote)?,
    };
    if name.is_empty() {
        return None;
    }
    Some(ToolCall { id: format!("call_{index}"), name, arguments })
}

/// `{"name": ..., "arguments": ...}`.
fn decode_json_body(payload: &str, args_are_string: bool) -> Option<(String, serde_json::Value)> {
    let v: serde_json::Value = serde_json::from_str(payload.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let raw = v.get("arguments").cloned().unwrap_or_else(|| serde_json::json!({}));
    let arguments = if args_are_string {
        // This template stringifies arguments, so the model writes them stringified too.
        match raw.as_str() {
            Some(s) => serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.to_string())),
            None => raw,
        }
    } else {
        raw
    };
    Some((name, arguments))
}

/// `name{key:value,...}`, Gemma-4's DSL: the name ahead of the body, bare or quoted keys, and
/// strings wrapped in `quote`.
///
/// The format has no escape for `quote`, so a value containing it is ambiguous on the way in and
/// unrepresentable on the way out. Such a body decodes to whatever the first closing `quote` says
/// and the disagreement is unresolvable here -- it has to be fixed in the format.
fn decode_dsl_body(payload: &str, quote: &str) -> Option<(String, serde_json::Value)> {
    let body_at = payload.find('{')?;
    let name = payload[..body_at].trim().to_string();
    let (arguments, rest) = dsl_object(&payload[body_at..], quote)?;
    // Trailing bytes mean the scan ended somewhere the body did not, so the decode is not a decode.
    if !rest.trim().is_empty() {
        return None;
    }
    Some((name, arguments))
}

/// `{` k `:` v (`,` k `:` v)* `}` -> an object, plus whatever follows it.
fn dsl_object<'a>(s: &'a str, q: &str) -> Option<(serde_json::Value, &'a str)> {
    let mut rest = s.strip_prefix('{')?.trim_start();
    let mut map = serde_json::Map::new();
    if let Some(after) = rest.strip_prefix('}') {
        return Some((serde_json::Value::Object(map), after));
    }
    loop {
        let (key, tail) = dsl_key(rest, q)?;
        rest = tail.trim_start().strip_prefix(':')?.trim_start();
        let (value, tail) = dsl_value(rest, q)?;
        map.insert(key, value);
        rest = tail.trim_start();
        match rest.chars().next()? {
            ',' => rest = rest[1..].trim_start(),
            '}' => return Some((serde_json::Value::Object(map), &rest[1..])),
            _ => return None,
        }
    }
}

/// `[` v (`,` v)* `]` -> an array, plus whatever follows it.
fn dsl_array<'a>(s: &'a str, q: &str) -> Option<(serde_json::Value, &'a str)> {
    let mut rest = s.strip_prefix('[')?.trim_start();
    let mut items = Vec::new();
    if let Some(after) = rest.strip_prefix(']') {
        return Some((serde_json::Value::Array(items), after));
    }
    loop {
        let (value, tail) = dsl_value(rest, q)?;
        items.push(value);
        rest = tail.trim_start();
        match rest.chars().next()? {
            ',' => rest = rest[1..].trim_start(),
            ']' => return Some((serde_json::Value::Array(items), &rest[1..])),
            _ => return None,
        }
    }
}

/// A key: `quote`-wrapped when the template escapes keys, bare when it does not. Both spellings
/// come out of the same template -- Gemma-4 quotes keys in a tool DECLARATION and not in a call.
fn dsl_key<'a>(s: &'a str, q: &str) -> Option<(String, &'a str)> {
    if let Some((v, rest)) = dsl_string(s, q) {
        return Some((v, rest));
    }
    let end = s.find(':')?;
    let key = s[..end].trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_string(), &s[end..]))
}

fn dsl_value<'a>(s: &'a str, q: &str) -> Option<(serde_json::Value, &'a str)> {
    if let Some((v, rest)) = dsl_string(s, q) {
        return Some((serde_json::Value::String(v), rest));
    }
    match s.chars().next()? {
        '{' => dsl_object(s, q),
        '[' => dsl_array(s, q),
        _ => {
            let end = s.find([',', '}', ']']).unwrap_or(s.len());
            let raw = s[..end].trim();
            let v = match raw {
                "true" => serde_json::Value::Bool(true),
                "false" => serde_json::Value::Bool(false),
                "null" | "" => serde_json::Value::Null,
                _ => serde_json::from_str(raw).unwrap_or_else(|_| serde_json::json!(raw)),
            };
            Some((v, &s[end..]))
        }
    }
}

fn dsl_string<'a>(s: &'a str, q: &str) -> Option<(String, &'a str)> {
    let inner = s.strip_prefix(q)?;
    let end = inner.find(q)?;
    Some((inner[..end].to_string(), &inner[end + q.len()..]))
}

/// What the streaming parser hands back, in the order the model produced it.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseOut {
    Text(String),
    Call(ToolCall),
}

#[derive(Debug, PartialEq)]
enum State {
    /// Outside a call. `held` is the tail that might be the beginning of `open`.
    Content,
    /// Inside one. `held` is the payload so far.
    InCall,
}

/// Incremental [`parse_completion`]. Feed decoded text as it arrives; take what is safe to release.
pub struct StreamingToolParser {
    syn: ToolSyntax,
    state: State,
    held: String,
    calls: usize,
    rejects: Vec<String>,
}

impl StreamingToolParser {
    pub fn new(syn: &ToolSyntax) -> Self {
        StreamingToolParser {
            syn: syn.clone(),
            state: State::Content,
            held: String::new(),
            calls: 0,
            rejects: Vec::new(),
        }
    }

    /// Spans that arrived delimited as calls and did not decode. See [`ParsedCompletion::rejects`].
    pub fn rejects(&self) -> &[String] {
        &self.rejects
    }

    /// Absorb a chunk and return everything now resolved.
    pub fn push(&mut self, chunk: &str) -> Vec<ParseOut> {
        self.held.push_str(chunk);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Content => {
                    if let Some(at) = self.held.find(&self.syn.open) {
                        let text: String = self.held.drain(..at).collect();
                        if !text.is_empty() {
                            out.push(ParseOut::Text(text));
                        }
                        self.held.drain(..self.syn.open.len());
                        self.state = State::InCall;
                        continue;
                    }
                    // Release everything except the longest tail that could still grow into `open`.
                    let keep = partial_suffix_len(&self.held, &self.syn.open);
                    if keep < self.held.len() {
                        let cut = self.held.len() - keep;
                        let text: String = self.held.drain(..cut).collect();
                        if !text.is_empty() {
                            out.push(ParseOut::Text(text));
                        }
                    }
                    return out;
                }
                State::InCall => {
                    let Some(at) = self.held.find(&self.syn.close) else { return out };
                    let payload: String = self.held.drain(..at).collect();
                    self.held.drain(..self.syn.close.len());
                    match decode_call(&self.syn, &payload, self.calls) {
                        Some(call) => {
                            self.calls += 1;
                            out.push(ParseOut::Call(call));
                        }
                        // Delimited but not a call: hand back exactly the bytes the model wrote.
                        None => {
                            push_reject(&mut self.rejects, &payload);
                            out.push(ParseOut::Text(format!(
                                "{}{payload}{}",
                                self.syn.open, self.syn.close
                            )))
                        }
                    }
                    self.state = State::Content;
                }
            }
        }
    }

    /// Release whatever is still held. An unterminated call is content, matching the buffered path.
    pub fn finish(&mut self) -> Vec<ParseOut> {
        let mut out = Vec::new();
        let held = std::mem::take(&mut self.held);
        let text = match self.state {
            State::Content => held,
            State::InCall => {
                let span = format!("{}{held}", self.syn.open);
                push_reject(&mut self.rejects, &span);
                span
            }
        };
        if !text.is_empty() {
            out.push(ParseOut::Text(text));
        }
        self.state = State::Content;
        out
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `pat`.
///
/// This is the whole hold-back rule. `pat` is short (a delimiter), so the naive scan is right and
/// the KMP version would be harder to read for no measurable gain.
fn partial_suffix_len(s: &str, pat: &str) -> usize {
    let max = pat.len().saturating_sub(1).min(s.len());
    (1..=max)
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && pat.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::chat_template::ChatTemplate;
    use crate::llm::detokenize::IncrementalDetokenizer;
    use crate::llm::tool_syntax::PayloadFormat;
    use crate::pipeline::ChatMessage;
    use std::path::PathBuf;

    fn qwen_syntax() -> ToolSyntax {
        ToolSyntax {
            open: "<tool_call>\n".into(),
            close: "\n</tool_call>".into(),
            payload: PayloadFormat::Json { args_are_string: false },
        }
    }

    /// Gemma-4's shape, written out so the DSL cases do not all depend on the artifact being there.
    fn gemma_syntax() -> ToolSyntax {
        ToolSyntax {
            open: "<|tool_call>call:".into(),
            close: "<tool_call|>".into(),
            payload: PayloadFormat::NamedDsl { quote: "<|\"|>".into() },
        }
    }

    fn real_qwen3_template() -> Option<ChatTemplate> {
        let path = if let Ok(dir) = std::env::var("QWEN3_TOKENIZER_DIR") {
            PathBuf::from(dir).join("tokenizer_config.json")
        } else {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("artifacts/qwen3-0.6b/tokenizer/tokenizer_config.json")
        };
        if !path.exists() {
            eprintln!("SKIP: {} missing", path.display());
            return None;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        Some(ChatTemplate::new(cfg["chat_template"].as_str().unwrap().to_string()))
    }

    /// The strongest test in this file. Render a call through the REAL template, probe the syntax
    /// from that same template, parse the render back, and require the call that comes out to be
    /// the one that went in. Render and parse are inverses descending from one source, so this
    /// catches a probe that is self-consistently WRONG -- which no assertion on literal delimiters
    /// can, because those assertions were written from the same reading of the template.
    #[test]
    fn a_rendered_call_round_trips_through_the_probe() {
        let Some(tmpl) = real_qwen3_template() else { return };
        let syn = ToolSyntax::probe(&tmpl).expect("Qwen3 renders tool calls");
        let call = ToolCall {
            id: "call_0".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({ "city": "Paris", "unit": "c" }),
        };
        let tools = [serde_json::json!({
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } }
        })];
        let prefix = tmpl
            .render_full(&[ChatMessage::new("user", "w?")], true, Some(false), &tools)
            .unwrap();
        let full = tmpl
            .render_full(
                &[
                    ChatMessage::new("user", "w?"),
                    ChatMessage::new("assistant", "").with_tool_calls(vec![call.clone()]),
                ],
                false,
                Some(false),
                &tools,
            )
            .unwrap();
        // What the MODEL would have had to emit, minus the template's own turn terminator.
        let emitted = full[prefix.len()..].replace("<|im_end|>\n", "");

        let parsed = parse_completion(&syn, &emitted);
        assert_eq!(parsed.calls.len(), 1, "content={:?}", parsed.content);
        assert_eq!(parsed.calls[0].name, call.name);
        assert_eq!(parsed.calls[0].arguments, call.arguments);
        assert_eq!(parsed.content, "");
    }

    #[test]
    fn text_before_a_call_stays_content() {
        let out = parse_completion(
            &qwen_syntax(),
            "Let me check.<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
        );
        assert_eq!(out.content, "Let me check.");
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "f");
        assert_eq!(out.calls[0].id, "call_0");
    }

    #[test]
    fn two_calls_in_one_completion_both_parse_and_get_distinct_ids() {
        let out = parse_completion(
            &qwen_syntax(),
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\
             <tool_call>\n{\"name\": \"b\", \"arguments\": {\"y\": 2}}\n</tool_call>",
        );
        assert_eq!(out.calls.len(), 2);
        assert_eq!((out.calls[0].name.as_str(), out.calls[1].name.as_str()), ("a", "b"));
        assert_eq!((out.calls[0].id.as_str(), out.calls[1].id.as_str()), ("call_0", "call_1"));
        assert_eq!(out.content, "");
    }

    /// A model that opens a call and never closes it produced garbage, and garbage is CONTENT.
    /// Fabricating a call from a truncated payload is the one outcome worse than no tool support:
    /// the client executes it.
    #[test]
    fn an_unterminated_call_is_content_not_a_call() {
        let out = parse_completion(&qwen_syntax(), "<tool_call>\n{\"name\": \"f\"");
        assert!(out.calls.is_empty());
        assert!(out.content.contains("<tool_call>"), "content={:?}", out.content);
    }

    /// Well-delimited but not a call. The span stays content rather than being dropped.
    #[test]
    fn a_delimited_non_call_payload_is_content() {
        for payload in ["not json at all", "[1,2,3]", "{\"arguments\": {}}", "{\"name\": 7}"] {
            let text = format!("<tool_call>\n{payload}\n</tool_call>");
            let out = parse_completion(&qwen_syntax(), &text);
            assert!(out.calls.is_empty(), "payload {payload:?} became a call");
            assert_eq!(out.content, text, "payload {payload:?} lost its text");
        }
    }

    /// `arguments` absent entirely -- a zero-argument tool. An empty object, not a refusal.
    #[test]
    fn a_call_with_no_arguments_key_gets_an_empty_object() {
        let out = parse_completion(&qwen_syntax(), "<tool_call>\n{\"name\": \"ping\"}\n</tool_call>");
        assert_eq!(out.calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn stringified_arguments_are_decoded_once_more() {
        let syn = ToolSyntax {
            payload: PayloadFormat::Json { args_are_string: true },
            ..qwen_syntax()
        };
        let out = parse_completion(
            &syn,
            "<tool_call>\n{\"name\": \"f\", \"arguments\": \"{\\\"city\\\": \\\"Paris\\\"}\"}\n</tool_call>",
        );
        assert_eq!(out.calls[0].arguments, serde_json::json!({ "city": "Paris" }));
    }

    // ---- streaming ----------------------------------------------------------------------------

    fn drain(p: &mut StreamingToolParser, chunks: &[&str]) -> (String, Vec<ToolCall>) {
        let (mut text, mut calls) = (String::new(), Vec::new());
        let mut take = |outs: Vec<ParseOut>| {
            for o in outs {
                match o {
                    ParseOut::Text(t) => text.push_str(&t),
                    ParseOut::Call(c) => calls.push(c),
                }
            }
        };
        for c in chunks {
            take(p.push(c));
        }
        take(p.finish());
        (text, calls)
    }

    /// `<tool_call>` arrives across several BPE tokens. A parser that forwards text as it lands
    /// leaks `<tool_` as assistant content and cannot retract it. One BYTE at a time is the worst
    /// case, and nothing may escape until the delimiter resolves.
    #[test]
    fn a_delimiter_split_across_tokens_is_never_leaked() {
        let syn = qwen_syntax();
        let mut p = StreamingToolParser::new(&syn);
        let full = "ok<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>";
        let chunks: Vec<String> = full.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let (text, calls) = drain(&mut p, &refs);
        assert_eq!(text, "ok", "leaked a partial delimiter as content");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }

    /// Hold-back must not swallow text that merely LOOKS like a delimiter start.
    #[test]
    fn a_false_start_is_released_as_content() {
        let syn = qwen_syntax();
        let mut p = StreamingToolParser::new(&syn);
        let (text, calls) = drain(&mut p, &["a<to", "ol_ca", "ke b"]);
        assert_eq!(text, "a<tool_cake b");
        assert!(calls.is_empty());
    }

    /// Streaming and buffered must agree on every split, or a client sees different answers from
    /// the same completion depending on a flag it did not set.
    #[test]
    fn streaming_agrees_with_buffered_at_every_split_point() {
        let syn = qwen_syntax();
        let full = "hi<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>bye";
        let want = parse_completion(&syn, full);
        for cut in 0..full.len() {
            if !full.is_char_boundary(cut) {
                continue;
            }
            let mut p = StreamingToolParser::new(&syn);
            let (text, calls) = drain(&mut p, &[&full[..cut], &full[cut..]]);
            assert_eq!(text, want.content, "content differs at cut {cut}");
            assert_eq!(calls, want.calls, "calls differ at cut {cut}");
        }
    }

    /// The unterminated case again, through the streaming path: `finish` must give the bytes back,
    /// delimiter included, not swallow them.
    #[test]
    fn streaming_finish_releases_an_unterminated_call_as_content() {
        let syn = qwen_syntax();
        let mut p = StreamingToolParser::new(&syn);
        let (text, calls) = drain(&mut p, &["<tool_call>\n{\"name\": \"f\""]);
        assert!(calls.is_empty());
        assert_eq!(text, "<tool_call>\n{\"name\": \"f\"");
    }

    // ---- Gemma-4's named DSL ------------------------------------------------------------------

    #[test]
    fn a_dsl_call_decodes_name_and_string_arguments() {
        let out = parse_completion(
            &gemma_syntax(),
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,unit:<|\"|>c<|\"|>}<tool_call|>",
        );
        assert_eq!(out.calls.len(), 1, "content={:?}", out.content);
        assert_eq!(out.calls[0].name, "get_weather");
        assert_eq!(out.calls[0].arguments, serde_json::json!({ "city": "Paris", "unit": "c" }));
        assert_eq!(out.content, "");
    }

    /// The DSL is untyped on the wire: only the wrapper says a value is a string, so an unwrapped
    /// scalar must not come back quoted and a wrapped digit must not come back as a number.
    #[test]
    fn dsl_scalars_keep_the_type_the_wrapper_gives_them() {
        let out = parse_completion(
            &gemma_syntax(),
            "<|tool_call>call:f{n:3,x:1.5,ok:true,none:null,s:<|\"|>7<|\"|>}<tool_call|>",
        );
        assert_eq!(
            out.calls[0].arguments,
            serde_json::json!({ "n": 3, "x": 1.5, "ok": true, "none": null, "s": "7" })
        );
    }

    #[test]
    fn dsl_nests_objects_and_arrays() {
        let out = parse_completion(
            &gemma_syntax(),
            "<|tool_call>call:f{q:{a:[1,2],b:{c:<|\"|>d<|\"|>}},e:[]}<tool_call|>",
        );
        assert_eq!(
            out.calls[0].arguments,
            serde_json::json!({ "q": { "a": [1, 2], "b": { "c": "d" } }, "e": [] })
        );
    }

    /// A `}` inside a quoted value must not close the body. The JSON balancer gets this via `"`;
    /// the DSL one has to know the template's own quote literal.
    #[test]
    fn a_brace_inside_a_dsl_string_does_not_end_the_call() {
        let out = parse_completion(
            &gemma_syntax(),
            "<|tool_call>call:f{expr:<|\"|>if (x) { y }<|\"|>}<tool_call|>tail",
        );
        assert_eq!(out.calls.len(), 1, "content={:?}", out.content);
        assert_eq!(out.calls[0].arguments, serde_json::json!({ "expr": "if (x) { y }" }));
        assert_eq!(out.content, "tail");
    }

    #[test]
    fn a_dsl_call_with_no_arguments_is_an_empty_object() {
        let out = parse_completion(&gemma_syntax(), "<|tool_call>call:ping{}<tool_call|>");
        assert_eq!(out.calls[0].name, "ping");
        assert_eq!(out.calls[0].arguments, serde_json::json!({}));
    }

    /// Garbage between the delimiters is content in BOTH families, and the DSL decoder must not be
    /// the lenient one -- a half-parsed body is a fabricated call.
    #[test]
    fn a_malformed_dsl_body_is_content() {
        for payload in ["f{a:", "f{a:1", "{a:1}", "f{a:1}trailing"] {
            let text = format!("<|tool_call>call:{payload}<tool_call|>");
            let out = parse_completion(&gemma_syntax(), &text);
            assert!(out.calls.is_empty(), "payload {payload:?} became a call");
            assert_eq!(out.content, text, "payload {payload:?} lost its text");
        }
    }

    // ---- diagnostics --------------------------------------------------------------------------

    /// The reason this exists: a body we read wrongly is otherwise invisible. It reaches the client
    /// as ordinary assistant text, which reads as the model rambling rather than as a parser miss.
    #[test]
    fn an_undecodable_body_is_recorded_as_a_reject() {
        let out = parse_completion(&gemma_syntax(), "<|tool_call>call:f{a:<tool_call|>");
        assert!(out.calls.is_empty());
        assert_eq!(out.rejects.len(), 1);
        assert!(out.rejects[0].contains("f{a:"), "reject={:?}", out.rejects[0]);
    }

    #[test]
    fn an_unterminated_call_is_recorded_as_a_reject_on_both_paths() {
        let syn = qwen_syntax();
        let buffered = parse_completion(&syn, "<tool_call>\n{\"name\": \"f\"");
        assert_eq!(buffered.rejects.len(), 1, "buffered path recorded nothing");

        let mut p = StreamingToolParser::new(&syn);
        drain(&mut p, &["<tool_call>\n{\"name\": \"f\""]);
        assert_eq!(p.rejects().len(), 1, "streaming path recorded nothing");
    }

    /// A completion with no delimiters at all is not a diagnosis waiting to happen.
    #[test]
    fn ordinary_text_records_nothing() {
        let out = parse_completion(&qwen_syntax(), "just an answer");
        assert!(out.rejects.is_empty());
    }

    fn real_gemma4_template() -> Option<ChatTemplate> {
        let path = match std::env::var("GEMMA4_TOKENIZER_DIR") {
            Ok(dir) => PathBuf::from(dir).join("tokenizer_config.json"),
            Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("artifacts/gemma4-12b/tokenizer/tokenizer_config.json"),
        };
        if !path.exists() {
            eprintln!("SKIP: {} missing", path.display());
            return None;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        Some(ChatTemplate::new(cfg["chat_template"].as_str().unwrap().to_string()))
    }

    /// The Qwen round-trip's counterpart on the other payload family, and the reason the DSL
    /// decoder exists: arguments that are not strings, nested, and written by the real template.
    /// Render and parse descend from one source, so this catches a decoder that is self-consistently
    /// wrong -- which the literal-by-literal assertions cannot.
    #[test]
    fn a_rendered_gemma4_call_round_trips_through_the_probe() {
        let Some(tmpl) = real_gemma4_template() else { return };
        let syn = ToolSyntax::probe(&tmpl).expect("Gemma-4 renders tool calls");
        let call = ToolCall {
            id: "call_0".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({
                "city": "Köln",
                "days": 3,
                "metric": true,
                "detail": { "hourly": false },
                "tags": ["a", "b"],
            }),
        };
        let tools = [serde_json::json!({
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } }
        })];
        let full = tmpl
            .render_full(
                &[
                    ChatMessage::new("user", "w?"),
                    ChatMessage::new("assistant", "").with_tool_calls(vec![call.clone()]),
                ],
                false,
                Some(false),
                &tools,
            )
            .unwrap();
        // What the MODEL would have had to write, located by the delimiters the TEMPLATE spells --
        // not by the probed ones, or the test would only be checking the probe against itself.
        let from = full.find("<|tool_call>").expect("template rendered a call");
        let to = full.find("<tool_call|>").expect("template closed the call") + "<tool_call|>".len();
        let parsed = parse_completion(&syn, &full[from..to]);

        assert_eq!(parsed.calls.len(), 1, "content={:?}", parsed.content);
        assert_eq!(parsed.calls[0].name, call.name);
        assert_eq!(parsed.calls[0].arguments, call.arguments);
        assert_eq!(parsed.content, "");
        assert!(parsed.rejects.is_empty());
    }

    /// The chain that actually runs: token ids -> detokenizer -> parser.
    ///
    /// Gemma-4 spells its call in SPECIAL ids, and the ordinary decode deletes every one of them.
    /// The parser is then handed `call:get_weather{city:Paris}` -- no delimiter to find, and a
    /// string it can no longer tell from a number -- so the request answers as though the model had
    /// never called anything. No test above the token level can see that, which is why this one
    /// starts at ids and carries its own negative control.
    #[test]
    fn gemma4_call_tokens_survive_the_detokenizer_and_parse() {
        let dir = PathBuf::from("../../artifacts/gemma4-12b/tokenizer");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: {} missing", dir.display());
            return;
        }
        let cfg = crate::llm::config::ModelConfig::load(&dir).expect("load Gemma-4");
        let syn = cfg.tool_syntax().expect("Gemma-4 renders tool calls").clone();
        assert!(!cfg.tool_special_strip.is_empty(), "its call literals are special ids");

        let enc = |t: &str| cfg.tokenizer.encode(t, false).unwrap().get_ids().to_vec();
        let id = |t: &str| cfg.tokenizer.token_to_id(t).unwrap_or_else(|| panic!("no id for {t}"));
        let mut ids = vec![id("<|tool_call>")];
        ids.extend(enc("call:get_weather{city:"));
        ids.push(id("<|\"|>"));
        ids.extend(enc("Paris"));
        ids.push(id("<|\"|>"));
        ids.extend(enc("}"));
        ids.push(id("<tool_call|>"));

        let decode = |mut d: IncrementalDetokenizer| {
            ids.iter().fold(String::new(), |mut acc, &t| {
                acc.push_str(&d.push(t, &cfg.tokenizer).unwrap());
                acc
            })
        };

        let kept = decode(IncrementalDetokenizer::keeping_special(cfg.tool_special_strip.clone()));
        let parsed = parse_completion(&syn, &kept);
        assert_eq!(parsed.calls.len(), 1, "decoded {kept:?}");
        assert_eq!(parsed.calls[0].name, "get_weather");
        assert_eq!(parsed.calls[0].arguments, serde_json::json!({ "city": "Paris" }));
        assert_eq!(parsed.content, "");

        // The control: the same ids through the ordinary decode produce no call at all.
        let dropped = decode(IncrementalDetokenizer::new());
        assert!(
            parse_completion(&syn, &dropped).calls.is_empty(),
            "special tokens survived a decode that skips them: {dropped:?}"
        );
    }

    /// Keeping specials is all-or-nothing, so the ones that are not part of the syntax have to go
    /// back out by hand -- otherwise turning tools on starts leaking control tokens into answers.
    #[test]
    fn a_control_token_outside_the_syntax_is_stripped_from_the_text() {
        let dir = PathBuf::from("../../artifacts/gemma4-12b/tokenizer");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: {} missing", dir.display());
            return;
        }
        let cfg = crate::llm::config::ModelConfig::load(&dir).expect("load Gemma-4");
        assert!(
            cfg.tool_special_strip.iter().any(|s| s == "<|turn>"),
            "a turn marker is not part of a tool call and must be on the strip list"
        );
        assert!(
            !cfg.tool_special_strip.iter().any(|s| s == "<|tool_call>" || s == "<|\"|>"),
            "the syntax's own literals must never be stripped: {:?}",
            cfg.tool_special_strip
        );

        let id = |t: &str| cfg.tokenizer.token_to_id(t).unwrap();
        let mut d = IncrementalDetokenizer::keeping_special(cfg.tool_special_strip.clone());
        let hi = cfg.tokenizer.encode("hi", false).unwrap().get_ids().to_vec();
        let ids: Vec<u32> = std::iter::once(id("<|turn>")).chain(hi).collect();
        let text = ids.iter().fold(String::new(), |mut acc, &t| {
            acc.push_str(&d.push(t, &cfg.tokenizer).unwrap());
            acc
        });
        assert_eq!(text, "hi");
        // And the deletion leaves a trace. A control token that vanishes without one is how a
        // construct this server does not model reaches a user as leftover text nobody can explain.
        assert_eq!(d.stripped(), ["<|turn>"]);
    }
}
