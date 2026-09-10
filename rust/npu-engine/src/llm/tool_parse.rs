//! Turn a model's raw completion into content plus tool calls, driven by the [`ToolSyntax`] probed
//! from its template.
//!
//! Two entry points over one scanner: [`parse_completion`] for a buffered response, and
//! [`StreamingToolParser`] for SSE. The streaming one is why this lives in the engine rather than
//! in the wire layer -- it has to HOLD BACK a tail that might be the start of a delimiter. A parser
//! that forwards text as it lands leaks `<tool_` to the client as assistant content and then cannot
//! retract it.
//!
//! Garbage is CONTENT. A call that opens and never closes, or whose payload is not an object with a
//! string `name`, comes back as text. Inventing a call from a truncated payload is the one outcome
//! worse than not supporting tools at all: the client would execute it.

use crate::llm::tool_syntax::ToolSyntax;
use crate::pipeline::ToolCall;

/// A completion split into what the user sees and what the client must execute.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParsedCompletion {
    pub content: String,
    pub calls: Vec<ToolCall>,
}

/// Split a finished completion. Text outside the delimiters is content, in order.
pub fn parse_completion(syn: &ToolSyntax, text: &str) -> ParsedCompletion {
    let mut out = ParsedCompletion::default();
    let mut rest = text;
    while let Some(open_at) = rest.find(&syn.open) {
        let after_open = open_at + syn.open.len();
        let Some(close_rel) = rest[after_open..].find(&syn.close) else {
            // Opened and never closed: the model was cut off mid-call. All of it is content.
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
            None => out.content.push_str(&rest[..after_open + close_rel + syn.close.len()]),
        }
        rest = &rest[after_open + close_rel + syn.close.len()..];
    }
    out.content.push_str(rest);
    out
}

/// `{"name": ..., "arguments": ...}` -> a call. `index` becomes the id: OpenAI requires one on
/// every call and no model emits it.
fn decode_call(syn: &ToolSyntax, payload: &str, index: usize) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(payload.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    let raw = v.get("arguments").cloned().unwrap_or_else(|| serde_json::json!({}));
    let arguments = if syn.args_are_string {
        // This template stringifies arguments, so the model writes them stringified too.
        match raw.as_str() {
            Some(s) => serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.to_string())),
            None => raw,
        }
    } else {
        raw
    };
    Some(ToolCall { id: format!("call_{index}"), name, arguments })
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
}

impl StreamingToolParser {
    pub fn new(syn: &ToolSyntax) -> Self {
        StreamingToolParser { syn: syn.clone(), state: State::Content, held: String::new(), calls: 0 }
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
                        None => out.push(ParseOut::Text(format!(
                            "{}{payload}{}",
                            self.syn.open, self.syn.close
                        ))),
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
            State::InCall => format!("{}{held}", self.syn.open),
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
    use crate::pipeline::ChatMessage;
    use std::path::PathBuf;

    fn qwen_syntax() -> ToolSyntax {
        ToolSyntax {
            open: "<tool_call>\n".into(),
            close: "\n</tool_call>".into(),
            args_are_string: false,
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
        let syn = ToolSyntax { args_are_string: true, ..qwen_syntax() };
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
}
