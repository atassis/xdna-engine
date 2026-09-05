//! Incremental detokenization and stop-sequence matching over a token stream.
//!
//! BPE tokens are not UTF-8 boundaries: a single new token can complete, or start, a multi-byte
//! codepoint. `tokenizers`' byte-level decoder recovers a truncated codepoint via lossy UTF-8
//! conversion, which surfaces as a trailing U+FFFD (replacement character) -- that is the signal
//! [`IncrementalDetokenizer`] waits on before emitting.

use crate::api::EngineError;

/// Decodes token ids to text. Implemented for `tokenizers::Tokenizer`; a trait so tests can inject a
/// fake decoder without building a real BPE vocabulary.
pub trait Detokenize {
    fn decode_ids(&self, ids: &[u32]) -> Result<String, EngineError>;
}

impl Detokenize for tokenizers::Tokenizer {
    fn decode_ids(&self, ids: &[u32]) -> Result<String, EngineError> {
        self.decode(ids, true).map_err(|e| EngineError::Load(format!("detokenize: {e}")))
    }
}

/// Streams text out of a growing token sequence, buffering across a token boundary that splits a
/// multi-byte UTF-8 codepoint. Re-decodes the whole sequence each push (the crate's own
/// `Tokenizer::decode` is already effectively O(n) in sequence length, and generation lengths here
/// are modest); a windowed decode is a future optimisation, not a correctness requirement.
pub struct IncrementalDetokenizer {
    ids: Vec<u32>,
    emitted: usize, // byte length of the decoded text already emitted
}

impl IncrementalDetokenizer {
    pub fn new() -> Self {
        IncrementalDetokenizer { ids: Vec::new(), emitted: 0 }
    }

    /// Push one new token id; returns the newly-completed text (may be empty -- that is correct
    /// while a multi-byte codepoint is still pending across a later token).
    pub fn push<D: Detokenize>(&mut self, tok: u32, d: &D) -> Result<String, EngineError> {
        self.ids.push(tok);
        let text = d.decode_ids(&self.ids)?;
        if text.ends_with('\u{FFFD}') || text.len() <= self.emitted {
            return Ok(String::new());
        }
        let new_text = text[self.emitted..].to_string();
        self.emitted = text.len();
        Ok(new_text)
    }
}

impl Default for IncrementalDetokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// What [`StopMatcher::feed`] releases: text safe to hand to the sink now, or the text emitted
/// right before a matched stop string (generation must end after this).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopFeed {
    Emit(String),
    Matched(String),
}

/// Detects a `stop` string that may straddle two detokenizer chunks: buffers only as much tail as
/// could still complete the longest configured stop string, and releases the rest immediately.
pub struct StopMatcher {
    stops: Vec<String>,
    buf: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), buf: String::new() }
    }

    fn max_len(&self) -> usize {
        self.stops.iter().map(|s| s.len()).max().unwrap_or(0)
    }

    /// Feed newly-decoded text.
    pub fn feed(&mut self, text: &str) -> StopFeed {
        if self.stops.is_empty() {
            return StopFeed::Emit(text.to_string());
        }
        self.buf.push_str(text);
        // The EARLIEST-occurring match across all configured stops wins, not the first stop in the
        // list that happens to match anywhere -- two stop strings can both be present with the
        // later-listed one occurring first in the text.
        let earliest = self.stops.iter().filter_map(|s| self.buf.find(s.as_str())).min();
        if let Some(pos) = earliest {
            let before = self.buf[..pos].to_string();
            self.buf.clear();
            return StopFeed::Matched(before);
        }
        // No match yet: release everything except a tail long enough to still complete the
        // longest stop string on a LATER feed, so a match spanning two feeds is never missed.
        let keep = self.max_len().saturating_sub(1);
        if self.buf.len() <= keep {
            return StopFeed::Emit(String::new());
        }
        let want_split = self.buf.len() - keep;
        let split = (0..=want_split).rev().find(|&i| self.buf.is_char_boundary(i)).unwrap_or(0);
        let released = self.buf[..split].to_string();
        self.buf.drain(..split);
        StopFeed::Emit(released)
    }

    /// Whatever is left buffered when generation ends with no stop string ever matching.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A fake `Detokenize` that returns a SCRIPTED decode per prefix length, simulating exactly
    /// the lossy-UTF8 signal (`tokenizers`' byte-level decoder inserting U+FFFD for a truncated
    /// multi-byte codepoint) without needing a real BPE vocabulary.
    struct Scripted(Vec<&'static str>);
    impl Detokenize for Scripted {
        fn decode_ids(&self, ids: &[u32]) -> Result<String, EngineError> {
            Ok(self.0[ids.len() - 1].to_string())
        }
    }

    #[test]
    fn buffers_through_a_pending_replacement_char_then_emits_the_full_codepoint() {
        // 3 pushes decode as: "x" + partial (FFFD) + partial (FFFD) + complete "x€".
        let d = Scripted(vec!["x", "x\u{FFFD}", "x\u{FFFD}", "x\u{20AC}"]);
        let mut dt = IncrementalDetokenizer::new();
        assert_eq!(dt.push(0, &d).unwrap(), "x");
        assert_eq!(dt.push(0, &d).unwrap(), "", "pending codepoint must emit nothing, not garbage");
        assert_eq!(dt.push(0, &d).unwrap(), "");
        assert_eq!(dt.push(0, &d).unwrap(), "\u{20AC}", "must emit exactly the newly-completed suffix");
    }

    #[test]
    fn ordinary_tokens_emit_immediately() {
        let d = Scripted(vec!["hel", "hello", "hello world"]);
        let mut dt = IncrementalDetokenizer::new();
        assert_eq!(dt.push(0, &d).unwrap(), "hel");
        assert_eq!(dt.push(0, &d).unwrap(), "lo");
        assert_eq!(dt.push(0, &d).unwrap(), " world");
    }

    /// Real Qwen3-0.6B tokenizer, real split: U+10FFFD (4-byte UTF-8) encodes to FOUR separate
    /// byte-level tokens on this vocabulary (verified directly against this exact tokenizer file),
    /// so decoding any strict prefix of them is genuinely lossy.
    #[test]
    fn real_qwen3_tokenizer_splits_a_rare_codepoint_across_tokens() {
        let path = qwen3_tokenizer_path();
        if !path.exists() {
            eprintln!("SKIP: {} missing -- `huggingface-cli download Qwen/Qwen3-0.6B`", path.display());
            return;
        }
        let tok = tokenizers::Tokenizer::from_file(&path).unwrap();
        let rare = char::from_u32(0x10FFFD).unwrap();
        let s = format!("x{rare}y");
        let ids = tok.encode(s.as_str(), false).unwrap().get_ids().to_vec();
        assert!(ids.len() >= 4, "expected the rare codepoint to split into multiple tokens, got {ids:?}");

        let mut dt = IncrementalDetokenizer::new();
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&dt.push(id, &tok).unwrap());
        }
        assert_eq!(out, s, "incremental output must equal a one-shot decode of the full sequence");
    }

    fn qwen3_tokenizer_path() -> PathBuf {
        let dir = std::env::var("QWEN3_TOKENIZER_DIR").unwrap_or_else(|_| {
            "/mnt/data/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/\
             c1899de289a04d12100db370d81485cdf75e47ca"
                .to_string()
        });
        PathBuf::from(dir).join("tokenizer.json")
    }

    #[test]
    fn stop_string_split_across_two_feeds_is_still_caught() {
        // "STOP" is 4 bytes, so 3 bytes of tail (" ST") must be retained across the feed boundary
        // for the match on the second feed to be found at all.
        let mut m = StopMatcher::new(vec!["STOP".to_string()]);
        assert_eq!(m.feed("hello ST"), StopFeed::Emit("hello".to_string()));
        assert_eq!(m.feed("OP world"), StopFeed::Matched(" ".to_string()));
    }

    #[test]
    fn stop_match_entirely_within_one_feed() {
        let mut m = StopMatcher::new(vec!["abc".to_string()]);
        assert_eq!(m.feed("xxabcyy"), StopFeed::Matched("xx".to_string()));
    }

    #[test]
    fn no_stop_configured_emits_everything_immediately() {
        let mut m = StopMatcher::new(vec![]);
        assert_eq!(m.feed("anything"), StopFeed::Emit("anything".to_string()));
    }

    #[test]
    fn unmatched_tail_is_returned_by_flush() {
        let mut m = StopMatcher::new(vec!["ZZZZ".to_string()]);
        assert_eq!(m.feed("hello ZZ"), StopFeed::Emit("hello".to_string()));
        assert_eq!(m.flush(), " ZZ");
    }

    #[test]
    fn multiple_stop_strings_the_earliest_match_wins() {
        let mut m = StopMatcher::new(vec!["world".to_string(), "hello".to_string()]);
        assert_eq!(m.feed("say hello world"), StopFeed::Matched("say ".to_string()));
    }

    #[test]
    fn split_point_never_lands_inside_a_utf8_codepoint() {
        // "€" is 3 bytes; with a 4-char stop string the retained tail is 3 bytes -- pushing the
        // split point back to a char boundary must not panic or corrupt the euro sign.
        let mut m = StopMatcher::new(vec!["WXYZ".to_string()]);
        match m.feed("€€€€") {
            StopFeed::Emit(s) => assert!(s.chars().all(|c| c == '€'), "must not slice mid-codepoint: {s:?}"),
            other => panic!("unexpected {other:?}"),
        }
    }
}
