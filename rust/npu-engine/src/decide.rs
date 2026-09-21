//! Typed decisions (TypeSafe's `/v1/systemone`) read off a language model's next-token logits, the
//! way SemIf does it: a fixed system turn, a JSON user turn naming each option by letter, and a
//! softmax over the letters' logits at the last prompt position -- no generation. The reference is
//! `scripts/qwen35_decide_ref.py`, whose prompts are byte-identical to SemIf's own on JevBench.

use serde_json::{json, Value};

use crate::pipeline::ChatMessage;

pub const SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed \
option. Respond with only its uppercase letter, with no explanation or reasoning.";
/// One answer slot per option, in option order; SemIf's own limit.
pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    Noul,
    Choice,
    Score,
}

#[derive(Debug, Clone)]
pub struct DecideQuestion {
    pub id: String,
    pub kind: QuestionKind,
    pub instructions: String,
    /// (key, description) in the order the letters are assigned.
    pub options: Vec<(String, String)>,
}

impl DecideQuestion {
    /// A yes/no question. Missing criteria get SemIf's default wording.
    pub fn noul(id: &str, instructions: &str, when_true: Option<&str>, when_false: Option<&str>) -> Self {
        let opt = |k: &str, d: Option<&str>| (k.to_string(), d.map_or(format!("The proposition is {k}."), str::to_string));
        Self { id: id.into(), kind: QuestionKind::Noul, instructions: instructions.into(),
               options: vec![opt("true", when_true), opt("false", when_false)] }
    }

    /// Options in the caller's order; an empty description falls back to the key.
    pub fn choice(id: &str, instructions: &str, criteria: Vec<(String, String)>) -> Self {
        let options = criteria.into_iter().map(|(k, d)| { let d = if d.is_empty() { k.clone() } else { d }; (k, d) }).collect();
        Self { id: id.into(), kind: QuestionKind::Choice, instructions: instructions.into(), options }
    }

    /// Ordered levels, keyed by index.
    pub fn score(id: &str, instructions: &str, levels: Vec<String>) -> Self {
        let options = levels.into_iter().enumerate().map(|(i, l)| (i.to_string(), l)).collect();
        Self { id: id.into(), kind: QuestionKind::Score, instructions: instructions.into(), options }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(2..=LETTERS.len()).contains(&self.options.len()) {
            return Err(format!("question `{}` has {} options; 2..={} are supported", self.id,
                               self.options.len(), LETTERS.len()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DecideRequest {
    /// The TypeSafe `state`: a string or any JSON value, rendered as SemIf's `evidence`.
    pub evidence: Value,
    pub questions: Vec<DecideQuestion>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecideAnswer {
    Noul { id: String, p_true: f64 },
    Choice { id: String, choice: String, probabilities: Vec<(String, f64)>, confidence: f64 },
    /// `score` is the probability-weighted level index (reflex's reading of an ordinal answer).
    Score { id: String, score: f64, probabilities: Vec<f64>, confidence: f64 },
}

/// What one question cost. `reused_tokens` were already primed when the question started (the
/// shared prefix restored from a snapshot, or a positional cache's ledger hit).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuestionStats {
    pub id: String,
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub batched_tokens: usize,
    pub stepwise_tokens: usize,
    pub restore_us: u64,
    pub prefill_us: u64,
    pub readout_us: u64,
}

/// A decide request's measurements, questions in request order. `shared_prefix_tokens` is 0 when
/// the questions were each primed from scratch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecideStats {
    pub shared_prefix_tokens: usize,
    pub prefix_us: u64,
    pub snapshot_us: u64,
    pub questions: Vec<QuestionStats>,
}

/// One answer per question in request order, and what producing them cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Decisions {
    pub answers: Vec<DecideAnswer>,
    pub stats: DecideStats,
}

/// SemIf's `direct_messages`: the user turn is Python `json.dumps(..., ensure_ascii=False)` of
/// {evidence, criterion, options[{letter, description}]}, each description `"<key>: <desc>"`.
pub fn messages(evidence: &Value, q: &DecideQuestion) -> Vec<ChatMessage> {
    let options: Vec<Value> = q.options.iter().zip(LETTERS.chars())
        .map(|((k, d), l)| json!({"letter": l.to_string(), "description": format!("{k}: {d}")}))
        .collect();
    let payload = json!({"evidence": evidence, "criterion": q.instructions, "options": options});
    vec![ChatMessage::new("system", SYSTEM),
         ChatMessage::new("user", crate::llm::chat_template::json_dumps_py(&payload))]
}

/// The answer from the option letters' logits, in option order. Softmax over those logits only, at
/// temperature 1, uncalibrated -- SemIf's readout. `confidence` is 1 - normalised entropy.
pub fn answer(q: &DecideQuestion, letter_logits: &[f32]) -> DecideAnswer {
    let m = letter_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let w: Vec<f64> = letter_logits.iter().map(|&x| (x as f64 - m).exp()).collect();
    let z: f64 = w.iter().sum();
    let p: Vec<f64> = w.iter().map(|x| x / z).collect();
    let entropy: f64 = p.iter().filter(|&&x| x > 0.0).map(|x| -x * x.ln()).sum();
    let confidence = 1.0 - entropy / (p.len() as f64).ln();
    let best = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let id = q.id.clone();
    match q.kind {
        QuestionKind::Noul => DecideAnswer::Noul { id, p_true: p[0] },
        QuestionKind::Choice => DecideAnswer::Choice {
            id, choice: q.options[best].0.clone(),
            probabilities: q.options.iter().map(|(k, _)| k.clone()).zip(p).collect(), confidence },
        QuestionKind::Score => DecideAnswer::Score {
            id, score: p.iter().enumerate().map(|(i, x)| i as f64 * x).sum(), probabilities: p, confidence },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JevBench `original-policy-01-0`, whose user turn the Python reference renders (and SemIf's
    /// own `direct_messages` renders identically) as the string below.
    #[test]
    fn the_user_turn_is_semifs_json_dumps_byte_for_byte() {
        let q = DecideQuestion::noul(
            "q", "Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied.",
            Some("Every required condition is established and no prohibition applies."),
            Some("A condition is missing or a prohibition applies."));
        let state = json!("Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago but has no receipt. Issue a refund.");
        let m = messages(&state, &q);
        assert_eq!(m[0].content, SYSTEM);
        assert_eq!(m[1].content, concat!(
            r#"{"evidence": "Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago but has no receipt. Issue a refund.", "#,
            r#""criterion": "Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied.", "#,
            r#""options": [{"letter": "A", "description": "true: Every required condition is established and no prohibition applies."}, "#,
            r#"{"letter": "B", "description": "false: A condition is missing or a prohibition applies."}]}"#));
    }

    #[test]
    fn answers_follow_the_letters_softmax() {
        let q = DecideQuestion::choice("c", "pick", vec![("a".into(), "x".into()), ("b".into(), "".into())]);
        assert_eq!(q.options[1].1, "b", "an empty description falls back to the key");
        match answer(&q, &[0.0, (3.0f32).ln()]) {
            DecideAnswer::Choice { choice, probabilities, confidence, .. } => {
                assert_eq!(choice, "b");
                assert!((probabilities[1].1 - 0.75).abs() < 1e-6, "f32 logits");
                assert!(confidence > 0.0 && confidence < 1.0);
            }
            other => panic!("{other:?}"),
        }
        let n = DecideQuestion::noul("n", "?", None, None);
        assert_eq!(n.options[1].1, "The proposition is false.");
        assert!(matches!(answer(&n, &[27.8729, 21.3354]), DecideAnswer::Noul { p_true, .. } if p_true > 0.998));
        let s = DecideQuestion::score("s", "?", vec!["lo".into(), "mid".into(), "hi".into()]);
        match answer(&s, &[0.0, 0.0, 0.0]) {
            DecideAnswer::Score { score, confidence, .. } => {
                assert!((score - 1.0).abs() < 1e-9);
                assert!(confidence.abs() < 1e-9, "a uniform answer has zero confidence");
            }
            other => panic!("{other:?}"),
        }
        assert!(DecideQuestion::score("s", "?", vec!["only".into()]).validate().is_err());
    }
}
