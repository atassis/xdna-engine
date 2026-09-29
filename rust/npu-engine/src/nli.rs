//! openjev's NLI readout: a 3-way head over the last token's final-normed hidden. Prompts,
//! decision hypotheses and long-premise windows mirror `scripts/nli_common.py`, which defines them;
//! `tests/refs/nli/common_golden.json` pins the two together.

use std::path::Path;

use crate::api::EngineError;
use crate::decide::{semif_noul_default, DecideQuestion, DecideStats, QuestionKind};

pub const DECIDE_HYPOTHESIS: &str = "The answer to \"{instr}\" is {label}: {crit}";
pub const RERANK_HYPOTHESIS: &str = "The correct answer is: {text}";

/// Python's `str.strip()`: Rust's `trim` misses U+001C..U+001F, which `str.isspace` counts.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// `template.format(premise=.., hypothesis=..)` over stripped fields, one substitution each, so a
/// premise that contains the text `{hypothesis}` is not substituted into.
pub fn prompt(template: &str, premise: &str, hypothesis: &str) -> Result<String, EngineError> {
    let bad = || EngineError::Load(format!("NLI template {template:?} needs one {{premise}} before one {{hypothesis}}"));
    let (pre, rest) = template.split_once("{premise}").ok_or_else(bad)?;
    let (mid, post) = rest.split_once("{hypothesis}").ok_or_else(bad)?;
    Ok(format!("{pre}{}{mid}{}{post}", py_strip(premise), py_strip(hypothesis)))
}

/// (label, hypothesis) per option, in the question's option order. Noul options are
/// `[true, false]` here and read as JevBench's `yes`/`no`, whose default rubric is `Yes`/`No`
/// where SemIf's is `The proposition is true.`.
pub fn decide_hypotheses(q: &DecideQuestion) -> Vec<(String, String)> {
    let instr = py_strip(&q.instructions);
    q.options.iter().map(|(k, d)| {
        let (label, crit) = match q.kind {
            QuestionKind::Noul => {
                let (l, dflt) = if k == "true" { ("yes", "Yes") } else { ("no", "No") };
                (l.to_string(), if *d == semif_noul_default(k) { dflt.to_string() } else { d.clone() })
            }
            _ => (k.clone(), d.clone()),
        };
        let h = DECIDE_HYPOTHESIS.replace("{instr}", instr).replacen("{label}", &label, 1).replacen("{crit}", &crit, 1);
        (label, h)
    }).collect()
}

/// openjev's windowing at width `w` chars, overlap `w / 12`.
pub fn char_windows(premise: &str, w: usize) -> Vec<String> {
    let c: Vec<char> = premise.chars().collect();
    if c.len() <= w {
        return vec![premise.to_string()];
    }
    let ov = w / 12;
    let step = w - ov;
    (0..(c.len() - ov).max(1)).step_by(step).map(|s| c[s..(s + w).min(c.len())].iter().collect()).collect()
}

/// The widest windows at which every (window, hypothesis) prompt fits `s` tokens.
pub fn fit_windows(template: &str, premise: &str, hyps: &[String],
                   n_tokens: &mut dyn FnMut(&str) -> Result<usize, EngineError>, s: usize)
    -> Result<Vec<String>, EngineError> {
    let mut w = premise.chars().count().max(1);
    loop {
        let wins = char_windows(premise, w);
        let mut fits = true;
        'all: for x in &wins {
            for h in hyps {
                if n_tokens(&prompt(template, x, h)?)? > s { fits = false; break 'all; }
            }
        }
        if fits { return Ok(wins); }
        if w <= 64 {
            return Err(EngineError::Unsupported(format!("a hypothesis alone does not fit {s} tokens")));
        }
        w = w * 9 / 10;
    }
}

/// `[window][option]` P(entailment) -> per option max over windows, normalised over options.
pub fn normalise_entailment(p: &[Vec<f64>]) -> Vec<f64> {
    let best: Vec<f64> = (0..p[0].len()).map(|o| p.iter().map(|w| w[o]).fold(f64::MIN, f64::max)).collect();
    let z = best.iter().sum::<f64>().max(1e-9);
    best.iter().map(|x| x / z).collect()
}

#[derive(Debug, Clone)]
pub struct NliHead {
    pub labels: Vec<String>,
    pub template: String,
    rows: Vec<Vec<f32>>,
    entail: usize,
}

impl NliHead {
    /// `dir/head.json` + `dir/score.bin` as `scripts/extract_nli_head.py` writes them.
    pub fn load(dir: &Path) -> Result<Self, EngineError> {
        let err = |m: String| EngineError::Load(format!("NLI head {}: {m}", dir.display()));
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("head.json"))
            .map_err(|e| err(format!("head.json: {e}")))?).map_err(|e| err(format!("head.json: {e}")))?;
        let labels: Vec<String> = serde_json::from_value(meta["labels"].clone()).map_err(|e| err(format!("labels: {e}")))?;
        let d = meta["d_model"].as_u64().ok_or_else(|| err("no d_model".into()))? as usize;
        let template = meta["template"].as_str().ok_or_else(|| err("no template".into()))?.to_string();
        let raw = std::fs::read(dir.join("score.bin")).map_err(|e| err(format!("score.bin: {e}")))?;
        if raw.len() != labels.len() * d * 2 {
            return Err(err(format!("score.bin is {} bytes, want {} x {d} bf16", raw.len(), labels.len())));
        }
        let all = crate::llm::npu_decode::unpack_bf16_bytes(&raw);
        Self::from_parts(labels, template, all.chunks(d).map(<[f32]>::to_vec).collect())
    }

    pub fn from_parts(labels: Vec<String>, template: String, rows: Vec<Vec<f32>>) -> Result<Self, EngineError> {
        let entail = labels.iter().position(|l| l == "entailment")
            .ok_or_else(|| EngineError::Load(format!("NLI labels {labels:?} carry no `entailment`")))?;
        prompt(&template, "", "")?;
        Ok(NliHead { labels, template, rows, entail })
    }

    pub fn entailment(&self) -> usize { self.entail }

    /// Softmax over the head's rows applied to `xf`, accumulated in f64.
    pub fn probs(&self, xf: &[f32]) -> Result<Vec<f64>, EngineError> {
        if xf.len() != self.rows[0].len() {
            return Err(EngineError::Device(format!("final hidden is {} wide, head wants {}", xf.len(), self.rows[0].len())));
        }
        let z: Vec<f64> = self.rows.iter().map(|r| r.iter().zip(xf).map(|(a, b)| *a as f64 * *b as f64).sum()).collect();
        let m = z.iter().copied().fold(f64::MIN, f64::max);
        let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
        let s: f64 = e.iter().sum();
        Ok(e.iter().map(|x| x / s).collect())
    }
}

/// `/predict` and `/rerank`'s request: independent (premise, hypothesis) pairs.
#[derive(Debug, Clone)]
pub struct NliRequest {
    pub pairs: Vec<(String, String)>,
}

/// Per pair, the label probabilities from the window with the highest entailment.
#[derive(Debug, Clone, PartialEq)]
pub struct NliScores {
    pub labels: Vec<String>,
    pub probs: Vec<Vec<f64>>,
    pub windows: Vec<usize>,
    pub stats: DecideStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn golden() -> Value {
        serde_json::from_str(include_str!("../../../tests/refs/nli/common_golden.json")).unwrap()
    }
    fn toy_tokens(s: &str) -> usize { s.chars().count() / 4 + 1 }

    #[test]
    fn prompts_match_python() {
        let g = golden();
        let t = g["template"].as_str().unwrap();
        for c in g["prompt"].as_array().unwrap() {
            assert_eq!(prompt(t, c["premise"].as_str().unwrap(), c["hypothesis"].as_str().unwrap()).unwrap(),
                       c["out"].as_str().unwrap());
        }
    }

    #[test]
    fn hypotheses_match_python() {
        for c in golden()["hypotheses"].as_array().unwrap() {
            let q = question_from_jevbench(&c["q"]);
            let want: Vec<(String, String)> = serde_json::from_value(c["out"].clone()).unwrap();
            assert_eq!(decide_hypotheses(&q), want);
        }
    }

    #[test]
    fn windows_and_fit_match_python() {
        let g = golden();
        let t = g["template"].as_str().unwrap();
        for c in g["windows"].as_array().unwrap() {
            let want: Vec<String> = serde_json::from_value(c["out"].clone()).unwrap();
            assert_eq!(char_windows(c["premise"].as_str().unwrap(), c["w"].as_u64().unwrap() as usize), want);
        }
        for c in g["fit"].as_array().unwrap() {
            let hyps: Vec<String> = serde_json::from_value(c["hyps"].clone()).unwrap();
            let want: Vec<String> = serde_json::from_value(c["out"].clone()).unwrap();
            let got = fit_windows(t, c["premise"].as_str().unwrap(), &hyps, &mut |s| Ok(toy_tokens(s)),
                                  c["s"].as_u64().unwrap() as usize).unwrap();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn normalise_matches_python() {
        for c in golden()["normalise"].as_array().unwrap() {
            let p: Vec<Vec<f64>> = serde_json::from_value(c["p"].clone()).unwrap();
            let want: Vec<f64> = serde_json::from_value(c["out"].clone()).unwrap();
            let got = normalise_entailment(&p);
            assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-12), "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn head_softmax_is_over_the_three_rows() {
        let h = NliHead::from_parts(vec!["contradiction".into(), "entailment".into(), "neutral".into()],
            "Premise: {premise}\nHypothesis: {hypothesis}".into(),
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let p = h.probs(&[2.0, 0.0]).unwrap();
        assert_eq!(h.entailment(), 1);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[1] > p[0] && p[1] > p[2]);
        assert!(h.probs(&[1.0]).is_err(), "a hidden of the wrong width is an error, not a short dot");
    }

    #[test]
    fn a_head_without_entailment_is_refused() {
        assert!(NliHead::from_parts(vec!["a".into(), "b".into()], "{premise}{hypothesis}".into(),
                                    vec![vec![0.0], vec![0.0]]).is_err());
    }

    /// JevBench question json -> our DecideQuestion, the way `/v1/systemone`'s parser builds it.
    fn question_from_jevbench(q: &Value) -> DecideQuestion {
        let instr = q["instructions"].as_str().unwrap();
        match q["type"].as_str().unwrap() {
            "noul" => DecideQuestion::noul("q", instr, q["criteria"]["true"].as_str(), q["criteria"]["false"].as_str()),
            "choice" => DecideQuestion::choice("q", instr, q["criteria"].as_object().unwrap().iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect()),
            _ => DecideQuestion::score("q", instr, q["criteria"].as_array().unwrap().iter()
                .map(|v| v.as_str().unwrap().to_string()).collect()),
        }
    }
}
