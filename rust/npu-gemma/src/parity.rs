//! Parity hook against the host-CPU oracle (`scripts/gemma_ref_generate.py`).
//!
//! The on-NPU decode is gated in two stages (see the turnkey device doc): (1) per-node rel-L2 <= 0.08 vs
//! the reference intermediates, then (2) end-to-end greedy token-sequence parity. This module owns stage
//! (2)'s comparison + the reusable rel-L2 metric for stage (1). CPU-only; no XRT.

use std::path::Path;
use std::process::Command;

/// Relative L2 error `||a - b|| / ||b||` (b = reference). The per-node gate is `<= 0.08`.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "rel_l2 length mismatch");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    if den == 0.0 {
        return num.sqrt() as f32;
    }
    (num.sqrt() / den.sqrt()) as f32
}

/// The oracle: per-step greedy argmax token ids + generated text (the ground truth to match).
#[derive(Debug, Clone)]
pub struct Oracle {
    pub step_argmax: Vec<i64>,
    pub generated_ids: Vec<i64>,
    pub text: String,
}

/// Run `scripts/gemma_ref_generate.py --dump-oracle <dir>` and read back `oracle.json`. `python` must be
/// the IRON/transformers venv interpreter; `repo_root` is the public repo root. CPU-only oracle.
pub fn run_oracle(python: &str, repo_root: &Path, model: &str, prompt: &str, dump_dir: &Path) -> std::io::Result<Oracle> {
    let script = repo_root.join("scripts/gemma_ref_generate.py");
    let status = Command::new(python)
        .env("CUDA_VISIBLE_DEVICES", "")
        .arg(&script)
        .args(["--model", model, "--prompt", prompt, "--dump-oracle"])
        .arg(dump_dir)
        .status()?;
    if !status.success() {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, "gemma_ref_generate.py failed"));
    }
    read_oracle(dump_dir)
}

/// Parse a previously dumped `oracle.json`.
pub fn read_oracle(dump_dir: &Path) -> std::io::Result<Oracle> {
    let raw = std::fs::read_to_string(dump_dir.join("oracle.json"))?;
    // Minimal hand-parse to avoid a serde dependency in this scaffold crate.
    let step_argmax = parse_i64_array(&raw, "step_argmax");
    let generated_ids = parse_i64_array(&raw, "generated_ids");
    let text = parse_string(&raw, "text");
    Ok(Oracle { step_argmax, generated_ids, text })
}

/// Compare the NPU-generated token ids against the oracle. Returns `Ok(())` on exact prefix parity, else
/// the first divergence index. Token-sequence parity is the e2e gate.
pub fn token_parity(npu_ids: &[i64], oracle: &Oracle) -> Result<(), usize> {
    for (i, (a, b)) in npu_ids.iter().zip(&oracle.step_argmax).enumerate() {
        if a != b {
            return Err(i);
        }
    }
    Ok(())
}

fn parse_i64_array(raw: &str, key: &str) -> Vec<i64> {
    let pat = format!("\"{key}\"");
    let Some(start) = raw.find(&pat) else { return Vec::new() };
    let Some(lb) = raw[start..].find('[') else { return Vec::new() };
    let s = start + lb + 1;
    let Some(rb) = raw[s..].find(']') else { return Vec::new() };
    raw[s..s + rb]
        .split(',')
        .filter_map(|t| t.trim().parse::<i64>().ok())
        .collect()
}

fn parse_string(raw: &str, key: &str) -> String {
    let pat = format!("\"{key}\"");
    let Some(start) = raw.find(&pat) else { return String::new() };
    let rest = &raw[start + pat.len()..];
    let Some(c) = rest.find(':') else { return String::new() };
    let after = &rest[c + 1..];
    let Some(q1) = after.find('"') else { return String::new() };
    let tail = &after[q1 + 1..];
    let Some(q2) = tail.find('"') else { return String::new() };
    tail[..q2].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_l2_zero_on_equal() {
        let a = [1.0, 2.0, 3.0];
        assert!(rel_l2(&a, &a) < 1e-9);
    }

    #[test]
    fn rel_l2_scales() {
        let b = [3.0, 4.0]; // ||b|| = 5
        let a = [3.0, 4.3]; // diff 0.3 -> 0.06
        assert!((rel_l2(&a, &b) - 0.06).abs() < 1e-5);
    }

    #[test]
    fn token_parity_detects_divergence() {
        let o = Oracle { step_argmax: vec![9079, 108, 651], generated_ids: vec![], text: String::new() };
        assert_eq!(token_parity(&[9079, 108, 651], &o), Ok(()));
        assert_eq!(token_parity(&[9079, 999], &o), Err(1));
    }

    #[test]
    fn parse_oracle_json() {
        let raw = r#"{ "text": "the answer", "step_argmax": [1, 2, 3], "generated_ids": [1, 2] }"#;
        assert_eq!(parse_i64_array(raw, "step_argmax"), vec![1, 2, 3]);
        assert_eq!(parse_string(raw, "text"), "the answer");
    }
}
