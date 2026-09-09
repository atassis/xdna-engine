//! Rendering a generation's measurements for a human, and reading one back off disk.
//!
//! Two forms, and the split is deliberate. Measuring is free, so it always happens; READING is not
//! free -- a wall of numbers under every one-line answer is noise. So the compact form goes to
//! STDERR after every generation (a pipe still gets clean text, a terminal always gets the numbers)
//! and the full breakdown waits to be asked for.
//!
//! The shape is a frame-time overlay, not a table of totals: average, then the tail, then who is
//! actually holding the frame up. A mean tok/s cannot show a stall and cannot say what to fix.

use std::path::Path;

use anyhow::{anyhow, Result};
use npu_engine::telemetry::wire;
use npu_engine::{GenerationReport, Summary};

fn ms(us: u64) -> f64 {
    us as f64 / 1e3
}

/// One line, for stderr after every generation.
pub fn one_line(s: &Summary) -> String {
    format!(
        "{:.1} tok/s · {:.1} ms/tok · 1% low {:.1} · ttft {:.0} ms · {} {:.0}%",
        s.tok_per_s,
        ms(s.itl_mean_us),
        ms(s.itl_p99_us),
        ms(s.ttft_us),
        s.bound.as_str(),
        s.bound_share * 100.0,
    )
}

/// The full breakdown.
///
/// The phase rows are shares of the DECODE window, not of a renormalized phase sum, so `residual`
/// is a real row: if the named phases do not add up, the gap is visible instead of being spread
/// silently across the rows that happen to be measured.
pub fn table(r: &GenerationReport) -> String {
    let s = r.summarize();
    let mut o = String::new();
    let pct = |v: u64| if s.decode_us == 0 { 0.0 } else { 100.0 * v as f64 / s.decode_us as f64 };

    o.push_str(&format!(
        "  {:.1} tok/s  ·  {:.1} ms/tok  ·  p95 {:.1}  ·  p99 {:.1}  ·  max {:.1}\n",
        s.tok_per_s, ms(s.itl_mean_us), ms(s.itl_p95_us), ms(s.itl_p99_us), ms(s.itl_max_us)));
    o.push_str(&format!(
        "  ttft {:.0} ms  = queue {:.1} · load {:.1} · tokenize {:.1} · prefill {:.1}\n",
        ms(s.ttft_us), ms(s.queue_us), ms(s.load_us), ms(s.tokenize_us), ms(s.prefill_us)));
    o.push_str(&format!(
        "  {} prompt + {} completion tokens  ·  decode {:.0} ms  ·  total {:.0} ms\n",
        s.prompt_tokens, s.completion_tokens, ms(s.decode_us), ms(s.total_us)));
    if let Some(p) = s.prompt_tok_per_s {
        o.push_str(&format!("  prefill {:.0} tok/s over {} tokens ({} batched, {} stepwise)\n",
            p, r.prefill.tokens, r.prefill.batched, r.prefill.stepwise));
    }

    o.push_str("  ── where the decode went ─────────────────────────────\n");
    let mut rows = vec![
        ("device", s.phases.step_us),
        ("sampling", s.phases.sample_us),
        ("detokenize", s.phases.detok_us),
        ("unattributed", s.residual_us),
    ];
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    let n = s.completion_tokens.max(1) as f64;
    for (name, v) in rows {
        o.push_str(&format!("  {name:<14} {:>8.3} ms/tok {:>6.1}%\n", ms(v) / n, pct(v)));
    }
    o.push_str(&format!("  {:<14} {:>8}    {}\n", "BOUND", s.bound.as_str(), s.bound.lever()));

    if let (Some(d), Some(t)) = (s.dispatches, s.transitions) {
        o.push_str(&format!("  dispatches {d} · context transitions {t}\n"));
    } else {
        o.push_str("  dispatches     not counted (set NPU_DISPATCH_LOG=1)\n");
    }
    o.push_str(&format!(
        "  conditions     {} · power mode {} · {}\n",
        r.conditions.engine_version,
        r.conditions.power_mode.as_deref().unwrap_or("unknown"),
        match r.conditions.resident {
            Some(true) => "model was resident",
            Some(false) => "COLD -- model loaded during this request",
            None => "residency unknown",
        }));
    if r.conditions.power_mode.is_none() {
        // Not a nag: without the mode this run cannot be compared to another one, and the usual
        // cause of an unexplained regression on this device is that the mode moved.
        o.push_str("                 (unpinned or unreadable -- a tok/s taken now is not comparable\n\
                     \x20                 to one taken under a different mode)\n");
    }
    o
}

/// Read a run log and render it. The overlay then has exactly one input type, and works on a run
/// that happened on another day or another machine.
pub fn from_log(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path)?;
    let run = wire::parse_run(&text).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let mut o = format!("{} · {} · {} steps\n", run.id, run.model, run.steps.len());
    match &run.summary {
        Some(s) => {
            o.push_str(&format!("  {:.1} tok/s · {:.1} ms/tok · p99 {:.1} · ttft {:.0} ms\n",
                s.tok_per_s, ms(s.itl_mean_us), ms(s.itl_p99_us), ms(s.ttft_us)));
            o.push_str(&format!("  bound: {} ({:.0}%) -- {}\n",
                s.bound.as_str(), s.bound_share * 100.0, s.bound.lever()));
            o.push_str(&format!("  power mode {} · engine {}\n",
                run.conditions.power_mode.as_deref().unwrap_or("unknown"),
                run.conditions.engine_version));
        }
        None => o.push_str("  (truncated: no summary line -- the run did not finish)\n"),
    }
    Ok(o)
}

/// Compare two run logs. Correctness first, then speed: a timing difference between two runs that
/// produced different tokens is not a regression, it is a different computation, and reporting the
/// milliseconds first invites reading it as one.
pub fn diff(a: &Path, b: &Path) -> Result<String> {
    let ra = wire::parse_run(&std::fs::read_to_string(a)?).map_err(|e| anyhow!("{}: {e}", a.display()))?;
    let rb = wire::parse_run(&std::fs::read_to_string(b)?).map_err(|e| anyhow!("{}: {e}", b.display()))?;
    let mut o = String::new();

    let divergence = ra.steps.iter().zip(&rb.steps).position(|(x, y)| x.token != y.token);
    match divergence {
        Some(i) => o.push_str(&format!(
            "TOKENS DIVERGE at seq {i}: {:?} vs {:?} -- the runs did not compute the same thing,\n\
             so the timings below are not comparable.\n",
            ra.steps[i].token, rb.steps[i].token)),
        None if ra.steps.len() != rb.steps.len() => o.push_str(&format!(
            "same tokens as far as both go, but lengths differ: {} vs {}\n",
            ra.steps.len(), rb.steps.len())),
        None => o.push_str(&format!("tokens identical ({} steps)\n", ra.steps.len())),
    }

    if ra.conditions.power_mode != rb.conditions.power_mode {
        o.push_str(&format!(
            "POWER MODE DIFFERS: {} vs {} -- on this device that alone moves the numbers.\n",
            ra.conditions.power_mode.as_deref().unwrap_or("unknown"),
            rb.conditions.power_mode.as_deref().unwrap_or("unknown")));
    }

    if let (Some(sa), Some(sb)) = (&ra.summary, &rb.summary) {
        let d = |x: f64, y: f64| if x == 0.0 { 0.0 } else { 100.0 * (y - x) / x };
        o.push_str(&format!("  tok/s     {:>8.1} -> {:>8.1}  ({:+.1}%)\n",
            sa.tok_per_s, sb.tok_per_s, d(sa.tok_per_s, sb.tok_per_s)));
        o.push_str(&format!("  ms/tok    {:>8.2} -> {:>8.2}  ({:+.1}%)\n",
            ms(sa.itl_mean_us), ms(sb.itl_mean_us), d(ms(sa.itl_mean_us), ms(sb.itl_mean_us))));
        o.push_str(&format!("  p99 ms    {:>8.2} -> {:>8.2}\n", ms(sa.itl_p99_us), ms(sb.itl_p99_us)));
        o.push_str(&format!("  ttft ms   {:>8.0} -> {:>8.0}\n", ms(sa.ttft_us), ms(sb.ttft_us)));
        o.push_str(&format!("  device ms {:>8.1} -> {:>8.1}\n",
            ms(sa.phases.step_us), ms(sb.phases.step_us)));
        o.push_str(&format!("  bound     {:>8} -> {:>8}\n", sa.bound.as_str(), sb.bound.as_str()));
    } else {
        o.push_str("  (one side has no summary -- nothing to compare)\n");
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use npu_engine::{StepPhases, StepRecord};

    fn run(step_us: u64, tokens: &[u32]) -> GenerationReport {
        let steps = tokens.iter().enumerate().map(|(i, t)| StepRecord {
            seq: i as u32,
            token: Some(*t),
            text: "x".into(),
            emit: "x".into(),
            t_us: 1_000 + 20_000 * i as u64,
            dt_us: if i == 0 { 1_000 } else { 20_000 },
            phases: StepPhases { step_us, sample_us: 100, detok_us: 20 },
            ..StepRecord::default()
        }).collect();
        GenerationReport {
            steps,
            generate_us: 100_000,
            usage: npu_engine::GenerateUsage { prompt_tokens: 5, completion_tokens: tokens.len() as u32 },
            ..GenerationReport::default()
        }
    }

    fn write(dir: &Path, name: &str, r: &GenerationReport) -> std::path::PathBuf {
        let m = wire::RunMeta { id: name.into(), created: 0, model: "m".into(), chat: true };
        let mut lines = vec![wire::header_line(&r.conditions, &m).to_string()];
        lines.extend(r.steps.iter().map(|s| wire::chunk_line(s, &m).to_string()));
        lines.push(wire::prefill_line(&r.prefill, &m).to_string());
        lines.push(wire::summary_line(r, &m, npu_engine::FinishReason::Stop).to_string());
        let p = dir.join(format!("{name}.jsonl"));
        std::fs::write(&p, lines.join("\n")).unwrap();
        p
    }

    #[test]
    fn the_one_line_form_names_the_bottleneck() {
        let line = one_line(&run(18_000, &[1, 2, 3]).summarize());
        assert!(line.contains("tok/s"), "{line}");
        assert!(line.contains("device"), "the verdict is on the line, not just the numbers: {line}");
    }

    #[test]
    fn the_table_shows_unattributed_time_as_its_own_row() {
        // 20 ms gaps with 2.1 ms of named phases: the table must show ~89% unattributed rather than
        // renormalizing the named rows to 100% and reporting "device 94%".
        let t = table(&run(2_000, &[1, 2, 3, 4]));
        assert!(t.contains("unattributed"), "{t}");
        assert!(t.contains("BOUND"), "{t}");
    }

    #[test]
    fn an_unknown_power_mode_is_called_out_as_uncomparable() {
        let t = table(&run(18_000, &[1, 2]));
        assert!(t.contains("power mode unknown"), "{t}");
        assert!(t.contains("not comparable"), "{t}");
    }

    #[test]
    fn diff_leads_with_token_divergence_not_with_milliseconds() {
        let d = tempfile::tempdir().unwrap();
        let a = write(d.path(), "a", &run(18_000, &[1, 2, 3]));
        let b = write(d.path(), "b", &run(9_000, &[1, 9, 3]));
        let out = diff(&a, &b).unwrap();
        assert!(out.starts_with("TOKENS DIVERGE at seq 1"), "{out}");
        assert!(out.contains("not comparable"), "{out}");
    }

    #[test]
    fn diff_of_identical_tokens_reports_the_speed_change() {
        let d = tempfile::tempdir().unwrap();
        let a = write(d.path(), "a", &run(18_000, &[1, 2, 3]));
        let b = write(d.path(), "b", &run(9_000, &[1, 2, 3]));
        let out = diff(&a, &b).unwrap();
        assert!(out.starts_with("tokens identical"), "{out}");
        assert!(out.contains("device ms"), "{out}");
    }

    #[test]
    fn a_truncated_log_is_reported_as_truncated_not_as_zero() {
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "a", &run(18_000, &[1, 2, 3]));
        let text = std::fs::read_to_string(&p).unwrap();
        let cut: Vec<&str> = text.lines().take(3).collect();
        std::fs::write(&p, cut.join("\n")).unwrap();
        assert!(from_log(&p).unwrap().contains("truncated"));
    }
}
