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

    // Split `device` further -- from `npu_xrt::dispatch_log`, which every device dispatch already
    // funnels through. Nested under BOUND rather than a peer row: it explains the one row above it,
    // and its own residual is host glue around the dispatch (step_us also counts whatever the
    // backend wraps the call in), not a second unattributed bucket.
    //
    // BY XCLBIN/STREAM, not "by design": the log keys on (xclbin, instruction stream), and a fused
    // decode issues its whole layer stack as one instruction stream inside one XRT dispatch. One row
    // is therefore not a degenerate case of a split, it is the ONLY case this instrument can ever
    // produce for a fused design -- rendering it as a table with one row would claim a per-operator
    // attribution that never happened, so it gets one honest line instead.
    if s.design_breakdown.len() == 1 {
        let d = &s.design_breakdown[0];
        let dev_us = d.secs * 1e6;
        o.push_str(&format!(
            "      one xclbin/stream ({}, x{} dispatches, {:.3} ms/tok) IS the device row above --\n\
             \x20     everything dispatches inside a single XRT call, so no sub-dispatch/per-operator\n\
             \x20     split is visible to the host; this is a limit of the instrument, not a result\n",
            d.label, d.dispatches, dev_us / 1e3 / n));
    } else if !s.design_breakdown.is_empty() {
        o.push_str("      ── device, by xclbin/stream ──────────────────────\n");
        let mut designs = s.design_breakdown.clone();
        designs.sort_by(|a, b| b.secs.partial_cmp(&a.secs).unwrap_or(std::cmp::Ordering::Equal));
        let named_us: f64 = designs.iter().map(|d| d.secs * 1e6).sum();
        for d in &designs {
            let dev_us = d.secs * 1e6;
            let share = if s.phases.step_us == 0 { 0.0 } else { 100.0 * dev_us / s.phases.step_us as f64 };
            o.push_str(&format!(
                "      {:<28} x{:<4} {:>8.3} ms/tok {:>6.1}%\n", d.label, d.dispatches, dev_us / 1e3 / n, share));
        }
        let residual_us = (s.phases.step_us as f64 - named_us).max(0.0);
        let residual_share = if s.phases.step_us == 0 { 0.0 } else { 100.0 * residual_us / s.phases.step_us as f64 };
        o.push_str(&format!(
            "      {:<28} {:<5} {:>8.3} ms/tok {:>6.1}%  (host glue around the dispatches)\n",
            "unattributed", "", residual_us / 1e3 / n, residual_share));
    }

    // Split `sampling` further, by internal stage -- from `sampling::sample`'s own timing, wired
    // through `SampleOutcome::timings`. Same nesting convention as device-by-design: shares are of
    // the SAMPLING row's own total, not of the decode window, and the residual is real host cost
    // (RNG draw setup, allocation) the four named stages do not claim.
    if let Some(sp) = s.phases.sample_phases {
        if s.phases.sample_us > 0 {
            o.push_str("      ── sampling, by stage ────────────────────────────\n");
            let stage_rows =
                [("penalties", sp.penalties_us), ("top_k", sp.top_k_us), ("top_p", sp.top_p_us), ("draw", sp.draw_us)];
            let named_us: u64 = stage_rows.iter().map(|(_, v)| v).sum();
            for (name, v) in stage_rows {
                let share = 100.0 * v as f64 / s.phases.sample_us as f64;
                o.push_str(&format!("      {:<28} {:>8.3} ms/tok {:>6.1}%\n", name, ms(v) / n, share));
            }
            let residual_us = s.phases.sample_us.saturating_sub(named_us);
            let residual_share = 100.0 * residual_us as f64 / s.phases.sample_us as f64;
            o.push_str(&format!(
                "      {:<28} {:>8.3} ms/tok {:>6.1}%  (host glue around the four stages)\n",
                "unattributed", ms(residual_us) / n, residual_share));
        }
    }

    if let (Some(d), Some(t)) = (s.dispatches, s.transitions) {
        o.push_str(&format!("  dispatches {d} · context transitions {t}\n"));
    } else {
        o.push_str("  dispatches     not counted (set NPU_DISPATCH_LOG=1, or pass --dispatch-log for this run)\n");
    }

    let p = &s.provenance;
    let mut bits = Vec::new();
    if let Some(h) = &p.head_dtype { bits.push(format!("head {h}")); }
    if let Some(m) = &p.mlp_dtype {
        bits.push(match p.quant_group {
            Some(g) => format!("mlp {m} (g={g})"),
            None => format!("mlp {m}"),
        });
    }
    if !p.fusion_flags.is_empty() { bits.push(format!("flags {}", p.fusion_flags.join(","))); }
    if let Some(v) = p.max_seq { bits.push(format!("max_seq {v}")); }
    if let Some(v) = p.n_past { bits.push(format!("n_past {v}")); }
    if bits.is_empty() {
        o.push_str("  arm            not reported by this backend\n");
    } else {
        o.push_str(&format!("  arm            {}\n", bits.join(" · ")));
    }
    if p.artifact_path.is_some() || p.artifact_hash.is_some() || p.toolchain_pin_hash.is_some() {
        o.push_str(&format!(
            "                 artifact {} ({}) · toolchain {}\n",
            p.artifact_path.as_deref().unwrap_or("?"),
            p.artifact_hash.as_deref().unwrap_or("?"),
            p.toolchain_pin_hash.as_deref().unwrap_or("?")));
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
    // The mode above is a coarse driver enum (Default/Low/.../Turbo), not a clock. On this chip the
    // actual AIE core clock (an 8-level 792-1800 MHz DPM ladder) has no runtime read path at all --
    // `GET_CURRENT_DPM_LEVEL` exists only on AIE4 firmware, and aie2p's `aie2_msg_priv.h` carries no
    // power/DPM MSG_OP. So any "% of peak" claim here must be stated as a cycle ratio, never scaled
    // by an assumed clock.
    o.push_str("  clock          not independently readable on this chip (power mode above is the\n\
                 \x20                only readable proxy) -- treat any %-of-peak figure as a cycle\n\
                 \x20                ratio, not a clock-scaled one\n");
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
    use npu_engine::telemetry::DesignCost;
    use npu_engine::{StepPhases, StepRecord};

    fn run(step_us: u64, tokens: &[u32]) -> GenerationReport {
        let steps = tokens.iter().enumerate().map(|(i, t)| StepRecord {
            seq: i as u32,
            token: Some(*t),
            text: "x".into(),
            emit: "x".into(),
            t_us: 1_000 + 20_000 * i as u64,
            dt_us: if i == 0 { 1_000 } else { 20_000 },
            phases: StepPhases { step_us, sample_us: 100, detok_us: 20, sample_phases: None },
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
    fn sampling_by_stage_nests_under_the_sampling_row_and_shares_its_total() {
        let mut r = run(18_000, &[1, 2, 3, 4]);
        for s in &mut r.steps {
            s.phases.sample_phases =
                Some(npu_engine::telemetry::SamplePhases { penalties_us: 10, top_k_us: 20, top_p_us: 30, draw_us: 15 });
        }
        let t = table(&r);
        assert!(t.contains("sampling, by stage"), "{t}");
        assert!(t.contains("penalties"), "{t}");
        assert!(t.contains("top_k"), "{t}");
        assert!(t.contains("top_p"), "{t}");
        assert!(t.contains("draw"), "{t}");
        // sample_us is 100/token, the four named stages sum to 75 -- the remaining 25 must show up
        // as its own unattributed row inside the nested block, not be silently absorbed into one of
        // the four. That is a SECOND "unattributed": the top-level decode residual row is always
        // there too (design_breakdown is empty here, so its own nested residual does not add a third).
        assert_eq!(t.matches("unattributed").count(), 2, "{t}");
    }

    #[test]
    fn a_single_stream_degrades_to_one_line_instead_of_a_fake_split() {
        // A fused decode always produces exactly this: one xclbin/stream carrying the whole device
        // row. Rendering it as a one-row table would claim a per-operator split that never happened.
        let mut r = run(18_000, &[1, 2, 3, 4]);
        r.design_breakdown =
            vec![DesignCost { label: "main:sequence".to_string(), dispatches: 4, secs: 4.0 * 17_000e-6 }];
        let t = table(&r);
        assert!(t.contains("one xclbin/stream"), "{t}");
        assert!(t.contains("main:sequence"), "{t}");
        assert!(!t.contains("by design"), "must not use the retired, misleading label: {t}");
        assert!(!t.contains("by xclbin/stream ──"), "one row must not render as a table: {t}");
    }

    #[test]
    fn multiple_streams_still_render_as_a_breakdown_table() {
        let mut r = run(18_000, &[1, 2, 3, 4]);
        r.design_breakdown = vec![
            DesignCost { label: "prefill.xclbin".to_string(), dispatches: 1, secs: 0.010 },
            DesignCost { label: "decode.xclbin".to_string(), dispatches: 3, secs: 0.041 },
        ];
        let t = table(&r);
        assert!(t.contains("by xclbin/stream"), "{t}");
        assert!(t.contains("prefill.xclbin"), "{t}");
        assert!(t.contains("decode.xclbin"), "{t}");
        assert!(!t.contains("one xclbin/stream"), "a real split must not use the degraded line: {t}");
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
