//! Price `segmentation.step_s`: what a bigger window hop buys in time and costs in accuracy.
//!
//! The hop is 0.1 x the window by default, so every second of audio is segmented and embedded ~10
//! times. Crops scale with windows, so the trade is close to linear in time -- but the accuracy
//! cost is NOT guessable, because two things move at once: the timeline gets coarser, and the crop
//! count falls toward `min_cluster_size` (12 upstream), below which the post-hoc reassignment pass
//! starts merging real speakers away.
//!
//! Accuracy is scored against whichever oracle the clip has, and the two are not equivalent:
//!   - `artifacts/pyannote/ref/<stem>/summary.json` -- pyannote's OWN answer, so "disagreement"
//!     means "differs from upstream", including where upstream is wrong;
//!   - `<clip>.json` with a `turns` list -- constructed ground truth, so disagreement means WRONG.
//! Our own step = the first swept value is always reported too, which isolates what the hop
//! changed from what we get wrong at every hop.
//!
//! A fourth argument names an EXTERNAL oracle instead -- another system's answer, in either
//! `{"segments":[...]}` or a bare `[{start,end,speaker}]` list. It is reported under its own name
//! because it is not truth: disagreement with it means "differs from that system", nothing more.
//!
//!   rust/target/release/sweep_diarize_step artifacts/pyannote/fixtures/conversation_2spk.wav \
//!       speaker-diarization-3.1 1.0,2.0,3.0,5.0
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use ndarray::{Array2, Array3};
use npu_engine::api::EngineError;
use npu_engine::capability::Segment;
use npu_engine::diarize::onnx::{OnnxEmbedder, OnnxSegmenter};
use npu_engine::diarize::types::{Crop, Segmenter, SpeakerEmbedder};
use npu_engine::diarize::{DiarizePipeline, Manifest};

/// One labelled span, from any oracle. `speaker` is an opaque id: comparison is permutation-
/// minimised, so its value never matters, only which spans share it.
#[derive(Clone, Debug)]
struct Span { start: f32, end: f32, speaker: String }

/// A span as every oracle format spells it. `speaker` is a string in pyannote's dumps and an
/// integer in sherpa's, so it is read untyped and stringified -- the comparison is permutation-
/// minimised and never looks at the value.
#[derive(serde::Deserialize)]
struct RefSeg { start: f32, end: f32, speaker: serde_json::Value }
#[derive(serde::Deserialize)]
struct RefSummary { segments: Vec<RefSeg> }

#[derive(serde::Deserialize)]
struct FixtureTurn { speaker: String, start_s: f32, end_s: f32 }
#[derive(serde::Deserialize)]
struct Fixture { turns: Vec<FixtureTurn> }

fn read_wav_i16(p: &Path) -> Vec<i16> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let pos = b.windows(4).position(|w| w == b"data").expect("no data chunk");
    b[pos + 8..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Wraps a stage to record its wall time. The engine only prints stage times to stderr behind
/// `NPU_DIARIZE_TIME`; a sweep needs them as values, and the traits make that a wrapper rather
/// than an engine change.
struct TimedSeg { inner: OnnxSegmenter, t: Rc<Cell<Duration>>, windows: Rc<Cell<usize>> }
impl Segmenter for TimedSeg {
    fn segment(&self, pcm: &[i16]) -> Result<(Array3<f32>, usize), EngineError> {
        let t0 = Instant::now();
        let r = self.inner.segment(pcm);
        self.t.set(self.t.get() + t0.elapsed());
        if let Ok((a, _)) = &r { self.windows.set(a.dim().0); }
        r
    }
}

struct TimedEmb {
    inner: OnnxEmbedder,
    t: Rc<Cell<Duration>>,
    crops: Rc<Cell<usize>>,
    /// Summed fraction of each crop's frames that are ACTIVE, and the fraction spanned by the
    /// active HULL (first active frame to last). The graph is exported with a frozen 10 s time
    /// axis, so every crop costs a full window whatever these say; the gap between them and 1.0
    /// is what a time-dynamic export would stop paying for.
    active: Rc<Cell<f64>>,
    hull: Rc<Cell<f64>>,
}
impl SpeakerEmbedder for TimedEmb {
    fn embed(&self, pcm: &[i16], crops: &[Crop]) -> Result<Array2<f32>, EngineError> {
        self.crops.set(self.crops.get() + crops.len());
        for c in crops {
            let n = c.weights.len().max(1) as f64;
            let on = c.weights.iter().filter(|&&w| w > 0.0).count() as f64;
            let first = c.weights.iter().position(|&w| w > 0.0);
            let last = c.weights.iter().rposition(|&w| w > 0.0);
            let span = match (first, last) { (Some(a), Some(b)) => (b - a + 1) as f64, _ => 0.0 };
            self.active.set(self.active.get() + on / n);
            self.hull.set(self.hull.get() + span / n);
        }
        let t0 = Instant::now();
        let r = self.inner.embed(pcm, crops);
        self.t.set(self.t.get() + t0.elapsed());
        r
    }
}

/// Fraction of the clip on which two answers disagree about WHO is speaking, minimised over
/// label permutations.
///
/// Compares the SET of active speakers per tick, not the first match: a diarization answer is
/// multi-label wherever speech overlaps, and scoring only the first span silently ignores exactly
/// the frames the masked-pooling path exists for.
fn disagreement(ours: &[Span], theirs: &[Span], end_s: f32, tick: f32) -> f32 {
    let ids = |v: &[Span]| -> Vec<String> {
        let mut s: Vec<String> = v.iter().map(|x| x.speaker.clone()).collect();
        s.sort(); s.dedup(); s
    };
    let (oi, ti) = (ids(ours), ids(theirs));
    if oi.is_empty() && ti.is_empty() { return 0.0 }
    // Map our labels onto theirs; pad the shorter side with unmatched slots so a run that finds
    // more speakers than the oracle is penalised rather than silently collapsed.
    let k = oi.len().max(ti.len());
    let mut perm: Vec<usize> = (0..k).collect();
    let mut perms = Vec::new();
    permute(&mut perm, 0, &mut perms);
    let n_ticks = (end_s / tick).ceil() as usize;
    let mut best = f32::INFINITY;
    for p in &perms {
        let mut bad = 0usize;
        for i in 0..n_ticks {
            let t = i as f32 * tick;
            let mut a: Vec<usize> = ours.iter().filter(|s| t >= s.start && t < s.end)
                .filter_map(|s| oi.iter().position(|x| *x == s.speaker))
                .filter_map(|idx| p.get(idx).copied()).collect();
            let mut b: Vec<usize> = theirs.iter().filter(|s| t >= s.start && t < s.end)
                .filter_map(|s| ti.iter().position(|x| *x == s.speaker)).collect();
            a.sort_unstable(); a.dedup();
            b.sort_unstable(); b.dedup();
            if a != b { bad += 1; }
        }
        best = best.min(bad as f32 * tick);
    }
    100.0 * best / end_s.max(1e-6)
}

fn permute(v: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if out.len() > 5040 { return }          // 7! -- past that the oracle is not a speaker set
    if k == v.len() { out.push(v.clone()); return }
    for i in k..v.len() { v.swap(k, i); permute(v, k + 1, out); v.swap(k, i); }
}

fn ref_spans(v: &[RefSeg]) -> Vec<Span> {
    v.iter().map(|r| Span { start: r.start, end: r.end, speaker: r.speaker.to_string() }).collect()
}

/// Read an oracle that is either `{"segments": [...]}` or a bare `[...]`.
fn read_oracle(p: &Path) -> Option<Vec<Span>> {
    let txt = std::fs::read_to_string(p).ok()?;
    if let Ok(s) = serde_json::from_str::<RefSummary>(&txt) { return Some(ref_spans(&s.segments)) }
    serde_json::from_str::<Vec<RefSeg>>(&txt).ok().map(|v| ref_spans(&v))
}

fn to_spans(segs: &[Segment]) -> Vec<Span> {
    segs.iter().map(|s| Span { start: s.start_s, end: s.end_s, speaker: s.speaker.to_string() })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = PathBuf::from(args.next().expect(
        "usage: sweep_diarize_step <clip.wav> [model] [step_s,step_s,...]"));
    let model = args.next().unwrap_or_else(|| "speaker-diarization-3.1".into());
    let steps: Vec<f32> = args.next().unwrap_or_else(|| "1.0,1.5,2.0,2.5,3.0,5.0".into())
        .split(',').map(|s| s.trim().parse().expect("step_s must be a number")).collect();
    let external = args.next().map(PathBuf::from);
    assert!(!steps.is_empty(), "need at least one step_s");

    let dir = Path::new("artifacts/pyannote").join(&model);
    let base: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("diarize.json")).expect("diarize.json")).expect("manifest");
    let pcm = read_wav_i16(&wav);
    let audio_s = pcm.len() as f32 / base.sample_rate as f32;
    let stem = wav.file_stem().unwrap().to_string_lossy().to_string();

    // Oracle. pyannote's own dump wins when present -- it is the model we claim parity WITH; the
    // constructed fixture is the fallback and means something different, so the header says which.
    let refp = Path::new("artifacts/pyannote/ref").join(&stem).join("summary.json");
    let fixp = wav.with_extension("json");
    let ext_name;
    let (oracle, oracle_name): (Option<Vec<Span>>, &str) = if let Some(e) = &external {
        ext_name = format!("external {}", e.display());
        (read_oracle(e).or_else(|| panic!("cannot read oracle {}", e.display())), &ext_name)
    } else if refp.is_file() {
        (read_oracle(&refp), "pyannote ref")
    } else if fixp.is_file() {
        match serde_json::from_str::<Fixture>(&std::fs::read_to_string(&fixp).unwrap()) {
            Ok(f) => (Some(f.turns.iter()
                .map(|t| Span { start: t.start_s, end: t.end_s, speaker: t.speaker.clone() })
                .collect()), "fixture truth"),
            Err(_) => (None, "none"),
        }
    } else { (None, "none") };

    println!("clip {} ({audio_s:.2}s)  model {model}  clustering {}  oracle: {oracle_name}",
             wav.display(), base.clustering.method);
    println!("{:>6} {:>8} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>6} {:>10} {:>10}",
             "step_s", "windows", "crops", "act%", "hull%", "seg_s", "emb_s", "clu_s", "total_s",
             "spk", "vs_oracle", "vs_base");

    let mut baseline: Option<Vec<Span>> = None;
    for &step in &steps {
        let mut m = base.clone();
        m.segmentation.step_s = step;
        // The pipeline owns the stages, so the counters are shared handles rather than fields
        // read back out of it -- the wrapper is MOVED into the Box, and a pointer taken before
        // that move would dangle.
        let (t_seg_c, n_win_c) = (Rc::new(Cell::new(Duration::ZERO)), Rc::new(Cell::new(0)));
        let (t_emb_c, n_crop_c) = (Rc::new(Cell::new(Duration::ZERO)), Rc::new(Cell::new(0)));
        let (act_c, hull_c) = (Rc::new(Cell::new(0.0f64)), Rc::new(Cell::new(0.0f64)));
        let seg = TimedSeg {
            inner: OnnxSegmenter::build(&m, &dir).expect("segmenter"),
            t: t_seg_c.clone(), windows: n_win_c.clone() };
        let emb = TimedEmb {
            inner: OnnxEmbedder::build(&m, &dir).expect("embedder"),
            t: t_emb_c.clone(), crops: n_crop_c.clone(),
            active: act_c.clone(), hull: hull_c.clone() };
        let p = DiarizePipeline::new(m.clone(), Box::new(seg), Box::new(emb), &dir)
            .expect("pipeline");
        let t0 = Instant::now();
        let segs = p.run(&pcm).expect("diarize");
        let total = t0.elapsed();
        let (t_seg, n_win, t_emb, n_crop) =
            (t_seg_c.get(), n_win_c.get(), t_emb_c.get(), n_crop_c.get());
        let t_clu = total.saturating_sub(t_seg + t_emb);

        let ours = to_spans(&segs);
        let n_spk = segs.iter().map(|s| s.speaker).collect::<std::collections::BTreeSet<_>>().len();
        let vs_or = oracle.as_ref()
            .map(|o| format!("{:.2}%", disagreement(&ours, o, audio_s, 0.01)))
            .unwrap_or_else(|| "-".into());
        let vs_base = match &baseline {
            None => "(base)".to_string(),
            Some(b) => format!("{:.2}%", disagreement(&ours, b, audio_s, 0.01)),
        };
        let nc = n_crop.max(1) as f64;
        println!("{step:>6.2} {n_win:>8} {n_crop:>7} {:>7.1} {:>7.1} {:>8.3} {:>8.3} {:>8.3} \
                  {:>8.3} {n_spk:>6} {vs_or:>10} {vs_base:>10}",
                 100.0 * act_c.get() / nc, 100.0 * hull_c.get() / nc,
                 t_seg.as_secs_f64(), t_emb.as_secs_f64(), t_clu.as_secs_f64(),
                 total.as_secs_f64());
        if baseline.is_none() { baseline = Some(ours); }
    }
}
