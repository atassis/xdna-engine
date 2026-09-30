//! Per-speaker activation masks, and aggregation of overlapping windows into one timeline.

use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Per-embedder-frame weights for one speaker of one window, with overlapping speech EXCLUDED.
///
/// The shipped pipeline runs `embedding_exclude_overlap: true`: WeSpeaker pools weighted
/// statistics over only the frames where this speaker is active ALONE. Frames where two speakers
/// overlap describe neither voice cleanly, so feeding them pulls both embeddings toward each other
/// -- precisely where clustering must keep them apart.
///
/// `n_out` is the EMBEDDER's frame count (fbank at ~100 Hz), which differs from the segmentation
/// frame count (~58.9 Hz); the mask is resampled nearest-neighbour.
pub fn exclusive_weights(activity: &Array3<bool>, window: usize, speaker: usize, n_out: usize)
    -> Vec<f32> {
    let (_, n_frames, n_speakers) = activity.dim();
    if n_frames == 0 { return vec![0.0; n_out]; }
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let f = ((i * n_frames) / n_out.max(1)).min(n_frames - 1);
        let mine = activity[[window, f, speaker]];
        let others = (0..n_speakers).any(|s| s != speaker && activity[[window, f, s]]);
        out.push(if mine && !others { 1.0 } else { 0.0 });
    }
    out
}

/// Whether a mask has any active frame. An all-zero mask cannot be embedded -- weighted pooling
/// over zero weight is undefined -- so such a (window, speaker) is dropped before the embedder.
pub fn has_speech(weights: &[f32]) -> bool {
    weights.iter().any(|&w| w > 0.0)
}

/// Overlapping windows -> one `[n_global_frames, n_clusters]` activation map.
///
/// Identity comes from CLUSTERING, not from window order: local speaker 0 of window 3 and local
/// speaker 0 of window 4 are unrelated until the clusterer says otherwise, which is why this takes
/// a `(window, local_speaker) -> cluster` map rather than assuming any correspondence.
///
/// Overlapping windows ACCUMULATE (each frame is covered by several windows at a 1 s hop over a
/// 10 s window), so the caller binarizes against a coverage-normalised threshold.
pub fn aggregate_windows(
    activity: &Array3<bool>,
    labels: &HashMap<(usize, usize), u32>,
    hop_frames: usize,
    n_global_frames: usize,
    n_clusters: usize,
) -> Array2<f32> {
    let (n_windows, n_frames, n_speakers) = activity.dim();
    let mut out = Array2::<f32>::zeros((n_global_frames, n_clusters.max(1)));
    for w in 0..n_windows {
        let base = w * hop_frames;
        for s in 0..n_speakers {
            // A (window, speaker) with no cluster was dropped before embedding (silent, or wholly
            // overlapped). It must contribute nothing rather than defaulting to cluster 0.
            let Some(&c) = labels.get(&(w, s)) else { continue };
            if (c as usize) >= out.ncols() { continue }
            for f in 0..n_frames {
                let g = base + f;
                if g >= n_global_frames { break }
                if activity[[w, f, s]] { out[[g, c as usize]] += 1.0; }
            }
        }
    }
    out
}

/// How many windows cover each global frame -- the denominator for binarization, so a frame near
/// the clip edge (covered by fewer windows) is not penalised against a mid-clip frame.
pub fn coverage(n_windows: usize, n_frames: usize, hop_frames: usize, n_global_frames: usize)
    -> Vec<f32> {
    let mut cov = vec![0.0f32; n_global_frames];
    for w in 0..n_windows {
        let base = w * hop_frames;
        for f in 0..n_frames {
            let g = base + f;
            if g >= n_global_frames { break }
            cov[g] += 1.0;
        }
    }
    cov
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::collections::HashMap;

    fn activity(frames: &[[bool; 3]]) -> Array3<bool> {
        let mut a = Array3::from_elem((1, frames.len(), 3), false);
        for (f, row) in frames.iter().enumerate() {
            for s in 0..3 { a[[0, f, s]] = row[s]; }
        }
        a
    }

    #[test]
    fn overlapping_frames_are_excluded_from_both_speakers() {
        let a = activity(&[[true, false, false], [true, true, false],
                           [false, true, false], [false, false, false]]);
        let w0 = exclusive_weights(&a, 0, 0, 4);
        let w1 = exclusive_weights(&a, 0, 1, 4);
        assert_eq!(w0, vec![1.0, 0.0, 0.0, 0.0], "overlap frame must not feed spk0's statistic");
        assert_eq!(w1, vec![0.0, 0.0, 1.0, 0.0], "nor spk1's");
    }

    #[test]
    fn weights_resample_to_the_embedder_frame_rate() {
        let a = activity(&[[true, false, false], [false, false, false]]);
        assert_eq!(exclusive_weights(&a, 0, 0, 4), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_speaker_with_no_exclusive_speech_yields_an_all_zero_mask() {
        let a = activity(&[[true, true, false], [true, true, false]]);
        assert_eq!(exclusive_weights(&a, 0, 0, 2), vec![0.0, 0.0]);
        assert!(!has_speech(&exclusive_weights(&a, 0, 0, 2)),
            "an all-zero mask must be detectable: it cannot be embedded");
    }

    #[test]
    fn windows_are_relabelled_by_cluster_then_summed_onto_a_global_timeline() {
        // w0 local spk0 on frames 0..2 ; w1 local spk0 on frames 2..4. Clusters say they are
        // DIFFERENT speakers despite both being "local speaker 0" -- what a naive stitcher fumbles.
        let mut act = Array3::from_elem((2, 4, 3), false);
        for f in 0..2 { act[[0, f, 0]] = true; }
        for f in 2..4 { act[[1, f, 0]] = true; }
        let labels: HashMap<(usize, usize), u32> =
            vec![((0usize, 0usize), 1u32), ((1, 0), 0)].into_iter().collect();
        let g = aggregate_windows(&act, &labels, 2, 6, 2);
        assert_eq!(g.dim(), (6, 2));
        assert!(g[[0, 1]] > 0.0 && g[[1, 1]] > 0.0, "window 0 frames land on cluster 1");
        assert_eq!(g[[0, 0]], 0.0);
        assert!(g[[4, 0]] > 0.0 && g[[5, 0]] > 0.0, "window 1 frames land on cluster 0");
        assert_eq!(g[[4, 1]], 0.0);
    }

    #[test]
    fn unlabelled_window_speakers_contribute_nothing() {
        let mut act = Array3::from_elem((1, 2, 3), false);
        act[[0, 0, 2]] = true;
        let labels = HashMap::new();
        let g = aggregate_windows(&act, &labels, 2, 2, 1);
        assert_eq!(g.sum(), 0.0, "a dropped (window, speaker) must not leak into the timeline");
    }

    #[test]
    fn coverage_counts_how_many_windows_touch_each_frame() {
        // 2 windows of 4 frames at hop 2 over 6 global frames: frames 2,3 are double-covered.
        let cov = coverage(2, 4, 2, 6);
        assert_eq!(cov, vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0]);
    }
}
