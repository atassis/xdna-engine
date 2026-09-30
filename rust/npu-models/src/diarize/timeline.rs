//! Global activation map -> `Vec<Segment>`.

use ndarray::Array2;

use crate::capability::Segment;

/// Binarize `[n_frames, n_clusters]` accumulated activation into per-speaker spans.
///
/// `coverage[f]` is how many windows covered frame `f`. Dividing by it is not cosmetic: at a 1 s hop
/// over a 10 s window, mid-clip frames are covered ~10x and clip-edge frames once, so an
/// unnormalised threshold silently trims the first and last seconds of every clip.
///
/// `min_duration_off_s` bridges gaps shorter than itself (upstream default `0.0`, no bridging).
pub fn to_segments(
    global: &Array2<f32>,
    coverage: &[f32],
    threshold: f32,
    frame_s: f32,
    min_duration_off_s: f32,
) -> Vec<Segment> {
    let (n_frames, n_clusters) = global.dim();
    let mut out = Vec::new();
    for c in 0..n_clusters {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        for f in 0..n_frames {
            let cov = coverage.get(f).copied().unwrap_or(1.0).max(1.0);
            let on = global[[f, c]] / cov >= threshold;
            match (on, start) {
                (true, None) => start = Some(f),
                (false, Some(s)) => { spans.push((s, f)); start = None; }
                _ => {}
            }
        }
        if let Some(s) = start { spans.push((s, n_frames)); }

        // Bridge short gaps before emitting, so a speaker's pause does not split their turn.
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (s, e) in spans {
            match merged.last_mut() {
                Some((_, pe)) if (s - *pe) as f32 * frame_s < min_duration_off_s => *pe = e,
                _ => merged.push((s, e)),
            }
        }
        for (s, e) in merged {
            out.push(Segment {
                start_s: s as f32 * frame_s,
                end_s: e as f32 * frame_s,
                speaker: c as u32,
            });
        }
    }
    out.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn contiguous_active_frames_become_one_segment() {
        let g = arr2(&[[0.0f32], [1.0], [1.0], [1.0], [0.0]]);
        let s = to_segments(&g, &[1.0; 5], 0.5, 0.1, 0.0);
        assert_eq!(s.len(), 1);
        assert!((s[0].start_s - 0.1).abs() < 1e-6, "got {:?}", s[0]);
        assert!((s[0].end_s - 0.4).abs() < 1e-6, "got {:?}", s[0]);
        assert_eq!(s[0].speaker, 0);
    }

    #[test]
    fn coverage_normalises_the_edges() {
        // Frame 0: 1 window, fully active. Mid frames: 2 windows. Unnormalised, frame 0 (raw 1.0)
        // would lose to a mid frame (raw 2.0) and the clip's first second would vanish.
        let g = arr2(&[[1.0f32], [2.0], [0.0]]);
        let s = to_segments(&g, &[1.0, 2.0, 2.0], 0.5, 0.1, 0.0);
        assert_eq!(s.len(), 1, "the clip-edge frame must survive: {s:?}");
        assert!((s[0].start_s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn two_speakers_produce_independent_overlapping_segments() {
        let g = arr2(&[[1.0f32, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let s = to_segments(&g, &[1.0; 3], 0.5, 1.0, 0.0);
        assert_eq!(s.len(), 2);
        let a = s.iter().find(|x| x.speaker == 0).unwrap();
        let b = s.iter().find(|x| x.speaker == 1).unwrap();
        assert!(a.start_s < b.end_s && b.start_s < a.end_s, "speech may overlap: {s:?}");
    }

    #[test]
    fn gaps_shorter_than_min_duration_off_are_bridged() {
        let g = arr2(&[[1.0f32], [0.0], [1.0]]);
        assert_eq!(to_segments(&g, &[1.0; 3], 0.5, 1.0, 0.0).len(), 2);
        assert_eq!(to_segments(&g, &[1.0; 3], 0.5, 1.0, 1.5).len(), 1);
    }

    #[test]
    fn an_empty_timeline_is_not_a_panic() {
        let g = ndarray::Array2::<f32>::zeros((0, 0));
        assert!(to_segments(&g, &[], 0.5, 0.1, 0.0).is_empty());
    }

    #[test]
    fn segments_come_out_sorted_by_start() {
        // Cluster 1 speaks first, cluster 0 second: per-cluster scan order must not leak out.
        let g = arr2(&[[0.0f32, 1.0], [0.0, 1.0], [1.0, 0.0]]);
        let s = to_segments(&g, &[1.0; 3], 0.5, 1.0, 0.0);
        assert_eq!(s.len(), 2);
        assert!(s[0].start_s <= s[1].start_s, "{s:?}");
        assert_eq!(s[0].speaker, 1);
    }
}
