//! Powerset -> per-speaker binary activity.
//!
//! segmentation-3.0 is a powerset classifier, not a multilabel one: each frame gets ONE class
//! naming the SET of active speakers. With `max_speakers_per_chunk = 3` and
//! `max_speakers_per_frame = 2` that is C(3,0)+C(3,1)+C(3,2) = 7 classes, which is why the head is
//! 7-wide and not 2^3.

use ndarray::{Array3, ArrayView3};

/// The class -> speaker-set table, in the order the checkpoint was trained with: the empty set,
/// then every singleton, then every pair.
pub const POWERSET_3: [&[usize]; 7] =
    [&[], &[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]];

/// `[n_windows, n_frames, n_classes]` logits -> `[n_windows, n_frames, n_speakers]` activity.
///
/// Argmax over classes, NOT a per-speaker threshold: the classes are mutually exclusive by
/// construction, so thresholding them independently can assert two disjoint sets at once.
pub fn decode_powerset(logits: &Array3<f32>, n_speakers: usize) -> Array3<bool> {
    decode_powerset_view(logits.view(), n_speakers)
}

pub fn decode_powerset_view(logits: ArrayView3<f32>, n_speakers: usize) -> Array3<bool> {
    let (nw, nf, nc) = logits.dim();
    let mut out = Array3::from_elem((nw, nf, n_speakers), false);
    for w in 0..nw {
        for f in 0..nf {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for c in 0..nc {
                let v = logits[[w, f, c]];
                if v > best_v { best_v = v; best = c; }
            }
            for &s in POWERSET_3[best.min(POWERSET_3.len() - 1)] {
                if s < n_speakers { out[[w, f, s]] = true; }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Build [1, n_frames, 7] logits that argmax to the given class per frame.
    fn logits(classes: &[usize]) -> Array3<f32> {
        let mut a = Array3::zeros((1, classes.len(), 7));
        for (f, &c) in classes.iter().enumerate() { a[[0, f, c]] = 1.0; }
        a
    }

    #[test]
    fn classes_map_to_the_speaker_sets_pyannote_uses() {
        assert_eq!(POWERSET_3, [
            &[][..], &[0][..], &[1][..], &[2][..], &[0, 1][..], &[0, 2][..], &[1, 2][..]]);
    }

    #[test]
    fn decode_marks_single_and_overlapping_speakers() {
        // frame 0: silence, 1: spk0, 2: spk2, 3: spk0+spk1
        let a = decode_powerset(&logits(&[0, 1, 3, 4]), 3);
        assert_eq!(a.shape(), &[1, 4, 3]);
        assert!(!a[[0, 0, 0]], "class 0 is non-speech");
        assert!(a[[0, 1, 0]]);
        assert!(!a[[0, 1, 1]]);
        assert!(a[[0, 2, 2]]);
        assert!(a[[0, 3, 0]] && a[[0, 3, 1]], "class 4 = {{0,1}} is the overlap case");
        assert!(!a[[0, 3, 2]]);
    }

    #[test]
    fn decode_takes_the_argmax_not_a_threshold() {
        // Every class scores, class 2 highest -> exactly speaker 1 active.
        let mut a = Array3::from_elem((1, 1, 7), 0.4f32);
        a[[0, 0, 2]] = 0.9;
        let d = decode_powerset(&a, 3);
        assert_eq!((d[[0, 0, 0]], d[[0, 0, 1]], d[[0, 0, 2]]), (false, true, false));
    }
}
