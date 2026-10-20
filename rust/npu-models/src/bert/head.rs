//! Embedding head: pool encoder outputs over valid tokens, optional L2-normalize.

use ndarray::Array2;
use npu_asr_host::batch::PassageSpans;
use crate::pipeline::Head;

pub struct EmbedHead {
    pub pooling: Pooling,
    pub normalize: bool,
}

impl EmbedHead {
    pub fn run_cls_packed(&self, encoded: &Array2<f32>, spans: &PassageSpans) -> Result<Vec<Vec<f32>>, String> {
        if self.pooling != Pooling::Cls || !self.normalize || encoded.nrows() != spans.rows() || encoded.ncols() == 0 {
            return Err("packed CLS head requires normalized CLS and matching nonzero output width".into());
        }
        spans.ranges().iter().map(|span| {
            let norm2: f32 = encoded.row(span.start).iter().map(|x| x * x).sum();
            if !norm2.is_finite() || norm2 <= 0.0 {
                return Err(format!("invalid CLS norm at packed row {}", span.start));
            }
            let norm = norm2.sqrt();
            Ok(encoded.row(span.start).iter().map(|x| x / norm).collect())
        }).collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Pooling { Mean, Cls }

impl Pooling {
    pub fn parse(s: &str) -> Pooling {
        match s { "cls" => Pooling::Cls, _ => Pooling::Mean }
    }
}

impl Head for EmbedHead {
    type Output = Vec<f32>;
    /// `encoded` is [seq, D]; pool over the first `valid_len` rows.
    fn run(&self, encoded: &Array2<f32>, valid_len: usize) -> Vec<f32> {
        let d = encoded.ncols();
        let vl = valid_len.max(1).min(encoded.nrows());
        let mut v = vec![0f32; d];
        match self.pooling {
            Pooling::Cls => {
                for j in 0..d { v[j] = encoded[[0, j]]; }
            }
            Pooling::Mean => {
                for t in 0..vl {
                    for j in 0..d { v[j] += encoded[[t, j]]; }
                }
                let inv = 1.0 / vl as f32;
                for x in v.iter_mut() { *x *= inv; }
            }
        }
        if self.normalize {
            let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-12);
            for x in v.iter_mut() { *x /= norm; }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn mean_pool_l2_norm() {
        let enc = array![[3.0f32, 0.0], [0.0, 0.0]]; // valid_len=1 -> mean = [3,0]
        let head = EmbedHead { pooling: Pooling::Mean, normalize: true };
        let v = head.run(&enc, 1);
        assert!((v[0] - 1.0).abs() < 1e-6 && v[1].abs() < 1e-6, "got {v:?}");
    }

    #[test]
    fn mean_pool_excludes_padding() {
        let enc = array![[2.0f32, 0.0], [4.0, 0.0]]; // valid_len=2 -> mean=[3,0]
        let head = EmbedHead { pooling: Pooling::Mean, normalize: false };
        let v = head.run(&enc, 2);
        assert!((v[0] - 3.0).abs() < 1e-6, "got {v:?}");
    }
}

#[cfg(test)]
mod packed_tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn cls_offsets_are_normalized_independently_and_invalid_rows_fail() {
        let spans = PassageSpans::from_lengths(&[2,1,1], 2, 4).unwrap();
        let head = EmbedHead { pooling: Pooling::Cls, normalize: true };
        let enc = array![[3.,4.], [99.,99.], [0.,2.], [4.,3.]];
        let v = head.run_cls_packed(&enc, &spans).unwrap();
        assert_eq!(v, vec![vec![0.6,0.8], vec![0.,1.], vec![0.8,0.6]]);
        assert_eq!(v[0], head.run(&enc.slice(s![..2,..]).to_owned(), 2));
        for invalid in [0., f32::NAN, f32::INFINITY] {
            let bad = Array2::from_elem((4,2), invalid);
            assert!(head.run_cls_packed(&bad, &spans).is_err());
        }
        assert!(head.run_cls_packed(&Array2::zeros((3,2)), &spans).is_err());
        let empty = PassageSpans::from_lengths(&[], 2, 4).unwrap();
        assert!(head.run_cls_packed(&Array2::zeros((0,2)), &empty).unwrap().is_empty());
    }
}
