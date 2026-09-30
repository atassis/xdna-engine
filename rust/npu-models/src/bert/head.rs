//! Embedding head: pool encoder outputs over valid tokens, optional L2-normalize.

use ndarray::Array2;
use crate::pipeline::Head;

pub struct EmbedHead {
    pub pooling: Pooling,
    pub normalize: bool,
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
