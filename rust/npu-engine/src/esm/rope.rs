//! ESM-2 RoPE: HF rotate-half convention. theta=10000. Applied to full head_dim.
use ndarray::Array2;

/// Precompute cos/sin tables of shape [seq, head_dim] for positions 0..seq.
pub fn tables(seq: usize, head_dim: usize) -> (Array2<f32>, Array2<f32>) {
    let half = head_dim / 2;
    let mut cos = Array2::<f32>::zeros((seq, head_dim));
    let mut sin = Array2::<f32>::zeros((seq, head_dim));
    for p in 0..seq {
        for i in 0..half {
            let freq = 1.0f32 / 10000f32.powf(2.0 * i as f32 / head_dim as f32);
            let a = p as f32 * freq;
            let (s, c) = a.sin_cos();
            cos[[p, i]] = c;
            cos[[p, i + half]] = c;
            sin[[p, i]] = s;
            sin[[p, i + half]] = s;
        }
    }
    (cos, sin)
}

/// Apply RoPE to x[seq, head_dim] in place: x*cos + rotate_half(x)*sin.
/// rotate_half([a,b]) = [-b, a] over the two halves.
pub fn apply(x: &mut Array2<f32>, cos: &Array2<f32>, sin: &Array2<f32>) {
    let (seq, hd) = x.dim();
    let half = hd / 2;
    for p in 0..seq {
        let row: Vec<f32> = x.row(p).to_vec();
        for i in 0..hd {
            let rot = if i < half { -row[i + half] } else { row[i - half] };
            x[[p, i]] = row[i] * cos[[p, i]] + rot * sin[[p, i]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn pos0_is_identity() {
        // at position 0, cos=1 sin=0 -> apply is identity
        let (cos, sin) = tables(1, 4);
        let mut x = array![[1.0f32, 2.0, 3.0, 4.0]];
        apply(&mut x, &cos, &sin);
        for (a, b) in x.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn rotate_half_shape() {
        let (cos, sin) = tables(8, 16);
        assert_eq!(cos.dim(), (8, 16));
        assert_eq!(sin.dim(), (8, 16));
    }
}
