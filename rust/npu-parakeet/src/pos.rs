//! Relative positional encoding (NeMo RelPositionalEncoding): length 2T-1, positions
//! [T-1 .. 0 .. -(T-1)] (positive reversed, then negative), sinusoidal. Verified rel 1.7e-7
//! vs the extracted `pos_enc` ref. d must be even.

use ndarray::prelude::*;

pub fn rel_pos_encoding(t: usize, d: usize) -> Array2<f32> {
    let half = d / 2;
    // div[i] = exp(2i * -ln(10000)/d) for i in 0..half  (over even indices 0,2,..)
    let div: Vec<f64> = (0..half).map(|i| (-(10000f64.ln()) * (2 * i) as f64 / d as f64).exp()).collect();

    let row = |signed_pos: f64| -> Vec<f64> {
        let mut r = vec![0f64; d];
        for i in 0..half {
            let a = signed_pos * div[i];
            r[2 * i] = a.sin();
            r[2 * i + 1] = a.cos();
        }
        r
    };

    let mut pe = Array2::<f32>::zeros((2 * t - 1, d));
    // positive positions reversed: pos = T-1, T-2, ..., 0  -> rows 0..T
    for k in 0..t {
        let pos = (t - 1 - k) as f64;
        let r = row(pos);
        for j in 0..d {
            pe[[k, j]] = r[j] as f32;
        }
    }
    // negative positions: pos = -1, -2, ..., -(T-1) -> rows T..2T-1
    for k in 1..t {
        let pos = -(k as f64);
        let r = row(pos);
        for j in 0..d {
            pe[[t - 1 + k, j]] = r[j] as f32;
        }
    }
    pe
}
