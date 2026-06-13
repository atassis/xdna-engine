//! Host ops (ndarray f32) — kept local so the reference matches the verified NumPy exactly.
//! Phase 3 replaces the matmuls (`.dot`) with `npu_asr::ctx2` NPU dispatches.

use ndarray::prelude::*;
use rayon::prelude::*;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu_inplace(x: &mut Array2<f32>) {
    x.mapv_inplace(|v| v * (1.0 / (1.0 + (-v).exp())));
}

/// LayerNorm over the last axis (per row), eps 1e-5. Rows in parallel.
pub fn layernorm(x: &Array2<f32>, g: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let (rows, ncol) = x.dim();
    let n = ncol as f32;
    let gs = g.as_slice().unwrap();
    let bs = b.as_slice().unwrap();
    let xs = x.as_standard_layout();
    let xs = xs.as_slice().unwrap();
    let mut out = vec![0f32; rows * ncol];
    out.par_chunks_mut(ncol).enumerate().for_each(|(r, orow)| {
        let row = &xs[r * ncol..r * ncol + ncol];
        let mu = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for c in 0..ncol {
            orow[c] = (row[c] - mu) * inv * gs[c] + bs[c];
        }
    });
    Array2::from_shape_vec((rows, ncol), out).unwrap()
}

/// Generic 2-D conv over [C_in, H, W] -> [C_out, Hout, Wout]; weight [C_out, C_in/groups, kh, kw].
/// Pointwise 1×1 (groups=1, stride=1) is a matmul; otherwise parallel over output channels.
pub fn conv2d(x: &Array3<f32>, w: &Array4<f32>, b: &Array1<f32>, stride: usize, pad: usize, groups: usize) -> Array3<f32> {
    let (ci, hin, win) = x.dim();
    let (co, cig, kh, kw) = w.dim();

    // Fast path: pointwise 1×1 conv == matmul W[Co,Ci] @ x[Ci, H*W].
    if kh == 1 && kw == 1 && groups == 1 && stride == 1 && pad == 0 {
        let w2 = w.to_shape((co, ci)).unwrap().to_owned(); // [Co, Ci]
        let x2 = x.to_shape((ci, hin * win)).unwrap().to_owned(); // [Ci, H*W]
        let mut y = w2.dot(&x2); // [Co, H*W]
        for co_idx in 0..co {
            let bc = b[co_idx];
            y.row_mut(co_idx).iter_mut().for_each(|v| *v += bc);
        }
        return y.to_shape((co, hin, win)).unwrap().to_owned();
    }

    let hout = (hin + 2 * pad - kh) / stride + 1;
    let wout = (win + 2 * pad - kw) / stride + 1;
    let mut xp = Array3::<f32>::zeros((ci, hin + 2 * pad, win + 2 * pad));
    xp.slice_mut(s![.., pad..pad + hin, pad..pad + win]).assign(x);

    let gci = ci / groups;
    let gco = co / groups;
    let st = stride as isize;
    // parallel over output channels; vectorized shift-add over the spatial plane (one strided
    // slice + scaled_add per (in-channel, kernel-tap) instead of per-output-element indexing).
    let planes: Vec<Array2<f32>> = (0..co).into_par_iter().map(|co_idx| {
        let g = co_idx / gco;
        let mut plane = Array2::<f32>::from_elem((hout, wout), b[co_idx]);
        for ic in 0..cig {
            let ci_idx = g * gci + ic;
            let chan = xp.index_axis(Axis(0), ci_idx); // [Hp, Wp]
            for i in 0..kh {
                for j in 0..kw {
                    let wv = w[[co_idx, ic, i, j]];
                    if wv == 0.0 { continue; }
                    // exactly hout×wout elements: start i, step stride, end = last_idx+1
                    let h_end = (i + (hout - 1) * stride + 1) as isize;
                    let w_end = (j + (wout - 1) * stride + 1) as isize;
                    let sub = chan.slice(s![i as isize..h_end; st, j as isize..w_end; st]);
                    plane.scaled_add(wv, &sub);
                }
            }
        }
        plane
    }).collect();
    let mut out = Array3::<f32>::zeros((co, hout, wout));
    for (co_idx, plane) in planes.into_iter().enumerate() {
        out.slice_mut(s![co_idx, .., ..]).assign(&plane);
    }
    out
}

/// Depthwise 1-D conv along time (k taps, pad both sides), per channel. x [C, T], taps [C, k], bias [C].
pub fn dwconv1d(x: &Array2<f32>, taps: &Array2<f32>, bias: &Array1<f32>, k: usize) -> Array2<f32> {
    let (c, t) = x.dim();
    let pad = (k - 1) / 2;
    let mut out = Array2::<f32>::zeros((c, t));
    for ch in 0..c {
        for ti in 0..t {
            let mut acc = bias[ch];
            for j in 0..k {
                // input index = ti + j - pad
                let src = ti as isize + j as isize - pad as isize;
                if src >= 0 && (src as usize) < t {
                    acc += taps[[ch, j]] * x[[ch, src as usize]];
                }
            }
            out[[ch, ti]] = acc;
        }
    }
    out
}

/// rel_shift (NeMo): bd [H, T, 2T-1] -> [H, T, T].
pub fn rel_shift(bd: &Array3<f32>, t: usize) -> Array3<f32> {
    let h = bd.dim().0;
    let p = bd.dim().2; // 2T-1
    // left-pad last dim by 1 -> [H, T, P+1]
    let mut padded = Array3::<f32>::zeros((h, t, p + 1));
    padded.slice_mut(s![.., .., 1..]).assign(bd);
    // reshape [H, P+1, T], drop first row -> [H, P, T], reshape [H, T, P], slice [:, :, :T]
    let mut out = Array3::<f32>::zeros((h, t, t));
    for hh in 0..h {
        // flat row-major over (T, P+1)
        let flat: Vec<f32> = padded.slice(s![hh, .., ..]).iter().copied().collect();
        // view as [P+1, T], drop first row => start at offset T, take P*T elems => view [P, T] then [T, P]
        // Equivalent: element (i,j) of [T,P] after reshape = flat[T + i*P + j]
        for i in 0..t {
            for j in 0..t {
                out[[hh, i, j]] = flat[t + i * p + j];
            }
        }
    }
    out
}
