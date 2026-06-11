//! Pure host-side tensor math for the GigaAM-v3 Conformer encoder (no NPU dependency).
//! These are the "glue" ops the fused encoder runs on the CPU: LayerNorm, RoPE,
//! multi-head attention (scores+softmax+context), GLU, sigmoid/SiLU, and the im2col
//! subsampling front-end. Ported from `npu_asr/fused.py` + `npu_asr/encoder.py`.

use ndarray::prelude::*;

/// bf16 round-to-nearest-even of an f32 (truncate-with-round of the mantissa). Matches numpy
/// ml_dtypes.bfloat16.
pub fn bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return f32::from_bits(((bits >> 16) | 0x0040) << 16);
    }
    let bias = 0x0000_7fff + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(bias)) & 0xffff_0000)
}

/// LayerNorm over the LAST axis with affine. x is [T, D]. gamma,beta len D.
/// var is POPULATION variance (ddof=0).
pub fn layer_norm(x: &Array2<f32>, gamma: &[f32], beta: &[f32], eps: f32) -> Array2<f32> {
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mean = row.sum() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
    out
}

/// LayerNorm NORMALIZE-ONLY over the last axis (no affine). [T,D] -> [T,D].
pub fn layer_norm_normalize(x: &Array2<f32>, eps: f32) -> Array2<f32> {
    let (t, d) = x.dim();
    let mut out = Array2::<f32>::zeros((t, d));
    for i in 0..t {
        let row = x.row(i);
        let mean = row.sum() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = (row[j] - mean) * inv;
        }
    }
    out
}

/// RoPE applied per-head. ln is [T, D] with D = n_heads*head_dim. cos,sin are [T, head_dim].
pub fn rope(
    ln: &Array2<f32>,
    cos: &Array2<f32>,
    sin: &Array2<f32>,
    n_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    let (t, d) = ln.dim();
    let half = head_dim / 2;
    let mut out = Array2::<f32>::zeros((t, d));
    for ti in 0..t {
        for h in 0..n_heads {
            let base = h * head_dim;
            for d_ in 0..head_dim {
                let xr = ln[[ti, base + d_]];
                let rot = if d_ < half {
                    -ln[[ti, base + d_ + half]]
                } else {
                    ln[[ti, base + d_ - half]]
                };
                out[[ti, base + d_]] = xr * cos[[ti, d_]] + rot * sin[[ti, d_]];
            }
        }
    }
    out
}

/// Multi-head scaled-dot-product attention. q,k,v are each [T, D], D = n_heads*head_dim.
pub fn mha(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    n_heads: usize,
    head_dim: usize,
    round_probs: bool,
) -> Array2<f32> {
    use rayon::prelude::*;
    let (t, d) = q.dim();
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Each head is independent — compute per-head context [T,HD] in parallel across cores.
    let ctxs: Vec<Array2<f32>> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let base = h * head_dim;
            let qh = q.slice(s![.., base..base + head_dim]); // [T, HD]
            let kh = k.slice(s![.., base..base + head_dim]);
            let vh = v.slice(s![.., base..base + head_dim]);
            // scores = (qh @ kh^T) * scale  -> [T, T]  (matrixmultiply, not scalar loops)
            let mut sc = qh.dot(&kh.t());
            sc.mapv_inplace(|x| x * scale);
            // row-wise softmax (max-stable), optionally bf16-rounding the probabilities
            for mut row in sc.rows_mut() {
                let mut maxv = f32::NEG_INFINITY;
                for &x in row.iter() {
                    if x > maxv {
                        maxv = x;
                    }
                }
                let mut sum = 0f32;
                for x in row.iter_mut() {
                    *x = (*x - maxv).exp();
                    sum += *x;
                }
                let inv = 1.0 / sum;
                for x in row.iter_mut() {
                    *x *= inv;
                    if round_probs {
                        *x = bf16_round(*x);
                    }
                }
            }
            sc.dot(&vh) // ctx [T, HD]
        })
        .collect();
    let mut out = Array2::<f32>::zeros((t, d));
    for (h, ctx) in ctxs.into_iter().enumerate() {
        let base = h * head_dim;
        out.slice_mut(s![.., base..base + head_dim]).assign(&ctx);
    }
    out
}

/// GLU: x is [2C, T] (channel-major). a = x[..C], g = x[C..]; out[c,t] = a[c,t] * sigmoid(g[c,t]).
pub fn glu(x: &Array2<f32>) -> Array2<f32> {
    let (two_c, t) = x.dim();
    let c = two_c / 2;
    let mut out = Array2::<f32>::zeros((c, t));
    for ci in 0..c {
        for ti in 0..t {
            let a = x[[ci, ti]];
            let g = x[[c + ci, ti]];
            out[[ci, ti]] = a / (1.0 + (-g).exp());
        }
    }
    out
}

/// depthwise conv1d k=5 'same' (pad=2): x[C,T], taps[C,5] -> [C,T] (f32). Parallel over channels.
/// out[c,t] = sum_{i=0..5} taps[c,i] * x[c, t+i-2]  (zero-padded). Matches `host_dwconv_k5` in
/// npu_asr/ops.py. Cheap enough that running it on the host (20 cores) beats the scalar NPU kernel.
pub fn dwconv_k5(x: &Array2<f32>, taps: &Array2<f32>) -> Array2<f32> {
    use rayon::prelude::*;
    let (c, t) = x.dim();
    let rows: Vec<Vec<f32>> = (0..c)
        .into_par_iter()
        .map(|ci| {
            let mut o = vec![0f32; t];
            for (ti, ov) in o.iter_mut().enumerate() {
                let mut acc = 0f32;
                for ki in 0..5usize {
                    let idx = ti as isize + ki as isize - 2;
                    if idx >= 0 && (idx as usize) < t {
                        acc += taps[[ci, ki]] * x[[ci, idx as usize]];
                    }
                }
                *ov = acc;
            }
            o
        })
        .collect();
    let mut out = Array2::<f32>::zeros((c, t));
    for (ci, row) in rows.into_iter().enumerate() {
        for (ti, val) in row.into_iter().enumerate() {
            out[[ci, ti]] = val;
        }
    }
    out
}

/// elementwise sigmoid 1/(1+e^-x)
pub fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// elementwise SiLU x*sigmoid(x)
pub fn silu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// 1D conv via im2col. x is [Cin, L], w is [Cout, Cin, k], b is [Cout].
/// Lout = (L + 2*pad - k)/stride + 1. Output [Cout, Lout].
pub fn im2col_conv1d(
    x: &Array2<f32>,
    w: &Array3<f32>,
    b: &[f32],
    stride: usize,
    pad: usize,
) -> Array2<f32> {
    let (cin, l) = x.dim();
    let (cout, cin_w, k) = w.dim();
    debug_assert_eq!(cin, cin_w);
    let lp = l + 2 * pad;
    let lout = (lp - k) / stride + 1;
    // padded input
    let mut xp = Array2::<f32>::zeros((cin, lp));
    for ci in 0..cin {
        for li in 0..l {
            xp[[ci, li + pad]] = x[[ci, li]];
        }
    }
    // cols: [Lout, Cin*k] with (Cin,k) flattened Cin-major (row = ci*k + ki)
    let mut cols = Array2::<f32>::zeros((lout, cin * k));
    for t in 0..lout {
        let start = t * stride;
        for ci in 0..cin {
            for ki in 0..k {
                cols[[t, ci * k + ki]] = xp[[ci, start + ki]];
            }
        }
    }
    // W2: [Cout, Cin*k] (same flatten order)
    let w2 = w
        .to_shape((cout, cin * k))
        .expect("reshape w")
        .to_owned();
    // out = (cols @ W2.T + b).T  -> [Cout, Lout]
    let prod = cols.dot(&w2.t()); // [Lout, Cout]
    let mut out = Array2::<f32>::zeros((cout, lout));
    for t in 0..lout {
        for co in 0..cout {
            out[[co, t]] = prod[[t, co]] + b[co];
        }
    }
    out
}

#[inline]
fn relu_inplace(a: &mut Array2<f32>) {
    a.mapv_inplace(|v| if v > 0.0 { v } else { 0.0 });
}

/// Subsampling front-end (pre_encode): 2x (conv1d k=5 stride=2 pad=2 + ReLU), then transpose.
/// audio is [64, 1600]. w0 [768,64,5] b0[768]; w2 [768,768,5] b2[768]. Returns [400, 768].
pub fn subsample(
    audio: &Array2<f32>,
    w0: &Array3<f32>,
    b0: &[f32],
    w2: &Array3<f32>,
    b2: &[f32],
) -> Array2<f32> {
    let mut h = im2col_conv1d(audio, w0, b0, 2, 2); // [768, 800]
    relu_inplace(&mut h);
    let mut h = im2col_conv1d(&h, w2, b2, 2, 2); // [768, 400]
    relu_inplace(&mut h);
    h.t().to_owned() // [400, 768]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maxdiff(a: &Array2<f32>, expected: &[f32]) -> f32 {
        a.iter()
            .zip(expected.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn test_bf16_round() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(1.0001), 1.0);
        assert_eq!(bf16_round(1.5), 1.5);
        assert!((bf16_round(3.14159) - 3.140625).abs() < 1e-6);
        assert_eq!(bf16_round(0.0), 0.0);
        assert!((bf16_round(-2.7) - (-2.703125)).abs() < 1e-6);
        assert_eq!(bf16_round(65504.0), 65536.0);
        assert_eq!(bf16_round(1.0000001), 1.0);
        assert!(bf16_round(f32::NAN).is_nan());
    }

    #[test]
    fn test_layer_norm() {
        let x = arr2(&[[1., 2., 3., 4.], [-1., 0., 2., 5.]]);
        let gamma = [1.0, 0.5, 2.0, 1.5];
        let beta = [0.1, -0.2, 0.3, 0.0];
        let out = layer_norm(&x, &gamma, &beta, 1e-5);
        let exp = [
            -1.24163544e+00, -4.23605919e-01, 1.19442368e+00, 2.01245308e+00, -9.91088390e-01,
            -5.27326524e-01, 7.36435413e-01, 2.29128551e+00,
        ];
        assert!(maxdiff(&out, &exp) < 1e-4);
    }

    #[test]
    fn test_layer_norm_normalize() {
        let x = arr2(&[[1., 2., 3., 4.], [-1., 0., 2., 5.]]);
        let out = layer_norm_normalize(&x, 1e-5);
        let exp = [
            -1.34163547e+00, -4.47211832e-01, 4.47211832e-01, 1.34163547e+00, -1.09108841e+00,
            -6.54653072e-01, 2.18217686e-01, 1.52752376e+00,
        ];
        assert!(maxdiff(&out, &exp) < 1e-4);
    }

    #[test]
    fn test_rope() {
        let ln = Array2::from_shape_vec(
            (2, 8),
            vec![
                -5.00000000e-01, -4.00000006e-01, -3.00000012e-01, -1.99999988e-01,
                -9.99999940e-02, 0.00000000e+00, 1.00000024e-01, 1.99999988e-01, 3.00000012e-01,
                4.00000036e-01, 5.00000000e-01, 6.00000024e-01, 7.00000048e-01, 8.00000072e-01,
                8.99999976e-01, 1.00000000e+00,
            ],
        )
        .unwrap();
        let cos = Array2::from_shape_vec(
            (2, 4),
            vec![
                1.00000000e+00, 9.55336511e-01, 8.25335622e-01, 6.21609926e-01, 3.62357736e-01,
                7.07371980e-02, -2.27202162e-01, -5.04846215e-01,
            ],
        )
        .unwrap();
        let sin = Array2::from_shape_vec(
            (2, 4),
            vec![
                0.00000000e+00, 2.95520216e-01, 5.64642489e-01, 7.83326924e-01, 9.32039082e-01,
                9.97494996e-01, 9.73847628e-01, 8.63209307e-01,
            ],
        )
        .unwrap();
        let out = rope(&ln, &cos, &sin, 2, 4);
        let exp = [
            -5.00000000e-01, -3.23030591e-01, -5.29921949e-01, -4.37652737e-01, -9.99999940e-02,
            -5.91040403e-02, 2.60693394e-02, 1.24321975e-01, -3.57312202e-01, -5.70202172e-01,
            1.78553224e-01, 4.23760116e-02, -5.85184753e-01, -9.40905213e-01, 4.77211416e-01,
            1.85721278e-01,
        ];
        assert!(maxdiff(&out, &exp) < 1e-4);
    }

    fn mha_inputs() -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = Array2::from_shape_vec(
            (3, 8),
            vec![
                1.76405239e+00, 4.00157213e-01, 9.78738010e-01, 2.24089313e+00, 1.86755800e+00,
                -9.77277875e-01, 9.50088441e-01, -1.51357204e-01, -1.03218853e-01, 4.10598516e-01,
                1.44043565e-01, 1.45427346e+00, 7.61037707e-01, 1.21675014e-01, 4.43863243e-01,
                3.33674341e-01, 1.49407911e+00, -2.05158263e-01, 3.13067704e-01, -8.54095757e-01,
                -2.55298972e+00, 6.53618574e-01, 8.64436209e-01, -7.42165029e-01,
            ],
        )
        .unwrap();
        let k = Array2::from_shape_vec(
            (3, 8),
            vec![
                2.26975465e+00, -1.45436573e+00, 4.57585156e-02, -1.87183857e-01, 1.53277922e+00,
                1.46935880e+00, 1.54947430e-01, 3.78162533e-01, -8.87785733e-01, -1.98079646e+00,
                -3.47912163e-01, 1.56348974e-01, 1.23029065e+00, 1.20237982e+00, -3.87326807e-01,
                -3.02302748e-01, -1.04855299e+00, -1.42001796e+00, -1.70627022e+00,
                1.95077538e+00, -5.09652197e-01, -4.38074291e-01, -1.25279534e+00, 7.77490377e-01,
            ],
        )
        .unwrap();
        let v = Array2::from_shape_vec(
            (3, 8),
            vec![
                -1.61389780e+00, -2.12740287e-01, -8.95466566e-01, 3.86902511e-01,
                -5.10805130e-01, -1.18063223e+00, -2.81822290e-02, 4.28331882e-01, 6.65172189e-02,
                3.02471906e-01, -6.34322107e-01, -3.62741172e-01, -6.72460437e-01,
                -3.59553158e-01, -8.13146293e-01, -1.72628260e+00, 1.77426144e-01,
                -4.01780933e-01, -1.63019836e+00, 4.62782264e-01, -9.07298386e-01, 5.19453958e-02,
                7.29090571e-01, 1.28982916e-01,
            ],
        )
        .unwrap();
        (q, k, v)
    }

    #[test]
    fn test_mha_no_round() {
        let (q, k, v) = mha_inputs();
        let out = mha(&q, &k, &v, 2, 4, false);
        let exp = [
            -1.18690836e+00, -2.22432256e-01, -1.02206445e+00, 3.63069683e-01, -6.10106885e-01,
            -7.53903568e-01, -2.43777782e-01, -3.99768412e-01, -8.78197402e-02, -2.48804614e-01,
            -1.34988821e+00, 3.03411782e-01, -6.29526436e-01, -7.01800525e-01, -1.78305149e-01,
            -3.61683249e-01, -1.44960690e+00, -1.79372147e-01, -8.93268466e-01, 3.33763868e-01,
            -7.83013344e-01, -2.58810401e-01, 2.32112199e-01, -2.59631425e-01,
        ];
        assert!(maxdiff(&out, &exp) < 1e-4);
    }

    #[test]
    fn test_mha_round() {
        let (q, k, v) = mha_inputs();
        let out = mha(&q, &k, &v, 2, 4, true);
        let exp = [
            -1.18585062e+00, -2.22294509e-01, -1.02140045e+00, 3.62818062e-01, -6.11286521e-01,
            -7.55464554e-01, -2.44101077e-01, -4.00213450e-01, -8.73454213e-02, -2.49094710e-01,
            -1.35141969e+00, 3.03717583e-01, -6.28988981e-01, -7.01031983e-01, -1.77443653e-01,
            -3.60385418e-01, -1.44732535e+00, -1.79080769e-01, -8.92088592e-01, 3.33228081e-01,
            -7.83035755e-01, -2.58758098e-01, 2.32214749e-01, -2.59537369e-01,
        ];
        assert!(maxdiff(&out, &exp) < 1e-4);
    }

    #[test]
    fn test_glu() {
        let x = arr2(&[
            [1.0, -1.0, 0.5],
            [2.0, 0.0, -0.5],
            [0.30000001, 1.0, -2.0],
            [-1.0, 0.5, 1.5],
        ]);
        let out = glu(&x);
        let exp = [
            5.74442506e-01, -7.31058598e-01, 5.96014671e-02, 5.37882805e-01, 0.00000000e+00,
            -4.08787251e-01,
        ];
        assert!(maxdiff(&out, &exp) < 1e-4);
    }

    #[test]
    fn test_sigmoid_silu() {
        let x = arr2(&[[-2., 0., 1.], [3., -1., 0.5]]);
        let sig = sigmoid(&x);
        let sig_exp = [
            1.19202934e-01, 5.00000000e-01, 7.31058598e-01, 9.52574134e-01, 2.68941402e-01,
            6.22459352e-01,
        ];
        assert!(maxdiff(&sig, &sig_exp) < 1e-4);
        let si = silu(&x);
        let silu_exp = [
            -2.38405868e-01, 0.00000000e+00, 7.31058598e-01, 2.85772252e+00, -2.68941402e-01,
            3.11229676e-01,
        ];
        assert!(maxdiff(&si, &silu_exp) < 1e-4);
    }

    #[test]
    fn test_im2col_conv1d() {
        let x = Array2::from_shape_vec(
            (2, 6),
            vec![
                1.13940072e+00, -1.23482585e+00, 4.02341634e-01, -6.84810102e-01,
                -8.70797157e-01, -5.78849673e-01, -3.11552525e-01, 5.61653413e-02,
                -1.16514981e+00, 9.00826514e-01, 4.65662450e-01, -1.53624368e+00,
            ],
        )
        .unwrap();
        let w = Array3::from_shape_vec(
            (3, 2, 3),
            vec![
                1.48825216e+00, 1.89588916e+00, 1.17877960e+00, -1.79924831e-01, -1.07075262e+00,
                1.05445170e+00, -4.03176934e-01, 1.22244501e+00, 2.08274975e-01, 9.76639032e-01,
                3.56366396e-01, 7.06573188e-01, 1.05000203e-02, 1.78587055e+00, 1.26912087e-01,
                4.01989371e-01, 1.88315070e+00, -1.34775901e+00,
            ],
        )
        .unwrap();
        let b = [-1.27048504e+00, 9.69396710e-01, -1.17312336e+00];
        let out = im2col_conv1d(&x, &w, &b, 2, 1);
        assert_eq!(out.dim(), (3, 3));
        let exp = [
            -1.73075795e-01, -9.65302646e-01, -6.90351105e+00, 2.03372622e+00, 2.09259462e+00,
            2.06949711e-02, 4.25869226e-02, -3.94014144e+00, 5.00613809e-01,
        ];
        assert!(maxdiff(&out, &exp) < 1e-4);
    }

    #[test]
    fn test_subsample() {
        // Reproducible closed-form inputs matching /tmp/ref3.py.
        let audio = Array2::from_shape_fn((64, 1600), |(i, j)| {
            (0.01 * (i as f32 * 1600.0 + j as f32)).sin()
        });
        let w0 = Array3::from_shape_fn((768, 64, 5), |(o, i, k)| {
            (0.01 * (0.001 * (o as f32 * 320.0 + i as f32 * 5.0 + k as f32)).cos()) as f32
        });
        let b0: Vec<f32> = (0..768).map(|o| 0.001 * (0.1 * o as f32).sin()).collect();
        let w2 = Array3::from_shape_fn((768, 768, 5), |(o, i, k)| {
            0.001 * (0.0005 * (o as f32 * 768.0 * 5.0 + i as f32 * 5.0 + k as f32)).cos()
        });
        let b2: Vec<f32> = (0..768).map(|o| 0.001 * (0.1 * o as f32).cos()).collect();

        let out = subsample(&audio, &w0, &b0, &w2, &b2);
        assert_eq!(out.dim(), (400, 768));

        // a few sampled non-zero elements
        assert!((out[[0, 0]] - 2.32914230e-03).abs() < 1e-4);
        assert!((out[[79, 745]] - 2.15513725e-03).abs() < 1e-4);
        assert!((out[[160, 26]] - 2.65060668e-03).abs() < 1e-4);
        assert!((out[[239, 753]] - 9.77828051e-04).abs() < 1e-4);
        assert!((out[[320, 39]] - 3.09958728e-03).abs() < 1e-4);

        // global checksum
        let sum: f32 = out.iter().sum();
        assert!((sum - 6.350164e+02).abs() / 6.350164e+02 < 1e-3);
    }
}
