//! VBx clustering: pyannote community-1's stage, in place of 3.1's agglomerative linkage.
//!
//! Two parts, both dense linear algebra:
//!
//!   1. **PLDA transform.** The 256-d embedding is centred, LDA-projected to 128-d, length-
//!      normalised twice and mapped into the PLDA latent space. In that space the model's
//!      within-class covariance is the IDENTITY and the across-class covariance is DIAGONAL
//!      (`psi`), which is what keeps step 2 free of matrix inversions.
//!   2. **Variational Bayes over an HMM.** Agglomerative clustering supplies the initial
//!      responsibilities; the VB loop then re-estimates speaker models and responsibilities until
//!      the ELBO stops improving. The speaker COUNT falls out of the objective -- `Fb` regularises
//!      it -- rather than being cut from a dendrogram at a threshold.
//!
//! The generalized eigenproblem that produces `plda_tr`/`plda_psi` is solved by the EXPORT script,
//! not here: it is a one-time transform of two fixed files, so shipping its result keeps this file
//! to matmul and avoids an eigensolver dependency for something that never varies at runtime.

use ndarray::{Array1, Array2};

use crate::api::EngineError;
use crate::diarize::types::Clusterer;

/// The learned matrices, loaded from the artifacts the export script wrote.
pub struct Plda {
    pub xvec_mean1: Array1<f32>,   // [256]
    pub xvec_lda: Array2<f32>,     // [256, 128]
    pub xvec_mean2: Array1<f32>,   // [128]
    pub plda_mu: Array1<f32>,      // [128]
    pub plda_tr: Array2<f32>,      // [128, 128]
    pub plda_psi: Array1<f32>,     // [128]
}

impl Plda {
    /// Load the matrices the export script wrote, resolving names against the manifest's dir.
    pub fn load(cfg: &crate::diarize::types::PldaCfg, dir: &std::path::Path)
        -> Result<Plda, EngineError> {
        let v1 = |n: &str| -> Result<Array1<f32>, EngineError> {
            ndarray_npy::read_npy(dir.join(n))
                .map_err(|e| EngineError::Load(format!("plda {n}: {e}")))
        };
        let v2 = |n: &str| -> Result<Array2<f32>, EngineError> {
            ndarray_npy::read_npy(dir.join(n))
                .map_err(|e| EngineError::Load(format!("plda {n}: {e}")))
        };
        Ok(Plda {
            xvec_mean1: v1(&cfg.xvec_mean1)?,
            xvec_lda: v2(&cfg.xvec_lda)?,
            xvec_mean2: v1(&cfg.xvec_mean2)?,
            plda_mu: v1(&cfg.plda_mu)?,
            plda_tr: v2(&cfg.plda_tr)?,
            plda_psi: v1(&cfg.plda_psi)?,
        })
    }
}

fn l2_normalise_rows(mut x: Array2<f32>) -> Array2<f32> {
    for mut row in x.rows_mut() {
        let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|v| v / n);
    }
    x
}

impl Plda {
    /// `[n, 256]` embeddings -> `[n, lda_dim]` features in the PLDA latent space.
    ///
    /// Mirrors pyannote's `xvec_tf` then `plda_tf`. The two `sqrt(dim) * l2_norm(..)` steps are
    /// not decoration: they are what makes the unit-variance assumption behind the VB model hold.
    pub fn transform(&self, emb: &Array2<f32>, lda_dim: usize) -> Array2<f32> {
        let d_in = self.xvec_lda.nrows();
        let d_out = self.xvec_lda.ncols();

        let centred = emb - &self.xvec_mean1;
        let y = l2_normalise_rows(centred) * (d_in as f32).sqrt();
        let projected = y.dot(&self.xvec_lda) - &self.xvec_mean2;
        let y = l2_normalise_rows(projected) * (d_out as f32).sqrt();

        let latent = (y - &self.plda_mu).dot(&self.plda_tr.t());
        let k = lda_dim.min(latent.ncols());
        latent.slice(ndarray::s![.., ..k]).to_owned()
    }
}

fn logsumexp_rows(x: &Array2<f32>) -> Array1<f32> {
    let mut out = Array1::zeros(x.nrows());
    for (i, row) in x.rows().into_iter().enumerate() {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !m.is_finite() { out[i] = m; continue }
        let s: f32 = row.iter().map(|v| (v - m).exp()).sum();
        out[i] = m + s.max(1e-30).ln();
    }
    out
}

/// The VB loop. `x` is `[T, D]` in the PLDA space, `phi` the `[D]` across-class diagonal.
/// `gamma` enters as the initial responsibilities `[T, K]` and is refined in place.
pub fn vbx(x: &Array2<f32>, phi: &Array1<f32>, fa: f32, fb: f32,
           mut gamma: Array2<f32>, max_iters: usize, epsilon: f32) -> Array2<f32> {
    let (t, d) = x.dim();
    if t == 0 { return gamma }
    let mut pi: Array1<f32> = {
        let k = gamma.ncols();
        Array1::from_elem(k, 1.0 / k as f32)
    };

    // Per-frame constant of the log-likelihood; hoisted because it never changes.
    let g: Array1<f32> = x.rows().into_iter()
        .map(|r| -0.5 * (r.iter().map(|v| v * v).sum::<f32>()
                         + d as f32 * (2.0 * std::f32::consts::PI).ln()))
        .collect::<Vec<_>>().into();
    let v = phi.mapv(f32::sqrt);
    let rho = x * &v;

    let mut prev_elbo = f32::NEG_INFINITY;
    for it in 0..max_iters.max(1) {
        // (17) and (16): speaker models from the current responsibilities.
        let counts = gamma.sum_axis(ndarray::Axis(0));                     // [K]
        let inv_l: Array2<f32> = Array2::from_shape_fn((gamma.ncols(), d), |(k, j)| {
            1.0 / (1.0 + fa / fb * counts[k] * phi[j])
        });
        let alpha: Array2<f32> = &inv_l * &(gamma.t().dot(&rho) * (fa / fb));  // [K, D]

        // (23): per-frame, per-speaker log-likelihood.
        let quad: Array1<f32> = (&inv_l + &alpha.mapv(|a| a * a)).dot(phi);    // [K]
        let mut log_p = rho.dot(&alpha.t());                                   // [T, K]
        for ((i, k), val) in log_p.indexed_iter_mut() {
            *val = fa * (*val - 0.5 * quad[k] + g[i]);
        }

        let lpi = pi.mapv(|p| (p + 1e-8).ln());
        let shifted = &log_p + &lpi;
        let log_p_x = logsumexp_rows(&shifted);
        gamma = Array2::from_shape_fn(shifted.dim(), |(i, k)| (shifted[[i, k]] - log_p_x[i]).exp());

        pi = gamma.sum_axis(ndarray::Axis(0));
        let s = pi.sum().max(1e-30);
        pi.mapv_inplace(|p| p / s);

        // (25): the ELBO, only used as the stopping rule.
        let reg: f32 = inv_l.iter().zip(alpha.iter())
            .map(|(l, a)| l.max(1e-30).ln() - l - a * a + 1.0).sum();
        let elbo = log_p_x.sum() + fb * 0.5 * reg;
        if it > 0 && (elbo - prev_elbo) < epsilon { break }
        prev_elbo = elbo;
    }
    gamma
}

/// community-1's clusterer.
pub struct VbxClusterer {
    pub plda: Plda,
    pub threshold: f32,
    pub fa: f32,
    pub fb: f32,
    pub max_iters: usize,
    pub init_smoothing: f32,
    pub lda_dim: usize,
}

impl Clusterer for VbxClusterer {
    fn cluster(&self, embeddings: &Array2<f32>) -> Result<Vec<u32>, EngineError> {
        let n = embeddings.nrows();
        if n == 0 { return Ok(Vec::new()) }
        if n == 1 { return Ok(vec![0]) }

        let fea = self.plda.transform(embeddings, self.lda_dim);
        let phi = self.plda.plda_psi.slice(ndarray::s![..fea.ncols()]).to_owned();

        // Agglomerative init, exactly as upstream: VB refines an existing partition rather than
        // starting from noise, so the same linkage 3.1 ships is reused here as a seed.
        //
        // EUCLIDEAN, not cosine. Upstream normalises the embeddings and runs centroid linkage with
        // metric="euclidean", cutting at `threshold`. Reading community-1's 0.6 as a cosine cut
        // merges every speaker into one cluster, and VB cannot split what the init handed it.
        let init = super::cluster::cluster_with(
            embeddings, self.threshold, 1, super::cluster::Metric::Euclidean);
        let k = super::cluster::n_clusters(&init).max(1);

        // One-hot, then softened: hard 0/1 responsibilities give the first VB step no gradient to
        // move a frame between speakers.
        let mut gamma = Array2::<f32>::zeros((n, k));
        for (i, &c) in init.iter().enumerate() { gamma[[i, c as usize]] = self.init_smoothing; }
        for mut row in gamma.rows_mut() {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            row.mapv_inplace(|v| { let e = (v - m).exp(); s += e; e });
            row.mapv_inplace(|v| v / s.max(1e-30));
        }

        let debug = std::env::var_os("NPU_DIARIZE_DEBUG").is_some();
        if debug {
            eprintln!("[vbx] {} crops, fea {:?}, phi[0..4]={:?}", n, fea.dim(),
                &phi.iter().take(4).collect::<Vec<_>>());
            for i in 0..n.min(6) {
                for j in (i + 1)..n.min(6) {
                    let (a, b) = (embeddings.row(i), embeddings.row(j));
                    let eu: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
                    let cs: f32 = 1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
                    eprintln!("[vbx]   emb {i}-{j}: euclid {eu:.4} cosdist {cs:.4} (thr {:.3})",
                        self.threshold);
                }
            }
            eprintln!("[vbx] ahc init -> {:?} ({} clusters)", init, k);
        }

        if debug {
            // Dump the exact VB inputs so the loop can be checked against the reference
            // implementation on identical data -- that separates a wrong PLDA transform from a
            // wrong VB iteration, which look the same from the outside.
            let _ = ndarray_npy::write_npy("/tmp/vbx_fea.npy", &fea);
            let _ = ndarray_npy::write_npy("/tmp/vbx_phi.npy", &phi);
            let _ = ndarray_npy::write_npy("/tmp/vbx_gamma_init.npy", &gamma);
            let _ = ndarray_npy::write_npy("/tmp/vbx_emb.npy", embeddings);
        }
        let gamma = vbx(&fea, &phi, self.fa, self.fb, gamma, self.max_iters, 1e-4);
        if debug {
            let _ = ndarray_npy::write_npy("/tmp/vbx_gamma_out.npy", &gamma);
            eprintln!("[vbx] gamma rows: {:?}", (0..n.min(6))
                .map(|i| gamma.row(i).iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>())
                .collect::<Vec<_>>());
        }

        // Hard-assign, then drop speakers VB emptied out and compact the labels.
        let mut labels: Vec<u32> = (0..n).map(|i| {
            let row = gamma.row(i);
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (k, &v) in row.iter().enumerate() { if v > bv { bv = v; best = k; } }
            best as u32
        }).collect();
        let mut map = std::collections::BTreeMap::new();
        for l in labels.iter() { let next = map.len() as u32; map.entry(*l).or_insert(next); }
        for l in labels.iter_mut() { *l = map[l]; }
        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};

    /// An identity PLDA: no centring, no projection, no rotation. Lets the VB loop be tested on
    /// known geometry without depending on the shipped matrices.
    fn identity_plda(d: usize) -> Plda {
        Plda {
            xvec_mean1: Array1::zeros(d),
            xvec_lda: Array2::from_shape_fn((d, d), |(i, j)| if i == j { 1.0 } else { 0.0 }),
            xvec_mean2: Array1::zeros(d),
            plda_mu: Array1::zeros(d),
            plda_tr: Array2::from_shape_fn((d, d), |(i, j)| if i == j { 1.0 } else { 0.0 }),
            plda_psi: Array1::from_elem(d, 1.0),
        }
    }

    #[test]
    fn the_transform_length_normalises_to_the_expected_scale() {
        // xvec_tf scales by sqrt(dim) after each l2-norm; with an identity LDA the rows should
        // come out with norm sqrt(d), which is the unit-variance assumption the VB model needs.
        let p = identity_plda(4);
        let e = arr2(&[[3.0f32, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]]);
        let f = p.transform(&e, 4);
        for row in f.rows() {
            let n = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 2.0).abs() < 1e-4, "expected norm sqrt(4)=2, got {n}");
        }
    }

    #[test]
    fn transform_truncates_to_lda_dim() {
        let p = identity_plda(6);
        let e = arr2(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]]);
        assert_eq!(p.transform(&e, 3).ncols(), 3);
        assert_eq!(p.transform(&e, 99).ncols(), 6, "a larger lda_dim must clamp, not panic");
    }

    #[test]
    fn vb_keeps_two_well_separated_groups_apart() {
        // Two tight clusters far apart; VB started from the correct partition must not merge them.
        let x = arr2(&[[ 5.0f32, 0.0], [ 5.1, 0.1], [-5.0, 0.0], [-5.1, -0.1]]);
        let phi = arr1(&[1.0f32, 1.0]);
        let mut g = Array2::<f32>::zeros((4, 2));
        for (i, k) in [(0, 0), (1, 0), (2, 1), (3, 1)] { g[[i, k]] = 1.0; }
        let out = vbx(&x, &phi, 0.07, 0.8, g, 20, 1e-4);
        let hard: Vec<usize> = (0..4).map(|i| {
            let r = out.row(i);
            (0..r.len()).max_by(|a, b| r[*a].partial_cmp(&r[*b]).unwrap()).unwrap()
        }).collect();
        assert_eq!(hard[0], hard[1], "{hard:?}");
        assert_eq!(hard[2], hard[3], "{hard:?}");
        assert_ne!(hard[0], hard[2], "separated groups must not merge: {hard:?}");
    }

    #[test]
    fn responsibilities_stay_a_distribution() {
        let x = arr2(&[[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0]]);
        let phi = arr1(&[1.0f32, 1.0]);
        let g = Array2::from_elem((3, 3), 1.0 / 3.0);
        let out = vbx(&x, &phi, 0.07, 0.8, g, 10, 1e-4);
        for row in out.rows() {
            let s: f32 = row.sum();
            assert!((s - 1.0).abs() < 1e-4, "rows must sum to 1, got {s}");
            assert!(row.iter().all(|v| *v >= 0.0 && v.is_finite()), "no NaN or negative mass");
        }
    }

    #[test]
    fn empty_and_single_inputs_do_not_panic() {
        let phi = arr1(&[1.0f32]);
        let e = Array2::<f32>::zeros((0, 1));
        assert_eq!(vbx(&e, &phi, 0.07, 0.8, Array2::zeros((0, 1)), 5, 1e-4).nrows(), 0);
        let c = VbxClusterer { plda: identity_plda(2), threshold: 0.6, fa: 0.07, fb: 0.8,
                               max_iters: 5, init_smoothing: 7.0, lda_dim: 2 };
        assert!(c.cluster(&Array2::<f32>::zeros((0, 2))).unwrap().is_empty());
        assert_eq!(c.cluster(&arr2(&[[1.0f32, 0.0]])).unwrap(), vec![0]);
    }

    #[test]
    fn labels_come_out_compact() {
        let c = VbxClusterer { plda: identity_plda(2), threshold: 0.6, fa: 0.07, fb: 0.8,
                               max_iters: 20, init_smoothing: 7.0, lda_dim: 2 };
        let e = arr2(&[[1.0f32, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, -0.1]]);
        let l = c.cluster(&e).unwrap();
        let max = *l.iter().max().unwrap() as usize;
        assert_eq!(max + 1, l.iter().collect::<std::collections::BTreeSet<_>>().len(),
            "gaps in labels break downstream sizing: {l:?}");
    }
}
