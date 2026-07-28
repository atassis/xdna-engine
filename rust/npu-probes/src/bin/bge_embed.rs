//! bge-base-en-v1.5 sentence embedding on the engine (host f32 first) — the IN-ENVELOPE target.
//!
//! BERT encoder, dimension-identical to whisper (768/12/12/3072), POST-norm, gelu, bidirectional,
//! CLS-pooled + L2-normalized embedding. Weights via scripts/convert_bge.py. Validates cosine
//! similarity against the HF golden (scripts/bge_reference.py -> ref_emb0.npy). This is the workload
//! the NPU is FOR (encoder = M=seq occupancy-bound; NPU GEMM routing actually helps, unlike M=1 decode).
//!
//!   cargo run -p npu-whisper --release --bin bge_embed

use ndarray::prelude::*;
use ndarray_npy::read_npy;
use npu_asr_host::{gelu, layer_norm};

#[cfg(feature = "npu")]
use npu_asr::ctx2::{CtxAOp, Epi, FfnMm2};
#[cfg(feature = "npu")]
use npu_whisper::npu::{apply_tiled, apply_tiled_mm2, WhisperNpu};

const NL: usize = 12;
const NH: usize = 12;
const HD: usize = 64;
const D: usize = 768;
const EPS: f32 = 1e-12;

fn l2(p: String) -> Array2<f32> { read_npy(&p).unwrap_or_else(|e| panic!("load {p}: {e}")) }
fn l1(p: String) -> Array1<f32> { read_npy(&p).unwrap_or_else(|e| panic!("load {p}: {e}")) }

struct Lin { w: Array2<f32>, b: Array1<f32> }
impl Lin {
    fn load(d: &str, n: &str) -> Self { Lin { w: l2(format!("{d}/{n}.weight.npy")), b: l1(format!("{d}/{n}.bias.npy")) } }
    fn fwd(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut y = x.dot(&self.w);
        y += &self.b.view().insert_axis(Axis(0));
        y
    }
}

struct Layer {
    q: Lin, k: Lin, v: Lin,
    attn_out: Lin, attn_ln_w: Array1<f32>, attn_ln_b: Array1<f32>,
    inter: Lin, out: Lin, out_ln_w: Array1<f32>, out_ln_b: Array1<f32>,
}

/// NPU GEMM ops for one BERT layer (q/k/v/attn_out: K=768; inter: K=768->3072; out: FfnMm2 K=3072->768).
/// Bias applied on-chip (Epi::Bias). Mirrors the whisper-encoder BlockOps. Attention + LN stay host.
#[cfg(feature = "npu")]
struct LayerNpu { q: CtxAOp, k: CtxAOp, v: CtxAOp, attn_out: CtxAOp, inter: CtxAOp, out: FfnMm2 }

struct Bge {
    word: Array2<f32>, pos: Array2<f32>, tt: Array2<f32>,
    emb_ln_w: Array1<f32>, emb_ln_b: Array1<f32>,
    layers: Vec<Layer>,
    // BGE_NPU=1: route the 6 GEMMs/layer through the NPU (reuse the whisper-encoder ctx2 K=768 path).
    #[cfg(feature = "npu")]
    npu: Option<(WhisperNpu, Vec<LayerNpu>)>,
}

impl Bge {
    fn load(root: &str) -> Self {
        let layers: Vec<Layer> = (0..NL).map(|l| {
            let d = format!("{root}/L{l}");
            Layer {
                q: Lin::load(&d, "q"), k: Lin::load(&d, "k"), v: Lin::load(&d, "v"),
                attn_out: Lin::load(&d, "attn_out"),
                attn_ln_w: l1(format!("{d}/attn_ln.weight.npy")), attn_ln_b: l1(format!("{d}/attn_ln.bias.npy")),
                inter: Lin::load(&d, "inter"), out: Lin::load(&d, "out"),
                out_ln_w: l1(format!("{d}/out_ln.weight.npy")), out_ln_b: l1(format!("{d}/out_ln.bias.npy")),
            }
        }).collect();
        #[cfg(feature = "npu")]
        let npu = if std::env::var("BGE_NPU").is_ok() {
            let wn = WhisperNpu::open(std::path::Path::new("."));
            let sh = wn.shared.clone();
            let bs = |a: &Array1<f32>| a.as_slice().unwrap().to_vec();
            let nl: Vec<LayerNpu> = layers.iter().map(|ly| LayerNpu {
                q: CtxAOp::new(sh.clone(), &ly.q.w, D, Epi::Bias, &bs(&ly.q.b)),
                k: CtxAOp::new(sh.clone(), &ly.k.w, D, Epi::Bias, &bs(&ly.k.b)),
                v: CtxAOp::new(sh.clone(), &ly.v.w, D, Epi::Bias, &bs(&ly.v.b)),
                attn_out: CtxAOp::new(sh.clone(), &ly.attn_out.w, D, Epi::Bias, &bs(&ly.attn_out.b)),
                inter: CtxAOp::new(sh.clone(), &ly.inter.w, 3072, Epi::Bias, &bs(&ly.inter.b)),
                out: FfnMm2::new(sh.clone(), &ly.out.w, &bs(&ly.out.b)),
            }).collect();
            eprintln!("[bge] NPU GEMM routing ON (BGE_NPU): 6 GEMMs/layer via ctx2 K=768");
            Some((wn, nl))
        } else { None };

        Bge {
            word: l2(format!("{root}/word_emb.npy")), pos: l2(format!("{root}/pos_emb.npy")),
            tt: l2(format!("{root}/tok_type_emb.npy")),
            emb_ln_w: l1(format!("{root}/emb_ln.weight.npy")), emb_ln_b: l1(format!("{root}/emb_ln.bias.npy")),
            layers,
            #[cfg(feature = "npu")]
            npu,
        }
    }

    // GEMM dispatch: NPU (apply_tiled, bias on-chip) when BGE_NPU, else host Lin.fwd (bias added).
    fn g_qkv(&self, x: &Array2<f32>, l: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        #[cfg(feature = "npu")]
        if let Some((_, nl)) = &self.npu {
            return (apply_tiled(&nl[l].q, x, D), apply_tiled(&nl[l].k, x, D), apply_tiled(&nl[l].v, x, D));
        }
        let ly = &self.layers[l];
        (ly.q.fwd(x), ly.k.fwd(x), ly.v.fwd(x))
    }
    fn g_attn_out(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        #[cfg(feature = "npu")]
        if let Some((_, nl)) = &self.npu { return apply_tiled(&nl[l].attn_out, x, D); }
        self.layers[l].attn_out.fwd(x)
    }
    fn g_inter(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        #[cfg(feature = "npu")]
        if let Some((_, nl)) = &self.npu { return apply_tiled(&nl[l].inter, x, 3072); }
        self.layers[l].inter.fwd(x)
    }
    fn g_out(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        #[cfg(feature = "npu")]
        if let Some((_, nl)) = &self.npu { return apply_tiled_mm2(&nl[l].out, x); }
        self.layers[l].out.fwd(x)
    }

    fn block(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        let t = x.nrows();
        let scale = (HD as f32).powf(-0.5);
        let ly = &self.layers[l];
        let (q, k, v) = self.g_qkv(x, l);
        let mut ctx = Array2::<f32>::zeros((t, D));
        for hd in 0..NH {
            let c0 = hd * HD;
            let qh = q.slice(s![.., c0..c0 + HD]);
            let kh = k.slice(s![.., c0..c0 + HD]);
            let vh = v.slice(s![.., c0..c0 + HD]);
            let mut sc = qh.dot(&kh.t()); // [T,T] bidirectional (no mask)
            sc *= scale;
            for i in 0..t {
                let mx = sc.row(i).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..t { let e = (sc[[i, j]] - mx).exp(); sc[[i, j]] = e; sum += e; }
                for j in 0..t { sc[[i, j]] /= sum; }
            }
            ctx.slice_mut(s![.., c0..c0 + HD]).assign(&sc.dot(&vh));
        }
        let attn = self.g_attn_out(&ctx, l);
        let h = layer_norm(&(x + &attn), ly.attn_ln_w.as_slice().unwrap(), ly.attn_ln_b.as_slice().unwrap(), EPS); // post-norm
        let inter = gelu(&self.g_inter(&h, l));
        let out = self.g_out(&inter, l);
        layer_norm(&(&h + &out), ly.out_ln_w.as_slice().unwrap(), ly.out_ln_b.as_slice().unwrap(), EPS) // post-norm
    }

    /// CLS-pooled, L2-normalized embedding for token ids.
    fn embed(&self, ids: &[i64]) -> Array1<f32> {
        let t = ids.len();
        let mut x = Array2::<f32>::zeros((t, D));
        for (i, &id) in ids.iter().enumerate() {
            let row = &self.word.row(id as usize) + &self.pos.row(i) + &self.tt.row(0);
            x.row_mut(i).assign(&row);
        }
        let mut x = layer_norm(&x, self.emb_ln_w.as_slice().unwrap(), self.emb_ln_b.as_slice().unwrap(), EPS);
        for l in 0..NL { x = self.block(&x, l); }
        let cls = x.row(0).to_owned();
        let norm = cls.dot(&cls).sqrt();
        cls / norm
    }
}

fn main() {
    let root = "artifacts/bge-base";
    let bge = Bge::load(root);
    let ids: Array1<i64> = read_npy(format!("{root}/ref_ids0.npy")).expect("ref_ids0.npy");
    let golden: Array1<f32> = read_npy(format!("{root}/ref_emb0.npy")).expect("ref_emb0.npy");
    let ids: Vec<i64> = ids.to_vec();

    // warm + time the real-sentence embed
    let _ = bge.embed(&ids);
    let t0 = std::time::Instant::now();
    let emb = bge.embed(&ids);
    let dt = t0.elapsed().as_secs_f64() * 1e3;

    let cos = emb.dot(&golden); // both L2-normalized
    let max_abs = emb.iter().zip(golden.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("[bge] {} tokens, embed in {:.1} ms", ids.len(), dt);
    println!("[bge] cosine_sim vs HF golden = {cos:.6}  max_abs_diff = {max_abs:.5}");
    // Embedding bar: cos > 0.99 (retrieval is robust to small bf16 perturbations). Host-f32 = exactly 1.0;
    // NPU GEMM (bf16) ~0.9989 = a benign wobble (no decode compounding, unlike A1's MHA).
    let ok = cos > 0.99;
    println!("[bge] RESULT: {}", if ok { "PASS (cos > 0.99 — embedding bar)" } else { "FAIL" });

    // Optional M-sweep: BGE_M="64,128,256,512" times the encoder at synthetic seq lengths (repeat ids)
    // to show where the NPU (occupancy-bound) overtakes host. No validation (synthetic input).
    if let Ok(ms) = std::env::var("BGE_M") {
        for m in ms.split(',').filter_map(|s| s.trim().parse::<usize>().ok()) {
            let synth: Vec<i64> = (0..m).map(|i| ids[i % ids.len()]).collect();
            let _ = bge.embed(&synth); // warm
            let t = std::time::Instant::now();
            let _ = bge.embed(&synth);
            println!("[bge] M={m:>4}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }
    std::process::exit(if ok { 0 } else { 2 });
}
