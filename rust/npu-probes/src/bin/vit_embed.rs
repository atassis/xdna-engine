//! google/vit-base-patch16-224 image classification on the engine (host f32) — a 4th model class (vision).
//!
//! ViT-base: same 768/12/12/3072 dims, PRE-norm, gelu, bidirectional. New vs bge: patch embedding
//! (16×16 patches → 768 via a flatten+matmul), a prepended CLS token, and a 1000-class head. Validates
//! logits (cosine + argmax) against the HF golden on a fixed random image (scripts/vit_reference.py).
//! Reuses the bge/whisper encoder shape → NPU GEMM routing would help (occupancy-bound), like bge.
//!
//!   cargo run -p npu-whisper --release --bin vit_embed

use ndarray::prelude::*;
use ndarray_npy::read_npy;
use npu_asr_host::{gelu, layer_norm};

const NL: usize = 12;
const NH: usize = 12;
const HD: usize = 64;
const D: usize = 768;
const NP: usize = 196; // 14×14 patches
const T: usize = 197; // + CLS
const EPS: f32 = 1e-12;

fn l2(p: String) -> Array2<f32> { read_npy(&p).unwrap_or_else(|e| panic!("load {p}: {e}")) }
fn l1(p: String) -> Array1<f32> { read_npy(&p).unwrap_or_else(|e| panic!("load {p}: {e}")) }

struct Lin { w: Array2<f32>, b: Array1<f32> }
impl Lin {
    fn load(d: &str, n: &str) -> Self { Lin { w: l2(format!("{d}/{n}.weight.npy")), b: l1(format!("{d}/{n}.bias.npy")) } }
    fn fwd(&self, x: &Array2<f32>) -> Array2<f32> { let mut y = x.dot(&self.w); y += &self.b.view().insert_axis(Axis(0)); y }
}

struct Layer {
    ln_before_w: Array1<f32>, ln_before_b: Array1<f32>,
    q: Lin, k: Lin, v: Lin, attn_out: Lin,
    ln_after_w: Array1<f32>, ln_after_b: Array1<f32>,
    inter: Lin, out: Lin,
}

struct Vit {
    patch: Lin, cls: Array1<f32>, pos: Array2<f32>,
    lnf_w: Array1<f32>, lnf_b: Array1<f32>,
    classifier: Lin,
    layers: Vec<Layer>,
}

impl Vit {
    fn load(root: &str) -> Self {
        let layers = (0..NL).map(|l| {
            let d = format!("{root}/L{l}");
            Layer {
                ln_before_w: l1(format!("{d}/ln_before.weight.npy")), ln_before_b: l1(format!("{d}/ln_before.bias.npy")),
                q: Lin::load(&d, "q"), k: Lin::load(&d, "k"), v: Lin::load(&d, "v"), attn_out: Lin::load(&d, "attn_out"),
                ln_after_w: l1(format!("{d}/ln_after.weight.npy")), ln_after_b: l1(format!("{d}/ln_after.bias.npy")),
                inter: Lin::load(&d, "inter"), out: Lin::load(&d, "out"),
            }
        }).collect();
        Vit {
            patch: Lin::load(root, "patch_proj"), cls: l1(format!("{root}/cls_token.npy")), pos: l2(format!("{root}/pos_emb.npy")),
            lnf_w: l1(format!("{root}/ln_final.weight.npy")), lnf_b: l1(format!("{root}/ln_final.bias.npy")),
            classifier: Lin::load(root, "classifier"), layers,
        }
    }

    fn block(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        let ly = &self.layers[l];
        let scale = (HD as f32).powf(-0.5);
        // pre-norm attention
        let h = layer_norm(x, ly.ln_before_w.as_slice().unwrap(), ly.ln_before_b.as_slice().unwrap(), EPS);
        let q = ly.q.fwd(&h); let k = ly.k.fwd(&h); let v = ly.v.fwd(&h);
        let mut ctx = Array2::<f32>::zeros((T, D));
        for hd in 0..NH {
            let c0 = hd * HD;
            let (qh, kh, vh) = (q.slice(s![.., c0..c0 + HD]), k.slice(s![.., c0..c0 + HD]), v.slice(s![.., c0..c0 + HD]));
            let mut sc = qh.dot(&kh.t()); sc *= scale; // [T,T] bidirectional
            for i in 0..T {
                let mx = sc.row(i).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..T { let e = (sc[[i, j]] - mx).exp(); sc[[i, j]] = e; sum += e; }
                for j in 0..T { sc[[i, j]] /= sum; }
            }
            ctx.slice_mut(s![.., c0..c0 + HD]).assign(&sc.dot(&vh));
        }
        let x = x + &ly.attn_out.fwd(&ctx);
        // pre-norm MLP
        let h = layer_norm(&x, ly.ln_after_w.as_slice().unwrap(), ly.ln_after_b.as_slice().unwrap(), EPS);
        let mlp = ly.out.fwd(&gelu(&ly.inter.fwd(&h)));
        x + &mlp
    }

    /// image [3,224,224] -> logits [1000].
    fn classify(&self, img: &Array3<f32>) -> Array1<f32> {
        // patch embed: 196 patches of [3,16,16] flattened (c,h,w) -> [196,768] @ patch[768,768]
        let mut patches = Array2::<f32>::zeros((NP, D));
        for p in 0..NP {
            let (pi, pj) = (p / 14, p % 14);
            for c in 0..3 {
                for hh in 0..16 {
                    for ww in 0..16 {
                        patches[[p, c * 256 + hh * 16 + ww]] = img[[c, pi * 16 + hh, pj * 16 + ww]];
                    }
                }
            }
        }
        let pe = self.patch.fwd(&patches); // [196,768]
        // prepend CLS, add position embeddings
        let mut x = Array2::<f32>::zeros((T, D));
        x.row_mut(0).assign(&self.cls);
        x.slice_mut(s![1.., ..]).assign(&pe);
        x += &self.pos;
        for l in 0..NL { x = self.block(&x, l); }
        let x = layer_norm(&x, self.lnf_w.as_slice().unwrap(), self.lnf_b.as_slice().unwrap(), EPS);
        let cls = x.row(0).to_owned().insert_axis(Axis(0)); // [1,768]
        self.classifier.fwd(&cls).row(0).to_owned() // [1000]
    }
}

fn main() {
    let root = "artifacts/vit-base";
    let vit = Vit::load(root);
    let img: Array3<f32> = read_npy(format!("{root}/ref_img.npy")).expect("ref_img.npy");
    let golden: Array1<f32> = read_npy(format!("{root}/ref_logits.npy")).expect("ref_logits.npy");

    let _ = vit.classify(&img);
    let t0 = std::time::Instant::now();
    let logits = vit.classify(&img);
    let dt = t0.elapsed().as_secs_f64() * 1e3;

    let dot = logits.dot(&golden);
    let cos = dot / (logits.dot(&logits).sqrt() * golden.dot(&golden).sqrt());
    let am = |a: &Array1<f32>| a.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
    let (mine, theirs) = (am(&logits), am(&golden));
    println!("[vit] classify in {dt:.1} ms  cos={cos:.6}  argmax mine={mine} hf={theirs}");
    let ok = cos > 0.9999 && mine == theirs;
    println!("[vit] RESULT: {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 2 });
}
