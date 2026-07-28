//! opt-125m greedy decode on the engine (host f32 first; the LLM-decode generalization beachhead).
//!
//! opt-125m is dimension-identical to whisper-small (768/12/12/3072/64), decoder-only, relu FFN,
//! learned positions (offset +2), pre-norm. Weights converted by scripts/convert_opt125m.py to
//! artifacts/opt-125m/ ([K,N] linears). This bin loads them and greedy-decodes, validating
//! token-for-token against the HF golden (scripts/opt125m_reference.py). NPU GEMV routing is a
//! follow-on (reuse the K=768 ctx_decode primitives).
//!
//!   cargo run -p npu-whisper --release --bin opt125m_decode

use std::path::Path;

use ndarray::prelude::*;
use ndarray_npy::read_npy;
use npu_asr_host::layer_norm;

const NL: usize = 12;
const NH: usize = 12;
const HD: usize = 64;
const D: usize = 768;
const EPS: f32 = 1e-5;

fn l2(path: String) -> Array2<f32> {
    read_npy(&path).unwrap_or_else(|e| panic!("load {path}: {e}"))
}
fn l1(path: String) -> Array1<f32> {
    read_npy(&path).unwrap_or_else(|e| panic!("load {path}: {e}"))
}

struct Lin {
    w: Array2<f32>, // [K, N]
    b: Array1<f32>,
    // weight-only int8 (OPT_INT8=1): per-column symmetric quant. w_i8[K*N] row-major + scale[N].
    // Tests the run-small-llms int8 lever for the bandwidth-bound LLM decode (weights = the LPDDR sink).
    i8: Option<(Vec<i8>, Vec<f32>)>,
    // weight-only int4 (OPT_INT4=1): per-column symmetric, 2 nibbles/byte (half the int8 bytes -> ~4x vs f32
    // if bandwidth-bound). packed[K*N/2] + scale[N]. N is even (768/3072).
    i4: Option<(Vec<u8>, Vec<f32>)>,
}
impl Lin {
    fn load(dir: &str, name: &str) -> Self {
        let w = l2(format!("{dir}/{name}.weight.npy"));
        let b = l1(format!("{dir}/{name}.bias.npy"));
        let (k, n) = w.dim();
        let colscale = |div: f32| -> Vec<f32> {
            (0..n).map(|c| {
                let amax = (0..k).map(|r| w[[r, c]].abs()).fold(0.0f32, f32::max);
                if amax > 0.0 { amax / div } else { 1.0 }
            }).collect()
        };
        let i8 = if std::env::var("OPT_INT8").is_ok() {
            let scale = colscale(127.0);
            let mut wi8 = vec![0i8; k * n];
            for r in 0..k {
                for c in 0..n {
                    wi8[r * n + c] = (w[[r, c]] / scale[c]).round().clamp(-127.0, 127.0) as i8;
                }
            }
            Some((wi8, scale))
        } else {
            None
        };
        let i4 = if std::env::var("OPT_INT4").is_ok() {
            let scale = colscale(7.0); // int4 symmetric range [-7,7]
            let mut packed = vec![0u8; k * n / 2];
            for r in 0..k {
                for c2 in 0..n / 2 {
                    let q = |c: usize| -> u8 {
                        ((w[[r, c]] / scale[c]).round().clamp(-7.0, 7.0) as i32 & 0x0F) as u8
                    };
                    let lo = q(2 * c2);
                    let hi = q(2 * c2 + 1);
                    packed[r * (n / 2) + c2] = lo | (hi << 4);
                }
            }
            Some((packed, scale))
        } else {
            None
        };
        Lin { w, b, i8, i4 }
    }
    fn fwd(&self, x: &Array2<f32>) -> Array2<f32> {
        if let Some((pk, scale)) = &self.i4 {
            let (m, k) = x.dim();
            let n = self.b.len();
            let nh = n / 2;
            let sext = |v: u8| -> i32 { ((v as i32) ^ 0x8) - 0x8 }; // 4-bit sign-extend
            let mut y = Array2::<f32>::zeros((m, n));
            for mi in 0..m {
                let yr = y.row_mut(mi).into_slice().unwrap();
                for ki in 0..k {
                    let a = x[[mi, ki]];
                    let row = &pk[ki * nh..(ki + 1) * nh];
                    for nj in 0..nh {
                        let byte = row[nj];
                        yr[2 * nj] += a * sext(byte & 0x0F) as f32;
                        yr[2 * nj + 1] += a * sext(byte >> 4) as f32;
                    }
                }
                for ni in 0..n {
                    yr[ni] = yr[ni] * scale[ni] + self.b[ni];
                }
            }
            return y;
        }
        if let Some((wi8, scale)) = &self.i8 {
            let (m, k) = x.dim();
            let n = self.b.len();
            let mut y = Array2::<f32>::zeros((m, n));
            for mi in 0..m {
                let yr = y.row_mut(mi);
                let yr = yr.into_slice().unwrap();
                for ki in 0..k {
                    let a = x[[mi, ki]];
                    let wrow = &wi8[ki * n..(ki + 1) * n];
                    for ni in 0..n {
                        yr[ni] += a * wrow[ni] as f32;
                    }
                }
                for ni in 0..n {
                    yr[ni] = yr[ni] * scale[ni] + self.b[ni];
                }
            }
            return y;
        }
        let mut y = x.dot(&self.w);
        y += &self.b.view().insert_axis(Axis(0));
        y
    }
}

struct Layer {
    ln_self_w: Array1<f32>,
    ln_self_b: Array1<f32>,
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    ln_ffn_w: Array1<f32>,
    ln_ffn_b: Array1<f32>,
    fc1: Lin,
    fc2: Lin,
}

struct Opt {
    emb_tok: Array2<f32>,
    emb_pos: Array2<f32>,
    lm: Array2<f32>, // [D, vocab]
    lnf_w: Array1<f32>,
    lnf_b: Array1<f32>,
    layers: Vec<Layer>,
}

impl Opt {
    fn load(root: &str) -> Self {
        let layers = (0..NL)
            .map(|l| {
                let d = format!("{root}/L{l}");
                Layer {
                    ln_self_w: l1(format!("{d}/ln_self.weight.npy")),
                    ln_self_b: l1(format!("{d}/ln_self.bias.npy")),
                    q: Lin::load(&d, "q"),
                    k: Lin::load(&d, "k"),
                    v: Lin::load(&d, "v"),
                    out: Lin::load(&d, "out"),
                    ln_ffn_w: l1(format!("{d}/ln_ffn.weight.npy")),
                    ln_ffn_b: l1(format!("{d}/ln_ffn.bias.npy")),
                    fc1: Lin::load(&d, "fc1"),
                    fc2: Lin::load(&d, "fc2"),
                }
            })
            .collect();
        Opt {
            emb_tok: l2(format!("{root}/embed_tokens.npy")),
            emb_pos: l2(format!("{root}/embed_positions.npy")),
            lm: l2(format!("{root}/lm_head.weight.npy")),
            lnf_w: l1(format!("{root}/ln_final.weight.npy")),
            lnf_b: l1(format!("{root}/ln_final.bias.npy")),
            layers,
        }
    }

    fn block(&self, x: &Array2<f32>, l: usize) -> Array2<f32> {
        let t = x.nrows();
        let scale = (HD as f32).powf(-0.5);
        let ly = &self.layers[l];
        // --- self-attention (pre-norm) ---
        let h = layer_norm(x, ly.ln_self_w.as_slice().unwrap(), ly.ln_self_b.as_slice().unwrap(), EPS);
        let q = ly.q.fwd(&h) * scale;
        let k = ly.k.fwd(&h);
        let v = ly.v.fwd(&h);
        let mut ctx = Array2::<f32>::zeros((t, D));
        for hd in 0..NH {
            let c0 = hd * HD;
            let qh = q.slice(s![.., c0..c0 + HD]); // [T,HD]
            let kh = k.slice(s![.., c0..c0 + HD]);
            let vh = v.slice(s![.., c0..c0 + HD]);
            let mut sc = qh.dot(&kh.t()); // [T,T]
            // causal mask + softmax per row
            for i in 0..t {
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    if sc[[i, j]] > mx {
                        mx = sc[[i, j]];
                    }
                }
                let mut sum = 0.0f32;
                for j in 0..t {
                    if j <= i {
                        let e = (sc[[i, j]] - mx).exp();
                        sc[[i, j]] = e;
                        sum += e;
                    } else {
                        sc[[i, j]] = 0.0;
                    }
                }
                for j in 0..=i {
                    sc[[i, j]] /= sum;
                }
            }
            let oh = sc.dot(&vh); // [T,HD]
            ctx.slice_mut(s![.., c0..c0 + HD]).assign(&oh);
        }
        let attn = ly.out.fwd(&ctx);
        let x = x + &attn;
        // --- FFN (pre-norm, relu) ---
        let h = layer_norm(&x, ly.ln_ffn_w.as_slice().unwrap(), ly.ln_ffn_b.as_slice().unwrap(), EPS);
        let mut f = ly.fc1.fwd(&h);
        f.mapv_inplace(|v| v.max(0.0)); // relu
        let f = ly.fc2.fwd(&f);
        x + &f
    }

    /// Full-sequence forward; returns logits for the LAST position [vocab].
    fn forward_last(&self, ids: &[i64]) -> Array1<f32> {
        let t = ids.len();
        let mut x = Array2::<f32>::zeros((t, D));
        for (i, &id) in ids.iter().enumerate() {
            let row = &self.emb_tok.row(id as usize) + &self.emb_pos.row(i + 2);
            x.row_mut(i).assign(&row);
        }
        for l in 0..NL {
            x = self.block(&x, l);
        }
        let x = layer_norm(&x, self.lnf_w.as_slice().unwrap(), self.lnf_b.as_slice().unwrap(), EPS);
        let last = x.row(t - 1).to_owned(); // [D]
        last.dot(&self.lm) // [vocab]
    }

    /// One decode step with a KV cache: process a single token `x`[1,D], appending its k/v to the
    /// per-layer caches and attending the new query over the full cache. Returns logits [vocab].
    fn step(&self, id: i64, pos: usize, kc: &mut [Array2<f32>], vc: &mut [Array2<f32>]) -> Array1<f32> {
        let mut x = (&self.emb_tok.row(id as usize) + &self.emb_pos.row(pos + 2))
            .insert_axis(Axis(0)); // [1,D]
        let scale = (HD as f32).powf(-0.5);
        for l in 0..NL {
            let ly = &self.layers[l];
            let h = layer_norm(&x, ly.ln_self_w.as_slice().unwrap(), ly.ln_self_b.as_slice().unwrap(), EPS);
            let q = ly.q.fwd(&h) * scale; // [1,D]
            let knew = ly.k.fwd(&h);
            let vnew = ly.v.fwd(&h);
            // append to cache
            kc[l] = if kc[l].nrows() == 0 { knew.clone() } else {
                ndarray::concatenate![Axis(0), kc[l].view(), knew.view()]
            };
            vc[l] = if vc[l].nrows() == 0 { vnew.clone() } else {
                ndarray::concatenate![Axis(0), vc[l].view(), vnew.view()]
            };
            let tk = kc[l].nrows();
            let mut ctx = Array2::<f32>::zeros((1, D));
            for hd in 0..NH {
                let c0 = hd * HD;
                let qh = q.slice(s![.., c0..c0 + HD]); // [1,HD]
                let kh = kc[l].slice(s![.., c0..c0 + HD]); // [tk,HD]
                let vh = vc[l].slice(s![.., c0..c0 + HD]);
                let mut sc = qh.dot(&kh.t()); // [1,tk] (attend all cached — no mask, causal by construction)
                let mx = sc.row(0).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..tk { let e = (sc[[0, j]] - mx).exp(); sc[[0, j]] = e; sum += e; }
                for j in 0..tk { sc[[0, j]] /= sum; }
                ctx.slice_mut(s![.., c0..c0 + HD]).assign(&sc.dot(&vh));
            }
            let attn = ly.out.fwd(&ctx);
            let xr = &x + &attn;
            let h = layer_norm(&xr, ly.ln_ffn_w.as_slice().unwrap(), ly.ln_ffn_b.as_slice().unwrap(), EPS);
            let mut f = ly.fc1.fwd(&h);
            f.mapv_inplace(|v| v.max(0.0));
            let f = ly.fc2.fwd(&f);
            x = xr + &f;
        }
        let x = layer_norm(&x, self.lnf_w.as_slice().unwrap(), self.lnf_b.as_slice().unwrap(), EPS);
        x.row(0).dot(&self.lm)
    }

    /// KV-cached greedy generation: prefill `prompt`, then emit `n_new` tokens.
    fn generate(&self, prompt: &[i64], n_new: usize) -> Vec<i64> {
        let mut kc: Vec<Array2<f32>> = (0..NL).map(|_| Array2::zeros((0, D))).collect();
        let mut vc: Vec<Array2<f32>> = (0..NL).map(|_| Array2::zeros((0, D))).collect();
        let mut logits = Array1::zeros(0);
        for (pos, &id) in prompt.iter().enumerate() {
            logits = self.step(id, pos, &mut kc, &mut vc);
        }
        let argmax = |l: &Array1<f32>| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as i64;
        let mut gen = Vec::new();
        let mut next = argmax(&logits);
        let mut pos = prompt.len();
        loop {
            gen.push(next);
            if gen.len() == n_new { break; }
            logits = self.step(next, pos, &mut kc, &mut vc);
            pos += 1;
            next = argmax(&logits);
        }
        gen
    }
}

fn main() {
    let root = "artifacts/opt-125m";
    assert!(Path::new(root).exists(), "missing {root} (run scripts/convert_opt125m.py)");
    let opt = Opt::load(root);

    // golden (from scripts/opt125m_reference.py): "The capital of France is"
    let prompt_ids: Vec<i64> = vec![2, 133, 812, 9, 1470, 16];
    let golden: Vec<i64> = vec![
        5, 812, 9, 5, 1515, 3497, 4, 50118, 50118, 133, 812, 9, 1470, 16, 5, 812, 9, 5, 1515, 3497,
    ];
    let n = golden.len();
    let t0 = std::time::Instant::now();
    let gen = opt.generate(&prompt_ids, n); // KV-cached decode
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "[opt125m] host-f32 KV-cached decode: {n} tokens in {:.0} ms = {:.1} ms/tok ({:.1} tok/s)",
        dt * 1e3, dt * 1e3 / n as f64, n as f64 / dt
    );
    let ok = gen == golden;
    println!("[opt125m] generated: {gen:?}");
    println!("[opt125m] golden:    {golden:?}");
    println!("[opt125m] RESULT: {}", if ok { "PASS (matches HF golden)" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 2 });
}
