//! Host-fp32 ResNet-18 forward via im2col2d + ndarray GEMM, gated vs an ORT ResNet-18 oracle.
//! Proves the conv->im2col->GEMM LOWERING is correct end-to-end before any NPU kernel exists.
//! Driven by artifacts/resnet18/manifest.json (a flat op list from scripts/export_resnet.py).
//! Run from repo root: rust/target/release/verify_resnet
//!   (needs artifacts/resnet18/ + libonnxruntime on LD_LIBRARY_PATH)
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ndarray::prelude::*;
use ndarray_npy::read_npy;
use npu_asr::conv_npu::ConvNpu;
use npu_asr_host::im2col2d;
use npu_onnx::{Env, Session, Tensor};
use npu_xrt::Device;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
struct Op {
    op: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    kh: usize,
    #[serde(default)]
    kw: usize,
    #[serde(default)]
    stride: usize,
    #[serde(default)]
    pad: usize,
    #[serde(default)]
    relu: bool,
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let mut md = 0f32;
    let mut mr = 0f32;
    for (x, y) in a.iter().zip(b) {
        md = md.max((x - y).abs());
        mr = mr.max(y.abs());
    }
    md / (mr + 1e-9)
}

/// one conv: x[Cin,H,W] -> y[Cout,Hout,Wout] via im2col2d + (cols @ wˆT) + bias.
fn conv(x: &Array3<f32>, w: &Array4<f32>, b: &Array1<f32>, op: &Op) -> Array3<f32> {
    let (cin, h, wd) = x.dim();
    let cout = w.dim().0;
    let out_h = (h + 2 * op.pad - op.kh) / op.stride + 1;
    let out_w = (wd + 2 * op.pad - op.kw) / op.stride + 1;
    let cols = im2col2d(x, op.kh, op.kw, op.stride, op.stride, op.pad, op.pad); // [P, cin*kh*kw]
    let k = cin * op.kh * op.kw;
    let w2 = w.to_shape((cout, k)).unwrap().to_owned(); // [cout, K]
    let prod = cols.dot(&w2.t()); // [P, cout]
    let mut y = Array3::<f32>::zeros((cout, out_h, out_w));
    for p in 0..out_h * out_w {
        let (oh, ow) = (p / out_w, p % out_w);
        for co in 0..cout {
            y[[co, oh, ow]] = prod[[p, co]] + b[co];
        }
    }
    y
}

fn maxpool3x3s2(x: &Array3<f32>) -> Array3<f32> {
    let (c, h, w) = x.dim();
    let (oh, ow) = ((h + 2 - 3) / 2 + 1, (w + 2 - 3) / 2 + 1); // 3x3 stride 2 pad 1
    let mut y = Array3::<f32>::from_elem((c, oh, ow), f32::NEG_INFINITY);
    for ch in 0..c {
        for i in 0..oh {
            for j in 0..ow {
                for di in 0..3 {
                    for dj in 0..3 {
                        let (ii, jj) = (i * 2 + di, j * 2 + dj);
                        if ii >= 1 && jj >= 1 && ii < h + 1 && jj < w + 1 {
                            y[[ch, i, j]] = y[[ch, i, j]].max(x[[ch, ii - 1, jj - 1]]);
                        }
                    }
                }
            }
        }
    }
    y
}

fn relu3(x: &mut Array3<f32>) {
    x.mapv_inplace(|v| v.max(0.0));
}

fn load_wb(dir: &Path, name: &str) -> (Array4<f32>, Array1<f32>) {
    let w: Array4<f32> = read_npy(dir.join(format!("{name}_w.npy"))).expect("weight npy");
    let b: Array1<f32> = read_npy(dir.join(format!("{name}_b.npy"))).expect("bias npy");
    (w, b)
}

fn main() {
    let dir = Path::new("artifacts/resnet18");
    let ops: Vec<Op> =
        serde_json::from_reader(std::fs::File::open(dir.join("manifest.json")).expect("manifest"))
            .expect("parse manifest");
    let input: Array3<f32> = read_npy(dir.join("input.npy")).expect("input npy");

    let npu_mode = std::env::args().any(|a| a == "--npu");
    let convnpu = if npu_mode {
        let root = std::env::var("PATCH_XCLBIN_ROOT").unwrap_or_else(|_| ".".into());
        let wa = PathBuf::from(root)
            .join("mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build");
        Some(ConvNpu::new(Rc::new(Device::open(0).expect("open NPU (stop voxd first)")), wa))
    } else {
        None
    };

    let mut x = input.clone();
    let mut skip: Option<Array3<f32>> = None;
    let mut logits: Option<Array1<f32>> = None;

    for op in &ops {
        match op.op.as_str() {
            "conv" => {
                let (w, b) = load_wb(dir, &op.name);
                let mut y = if let Some(cn) = &convnpu {
                    let yn = cn.conv(&x, &w, &b, op.kh, op.kw, op.stride, op.pad);
                    let yh = conv(&x, &w, &b, op);
                    let r = rel(yn.as_slice().unwrap(), yh.as_slice().unwrap());
                    println!("  [npu] {:<8} rel vs host = {:.3e}", op.name, r);
                    yn
                } else {
                    conv(&x, &w, &b, op)
                };
                if op.relu {
                    relu3(&mut y);
                }
                x = y;
            }
            "maxpool" => x = maxpool3x3s2(&x),
            "block_start" => skip = Some(x.clone()),
            "downsample" => {
                let (w, b) = load_wb(dir, &op.name);
                let s = skip.as_ref().expect("downsample without block_start");
                skip = Some(if let Some(cn) = &convnpu {
                    cn.conv(s, &w, &b, op.kh, op.kw, op.stride, op.pad)
                } else {
                    conv(s, &w, &b, op)
                }); // project the saved skip (1x1 stride2)
            }
            "residual_relu" => {
                let s = skip.take().expect("residual without skip");
                x = &x + &s;
                relu3(&mut x);
            }
            "globalavgpool" => {
                let (c, h, wd) = x.dim();
                let mut pooled = Array1::<f32>::zeros(c);
                for ch in 0..c {
                    let mut acc = 0f32;
                    for i in 0..h {
                        for j in 0..wd {
                            acc += x[[ch, i, j]];
                        }
                    }
                    pooled[ch] = acc / (h * wd) as f32;
                }
                // stash pooled as a 1x1 feature map for the fc step
                x = pooled.into_shape_with_order((c, 1, 1)).unwrap();
            }
            "fc" => {
                let w: Array2<f32> = read_npy(dir.join("fc_w.npy")).expect("fc_w"); // [1000,512]
                let b: Array1<f32> = read_npy(dir.join("fc_b.npy")).expect("fc_b");
                let c = x.dim().0;
                let pooled = x.to_shape(c).unwrap().to_owned(); // [512]
                logits = Some(w.dot(&pooled) + &b); // [1000]
            }
            other => panic!("unknown op {other}"),
        }
    }

    let ours = logits.expect("no fc op produced logits");

    // oracle
    let env = Env::new().expect("onnx env");
    let sess = Session::load(&env, dir.join("resnet18.onnx").to_str().unwrap()).expect("onnx");
    let xs = input.as_standard_layout();
    let isl = xs.as_slice().unwrap();
    let out = sess
        .run(&[("input", Tensor::F32(isl, vec![1, 3, 224, 224]))], &["logits"])
        .expect("onnx run");

    let r = rel(ours.as_slice().unwrap(), out.f32(0));
    let (label, tol, pass) = if npu_mode {
        ("NPU bf16", 0.08f32, r <= 0.08)
    } else {
        ("host-fp32", 1e-3f32, r < 1e-3)
    };
    println!("ResNet-18 {label} logits rel vs ORT = {r:.3e}  (need <= {tol:.0e})");
    assert!(pass, "ResNet-18 gate FAILED: rel {r} > {tol}");
    println!("{} GATE PASS", if npu_mode { "NPU" } else { "LOWERING" });
}
