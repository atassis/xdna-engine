//! The resident frontier: runs the schedule over a luma plane. CPU backend = ground truth (pure Rust
//! conv2d). NPU backend (Task 4b) = host im2col -> NativeKernel whole-array GEMM. One interface, two
//! backends, so the whole engine is testable with no device.
use crate::schedule::{Op, Schedule};
use crate::{Plane, SrError};

/// A feature map [C, H, W], row-major per channel.
pub(crate) struct Feat {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

/// Baked conv weights [Cout, Cin, k, k] + bias [Cout], loaded from the arena.
pub(crate) struct ConvW {
    pub cout: usize,
    pub cin: usize,
    pub k: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct Frontier {
    use_npu: bool,
    weights: Vec<ConvW>,
    #[allow(dead_code)]
    npu: Option<npu_backend::NpuGemm>,
}

impl Frontier {
    pub fn build(sched: &Schedule, use_npu: bool) -> Result<Frontier, SrError> {
        let weights = load_conv_weights(sched)?;
        let npu = if use_npu {
            Some(npu_backend::NpuGemm::build(&weights)?)
        } else {
            None
        };
        Ok(Frontier {
            use_npu,
            weights,
            npu,
        })
    }

    pub fn run(&mut self, sched: &Schedule, y: &Plane) -> Result<Plane, SrError> {
        let mut feat = Feat {
            c: 1,
            h: y.h,
            w: y.w,
            data: y.data.clone(),
        };
        let mut ci = 0usize;
        for op in &sched.ops {
            match op {
                Op::Conv2d { k, pad, relu, .. } => {
                    let cw = &self.weights[ci];
                    ci += 1;
                    feat = if self.use_npu {
                        self.npu
                            .as_mut()
                            .unwrap()
                            .conv(ci - 1, &feat, cw, *k, *pad, *relu)?
                    } else {
                        cpu_conv(&feat, cw, *k, *pad, *relu)
                    };
                }
                Op::PixelShuffle { r } => {
                    feat = pixel_shuffle(&feat, *r);
                }
            }
        }
        if feat.c != 1 {
            return Err(SrError::Frame(format!(
                "final feat has {} channels, want 1",
                feat.c
            )));
        }
        Ok(Plane {
            w: feat.w,
            h: feat.h,
            data: feat.data,
        })
    }
}

/// Pure-Rust conv2d, same-pad, direct form (ground truth). [Cin,H,W] -> [Cout,H,W].
fn cpu_conv(x: &Feat, cw: &ConvW, k: usize, pad: usize, relu: bool) -> Feat {
    let (cin, h, w) = (x.c, x.h, x.w);
    let cout = cw.cout;
    assert_eq!(cin, cw.cin, "conv Cin mismatch");
    assert_eq!(k, cw.k, "conv k mismatch");
    let at = |c: usize, y: isize, xx: isize| -> f32 {
        if y < 0 || xx < 0 || y as usize >= h || xx as usize >= w {
            0.0
        } else {
            x.data[c * h * w + y as usize * w + xx as usize]
        }
    };
    let mut out = vec![0f32; cout * h * w];
    for oc in 0..cout {
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = cw.b[oc];
                for ic in 0..cin {
                    for ky in 0..k {
                        for kx in 0..k {
                            let wv = cw.w[((oc * cin + ic) * k + ky) * k + kx];
                            acc += wv
                                * at(
                                    ic,
                                    oy as isize + ky as isize - pad as isize,
                                    ox as isize + kx as isize - pad as isize,
                                );
                        }
                    }
                }
                out[oc * h * w + oy * w + ox] = if relu { acc.max(0.0) } else { acc };
            }
        }
    }
    Feat {
        c: cout,
        h,
        w,
        data: out,
    }
}

/// Depth-to-space by r, CRD order (matches numpy reshape(C,r,r,H,W).transpose(0,3,1,4,2)).
fn pixel_shuffle(x: &Feat, r: usize) -> Feat {
    let c = x.c / (r * r);
    let (h, w) = (x.h, x.w);
    let (oh, ow) = (h * r, w * r);
    let mut out = vec![0f32; c * oh * ow];
    for oc in 0..c {
        for ry in 0..r {
            for rx in 0..r {
                let ic = oc * r * r + ry * r + rx;
                for yy in 0..h {
                    for xx in 0..w {
                        out[oc * oh * ow + (yy * r + ry) * ow + (xx * r + rx)] =
                            x.data[ic * h * w + yy * w + xx];
                    }
                }
            }
        }
    }
    Feat {
        c,
        h: oh,
        w: ow,
        data: out,
    }
}

/// Load the baked conv weights from the arena the converter produced.
fn load_conv_weights(sched: &Schedule) -> Result<Vec<ConvW>, SrError> {
    let loaded = npu_weights::arena::load(std::path::Path::new(&sched.arena), &sched.name)
        .map_err(|e| SrError::Load(format!("arena {}: {e}", sched.arena)))?;
    let mut out = Vec::new();
    for op in &sched.ops {
        if let Op::Conv2d {
            weights, cin, cout, ..
        } = op
        {
            let (_ws, w) = loaded
                .tensor_f32(&format!("{weights}_w"))
                .map_err(|e| SrError::Load(format!("{weights}_w: {e}")))?;
            let (_bs, b) = loaded
                .tensor_f32(&format!("{weights}_b"))
                .map_err(|e| SrError::Load(format!("{weights}_b: {e}")))?;
            let k = op_k(op);
            out.push(ConvW {
                cout: *cout,
                cin: *cin,
                k,
                w,
                b,
            });
        }
    }
    Ok(out)
}

fn op_k(op: &Op) -> usize {
    match op {
        Op::Conv2d { k, .. } => *k,
        _ => 0,
    }
}

// --- NPU backend: host im2col -> NativeKernel whole-array GEMM (the M1 host-im2col path in Rust) ---
//
// One 576x256 whole-array GEMM kernel serves ALL four ESPCN convs: NativeKernel zero-pads each conv's
// real (Kf=Cin*k*k <= 576, Cout <= 256) up to the kernel dims, so we build ONE xclbin, not four. Per
// conv: host im2col -> C[M,Cout] = A[M,Kf] @ B[Kf,Cout] (+bias, +relu). M (=H*W) is tiled to PAD_M=512.
//
// Requires the whole_array xclbin + insts at:
//   mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/
//     final_512x576x256_32x32x32_8c.xclbin  +  insts_512x576x256_32x32x32_8c.txt
// built via `make M=512 K=576 N=256 n_aie_cols=8` under the fork toolchain (insts .bin copied to .txt).
mod npu_backend {
    use super::{ConvW, Feat};
    use crate::SrError;
    use ndarray::Array2;
    use npu_engine::esm::native::{NativeKernel, NativeWeight, PAD_M};
    use npu_xrt::Device;
    use std::path::Path;
    use std::rc::Rc;

    const KERNEL_K: usize = 576; // max Cin*k*k across ESPCN convs (conv2/conv3 = 64*9)
    const KERNEL_N: usize = 256; // whole-array tiling needs N % (32*8) == 0; Cout <= 64 pads up
    const TILE: &str = "32x32x32";
    const WA: &str =
        "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build";

    pub struct NpuGemm {
        kernel: Rc<NativeKernel>,
        weights: Vec<NativeWeight>, // one per conv2d op, in schedule order
    }

    impl NpuGemm {
        pub fn build(convs: &[ConvW]) -> Result<NpuGemm, SrError> {
            if !crate::npu_available() {
                return Err(SrError::NotAvailable);
            }
            let dev = Rc::new(Device::open(0).map_err(|e| SrError::Device(format!("open: {e}")))?);
            let kernel = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                NativeKernel::load(&dev, Path::new(WA), KERNEL_K, KERNEL_N, TILE)
            }))
            .map_err(|_| {
                SrError::Device(format!(
                    "load whole_array xclbin from {WA} (need final_{PAD_M}x{KERNEL_K}x{KERNEL_N}_{TILE}_8c.xclbin + insts .txt)"
                ))
            })?;
            let mut weights = Vec::new();
            for cw in convs {
                // B = weight [Cout,Cin,k,k] -> [Cout, Kf] -> transpose -> [Kf, Cout].
                let kf = cw.cin * cw.k * cw.k;
                let mut b = Array2::<f32>::zeros((kf, cw.cout));
                for oc in 0..cw.cout {
                    for j in 0..kf {
                        b[[j, oc]] = cw.w[oc * kf + j];
                    }
                }
                weights.push(kernel.weight(&b));
            }
            Ok(NpuGemm { kernel, weights })
        }

        pub fn conv(
            &mut self,
            idx: usize,
            x: &Feat,
            cw: &ConvW,
            k: usize,
            pad: usize,
            relu: bool,
        ) -> Result<Feat, SrError> {
            let (cin, h, w) = (x.c, x.h, x.w);
            let kf = cin * k * k;
            let m = h * w;
            // Host im2col: A[M, Kf], column order ic*(k*k) + ky*k + kx (matches the weight reshape).
            let at = |c: usize, yy: isize, xx: isize| -> f32 {
                if yy < 0 || xx < 0 || yy as usize >= h || xx as usize >= w {
                    0.0
                } else {
                    x.data[c * h * w + yy as usize * w + xx as usize]
                }
            };
            let weight = &self.weights[idx];
            let mut out = vec![0f32; cw.cout * h * w];
            // Tile M into chunks of PAD_M so H*W frames larger than 512 px work.
            let mut p0 = 0usize;
            while p0 < m {
                let rows = (m - p0).min(PAD_M);
                let mut a = Array2::<f32>::zeros((rows, kf));
                for r in 0..rows {
                    let p = p0 + r;
                    let (oy, ox) = (p / w, p % w);
                    for ic in 0..cin {
                        for ky in 0..k {
                            for kx in 0..k {
                                a[[r, ic * (k * k) + ky * k + kx]] = at(
                                    ic,
                                    oy as isize + ky as isize - pad as isize,
                                    ox as isize + kx as isize - pad as isize,
                                );
                            }
                        }
                    }
                }
                let c = self.kernel.matmul(weight, &a, cw.cout, Some(&cw.b)); // [rows, Cout]
                for r in 0..rows {
                    let p = p0 + r;
                    for oc in 0..cw.cout {
                        let v = c[[r, oc]];
                        out[oc * h * w + p] = if relu { v.max(0.0) } else { v };
                    }
                }
                p0 += rows;
            }
            Ok(Feat {
                c: cw.cout,
                h,
                w,
                data: out,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_shuffle_crd_order() {
        // 9 channels, 1x1 spatial, r=3 -> 1 channel 3x3; CRD maps channel c -> (c/3, c%3).
        let x = Feat {
            c: 9,
            h: 1,
            w: 1,
            data: (0..9).map(|i| i as f32).collect(),
        };
        let y = pixel_shuffle(&x, 3);
        assert_eq!((y.c, y.h, y.w), (1, 3, 3));
        // out[(ry, rx)] = channel ry*3 + rx
        for ry in 0..3 {
            for rx in 0..3 {
                assert_eq!(y.data[ry * 3 + rx], (ry * 3 + rx) as f32);
            }
        }
    }
}
