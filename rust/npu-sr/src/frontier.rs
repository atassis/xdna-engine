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

// --- NPU backend: real implementation lands in Task 4b (device-gated) ---
mod npu_backend {
    use super::{ConvW, Feat};
    use crate::SrError;

    pub struct NpuGemm;
    impl NpuGemm {
        pub fn build(_w: &[ConvW]) -> Result<NpuGemm, SrError> {
            Err(SrError::Device(
                "npu frontier not built yet (Task 4b)".into(),
            ))
        }
        pub fn conv(
            &mut self,
            _idx: usize,
            _x: &Feat,
            _cw: &ConvW,
            _k: usize,
            _pad: usize,
            _relu: bool,
        ) -> Result<Feat, SrError> {
            Err(SrError::Device("npu conv not built yet (Task 4b)".into()))
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
