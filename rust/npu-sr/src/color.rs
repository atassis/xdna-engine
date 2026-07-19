//! Y-only colorspace path: extract the luma plane for the net; bicubic-upsample Cb/Cr; merge back to
//! interleaved RGB8. Matches the M1 quality path (espcn_quality.py): ESPCN on Y, bicubic on chroma.
use crate::Plane;

/// Bicubic (Catmull-Rom, a=-0.5) resample of a single plane to (dst_w, dst_h). Close to PIL BICUBIC for
/// the chroma path (chroma detail is not the SR claim). 2-D kernel over a 4x4 neighborhood.
pub fn bicubic(src: &Plane, dst_w: usize, dst_h: usize) -> Plane {
    fn wk(t: f32) -> f32 {
        // Catmull-Rom kernel, a=-0.5
        let a = -0.5;
        let t = t.abs();
        if t <= 1.0 {
            (a + 2.0) * t * t * t - (a + 3.0) * t * t + 1.0
        } else if t < 2.0 {
            a * t * t * t - 5.0 * a * t * t + 8.0 * a * t - 4.0 * a
        } else {
            0.0
        }
    }
    let sample = |data: &[f32], w_: usize, h_: usize, fx: f32, fy: f32| -> f32 {
        let (ix, iy) = (fx.floor() as isize, fy.floor() as isize);
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for m in -1..=2 {
            for n in -1..=2 {
                let px = (ix + n).clamp(0, w_ as isize - 1) as usize;
                let py = (iy + m).clamp(0, h_ as isize - 1) as usize;
                let wt = wk(fx - (ix + n) as f32) * wk(fy - (iy + m) as f32);
                acc += wt * data[py * w_ + px];
                wsum += wt;
            }
        }
        if wsum != 0.0 {
            acc / wsum
        } else {
            acc
        }
    };
    let sx = src.w as f32 / dst_w as f32;
    let sy = src.h as f32 / dst_h as f32;
    let mut out = vec![0f32; dst_w * dst_h];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let fy = (y as f32 + 0.5) * sy - 0.5;
            out[y * dst_w + x] = sample(&src.data, src.w, src.h, fx, fy);
        }
    }
    Plane {
        w: dst_w,
        h: dst_h,
        data: out,
    }
}

/// BT.601 full-range RGB8 -> (Y, Cb, Cr) planes; Y in [0,1], Cb/Cr in [0,1] centered at 0.5.
pub fn rgb8_to_ycbcr(rgb: &[u8], w: usize, h: usize) -> (Plane, Plane, Plane) {
    let mut y = vec![0f32; w * h];
    let mut cb = vec![0f32; w * h];
    let mut cr = vec![0f32; w * h];
    for i in 0..w * h {
        let (r, g, b) = (rgb[3 * i] as f32, rgb[3 * i + 1] as f32, rgb[3 * i + 2] as f32);
        y[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        cb[i] = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0) / 255.0;
        cr[i] = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0) / 255.0;
    }
    (
        Plane { w, h, data: y },
        Plane { w, h, data: cb },
        Plane { w, h, data: cr },
    )
}

/// Merge upscaled Y with upscaled Cb/Cr back to interleaved RGB8. All three planes same dims.
pub fn ycbcr_to_rgb8(y: &Plane, cb: &Plane, cr: &Plane) -> Vec<u8> {
    let (w, h) = (y.w, y.h);
    let mut rgb = vec![0u8; w * h * 3];
    for i in 0..w * h {
        let yy = y.data[i] * 255.0;
        let cbb = cb.data[i] * 255.0 - 128.0;
        let crr = cr.data[i] * 255.0 - 128.0;
        let r = yy + 1.402 * crr;
        let g = yy - 0.344136 * cbb - 0.714136 * crr;
        let b = yy + 1.772 * cbb;
        rgb[3 * i] = r.round().clamp(0.0, 255.0) as u8;
        rgb[3 * i + 1] = g.round().clamp(0.0, 255.0) as u8;
        rgb[3 * i + 2] = b.round().clamp(0.0, 255.0) as u8;
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bicubic_identity_on_same_size() {
        let p = Plane {
            w: 4,
            h: 4,
            data: (0..16).map(|i| i as f32).collect(),
        };
        let q = bicubic(&p, 4, 4);
        for i in 0..16 {
            assert!(
                (q.data[i] - p.data[i]).abs() < 1e-3,
                "idx {i}: {} vs {}",
                q.data[i],
                p.data[i]
            );
        }
    }
    #[test]
    fn rgb_roundtrip_gray() {
        let rgb = vec![128u8; 3 * 4]; // 2x2 gray
        let (y, cb, cr) = rgb8_to_ycbcr(&rgb, 2, 2);
        let back = ycbcr_to_rgb8(&y, &cb, &cr);
        for i in 0..12 {
            assert!((back[i] as i32 - 128).abs() <= 1, "idx {i} = {}", back[i]);
        }
    }
    #[test]
    fn bicubic_doubles_dims() {
        let p = Plane {
            w: 2,
            h: 2,
            data: vec![0.0, 1.0, 1.0, 0.0],
        };
        let q = bicubic(&p, 4, 4);
        assert_eq!((q.w, q.h), (4, 4));
    }
}
