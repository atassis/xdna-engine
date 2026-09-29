//! FSR1 (EASU+RCAS) x3 as a fixed-function NPU backend: one exported streamed design
//! (`designs/fsr1/fsr1_frame.py export`) upscales a whole frame in one dispatch. The frame is cut
//! into source tiles on the host; the tile contract is `designs/fsr1/kernel/fsr1_bf16.cc`'s and
//! the tiling mirrors `fsr1_frame.py`'s `plan`/`pack`/`unpack`.
use crate::SrError;
use npu_xrt::{Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};
use serde::Deserialize;
use std::path::Path;
use std::rc::Rc;

const OPCODE: u32 = 3;
const FLAG_TOP: u8 = 1;
const FLAG_BOTTOM: u8 = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct TileGeom {
    pub tw: usize,
    pub th: usize,
    pub pad: usize,
    pub rs: usize,
    pub in_bytes: usize,
    pub out_bytes: usize,
}

/// `fsr1.json`, written by the exporter next to the xclbin it describes.
#[derive(Debug, Clone, Deserialize)]
pub struct Fsr1Config {
    pub name: String,
    pub scale: usize,
    pub in_w: usize,
    pub in_h: usize,
    pub tile: TileGeom,
    pub n_tiles: usize,
    pub xclbin: String,
    pub insts: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Tile {
    x0: usize,
    lo: usize,
    hi: usize,
    y0: usize,
    flags: u8,
}

fn plan(w: usize, h: usize, g: &TileGeom) -> Result<Vec<Tile>, SrError> {
    if w < g.tw || h % g.th != 0 {
        return Err(SrError::Load(format!("{w}x{h}: need w >= {} and h % {} == 0", g.tw, g.th)));
    }
    let mut cols = Vec::new();
    let (mut covered, mut x0) = (0usize, 0usize);
    while covered < w {
        x0 = x0.min(w - g.tw);
        let lo = (if x0 == 0 { 0 } else { 1 }).max(covered.saturating_sub(x0));
        let hi = if x0 + g.tw == w { g.tw - 1 } else { g.tw - 2 };
        cols.push((x0, lo, hi));
        covered = x0 + hi + 1;
        x0 += g.tw - 2;
    }
    let mut tiles = Vec::with_capacity(cols.len() * h / g.th);
    for y0 in (0..h).step_by(g.th) {
        let flags = if y0 == 0 { FLAG_TOP } else { 0 } | if y0 + g.th == h { FLAG_BOTTOM } else { 0 };
        tiles.extend(cols.iter().map(|&(x0, lo, hi)| Tile { x0, lo, hi, y0, flags }));
    }
    Ok(tiles)
}

pub struct Fsr1Engine {
    cfg: Fsr1Config,
    tiles: Vec<Tile>,
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_in: Bo,
    bo_out: Bo,
    inbuf: Vec<u8>,
    outbuf: Vec<u8>,
    _dev: Device,
}

impl Fsr1Engine {
    /// `dir` is the directory holding `fsr1.json`; its `xclbin`/`insts` resolve against it.
    pub fn load(cfg: Fsr1Config, dir: &Path) -> Result<Fsr1Engine, SrError> {
        let tiles = plan(cfg.in_w, cfg.in_h, &cfg.tile)?;
        if tiles.len() != cfg.n_tiles {
            return Err(SrError::Load(format!(
                "{}: plan has {} tiles, design was exported for {}", cfg.name, tiles.len(), cfg.n_tiles)));
        }
        let xclbin = dir.join(&cfg.xclbin);
        let insts_path = dir.join(&cfg.insts);
        let insts = std::fs::read(&insts_path)
            .map_err(|e| SrError::Load(format!("read {}: {e}", insts_path.display())))?;
        let dev = Device::open(0).map_err(SrError::Device)?;
        let kern = dev.load_kernel(&xclbin.to_string_lossy(), None).map_err(SrError::Load)?;
        let g = |arg: i32| kern.group_id(arg).map_err(SrError::Device);
        let instr = dev.alloc_bo(&kern, insts.len(), FLAG_CACHEABLE, g(1)?).map_err(SrError::Device)?;
        instr.write_bytes(&insts).map_err(SrError::Device)?;
        instr.sync_to_device().map_err(SrError::Device)?;
        let in_len = cfg.n_tiles * cfg.tile.in_bytes;
        let out_len = cfg.n_tiles * cfg.tile.out_bytes;
        let bo_in = dev.alloc_bo(&kern, in_len, FLAG_HOST_ONLY, g(3)?).map_err(SrError::Device)?;
        let bo_out = dev.alloc_bo(&kern, out_len, FLAG_HOST_ONLY, g(4)?).map_err(SrError::Device)?;
        Ok(Fsr1Engine {
            tiles,
            kern,
            instr,
            n_instr: insts.len() / 4,
            bo_in,
            bo_out,
            inbuf: vec![0; in_len],
            outbuf: vec![0; out_len],
            _dev: dev,
            cfg,
        })
    }

    pub fn scale(&self) -> usize {
        self.cfg.scale
    }

    pub fn upscale_rgb8(&mut self, rgb: &[u8], w: usize, h: usize)
        -> Result<(Vec<u8>, usize, usize), SrError> {
        if (w, h) != (self.cfg.in_w, self.cfg.in_h) {
            return Err(SrError::Frame(format!(
                "{}: exported for {}x{}, got {w}x{h}", self.cfg.name, self.cfg.in_w, self.cfg.in_h)));
        }
        let t0 = std::time::Instant::now();
        self.pack(rgb);
        let t1 = std::time::Instant::now();
        self.bo_in.write_bytes(&self.inbuf).map_err(SrError::Device)?;
        self.bo_in.sync_to_device().map_err(SrError::Device)?;
        let t2 = std::time::Instant::now();
        self.kern
            .run_kernel(OPCODE, &self.instr, self.n_instr, &[&self.bo_in, &self.bo_out])
            .map_err(SrError::Device)?;
        let t3 = std::time::Instant::now();
        self.bo_out.sync_from_device().map_err(SrError::Device)?;
        self.bo_out.read_bytes(&mut self.outbuf).map_err(SrError::Device)?;
        let t4 = std::time::Instant::now();
        let (ow, oh) = (w * 3, h * 3);
        let mut out = vec![0u8; ow * oh * 3];
        self.unpack(&mut out, ow);
        if std::env::var_os("NPU_SR_TIMING").is_some() {
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
            eprintln!("[fsr1] pack {:.2} upload {:.2} dispatch {:.2} readback {:.2} unpack {:.2} ms",
                      ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, std::time::Instant::now()));
        }
        Ok((out, ow, oh))
    }

    /// Tile i = 3 planes x (th+2*pad) rows x rs, column x0-pad.. clamped to the frame, then the
    /// flags byte.
    fn pack(&mut self, rgb: &[u8]) {
        let (w, h, g) = (self.cfg.in_w, self.cfg.in_h, &self.cfg.tile);
        let rows = g.th + 2 * g.pad;
        for (t, buf) in self.tiles.iter().zip(self.inbuf.chunks_exact_mut(g.in_bytes)) {
            for r in 0..rows {
                let y = (t.y0 + r).saturating_sub(g.pad).min(h - 1);
                for i in 0..g.tw + 2 * g.pad {
                    let x = (t.x0 + i).saturating_sub(g.pad).min(w - 1);
                    let px = &rgb[3 * (y * w + x)..3 * (y * w + x) + 3];
                    for c in 0..3 {
                        buf[(c * rows + r) * g.rs + i] = px[c];
                    }
                }
            }
            buf[3 * rows * g.rs] = t.flags;
        }
    }

    /// Tile output is phase-planar u8 [c][oy][px][t]; output column 3t+px.
    fn unpack(&self, out: &mut [u8], ow: usize) {
        let g = &self.cfg.tile;
        let oh_t = 3 * g.th;
        for (t, buf) in self.tiles.iter().zip(self.outbuf.chunks_exact(g.out_bytes)) {
            for oy in 0..oh_t {
                let row = (3 * t.y0 + oy) * ow;
                for lane in t.lo..=t.hi {
                    for px in 0..3 {
                        let o = 3 * (row + 3 * (t.x0 + lane) + px);
                        for c in 0..3 {
                            out[o + c] = buf[((c * oh_t + oy) * 3 + px) * g.tw + lane];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> TileGeom {
        TileGeom { tw: 32, th: 12, pad: 2, rs: 64, in_bytes: 3 * 16 * 64 + 64, out_bytes: 3 * 36 * 96 }
    }

    /// Same plan as fsr1_frame.py for the exported 640x360 design: 22 column tiles, the last
    /// shifted to end on the frame edge.
    #[test]
    fn plan_matches_the_python_exporter() {
        let t = plan(640, 360, &geom()).unwrap();
        assert_eq!(t.len(), 660);
        assert_eq!(t[0], Tile { x0: 0, lo: 0, hi: 30, y0: 0, flags: FLAG_TOP });
        assert_eq!(t[1], Tile { x0: 30, lo: 1, hi: 30, y0: 0, flags: FLAG_TOP });
        assert_eq!(t[21], Tile { x0: 608, lo: 23, hi: 31, y0: 0, flags: FLAG_TOP });
        assert_eq!(t[659].flags, FLAG_BOTTOM);
    }

    /// Every output column is written by exactly one tile.
    #[test]
    fn plan_covers_each_column_once() {
        for w in [32, 33, 62, 640, 1280] {
            let mut n = vec![0u32; w];
            for t in plan(w, 12, &geom()).unwrap() {
                for l in t.lo..=t.hi {
                    n[t.x0 + l] += 1;
                }
            }
            assert!(n.iter().all(|&c| c == 1), "w={w}");
        }
    }
}
