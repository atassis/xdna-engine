//! On-NPU full (bidirectional) attention for the Whisper encoder.
//!
//! Loads the static-shape MHA xclbin (`gen_encoder_mha.py`, h=12 s=1500 d=64 causal=False) and replaces
//! the host `mha(&q,&k,&v,…)`. Q/K/V/O live in resident BOs in the op's `[heads, seq_pad, d]` bf16
//! layout (head-major, seq padded 1500→1536). The kernel masks the padded KV columns internally
//! (S_kv_effective=1500 baked into the static design), so the pad rows are don't-care.
//!
//! ABI: kernel(opcode=3, instr[gid1], n_instr, Q[gid3], K[gid4], V[gid5], O[gid6]) via `run_mha`.

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_xrt::{pack_f32_to_bf16, unpack_bf16_to_f32, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const HEADS: usize = 12;
const D: usize = 64;
const SEQ: usize = 1500;
const SEQ_PAD: usize = 1536;
const DMODEL: usize = HEADS * D; // 768
const ELEMS: usize = HEADS * SEQ_PAD * D; // 1_179_648
const OPCODE: u32 = 3;

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn u16_bytes_mut(v: &mut [u16]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

pub struct MhaNpu {
    kern: Rc<Kernel>,
    instr: Bo,
    n_instr: usize,
    bo_q: Bo,
    bo_k: Bo,
    bo_v: Bo,
    bo_o: Bo,
}

impl MhaNpu {
    /// Load the MHA xclbin + insts onto an already-open Device (single-tenant; reuse the handle).
    pub fn open(dev: &Rc<Device>, xclbin: &Path, insts: &Path) -> Result<Self, String> {
        let kern = dev
            .load_kernel(xclbin.to_str().unwrap(), None)
            .map_err(|e| format!("MhaNpu: load {}: {e}", xclbin.display()))?;
        let ibytes = std::fs::read(insts).map_err(|e| format!("MhaNpu: read insts {}: {e}", insts.display()))?;
        let n_instr = ibytes.len() / 4; // 4 bytes/instr
        let g = |i| kern.group_id(i).unwrap();

        let instr = dev
            .alloc_bo(&kern, ibytes.len(), FLAG_CACHEABLE, g(1))
            .map_err(|e| format!("MhaNpu: alloc instr BO: {e}"))?;
        instr.write_bytes(&ibytes).map_err(|e| format!("MhaNpu: write instr: {e}"))?;
        instr.sync_to_device().map_err(|e| format!("MhaNpu: sync instr: {e}"))?;

        let nbytes = ELEMS * 2; // bf16
        let mk = |gid, what: &str| {
            dev.alloc_bo(&kern, nbytes, FLAG_HOST_ONLY, gid)
                .map_err(|e| format!("MhaNpu: alloc {what} BO: {e}"))
        };
        let bo_q = mk(g(3), "Q")?;
        let bo_k = mk(g(4), "K")?;
        let bo_v = mk(g(5), "V")?;
        let bo_o = mk(g(6), "O")?;
        eprintln!("[MhaNpu] loaded {} (h={HEADS} s={SEQ} d={D}, {n_instr} instr)", xclbin.display());
        Ok(MhaNpu { kern, instr, n_instr, bo_q, bo_k, bo_v, bo_o })
    }

    /// `q`,`k`,`v`: `[SEQ, 768]` host f32 → context `[SEQ, 768]` host f32.
    pub fn forward(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        self.upload(q, &self.bo_q);
        self.upload(k, &self.bo_k);
        self.upload(v, &self.bo_v);
        self.kern
            .run_mha(OPCODE, &self.instr, self.n_instr, &self.bo_q, &self.bo_k, &self.bo_v, &self.bo_o)
            .expect("MhaNpu: run_mha");
        self.bo_o.sync_from_device().expect("MhaNpu: sync O");

        let mut obf = vec![0u16; ELEMS];
        self.bo_o.read_bytes(u16_bytes_mut(&mut obf)).expect("MhaNpu: read O");
        let mut of32 = vec![0f32; ELEMS];
        unpack_bf16_to_f32(&obf, &mut of32);
        // O is [HEADS, SEQ_PAD, D] head-major; gather the valid [SEQ, 768].
        let mut ctx = Array2::<f32>::zeros((SEQ, DMODEL));
        for h in 0..HEADS {
            for s in 0..SEQ {
                let base = h * SEQ_PAD * D + s * D;
                for dd in 0..D {
                    ctx[[s, h * D + dd]] = of32[base + dd];
                }
            }
        }
        ctx
    }

    /// Pack `[SEQ, 768]` host f32 into the op's `[HEADS, SEQ_PAD, D]` bf16 BO (pad rows zeroed).
    fn upload(&self, x: &Array2<f32>, bo: &Bo) {
        let mut buf = vec![0f32; ELEMS];
        for h in 0..HEADS {
            for s in 0..SEQ {
                let base = h * SEQ_PAD * D + s * D;
                for dd in 0..D {
                    buf[base + dd] = x[[s, h * D + dd]];
                }
            }
        }
        let mut bf = vec![0u16; ELEMS];
        pack_f32_to_bf16(&buf, &mut bf);
        bo.write_bytes(u16_bytes(&bf)).expect("MhaNpu: write data");
        bo.sync_to_device().expect("MhaNpu: sync data");
    }
}
