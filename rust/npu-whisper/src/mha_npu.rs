//! On-NPU full (bidirectional) attention for the Whisper encoder.
//!
//! Loads the static-shape MHA xclbin (`gen_encoder_mha.py --heads H`, s=1500 d=64, non-causal) and replaces
//! the host `mha(&q,&k,&v,…)`. Q/K/V/O live in resident BOs in the op's `[heads, seq_pad, d]` bf16
//! layout (head-major, seq padded 1500→1536). The kernel masks the padded KV columns internally
//! (S_kv_effective=1500 baked into the static design), so the pad rows are don't-care.
//!
//! ABI: kernel(opcode=3, instr[gid1], n_instr, Q[gid3], K[gid4], V[gid5], O[gid6]) via `run_mha`.

use std::path::Path;
use std::rc::Rc;

use ndarray::prelude::*;
use npu_xrt::{pack_f32_to_bf16, unpack_bf16_to_f32, Bo, Device, Kernel, FLAG_CACHEABLE, FLAG_HOST_ONLY};

const D: usize = 64; // every Whisper size: 768/12 == 1280/20 == 64
const SEQ: usize = 1500;
const SEQ_PAD: usize = 1536;
const OPCODE: u32 = 3;

fn u16_bytes(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn u16_bytes_mut(v: &mut [u16]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

pub struct MhaNpu {
    /// Head count this xclbin was built for. Not a constant: whisper-small is 12, large-v3-turbo 20,
    /// and the buffer layout is `[heads, seq_pad, d]`, so a wrong value silently misreads the output.
    heads: usize,
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
    pub fn open(dev: &Rc<Device>, heads: usize, xclbin: &Path, insts: &Path) -> Result<Self, String> {
        let elems = heads * SEQ_PAD * D;
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

        let nbytes = elems * 2; // bf16
        let mk = |gid, what: &str| {
            dev.alloc_bo(&kern, nbytes, FLAG_HOST_ONLY, gid)
                .map_err(|e| format!("MhaNpu: alloc {what} BO: {e}"))
        };
        let bo_q = mk(g(3), "Q")?;
        let bo_k = mk(g(4), "K")?;
        let bo_v = mk(g(5), "V")?;
        let bo_o = mk(g(6), "O")?;
        eprintln!("[MhaNpu] loaded {} (h={heads} s={SEQ} d={D}, {n_instr} instr)", xclbin.display());
        Ok(MhaNpu { heads, kern, instr, n_instr, bo_q, bo_k, bo_v, bo_o })
    }

    /// `q`,`k`,`v`: `[SEQ, heads*64]` host f32 → context `[SEQ, heads*64]` host f32.
    pub fn forward(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        self.upload(q, &self.bo_q);
        self.upload(k, &self.bo_k);
        self.upload(v, &self.bo_v);
        self.kern
            .run_mha(OPCODE, &self.instr, self.n_instr, &self.bo_q, &self.bo_k, &self.bo_v, &self.bo_o)
            .expect("MhaNpu: run_mha");
        self.bo_o.sync_from_device().expect("MhaNpu: sync O");

        let elems = self.heads * SEQ_PAD * D;
        let mut obf = vec![0u16; elems];
        self.bo_o.read_bytes(u16_bytes_mut(&mut obf)).expect("MhaNpu: read O");
        let mut of32 = vec![0f32; elems];
        unpack_bf16_to_f32(&obf, &mut of32);
        // O is [heads, SEQ_PAD, D] head-major; gather the valid [SEQ, heads*D].
        let mut ctx = Array2::<f32>::zeros((SEQ, self.heads * D));
        for h in 0..self.heads {
            for s in 0..SEQ {
                let base = h * SEQ_PAD * D + s * D;
                for dd in 0..D {
                    ctx[[s, h * D + dd]] = of32[base + dd];
                }
            }
        }
        ctx
    }

    /// Pack `[SEQ, heads*D]` host f32 into the op's `[heads, SEQ_PAD, D]` bf16 BO (pad rows zeroed).
    fn upload(&self, x: &Array2<f32>, bo: &Bo) {
        let elems = self.heads * SEQ_PAD * D;
        let mut buf = vec![0f32; elems];
        for h in 0..self.heads {
            for s in 0..SEQ {
                let base = h * SEQ_PAD * D + s * D;
                for dd in 0..D {
                    buf[base + dd] = x[[s, h * D + dd]];
                }
            }
        }
        let mut bf = vec![0u16; elems];
        pack_f32_to_bf16(&buf, &mut bf);
        bo.write_bytes(u16_bytes(&bf)).expect("MhaNpu: write data");
        bo.sync_to_device().expect("MhaNpu: sync data");
    }
}
