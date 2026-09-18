//! Device gate for the AR embedding gather driven from RUST, on the full-ELF path.
//!
//! This closes the last non-GEMM gap between the AR bricks and `npu-s2`: `embedding-gather` was
//! device-green only through pyxrt, and `npu-s2`'s `S2Design` dispatches registered xclbins, which
//! structurally cannot carry a parameter scratchpad (XRT builds a plain `xrt::kernel`'s module
//! empty, so `get_ctrl_scratchpad_bo()` throws -- see `ScratchpadParams`' own doc comment for the
//! three `xrt_kernel.cpp` citations). So the AR gather has to go through `ElfResident`, and this
//! proves it does.
//!
//! Unlike `scratchpad_resident_probe`, the offset is resolved BY NAME through `ScratchpadParams`
//! rather than written to a hardcoded byte 0 -- the parser exists precisely so a design's parameter
//! layout stops being a magic constant at the call site.
//!
//! Gate: exact equality against the selected row (a gather is zero-compute, so anything short of
//! exact is a defect, not noise) AND run-to-run bit-identity, which is this project's blocking
//! check. `idx` is repeated deliberately: the same offset must re-dispatch identically.
//!
//! Usage: ar_gather_probe <aie.elf> <params.txt>   (single-tenant; hold scripts/npu_lock.sh)

use npu_xrt::{Device, ScratchpadParams, FLAG_HOST_ONLY};
use std::path::Path;

// Must match the built ELF: gen_embedding_gather.build_design(n_rows=64, d=2560, chunk_n=16).
// D=2560 is the real AR table width for all three tables; N_ROWS is a test-sized stand-in for the
// 155776/40960/4096-row real tables, which the design's L3-resident contract does not care about.
const N_ROWS: usize = 64;
const D: usize = 2560;
const PARAM: &str = "row_off";

/// bf16 by truncating f32's high half -- so every table value is exactly representable and the
/// comparison below can demand equality rather than a tolerance.
fn bf16_of(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Host-side clamp + element-unit offset, mirroring `gen_embedding_gather.row_offset_elements`.
/// The kernel never sees an index (row selection happens in the DMA), so the clamp is the host's.
fn row_offset_elements(idx: i64) -> i32 {
    (idx.clamp(0, N_ROWS as i64 - 1) * D as i64) as i32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let elf_path = args.next().expect("usage: ar_gather_probe <aie.elf> <params.txt>");
    let params_path = args.next().expect("usage: ar_gather_probe <aie.elf> <params.txt>");

    let elf = std::fs::read(&elf_path).unwrap_or_else(|e| panic!("read {elf_path}: {e}"));
    let params = ScratchpadParams::parse(Path::new(&params_path))
        .unwrap_or_else(|e| panic!("parse {params_path}: {e}"));

    let dev = Device::open(0).expect("open device 0");
    let res = dev
        .open_elf_resident(&elf, Some("embgather_verify:sequence"))
        .expect("open_elf_resident (ELF must be a --get-scratchpad-parameters build)");
    println!("[ar_gather] resident open OK, ctrl scratchpad = {} bytes, params declare {} bytes",
             res.scratchpad_size(), params.size_bytes());

    let table: Vec<u16> = (0..N_ROWS * D).map(|i| bf16_of(i as f32 * 1e-4)).collect();
    let mut table_bytes = Vec::with_capacity(table.len() * 2);
    for v in &table {
        table_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let in_bo = dev.alloc_bo_raw(table_bytes.len(), FLAG_HOST_ONLY, 0).expect("alloc table");
    let out_bo = dev.alloc_bo_raw(D * 2, FLAG_HOST_ONLY, 0).expect("alloc out");
    in_bo.write_bytes(&table_bytes).unwrap();
    in_bo.sync_to_device().unwrap();
    res.bind(&[&in_bo, &out_bo]).expect("bind arenas");

    let mut all_ok = true;
    for &idx in &[0i64, N_ROWS as i64 - 1, 17, 17, 5, 40] {
        let off = row_offset_elements(idx);
        let want = &table[off as usize..off as usize + D];

        let mut reads = Vec::new();
        for _ in 0..2 {
            out_bo.write_bytes(&vec![0u8; D * 2]).unwrap();
            out_bo.sync_to_device().unwrap();
            params.write_i32(&res, PARAM, off).expect("scratchpad write_i32");
            res.dispatch().expect("resident dispatch");
            out_bo.sync_from_device().unwrap();
            let mut got = vec![0u8; D * 2];
            out_bo.read_bytes(&mut got).unwrap();
            reads.push(got.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>());
        }

        let determ = reads[0] == reads[1];
        let exact = reads[0] == want;
        let nonzero = reads[0].iter().any(|&v| v != 0);
        let ok = determ && exact && nonzero;
        all_ok &= ok;
        println!("  idx={idx:3} row_off={off:7} exact={exact} run2run={} nonzero={nonzero} -> {}",
                 if determ { "bit-identical" } else { "MISMATCH" },
                 if ok { "PASS" } else { "FAIL" });
    }
    println!("RESULT: {}", if all_ok { "PASS -- Rust drives the AR embedding gather on the ELF path" } else { "FAIL" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
