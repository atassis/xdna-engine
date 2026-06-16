//! Milestone-A validation of the npu-xrt resident scratchpad path (Option C piece 1) against the
//! upstream `scratchpad_addr_offset` test ELF. Registers the ELF ONCE, binds [in, out], then for each
//! runtime offset writes the i32 into the ctrl scratchpad + dispatches — proving our shim (not pyxrt)
//! drives `aiex.scratchpad_parameter` correctly. The kernel passes 8 i32 from input[offset..offset+8].
//!
//! Usage: scratchpad_resident_probe <aie.elf>   (single-tenant; expects input arange(32)).

use npu_xrt::{Device, FLAG_HOST_ONLY};

const N_IN: usize = 32;
const N_OUT: usize = 8;

fn main() {
    let elf_path = std::env::args().nth(1).expect("usage: scratchpad_resident_probe <aie.elf>");
    let elf = std::fs::read(&elf_path).unwrap_or_else(|e| panic!("read {elf_path}: {e}"));

    let dev = Device::open(0).expect("open device 0");
    let res = dev
        .open_elf_resident(&elf, Some("test:sequence"))
        .expect("open_elf_resident (ELF must be a scratchpad-parameter build)");
    let sp = res.scratchpad_size();
    println!("[probe] resident open OK, ctrl scratchpad size = {sp} bytes");
    assert!(sp > 0, "ELF has no ctrl scratchpad");

    let in_bo = dev.alloc_bo_raw(N_IN * 4, FLAG_HOST_ONLY, 0).expect("alloc in");
    let out_bo = dev.alloc_bo_raw(N_OUT * 4, FLAG_HOST_ONLY, 0).expect("alloc out");

    // input = [0,1,...,31] as i32 LE
    let mut in_bytes = Vec::with_capacity(N_IN * 4);
    for i in 0..N_IN as i32 {
        in_bytes.extend_from_slice(&i.to_le_bytes());
    }
    in_bo.write_bytes(&in_bytes).unwrap();
    in_bo.sync_to_device().unwrap();

    res.bind(&[&in_bo, &out_bo]).expect("bind arenas");

    let mut all_ok = true;
    for &offset in &[0i32, 8, 16] {
        // write the runtime offset (i32, element units) into the scratchpad at byte 0, then dispatch
        res.write_scratchpad(0, &offset.to_le_bytes()).expect("write scratchpad");
        res.dispatch().expect("resident dispatch");
        out_bo.sync_from_device().unwrap();
        let mut out_bytes = vec![0u8; N_OUT * 4];
        out_bo.read_bytes(&mut out_bytes).unwrap();
        let got: Vec<i32> =
            out_bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let expected: Vec<i32> = (offset..offset + N_OUT as i32).collect();
        let ok = got == expected;
        all_ok &= ok;
        println!("  offset={offset:2} expected={expected:?} got={got:?} {}", if ok { "PASS" } else { "FAIL" });
    }
    println!("RESULT: {}", if all_ok { "ALL PASS — shim resident scratchpad drives runtime DMA offset on ONE ELF" } else { "FAIL" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
