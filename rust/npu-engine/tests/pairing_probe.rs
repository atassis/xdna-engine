//! Device-free report of what `NpuDecodeStep::with_prefill` would decide about a real artifact
//! pair, by running the ACTUAL checks rather than a second implementation of them.
//!
//! `scripts/check_prefill_arena_pairing.py` covers the arena half in Python and is what install
//! gates on; this covers the model-level half, which only exists in Rust and until now could not
//! be exercised without opening the device.
//!
//! Prints, asserts nothing -- a deliberate negative probe is a normal use.
//!
//!   DECODE_DIR=<dir> PREFILL_DIR=<dir> \
//!     cargo test -p npu-engine --test pairing_probe -- --nocapture
use npu_engine::llm::LlmArtifact;
use std::path::Path;

#[test]
fn report_pairing() {
    let (Ok(d), Ok(p)) = (std::env::var("DECODE_DIR"), std::env::var("PREFILL_DIR")) else {
        eprintln!("skip: set DECODE_DIR and PREFILL_DIR");
        return;
    };
    let da = LlmArtifact::load(Path::new(&d)).expect("load decode");
    let pa = LlmArtifact::load_prefill(Path::new(&p)).expect("load prefill");
    let wins = |a: &LlmArtifact| {
        a.kv_windows.iter().map(|&(_, hd, c, _, _, _)| (hd, c)).collect::<Vec<_>>()
    };
    eprintln!("decode  L={} S={} kv_block={} kv_windows={:?}",
              da.n_layers, da.max_seq, da.kv_block, wins(&da));
    eprintln!("prefill L={} S={} kv_block={} M={} kv_windows={:?}",
              pa.n_layers, pa.max_seq, pa.kv_block, pa.batch, wins(&pa));
    for (what, r) in [("check_prefill_pairing", da.check_prefill_pairing(&pa)),
                      ("check_shared_layout_agrees", da.check_shared_layout_agrees(&pa))] {
        match r {
            Ok(()) => eprintln!("{what}: PASS"),
            Err(e) => eprintln!("{what}: FAIL -- {e}"),
        }
    }
    let caps: Vec<usize> = pa.kv_windows.iter().map(|&(_, _, c, _, _, _)| c).collect();
    let ring_caps: Vec<usize> =
        pa.mask_ring.as_ref().map(|mr| mr.geoms.iter().map(|&(_, c)| c).collect())
            .unwrap_or_default();
    eprintln!("mask_ring={:?}", pa.mask_ring.as_ref().map(|mr| &mr.geoms));
    eprintln!("batchable_window = {} of S={}",
              npu_engine::llm::npu_prefill::batchable_window(pa.max_seq, pa.batch, &caps, &ring_caps),
              pa.max_seq);
}
