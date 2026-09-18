//! Gates `npu_s2::chain::chain_offset` against `decoder_chain.chain_offset()`'s own REAL output
//! (not a reimplementation of its formula) -- `tests/fixtures/chain_offset_cases.json`, generated
//! by importing `decoder_chain` (toolchain-env python, `aie.iron` importable) and calling the
//! shipped function directly. Safe to commit: these are pure architecture numbers (kernel sizes,
//! strides), not model weight content -- see the fixture's own `_generated_by` note.
//!
//! Device-free: `chain_offset` touches no design, no GGUF, no XRT.

use npu_s2::chain::{chain_offset, StageOffset};

#[test]
fn chain_offset_matches_python_rail_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chain_offset_cases.json");
    let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();

    let ctx_head = fixture["CTX_HEAD"].as_i64().unwrap();
    let ctx_tail = fixture["CTX_TAIL"].as_i64().unwrap();
    let res_ctx = fixture["RES_CTX_PER_STAGE"].as_i64().unwrap();
    let up_ctx = fixture["UP_CTX_PER_STAGE"].as_i64().unwrap();
    let strides = &fixture["STRIDES"];
    let stages: Vec<StageOffset> = (1..=4)
        .map(|s| StageOffset { up_ctx, stride: strides[s.to_string()].as_i64().unwrap(), res_ctx })
        .collect();

    let cases = fixture["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for c in cases {
        let latent_start = c["latent_start"].as_i64().unwrap();
        let latent_len = c["latent_len"].as_i64().unwrap();
        let want = (c["audio_start"].as_i64().unwrap(), c["audio_len"].as_i64().unwrap());
        let got = chain_offset(ctx_head, ctx_tail, &stages, latent_start, latent_len).unwrap();
        assert_eq!(got, want, "latent_start={latent_start} latent_len={latent_len}");
    }
}
