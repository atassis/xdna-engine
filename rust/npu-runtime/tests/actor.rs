// Integration test for the device actor + Handle, using the mock loader (gated behind `testkit`).
// Run with: cargo test -p npu-runtime --features testkit
#![cfg(feature = "testkit")]
use npu_engine::capability::{Capability, Request, Response};
use npu_runtime::actor::start;
use npu_runtime::config::{Config, Defaults, ModelCfg, ServerCfg};
use npu_runtime::loader::mock::MockLoader;
use std::collections::BTreeMap;

#[test]
fn actor_serves_and_echoes_model() {
    let mut t = BTreeMap::new();
    t.insert("bge".to_string(), Ok((Capability::EMBED, 1)));
    t.insert("asr".to_string(), Ok((Capability::ASR, 1)));
    let cfg = Config {
        server: ServerCfg { max_resident: 8, ..Default::default() },
        defaults: Defaults::from_pairs([
            (Capability::ASR, "asr".to_string()), (Capability::EMBED, "bge".to_string())]),
        models: vec![
            ModelCfg { name: "bge".into(), scenario: "x".into() },
            ModelCfg { name: "asr".into(), scenario: "y".into() },
        ],
    };
    let (h, join) = start(cfg, Box::new(MockLoader { table: t })).unwrap();
    let e = h.embed(None, "hi").unwrap();
    assert_eq!(e.model, "bge");
    assert_eq!(e.value.len(), 8);
    let tr = h.transcribe(None, vec![0i16; 4], 16_000).unwrap();
    assert_eq!(tr.model, "asr");
    assert_eq!(tr.value, "mock-text");
    assert_eq!(h.status().len(), 2);
    h.shutdown();
    join.join().unwrap();
}

/// The same two models through one slot: every request that names a different model swaps it in.
#[test]
fn actor_hot_swaps_at_one_slot() {
    let mut t = BTreeMap::new();
    t.insert("bge".to_string(), Ok((Capability::EMBED, 1)));
    t.insert("asr".to_string(), Ok((Capability::ASR, 1)));
    let cfg = Config {
        server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
        defaults: Defaults::from_pairs([
            (Capability::ASR, "asr".to_string()), (Capability::EMBED, "bge".to_string())]),
        models: vec![
            ModelCfg { name: "asr".into(), scenario: "x".into() },
            ModelCfg { name: "bge".into(), scenario: "y".into() },
        ],
    };
    let (h, join) = start(cfg, Box::new(MockLoader { table: t })).unwrap();
    for _ in 0..2 {
        assert_eq!(h.embed(None, "hi").unwrap().model, "bge");
        assert_eq!(h.transcribe(None, vec![0i16; 4], 16_000).unwrap().model, "asr");
    }
    assert_eq!(h.status().iter().filter(|s| s.idle_s.is_some()).count(), 1,
        "exactly one model may hold the single-tenant device");
    h.shutdown();
    join.join().unwrap();
}

/// A capability with no typed `Handle` helper and no `ModelKind` variant, served end to end through
/// `Handle::serve`. This is the property the whole rewiring exists for: adding TTS cost a scenario
/// entry and nothing in the actor, the registry or the router.
#[test]
fn actor_serves_a_capability_with_no_typed_helper() {
    let mut t = BTreeMap::new();
    t.insert("kokoro".to_string(), Ok((Capability::TTS, 1)));
    let cfg = Config {
        server: ServerCfg { max_resident: 1, idle_unload_s: 0, ..Default::default() },
        defaults: Defaults::from_pairs([(Capability::TTS, "kokoro".to_string())]),
        models: vec![ModelCfg { name: "kokoro".into(), scenario: "x".into() }],
    };
    let (h, join) = start(cfg, Box::new(MockLoader { table: t })).unwrap();
    let s = h.serve(Capability::TTS, None, Request::Text("hello".into())).unwrap();
    assert_eq!(s.model, "kokoro");
    match s.value {
        Response::Audio { pcm, sample_rate } => { assert_eq!(pcm.len(), 8); assert_eq!(sample_rate, 24_000); }
        other => panic!("tts returned a {} response", other.shape()),
    }
    // ...and asking that model for a capability it does not have is still an error, not a swap.
    assert!(h.serve(Capability::ASR, Some("kokoro"), Request::Audio { pcm: vec![0i16; 4], sample_rate: 16_000 }).is_err());
    h.shutdown();
    join.join().unwrap();
}

/// The owner's constraint, pinned: adding a diarize capability cannot move the configured ASR
/// default, and an unconfigured diarize request must fail loudly rather than fall back to the ASR
/// model that happens to be resident.
#[test]
fn diarize_routes_by_capability_and_leaves_the_asr_default_alone() {
    let mut table = BTreeMap::new();
    table.insert("parakeet".to_string(), Ok((Capability::ASR, 1u64)));
    let l = MockLoader { table };
    let cfg = Config {
        server: ServerCfg { max_resident: 2, ..Default::default() },
        defaults: Defaults::from_pairs([(Capability::ASR, "parakeet".to_string())]),
        models: vec![ModelCfg { name: "parakeet".into(), scenario: "x".into() }],
    };
    let (h, join) = start(cfg, Box::new(l)).unwrap();
    let Err(e) = h.diarize(None, vec![0i16; 16], 16_000) else {
        panic!("an unconfigured diarize capability must not resolve to the resident asr model");
    };
    assert!(e.to_string().contains("diarize"), "must name the missing capability, got: {e}");
    assert!(h.transcribe(None, vec![0i16; 16], 16_000).is_ok(),
        "the asr default must still resolve after a diarize capability exists");
    h.shutdown();
    let _ = join.join();
}
