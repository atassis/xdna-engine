// Integration test for the device actor + Handle, using the mock loader (gated behind `testkit`).
// Run with: cargo test -p npu-runtime --features testkit
#![cfg(feature = "testkit")]
use npu_engine::ModelKind;
use npu_runtime::actor::start;
use npu_runtime::config::{Config, Defaults, ModelCfg, ServerCfg};
use npu_runtime::loader::mock::MockLoader;
use std::collections::BTreeMap;

#[test]
fn actor_serves_and_echoes_model() {
    let mut t = BTreeMap::new();
    t.insert("bge".to_string(), Ok((ModelKind::Embed, 1)));
    t.insert("asr".to_string(), Ok((ModelKind::Asr, 1)));
    let cfg = Config {
        server: ServerCfg { max_resident: 8, ..Default::default() },
        defaults: Defaults { asr: Some("asr".into()), embed: Some("bge".into()) },
        models: vec![
            ModelCfg { name: "bge".into(), scenario: "x".into() },
            ModelCfg { name: "asr".into(), scenario: "y".into() },
        ],
    };
    let (h, join) = start(cfg, Box::new(MockLoader { table: t }));
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
