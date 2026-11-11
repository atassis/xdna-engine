use npu_weights::source_native::{
    plan::{WeightContract, WeightOp, WeightPlan},
    safetensors::SafetensorsFile,
    Limits,
};
use serde_json::json;
use std::collections::BTreeMap;

fn source(dir: &tempfile::TempDir) -> SafetensorsFile {
    source_with_first(dir, 0x3f80)
}
fn source_with_first(dir: &tempfile::TempDir, first: u16) -> SafetensorsFile {
    let header =
        serde_json::to_vec(&json!({"w":{"dtype":"BF16","shape":[3,2],"data_offsets":[0,12]}}))
            .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    for bits in [first, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0] {
        bytes.extend(bits.to_le_bytes());
    }
    let path = dir.path().join("model.safetensors");
    std::fs::write(&path, bytes).unwrap();
    SafetensorsFile::open(path, Limits::default()).unwrap()
}
fn contract() -> WeightContract {
    serde_json::from_value(
        json!({"version":1,"config":{"model_type":"fixture"},"buffers":{"out":[
        {"op":"rows","tensor":"w","shape":[3,2],"first":2,"count":1},
        {"op":"rows","tensor":"w","shape":[3,2],"first":0,"count":1,"scale":0.5},
        {"op":"zero","bytes":4}
    ]},"embedding":{"tensor":"w","shape":[3,2]}}),
    )
    .unwrap()
}
fn bind(c: &WeightContract, s: &SafetensorsFile) -> anyhow::Result<WeightPlan> {
    WeightPlan::bind(
        c,
        s.tensors(),
        &BTreeMap::from([("out".into(), 12)]),
        &json!({"model_type":"fixture"}),
    )
}
fn render(p: &WeightPlan) -> Vec<u8> {
    let mut got = Vec::new();
    p.write_buffer("out", &mut [0; 3], |off, b| {
        assert_eq!(off, got.len());
        got.extend_from_slice(b);
        Ok(())
    })
    .unwrap();
    got
}
#[test]
fn ordered_bounded_plan_survives_source_reader_and_propagates_sink_error() {
    let dir = tempfile::tempdir().unwrap();
    let source = source(&dir);
    let plan = bind(&contract(), &source).unwrap();
    drop(source);
    assert_eq!(
        render(&plan),
        [0xa0, 0x40, 0xc0, 0x40, 0x00, 0x3f, 0x80, 0x3f, 0, 0, 0, 0]
    );
    assert!(plan
        .write_buffer("out", &mut [0; 2], |_, _| anyhow::bail!("sink failed"))
        .unwrap_err()
        .to_string()
        .contains("sink failed"));
    assert!(plan
        .write_buffer("out", &mut [0; 1], |_, _| Ok(()))
        .is_err());
    assert!(plan
        .write_buffer("missing", &mut [0; 2], |_, _| Ok(()))
        .is_err());
    assert_eq!(plan.embedding().shape, [3, 2]);
}
#[test]
fn rejects_bad_contract_before_any_write() {
    let dir = tempfile::tempdir().unwrap();
    let source = source(&dir);
    let expected = BTreeMap::from([("out".into(), 12)]);
    for change in [
        "shape",
        "range",
        "missing",
        "version",
        "empty",
        "empty_config",
        "nan",
        "overflow",
    ] {
        let mut c = contract();
        let mut value = serde_json::to_value(&c).unwrap();
        match change {
            "shape" => value["buffers"]["out"][0]["shape"] = json!([2, 3]),
            "range" => value["buffers"]["out"][0]["first"] = json!(3),
            "missing" => value["buffers"]["out"][0]["tensor"] = json!("absent"),
            "version" => value["version"] = json!(2),
            "empty" => value["buffers"] = json!({}),
            "empty_config" => value["config"] = json!({}),
            "overflow" => value["buffers"]["out"][0]["first"] = json!(usize::MAX),
            "nan" => {
                c.buffers.get_mut("out").unwrap().push(WeightOp::Rows {
                    tensor: "w".into(),
                    shape: vec![3, 2],
                    first: 0,
                    count: 1,
                    scale: Some(f32::NAN),
                });
                assert!(bind(&c, &source).is_err());
                continue;
            }
            _ => unreachable!(),
        }
        c = serde_json::from_value(value).unwrap();
        assert!(bind(&c, &source).is_err(), "{change}");
    }
    for expected in [
        BTreeMap::from([("out".into(), 10)]),
        BTreeMap::from([("out".into(), 12), ("extra".into(), 2)]),
    ] {
        assert!(WeightPlan::bind(
            &contract(),
            source.tensors(),
            &expected,
            &json!({"model_type":"fixture"})
        )
        .is_err());
    }
    let mut tensors = source.tensors().clone();
    tensors.get_mut("w").unwrap().dtype = npu_weights::source_native::ScalarType::F16;
    assert!(WeightPlan::bind(
        &contract(),
        &tensors,
        &expected,
        &json!({"model_type":"fixture"})
    )
    .is_err());
    assert!(WeightPlan::bind(
        &contract(),
        source.tensors(),
        &expected,
        &json!({"model_type":"other"})
    )
    .is_err());
    assert!(WeightPlan::bind_config_json(
        &contract(),
        source.tensors(),
        &expected,
        br#"{"model_type":"wrong","model_type":"fixture"}"#
    )
    .is_err());
}
#[test]
fn schema_rejects_unknown_operations_fields_and_duplicate_keys() {
    let mut v = serde_json::to_value(contract()).unwrap();
    v["buffers"]["out"][0]["rounding"] = json!("truncate");
    assert!(serde_json::from_value::<WeightContract>(v).is_err());
    assert!(
        WeightContract::from_metadata_json(br#"{"weight_inputs":{},"weight_inputs":{}}"#).is_err()
    );
    let bytes = serde_json::to_vec(&json!({"weight_inputs":contract()})).unwrap();
    assert!(WeightContract::from_metadata_json(&bytes)
        .unwrap()
        .is_some());
}
#[test]
fn explicit_rne_and_wrong_order_or_omitted_scale_are_observable() {
    let dir = tempfile::tempdir().unwrap();
    let source = source(&dir);
    let mut c = contract();
    let baseline = render(&bind(&c, &source).unwrap());
    c.buffers.get_mut("out").unwrap().swap(0, 1);
    assert_ne!(render(&bind(&c, &source).unwrap()), baseline);
    c = contract();
    if let WeightOp::Rows { scale, .. } = &mut c.buffers.get_mut("out").unwrap()[1] {
        *scale = None;
    }
    assert_ne!(render(&bind(&c, &source).unwrap()), baseline);
    for (factor, expected) in [(1.00390625f32, 0x3f80u16), (1.01171875f32, 0x3f82u16)] {
        c = contract();
        c.buffers.get_mut("out").unwrap()[0] = WeightOp::Rows {
            tensor: "w".into(),
            shape: vec![3, 2],
            first: 0,
            count: 1,
            scale: Some(factor),
        };
        let bytes = render(&bind(&c, &source).unwrap());
        assert_eq!(&bytes[..2], expected.to_le_bytes());
    }
}
#[test]
fn weight_values_do_not_change_contract_and_large_zero_fill_is_bounded() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let first = source(&a);
    let second = source_with_first(&b, 0x4100);
    let c = contract();
    let identity = serde_json::to_vec(&c).unwrap();
    assert_ne!(
        render(&bind(&c, &first).unwrap()),
        render(&bind(&c, &second).unwrap())
    );
    assert_eq!(identity, serde_json::to_vec(&c).unwrap());
    let mut zero = c.clone();
    zero.buffers
        .insert("out".into(), vec![WeightOp::Zero { bytes: 1024 * 1024 }]);
    let plan = WeightPlan::bind(
        &zero,
        first.tensors(),
        &BTreeMap::from([("out".into(), 1024 * 1024)]),
        &json!({"model_type":"fixture"}),
    )
    .unwrap();
    let mut count = 0;
    plan.write_buffer("out", &mut [0xff; 7], |off, bytes| {
        assert_eq!(off, count);
        assert!(bytes.len() <= 6 && bytes.iter().all(|&b| b == 0));
        count += bytes.len();
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 1024 * 1024);
}

#[test]
fn scaled_values_reject_overflow_during_bf16_rounding() {
    for first in [0x7f7f, 0xff7f] {
        let dir = tempfile::tempdir().unwrap();
        let source = source_with_first(&dir, first);
        let mut c = contract();
        c.buffers.insert(
            "out".into(),
            vec![WeightOp::Rows {
                tensor: "w".into(),
                shape: vec![3, 2],
                first: 0,
                count: 1,
                scale: Some(1.002),
            }],
        );
        let plan = WeightPlan::bind(
            &c,
            source.tensors(),
            &BTreeMap::from([("out".into(), 4)]),
            &json!({"model_type":"fixture"}),
        )
        .unwrap();
        let error = plan
            .write_buffer("out", &mut [0; 4], |_, _| Ok(()))
            .unwrap_err();
        assert!(error.to_string().contains("non-finite scaled BF16 value"));
    }
}
