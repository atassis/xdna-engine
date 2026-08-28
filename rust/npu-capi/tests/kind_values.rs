//! The C ABI's kind values are a published contract: 0 = asr, 1 = embed, 2 = diarize. A renumber
//! silently breaks every compiled consumer, so it is pinned by a test rather than by a comment.
#[test]
fn kind_discriminants_are_append_only() {
    use npu_engine::ModelKind;
    let code = |k: ModelKind| match k {
        ModelKind::Asr => 0,
        ModelKind::Embed => 1,
        ModelKind::Diarize => 2,
    };
    assert_eq!(code(ModelKind::Asr), 0);
    assert_eq!(code(ModelKind::Embed), 1);
    assert_eq!(code(ModelKind::Diarize), 2);
}
