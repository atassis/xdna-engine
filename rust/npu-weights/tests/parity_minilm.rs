mod common;

// rust/npu-weights/tests/parity_minilm.rs
//
// Parity for sentence-transformers BERT-family text-embedding models against the Python oracle
// (export_minilm.py). all-MiniLM-L6-v2 = 6-layer BERT (exercises the layer-count inference in the
// generalized `bert` arch); bge-small/e5-small = 12-layer BERT-family (same arch, minor naming).
use std::path::Path;

fn check_one(sub: &str, hf: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join(format!("artifacts/{sub}/encoder"));
    if !refs.join("emb/word_emb.npy").exists() {
        eprintln!("SKIP {sub}: oracle missing - run .venv/bin/python scripts/export_minilm.py {hf} {sub}");
        return;
    }
    common::bake_and_verify(sub, &format!("hf:{hf}"), "bert", &refs);
}

#[test]
fn minilm_l6_checkpoint_matches_python_oracle() {
    check_one("minilm-l6", "sentence-transformers/all-MiniLM-L6-v2");
}

#[test]
fn bge_small_checkpoint_matches_python_oracle() {
    check_one("bge-small", "BAAI/bge-small-en-v1.5");
}

#[test]
fn e5_small_checkpoint_matches_python_oracle() {
    check_one("e5-small", "intfloat/e5-small-v2");
}

#[test]
fn multilingual_e5_small_checkpoint_matches_python_oracle() {
    // multilingual-e5-small reports model_type=bert / BertModel (12 layers, hidden 384) - a
    // multilingual MiniLM backbone, so it rides the generalized `bert` arch unchanged. Only the
    // word-embedding vocab is larger (multilingual); the XLM-R tokenizer omits token_type_ids
    // (export_minilm.py synthesizes zeros for the ONNX oracle - a tokenizer concern, no weight change).
    check_one("multilingual-e5-small", "intfloat/multilingual-e5-small");
}
