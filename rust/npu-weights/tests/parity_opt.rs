// rust/npu-weights/tests/parity_opt.rs
// Bakes facebook/opt-125m via the `opt` arch and checks every baked tensor against the Python oracle
// npy (scripts/convert_opt125m.py). Refs dir is the model root `artifacts/opt-125m`; arena names
// (embed_tokens, embed_positions, ln_final.*, lm_head.weight, L{i}/...) map directly to the oracle's
// npy paths. The hf: source resolves a F16 safetensors (oracle used the F32 pickle) - well within the
// 5e-2 tolerance. Gated on oracle presence.
//
// NOTE: facebook/opt-125m's `main` revision ships only pytorch_model.bin (no safetensors); the F16
// safetensors lives on a separate commit. We pin that revision so the safetensors-only source backend
// resolves it (offline-safe from the HF cache).
use std::path::Path;
use std::process::Command;

const OPT_ST_REV: &str = "1f9886ce095904096e22b0f4d9e7ba932fa7df2a";

#[test]
fn opt_125m_arena_matches_python_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    let refs = root.join("artifacts/opt-125m");
    if !refs.join("embed_tokens.npy").exists() {
        eprintln!("SKIP opt: oracle missing - run .venv/bin/python scripts/convert_opt125m.py");
        return;
    }
    let arena = root.join("target/test-arenas/opt-125m.safetensors");
    let bin = env!("CARGO_BIN_EXE_npu-weights");
    let st = Command::new(bin)
        .current_dir(root)
        .args(["bake", "--source", &format!("hf:facebook/opt-125m@{OPT_ST_REV}"), "--arch", "opt",
               "--arena", arena.to_str().unwrap(), "--force"])
        .status().unwrap();
    assert!(st.success(), "bake failed for opt-125m");
    let out = Command::new(bin)
        .current_dir(root)
        .args(["verify", "--arena", arena.to_str().unwrap(), "--arch", "opt",
               "--refs", refs.to_str().unwrap()])
        .output().unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "verify failed for opt:\n{s}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("PARITY PASS"), "no parity pass for opt:\n{s}");
}
