//! Smoke test: run the GigaAM-v3 RNNT decoder.onnx via the onnxruntime C-shim. Proves onnxruntime
//! works from Rust (the whole point of option B). Run from repo root.
use npu_onnx::{Env, Session, Tensor};

fn main() {
    let env = Env::new().expect("env");
    let dec = Session::load(&env, "artifacts/asr/decoder.onnx").expect("load decoder");
    // x=[[blank=33]] i64; h,c zeros [1,1,320] f32
    let x = vec![33i64];
    let h = vec![0f32; 320];
    let c = vec![0f32; 320];
    let out = dec
        .run(
            &[
                ("x", Tensor::I64(&x, vec![1, 1])),
                ("h.1", Tensor::F32(&h, vec![1, 1, 320])),
                ("c.1", Tensor::F32(&c, vec![1, 1, 320])),
            ],
            &["dec", "h", "c"],
        )
        .expect("run decoder");
    println!("decoder ran: {} outputs", out.len());
    for (i, name) in ["dec", "h", "c"].iter().enumerate() {
        let s = out.shape(i);
        let d = out.f32(i);
        println!("  {name}: shape={s:?}  first4={:?}", &d[..4.min(d.len())]);
    }
    println!("ONNX-from-Rust (onnxruntime C-shim): OK");
}
