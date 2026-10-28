//! Fixed capture protocol shared with retained resident-forward baseline controls.
//! Backend recovery remains visible in stderr; recovered captures alone are not acceptance.
use std::{fs, path::PathBuf, rc::Rc, time::Instant};
use npu_models::llm::{CacheState, DecodeStep, LadderResidentForward};
use npu_xrt::Device;

const USAGE: &str = "usage: npu-dev resident-repeatability <model-root> <new-output-directory>\nRequires an authorized, exclusive NPU window and NPU_RESIDENT_REUSE_KV=0.";

fn inputs(args: &[String]) -> Result<(PathBuf, PathBuf), &'static str> {
    if args.len() != 3 {
        return Err(USAGE);
    }
    Ok((PathBuf::from(&args[1]), PathBuf::from(&args[2])))
}

pub fn run(args: Vec<String>) {
    if args.len() == 2 && matches!(args[1].as_str(), "-h" | "--help") {
        eprintln!("{USAGE}");
        return;
    }
    let (model_root, output) = inputs(&args).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    assert!(model_root.join("meta.json").is_file(), "model-root has no meta.json");
    fs::create_dir(&output).expect("output directory must be new");
    let device = Rc::new(Device::open(0).unwrap());
    let mut model = LadderResidentForward::open(&device, &model_root).unwrap();
    let ids: Vec<u32> = (0..128).map(|i| 1000 + (i * 7919) % 200_000).collect();
    for trial in 0..3 {
        assert!(matches!(model.reset().unwrap(), CacheState::Cleared),
                "set NPU_RESIDENT_REUSE_KV=0 for this gate");
        let at = model.prefill(&ids, 0).unwrap();
        println!("TRIAL {trial} prefill_at {at}");
        for pos in at..at + 24 {
            let t = Instant::now();
            let values = model.step(ids[pos % ids.len()], pos).unwrap();
            let elapsed_ms = t.elapsed().as_secs_f64() * 1e3;
            assert!(values.iter().all(|v| v.is_finite()));
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            fs::write(output.join(format!("t{trial}-s{pos}.f32")), bytes).unwrap();
            println!("MATCH trial {trial} pos {pos} logits {} dispatch_wall_ms {elapsed_ms:.3}", values.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_arguments_are_rejected_without_a_device() {
        assert!(inputs(&["probe".into(), "model".into()]).is_err());
    }

    #[test]
    fn capture_paths_are_explicit() {
        let (model, output) = inputs(&["probe".into(), "model".into(), "captures".into()]).unwrap();
        assert_eq!(model, PathBuf::from("model"));
        assert_eq!(output, PathBuf::from("captures"));
    }
}
