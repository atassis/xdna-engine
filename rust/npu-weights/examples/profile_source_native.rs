//! Host-only read/hash/transform probes. Run optimized; these do not allocate NPU storage.
use anyhow::{ensure, Context, Result};
use npu_weights::source_native::{
    plan::{WeightContract, WeightPlan},
    safetensors::SafetensorsFile,
    Limits,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, hint::black_box, io::Read, path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 4 || args.len() == 5,
        "usage: profile_source_native <read|source-read|hash|plan|prepared-read> <checkpoint-dir> <meta.json> [scratch-bytes]");
    let mode = &args[1];
    ensure!(
        ["read", "source-read", "hash", "plan", "prepared-read"].contains(&mode.as_str()),
        "unknown mode"
    );
    let size = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(65536usize);
    ensure!(
        (2..=16 * 1024 * 1024).contains(&size),
        "scratch must be 2 bytes..16 MiB"
    );
    let root = Path::new(&args[2]);
    let started = Instant::now();
    let metadata_bytes = std::fs::read(&args[3])?;
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)?;
    let contract =
        WeightContract::from_metadata_json(&metadata_bytes)?.context("missing weight_inputs")?;
    let config = std::fs::read(root.join("config.json"))?;
    let metadata_ms = started.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let source = SafetensorsFile::open(root.join("model.safetensors"), Limits::default())?;
    let header_ms = start.elapsed().as_secs_f64() * 1000.0;
    let expected: BTreeMap<String, usize> = metadata["weights"]
        .as_array()
        .context("weights array")?
        .iter()
        .map(|name| {
            let name = name.as_str().context("weight name")?;
            let mut components = Path::new(name).components();
            ensure!(
                matches!(components.next(), Some(std::path::Component::Normal(_)))
                    && components.next().is_none(),
                "weight name must be one path component"
            );
            let len = metadata["layout"][name]["len"]
                .as_u64()
                .context("weight extent")?;
            Ok((name.to_string(), usize::try_from(len)?))
        })
        .collect::<Result<_>>()?;
    let start = Instant::now();
    let plan = WeightPlan::bind_config_json(&contract, source.tensors(), &expected, &config)?;
    let bind_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut scratch = vec![0; size];
    let mut bytes = 0u64;
    let mut calls = 0u64;
    let mut digest = None;
    let start = Instant::now();
    match mode.as_str() {
        "plan" => {
            for name in metadata["weights"].as_array().context("weights array")? {
                plan.write_buffer(
                    name.as_str().context("weight name")?,
                    &mut scratch,
                    |_, chunk| {
                        black_box(chunk);
                        bytes += chunk.len() as u64;
                        calls += 1;
                        Ok(())
                    },
                )?;
            }
        }
        "prepared-read" => {
            let buffers = Path::new(&args[3])
                .parent()
                .context("metadata parent")?
                .join("buffers");
            for name in metadata["weights"].as_array().context("weights array")? {
                let name = name.as_str().context("weight name")?;
                let mut file = File::open(buffers.join(format!("{name}.bin")))?;
                ensure!(
                    file.metadata()?.len() == expected[name] as u64,
                    "prepared extent mismatch: {name}"
                );
                loop {
                    let n = file.read(&mut scratch)?;
                    if n == 0 {
                        break;
                    }
                    black_box(&scratch[..n]);
                    bytes += n as u64;
                    calls += 1;
                }
            }
        }
        "read" => {
            let mut file = File::open(root.join("model.safetensors"))?;
            loop {
                let n = file.read(&mut scratch)?;
                if n == 0 {
                    break;
                }
                black_box(&scratch[..n]);
                bytes += n as u64;
                calls += 1;
            }
        }
        "source-read" | "hash" => {
            let file = plan.embedding().span.source();
            let mut hash = Sha256::new();
            while bytes < file.len() {
                let n = (file.len() - bytes).min(size as u64) as usize;
                file.read_exact_at(bytes, &mut scratch[..n])?;
                if mode == "hash" {
                    hash.update(&scratch[..n]);
                } else {
                    black_box(&scratch[..n]);
                }
                bytes += n as u64;
                calls += 1;
            }
            if mode == "hash" {
                digest = Some(format!("{:x}", hash.finalize()));
            }
        }
        _ => unreachable!(),
    }
    let payload_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{}",
        serde_json::json!({"mode":mode,"scratch_bytes":size,"bytes":bytes,"calls":calls,
        "metadata_ms":metadata_ms,"source_header_ms":header_ms,"bind_ms":bind_ms,
        "payload_ms":payload_ms,"total_ms":started.elapsed().as_secs_f64()*1000.0,"source_sha256":digest})
    );
    Ok(())
}
