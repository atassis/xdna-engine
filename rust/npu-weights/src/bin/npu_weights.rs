// rust/npu-weights/src/bin/npu_weights.rs
use anyhow::Result;
use clap::{Parser, Subcommand};
use npu_weights::{arena, spec::ModelSpec, spec::Source};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "npu-weights")]
struct Cli { #[command(subcommand)] cmd: Cmd }

#[derive(Subcommand)]
enum Cmd {
    /// Bake source weights into a BF16 arena (skips if fresh, unless --force).
    Bake { #[arg(long)] source: String, #[arg(long)] arch: String,
           #[arg(long)] arena: Option<PathBuf>, #[arg(long)] force: bool },
    /// mmap-load an arena and print tensor stats.
    Load { #[arg(long)] arena: PathBuf, #[arg(long)] arch: String },
    /// Verify arena tensors match a directory of reference .npy within tolerance.
    Verify { #[arg(long)] arena: PathBuf, #[arg(long)] arch: String, #[arg(long)] refs: PathBuf },
}

fn repo_root() -> PathBuf { std::env::current_dir().unwrap() }

fn ensure_arena(source: &str, arch_name: &str, arena: Option<PathBuf>, force: bool) -> Result<PathBuf> {
    // Single source of truth: the library helper resolves + bakes (the engine calls the same path).
    let spec = ModelSpec { source: Source::parse(source)?, arch: arch_name.into(), arena };
    let path = spec.ensure_arena(&repo_root(), force)?;
    println!("arena ready: {}", path.display());
    Ok(path)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Bake { source, arch, arena, force } => { ensure_arena(&source, &arch, arena, force)?; }
        Cmd::Load { arena, arch } => {
            let l = arena::load(&arena, &arch)?;
            println!("arch={} version={} tensors={}", l.arch, l.meta_version, l.names.len());
            for n in l.names.iter().take(5) { let (sh, _) = l.tensor_f32(n)?; println!("  {n} {sh:?}"); }
        }
        Cmd::Verify { arena, arch, refs } => {
            let l = arena::load(&arena, &arch)?;
            let (n, max) = verify_against_npy(&l, &refs)?;
            println!("verified {n} tensors; max abs rel-err {max:.4e}");
            anyhow::ensure!(max < 5e-2, "parity FAILED: max rel-err {max:.4e} >= 5e-2");
            println!("PARITY PASS");
        }
    }
    Ok(())
}

/// Compare each arena tensor to refs/<name>.npy (name '/'->path). Returns (count, max rel-err).
fn verify_against_npy(l: &arena::Loaded, refs: &std::path::Path) -> Result<(usize, f32)> {
    use ndarray_npy::read_npy;
    use ndarray::ArrayD;
    let mut max = 0f32; let mut n = 0usize;
    let mut worst = String::new();
    for name in &l.names {
        let p = refs.join(format!("{name}.npy"));
        if !p.exists() { continue; }              // refs may be a subset
        let r: ArrayD<f32> = read_npy(&p)?;
        let (_sh, got) = l.tensor_f32(name)?;
        let exp: Vec<f32> = r.iter().cloned().collect();
        anyhow::ensure!(exp.len() == got.len(), "len mismatch for {name}: {} vs {}", exp.len(), got.len());
        for (a, b) in got.iter().zip(exp.iter()) {
            let denom = b.abs().max(1e-3);
            let e = (a - b).abs() / denom;
            if e > max { max = e; worst = name.clone(); }
        }
        n += 1;
    }
    if !worst.is_empty() { eprintln!("worst tensor: {worst} (rel-err {max:.4e})"); }
    Ok((n, max))
}
