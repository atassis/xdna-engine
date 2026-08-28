//! DEPRECATED shim. The weight tooling now lives in the single engine entrypoint as
//! `npu weights <bake|load|verify>`.
//!
//! Kept rather than deleted so existing scripts and muscle memory fail LOUDLY with a pointer
//! instead of "command not found", and so a stale binary left in ~/.local/bin cannot go on quietly
//! doing the old thing. It forwards every argument, so nothing that worked stops working.
//!
//! Why it moved: `npu` is documented as the single entrypoint, `npu bake` already overlapped
//! `npu-weights bake`, this binary resolved its repo root from the current working directory (the
//! dependency the service install just removed), and a second binary is a second shell-completion
//! surface with none of the coverage the main one is tested for.
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() -> ! {
    let args: Vec<String> = std::env::args().skip(1).collect();
    eprintln!("npu-weights is DEPRECATED; use: npu weights {}", args.join(" "));
    // exec rather than spawn: the replacement inherits the pid, so exit codes and signals reach the
    // caller unchanged and a script wrapping this sees no difference.
    let err = Command::new("npu").arg("weights").args(&args).exec();
    eprintln!("could not exec `npu`: {err}. Is it on PATH? (install.sh puts it in ~/.local/bin)");
    std::process::exit(127);
}
