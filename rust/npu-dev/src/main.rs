//! Dispatcher for the npu-dev subcommands: device probes, parity gates and kernel build tools.
//! `SUBCOMMANDS` drives the help table; `lookup` is the plain match that resolves a name to its
//! `run` function without calling it, so the two can be checked against each other in tests.

mod cmd;

const SUBCOMMANDS: &[(&str, &str)] = &[
    ("kernels-build", "Build every declared kernel reported Missing, then publish and re-verify."),
    ("kernels-manifest", "Regenerate kernel_manifest.json for one or more artifact directories."),
    ("kernels-verify", "Check declared_kernels.json against what actually exists under a kernels root."),
    ("s2-chain", "Device probe for the full S2 decoder-chain driver loop (head/stage/tail/chain)."),
    ("s2-design", "Device probe for one exported S2 codec design (rel-L2 + determinism gate)."),
    ("verify-parakeet", "Verify the Rust Parakeet host reference encoder vs ONNX reference activations."),
    ("verify-whisper", "Gate the Rust Whisper-small host reference encoder vs ONNX golden activations."),
    ("verify-whisper-decode", "Parity test: host-f32 Whisper-small decoder vs the ONNX decoder graphs."),
    ("parakeet-encode", "Encode mel spectrograms through the Parakeet NPU encoder."),
    ("whisper-e2e", "End-to-end Whisper-small ASR latency bench with per-stage breakdown."),
    ("fused-elf", "Generic on-device probe for a fused full ELF via the shim + FusedArena."),
    ("prefill-token-gate", "Tier 2 gate for the batched prefill path: greedy-decode from a primed prompt."),
    ("prefill-time", "Time priming a prompt, batched vs one token at a time."),
    ("prefill-golden", "Localise a batched-prefill divergence to a layer against the CPU golden."),
    ("mha-decode", "Parity probe for the on-chip single-query MHA decode kernel."),
    ("conveyor-parity", "On-device parity for the 8-head relpos conveyor vs a host reference."),
    ("tcache-parity", "Arm-vs-arm token parity for the transposed self-V cache via the fused decoder."),
];

/// Plain match, no clap: resolves a subcommand name to its `run` function pointer without
/// calling it, so tests can check name/arm agreement without touching the NPU.
fn lookup(sub: &str) -> Option<fn(Vec<String>)> {
    Some(match sub {
        "kernels-build" => cmd::kernels_build::run,
        "kernels-manifest" => cmd::kernels_manifest::run,
        "kernels-verify" => cmd::kernels_verify::run,
        "s2-chain" => cmd::s2_chain::run,
        "s2-design" => cmd::s2_design::run,
        "verify-parakeet" => cmd::verify_parakeet::run,
        "verify-whisper" => cmd::verify_whisper::run,
        "verify-whisper-decode" => cmd::verify_whisper_decode::run,
        "parakeet-encode" => cmd::parakeet_encode::run,
        "whisper-e2e" => cmd::whisper_e2e::run,
        "fused-elf" => cmd::fused_elf::run,
        "prefill-token-gate" => cmd::prefill_token_gate::run,
        "prefill-time" => cmd::prefill_time::run,
        "prefill-golden" => cmd::prefill_golden::run,
        "mha-decode" => cmd::mha_decode::run,
        "conveyor-parity" => cmd::conveyor_parity::run,
        "tcache-parity" => cmd::tcache_parity::run,
        _ => return None,
    })
}

fn print_usage() {
    eprintln!("usage: npu-dev <subcommand> [args...]");
    eprintln!();
    eprintln!("subcommands:");
    let width = SUBCOMMANDS.iter().map(|(name, _)| name.len()).max().unwrap_or(0);
    for (name, desc) in SUBCOMMANDS {
        eprintln!("  {name:<width$}  {desc}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        None => {
            print_usage();
            std::process::exit(2);
        }
        Some("-h") | Some("--help") => {
            print_usage();
            std::process::exit(0);
        }
        Some(sub) => match lookup(sub) {
            Some(run) => {
                let argv = std::iter::once(format!("npu-dev {sub}")).chain(args[2..].iter().cloned()).collect();
                run(argv);
            }
            None => {
                eprintln!("npu-dev: unknown subcommand `{sub}`\n");
                print_usage();
                std::process::exit(2);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_table_and_match_arms_agree() {
        for (name, _) in SUBCOMMANDS {
            assert!(lookup(name).is_some(), "`{name}` is in SUBCOMMANDS but has no match arm in lookup()");
        }
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        assert!(lookup("not-a-real-subcommand").is_none());
    }
}
