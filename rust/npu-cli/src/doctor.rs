//! `npu doctor` -- read-only self-test: what xrt-smi reports about the device, whether the power
//! mode is pinned, who currently holds a hardware context, which config file is in effect and why,
//! whether the configured models' artifacts resolve on disk, and whether a service is running.
//!
//! Everything here is read-only. It shells out to `xrt-smi examine` (verified unprivileged, takes
//! no hardware context) and reads files; it never opens `/dev/accel`, never dispatches, never
//! starts the engine. Scope: reporting only -- no device-side determinism gate. That is a followup,
//! and it would make this command take a hardware context, which defeats the point of a command an
//! operator runs to see who else is using the device.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::Value;

use npu_runtime::config::{Config, ModelCfg};

/// How long a single `xrt-smi` invocation gets before doctor gives up on it. Generous for a
/// read-only report (`examine` is near-instant when the device answers at all), tight enough that
/// a wedged driver cannot hang the command an operator reaches for BECAUSE something looks wedged.
const XRT_SMI_TIMEOUT: Duration = Duration::from_secs(10);

/// Mirrors `scripts/npu_power_mode.py`'s `PINNED_MODES`. Kept as a second copy rather than a shared
/// constant because one side is Rust and the other Python; if the driver's mode names change, both
/// need editing and this comment is the pointer to the other one.
const PINNED_POWER_MODES: &[&str] = &["powersaver", "balanced", "performance", "turbo"];

/// `scripts/npu_power_mode.py`'s `SET_CMD`/`QUIESCE_CMD`/`RECOMMENDED`, verbatim: doctor prints the
/// fix, it never runs it. Setting the mode needs `CAP_SYS_ADMIN` (`AMDXDNA_SET_STATE` is
/// `DRM_ROOT_ONLY`); no udev rule grants that, so the only honest move here is naming the root
/// command, same as the script already does.
const QUIESCE_CMD: &str = "systemctl --user stop xdna-engine.service npu-vox.service";
const PIN_CMD: &str = "sudo xrt-smi configure --pmode turbo --force";
const RESTORE_CMD: &str = "sudo xrt-smi configure --pmode default --force";

fn power_mode_is_pinned(mode: &str) -> bool {
    PINNED_POWER_MODES.contains(&mode.to_lowercase().as_str())
}

// --- xrt-smi subprocess, with a timeout std::process::Command does not give you for free ---

/// Run `program` with `args`, killing it if it outlives `timeout` rather than blocking forever the
/// way `Command::output()` would. Reads stdout/stderr only after the child has exited, so this is
/// only correct for commands with small output (well under the ~64KB pipe buffer) -- true of every
/// `xrt-smi examine` report, not true in general.
fn run_with_timeout(program: &str, args: &[String], timeout: Duration) -> Option<Output> {
    let mut child = Command::new(program).args(args)
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().ok()?;
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            use std::io::Read;
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            if let Some(mut o) = child.stdout.take() { let _ = o.read_to_end(&mut stdout); }
            if let Some(mut e) = child.stderr.take() { let _ = e.read_to_end(&mut stderr); }
            return Some(Output { status, stdout, stderr });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Run one `xrt-smi examine [-r <report>] -f json -o <tmp>` and return the parsed JSON. `-o` is
/// mandatory for JSON output (xrt-smi refuses to write it to stdout), so this writes to a
/// per-process temp file and reads it back -- the same shape `scripts/npu_power_mode.py` already
/// uses for the platform report.
fn xrt_smi_json(report: Option<&str>, tag: &str) -> Result<Value, String> {
    let tmp = std::env::temp_dir().join(format!("npu-doctor-{}-{tag}.json", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let mut args: Vec<String> = vec!["examine".into()];
    if let Some(r) = report { args.push("-r".into()); args.push(r.into()); }
    args.push("-f".into()); args.push("json".into());
    args.push("-o".into()); args.push(tmp.to_string_lossy().into_owned());
    args.push("--force".into());
    let out = run_with_timeout("xrt-smi", &args, XRT_SMI_TIMEOUT)
        .ok_or_else(|| "xrt-smi timed out or is not on PATH".to_string())?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("xrt-smi exited with {}: {}", out.status,
            String::from_utf8_lossy(&out.stderr).trim()));
    }
    let body = std::fs::read_to_string(&tmp)
        .map_err(|e| format!("xrt-smi reported success but wrote no JSON: {e}"))?;
    let _ = std::fs::remove_file(&tmp);
    serde_json::from_str(&body).map_err(|e| format!("parsing xrt-smi JSON: {e}"))
}

// --- parsing (pure, tested against captured xrt-smi output) ---

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExamineInfo {
    pub xrt_version: Option<String>,
    pub xrt_branch: Option<String>,
    pub amdxdna_version: Option<String>,
    pub firmware_version: Option<String>,
    pub device_name: Option<String>,
    pub device_bdf: Option<String>,
}

/// `xrt-smi examine -f json` (no `-r`, the default/summary report).
fn parse_examine(v: &Value) -> ExamineInfo {
    let host = v.get("system").and_then(|s| s.get("host"));
    let xrt = host.and_then(|h| h.get("xrt"));
    let str_at = |val: Option<&Value>| val.and_then(Value::as_str).map(str::to_string);
    let amdxdna_version = xrt.and_then(|x| x.get("drivers")).and_then(Value::as_array)
        .and_then(|drivers| drivers.iter()
            .find(|d| d.get("name").and_then(Value::as_str) == Some("amdxdna")))
        .and_then(|d| d.get("version")).and_then(Value::as_str).map(str::to_string);
    let dev = host.and_then(|h| h.get("devices")).and_then(Value::as_array).and_then(|a| a.first());
    ExamineInfo {
        xrt_version: str_at(xrt.and_then(|x| x.get("version"))),
        xrt_branch: str_at(xrt.and_then(|x| x.get("branch"))),
        amdxdna_version,
        firmware_version: str_at(dev.and_then(|d| d.get("firmware_version"))),
        device_name: str_at(dev.and_then(|d| d.get("name"))),
        device_bdf: str_at(dev.and_then(|d| d.get("bdf"))),
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlatformInfo {
    pub power_mode: Option<String>,
    pub total_columns: Option<String>,
}

/// `xrt-smi examine -r platform -f json`.
fn parse_platform(v: &Value) -> Option<PlatformInfo> {
    let plat = v.get("devices")?.as_array()?.first()?.get("platforms")?.as_array()?.first()?;
    Some(PlatformInfo {
        power_mode: plat.get("status").and_then(|s| s.get("power_mode"))
            .and_then(Value::as_str).map(|s| s.trim().to_string()),
        total_columns: plat.get("static_region").and_then(|s| s.get("total_columns"))
            .and_then(Value::as_str).map(|s| s.trim().to_string()),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct HwContext {
    pub pid: String,
    /// From `/proc/<pid>/comm`, not xrt-smi's own `process_name` field -- observed "N/A" on this
    /// driver/xrt-smi pairing (2.26.0/2.21.75) for every context, so the field the report actually
    /// carries is useless for "who is holding the device" and the question needs answering anyway.
    pub process: Option<String>,
    pub context_id: String,
    pub status: String,
    pub instr_bo: String,
}

/// `xrt-smi examine -r aie-partitions -f json`. `partitions` is `""` (a bare empty string, not `[]`)
/// when nothing holds the device -- observed directly, not documented -- so this reads it via
/// `.as_array()` and treats anything that is not an array as "no contexts" rather than erroring.
fn parse_hw_contexts(v: &Value) -> Vec<HwContext> {
    let mut out = Vec::new();
    let Some(devices) = v.get("devices").and_then(Value::as_array) else { return out };
    for dev in devices {
        let Some(partitions) = dev.get("aie_partitions").and_then(|p| p.get("partitions"))
            .and_then(Value::as_array) else { continue };
        for part in partitions {
            let Some(ctxs) = part.get("hw_contexts").and_then(Value::as_array) else { continue };
            for c in ctxs {
                let s = |k: &str| c.get(k).and_then(Value::as_str).unwrap_or("?").to_string();
                let pid = s("pid");
                let process = proc_comm(&pid).or_else(|| {
                    c.get("process_name").and_then(Value::as_str)
                        .filter(|p| *p != "N/A").map(str::to_string)
                });
                out.push(HwContext {
                    process, pid, context_id: s("context_id"), status: s("status"),
                    instr_bo: s("instr_bo_mem"),
                });
            }
        }
    }
    out
}

/// `/proc/<pid>/comm`, best-effort. `None` on any failure (pid gone, not our /proc, unparseable) --
/// the caller falls back to whatever xrt-smi itself reported.
fn proc_comm(pid: &str) -> Option<String> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}

// --- config + artifact resolution ---

pub struct ModelArtifactCheck {
    pub name: String,
    pub scenario_path: String,
    pub ok: bool,
    pub detail: String,
}

/// Whether `m`'s scenario file parses and, if it declares a legacy `artifacts.weights` dir, whether
/// that dir exists and is non-empty. This is NOT install.sh's §4c bash chain (that logic is inline
/// shell with no Rust entry point to call -- see the commit body) -- it reads the same two facts
/// through `ScenarioConfig`, the typed struct the engine itself loads scenarios with, rather than
/// re-deriving them with a second `grep -oP` parser that could drift from the first.
fn scenario_artifact_check(m: &ModelCfg, root: &Path) -> ModelArtifactCheck {
    let p = Path::new(&m.scenario);
    let scenario_path = if p.is_absolute() { p.to_path_buf() } else { root.join(p) };
    let display = scenario_path.display().to_string();
    match npu_engine::config::ScenarioConfig::load(&scenario_path) {
        Err(e) => ModelArtifactCheck {
            name: m.name.clone(), scenario_path: display, ok: false,
            detail: format!("scenario error: {e}"),
        },
        Ok(sc) if sc.artifacts.weights.is_empty() => ModelArtifactCheck {
            name: m.name.clone(), scenario_path: display, ok: true,
            detail: "scenario OK (no legacy weights dir declared)".into(),
        },
        Ok(sc) => {
            let w = Path::new(&sc.artifacts.weights);
            let wp = if w.is_absolute() { w.to_path_buf() } else { root.join(w) };
            let has_content = std::fs::read_dir(&wp).map(|mut d| d.next().is_some()).unwrap_or(false);
            ModelArtifactCheck {
                name: m.name.clone(), scenario_path: display, ok: has_content,
                detail: if has_content { format!("weights OK: {}", wp.display()) }
                        else { format!("weights missing/empty: {}", wp.display()) },
            }
        }
    }
}

// --- report assembly ---

struct DoctorReport {
    device_node_present: bool,
    examine: Result<ExamineInfo, String>,
    platform: Result<Option<PlatformInfo>, String>,
    hw_contexts: Result<Vec<HwContext>, String>,
    config_path: PathBuf,
    config_source: &'static str,
    config_exists: bool,
    config_error: Option<String>,
    engine_root: Option<PathBuf>,
    model_checks: Vec<ModelArtifactCheck>,
    /// The one artifact check that already had a device-free Rust code path before this command
    /// existed: `serve()` calls this same function before `start()`. Reused verbatim, not
    /// re-derived -- it is scoped to the parakeet resident xclbin build only (see its own doc
    /// comment), which is why `model_checks` above exists alongside it rather than instead of it.
    /// `None` when the engine root could not be resolved at all.
    parakeet_preflight: Option<Result<()>>,
    service: Option<(u64, Value)>,
}

impl DoctorReport {
    fn gather(cli: &crate::cli_def::Cli) -> Self {
        let (config_path, config_source) = crate::config_path_and_source(cli);
        let config_exists = config_path.exists();
        let (cfg, config_error) = match Config::load(&config_path) {
            Ok(c) => (c, None),
            Err(e) => (Config::default(), Some(e)),
        };
        let engine_root = crate::root(&cfg, &config_path).ok();

        let examine = xrt_smi_json(None, "examine").map(|v| parse_examine(&v));
        let platform = xrt_smi_json(Some("platform"), "platform").map(|v| parse_platform(&v));
        let hw_contexts = xrt_smi_json(Some("aie-partitions"), "aie-partitions")
            .map(|v| parse_hw_contexts(&v));
        let device_node_present = npu_engine::Engine::available();

        let model_checks: Vec<ModelArtifactCheck> = match &engine_root {
            Some(r) => cfg.models.iter().map(|m| scenario_artifact_check(m, r)).collect(),
            None => vec![],
        };
        let parakeet_preflight = engine_root.as_ref()
            .map(|r| crate::preflight_artifacts(&cfg, r));
        let service = crate::read_live_status(cfg.server.port);

        DoctorReport {
            device_node_present, examine, platform, hw_contexts,
            config_path, config_source, config_exists, config_error, engine_root,
            model_checks, parakeet_preflight, service,
        }
    }

    fn print_table(&self) {
        println!("== Device ==");
        println!("device node (/dev/accel/accel0)  {}",
            if self.device_node_present { "present" } else { "ABSENT" });
        match &self.examine {
            Ok(e) => {
                println!("XRT               {} (branch {})",
                    e.xrt_version.as_deref().unwrap_or("?"), e.xrt_branch.as_deref().unwrap_or("?"));
                println!("amdxdna driver    {}", e.amdxdna_version.as_deref().unwrap_or("?"));
                println!("NPU firmware      {}", e.firmware_version.as_deref().unwrap_or("?"));
                println!("device            {} ({})",
                    e.device_name.as_deref().unwrap_or("?"), e.device_bdf.as_deref().unwrap_or("?"));
            }
            Err(e) => println!("xrt-smi examine: unavailable ({e})"),
        }
        match &self.platform {
            Ok(Some(p)) => println!("total columns     {}", p.total_columns.as_deref().unwrap_or("?")),
            Ok(None) => println!("total columns     ? (unexpected xrt-smi platform JSON shape)"),
            Err(e) => println!("total columns     unavailable ({e})"),
        }

        println!("\n== Power mode ==");
        match &self.platform {
            Ok(Some(p)) => {
                let mode = p.power_mode.as_deref().unwrap_or("?");
                if power_mode_is_pinned(mode) {
                    println!("mode              {mode} (pinned)");
                } else {
                    println!("mode              {mode} (UNPINNED -- a timed %-of-peak figure taken now");
                    println!("                  is not comparable across runs; the DPM ramp aliases");
                    println!("                  with whatever you are measuring)");
                    println!("  pin (root):     {QUIESCE_CMD} && {PIN_CMD}");
                    println!("  restore (root): {RESTORE_CMD}");
                }
            }
            Ok(None) => println!("mode              ? (unexpected xrt-smi platform JSON shape)"),
            Err(e) => println!("mode              unavailable ({e})"),
        }

        println!("\n== Hardware contexts (who is holding the device) ==");
        match &self.hw_contexts {
            Ok(ctxs) if ctxs.is_empty() => println!("(none -- device is free)"),
            Ok(ctxs) => {
                println!("{:<10} {:<16} {:<8} {:<8} {}", "PID", "PROCESS", "CTX ID", "STATUS", "INSTR BO");
                for c in ctxs {
                    println!("{:<10} {:<16} {:<8} {:<8} {}",
                        c.pid, c.process.as_deref().unwrap_or("?"), c.context_id, c.status, c.instr_bo);
                }
            }
            Err(e) => println!("unavailable ({e})"),
        }

        println!("\n== Config ==");
        println!("in effect         {}", self.config_path.display());
        println!("selected by       {}", self.config_source);
        println!("file exists       {}",
            if self.config_exists { "yes" } else { "NO -- running on compiled-in defaults" });
        if let Some(e) = &self.config_error { println!("PARSE ERROR       {e}"); }
        match &self.engine_root {
            Some(r) => println!("engine root       {}", r.display()),
            None => println!("engine root       could not be resolved"),
        }

        println!("\n== Model artifacts ==");
        if self.model_checks.is_empty() {
            println!("(no models configured, or engine root could not be resolved)");
        } else {
            for c in &self.model_checks {
                println!("{:<22} {:<8} {}  [{}]",
                    c.name, if c.ok { "OK" } else { "FAIL" }, c.detail, c.scenario_path);
            }
        }
        match &self.parakeet_preflight {
            Some(Ok(())) => println!("resident xclbin (parakeet, NPU-side)  OK"),
            Some(Err(e)) => println!("resident xclbin (parakeet, NPU-side)  FAIL: {e}"),
            None => {}
        }

        println!("\n== Service ==");
        match &self.service {
            Some((age, v)) => {
                let pid = v.get("pid").and_then(Value::as_u64).map(|p| p.to_string()).unwrap_or_default();
                let port = v.get("port").and_then(Value::as_u64).map(|p| p.to_string())
                    .unwrap_or_else(|| "?".into());
                println!("running           yes (pid {pid}, status published {age}s ago, port {port})");
            }
            None => println!("running           no"),
        }
    }

    fn to_json(&self) -> Value {
        let examine_j = match &self.examine {
            Ok(e) => serde_json::json!({
                "xrt_version": e.xrt_version, "xrt_branch": e.xrt_branch,
                "amdxdna_version": e.amdxdna_version, "firmware_version": e.firmware_version,
                "device_name": e.device_name, "device_bdf": e.device_bdf,
            }),
            Err(e) => serde_json::json!({"error": e}),
        };
        let (power_mode, power_pinned, total_columns) = match &self.platform {
            Ok(Some(p)) => (p.power_mode.clone(), p.power_mode.as_deref().map(power_mode_is_pinned),
                            p.total_columns.clone()),
            _ => (None, None, None),
        };
        let hw_j: Value = match &self.hw_contexts {
            Ok(ctxs) => serde_json::json!(ctxs.iter().map(|c| serde_json::json!({
                "pid": c.pid, "process": c.process, "context_id": c.context_id,
                "status": c.status, "instr_bo": c.instr_bo,
            })).collect::<Vec<_>>()),
            Err(e) => serde_json::json!({"error": e}),
        };
        let models_j: Vec<Value> = self.model_checks.iter().map(|c| serde_json::json!({
            "name": c.name, "scenario": c.scenario_path, "ok": c.ok, "detail": c.detail,
        })).collect();
        let parakeet_j = self.parakeet_preflight.as_ref().map(|r| match r {
            Ok(()) => serde_json::json!({"ok": true}),
            Err(e) => serde_json::json!({"ok": false, "detail": e.to_string()}),
        });
        let service_j = match &self.service {
            Some((age, v)) => serde_json::json!({
                "running": true, "pid": v.get("pid"), "port": v.get("port"), "age_s": age,
            }),
            None => serde_json::json!({"running": false}),
        };
        serde_json::json!({
            "device_node_present": self.device_node_present,
            "examine": examine_j,
            "power_mode": power_mode, "power_pinned": power_pinned, "total_columns": total_columns,
            "hw_contexts": hw_j,
            "config": {
                "path": self.config_path.display().to_string(), "selected_by": self.config_source,
                "exists": self.config_exists, "parse_error": self.config_error,
                "engine_root": self.engine_root.as_ref().map(|r| r.display().to_string()),
            },
            "models": models_j,
            "parakeet_resident_preflight": parakeet_j,
            "service": service_j,
        })
    }
}

pub fn doctor(cli: &crate::cli_def::Cli, as_json: bool) -> Result<()> {
    let report = DoctorReport::gather(cli);
    if as_json { println!("{}", report.to_json()); } else { report.print_table(); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Value {
        let s = std::fs::read_to_string(format!("tests/fixtures/{name}")).expect(name);
        serde_json::from_str(&s).expect(name)
    }

    #[test]
    fn parses_examine_report() {
        let e = parse_examine(&fixture("examine.json"));
        assert_eq!(e.xrt_version.as_deref(), Some("2.21.75"));
        assert_eq!(e.xrt_branch.as_deref(), Some("makepkg"));
        assert_eq!(e.amdxdna_version.as_deref(), Some("2.26.0_20260901"));
        assert_eq!(e.firmware_version.as_deref(), Some("1.1.2.64"));
        assert_eq!(e.device_name.as_deref(), Some("NPU Gorgon Point 1"));
        assert_eq!(e.device_bdf.as_deref(), Some("0000:c5:00.1"));
    }

    #[test]
    fn parses_platform_report() {
        let p = parse_platform(&fixture("platform.json")).expect("platform present");
        assert_eq!(p.power_mode.as_deref(), Some("Default"));
        assert_eq!(p.total_columns.as_deref(), Some("8"));
    }

    #[test]
    fn default_power_mode_is_unpinned() {
        assert!(!power_mode_is_pinned("Default"));
        assert!(!power_mode_is_pinned("default"));
    }

    #[test]
    fn every_documented_pinned_mode_is_recognized_case_insensitively() {
        for m in ["powersaver", "balanced", "performance", "turbo", "TURBO", "Performance"] {
            assert!(power_mode_is_pinned(m), "{m} should be pinned");
        }
    }

    #[test]
    fn parses_hw_contexts_when_something_holds_the_device() {
        let ctxs = parse_hw_contexts(&fixture("aie-partitions.json"));
        assert_eq!(ctxs.len(), 4);
        assert_eq!(ctxs[0].pid, "144071");
        assert_eq!(ctxs[0].context_id, "95");
        assert_eq!(ctxs[0].status, "Idle");
        assert_eq!(ctxs[0].instr_bo, "6696 KB");
        for c in &ctxs { assert_eq!(c.pid, "144071"); }
    }

    /// xrt-smi reports `"partitions": ""` (a bare string, not `[]`) when nothing holds the device --
    /// captured directly off this box with the service stopped. A parser that assumed an array
    /// unconditionally would panic on `.as_array().unwrap()` on exactly the healthy, idle case.
    #[test]
    fn empty_partitions_string_parses_as_no_contexts_not_an_error() {
        let ctxs = parse_hw_contexts(&fixture("aie-partitions-empty.json"));
        assert!(ctxs.is_empty());
    }

    #[test]
    fn hw_contexts_tolerates_an_unexpected_top_level_shape() {
        assert!(parse_hw_contexts(&serde_json::json!({"unexpected": true})).is_empty());
        assert!(parse_hw_contexts(&serde_json::json!(null)).is_empty());
    }

    #[test]
    fn platform_tolerates_an_unexpected_top_level_shape() {
        assert!(parse_platform(&serde_json::json!({"devices": []})).is_none());
        assert!(parse_platform(&serde_json::json!(null)).is_none());
    }

    #[test]
    fn examine_tolerates_a_missing_system_block() {
        let e = parse_examine(&serde_json::json!({}));
        assert_eq!(e, ExamineInfo::default());
    }
}
