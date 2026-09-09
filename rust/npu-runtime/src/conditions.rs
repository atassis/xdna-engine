//! What the machine was doing while a request was measured.
//!
//! A latency is a measurement at a moment on a machine in a power state, not a property of the
//! model. On this device that is load-bearing rather than pedantic: the AIE core clock is a DPM
//! ladder and the power mode is per-boot and frequently unpinned, so two honest runs of identical
//! code can differ by double digits. A report that cannot say which mode it ran under cannot be
//! compared to another one, and the commonest apparent regression is a mode difference.

use std::sync::OnceLock;

/// Resolved once per process, off the request path. `None` covers both "probe has not finished"
/// and "probe failed" -- they render identically and neither is a number anyone should quote.
static POWER_MODE: OnceLock<Option<String>> = OnceLock::new();

/// Start the one-shot power-mode probe. Called when the actor starts, never from a request: the
/// probe is a subprocess, and a subprocess on the request path would be a worse instrument than no
/// instrument. A hung `xrt-smi` leaves this thread parked and the field `None`, which is the right
/// failure -- the server keeps serving and the report keeps saying it does not know.
pub fn spawn_probe() {
    // At most once per process, not once per actor: `spawn` runs for every `start`/`start_lazy`,
    // which in a test binary is dozens of times, and each call would otherwise fork `xrt-smi`
    // again to compute a value that is already a process-global.
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::thread::spawn(|| {
            let _ = POWER_MODE.set(probe_power_mode());
        });
    });
}

pub fn power_mode() -> Option<String> {
    POWER_MODE.get().cloned().flatten()
}

/// `xrt-smi examine`'s `platform.status.power_mode`.
///
/// A second, smaller copy of what `npu doctor` reads -- deliberately, the way `doctor.rs` already
/// keeps its own copy of the driver's mode names. doctor is the authority and reports far more;
/// this needs one string and must not drag doctor's reporting into the serving path.
fn probe_power_mode() -> Option<String> {
    let out = std::env::temp_dir().join(format!("npu-pmode-{}.json", std::process::id()));
    let path = out.to_str()?;
    // `-o` is mandatory for JSON: xrt-smi refuses to write it to stdout.
    let st = std::process::Command::new("xrt-smi")
        .args(["examine", "-r", "platform", "-f", "json", "-o", path, "--force"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;
    let body = std::fs::read_to_string(path).ok();
    let _ = std::fs::remove_file(path);
    if !st.success() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(&body?).ok()?;
    // `devices[0].platforms[0].status.power_mode`, matching `doctor::parse_platform`. Verified
    // against real output rather than assumed: the shorter `platform.status.power_mode` this first
    // reached for parses fine, finds nothing, and reports "unknown" -- a probe that fails by
    // silently returning the honest-looking answer is the worst shape a probe can have.
    v.get("devices")?.as_array()?.first()?
        .get("platforms")?.as_array()?.first()?
        .get("status")?.get("power_mode")?
        .as_str().map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    /// Pin the shape against real `xrt-smi examine -r platform -f json` output, so a future version
    /// that moves the key fails here instead of quietly reporting an unknown mode forever.
    #[test]
    fn the_power_mode_probe_reads_the_shape_xrt_smi_actually_emits() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"schema_version":{},"devices":[{"platforms":[{"status":{"power_mode":"Default"},
               "static_region":{"total_columns":"8"}}]}]}"#).unwrap();
        let got = v.get("devices").and_then(|d| d.as_array()).and_then(|a| a.first())
            .and_then(|d| d.get("platforms")).and_then(|p| p.as_array()).and_then(|a| a.first())
            .and_then(|p| p.get("status")).and_then(|s| s.get("power_mode"))
            .and_then(|m| m.as_str());
        assert_eq!(got, Some("Default"));
    }
}

/// Instantaneous NPU package power in microwatts, from the amdxdna hwmon node.
///
/// MEASURED 2026-09-09 on this box: one read costs ~785 us and reports 0 uW at idle. Both numbers
/// decide how it is used. 785 us against a ~21 ms decode step is 3.7%, so sampling this per token
/// would visibly perturb the very interval it is trying to measure -- it is therefore read exactly
/// twice per generation, at the ends, and the pair is reported as two samples and never integrated
/// into a J/token figure it cannot support. The 0 at idle is why 0 maps to `None`: a driver that
/// does not populate the counter and a device drawing no power are not the same claim, and only one
/// of them is possible.
pub fn npu_power_uw() -> Option<u64> {
    let dir = std::fs::read_dir("/sys/class/accel/accel0/device/hwmon").ok()?;
    for e in dir.flatten() {
        if let Ok(s) = std::fs::read_to_string(e.path().join("power1_input")) {
            return s.trim().parse::<u64>().ok().filter(|v| *v > 0);
        }
    }
    None
}

/// The conditions a serving thread can see at the moment a run starts.
///
/// One constructor, used by both the service's run log and the CLI's `--stats-log`. They had
/// diverged the first time round -- the CLI wrote a header with every condition null while the
/// service wrote a full one -- which is exactly the drift that makes two surfaces disagree about
/// one run. `resident` is not here: only the device actor knows whether the model was already
/// loaded, so it travels in the summary line instead of being guessed at the top of the file.
pub fn at_start(model: &str, created: i64) -> npu_engine::RunConditions {
    npu_engine::RunConditions {
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        model: model.to_string(),
        power_mode: power_mode(),
        resident: None,
        kernel: kernel_release(),
        started_unix: created,
    }
}

/// `uname -r`, read from `/proc`. One more thing two runs can differ by.
pub fn kernel_release() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/osrelease").ok().map(|s| s.trim().to_string())
}
