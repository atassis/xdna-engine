//! The live model list, published as a file the CLI reads.
//!
//! Why a file rather than a socket or the HTTP port. `RuntimeDirectory=` makes systemd create this
//! directory when the service starts and REMOVE it when the service stops, so the file's presence
//! is the liveness signal and the init system maintains it -- no probe, no handshake. A reader
//! cannot hang on a wedged service, because it reads bytes instead of connecting; it cannot mistake
//! another server for this one, because the path is ours by construction where port 11434 is a
//! shared default with ollama and FLM; and it needs no protocol, port or timeout.
//!
//! The document carries the PORT because the path does not: `$XDG_RUNTIME_DIR/xdna-engine` is
//! per-USER, so a second engine started on another port -- a test instance, a peer session -- writes
//! here too. That happened within the hour of this file existing, and made `npu models` report a
//! foreign instance's models as the service's. A reader must check the port it expects, the same way
//! `preflight_serve` checks who holds a socket rather than assuming.
//!
//! The cost is staleness, which is why `written_unix` is in the document. A reader states the age
//! and lets the human judge -- a stale answer carrying its age beats a connection that never
//! returns, which is the failure mode a socket would have on a device call that never comes back.

use crate::registry::ModelStatus;
use std::path::PathBuf;

pub const FILE_NAME: &str = "status.json";

/// The directory both sides agree on: what systemd hands the service in `RUNTIME_DIRECTORY`, and
/// what `RuntimeDirectory=xdna-engine` resolves to for a user unit -- `$XDG_RUNTIME_DIR/xdna-engine`
/// -- which is what a CLI outside the service can compute. `None` when neither is set, in which
/// case publishing is simply skipped rather than guessed at.
pub fn dir() -> Option<PathBuf> {
    dir_from(std::env::var("RUNTIME_DIRECTORY").ok().as_deref(),
             std::env::var("XDG_RUNTIME_DIR").ok().as_deref())
}

/// [`dir`] with the environment as arguments, so the agreement between the two sides is testable
/// without mutating process env -- which races every other test in a parallel run.
pub fn dir_from(runtime_directory: Option<&str>, xdg_runtime_dir: Option<&str>) -> Option<PathBuf> {
    if let Some(d) = runtime_directory {
        // systemd may hand several colon-separated paths; taking the whole string would make a
        // directory name containing a colon that silently never matches the reader's.
        return d.split(':').next().filter(|s| !s.is_empty()).map(PathBuf::from);
    }
    xdg_runtime_dir.filter(|s| !s.is_empty()).map(|d| PathBuf::from(d).join("xdna-engine"))
}

pub fn path() -> Option<PathBuf> { dir().map(|d| d.join(FILE_NAME)) }

/// Publish the current status. Best-effort: a service that cannot write its status file must keep
/// serving, so every failure here is silent by design -- the reader sees a missing or old file,
/// which is exactly the state that happened.
///
/// Written to a temporary and renamed, because a reader that catches a half-written file would get
/// a parse error that reads like a corrupt service rather than a race.
pub fn publish(port: u16, status: &[ModelStatus]) {
    let Some(p) = path() else { return };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let body = crate::http::models_json(status);
    let doc = format!(
        "{{\"written_unix\":{now},\"pid\":{},\"port\":{port},\"models\":{body}}}",
        std::process::id());
    let tmp = p.with_extension("json.tmp");
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&tmp, doc.as_bytes()).is_ok() {
        let _ = std::fs::rename(&tmp, &p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_directory_takes_the_first_of_several() {
        assert_eq!(dir_from(Some("/run/user/1000/a:/run/user/1000/b"), None).unwrap(),
                   PathBuf::from("/run/user/1000/a"));
    }

    /// Both sides must compute the same path, or the service publishes where nothing reads. That is
    /// the correspondence this module exists to keep single, and `RuntimeDirectory=xdna-engine` in a
    /// USER unit resolves to exactly `$XDG_RUNTIME_DIR/xdna-engine`.
    #[test]
    fn xdg_fallback_matches_what_a_user_unit_creates() {
        assert_eq!(dir_from(None, Some("/run/user/1000")).unwrap(),
                   PathBuf::from("/run/user/1000/xdna-engine"));
    }

    /// Neither set: publish nothing rather than guess a path. A status file in the wrong place is
    /// worse than none -- a reader would find it absent and report the service down while it runs.
    #[test]
    fn no_environment_means_no_path_rather_than_a_guess() {
        assert!(dir_from(None, None).is_none());
        assert!(dir_from(Some(""), Some("")).is_none());
    }
}
