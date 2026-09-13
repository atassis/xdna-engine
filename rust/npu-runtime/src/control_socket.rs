//! The CLI's transport to the service: a unix socket beside where `status.json` used to live,
//! served OFF the actor thread so a busy device call can never hang a query the way
//! `Handle::status()` can -- that sends `Cmd::Status` into the actor's channel, which queues behind
//! whatever `Cmd::Generate`/`Cmd::Serve` is already in flight.
//!
//! `GET /v1/models` is answered here, directly, from [`LiveStatus`] -- a snapshot the actor updates
//! after every command and idle sweep, the same moments it used to write `status.json`. Every other
//! request is unchanged HTTP shape and falls through to [`crate::http::route`], so a command lands
//! on this transport by nothing more than being reachable here instead of only over TCP.
use std::io::BufReader;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::actor::Handle;
use crate::http::{parse_request, respond, route, Body};
use crate::registry::ModelStatus;

pub const SOCKET_NAME: &str = "control.sock";
const SOCKET_TIMEOUT: Duration = Duration::from_secs(60);

/// The directory both sides agree on -- `status.json`'s own rule, unchanged by the move to a
/// socket: `RUNTIME_DIRECTORY` (what systemd hands a user unit) first, else `$XDG_RUNTIME_DIR/npu`,
/// else `None` rather than a guessed path.
pub fn dir() -> Option<PathBuf> {
    dir_from(std::env::var("RUNTIME_DIRECTORY").ok().as_deref(),
             std::env::var("XDG_RUNTIME_DIR").ok().as_deref())
}

pub fn dir_from(runtime_directory: Option<&str>, xdg_runtime_dir: Option<&str>) -> Option<PathBuf> {
    if let Some(d) = runtime_directory {
        // systemd may hand several colon-separated paths; the whole string would make a directory
        // name containing a colon that silently never matches the reader's.
        return d.split(':').next().filter(|s| !s.is_empty()).map(PathBuf::from);
    }
    xdg_runtime_dir.filter(|s| !s.is_empty()).map(|d| PathBuf::from(d).join("npu"))
}

/// `$NPU_SOCKET_ENDPOINT` if set -- an explicit override, for a container or test where
/// `XDG_RUNTIME_DIR` is not set up the normal desktop way -- else the `RuntimeDirectory`-derived
/// default. `npu serve` binds this; every socket-based CLI command connects to it.
pub fn socket_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NPU_SOCKET_ENDPOINT") { return Some(PathBuf::from(p)); }
    dir().map(|d| d.join(SOCKET_NAME))
}

/// Bind the control socket, clearing a stale path first: `UnixListener::bind` fails `AddrInUse` on
/// an existing file regardless of whether anything is listening, and an unclean shutdown (a crash, a
/// `SIGKILL`) can leave one behind even though `RuntimeDirectory=` cleans the whole directory only
/// across a stop/start cycle.
///
/// A path that answers a connection is a DIFFERENT thing from a stale one, and only the second may
/// be removed -- unlinking a live listener's path would not stop it, it would just steal the name out
/// from under it, so every later client reaches the wrong process instead of the one already serving.
/// `AddrInUse` on the bind below is the honest failure for that case, the same as a TCP port already
/// held (`preflight_serve`).
pub fn bind(path: &Path) -> std::io::Result<UnixListener> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    if path.exists() && UnixStream::connect(path).is_err() {
        std::fs::remove_file(path)?;
    }
    UnixListener::bind(path)
}

/// The out-of-band status document. The actor renders it fresh after every command and sweep and
/// stores it here; a reader takes the current bytes under one uncontended lock, never sending
/// anything to the actor's channel.
#[derive(Clone, Default)]
pub struct LiveStatus(Arc<Mutex<String>>);

impl LiveStatus {
    pub fn set(&self, doc: String) { *self.0.lock().unwrap() = doc; }
    pub fn get(&self) -> String { self.0.lock().unwrap().clone() }
}

/// The same document shape `status.json` carried, so `npu model ls`/`npu top` need no format change
/// to read this instead of a file.
pub fn render(port: u16, started_unix: u64, status: &[ModelStatus]) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!(
        "{{\"written_unix\":{now},\"started_unix\":{started_unix},\"pid\":{},\"port\":{port},\"models\":{}}}",
        std::process::id(), crate::http::models_json(status))
}

/// Blocking accept loop, one connection at a time -- control traffic is rare and every request
/// answered here is immediate, so there is no case for the concurrency `http::serve_on` also does
/// not have.
pub fn serve(listener: UnixListener, handle: Handle, live: LiveStatus, cfg_path: PathBuf) {
    for stream in listener.incoming() {
        match stream {
            Ok(s) => { if let Err(e) = handle_conn(s, &handle, &live, &cfg_path) {
                eprintln!("[npu-control] {e}");
            } }
            Err(e) => eprintln!("[npu-control] accept: {e}"),
        }
    }
}

fn handle_conn(mut stream: UnixStream, handle: &Handle, live: &LiveStatus, cfg_path: &Path)
    -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(SOCKET_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SOCKET_TIMEOUT));
    let mut reader = BufReader::new(stream.try_clone()?);
    let req = match parse_request(&mut reader) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData =>
            return respond(&mut stream, 413, &"{\"error\":\"too large\"}".into()),
        Err(e) => return Err(e),
    };
    // The one path that must never reach `route()`: that would call `Handle::status()`, which is
    // exactly the hang this transport exists to avoid. Every other path -- including a streamed
    // generation -- is unchanged `route()`/`respond()`, the same code the TCP surface runs.
    let (code, resp) = if req.method == "GET" && req.path == "/v1/models" {
        (200, Body::Json(live.get()))
    } else {
        route(&req, handle, cfg_path)
    };
    respond(&mut stream, code, &resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ported verbatim from `status_file.rs`'s own test of the same name: systemd may hand several
    /// colon-separated paths, and taking the whole string would make a directory name containing a
    /// colon that silently never matches the reader's.
    #[test]
    fn runtime_directory_takes_the_first_of_several() {
        assert_eq!(dir_from(Some("/run/user/1000/a:/run/user/1000/b"), None).unwrap(),
                   PathBuf::from("/run/user/1000/a"));
    }

    /// Both sides must compute the same directory, or the actor publishes where nothing listens.
    /// `RuntimeDirectory=npu` in a user unit resolves to exactly `$XDG_RUNTIME_DIR/npu`.
    #[test]
    fn xdg_fallback_matches_what_a_user_unit_creates() {
        assert_eq!(dir_from(None, Some("/run/user/1000")).unwrap(),
                   PathBuf::from("/run/user/1000/npu"));
    }

    /// Neither set: no path rather than a guess. A socket in the wrong place is worse than none --
    /// a client would get `ECONNREFUSED`/`ENOENT` either way, but a guessed path could collide.
    #[test]
    fn no_environment_means_no_path_rather_than_a_guess() {
        assert!(dir_from(None, None).is_none());
        assert!(dir_from(Some(""), Some("")).is_none());
    }

    #[test]
    fn bind_clears_a_stale_path_left_by_an_unclean_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SOCKET_NAME);
        // Stands in for a leftover from a crash: a path that exists but answers no one.
        std::fs::write(&path, b"").unwrap();
        assert!(bind(&path).is_ok(), "a stale, unanswering path must not block a fresh bind");
    }

    /// The property `bind`'s doc comment argues for: a second instance must not silently steal the
    /// path out from under a listener that is genuinely alive and answering.
    #[test]
    fn bind_refuses_to_steal_a_socket_something_is_still_answering() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SOCKET_NAME);
        let first = bind(&path).unwrap();
        let err = bind(&path).expect_err("a live listener's path must not be stolen");
        assert_eq!(err.kind(), std::io::ErrorKind::AddrInUse);
        // The original listener must still be reachable -- the failed second bind must not have
        // touched it.
        drop(UnixStream::connect(&path).expect("the first listener must still answer"));
        drop(first);
    }
}
