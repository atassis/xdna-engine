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
use std::time::{Duration, Instant};

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

/// The out-of-band status snapshot. The actor republishes it after every command and every sweep;
/// a reader clones it under one uncontended lock and never sends anything to the actor's channel.
///
/// STRUCTURED rather than a rendered document, because two surfaces need different renderings of
/// the same facts and a reader that parses a string back into them is a second format to keep in
/// agreement with the first.
#[derive(Clone, Default)]
pub struct LiveStatus {
    inner: Arc<Mutex<Snapshot>>,
    /// Process constants, kept here so a READER can render the whole document without reaching for
    /// the config the actor owns -- which is the reach this type exists to avoid.
    port: u16,
    started_unix: u64,
}

/// What the actor last published, and when.
///
/// The timestamp is not decoration. A reader that cannot say how old its answer is cannot tell
/// "nothing is happening" from "the actor has not come round the loop in 22 minutes" -- and the
/// second is the one worth acting on.
#[derive(Clone)]
pub struct Snapshot {
    pub models: Arc<Vec<ModelStatus>>,
    pub at: Instant,
    /// What the actor is inside RIGHT NOW, when that is not visible in `models`.
    ///
    /// A model being served shows up as `busy`; a model being LOADED does not, because loading
    /// happens before there is anything to mark. That made the longest single thing the actor does
    /// -- a 15 GB load -- the one thing it never reported, so a caller waiting on it saw an idle
    /// snapshot going stale and no reason for either.
    pub doing: Option<String>,
}

impl Default for Snapshot {
    fn default() -> Snapshot {
        Snapshot { models: Arc::new(Vec::new()), at: Instant::now(), doing: None }
    }
}

impl LiveStatus {
    pub fn new(port: u16, started_unix: u64) -> LiveStatus {
        LiveStatus { inner: Default::default(), port, started_unix }
    }

    pub fn set(&self, models: Vec<ModelStatus>) {
        self.set_doing(models, None)
    }

    /// Publish, naming what the actor is about to do. Call it BEFORE the long thing, not after:
    /// the point is to be readable while it runs.
    pub fn set_doing(&self, models: Vec<ModelStatus>, doing: Option<String>) {
        *self.inner.lock().unwrap() =
            Snapshot { models: Arc::new(models), at: Instant::now(), doing };
    }

    pub fn get(&self) -> Snapshot {
        self.inner.lock().unwrap().clone()
    }

    /// The `status.json`-shaped document `npu model ls` / `npu top` read.
    pub fn doc(&self) -> String {
        render(self.port, self.started_unix, &self.get().models)
    }
}

/// The same document shape `status.json` carried, so `npu model ls`/`npu top` need no format change
/// to read this instead of a file.
pub fn render(port: u16, started_unix: u64, status: &[ModelStatus]) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!(
        "{{\"written_unix\":{now},\"started_unix\":{started_unix},\"pid\":{},\"port\":{port},\
         \"npu_cold_wakes\":{},\"models\":{}}}",
        std::process::id(), crate::conditions::cold_wake_count(), crate::http::models_json(status))
}

/// Accept loop, one thread per connection.
///
/// It used to serve connections inline, on the reasoning that control traffic is rare and every
/// answer here is immediate. The second half was never true: a status read is served from
/// [`LiveStatus`] and is immediate, but anything MUTATING goes to the actor and waits as long as
/// the actor is busy. Inline, one `npu model stop` behind a long generation stopped the socket
/// answering at all -- including the status reads that would have explained why.
pub fn serve(listener: UnixListener, handle: Handle, live: LiveStatus, cfg_path: PathBuf) {
    for stream in listener.incoming() {
        let s = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[npu-control] accept: {e}");
                continue;
            }
        };
        let (handle, live, cfg_path) = (handle.clone(), live.clone(), cfg_path.clone());
        let _ = std::thread::Builder::new().name("npu-control".into()).spawn(move || {
            if let Err(e) = handle_conn(s, &handle, &live, &cfg_path) {
                eprintln!("[npu-control] {e}");
            }
        });
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
        (200, Body::Json(live.doc()))
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
