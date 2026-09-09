//! The JSONL run log: one file per generation, written as the tokens stream.
//!
//! Best-effort by construction. Every failure path here is a silent no-op, because a log that
//! cannot be written must never be able to fail a request -- the log exists to explain a request,
//! not to be a precondition for one.
//!
//! Off unless `NPU_TELEMETRY_LOG` names a directory. The reason to have it at all is the case the
//! per-request opt-in cannot serve: the run you did not know you would need to explain. By the time
//! you know, it has already happened.

use std::fs::File;
use std::io::Write;

use npu_engine::telemetry::wire;

pub struct RunLog {
    f: File,
}

impl RunLog {
    /// Open `$NPU_TELEMETRY_LOG/<id>.jsonl`, or `None` when the variable is unset or the directory
    /// cannot be created. The id is a completion id (`chatcmpl-<hex>`), so it is already unique and
    /// already appears in the response the client got -- which is what lets a user match a log file
    /// to the answer that puzzled them.
    pub fn open(id: &str) -> Option<RunLog> {
        RunLog::open_in(std::env::var_os("NPU_TELEMETRY_LOG")?, id)
    }

    /// The same thing with the directory named outright. Tests use this: reading the environment
    /// is the one part of `open` that cannot be exercised in parallel without two tests fighting
    /// over a process-global.
    pub fn open_in(dir: impl AsRef<std::path::Path>, id: &str) -> Option<RunLog> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).ok()?;
        File::create(dir.join(format!("{id}.jsonl"))).ok().map(|f| RunLog { f })
    }

    pub fn line(&mut self, v: &serde_json::Value) {
        let _ = writeln!(self.f, "{v}");
    }

    /// The header, from what the serving thread can see. `resident` is not among it -- only the
    /// actor knows whether the model was already loaded -- so it travels in the summary line
    /// instead of being guessed here.
    pub fn header(&mut self, m: &wire::RunMeta) {
        self.line(&wire::header_line(&crate::conditions::at_start(&m.model, m.created), m));
    }
}
