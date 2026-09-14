//! A cancellation signal that reaches the device loop.
//!
//! The generation sink already carries an abort path -- a sink returning `false` ends the run with
//! [`FinishReason::Aborted`](crate::FinishReason::Aborted) -- but it can only be consulted when
//! there is something to deliver. Prefill delivers nothing, so a prompt long enough to matter is
//! exactly the prompt that cannot be stopped: measured 2026-09-14, a ~6k-token Gemma-4 prefill ran
//! ~60 minutes at ~596 ms/token and cancelling in the client changed nothing.
//!
//! This is the signal that does not depend on traffic existing. One flag, set by whoever learns the
//! work is unwanted, polled by the generator at every dispatch boundary.
//!
//! **The floor is one dispatch.** A dispatch already submitted cannot be taken back -- the host is
//! parked in the driver until the device finishes -- so cancellation lands within one dispatch
//! (~530-600 ms on Gemma-4, ~28 ms on Qwen3), never inside one. Lowering that floor needs a driver
//! surface that does not exist yet; see task `amdxdna-no-abort-for-in-flight-commands`.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Why a generation was cancelled.
///
/// Carried rather than collapsed to a bool: "the client hung up" and "the deadline expired" are
/// different operational facts, and a run record that cannot tell them apart cannot answer the
/// question an operator actually asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    /// The client closed the connection.
    PeerGone,
    /// The request outlived its deadline.
    Deadline,
    /// An operator asked for it.
    Operator,
    /// The server is shutting down.
    Shutdown,
}

impl CancelReason {
    pub fn as_str(self) -> &'static str {
        match self {
            CancelReason::PeerGone => "peer-gone",
            CancelReason::Deadline => "deadline",
            CancelReason::Operator => "operator",
            CancelReason::Shutdown => "shutdown",
        }
    }

    fn code(self) -> u8 {
        match self {
            CancelReason::PeerGone => 1,
            CancelReason::Deadline => 2,
            CancelReason::Operator => 3,
            CancelReason::Shutdown => 4,
        }
    }

    fn from_code(c: u8) -> Option<CancelReason> {
        match c {
            1 => Some(CancelReason::PeerGone),
            2 => Some(CancelReason::Deadline),
            3 => Some(CancelReason::Operator),
            4 => Some(CancelReason::Shutdown),
            _ => None,
        }
    }
}

/// A shared cancel flag. Cloning shares the signal; a clone is another holder, not another request.
///
/// `Default` is an uncancelled token nobody else holds, which is what a caller that never cancels
/// wants -- so adding this to a params struct costs an allocation and changes no behaviour.
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<AtomicU8>);

impl Cancel {
    pub fn new() -> Cancel {
        Cancel::default()
    }

    /// Request cancellation. Idempotent, and the FIRST reason wins: a run the client abandoned and
    /// that shutdown then swept is still reported as the client leaving, which is the fact that
    /// explains it.
    pub fn cancel(&self, reason: CancelReason) {
        let _ = self.0.compare_exchange(0, reason.code(), Ordering::Relaxed, Ordering::Relaxed);
    }

    /// `Relaxed` throughout: this orders nothing, it only publishes one byte. The generator polls it
    /// between device dispatches, where a missed update costs one more dispatch and nothing else.
    pub fn reason(&self) -> Option<CancelReason> {
        CancelReason::from_code(self.0.load(Ordering::Relaxed))
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_token_is_live_and_clones_share_one_signal() {
        let a = Cancel::new();
        let b = a.clone();
        assert!(!a.is_cancelled());
        b.cancel(CancelReason::PeerGone);
        assert!(a.is_cancelled(), "a clone is another holder, not another request");
        assert_eq!(a.reason(), Some(CancelReason::PeerGone));
    }

    /// The first reason is the explaining one. Shutdown sweeping a run the client had already
    /// abandoned must not rewrite why it ended.
    #[test]
    fn the_first_reason_wins() {
        let c = Cancel::new();
        c.cancel(CancelReason::PeerGone);
        c.cancel(CancelReason::Shutdown);
        assert_eq!(c.reason(), Some(CancelReason::PeerGone));
    }

    #[test]
    fn default_is_uncancelled() {
        assert!(!Cancel::default().is_cancelled());
        assert_eq!(Cancel::default().reason(), None);
    }

    #[test]
    fn every_reason_round_trips_its_code_and_name() {
        for r in [CancelReason::PeerGone, CancelReason::Deadline,
                  CancelReason::Operator, CancelReason::Shutdown] {
            assert_eq!(CancelReason::from_code(r.code()), Some(r));
            assert!(!r.as_str().is_empty());
        }
        assert_eq!(CancelReason::from_code(0), None, "0 is the live state, not a reason");
    }
}
