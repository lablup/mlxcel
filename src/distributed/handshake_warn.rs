// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Rate limiter for the warnings an authenticated listener logs about
//! connections it refused (#2655, #2692).
//!
//! A party without the cluster secret can open sockets at will, so every
//! refusal it causes (a pending-pool refusal, a hello with bad magic, a stall
//! to the handshake timeout) is logged at warn only a few times per minute
//! and at debug afterwards. The budget is split in two classes so that a
//! trickle of such junk cannot push the one line an operator needs down to
//! debug:
//!
//! - [`HandshakeWarnClass::Noise`]: refusals a stranger can produce with no
//!   protocol knowledge at all and that carry no operator action
//!   (pending-pool refusals, malformed or absent hellos, timeouts,
//!   unparseable identities).
//! - [`HandshakeWarnClass::Diagnostic`]: the typed refusals that tell the
//!   operator what to fix: a probable secret mismatch
//!   ([`ClientAbortedAfterServerProof`], [`HandshakeRejected`]), an identity
//!   or suite that does not match ([`IdentityNotOwned`],
//!   [`FrameSuiteMismatch`]), and a frame that failed authentication after the
//!   handshake ([`FrameRejected`]).
//!
//! Each class has its own window of [`HANDSHAKE_WARN_PER_WINDOW`] warnings
//! per [`HANDSHAKE_WARN_WINDOW`]. When a window rolls over and the previous
//! one suppressed lines, the first call of the new window also logs how many
//! were suppressed, so a sustained flood still leaves one summary line per
//! minute in the log.
//!
//! Residual risk: the `Diagnostic` class is not reserved for authenticated
//! parties. A party without the secret can still produce
//! `ClientAbortedAfterServerProof` (send a well-formed hello naming its own
//! IP, then close after the server proof), and the `IdentityNotOwned` and
//! `FrameSuiteMismatch` refusals are decided from the hello alone, before
//! the client authenticates, so it can trigger those too. Any of them can
//! saturate the `Diagnostic` window. The rollover summary is what keeps the
//! operator informed then. Warning on the first event per source instead would need a
//! per-source map that an IPv6 source defeats by rotating addresses.
//!
//! Used by: `tcp_transport::accept_loop` and `handle_connection`,
//! `stage_assignment` (its listener and rejected-frame paths), and the
//! disaggregated router's result demux (`server::router_front`).

use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::cluster_handshake::{
    ClientAbortedAfterServerProof, FrameSuiteMismatch, HandshakeRejected, IdentityNotOwned,
};
use super::frame_protection::FrameRejected;

/// Warnings each class may log at warn per window; the rest go to debug.
pub(crate) const HANDSHAKE_WARN_PER_WINDOW: u32 = 5;
/// Length of one warning window.
pub(crate) const HANDSHAKE_WARN_WINDOW: Duration = Duration::from_secs(60);

/// Which budget a refusal draws from. See the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum HandshakeWarnClass {
    /// Producible by anyone without the secret; bounded tightly.
    Noise,
    /// A typed refusal the operator needs to see.
    Diagnostic,
}

impl HandshakeWarnClass {
    fn index(self) -> usize {
        match self {
            Self::Noise => 0,
            Self::Diagnostic => 1,
        }
    }
}

impl fmt::Display for HandshakeWarnClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Noise => "noise",
            Self::Diagnostic => "diagnostic",
        })
    }
}

/// Classify a failed inbound handshake for the warning budget: `Diagnostic`
/// when the error chain holds one of the typed refusals, `Noise` otherwise
/// (bad magic, a malformed hello, a timeout, an I/O error).
pub(crate) fn handshake_warn_class(err: &anyhow::Error) -> HandshakeWarnClass {
    let diagnostic = err.chain().any(|cause| {
        cause
            .downcast_ref::<ClientAbortedAfterServerProof>()
            .is_some()
            || cause.downcast_ref::<HandshakeRejected>().is_some()
            || cause.downcast_ref::<IdentityNotOwned>().is_some()
            || cause.downcast_ref::<FrameSuiteMismatch>().is_some()
            || cause.downcast_ref::<FrameRejected>().is_some()
    });
    if diagnostic {
        HandshakeWarnClass::Diagnostic
    } else {
        HandshakeWarnClass::Noise
    }
}

/// What [`WarnBudget::allow_at`] decided for one event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WarnDecision {
    /// Log this event at warn (`true`) or at debug (`false`).
    pub warn: bool,
    /// Events of the same class that were logged at debug in the window that
    /// just ended. Non-zero only on the first event of a new window.
    pub suppressed_in_previous_window: u32,
}

/// One window of a class: when it started and how many events it has seen.
#[derive(Debug, Clone, Copy)]
struct Window {
    start: Instant,
    count: u32,
}

/// Per-class warning windows. Thread-safe; one instance is process-wide
/// ([`handshake_warn_allowed`]), and tests build their own to inject time.
#[derive(Debug)]
pub(crate) struct WarnBudget {
    windows: Mutex<[Option<Window>; 2]>,
    per_window: u32,
    window: Duration,
}

impl WarnBudget {
    /// A budget of `per_window` warnings per `window` for each class.
    pub(crate) const fn new(per_window: u32, window: Duration) -> Self {
        Self {
            windows: Mutex::new([None, None]),
            per_window,
            window,
        }
    }

    /// Account one event of `class` happening at `now`.
    pub(crate) fn allow_at(&self, class: HandshakeWarnClass, now: Instant) -> WarnDecision {
        let mut windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        let slot = &mut windows[class.index()];
        match slot {
            Some(window) if now.saturating_duration_since(window.start) < self.window => {
                window.count = window.count.saturating_add(1);
                WarnDecision {
                    warn: window.count <= self.per_window,
                    suppressed_in_previous_window: 0,
                }
            }
            _ => {
                let suppressed = slot
                    .map(|window| window.count.saturating_sub(self.per_window))
                    .unwrap_or(0);
                *slot = Some(Window {
                    start: now,
                    count: 1,
                });
                WarnDecision {
                    warn: true,
                    suppressed_in_previous_window: suppressed,
                }
            }
        }
    }
}

static BUDGET: WarnBudget = WarnBudget::new(HANDSHAKE_WARN_PER_WINDOW, HANDSHAKE_WARN_WINDOW);

/// Whether a refusal of `class` may be logged at warn right now (debug
/// otherwise), against the process-wide budget. When the class's window has
/// just rolled over and the previous window suppressed lines, one summary
/// warn with their count is logged here.
pub(crate) fn handshake_warn_allowed(class: HandshakeWarnClass) -> bool {
    let decision = BUDGET.allow_at(class, Instant::now());
    if decision.suppressed_in_previous_window > 0 {
        tracing::warn!(
            "{} cluster handshake warnings ({class}) were logged at debug in the last {}s",
            decision.suppressed_in_previous_window,
            HANDSHAKE_WARN_WINDOW.as_secs()
        );
    }
    decision.warn
}

#[cfg(test)]
#[path = "handshake_warn_tests.rs"]
mod tests;
