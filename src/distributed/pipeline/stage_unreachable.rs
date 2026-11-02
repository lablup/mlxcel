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

//! The stage-caused exchange failures that open the availability gate
//! (#2790 item 5).
//!
//! An exchange can fail because the stage did not answer within the wait
//! or because the command could not be sent to it; both say something
//! about the stage. A serialization error, a missing local address or a
//! closed receive channel are the coordinator's own failures and stay plain
//! `anyhow` errors, so `RemotePipelineRuntime::exchange` opens the gate only
//! for a [`StageUnreachable`] in the error.
//!
//! Used by: `pipeline::runtime` (`exchange_within` builds it, `exchange`
//! and the gate's open line read it).

use std::fmt;
use std::time::Duration;

/// Why a stage could not be reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StageUnreachableCause {
    /// No matching reply arrived within the wait.
    Timeout(Duration),
    /// The transport could not deliver the command; the reason is the
    /// transport's error text.
    SendFailed(String),
}

/// An exchange failed because of the stage it went to.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) struct StageUnreachable {
    /// The address the command went to.
    pub(crate) peer: String,
    pub(crate) cause: StageUnreachableCause,
}

impl fmt::Display for StageUnreachable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            // The text predates the type; callers and tests match on
            // "timed out".
            StageUnreachableCause::Timeout(waited) => {
                write!(f, "remote stage {} timed out after {waited:?}", self.peer)
            }
            StageUnreachableCause::SendFailed(reason) => {
                write!(
                    f,
                    "remote stage {} could not be reached: {reason}",
                    self.peer
                )
            }
        }
    }
}

/// The cause a failed availability probe reports when it reopens the gate:
/// the `StageUnreachable` cause carried by the error chain, so a probe whose
/// send failed keeps its reason, and `Timeout(fallback_wait)` for everything
/// else (a probe that was sent and not answered).
pub(crate) fn probe_failure_cause(
    err: &anyhow::Error,
    fallback_wait: Duration,
) -> StageUnreachableCause {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<StageUnreachable>())
        .map_or(
            StageUnreachableCause::Timeout(fallback_wait),
            |unreachable| unreachable.cause.clone(),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_probe_send_keeps_its_reason_and_a_timeout_keeps_the_wait() {
        let wait = Duration::from_secs(2);
        let send = anyhow::Error::new(StageUnreachable {
            peer: "stage".to_string(),
            cause: StageUnreachableCause::SendFailed("connection refused".to_string()),
        })
        .context("probe failed");
        assert_eq!(
            probe_failure_cause(&send, wait),
            StageUnreachableCause::SendFailed("connection refused".to_string())
        );
        let timeout = anyhow::Error::new(StageUnreachable {
            peer: "stage".to_string(),
            cause: StageUnreachableCause::Timeout(Duration::from_secs(3)),
        });
        assert_eq!(
            probe_failure_cause(&timeout, wait),
            StageUnreachableCause::Timeout(Duration::from_secs(3))
        );
        assert_eq!(
            probe_failure_cause(&anyhow::anyhow!("other"), wait),
            StageUnreachableCause::Timeout(wait)
        );
    }

    #[test]
    fn display_keeps_the_timeout_text_and_names_the_send_reason() {
        let timeout = StageUnreachable {
            peer: "10.0.0.2:9000".to_string(),
            cause: StageUnreachableCause::Timeout(Duration::from_secs(30)),
        };
        assert_eq!(
            timeout.to_string(),
            "remote stage 10.0.0.2:9000 timed out after 30s"
        );
        let send = StageUnreachable {
            peer: "10.0.0.2:9000".to_string(),
            cause: StageUnreachableCause::SendFailed("connection refused".to_string()),
        };
        assert_eq!(
            send.to_string(),
            "remote stage 10.0.0.2:9000 could not be reached: connection refused"
        );
    }

    #[test]
    fn a_wrapped_cause_is_found_by_downcast() {
        let err = anyhow::Error::new(StageUnreachable {
            peer: "stage".to_string(),
            cause: StageUnreachableCause::Timeout(Duration::from_millis(5)),
        })
        .context("forward failed");
        assert!(
            err.chain()
                .any(|cause| cause.downcast_ref::<StageUnreachable>().is_some())
        );
        assert!(
            anyhow::anyhow!("serialization failed")
                .downcast_ref::<StageUnreachable>()
                .is_none()
        );
    }
}
