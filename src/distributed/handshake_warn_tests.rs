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

use std::net::{IpAddr, Ipv4Addr};

use super::*;
use crate::distributed::cluster_handshake::HANDSHAKE_TIMEOUT;
use crate::distributed::frame_protection::FrameSuite;

fn budget() -> WarnBudget {
    WarnBudget::new(HANDSHAKE_WARN_PER_WINDOW, HANDSHAKE_WARN_WINDOW)
}

/// Junk connections cannot push the typed refusals down to debug: each class
/// has its own window.
#[test]
fn noise_flood_does_not_suppress_diagnostic_warnings() {
    let budget = budget();
    let now = Instant::now();
    let mut noise_warned = 0;
    for i in 0..50u64 {
        let at = now + Duration::from_millis(i * 10);
        if budget.allow_at(HandshakeWarnClass::Noise, at).warn {
            noise_warned += 1;
        }
    }
    assert_eq!(noise_warned, HANDSHAKE_WARN_PER_WINDOW);
    let decision = budget.allow_at(HandshakeWarnClass::Diagnostic, now + Duration::from_secs(1));
    assert_eq!(
        decision,
        WarnDecision {
            warn: true,
            suppressed_in_previous_window: 0,
            previous_window_ended_ago: Duration::ZERO,
        }
    );
    // And the diagnostic window has its own full budget.
    let mut diagnostic_warned = 1;
    for i in 0..10u64 {
        let at = now + Duration::from_secs(2) + Duration::from_millis(i);
        if budget.allow_at(HandshakeWarnClass::Diagnostic, at).warn {
            diagnostic_warned += 1;
        }
    }
    assert_eq!(diagnostic_warned, HANDSHAKE_WARN_PER_WINDOW);
}

/// The first event after a window rolls over reports how many events of its
/// class the previous window logged at debug.
#[test]
fn rollover_reports_suppressed_count() {
    let budget = budget();
    let now = Instant::now();
    for i in 0..8u64 {
        let decision = budget.allow_at(HandshakeWarnClass::Noise, now + Duration::from_secs(i));
        assert_eq!(decision.warn, i < u64::from(HANDSHAKE_WARN_PER_WINDOW));
        assert_eq!(decision.suppressed_in_previous_window, 0);
    }
    let later = budget.allow_at(HandshakeWarnClass::Noise, now + Duration::from_secs(61));
    assert_eq!(
        later,
        WarnDecision {
            warn: true,
            suppressed_in_previous_window: 3,
            // The window started at `now` and ended at `now + 60s`.
            previous_window_ended_ago: Duration::from_secs(1),
        }
    );
    // The summary is reported once: the next event in the new window carries
    // no count, and a window that suppressed nothing reports zero.
    let next = budget.allow_at(HandshakeWarnClass::Noise, now + Duration::from_secs(62));
    assert_eq!(next.suppressed_in_previous_window, 0);
    let quiet = budget.allow_at(HandshakeWarnClass::Noise, now + Duration::from_secs(200));
    assert_eq!(
        quiet,
        WarnDecision {
            warn: true,
            suppressed_in_previous_window: 0,
            // The window that started at `now + 61s` ended at `now + 121s`.
            previous_window_ended_ago: Duration::from_secs(79),
        }
    );
    // The other class's window is untouched by the rollover.
    let other = budget.allow_at(
        HandshakeWarnClass::Diagnostic,
        now + Duration::from_secs(200),
    );
    assert_eq!(
        other,
        WarnDecision {
            warn: true,
            suppressed_in_previous_window: 0,
            previous_window_ended_ago: Duration::ZERO,
        }
    );
}

/// The typed refusals are diagnostic, also when wrapped in context; anything
/// a stranger can produce without the secret is noise.
#[test]
fn classifies_handshake_errors() {
    let diagnostic: Vec<anyhow::Error> = vec![
        ClientAbortedAfterServerProof.into(),
        HandshakeRejected("secret mismatch".to_string()).into(),
        IdentityNotOwned {
            advertised: "10.0.0.9:1".to_string(),
            source_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
        }
        .into(),
        FrameSuiteMismatch {
            peer_suite_id: 0,
            local: FrameSuite::default(),
        }
        .into(),
        FrameRejected.into(),
        anyhow::Error::from(HandshakeRejected("wrapped".to_string()))
            .context("cluster handshake with 10.0.0.9:1 failed"),
    ];
    for err in &diagnostic {
        assert_eq!(
            handshake_warn_class(err),
            HandshakeWarnClass::Diagnostic,
            "{err:#}"
        );
    }
    let noise: Vec<anyhow::Error> = vec![
        anyhow::anyhow!("cluster handshake timed out after {HANDSHAKE_TIMEOUT:?}"),
        anyhow::anyhow!("bad magic"),
        std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof").into(),
    ];
    for err in &noise {
        assert_eq!(
            handshake_warn_class(err),
            HandshakeWarnClass::Noise,
            "{err:#}"
        );
    }
    assert_eq!(HandshakeWarnClass::Noise.to_string(), "noise");
    assert_eq!(HandshakeWarnClass::Diagnostic.to_string(), "diagnostic");
    assert_eq!(HandshakeWarnClass::RouterDrop.to_string(), "router-drop");
    assert_eq!(HandshakeWarnClass::AtCapacity.to_string(), "at-capacity");
}

/// An authenticated frame of an unknown kind is a diagnostic (a
/// version-skewed member), also when wrapped in context (#2712 item 4).
#[test]
fn classifies_unknown_authenticated_kind_as_diagnostic() {
    use crate::distributed::tcp_frame::UnknownAuthenticatedKind;

    let bare: anyhow::Error = UnknownAuthenticatedKind { kind: 200 }.into();
    assert_eq!(handshake_warn_class(&bare), HandshakeWarnClass::Diagnostic);
    let wrapped = anyhow::Error::from(UnknownAuthenticatedKind { kind: 200 })
        .context("assignment listener: read frame");
    assert_eq!(
        handshake_warn_class(&wrapped),
        HandshakeWarnClass::Diagnostic
    );
}

/// The router's dropped result frames and the pool's capacity refusals have
/// their own windows (#2712 item 3): a flood of either leaves the
/// diagnostic budget untouched.
#[test]
fn router_drops_do_not_suppress_diagnostic_warnings() {
    let budget = budget();
    let now = Instant::now();
    let mut drops_warned = 0;
    for i in 0..50u64 {
        let at = now + Duration::from_millis(i * 10);
        if budget.allow_at(HandshakeWarnClass::RouterDrop, at).warn {
            drops_warned += 1;
        }
    }
    assert_eq!(drops_warned, HANDSHAKE_WARN_PER_WINDOW);
    for i in 0..50u64 {
        let at = now + Duration::from_millis(500 + i * 10);
        budget.allow_at(HandshakeWarnClass::AtCapacity, at);
    }
    let decision = budget.allow_at(HandshakeWarnClass::Diagnostic, now + Duration::from_secs(1));
    assert_eq!(
        decision,
        WarnDecision {
            warn: true,
            suppressed_in_previous_window: 0,
            previous_window_ended_ago: Duration::ZERO,
        }
    );
    // And neither class is suppressed by the other's exhausted window.
    assert!(
        !budget
            .allow_at(HandshakeWarnClass::RouterDrop, now + Duration::from_secs(2))
            .warn
    );
    assert!(
        !budget
            .allow_at(HandshakeWarnClass::AtCapacity, now + Duration::from_secs(2))
            .warn
    );
}

/// The rollover summary is written by the next event of its class, which
/// may arrive long after the window ended: the decision carries the age of
/// that window so the line can say when it ended instead of "in the last
/// 60s" (#2712 item 3).
#[test]
fn rollover_summary_reports_the_window_age() {
    let budget = budget();
    let now = Instant::now();
    for i in 0..8u64 {
        budget.allow_at(HandshakeWarnClass::Diagnostic, now + Duration::from_secs(i));
    }
    let later = budget.allow_at(
        HandshakeWarnClass::Diagnostic,
        now + Duration::from_secs(3600),
    );
    assert_eq!(
        later,
        WarnDecision {
            warn: true,
            suppressed_in_previous_window: 3,
            previous_window_ended_ago: Duration::from_secs(3540),
        }
    );
}
