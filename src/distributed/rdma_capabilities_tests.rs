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

//! Unit tests for the RDMA capability probe and protocol-version negotiation.

use super::{
    RDMA_PROTOCOL_VERSION, RdmaAcceleration, RdmaCapabilities, negotiate_protocol_version,
    os_family, probe_capabilities,
};

#[test]
fn capabilities_default_to_current_protocol_version() {
    let cap = RdmaCapabilities::accelerated(RdmaAcceleration::LinuxIoUring);
    assert_eq!(cap.protocol_version, RDMA_PROTOCOL_VERSION);
    assert!(cap.is_accelerated());
    assert!(cap.reason.is_none());
}

#[test]
fn tcp_fallback_carries_reason() {
    let cap = RdmaCapabilities::tcp_fallback("os=linux, driver=missing");
    assert!(!cap.is_accelerated());
    assert_eq!(cap.acceleration, RdmaAcceleration::TcpFallback);
    assert_eq!(cap.reason.as_deref(), Some("os=linux, driver=missing"));
}

#[test]
fn protocol_version_negotiation_succeeds_for_matching_peer() {
    assert_eq!(
        negotiate_protocol_version(RDMA_PROTOCOL_VERSION).unwrap(),
        RDMA_PROTOCOL_VERSION
    );
}

#[test]
fn protocol_version_negotiation_rejects_zero() {
    let err = negotiate_protocol_version(0).unwrap_err();
    assert!(err.contains("peer_version=0"));
    assert!(err.contains(os_family()));
}

#[test]
fn protocol_version_negotiation_rejects_mismatch() {
    let err = negotiate_protocol_version(RDMA_PROTOCOL_VERSION.wrapping_add(7)).unwrap_err();
    assert!(err.contains("peer_version="));
    assert!(err.contains("incompatible"));
}

#[test]
fn probe_returns_structured_result() {
    let cap = probe_capabilities();
    // Protocol version is always populated.
    assert_eq!(cap.protocol_version, RDMA_PROTOCOL_VERSION);
    match cap.acceleration {
        RdmaAcceleration::TcpFallback => {
            assert!(cap.reason.is_some(), "fallback must carry a reason");
            assert!(
                cap.reason.as_deref().unwrap().contains(os_family()),
                "fallback reason must name the OS family for observability"
            );
        }
        _ => {
            // Accelerated paths do not need a reason — but if present it must
            // be informative, not empty.
            if let Some(reason) = cap.reason.as_deref() {
                assert!(!reason.is_empty());
            }
        }
    }
}

#[test]
fn os_family_is_never_empty() {
    assert!(!os_family().is_empty());
}

#[test]
fn acceleration_display_strings_are_stable() {
    // These strings appear in logs and must stay stable for operators' grep
    // recipes and dashboards.
    assert_eq!(
        format!("{}", RdmaAcceleration::LinuxIoUring),
        "io_uring_registered"
    );
    assert_eq!(
        format!("{}", RdmaAcceleration::MacosKqueueRegistered),
        "kqueue_registered"
    );
    assert_eq!(format!("{}", RdmaAcceleration::TcpFallback), "tcp_fallback");
}
