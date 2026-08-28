// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Address-policy tests for remote media URLs (issue #1451).
//!
//! Nothing here opens a socket or resolves a name: every case goes through the
//! synchronous shape check or the pure address predicate, so the suite is
//! deterministic on a machine with no network and cannot be steered by whatever
//! the local resolver happens to answer.

use super::*;

fn url(raw: &str) -> reqwest::Url {
    reqwest::Url::parse(raw).expect("test URL parses")
}

#[test]
fn loopback_and_unspecified_addresses_are_refused() {
    for addr in [
        "127.0.0.1",
        "127.4.5.6",
        "0.0.0.0",
        "::1",
        "::",
        "::ffff:127.0.0.1",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
    }
}

#[test]
fn the_cloud_metadata_addresses_are_refused() {
    // 169.254.169.254 is EC2 / GCE / Azure instance metadata, reachable from
    // inside the instance and the single highest-value SSRF target.
    for addr in ["169.254.169.254", "169.254.0.1", "fe80::1"] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
    }
}

#[test]
fn private_and_reserved_ranges_are_refused() {
    for addr in [
        "10.0.0.1",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.1.1",
        "100.64.0.1",
        "192.0.0.1",
        "192.0.2.1",
        "198.18.0.1",
        "198.51.100.1",
        "203.0.113.1",
        "240.0.0.1",
        "255.255.255.255",
        "224.0.0.1",
        "fc00::1",
        "fd12:3456::1",
        "2001:db8::1",
        "ff02::1",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(!is_public_address(parsed), "{addr} must be refused");
    }
}

#[test]
fn ordinary_public_addresses_are_allowed() {
    // Over-blocking is a compatibility bug, so the neighbours of every
    // reserved prefix are asserted too.
    for addr in [
        "8.8.8.8",
        "1.1.1.1",
        "172.15.0.1",
        "172.32.0.1",
        "100.63.255.255",
        "100.128.0.1",
        "192.0.1.1",
        "192.0.3.1",
        "198.17.255.255",
        "198.20.0.1",
        "198.51.99.1",
        "198.51.101.1",
        "203.0.112.1",
        "203.0.114.1",
        "8.8.4.4",
        "2606:4700::1111",
        "::ffff:8.8.8.8",
    ] {
        let parsed: IpAddr = addr.parse().expect("address parses");
        assert!(is_public_address(parsed), "{addr} must be allowed");
    }
}

#[test]
fn a_non_http_scheme_is_refused() {
    for raw in [
        "file:///etc/passwd",
        "ftp://example.com/x.png",
        "gopher://x/",
    ] {
        let parsed = reqwest::Url::parse(raw).expect("test URL parses");
        assert!(
            matches!(
                ensure_media_url_shape_with(&parsed, false),
                Err(MediaUrlError::Scheme { .. })
            ),
            "{raw} must be refused"
        );
    }
}

#[test]
fn credentials_in_a_media_url_are_refused() {
    assert!(matches!(
        ensure_media_url_shape_with(&url("http://user:pass@example.com/a.png"), false),
        Err(MediaUrlError::Credentials)
    ));
    assert!(matches!(
        ensure_media_url_shape_with(&url("http://user@example.com/a.png"), false),
        Err(MediaUrlError::Credentials)
    ));
}

#[test]
fn an_ip_literal_host_is_checked_without_resolution() {
    for raw in [
        "http://127.0.0.1:9090/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:8080/x.png",
        "http://10.1.2.3/internal.png",
        "http://[::ffff:127.0.0.1]/x.png",
    ] {
        let parsed = url(raw);
        assert!(
            matches!(
                ensure_media_url_shape_with(&parsed, false),
                Err(MediaUrlError::BlockedAddress { .. })
            ),
            "{raw} must be refused"
        );
    }
    assert!(ensure_media_url_shape_with(&url("https://8.8.8.8/a.png"), false).is_ok());
}

#[test]
fn a_public_hostname_passes_the_shape_check() {
    // The shape check is name-agnostic; resolution happens in the async half.
    assert!(ensure_media_url_shape_with(&url("https://example.com/a.png"), false).is_ok());
    assert!(ensure_media_url_shape_with(&url("http://example.com:8080/a.png"), false).is_ok());
}

#[test]
fn a_blocked_peer_address_is_reported_with_the_host_and_the_escape_hatch() {
    let addr: SocketAddr = "127.0.0.1:9090".parse().expect("socket address");
    let err = ensure_socket_addr_allowed_with("images.example.com", addr, false)
        .expect_err("loopback is refused");
    let rendered = err.to_string();
    assert!(rendered.contains("images.example.com"), "{rendered}");
    assert!(rendered.contains("127.0.0.1"), "{rendered}");
    assert!(
        rendered.contains(ALLOW_PRIVATE_MEDIA_URLS_ENV),
        "the operator has to be told how to opt in: {rendered}"
    );
}

#[test]
fn a_public_peer_address_passes() {
    let addr: SocketAddr = "93.184.216.34:443".parse().expect("socket address");
    assert!(ensure_socket_addr_allowed_with("example.com", addr, false).is_ok());
}
