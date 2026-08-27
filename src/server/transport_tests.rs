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

//! Unit tests for the b10621 transport value resolution (#1432).

use std::path::PathBuf;
use std::time::Duration;

use super::{
    DEFAULT_HTTP_TIMEOUT_SECS, HttpTimeouts, ListenTarget, resolve_api_prefix,
    resolve_http_threads, resolve_listen_target, resolve_sse_ping_interval,
    timeout_migration_warning,
};

#[test]
fn host_ending_in_sock_is_a_unix_socket_at_any_port() {
    for port in [0u16, 8080] {
        let resolved = resolve_listen_target("/tmp/mlxcel.sock", port).expect("resolves");
        assert_eq!(
            resolved.target,
            ListenTarget::Unix(PathBuf::from("/tmp/mlxcel.sock")),
            "a .sock host is a socket whatever --port says (b10621 rule)"
        );
        assert!(resolved.legacy_socket_warning.is_none());
    }
}

#[test]
fn legacy_port_zero_socket_path_still_binds_a_socket_with_a_warning() {
    let resolved = resolve_listen_target("/tmp/legacy-mlxcel", 0).expect("resolves");
    assert_eq!(
        resolved.target,
        ListenTarget::Unix(PathBuf::from("/tmp/legacy-mlxcel"))
    );
    let warning = resolved
        .legacy_socket_warning
        .expect("the legacy spelling must warn");
    assert!(warning.contains(".sock"), "{warning}");
}

#[test]
fn port_zero_on_a_normal_host_is_an_ephemeral_tcp_bind() {
    let resolved = resolve_listen_target("127.0.0.1", 0).expect("resolves");
    assert_eq!(
        resolved.target,
        ListenTarget::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        },
        "b10621 reads --port 0 as bind_to_any_port"
    );
    assert!(resolved.legacy_socket_warning.is_none());
}

#[test]
fn tcp_is_the_default_resolution() {
    let resolved = resolve_listen_target("0.0.0.0", 8080).expect("resolves");
    assert_eq!(
        resolved.target,
        ListenTarget::Tcp {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    );
}

#[test]
fn empty_host_is_rejected() {
    let err = resolve_listen_target("   ", 8080).expect_err("empty host must fail");
    assert!(err.to_string().contains("--host"), "{err}");
}

#[test]
fn http_timeout_keeps_zero_rather_than_treating_it_as_disabled() {
    assert_eq!(
        HttpTimeouts::from_secs(0),
        HttpTimeouts {
            read: Duration::ZERO,
            write: Duration::ZERO
        },
        "b10621 passes 0 through to select(), so mlxcel must not reinterpret it"
    );
}

#[test]
fn http_timeout_default_is_the_b10621_hour() {
    assert_eq!(DEFAULT_HTTP_TIMEOUT_SECS, 3600);
    assert_eq!(
        HttpTimeouts::default(),
        HttpTimeouts::from_secs(3600),
        "read and write share one --timeout value upstream"
    );
}

#[test]
fn api_prefix_default_is_empty() {
    assert_eq!(resolve_api_prefix("").expect("empty is the default"), "");
    assert_eq!(resolve_api_prefix("   ").expect("blank is the default"), "");
}

#[test]
fn api_prefix_accepts_a_leading_slash_without_a_trailing_one() {
    assert_eq!(resolve_api_prefix("/llama").expect("valid"), "/llama");
    assert_eq!(
        resolve_api_prefix("/a/b").expect("nested prefixes are valid"),
        "/a/b"
    );
}

#[test]
fn api_prefix_rejects_the_shapes_that_would_404_every_route() {
    for bad in ["llama", "/llama/", "/", "//llama", "/llama?x=1", "/llama#f"] {
        let err = resolve_api_prefix(bad)
            .err()
            .unwrap_or_else(|| panic!("{bad:?} must be rejected"));
        assert!(
            err.to_string().contains("--api-prefix"),
            "the diagnostic must name the flag: {err}"
        );
    }
}

#[test]
fn api_prefix_rejects_spaces_and_control_characters() {
    assert!(resolve_api_prefix("/lla ma").is_err());
    assert!(resolve_api_prefix("/lla\nma").is_err());
}

#[test]
fn threads_http_honors_an_explicit_positive_value() {
    assert_eq!(resolve_http_threads(1, 4), 1);
    assert_eq!(resolve_http_threads(7, 4), 7);
}

#[test]
fn threads_http_auto_reserves_four_threads_above_the_slot_count() {
    // b10621: max(n_parallel + 4, hardware_concurrency() - 1), never zero.
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let resolved = resolve_http_threads(-1, 64);
    assert_eq!(
        resolved,
        68.max(available.saturating_sub(1)).max(1),
        "a large --parallel must dominate the hardware term"
    );
    assert!(resolve_http_threads(-1, 1) >= 1, "never resolves to zero");
    assert_eq!(
        resolve_http_threads(0, 4),
        resolve_http_threads(-1, 4),
        "any value below 1 selects the automatic sizing"
    );
}

#[test]
fn sse_ping_interval_negative_disables_pings() {
    assert_eq!(resolve_sse_ping_interval(-1).expect("valid"), None);
    assert_eq!(resolve_sse_ping_interval(-99).expect("valid"), None);
}

#[test]
fn sse_ping_interval_positive_is_a_duration() {
    assert_eq!(
        resolve_sse_ping_interval(30).expect("valid"),
        Some(Duration::from_secs(30))
    );
}

#[test]
fn sse_ping_interval_zero_is_rejected() {
    let err = resolve_sse_ping_interval(0).expect_err("0 must be rejected");
    assert!(err.to_string().contains("-1 to disable"), "{err}");
}

#[test]
fn timeout_migration_warning_fires_only_on_the_ambiguous_command_line() {
    assert!(
        timeout_migration_warning(true, false).is_some(),
        "--timeout alone is the command line whose meaning changed"
    );
    assert!(
        timeout_migration_warning(true, true).is_none(),
        "naming both controls is unambiguous"
    );
    assert!(
        timeout_migration_warning(false, false).is_none(),
        "defaults are not a migration"
    );
    assert!(timeout_migration_warning(false, true).is_none());
}

#[test]
fn timeout_migration_warning_names_both_spellings_and_defaults() {
    let warning = timeout_migration_warning(true, false).expect("warns");
    assert!(warning.contains("--decode-timeout"), "{warning}");
    assert!(warning.contains("MLXCEL_DECODE_TIMEOUT"), "{warning}");
    assert!(warning.contains("3600"), "{warning}");
    assert!(warning.contains("600"), "{warning}");
}
