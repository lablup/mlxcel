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

//! Unit tests for `--ssl-cert-file` / `--ssl-key-file` handling (#1432).
//!
//! The keypair is generated per test rather than checked in, so the
//! repository carries no private key and the material cannot expire.

use std::path::PathBuf;

use super::{TlsPaths, build_server_config, resolve_tls_paths};

/// Write a fresh self-signed certificate and its key into `dir`, returning
/// their paths.
pub(crate) fn write_self_signed(dir: &std::path::Path) -> TlsPaths {
    let issued = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("rcgen issues a self-signed certificate");
    let cert = dir.join("cert.pem");
    let key = dir.join("key.pem");
    std::fs::write(&cert, issued.cert.pem()).expect("write cert");
    std::fs::write(&key, issued.key_pair.serialize_pem()).expect("write key");
    TlsPaths { cert, key }
}

#[test]
fn neither_flag_means_plaintext_http() {
    assert_eq!(resolve_tls_paths(None, None).expect("valid"), None);
}

#[test]
fn both_flags_pair_into_tls_paths() {
    let paths = resolve_tls_paths(
        Some(PathBuf::from("/tmp/c.pem")),
        Some(PathBuf::from("/tmp/k.pem")),
    )
    .expect("valid")
    .expect("TLS enabled");
    assert_eq!(paths.cert, PathBuf::from("/tmp/c.pem"));
    assert_eq!(paths.key, PathBuf::from("/tmp/k.pem"));
}

#[test]
fn a_certificate_without_a_key_is_refused_rather_than_served_in_plaintext() {
    let err = resolve_tls_paths(Some(PathBuf::from("/tmp/c.pem")), None)
        .expect_err("half a TLS configuration must not start");
    assert!(err.to_string().contains("--ssl-key-file"), "{err}");
}

#[test]
fn a_key_without_a_certificate_is_refused() {
    let err = resolve_tls_paths(None, Some(PathBuf::from("/tmp/k.pem")))
        .expect_err("half a TLS configuration must not start");
    assert!(err.to_string().contains("--ssl-cert-file"), "{err}");
}

#[test]
fn a_valid_keypair_builds_a_server_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = write_self_signed(dir.path());
    build_server_config(&paths).expect("a self-signed pair is accepted");
}

#[test]
fn a_missing_certificate_file_names_the_flag_and_the_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = write_self_signed(dir.path());
    let missing = TlsPaths {
        cert: dir.path().join("absent.pem"),
        key: paths.key,
    };
    let err = build_server_config(&missing).expect_err("a missing certificate must fail");
    let text = format!("{err:#}");
    assert!(text.contains("--ssl-cert-file"), "{text}");
    assert!(text.contains("absent.pem"), "{text}");
}

#[test]
fn a_missing_key_file_names_the_flag_and_the_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = write_self_signed(dir.path());
    let missing = TlsPaths {
        cert: paths.cert,
        key: dir.path().join("absent-key.pem"),
    };
    let err = build_server_config(&missing).expect_err("a missing key must fail");
    let text = format!("{err:#}");
    assert!(text.contains("--ssl-key-file"), "{text}");
    assert!(text.contains("absent-key.pem"), "{text}");
}

#[test]
fn a_file_with_no_certificate_block_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = write_self_signed(dir.path());
    let empty = dir.path().join("empty.pem");
    std::fs::write(&empty, "# not a certificate\n").expect("write");
    let err = build_server_config(&TlsPaths {
        cert: empty,
        key: paths.key,
    })
    .expect_err("an empty PEM must fail");
    assert!(
        format!("{err:#}").contains("no CERTIFICATE block"),
        "{err:#}"
    );
}

#[test]
fn a_file_with_no_private_key_block_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = write_self_signed(dir.path());
    let empty = dir.path().join("empty-key.pem");
    std::fs::write(&empty, "# not a key\n").expect("write");
    let err = build_server_config(&TlsPaths {
        cert: paths.cert,
        key: empty,
    })
    .expect_err("an empty PEM must fail");
    assert!(
        format!("{err:#}").contains("no PRIVATE KEY block"),
        "{err:#}"
    );
}

#[test]
fn a_certificate_that_does_not_match_the_key_is_rejected() {
    let a = tempfile::tempdir().expect("tempdir");
    let b = tempfile::tempdir().expect("tempdir");
    let first = write_self_signed(a.path());
    let second = write_self_signed(b.path());
    let err = build_server_config(&TlsPaths {
        cert: first.cert,
        key: second.key,
    })
    .expect_err("a mismatched pair must fail");
    let text = format!("{err:#}");
    assert!(text.contains("does not match"), "{text}");
}
