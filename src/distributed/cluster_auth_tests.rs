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

use super::*;

fn temp_secret_file(name: &str, contents: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mlxcel-{name}-{}", uuid::Uuid::new_v4()));
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn secret_from_bytes_trims_one_trailing_newline_and_enforces_minimum() {
    let secret = ClusterSecret::from_bytes(b"0123456789abcdef\n".to_vec()).unwrap();
    assert_eq!(secret.as_bytes(), b"0123456789abcdef");
    let secret = ClusterSecret::from_bytes(b"0123456789abcdef\r\n".to_vec()).unwrap();
    assert_eq!(secret.as_bytes(), b"0123456789abcdef");
    let err = ClusterSecret::from_bytes(b"short\n".to_vec()).unwrap_err();
    assert!(err.to_string().contains("too short"), "{err}");
    let err = ClusterSecret::from_bytes(b"".to_vec()).unwrap_err();
    assert!(err.to_string().contains("too short"), "{err}");
}

#[test]
fn secret_debug_is_redacted() {
    let secret = ClusterSecret::from_bytes(b"0123456789abcdef".to_vec()).unwrap();
    let debug = format!("{secret:?}");
    assert!(!debug.contains("0123456789abcdef"));
    assert!(debug.contains("redacted"));
    let auth = ClusterAuth::new(&secret);
    assert!(format!("{auth:?}").contains("redacted"));
    // The sources struct carries the raw `MLXCEL_CLUSTER_SECRET` value.
    let sources = ClusterSecretSources {
        env_value: Some("0123456789abcdef-env-value".to_string()),
        ..Default::default()
    };
    let debug = format!("{sources:?}");
    assert!(!debug.contains("0123456789abcdef-env-value"), "{debug}");
    assert!(debug.contains("redacted"), "{debug}");
}

#[test]
fn hmac_sign_verify_round_trip_and_domain_separation() {
    let secret = ClusterSecret::random();
    let auth = ClusterAuth::new(&secret);
    let tag = auth.sign(b"label-a", b"message");
    assert!(auth.verify(b"label-a", b"message", &tag));
    assert!(!auth.verify(b"label-b", b"message", &tag));
    assert!(!auth.verify(b"label-a", b"messagf", &tag));
    assert!(!auth.verify(b"label-a", b"message", &tag[..31]));
    let other = ClusterAuth::new(&ClusterSecret::random());
    assert!(!other.verify(b"label-a", b"message", &tag));
}

#[test]
fn random_secrets_and_nonces_differ() {
    let a = ClusterSecret::random();
    let b = ClusterSecret::random();
    assert_ne!(a, b);
    assert_eq!(a.to_hex().len(), 64);
    let auth = ClusterAuth::new(&a);
    assert_ne!(auth.random_nonce(), auth.random_nonce());
    assert_ne!(auth.random_u64(), auth.random_u64());
}

#[test]
fn sources_resolve_cli_file_first() {
    let cli = temp_secret_file("cli", b"cli-secret-0123456789\n");
    let env_file = temp_secret_file("envfile", b"env-file-secret-0123456789");
    let sources = ClusterSecretSources {
        cli_file: Some(cli.clone()),
        env_file: Some(env_file.clone()),
        env_value: Some("env-value-secret-0123456789".to_string()),
        toml_file: None,
    };
    let secret = sources.resolve().unwrap().unwrap();
    assert_eq!(secret.as_bytes(), b"cli-secret-0123456789");
    assert_eq!(sources.file_path(), Some(cli.as_path()));
    let _ = std::fs::remove_file(cli);
    let _ = std::fs::remove_file(env_file);
}

#[test]
fn sources_reject_both_env_variables_set() {
    let sources = ClusterSecretSources {
        cli_file: None,
        env_file: Some(PathBuf::from("/nonexistent/secret")),
        env_value: Some("env-value-secret-0123456789".to_string()),
        toml_file: None,
    };
    let err = sources.resolve().unwrap_err().to_string();
    assert!(err.contains(CLUSTER_SECRET_FILE_ENV), "{err}");
    assert!(err.contains(CLUSTER_SECRET_ENV), "{err}");
}

#[test]
fn sources_fall_through_env_value_then_toml_then_none() {
    let sources = ClusterSecretSources {
        env_value: Some("env-value-secret-0123456789".to_string()),
        ..Default::default()
    };
    let secret = sources.resolve().unwrap().unwrap();
    assert_eq!(secret.as_bytes(), b"env-value-secret-0123456789");

    let toml = temp_secret_file("toml", b"toml-secret-0123456789\n");
    let sources = ClusterSecretSources {
        toml_file: Some(toml.clone()),
        ..Default::default()
    };
    let secret = sources.resolve().unwrap().unwrap();
    assert_eq!(secret.as_bytes(), b"toml-secret-0123456789");
    let _ = std::fs::remove_file(toml);

    assert!(ClusterSecretSources::default().resolve().unwrap().is_none());
}

#[test]
fn sources_report_missing_file_with_source_context() {
    let sources = ClusterSecretSources {
        cli_file: Some(PathBuf::from("/nonexistent/mlxcel/cluster.secret")),
        ..Default::default()
    };
    let err = format!("{:#}", sources.resolve().unwrap_err());
    assert!(err.contains(CLUSTER_SECRET_FILE_FLAG), "{err}");
    assert!(err.contains("/nonexistent/mlxcel/cluster.secret"), "{err}");
}

#[test]
fn sources_from_env_reads_process_environment() {
    let _guard = crate::test_support::env_lock::env_lock();
    // SAFETY: the env lock serializes every test that mutates the process
    // environment, and both variables are restored below.
    unsafe {
        std::env::set_var(CLUSTER_SECRET_ENV, "process-env-secret-0123456789");
        std::env::remove_var(CLUSTER_SECRET_FILE_ENV);
    }
    let sources = ClusterSecretSources::from_env(None, Some(Path::new("/tmp/x")));
    assert_eq!(
        sources.env_value.as_deref(),
        Some("process-env-secret-0123456789")
    );
    assert!(sources.env_file.is_none());
    assert_eq!(sources.toml_file.as_deref(), Some(Path::new("/tmp/x")));
    // SAFETY: still under the env lock acquired above.
    unsafe {
        std::env::remove_var(CLUSTER_SECRET_ENV);
    }
}
