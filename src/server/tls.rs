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

//! TLS material for `--ssl-cert-file` / `--ssl-key-file` (#1432).
//!
//! `llama-server` b10621 constructs an `httplib::SSLServer` when both
//! `params.ssl_file_cert` and `params.ssl_file_key` are non-empty, and refuses
//! to start when the binary was built without SSL support. mlxcel always has
//! rustls available, so the only startup failures here are a half-configured
//! pair or unreadable / malformed PEM material, and both are reported with the
//! offending path named.
//!
//! The provider is pinned to `ring` explicitly rather than going through
//! `rustls`'s process-default provider. A process default is installed by
//! whichever dependency gets there first (mlxcel also links a rustls-backed
//! `reqwest` for model downloads), and `ServerConfig::builder()` panics rather
//! than erroring when the default is ambiguous. A server must not be able to
//! abort at bind time because of a link-order detail.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rustls_pki_types::pem::PemObject;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio_rustls::rustls::{ServerConfig as RustlsServerConfig, crypto::ring};

/// Paths to the PEM certificate chain and private key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsPaths {
    pub cert: PathBuf,
    pub key: PathBuf,
}

/// Pair `--ssl-cert-file` and `--ssl-key-file` into an optional [`TlsPaths`].
///
/// b10621 enables TLS only when both are set and silently serves plaintext
/// when exactly one is. mlxcel refuses instead: a deployment that set only the
/// certificate almost certainly meant to serve HTTPS, and starting in plaintext
/// on the same port is the failure mode that matters.
pub fn resolve_tls_paths(cert: Option<PathBuf>, key: Option<PathBuf>) -> Result<Option<TlsPaths>> {
    match (cert, key) {
        (None, None) => Ok(None),
        (Some(cert), Some(key)) => Ok(Some(TlsPaths { cert, key })),
        (Some(cert), None) => bail!(
            "--ssl-cert-file {} was given without --ssl-key-file. TLS needs both a certificate \
             and its private key; pass --ssl-key-file, or drop --ssl-cert-file to serve plaintext \
             HTTP",
            cert.display()
        ),
        (None, Some(key)) => bail!(
            "--ssl-key-file {} was given without --ssl-cert-file. TLS needs both a certificate \
             and its private key; pass --ssl-cert-file, or drop --ssl-key-file to serve plaintext \
             HTTP",
            key.display()
        ),
    }
}

/// Read a PEM certificate chain.
fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>> {
    let certs: Vec<_> = CertificateDer::pem_file_iter(path)
        .with_context(|| format!("--ssl-cert-file {}: cannot open", path.display()))?
        .collect::<std::result::Result<_, _>>()
        .with_context(|| {
            format!(
                "--ssl-cert-file {}: not a readable PEM certificate chain",
                path.display()
            )
        })?;
    if certs.is_empty() {
        bail!(
            "--ssl-cert-file {}: contains no CERTIFICATE block. Point it at a PEM-encoded \
             certificate chain",
            path.display()
        );
    }
    Ok(certs)
}

/// Read a PEM private key, accepting PKCS#8, PKCS#1 and SEC1 encodings.
///
/// cpp-httplib hands the file to OpenSSL, which accepts the same three, so
/// restricting to PKCS#8 here would reject keys b10621 serves with.
fn load_key(path: &Path) -> Result<PrivateKeyDer<'static>> {
    if !path.is_file() {
        bail!("--ssl-key-file {}: cannot open", path.display());
    }
    PrivateKeyDer::from_pem_file(path).map_err(|e| {
        anyhow::anyhow!(
            "--ssl-key-file {}: contains no PRIVATE KEY block ({e}). Point it at a PEM-encoded \
             PKCS#8, PKCS#1 or SEC1 private key",
            path.display()
        )
    })
}

/// Build the rustls server configuration from the configured PEM files.
///
/// Every failure names the flag and the path, because a TLS misconfiguration
/// surfaces at bind time with no request to attach the error to.
pub fn build_server_config(paths: &TlsPaths) -> Result<Arc<RustlsServerConfig>> {
    let certs = load_certs(&paths.cert)?;
    let key = load_key(&paths.key)?;
    let config = RustlsServerConfig::builder_with_provider(Arc::new(ring::default_provider()))
        .with_safe_default_protocol_versions()
        .context("rustls rejected the default protocol version set")?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .with_context(|| {
            format!(
                "the certificate in --ssl-cert-file {} does not match the key in \
                 --ssl-key-file {}",
                paths.cert.display(),
                paths.key.display()
            )
        })?;
    Ok(Arc::new(config))
}

#[cfg(test)]
#[path = "tls_tests.rs"]
pub(crate) mod tls_tests;
