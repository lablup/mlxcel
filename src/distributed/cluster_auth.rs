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

//! Pre-shared per-cluster secret and the HMAC primitives built on it (#2637).
//!
//! Every distributed pipeline role (coordinator, remote stage, zero-config
//! peer) shares one operator-supplied secret. It keys three things:
//!
//! - the connect-time mutual handshake in the TCP transport core
//!   ([`super::cluster_handshake`]),
//! - the MAC on UDP discovery beacons ([`super::discovery_beacon`]),
//! - the per-exchange nonces the coordinator uses to correlate stage replies.
//!
//! The secret never appears in `Debug` or `Display` output and is never
//! logged. Loading goes through [`ClusterSecretSources`] so the precedence
//! (CLI file flag, then `MLXCEL_CLUSTER_SECRET_FILE`, then
//! `MLXCEL_CLUSTER_SECRET`, then the cluster TOML `secret_file`) lives in one
//! place and is unit-testable without touching the process environment.
//!
//! Used by: tcp_transport, connection_pool, cluster_init, pipeline runtime,
//! pipeline remote_service, server startup.

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ring::rand::{SecureRandom, SystemRandom};
use ring::{hkdf, hmac};

/// Minimum accepted secret length in bytes, after trimming one trailing
/// newline. 16 bytes (128 bits) is the floor for an HMAC key that an attacker
/// cannot brute-force online; `openssl rand -hex 32` produces 64.
pub const MIN_CLUSTER_SECRET_BYTES: usize = 16;

/// Environment variable carrying the secret value itself.
pub const CLUSTER_SECRET_ENV: &str = "MLXCEL_CLUSTER_SECRET";
/// Environment variable carrying a path to a file holding the secret.
pub const CLUSTER_SECRET_FILE_ENV: &str = "MLXCEL_CLUSTER_SECRET_FILE";
/// CLI flag carrying a path to a file holding the secret.
pub const CLUSTER_SECRET_FILE_FLAG: &str = "--cluster-secret-file";

/// Operator-facing summary of every place the secret may come from, used in
/// the fail-closed startup error so the message names the flag and env vars.
pub const CLUSTER_SECRET_SOURCES_HELP: &str = "pass --cluster-secret-file <PATH>, set \
     MLXCEL_CLUSTER_SECRET_FILE=<PATH> or MLXCEL_CLUSTER_SECRET=<VALUE>, or add \
     `secret_file = \"<PATH>\"` under [cluster] in the cluster TOML \
     (generate one with `openssl rand -hex 32 > cluster.secret`)";

/// The raw pre-shared secret. Redacted in `Debug`; no `Display`.
#[derive(Clone, PartialEq, Eq)]
pub struct ClusterSecret(Vec<u8>);

impl ClusterSecret {
    /// Build a secret from raw bytes.
    ///
    /// Trims exactly one trailing `\n` or `\r\n` (the common artefact of
    /// `echo` or an editor writing the secret file) and rejects anything
    /// shorter than [`MIN_CLUSTER_SECRET_BYTES`] afterwards.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let mut bytes: Vec<u8> = bytes.into();
        if bytes.ends_with(b"\r\n") {
            bytes.truncate(bytes.len() - 2);
        } else if bytes.ends_with(b"\n") {
            bytes.truncate(bytes.len() - 1);
        }
        if bytes.len() < MIN_CLUSTER_SECRET_BYTES {
            bail!(
                "cluster secret is too short ({} bytes after trimming one trailing newline); \
                 it must be at least {MIN_CLUSTER_SECRET_BYTES} bytes, for example \
                 `openssl rand -hex 32`",
                bytes.len()
            );
        }
        Ok(Self(bytes))
    }

    /// Read the secret from a file. On unix, warns (once per load) when the
    /// file is group- or world-readable; the secret is still accepted so an
    /// operator with a deliberately shared file is not locked out.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read cluster secret file {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(path) {
                let mode = meta.permissions().mode();
                if mode & 0o077 != 0 {
                    tracing::warn!(
                        "cluster secret file {} is readable by group or others (mode {:o}); \
                         run `chmod 600` on it",
                        path.display(),
                        mode & 0o777
                    );
                }
            }
        }
        Self::from_bytes(bytes)
            .with_context(|| format!("invalid cluster secret in {}", path.display()))
    }

    /// A fresh 32-byte random secret. Test harnesses and CI use this so no
    /// real secret is ever committed; every process of one logical cluster
    /// must still share the same value.
    pub fn random() -> Self {
        let mut bytes = vec![0u8; 32];
        SystemRandom::new()
            .fill(&mut bytes)
            .expect("system CSPRNG unavailable");
        Self(bytes)
    }

    /// Hex encoding, for harnesses that hand the secret to a child process
    /// through `MLXCEL_CLUSTER_SECRET` or a temp file. Never log the result.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for ClusterSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ClusterSecret(<redacted>)")
    }
}

impl Drop for ClusterSecret {
    fn drop(&mut self) {
        // Best-effort scrub; not a guarantee against copies the allocator or
        // the optimizer may have made, but it keeps the key out of freed heap
        // in the common case.
        for b in self.0.iter_mut() {
            // SAFETY: `b` is a valid, exclusively borrowed `u8` inside the
            // vector we own; a volatile write of zero through it is sound
            // and only prevents the compiler from eliding the scrub.
            unsafe { std::ptr::write_volatile(b, 0) };
        }
    }
}

/// Where a secret may be supplied. Collected once at startup and resolved by
/// [`ClusterSecretSources::resolve`], in this precedence order:
///
/// 1. `--cluster-secret-file <PATH>` ([`Self::cli_file`])
/// 2. `MLXCEL_CLUSTER_SECRET_FILE=<PATH>` ([`Self::env_file`])
/// 3. `MLXCEL_CLUSTER_SECRET=<VALUE>` ([`Self::env_value`])
/// 4. `[cluster] secret_file = "<PATH>"` in the cluster TOML
///    ([`Self::toml_file`], already resolved against the TOML's directory)
///
/// Setting both environment variables is a configuration error rather than
/// a silent pick, because the two can disagree without anyone noticing.
///
/// `Debug` redacts [`Self::env_value`], which holds the secret itself.
#[derive(Default, Clone)]
pub struct ClusterSecretSources {
    pub cli_file: Option<PathBuf>,
    pub env_file: Option<PathBuf>,
    pub env_value: Option<String>,
    pub toml_file: Option<PathBuf>,
}

impl fmt::Debug for ClusterSecretSources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClusterSecretSources")
            .field("cli_file", &self.cli_file)
            .field("env_file", &self.env_file)
            .field("env_value", &self.env_value.as_ref().map(|_| "<redacted>"))
            .field("toml_file", &self.toml_file)
            .finish()
    }
}

impl ClusterSecretSources {
    /// Gather the sources from the CLI flag, the process environment, and
    /// the cluster TOML (if one was loaded).
    pub fn from_env(cli_file: Option<&Path>, toml_file: Option<&Path>) -> Self {
        let env_file = std::env::var_os(CLUSTER_SECRET_FILE_ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let env_value = std::env::var(CLUSTER_SECRET_ENV)
            .ok()
            .filter(|v| !v.is_empty());
        Self {
            cli_file: cli_file.map(Path::to_path_buf),
            env_file,
            env_value,
            toml_file: toml_file.map(Path::to_path_buf),
        }
    }

    /// The highest-precedence source that is set, or `None` when no secret
    /// was supplied anywhere.
    pub fn resolve(&self) -> Result<Option<ClusterSecret>> {
        if let Some(path) = &self.cli_file {
            return ClusterSecret::from_file(path)
                .with_context(|| format!("{CLUSTER_SECRET_FILE_FLAG} {}", path.display()))
                .map(Some);
        }
        if self.env_file.is_some() && self.env_value.is_some() {
            bail!(
                "both {CLUSTER_SECRET_FILE_ENV} and {CLUSTER_SECRET_ENV} are set; keep only one \
                 so the cluster secret has a single unambiguous source"
            );
        }
        if let Some(path) = &self.env_file {
            return ClusterSecret::from_file(path)
                .with_context(|| format!("{CLUSTER_SECRET_FILE_ENV}={}", path.display()))
                .map(Some);
        }
        if let Some(value) = &self.env_value {
            return ClusterSecret::from_bytes(value.as_bytes())
                .with_context(|| format!("{CLUSTER_SECRET_ENV} is set but invalid"))
                .map(Some);
        }
        if let Some(path) = &self.toml_file {
            return ClusterSecret::from_file(path)
                .with_context(|| format!("cluster TOML secret_file = {}", path.display()))
                .map(Some);
        }
        Ok(None)
    }

    /// The file path the secret came from, if it came from a file. The
    /// zero-config coordinator records it in the generated TOML so peers that
    /// copy the file keep the same `secret_file` reference.
    pub fn file_path(&self) -> Option<&Path> {
        self.cli_file
            .as_deref()
            .or(self.env_file.as_deref())
            .or(self.toml_file.as_deref())
    }
}

/// HMAC-SHA256 keyed by the cluster secret, plus a CSPRNG for nonces.
///
/// Shared (`Arc`) between the transport, the discovery receiver, and the
/// pipeline runtime of one process. `Debug` prints no key material.
pub struct ClusterAuth {
    key: hmac::Key,
    /// Kept for HKDF extraction of per-connection frame keys (#2656); never
    /// read for any other purpose and scrubbed on drop with the secret.
    secret: ClusterSecret,
    rng: SystemRandom,
}

impl ClusterAuth {
    pub fn new(secret: &ClusterSecret) -> Self {
        Self {
            key: hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes()),
            secret: secret.clone(),
            rng: SystemRandom::new(),
        }
    }

    /// HKDF-SHA256 extract with the cluster secret as IKM. The per-connection
    /// frame keys expand from the returned PRK ([`super::frame_protection`]).
    pub fn hkdf_extract(&self, salt: &[u8]) -> hkdf::Prk {
        hkdf::Salt::new(hkdf::HKDF_SHA256, salt).extract(self.secret.as_bytes())
    }

    /// `HMAC(K, label || 0x00 || message)`. The label domain-separates the
    /// handshake directions and the beacon MAC so a tag minted for one
    /// purpose can never verify for another.
    pub fn sign(&self, label: &[u8], message: &[u8]) -> [u8; 32] {
        let mut ctx = hmac::Context::with_key(&self.key);
        ctx.update(label);
        ctx.update(&[0u8]);
        ctx.update(message);
        let tag = ctx.sign();
        let mut out = [0u8; 32];
        out.copy_from_slice(tag.as_ref());
        out
    }

    /// Constant-time verification of a tag produced by [`Self::sign`].
    pub fn verify(&self, label: &[u8], message: &[u8], tag: &[u8]) -> bool {
        if tag.len() != 32 {
            return false;
        }
        let mut data = Vec::with_capacity(label.len() + 1 + message.len());
        data.extend_from_slice(label);
        data.push(0u8);
        data.extend_from_slice(message);
        hmac::verify(&self.key, &data, tag).is_ok()
    }

    /// 32 random bytes from the system CSPRNG.
    pub fn random_nonce(&self) -> [u8; 32] {
        let mut nonce = [0u8; 32];
        self.rng
            .fill(&mut nonce)
            .expect("system CSPRNG unavailable");
        nonce
    }

    /// A random 64-bit value for per-exchange correlation.
    pub fn random_u64(&self) -> u64 {
        let mut bytes = [0u8; 8];
        self.rng
            .fill(&mut bytes)
            .expect("system CSPRNG unavailable");
        u64::from_be_bytes(bytes)
    }
}

impl fmt::Debug for ClusterAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ClusterAuth(<redacted>)")
    }
}

#[cfg(test)]
#[path = "cluster_auth_tests.rs"]
mod tests;
