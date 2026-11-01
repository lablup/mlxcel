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

//! Process-wide rustls crypto provider for `reqwest` clients.
//!
//! `reqwest` is built with `rustls-no-provider` so that `aws-lc-rs` (a C/cmake
//! build) stays out of the dependency tree; the server's own HTTPS listener
//! already pins the `ring` provider. With that feature `reqwest` panics while
//! building a client unless a process default [`CryptoProvider`] is installed,
//! TLS or not. Every `reqwest::Client` in this crate is therefore created
//! through [`client_builder`] (or after [`install_default_crypto_provider`]).
//!
//! [`CryptoProvider`]: tokio_rustls::rustls::crypto::CryptoProvider

use tokio_rustls::rustls::crypto::{CryptoProvider, ring};

/// Install `ring` as the process default rustls provider, if none is set.
///
/// Idempotent and thread-safe: later calls, and calls racing with another
/// installer, are no-ops. A provider installed by someone else is left alone.
pub fn install_default_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // `Err` only means another thread installed a provider first.
        let _ = ring::default_provider().install_default();
    }
}

/// A [`reqwest::ClientBuilder`] that is safe to `build()` in this process.
pub fn client_builder() -> reqwest::ClientBuilder {
    install_default_crypto_provider();
    reqwest::Client::builder()
}

/// Equivalent of [`reqwest::Client::new`] for a process using the `ring`
/// provider. Panics when the client cannot be built, like `Client::new`.
pub fn client() -> reqwest::Client {
    client_builder()
        .build()
        .expect("reqwest client builds with the ring provider installed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_is_idempotent_and_leaves_a_default_provider() {
        install_default_crypto_provider();
        install_default_crypto_provider();
        assert!(CryptoProvider::get_default().is_some());
    }

    #[test]
    fn client_builds_without_panicking_under_rustls_no_provider() {
        // reqwest panics here when no process default provider is installed.
        let _ = client();
        let _ = client_builder().build().expect("builder builds a client");
    }
}
