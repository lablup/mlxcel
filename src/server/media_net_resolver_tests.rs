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

//! [`PolicyResolver`] tests (issue #2644).
//!
//! These resolve `localhost`, which every test host maps to loopback through
//! its hosts file, so they need no network.

use reqwest::dns::Resolve;

use super::*;

#[tokio::test]
async fn a_name_that_resolves_only_to_loopback_is_refused() {
    let err = resolve_allowed_addresses("localhost", false)
        .await
        .expect_err("localhost resolves only to loopback");
    assert!(
        matches!(err, MediaUrlError::BlockedAddress { ref host, address }
            if host == "localhost" && address.is_loopback()),
        "{err:?}"
    );
}

#[tokio::test]
async fn the_opt_in_keeps_every_resolved_address() {
    let addrs = resolve_allowed_addresses("localhost", true)
        .await
        .expect("localhost resolves");
    assert!(!addrs.is_empty());
    assert!(
        addrs.iter().all(|addr| addr.ip().is_loopback()),
        "{addrs:?}"
    );
}

#[tokio::test]
async fn the_client_resolver_reads_the_opt_in_on_every_resolution() {
    // The media client is built once per process, so the resolver must follow
    // the opt-in as it is now, not as it was when the client was built.
    let resolver = PolicyResolver;
    {
        let _private = deny_private_media_urls_in_tests();
        let err = match resolver
            .resolve("localhost".parse().expect("name parses"))
            .await
        {
            Ok(addrs) => panic!("refused name resolved to {:?}", addrs.collect::<Vec<_>>()),
            Err(err) => err,
        };
        assert!(
            matches!(
                err.downcast_ref::<MediaUrlError>(),
                Some(MediaUrlError::BlockedAddress { .. })
            ),
            "{err:?}"
        );
    }
    let _private = allow_private_media_urls_in_tests();
    let addrs: Vec<SocketAddr> = resolver
        .resolve("localhost".parse().expect("name parses"))
        .await
        .expect("the opt-in admits loopback")
        .collect();
    assert!(!addrs.is_empty());
}
