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

//! [`PolicyResolver`] tests (issue #2644) and the DNS lookup bound (issue #2728).
//!
//! These resolve `localhost`, which every test host maps to loopback through
//! its hosts file, or use an injected resolver, so they need no network.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use reqwest::dns::Resolve;
use tokio::sync::Semaphore;
use tokio::time::Instant;

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
async fn a_trailing_dot_name_is_judged_by_what_it_resolves_to() {
    // `localhost.` is a different string from `localhost` but the same name to
    // the system resolver. The policy filters the resolved addresses, not the
    // spelling, so it is refused whether it resolves to loopback or not at all.
    let err = resolve_allowed_addresses("localhost.", false)
        .await
        .expect_err("localhost. is never a public name");
    assert!(
        matches!(
            err,
            MediaUrlError::BlockedAddress { .. } | MediaUrlError::Unresolvable { .. }
        ),
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

fn sock(ip: &str) -> SocketAddr {
    SocketAddr::new(ip.parse().expect("ip literal"), 0)
}

#[test]
fn filter_keeps_only_public_answers_from_a_mixed_set() {
    let kept = filter_allowed_addresses(
        "mixed.example",
        vec![sock("10.0.0.5"), sock("93.184.216.34"), sock("127.0.0.1")],
        false,
    )
    .expect("one public answer");
    assert_eq!(kept, vec![sock("93.184.216.34")]);
}

#[test]
fn filter_refuses_an_all_private_answer_set() {
    let err = filter_allowed_addresses("p.example", vec![sock("10.0.0.5"), sock("::1")], false)
        .expect_err("no public answer");
    assert!(
        matches!(err, MediaUrlError::BlockedAddress { ref host, address }
            if host == "p.example" && address == sock("10.0.0.5").ip()),
        "{err:?}"
    );
}

#[test]
fn filter_reports_an_empty_answer_as_unresolvable() {
    for allow_private in [false, true] {
        let err = filter_allowed_addresses("none.example", Vec::new(), allow_private)
            .expect_err("empty answer");
        assert!(matches!(err, MediaUrlError::Unresolvable { .. }), "{err:?}");
    }
}

#[test]
fn filter_with_the_opt_in_keeps_everything() {
    let all = vec![sock("10.0.0.5"), sock("93.184.216.34")];
    let kept = filter_allowed_addresses("x.example", all.clone(), true).expect("opt-in");
    assert_eq!(kept, all);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dns_lookups_are_bounded_by_the_permit_pool() {
    // A lookup whose caller timed out keeps its blocking thread until the
    // resolver returns, so the bound has to count those threads too.
    const PERMITS: usize = 4;
    const LOOKUPS: usize = 100;
    const SLOW: Duration = Duration::from_secs(2);
    const TIMEOUT: Duration = Duration::from_millis(200);
    // Slack on top of the timeout for a loaded test host. It stays well under
    // `SLOW`, so a lookup that waited for the resolver would still fail it.
    const SLACK: Duration = Duration::from_secs(1);

    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let slow_resolver = {
        let active = active.clone();
        let peak = peak.clone();
        move |_host: &str, port: u16| -> std::io::Result<Vec<SocketAddr>> {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(SLOW);
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(vec![SocketAddr::from(([93, 184, 216, 34], port))])
        }
    };
    let permits = Arc::new(Semaphore::new(PERMITS));

    let lookups: Vec<_> = (0..LOOKUPS)
        .map(|index| {
            let resolve = slow_resolver.clone();
            let permits = permits.clone();
            tokio::spawn(async move {
                let host = format!("slow{index}.test");
                let started = Instant::now();
                let outcome = lookup_bounded_with(resolve, &host, 443, TIMEOUT, permits).await;
                (host, outcome, started.elapsed())
            })
        })
        .collect();
    for lookup in lookups {
        let (host, outcome, elapsed) = lookup.await.expect("lookup task");
        assert!(
            matches!(&outcome, Err(MediaUrlError::Unresolvable { host: h }) if *h == host),
            "{host}: {outcome:?}"
        );
        assert!(elapsed <= TIMEOUT + SLACK, "{host} took {elapsed:?}");
    }
    assert!(
        peak.load(Ordering::SeqCst) <= PERMITS,
        "{} lookups occupied blocking threads at once",
        peak.load(Ordering::SeqCst)
    );

    // The abandoned lookups hand their permits back once the resolver returns.
    let drained = tokio::time::timeout(Duration::from_secs(15), async {
        while permits.available_permits() < PERMITS || active.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        drained.is_ok(),
        "the slow lookups never released their permits"
    );
    let addrs = lookup_bounded_with(slow_resolver, "late.test", 443, SLOW * 5, permits)
        .await
        .expect("a lookup after the slow ones finish resolves");
    assert_eq!(addrs, vec![SocketAddr::from(([93, 184, 216, 34], 443))]);
    assert!(peak.load(Ordering::SeqCst) <= PERMITS);
}
