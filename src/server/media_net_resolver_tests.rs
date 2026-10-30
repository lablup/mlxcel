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

//! [`PolicyResolver`] tests (issue #2644), the DNS lookup bound (issue #2728),
//! and its per-domain fairness, permit wait and pre-flight pin (issue #2764).
//!
//! These resolve `localhost`, which every test host maps to loopback through
//! its hosts file, or use an injected resolver, so they need no network.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use reqwest::dns::Resolve;
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
    const LIMITS: DnsLimits = DnsLimits {
        permit_wait: Duration::from_millis(200),
        lookup_timeout: Duration::from_millis(200),
        per_domain: 4,
    };
    // Slack on top of the two bounds for a loaded test host. It stays well
    // under `SLOW`, so a lookup that waited for the resolver would still fail.
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
    // Every host is its own domain key, so only the global pool binds here.
    let pool = Arc::new(DnsPool::new(PERMITS));

    let lookups: Vec<_> = (0..LOOKUPS)
        .map(|index| {
            let resolve = slow_resolver.clone();
            let pool = pool.clone();
            tokio::spawn(async move {
                let host = format!("slow{index}.test");
                let started = Instant::now();
                let outcome = lookup_bounded_with(resolve, &host, 443, LIMITS, &pool).await;
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
        assert!(
            elapsed <= LIMITS.permit_wait + LIMITS.lookup_timeout + SLACK,
            "{host} took {elapsed:?}"
        );
    }
    assert!(
        peak.load(Ordering::SeqCst) <= PERMITS,
        "{} lookups occupied blocking threads at once",
        peak.load(Ordering::SeqCst)
    );

    // The abandoned lookups hand their permits back once the resolver returns.
    let drained = tokio::time::timeout(Duration::from_secs(15), async {
        while pool.available_global() < PERMITS || active.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        drained.is_ok(),
        "the slow lookups never released their permits"
    );
    let patient = DnsLimits {
        lookup_timeout: SLOW * 5,
        ..LIMITS
    };
    let addrs = lookup_bounded_with(slow_resolver, "late.test", 443, patient, &pool)
        .await
        .expect("a lookup after the slow ones finish resolves");
    assert_eq!(addrs, vec![SocketAddr::from(([93, 184, 216, 34], 443))]);
    assert!(peak.load(Ordering::SeqCst) <= PERMITS);
}

#[test]
fn the_domain_key_is_the_last_two_labels() {
    for (host, key) in [
        ("a1.slow.test", "slow.test"),
        ("A.B.Slow.Test", "slow.test"),
        ("a.slow.test.", "slow.test"),
        ("slow.test", "slow.test"),
        ("localhost", "localhost"),
        // The documented limit: a multi-label public suffix is one key.
        ("example.co.uk", "co.uk"),
    ] {
        assert_eq!(domain_key(host), key, "{host}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_domain_cannot_fill_the_dns_pool() {
    // Many subdomains of one domain whose nameserver never answers must not
    // take every permit: another domain still resolves at once (issue #2764).
    const GLOBAL: usize = 8;
    const LOOKUPS: usize = 40;
    const SLOW: Duration = Duration::from_secs(2);
    const LIMITS: DnsLimits = DnsLimits {
        permit_wait: Duration::from_millis(300),
        lookup_timeout: Duration::from_millis(500),
        per_domain: 4,
    };

    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let resolver = {
        let active = active.clone();
        let peak = peak.clone();
        move |host: &str, port: u16| -> std::io::Result<Vec<SocketAddr>> {
            if host.ends_with(".slow.test") {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(SLOW);
                active.fetch_sub(1, Ordering::SeqCst);
            }
            Ok(vec![SocketAddr::from(([93, 184, 216, 34], port))])
        }
    };
    let pool = Arc::new(DnsPool::new(GLOBAL));
    let slow: Vec<_> = (0..LOOKUPS)
        .map(|index| {
            let resolve = resolver.clone();
            let pool = pool.clone();
            tokio::spawn(async move {
                let host = format!("a{index}.slow.test");
                lookup_bounded_with(resolve, &host, 443, LIMITS, &pool).await
            })
        })
        .collect();
    // Let the slow lookups take every slot their domain can get.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let started = Instant::now();
    let other = lookup_bounded_with(resolver, "ok.other.test", 443, LIMITS, &pool).await;
    let elapsed = started.elapsed();
    for lookup in slow {
        let outcome = lookup.await.expect("lookup task");
        assert!(
            matches!(outcome, Err(MediaUrlError::Unresolvable { .. })),
            "{outcome:?}"
        );
    }
    let drained = tokio::time::timeout(Duration::from_secs(15), async {
        while active.load(Ordering::SeqCst) > 0 || pool.available_global() < GLOBAL {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        drained.is_ok(),
        "the slow lookups never released their permits"
    );
    other.expect("another domain still resolves while slow.test is stuck");
    assert!(
        elapsed <= Duration::from_millis(200),
        "ok.other.test took {elapsed:?}"
    );
    assert_eq!(
        peak.load(Ordering::SeqCst),
        LIMITS.per_domain,
        "slow.test lookups holding resolver threads at once"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn permit_wait_does_not_eat_resolver_time() {
    // The second lookup waits about 200 ms for the only permit, then its
    // resolver takes 400 ms: 600 ms in all, past the 500 ms lookup timeout,
    // which must start only once the permit is held (issue #2764).
    const LIMITS: DnsLimits = DnsLimits {
        permit_wait: Duration::from_millis(300),
        lookup_timeout: Duration::from_millis(500),
        per_domain: 4,
    };
    let resolver = |host: &str, port: u16| -> std::io::Result<Vec<SocketAddr>> {
        let hold = if host == "first.a.test" { 200 } else { 400 };
        std::thread::sleep(Duration::from_millis(hold));
        Ok(vec![SocketAddr::from(([93, 184, 216, 34], port))])
    };
    let pool = Arc::new(DnsPool::new(1));
    let first = {
        let pool = pool.clone();
        tokio::spawn(async move {
            lookup_bounded_with(resolver, "first.a.test", 443, LIMITS, &pool).await
        })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    let second = lookup_bounded_with(resolver, "second.b.test", 443, LIMITS, &pool).await;
    first
        .await
        .expect("first lookup task")
        .expect("first lookup resolves");
    second.expect("the resolver got its whole timeout after the permit wait");
}

#[test]
fn idle_domain_keys_are_pruned_and_busy_ones_kept() {
    let pool = DnsPool::new(4);
    let busy = pool.domain("busy.test", 4);
    let _held = busy.clone().try_acquire_owned().expect("a free permit");
    for index in 0..1100 {
        drop(pool.domain(&format!("d{index}.test"), 4));
    }
    assert!(pool.tracked_domains() <= 1024, "{}", pool.tracked_domains());
    assert!(Arc::ptr_eq(&pool.domain("busy.test", 4), &busy));
}

#[test]
fn pins_expire_and_the_oldest_is_evicted_past_capacity() {
    let store = PinStore::new(Duration::from_secs(30), 2);
    let start = std::time::Instant::now();
    let answer = [SocketAddr::from(([93, 184, 216, 34], 443))];
    store.pin_at("One.Test", &answer, start);
    // Ports are cleared, and the host is matched case-insensitively.
    assert_eq!(
        store.get_at("one.test", start),
        Some(vec![SocketAddr::from(([93, 184, 216, 34], 0))])
    );
    store.pin_at("two.test", &answer, start + Duration::from_secs(1));
    store.pin_at("three.test", &answer, start + Duration::from_secs(2));
    assert_eq!(store.host_count(), 2);
    assert!(store.get_at("one.test", start).is_none(), "oldest evicted");
    assert!(store.get_at("two.test", start).is_some());
    let later = start + Duration::from_secs(31);
    assert!(store.get_at("two.test", later).is_none(), "expired");
    // New hosts still fit once the older pins expire.
    store.pin_at("four.test", &answer, later);
    store.pin_at("five.test", &answer, later);
    assert!(store.get_at("four.test", later).is_some());
    assert!(store.get_at("five.test", later).is_some());
}

/// The media client's own [`PolicyResolver`], with every address it lets
/// through dialled on loopback instead, so names registered with a public
/// answer reach a local test origin over the real connect path.
#[derive(Debug, Clone, Copy)]
struct LoopbackDialer;

impl Resolve for LoopbackDialer {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        Box::pin(async move {
            let addrs = PolicyResolver.resolve(name).await?;
            let local: Vec<SocketAddr> = addrs
                .map(|addr| SocketAddr::from(([127, 0, 0, 1], addr.port())))
                .collect();
            Ok(Box::new(local.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// A loopback origin that answers every request with `response`.
async fn serve_canned(response: String) -> (u16, tokio::task::JoinHandle<()>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind origin");
    let port = listener.local_addr().expect("origin addr").port();
    let task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let _ = socket.write_all(response.as_bytes()).await;
        }
    });
    (port, task)
}

#[tokio::test]
async fn connect_reuses_the_preflight_answer() {
    // One fetch of a hostname URL whose origin redirects once to another
    // hostname: each hop's pre-flight check resolves its name, and the
    // connect must reuse that answer instead of resolving again (issue #2764).
    const ORIGIN: &str = "origin.preflight-pin.test";
    const HOP: &str = "hop.preflight-pin.test";
    let _private = deny_private_media_urls_in_tests();
    let public: IpAddr = "93.184.216.34".parse().expect("address");
    test_answers::register(ORIGIN, vec![public]);
    test_answers::register(HOP, vec![public]);

    let (hop_port, hop_task) = serve_canned(
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".to_owned(),
    )
    .await;
    let (origin_port, origin_task) = serve_canned(format!(
        "HTTP/1.1 302 Found\r\nLocation: http://{HOP}:{hop_port}/x.png\r\n\
         Content-Length: 0\r\nConnection: close\r\n\r\n"
    ))
    .await;
    let client = crate::server::media::media_client_builder()
        .no_proxy()
        .dns_resolver(Arc::new(LoopbackDialer))
        .build()
        .expect("media client builds");

    // What the fetch path does: the pre-flight check, then the redirect chain.
    let first = ensure_media_url_allowed(&format!("http://{ORIGIN}:{origin_port}/start.png"))
        .await
        .expect("the origin passes the pre-flight check");
    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        send_following_redirects(&client, first),
    )
    .await
    .expect("the fetch finishes within its bound");
    origin_task.abort();
    hop_task.abort();
    let response = match outcome {
        Ok(response) => response,
        Err(err) => panic!("the fetch failed: {err:?}"),
    };
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(
        (test_answers::lookups(ORIGIN), test_answers::lookups(HOP)),
        (1, 1),
        "lookups per hop (origin, hop)"
    );
}
