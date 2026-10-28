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

//! Redirect following through [`send_following_redirects`] (issue #2693).
//!
//! Behind a forward proxy reqwest resolves only the proxy's host, so a hostname
//! redirect target never reached `PolicyResolver`; the helper now checks every
//! hop itself. Every listener here is on `127.0.0.1`, the proxy is addressed by
//! IP literal, and `media.test` is a name only the proxy knows, so nothing
//! leaves the host and nothing depends on the local resolver except
//! `localhost`.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

use super::super::{media_client_builder, media_send_failure};
use crate::server::media_net::{
    MAX_MEDIA_REQUESTS, MediaSendError, MediaUrlError, allow_private_media_urls_in_tests,
    check_hop_with, deny_private_media_urls_in_tests, send_following_redirects,
    send_following_redirects_within,
};

const GENERIC: &str = "media URL host is not allowed or could not be resolved";

/// Read one request head, up to and including the blank line.
async fn read_head(socket: &mut TcpStream) -> String {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") && head.len() < 16 * 1024 {
        match socket.read(&mut byte).await {
            Ok(1) => head.push(byte[0]),
            _ => break,
        }
    }
    String::from_utf8_lossy(&head).into_owned()
}

/// A `Connection: close` response with an empty body unless one is given.
fn reply(status: &str, headers: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// A `302 Found` naming `location`.
fn found(location: &str) -> String {
    reply("302 Found", &format!("Location: {location}\r\n"), "")
}

/// Aborts the listener tasks when a test ends, pass or fail.
struct Tasks(Vec<JoinHandle<()>>);

impl Drop for Tasks {
    fn drop(&mut self) {
        for task in &self.0 {
            task.abort();
        }
    }
}

/// An origin that answers each request path with `respond(path)`, after
/// `delay`; `None` holds the connection open without answering. Returns its
/// address and the paths it was asked for.
async fn spawn_origin(
    delay: Duration,
    respond: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
) -> (SocketAddr, Arc<Mutex<Vec<String>>>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind origin");
    let addr = listener.local_addr().expect("origin addr");
    let paths = Arc::new(Mutex::new(Vec::new()));
    let seen = paths.clone();
    let respond = Arc::new(respond);
    let task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let seen = seen.clone();
            let respond = respond.clone();
            tokio::spawn(async move {
                let head = read_head(&mut socket).await;
                let path = head
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_owned();
                seen.lock().expect("origin log").push(path.clone());
                tokio::time::sleep(delay).await;
                match respond(&path) {
                    Some(response) => {
                        let _ = socket.write_all(response.as_bytes()).await;
                    }
                    None => tokio::time::sleep(Duration::from_secs(60)).await,
                }
            });
        }
    });
    (addr, paths, task)
}

/// A listener that records whether anything connected to it at all.
async fn spawn_internal() -> (u16, Arc<AtomicBool>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind internal");
    let port = listener.local_addr().expect("internal addr").port();
    let reached = Arc::new(AtomicBool::new(false));
    let flag = reached.clone();
    let task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            flag.store(true, Ordering::SeqCst);
            let _ = socket.write_all(reply("200 OK", "", "ok").as_bytes()).await;
        }
    });
    (port, reached, task)
}

/// A forward proxy that records every request target (the absolute-form URL,
/// or a CONNECT authority), relays targets on host `media.test` to `origin`,
/// and answers 502 to anything else.
async fn spawn_proxy(origin: SocketAddr) -> (SocketAddr, Arc<Mutex<Vec<String>>>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    let log = Arc::new(Mutex::new(Vec::new()));
    let targets = log.clone();
    let task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let targets = targets.clone();
            tokio::spawn(async move {
                let head = read_head(&mut socket).await;
                let mut words = head.split_whitespace();
                let method = words.next().unwrap_or_default().to_owned();
                let target = words.next().unwrap_or_default().to_owned();
                targets.lock().expect("proxy log").push(target.clone());
                let relayed = reqwest::Url::parse(&target)
                    .ok()
                    .filter(|url| method != "CONNECT" && url.host_str() == Some("media.test"));
                let Some(url) = relayed else {
                    let _ = socket
                        .write_all(reply("502 Bad Gateway", "", "").as_bytes())
                        .await;
                    return;
                };
                let Ok(mut upstream) = TcpStream::connect(origin).await else {
                    return;
                };
                let path = match url.query() {
                    Some(query) => format!("{}?{query}", url.path()),
                    None => url.path().to_owned(),
                };
                let request =
                    format!("GET {path} HTTP/1.1\r\nHost: media.test\r\nConnection: close\r\n\r\n");
                let _ = upstream.write_all(request.as_bytes()).await;
                let mut response = Vec::new();
                let _ = upstream.read_to_end(&mut response).await;
                let _ = socket.write_all(&response).await;
            });
        }
    });
    (addr, log, task)
}

/// The production client configuration, sent through `proxy`.
fn proxied_client(proxy: SocketAddr) -> reqwest::Client {
    media_client_builder()
        .proxy(reqwest::Proxy::all(format!("http://{proxy}")).expect("proxy URL"))
        .build()
        .expect("media client builds")
}

/// The production client configuration with no proxy, whatever the
/// environment of the test process says.
fn direct_client() -> reqwest::Client {
    media_client_builder()
        .no_proxy()
        .build()
        .expect("media client builds")
}

fn url(raw: &str) -> reqwest::Url {
    reqwest::Url::parse(raw).expect("test URL parses")
}

#[tokio::test]
async fn behind_a_proxy_a_redirect_onto_an_internal_hostname_never_leaves() {
    let _private = deny_private_media_urls_in_tests();
    let (internal_port, reached, internal) = spawn_internal().await;
    let location = format!("http://localhost:{internal_port}/x.png");
    let (origin_addr, _, origin) =
        spawn_origin(Duration::ZERO, move |_| Some(found(&location))).await;
    let (proxy_addr, log, proxy) = spawn_proxy(origin_addr).await;
    let _tasks = Tasks(vec![internal, origin, proxy]);

    let outcome = tokio::time::timeout(
        Duration::from_secs(8),
        send_following_redirects(
            &proxied_client(proxy_addr),
            url("http://media.test/start.png"),
        ),
    )
    .await
    .expect("the fetch finishes within its bound");

    let refusal = match outcome {
        Err(MediaSendError::Refused(refusal)) => refusal,
        Ok(response) => panic!("followed to {} ({})", response.url(), response.status()),
        Err(other) => panic!("not a policy refusal: {other:?}"),
    };
    assert!(
        matches!(&refusal, MediaUrlError::BlockedAddress { host, .. } if host == "localhost"),
        "{refusal:?}"
    );
    assert_eq!(
        *log.lock().expect("proxy log"),
        vec!["http://media.test/start.png".to_owned()],
        "the proxy was asked for the internal host"
    );
    assert!(
        !reached.load(Ordering::SeqCst),
        "the internal listener was contacted"
    );
    // The text the client sees for that hop is the pre-flight's generic one.
    let message = media_send_failure(
        "image",
        "http://media.test/start.png",
        MediaSendError::Refused(refusal),
    )
    .to_string();
    assert_eq!(message, format!("image URL rejected: {GENERIC}"));
}

#[tokio::test]
async fn behind_a_proxy_a_redirect_onto_an_internal_literal_never_leaves() {
    let _private = deny_private_media_urls_in_tests();
    let (internal_port, reached, internal) = spawn_internal().await;
    let location = format!("http://127.0.0.1:{internal_port}/x.png");
    let (origin_addr, _, origin) =
        spawn_origin(Duration::ZERO, move |_| Some(found(&location))).await;
    let (proxy_addr, log, proxy) = spawn_proxy(origin_addr).await;
    let _tasks = Tasks(vec![internal, origin, proxy]);

    let outcome = send_following_redirects(
        &proxied_client(proxy_addr),
        url("http://media.test/start.png"),
    )
    .await;
    assert!(
        matches!(
            outcome,
            Err(MediaSendError::Refused(
                MediaUrlError::BlockedAddress { .. }
            ))
        ),
        "{outcome:?}"
    );
    assert_eq!(log.lock().expect("proxy log").len(), 1);
    assert!(!reached.load(Ordering::SeqCst));
}

#[tokio::test]
async fn a_redirect_chain_through_the_proxy_reaches_its_final_response() {
    let _private = allow_private_media_urls_in_tests();
    let (origin_addr, paths, origin) = spawn_origin(Duration::ZERO, |path| {
        Some(match path {
            "/start.png" => found("http://media.test/hop.png"),
            // Relative, so it is joined against the hop's own URL.
            "/hop.png" => found("img/final.png"),
            "/img/final.png" => reply("200 OK", "Content-Type: image/png\r\n", "final"),
            _ => reply("404 Not Found", "", ""),
        })
    })
    .await;
    let (proxy_addr, log, proxy) = spawn_proxy(origin_addr).await;
    let _tasks = Tasks(vec![origin, proxy]);

    let response = send_following_redirects(
        &proxied_client(proxy_addr),
        url("http://media.test/start.png"),
    )
    .await
    .expect("the chain is followed");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(response.url().as_str(), "http://media.test/img/final.png");
    assert_eq!(response.text().await.expect("body"), "final");
    assert_eq!(
        *log.lock().expect("proxy log"),
        [
            "http://media.test/start.png",
            "http://media.test/hop.png",
            "http://media.test/img/final.png",
        ]
    );
    assert_eq!(paths.lock().expect("origin log").len(), 3);
}

#[tokio::test]
async fn a_fifth_redirect_is_refused_after_exactly_five_requests() {
    let _private = allow_private_media_urls_in_tests();
    let counter = Arc::new(AtomicUsize::new(0));
    let next = counter.clone();
    let (origin_addr, paths, origin) = spawn_origin(Duration::ZERO, move |_| {
        let hop = next.fetch_add(1, Ordering::SeqCst) + 1;
        Some(found(&format!("/hop{hop}.png")))
    })
    .await;
    let _tasks = Tasks(vec![origin]);

    let outcome = send_following_redirects(
        &direct_client(),
        url(&format!("http://{origin_addr}/start.png")),
    )
    .await;
    match outcome {
        Err(MediaSendError::Refused(MediaUrlError::TooManyRedirects { url })) => {
            assert_eq!(url, format!("http://{origin_addr}/hop5.png"));
        }
        other => panic!("expected TooManyRedirects, got {other:?}"),
    }
    assert_eq!(MAX_MEDIA_REQUESTS, 5);
    assert_eq!(paths.lock().expect("origin log").len(), MAX_MEDIA_REQUESTS);
}

#[tokio::test]
async fn a_redirect_without_a_location_is_a_bad_redirect() {
    let _private = allow_private_media_urls_in_tests();
    let (origin_addr, _, origin) =
        spawn_origin(Duration::ZERO, |_| Some(reply("302 Found", "", ""))).await;
    let _tasks = Tasks(vec![origin]);

    let outcome = send_following_redirects(
        &direct_client(),
        url(&format!("http://{origin_addr}/x.png")),
    )
    .await;
    let err = match outcome {
        Err(err @ MediaSendError::BadRedirect { .. }) => err,
        other => panic!("expected BadRedirect, got {other:?}"),
    };
    assert!(
        matches!(err, MediaSendError::BadRedirect { status } if status == reqwest::StatusCode::FOUND),
        "{err:?}"
    );
    assert_eq!(
        media_send_failure("image", "http://x/", err).to_string(),
        "image URL returned redirect 302 Found without a usable Location header"
    );
}

#[tokio::test]
async fn an_unfollowed_redirect_status_is_returned_unchanged() {
    let _private = allow_private_media_urls_in_tests();
    let (origin_addr, paths, origin) = spawn_origin(Duration::ZERO, |_| {
        Some(reply(
            "304 Not Modified",
            "Location: /elsewhere.png\r\n",
            "",
        ))
    })
    .await;
    let _tasks = Tasks(vec![origin]);

    let response = send_following_redirects(
        &direct_client(),
        url(&format!("http://{origin_addr}/x.png")),
    )
    .await
    .expect("a 304 is a response");
    assert_eq!(response.status(), reqwest::StatusCode::NOT_MODIFIED);
    assert_eq!(paths.lock().expect("origin log").len(), 1);
}

#[tokio::test]
async fn the_whole_chain_shares_one_deadline() {
    let _private = allow_private_media_urls_in_tests();
    // The first hop answers; the second never does.
    let (origin_addr, _, origin) = spawn_origin(Duration::ZERO, |path| {
        (path == "/start.png").then(|| found("/stall.png"))
    })
    .await;
    let _tasks = Tasks(vec![origin]);

    let started = Instant::now();
    let outcome = send_following_redirects_within(
        &direct_client(),
        url(&format!("http://{origin_addr}/start.png")),
        Duration::from_millis(300),
    )
    .await;
    let elapsed = started.elapsed();
    match outcome {
        Err(MediaSendError::Transport(err)) => assert!(err.is_timeout(), "{err:?}"),
        other => panic!("expected a transport timeout, got {other:?}"),
    }
    assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
}

#[tokio::test]
async fn hops_that_each_fit_the_budget_still_share_it() {
    let _private = allow_private_media_urls_in_tests();
    // Each hop answers in 150 ms, well inside a 300 ms budget on its own, but
    // three of them cannot fit in one: a per-hop timeout would reach the 200.
    let (origin_addr, _, origin) = spawn_origin(Duration::from_millis(150), |path| {
        Some(match path {
            "/start.png" => found("/a.png"),
            "/a.png" => found("/b.png"),
            _ => reply("200 OK", "", "late"),
        })
    })
    .await;
    let _tasks = Tasks(vec![origin]);

    let outcome = send_following_redirects_within(
        &direct_client(),
        url(&format!("http://{origin_addr}/start.png")),
        Duration::from_millis(300),
    )
    .await;
    match outcome {
        Err(MediaSendError::Transport(err)) => assert!(err.is_timeout(), "{err:?}"),
        other => panic!("expected a transport timeout, got {other:?}"),
    }
}

#[tokio::test]
async fn a_hop_lookup_that_outlives_the_deadline_is_a_timeout() {
    // The chain deadline running out while a hop is resolved is a timeout,
    // not a refusal of the host (issue #2713).
    let started = Instant::now();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
    let outcome = check_hop_with("slow.test", deadline, std::future::pending()).await;
    let elapsed = started.elapsed();
    match outcome {
        Err(MediaSendError::DeadlineExceeded { host }) => assert_eq!(host, "slow.test"),
        other => panic!("expected DeadlineExceeded, got {other:?}"),
    }
    assert!(elapsed < Duration::from_secs(1), "took {elapsed:?}");
}

#[tokio::test]
async fn a_hop_check_that_fails_in_time_is_still_a_refusal() {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let refusal = MediaUrlError::Unresolvable {
        host: "gone.test".to_owned(),
    };
    let outcome = check_hop_with("gone.test", deadline, async { Err(refusal.clone()) }).await;
    match outcome {
        Err(MediaSendError::Refused(err)) => assert_eq!(err, refusal),
        other => panic!("expected Refused, got {other:?}"),
    }
}
