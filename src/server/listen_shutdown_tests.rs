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

//! Accept-loop shutdown and connection drain (#3483).
//!
//! Real listeners and real clients: what is under test is when `serve`
//! returns relative to the connections it holds, and what those connections
//! see on the wire.

use std::time::{Duration, Instant};

use axum::Router;
use axum::routing::get;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::{ServeOptions, serve};
use crate::server::transport::{HttpTimeouts, ListenTarget};

fn router() -> Router {
    Router::new()
        .route("/fast", get(|| async { "ok" }))
        .route(
            "/slow",
            get(|| async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                "slow-ok"
            }),
        )
        .route("/stream", get(stream))
}

/// An SSE stream of three events 500 ms apart, then the `[DONE]` sentinel a
/// chat-completions stream ends with.
async fn stream() -> axum::response::Response {
    use axum::response::IntoResponse;
    use axum::response::sse::Event;
    let events = futures::stream::unfold(0u32, |n| async move {
        if n > 3 {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        let data = if n == 3 {
            "[DONE]".to_string()
        } else {
            format!("chunk-{n}")
        };
        Some((
            Ok::<_, std::convert::Infallible>(Event::default().data(data)),
            n + 1,
        ))
    });
    axum::response::Sse::new(events).into_response()
}

struct Running {
    addr: String,
    shutdown: CancellationToken,
    served: JoinHandle<anyhow::Result<()>>,
}

async fn start(target: ListenTarget, drain: Duration) -> Running {
    let shutdown = CancellationToken::new();
    let options = ServeOptions {
        timeouts: HttpTimeouts::from_secs(30),
        reuse_port: false,
        tls: None,
        shutdown: shutdown.clone(),
        drain,
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    let served = tokio::spawn(async move {
        let mut tx = Some(tx);
        serve(&target, router(), options, |addr| {
            let _ = tx.take().expect("announced once").send(addr.to_string());
        })
        .await
    });
    let announced = rx.await.expect("the server announces its address");
    let addr = announced
        .rsplit_once("://")
        .map(|(_, rest)| rest.to_string())
        .unwrap_or(announced);
    Running {
        addr,
        shutdown,
        served,
    }
}

async fn start_tcp(drain: Duration) -> Running {
    start(
        ListenTarget::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        },
        drain,
    )
    .await
}

/// Cancel the token and wait (bounded) for `serve` to return; the elapsed
/// time is measured from the cancel.
async fn stop(running: Running, bound: Duration) -> (anyhow::Result<()>, Duration) {
    let started = Instant::now();
    running.shutdown.cancel();
    let served = tokio::time::timeout(bound, running.served)
        .await
        .unwrap_or_else(|_| panic!("serve did not return within {bound:?} of the shutdown"))
        .expect("serve task");
    (served, started.elapsed())
}

#[tokio::test]
async fn an_idle_listener_returns_at_once_and_refuses_new_connections() {
    let running = start_tcp(Duration::from_secs(30)).await;
    let addr = running.addr.clone();
    let (served, elapsed) = stop(running, Duration::from_secs(5)).await;
    served.expect("a shutdown is a clean return");
    assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
    assert!(
        TcpStream::connect(&addr).await.is_err(),
        "the listener is closed, so a new connection is refused"
    );
}

#[tokio::test]
async fn a_connected_client_that_sent_nothing_does_not_hold_the_drain() {
    let running = start_tcp(Duration::from_secs(30)).await;
    let mut silent = TcpStream::connect(&running.addr).await.expect("connect");
    // Let the server accept it before the shutdown.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (served, elapsed) = stop(running, Duration::from_secs(10)).await;
    served.expect("clean return");
    assert!(
        elapsed < Duration::from_secs(3),
        "an idle connection must close at once, not wait out the 30 s drain: {elapsed:?}"
    );
    let mut buf = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(2), silent.read_to_end(&mut buf)).await;
    assert!(read.is_ok(), "the idle connection is closed by the server");
}

#[tokio::test]
async fn an_idle_keep_alive_connection_closes_on_shutdown() {
    let running = start_tcp(Duration::from_secs(30)).await;
    let mut client = TcpStream::connect(&running.addr).await.expect("connect");
    client
        .write_all(b"GET /fast HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("write");
    let mut head = [0u8; 256];
    let n = client.read(&mut head).await.expect("response");
    assert!(String::from_utf8_lossy(&head[..n]).contains("200 OK"));
    let (served, elapsed) = stop(running, Duration::from_secs(10)).await;
    served.expect("clean return");
    assert!(elapsed < Duration::from_secs(3), "took {elapsed:?}");
}

#[tokio::test]
async fn an_in_flight_stream_runs_to_completion_before_serve_returns() {
    let running = start_tcp(Duration::from_secs(30)).await;
    let mut client = TcpStream::connect(&running.addr).await.expect("connect");
    client
        .write_all(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("write");
    // The response head is out; the shutdown lands mid-stream.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let reader = tokio::spawn(async move {
        let mut buf = Vec::new();
        let _ = tokio::time::timeout(Duration::from_secs(10), client.read_to_end(&mut buf)).await;
        buf
    });
    let (served, elapsed) = stop(running, Duration::from_secs(10)).await;
    served.expect("clean return");
    let body = String::from_utf8_lossy(&reader.await.expect("reader")).into_owned();
    assert!(
        body.contains("data: [DONE]"),
        "the stream must finish before its connection closes: {body}"
    );
    assert!(
        elapsed >= Duration::from_millis(1000),
        "serve returned before the stream finished: {elapsed:?}"
    );
}

#[tokio::test]
async fn the_drain_deadline_aborts_a_response_still_in_progress() {
    let running = start_tcp(Duration::from_secs(1)).await;
    let mut client = TcpStream::connect(&running.addr).await.expect("connect");
    client
        .write_all(b"GET /slow HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("write");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (served, elapsed) = stop(running, Duration::from_secs(10)).await;
    served.expect("a drain that hit the deadline is still a clean return");
    assert!(
        elapsed >= Duration::from_millis(900) && elapsed < Duration::from_secs(3),
        "serve should return about 1 s after the shutdown: {elapsed:?}"
    );
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(2), client.read_to_end(&mut buf)).await;
    assert!(
        !String::from_utf8_lossy(&buf).contains("slow-ok"),
        "the aborted response must not complete"
    );
}

#[tokio::test]
async fn a_zero_drain_aborts_in_flight_responses_at_once() {
    let running = start_tcp(Duration::ZERO).await;
    let mut client = TcpStream::connect(&running.addr).await.expect("connect");
    client
        .write_all(b"GET /slow HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("write");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (served, elapsed) = stop(running, Duration::from_secs(5)).await;
    served.expect("clean return");
    assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
}

#[tokio::test]
async fn a_shutdown_requested_before_the_bind_does_not_serve() {
    let shutdown = CancellationToken::new();
    shutdown.cancel();
    let options = ServeOptions {
        timeouts: HttpTimeouts::from_secs(30),
        reuse_port: false,
        tls: None,
        shutdown,
        drain: Duration::from_secs(30),
    };
    let target = ListenTarget::Tcp {
        host: "127.0.0.1".to_string(),
        port: 0,
    };
    let mut announced = false;
    let served = tokio::time::timeout(
        Duration::from_secs(5),
        serve(&target, router(), options, |_| announced = true),
    )
    .await
    .expect("returns at once");
    served.expect("clean return");
    assert!(!announced, "nothing was bound, so nothing is announced");
}

#[cfg(unix)]
#[tokio::test]
async fn a_unix_socket_listener_removes_its_socket_file_on_shutdown() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("server.sock");
    let running = start(ListenTarget::Unix(path.clone()), Duration::from_secs(30)).await;
    assert!(path.exists(), "the socket is bound");
    let (served, _) = stop(running, Duration::from_secs(5)).await;
    served.expect("clean return");
    assert!(
        std::fs::symlink_metadata(&path).is_err(),
        "the socket file is removed after the listener is dropped"
    );
}
