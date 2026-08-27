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

//! Socket-level tests for the b10621 transport (#1432).
//!
//! These drive real listeners and real clients, because the properties under
//! test are exactly the ones a router-level `oneshot` cannot see: whether the
//! socket options were applied, whether a half-open connection is closed on
//! the `--timeout` budget, and whether a slow response survives it.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::get;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use rustls_pki_types::pem::PemObject;

use super::{ServeOptions, bind_tcp, serve};
use crate::server::tls::{TlsPaths, build_server_config};
use crate::server::transport::{HttpTimeouts, ListenTarget};

/// Route set used by the socket tests: an instant reply, a deliberately
/// stalled one standing in for a long prefill, and an SSE stream whose first
/// event is equally late.
fn test_router() -> Router {
    Router::new()
        .route("/fast", get(|| async { "ok" }))
        .route(
            "/slow",
            get(|| async {
                tokio::time::sleep(Duration::from_secs(3)).await;
                "slow-ok"
            }),
        )
        .route("/slow-sse", get(slow_sse))
}

/// An SSE stream that stays silent for three seconds, then emits one event and
/// ends. This is the shape of a long prefill: the response head is already on
/// the wire while the body produces nothing.
async fn slow_sse() -> axum::response::Response {
    use axum::response::IntoResponse;
    let stream = futures::stream::once(async {
        tokio::time::sleep(Duration::from_secs(3)).await;
        Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data("late"))
    });
    axum::response::Sse::new(stream).into_response()
}

fn options(timeout_secs: u64) -> ServeOptions {
    ServeOptions {
        timeouts: HttpTimeouts::from_secs(timeout_secs),
        reuse_port: false,
        tls: None,
    }
}

/// Start the server on an ephemeral port and return the bound address.
async fn spawn_server(options: ServeOptions) -> String {
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let target = ListenTarget::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        };
        let mut tx = Some(tx);
        let _ = serve(&target, test_router(), options, |addr| {
            let _ = tx.take().expect("announced once").send(addr.to_string());
        })
        .await;
    });
    rx.await.expect("the server announces its address")
}

fn host_port(announced: &str) -> String {
    announced
        .rsplit_once("://")
        .map(|(_, rest)| rest.to_string())
        .unwrap_or_else(|| announced.to_string())
}

async fn read_to_end_with_deadline(stream: &mut TcpStream, deadline: Duration) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(deadline, stream.read_to_end(&mut buf)).await;
    buf
}

#[tokio::test]
async fn binding_port_zero_reports_the_ephemeral_port() {
    let listener = bind_tcp("127.0.0.1", 0, false).expect("binds");
    let local = listener.local_addr().expect("local addr");
    assert_ne!(local.port(), 0, "b10621 reports the port it actually got");
}

#[tokio::test]
async fn a_second_bind_without_reuse_port_fails() {
    let first = bind_tcp("127.0.0.1", 0, false).expect("binds");
    let port = first.local_addr().expect("local addr").port();
    assert!(
        bind_tcp("127.0.0.1", port, false).is_err(),
        "SO_REUSEADDR alone must not let a second listener share a live port"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn reuse_port_lets_two_listeners_share_one_port() {
    let first = bind_tcp("127.0.0.1", 0, true).expect("binds");
    let port = first.local_addr().expect("local addr").port();
    bind_tcp("127.0.0.1", port, true).expect("--reuse-port must allow a second bind");
}

#[tokio::test]
async fn an_unresolvable_host_is_a_named_startup_error() {
    let err = bind_tcp("no-such-host.invalid", 8080, false).expect_err("must fail");
    assert!(
        format!("{err:#}").contains("no-such-host.invalid"),
        "{err:#}"
    );
}

#[tokio::test]
async fn a_bound_server_answers_an_ordinary_request() {
    let announced = spawn_server(options(30)).await;
    let mut stream = TcpStream::connect(host_port(&announced))
        .await
        .expect("connect");
    stream
        .write_all(b"GET /fast HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .expect("write");
    let body = read_to_end_with_deadline(&mut stream, Duration::from_secs(10)).await;
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("200 OK"), "{text}");
    assert!(text.contains("ok"), "{text}");
}

#[tokio::test]
async fn a_client_that_never_finishes_its_request_is_closed_on_the_timeout_budget() {
    // The socket read timeout, not the decode watchdog, is what bounds this:
    // no model work has started, so there is nothing for a decode timeout to
    // cancel.
    let announced = spawn_server(options(1)).await;
    let mut stream = TcpStream::connect(host_port(&announced))
        .await
        .expect("connect");
    stream
        .write_all(b"GET /fast HTTP/1.1\r\nHost: x\r\n")
        .await
        .expect("partial head");

    let mut buf = Vec::new();
    let outcome = tokio::time::timeout(Duration::from_secs(15), stream.read_to_end(&mut buf)).await;
    assert!(
        outcome.is_ok(),
        "--timeout 1 must close a half-open request well inside 15s"
    );
}

#[tokio::test]
async fn a_slow_response_is_not_cut_off_by_the_socket_timeout() {
    // `--timeout` bounds socket reads and writes, not how long a handler may
    // take. A three-second response must survive a one-second socket budget,
    // which is what proves the HTTP timeout and the decode watchdog are
    // independent controls.
    let announced = spawn_server(options(1)).await;
    let mut stream = TcpStream::connect(host_port(&announced))
        .await
        .expect("connect");
    stream
        .write_all(b"GET /slow HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .expect("write");
    let body = read_to_end_with_deadline(&mut stream, Duration::from_secs(20)).await;
    let text = String::from_utf8_lossy(&body);
    assert!(
        text.contains("slow-ok"),
        "a 3s handler must complete under --timeout 1: {text}"
    );
}

#[tokio::test]
async fn tls_is_refused_on_a_unix_domain_socket() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths = crate::server::tls::tls_tests::write_self_signed(dir.path());
    let tls = build_server_config(&paths).expect("valid pair");
    let target = ListenTarget::Unix(dir.path().join("mlxcel.sock"));
    let err = serve(
        &target,
        test_router(),
        ServeOptions {
            timeouts: HttpTimeouts::default(),
            reuse_port: false,
            tls: Some(tls),
        },
        |_| {},
    )
    .await
    .expect_err("TLS over a Unix socket must be refused");
    assert!(format!("{err:#}").contains("--ssl-cert-file"), "{err:#}");
}

#[tokio::test]
async fn a_unix_socket_listener_answers_requests() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mlxcel.sock");
    let target = ListenTarget::Unix(path.clone());
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let mut tx = Some(tx);
        let _ = serve(&target, test_router(), options(30), |addr| {
            let _ = tx.take().expect("announced once").send(addr.to_string());
        })
        .await;
    });
    let announced = rx.await.expect("announced");
    assert!(announced.starts_with("unix://"), "{announced}");

    let mut stream = tokio::net::UnixStream::connect(&path)
        .await
        .expect("connect to socket");
    stream
        .write_all(b"GET /fast HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .expect("write");
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut buf))
        .await
        .expect("response arrives")
        .expect("read");
    assert!(String::from_utf8_lossy(&buf).contains("200 OK"));
}

#[tokio::test]
async fn a_tls_listener_completes_a_handshake_and_answers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let paths: TlsPaths = crate::server::tls::tls_tests::write_self_signed(dir.path());
    let server_config = build_server_config(&paths).expect("valid pair");

    let (tx, rx) = tokio::sync::oneshot::channel();
    let opts = ServeOptions {
        timeouts: HttpTimeouts::from_secs(30),
        reuse_port: false,
        tls: Some(server_config),
    };
    tokio::spawn(async move {
        let target = ListenTarget::Tcp {
            host: "127.0.0.1".to_string(),
            port: 0,
        };
        let mut tx = Some(tx);
        let _ = serve(&target, test_router(), opts, |addr| {
            let _ = tx.take().expect("announced once").send(addr.to_string());
        })
        .await;
    });
    let announced = rx.await.expect("announced");
    assert!(
        announced.starts_with("https://"),
        "a TLS listener announces https: {announced}"
    );

    let mut roots = tokio_rustls::rustls::RootCertStore::empty();
    for cert in tokio_rustls::rustls::pki_types::CertificateDer::pem_file_iter(&paths.cert)
        .expect("cert file opens")
    {
        roots.add(cert.expect("cert der")).expect("trust anchor");
    }
    let client_config = tokio_rustls::rustls::ClientConfig::builder_with_provider(Arc::new(
        tokio_rustls::rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("protocol versions")
    .with_root_certificates(roots)
    .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));

    let tcp = TcpStream::connect(host_port(&announced))
        .await
        .expect("connect");
    let server_name = tokio_rustls::rustls::pki_types::ServerName::try_from("localhost")
        .expect("server name")
        .to_owned();
    let mut tls = connector
        .connect(server_name, tcp)
        .await
        .expect("TLS handshake succeeds against the configured certificate");
    tls.write_all(b"GET /fast HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write");
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), tls.read_to_end(&mut buf))
        .await
        .expect("response arrives")
        .expect("read");
    let text = String::from_utf8_lossy(&buf);
    assert!(text.contains("200 OK"), "{text}");
    assert!(text.contains("ok"), "{text}");
}

#[tokio::test]
async fn a_silent_sse_stream_is_not_cut_off_by_the_socket_timeout() {
    // The response head is written immediately and the body then produces
    // nothing for three seconds. Under `--timeout 1` the read budget is stood
    // down for the whole exchange, so the stream survives; only a stalled
    // model would be cancelled here, and that is the decode watchdog's job.
    let announced = spawn_server(options(1)).await;
    let mut stream = TcpStream::connect(host_port(&announced))
        .await
        .expect("connect");
    stream
        .write_all(b"GET /slow-sse HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .expect("write");
    let body = read_to_end_with_deadline(&mut stream, Duration::from_secs(20)).await;
    let text = String::from_utf8_lossy(&body);
    assert!(
        text.contains("data: late"),
        "a 3s-silent SSE stream must survive --timeout 1: {text}"
    );
}

#[tokio::test]
async fn a_keep_alive_connection_is_bounded_again_after_a_response() {
    // The read budget re-arms when the response body finishes, so an idle
    // keep-alive connection is closed on the budget instead of being held
    // open forever.
    let announced = spawn_server(options(1)).await;
    let mut stream = TcpStream::connect(host_port(&announced))
        .await
        .expect("connect");
    stream
        .write_all(b"GET /fast HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("write");
    let mut buf = Vec::new();
    let outcome = tokio::time::timeout(Duration::from_secs(15), stream.read_to_end(&mut buf)).await;
    assert!(
        outcome.is_ok(),
        "an idle keep-alive connection must be closed on the --timeout budget"
    );
    assert!(
        String::from_utf8_lossy(&buf).contains("200 OK"),
        "the response itself must still have been delivered"
    );
}
