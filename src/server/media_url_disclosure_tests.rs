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

//! Remote media URL refusals must not tell the client which internal names
//! exist or what they resolve to, and a redirect must not reach an internal
//! hostname (issue #2644).
//!
//! Each case resolves `localhost`, which every test host maps to loopback
//! through its hosts file, so no external network is needed. The redirect-hop
//! cases at the end check that a hop URL the origin named never reaches the
//! client's error text (issue #2728).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::super::{media_client_builder, media_send_failure, media_url_rejected};
use super::{
    ImageInputLimits, build_chat_request, fetch_labeled_image, try_extract_chat_audio_data,
};
use crate::server::media_net::{
    ALLOW_PRIVATE_MEDIA_URLS_ENV, MEDIA_HOST_REFUSED, MediaSendError, MediaUrlError,
    allow_private_media_urls_in_tests, deny_private_media_urls_in_tests, send_following_redirects,
};
use crate::server::types::ContentPart;
use crate::server::types::request::InputAudio;

const GENERIC: &str = "media URL host is not allowed or could not be resolved";

/// Fail when `text` names the refused host, a loopback address, or the opt-in.
fn assert_discloses_nothing(text: &str) {
    for leaked in [
        "localhost",
        "127.0.0.1",
        "::1",
        ALLOW_PRIVATE_MEDIA_URLS_ENV,
    ] {
        assert!(!text.contains(leaked), "{leaked:?} leaked: {text}");
    }
}

#[tokio::test]
async fn a_labeled_image_refusal_is_generic() {
    // The text embeddings, rerank and image generation return as their 400.
    let _private = deny_private_media_urls_in_tests();
    let err = fetch_labeled_image(
        "http://localhost:9/x.png",
        "documents[0]",
        ImageInputLimits::default(),
    )
    .await
    .expect_err("a loopback name is refused");
    assert_eq!(
        err.error.message,
        format!("documents[0]: failed to read image: image URL rejected: {GENERIC}")
    );
    assert_discloses_nothing(&err.error.message);
}

#[tokio::test]
async fn a_chat_audio_url_refusal_is_generic() {
    let _private = deny_private_media_urls_in_tests();
    let request = build_chat_request(vec![ContentPart::InputAudio {
        input_audio: InputAudio {
            data: "http://localhost:9/clip.wav".to_string(),
            format: "wav".to_string(),
        },
    }]);
    let message = try_extract_chat_audio_data(&request)
        .await
        .expect_err("a loopback name is refused")
        .to_string();
    assert!(message.contains(GENERIC), "{message}");
    assert_discloses_nothing(&message);
}

#[tokio::test]
async fn a_redirect_onto_an_internal_hostname_never_reaches_it() {
    let _private = deny_private_media_urls_in_tests();

    // The internal service: records any connection at all.
    let internal = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind internal");
    let internal_port = internal.local_addr().expect("internal addr").port();
    let reached = Arc::new(AtomicBool::new(false));
    let reached_flag = reached.clone();
    let internal_task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = internal.accept().await {
            reached_flag.store(true, Ordering::SeqCst);
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .await;
        }
    });

    // The "public" origin: answers every request with a 302 onto the internal
    // service by name. It is handed to the redirect helper directly, which
    // skips the call site's pre-flight check the way a genuinely public origin
    // passes it; being an IP literal it never touches the resolver itself.
    let origin = TcpListener::bind("127.0.0.1:0").await.expect("bind origin");
    let origin_addr = origin.local_addr().expect("origin addr");
    let redirect = format!(
        "HTTP/1.1 302 Found\r\nLocation: http://localhost:{internal_port}/x.png\r\n\
         Content-Length: 0\r\nConnection: close\r\n\r\n"
    );
    let origin_task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = origin.accept().await {
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let _ = socket.write_all(redirect.as_bytes()).await;
        }
    });

    // No proxy, whatever the test process's environment says, so this is the
    // direct path; `media_net_proxy_tests.rs` covers the proxied one.
    let client = media_client_builder()
        .no_proxy()
        .build()
        .expect("media client builds");
    let start = reqwest::Url::parse(&format!("http://{origin_addr}/start.png")).expect("URL");
    let outcome = tokio::time::timeout(
        Duration::from_secs(8),
        send_following_redirects(&client, start),
    )
    .await
    .expect("the fetch finishes within its bound");
    origin_task.abort();
    internal_task.abort();

    let err = match outcome {
        Ok(response) => panic!(
            "redirect was followed to {} (status {})",
            response.url(),
            response.status()
        ),
        Err(err) => err,
    };
    assert!(
        !reached.load(Ordering::SeqCst),
        "the internal listener was contacted"
    );
    assert!(
        matches!(
            &err,
            MediaSendError::Refused(MediaUrlError::BlockedAddress { host, .. }) if host == "localhost"
        ),
        "the refusal is not the address policy's: {err:?}"
    );
}

#[test]
fn a_redirect_deadline_names_no_host_and_is_not_a_refusal() {
    let text = media_send_failure(
        "image",
        "http://origin.test/a.png",
        MediaSendError::DeadlineExceeded {
            host: "slow.test".to_owned(),
        },
    )
    .to_string();
    assert!(text.contains("deadline"), "{text}");
    assert!(!text.contains("slow.test"), "{text}");
    assert!(!text.contains(MEDIA_HOST_REFUSED), "{text}");
    assert!(!text.contains("rejected"), "{text}");
}

#[test]
fn a_hop_refusal_renders_like_the_preflight_refusal() {
    let url = "http://origin.test/a.png";
    for err in [
        MediaUrlError::BlockedAddress {
            host: "internal.test".to_owned(),
            address: "10.0.0.5".parse().expect("address"),
        },
        MediaUrlError::Unresolvable {
            host: "gone.test".to_owned(),
        },
    ] {
        let preflight = media_url_rejected("image", url, &err).to_string();
        let hop =
            media_send_failure("image", url, MediaSendError::Refused(err.clone())).to_string();
        assert_eq!(hop, preflight);
        assert_eq!(preflight, format!("image URL rejected: {GENERIC}"));
    }
}

/// A loopback origin that answers every request with `response`.
async fn serve_canned(response: String) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind origin");
    let addr = listener.local_addr().expect("origin addr");
    let task = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let _ = socket.write_all(response.as_bytes()).await;
        }
    });
    (addr, task)
}

/// The submitted URL and the client-facing error for an image URL whose
/// origin answers every request with a redirect to `hop`.
///
/// Goes through [`fetch_labeled_image`], which renders the whole error chain
/// into the 400 the embeddings, rerank and image-generation routes return.
async fn redirected_fetch_outcome(hop: &str) -> (String, String) {
    let (origin_addr, origin_task) = serve_canned(format!(
        "HTTP/1.1 302 Found\r\nLocation: {hop}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    ))
    .await;
    let url = format!("http://{origin_addr}/start.png");
    let outcome = tokio::time::timeout(
        Duration::from_secs(15),
        fetch_labeled_image(&url, "input[0]", ImageInputLimits::default()),
    )
    .await
    .expect("the fetch finishes within its bound");
    origin_task.abort();
    let message = match outcome {
        Ok(_) => panic!("a failing redirect hop produced an image"),
        Err(err) => err.error.message,
    };
    (url, message)
}

/// [`redirected_fetch_outcome`]'s error, which names the submitted URL.
async fn redirected_fetch_error(hop: &str) -> String {
    let (url, message) = redirected_fetch_outcome(hop).await;
    // The URL the client sent is its own to see.
    assert!(message.contains(&url), "{message}");
    message
}

#[tokio::test]
async fn redirect_hop_transport_error_omits_hop_url() {
    // The hop is named by the origin's `Location`, which an attacker controls;
    // reqwest's error text carries it unless it is stripped (issue #2728).
    let _private = allow_private_media_urls_in_tests();
    let closed = TcpListener::bind("127.0.0.1:0").await.expect("bind probe");
    let closed_port = closed.local_addr().expect("probe addr").port();
    drop(closed);
    let message =
        redirected_fetch_error(&format!("http://127.0.0.1:{closed_port}/hop-secret-path")).await;
    assert!(
        !message.contains("hop-secret-path"),
        "the redirect hop URL leaked: {message}"
    );
}

#[tokio::test]
async fn redirect_hop_error_status_omits_hop_url() {
    let _private = allow_private_media_urls_in_tests();
    let (hop_addr, hop_task) = serve_canned(
        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned(),
    )
    .await;
    let message = redirected_fetch_error(&format!("http://{hop_addr}/hop-secret-path")).await;
    hop_task.abort();
    assert!(message.contains("404"), "{message}");
    assert!(
        !message.contains("hop-secret-path"),
        "the redirect hop URL leaked: {message}"
    );
}

#[tokio::test]
async fn too_many_redirects_omits_hop_url() {
    // Every answer redirects again, so the chain runs out of requests on a hop
    // the origin's `Location` named (issue #2728).
    let _private = allow_private_media_urls_in_tests();
    let (_, message) = redirected_fetch_outcome("/hop-secret-path").await;
    assert!(message.contains("redirect chain exceeded"), "{message}");
    assert!(
        !message.contains("hop-secret-path"),
        "the redirect hop URL leaked: {message}"
    );
}

#[tokio::test]
async fn redirect_hop_body_read_error_omits_hop_url() {
    // The hop promises more bytes than it sends and closes, so the failure
    // comes from the body read rather than from the status line (issue #2728).
    let _private = allow_private_media_urls_in_tests();
    let (hop_addr, hop_task) = serve_canned(
        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 4096\r\n\
         Connection: close\r\n\r\npartial"
            .to_owned(),
    )
    .await;
    let message = redirected_fetch_error(&format!("http://{hop_addr}/hop-secret-path")).await;
    hop_task.abort();
    assert!(message.contains("response body"), "{message}");
    assert!(
        !message.contains("hop-secret-path"),
        "the redirect hop URL leaked: {message}"
    );
}

#[tokio::test]
async fn redirect_to_a_non_http_scheme_omits_the_scheme() {
    // The scheme of a redirect target comes from the origin's `Location`, so
    // echoing it would let the origin plant a chosen token in the client's
    // error (issue #2764).
    // A refusal renders without the submitted URL, as in
    // `too_many_redirects_omits_hop_url`, so the URL-echo helper does not apply.
    let _private = allow_private_media_urls_in_tests();
    let (_, message) = redirected_fetch_outcome("secret-token-123://x/y").await;
    assert!(
        !message.contains("secret-token-123"),
        "the redirect target's scheme leaked: {message}"
    );
    assert!(
        message.contains("redirect target uses an unsupported scheme"),
        "{message}"
    );
}
