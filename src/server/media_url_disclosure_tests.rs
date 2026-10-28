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
//! through its hosts file, so no external network is needed.

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
    deny_private_media_urls_in_tests, send_following_redirects,
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
