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

//! Redirect following for remote media fetches (check 3 in the `media_net`
//! module doc, issue #2693).
//!
//! The shared media client is built with `reqwest::redirect::Policy::none()`,
//! and this helper follows the chain instead, so every hop goes through the
//! full [`ensure_media_url_allowed`] check (shape, local resolution, address)
//! before a byte is sent toward it. That holds whether or not a proxy carries
//! the hop: behind a proxy reqwest resolves only the proxy's own host, so a
//! hostname hop would otherwise be resolved by the proxy alone and never be
//! checked by mlxcel.

use std::future::Future;
use std::time::Duration;

use tokio::time::Instant;

use super::{MediaUrlError, ensure_media_url_allowed};

/// Total deadline for one media fetch: the whole redirect chain and the final
/// response's body read share it.
pub(crate) const MEDIA_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Requests one fetch may send, the first included. A redirect answered to the
/// last of them fails with [`MediaUrlError::TooManyRedirects`].
pub(crate) const MAX_MEDIA_REQUESTS: usize = 5;

/// Why [`send_following_redirects`] produced no response.
#[derive(Debug)]
pub(crate) enum MediaSendError {
    /// Per-hop policy refusal, `TooManyRedirects`, or a [`MediaUrlError`] found
    /// by [`MediaUrlError::find_in`] inside a reqwest error (`PolicyResolver`).
    Refused(MediaUrlError),
    /// Any other failure to send a hop or read its response head, with its URL
    /// stripped so the error text names no hop.
    Transport(reqwest::Error),
    /// A followed 3xx whose `Location` is missing, not UTF-8, or does not join.
    BadRedirect { status: reqwest::StatusCode },
    /// The chain deadline ran out while a redirect target was being resolved
    /// and checked. A timeout, not a refusal: the client is told the deadline
    /// ran out, without the host. `host` is for debug logging only.
    DeadlineExceeded { host: String },
}

/// Send `first` and follow its redirects within [`MEDIA_FETCH_TIMEOUT`].
///
/// `first` must already have passed [`ensure_media_url_allowed`] at the call
/// site; every later hop is checked here.
///
/// # Errors
///
/// See [`MediaSendError`].
pub(crate) async fn send_following_redirects(
    client: &reqwest::Client,
    first: reqwest::Url,
) -> Result<reqwest::Response, MediaSendError> {
    send_following_redirects_within(client, first, MEDIA_FETCH_TIMEOUT).await
}

/// [`send_following_redirects`] with an explicit chain deadline.
///
/// One deadline covers the whole chain: each hop is sent with whatever is left
/// of `budget` as its total timeout, which also bounds the returned response's
/// body read, so the worst case stays `budget` however many hops there are.
/// Only 301, 302, 303, 307 and 308 are followed, always as a body-less `GET`
/// with no `Referer`; any other status, 3xx included, is returned unchanged.
///
/// # Errors
///
/// See [`MediaSendError`].
pub(crate) async fn send_following_redirects_within(
    client: &reqwest::Client,
    first: reqwest::Url,
    budget: Duration,
) -> Result<reqwest::Response, MediaSendError> {
    let deadline = Instant::now() + budget;
    let mut url = first;
    let mut sent = 0usize;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let response = client
            .get(url)
            .timeout(remaining)
            .send()
            .await
            .map_err(transport_or_refusal)?;
        sent += 1;
        let status = response.status();
        if !is_followed_redirect(status) {
            return Ok(response);
        }
        let next = redirect_target(&response).ok_or(MediaSendError::BadRedirect { status })?;
        // The intermediate response is never read. Dropping it with a body
        // still unread closes that connection instead of draining a body the
        // origin chose; an empty redirect body may let the connection return
        // to the pool, which is harmless.
        drop(response);
        if sent >= MAX_MEDIA_REQUESTS {
            return Err(MediaSendError::Refused(MediaUrlError::TooManyRedirects {
                url: next.to_string(),
            }));
        }
        url = check_hop(next, deadline).await?;
    }
}

/// Statuses whose `Location` is followed, as reqwest's own policy did.
fn is_followed_redirect(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 301 | 302 | 303 | 307 | 308)
}

/// The URL a redirect names, joined against the URL that answered it.
fn redirect_target(response: &reqwest::Response) -> Option<reqwest::Url> {
    let location = response
        .headers()
        .get(reqwest::header::LOCATION)?
        .to_str()
        .ok()?;
    response.url().join(location).ok()
}

/// Run the full policy check on a redirect target, inside the chain deadline.
///
/// The local lookup is bounded by the deadline as well, so a slow resolver
/// cannot stretch the chain past it. A refused scheme or malformed target is
/// reported through [`MediaUrlError::for_redirect_target`], so the scheme or
/// URL the origin's `Location` chose never reaches the client (issue #2764).
/// See [`check_hop_with`] for how each outcome is reported.
async fn check_hop(next: reqwest::Url, deadline: Instant) -> Result<reqwest::Url, MediaSendError> {
    let host = next.host_str().unwrap_or_default();
    let check = async {
        ensure_media_url_allowed(next.as_str())
            .await
            .map_err(MediaUrlError::for_redirect_target)
    };
    check_hop_with(host, deadline, check).await
}

/// [`check_hop`] with the policy check supplied, so tests can stall it.
///
/// A refusal from `check`, an unresolvable host included, is
/// [`MediaSendError::Refused`]. Running out of `deadline` before `check`
/// finishes is [`MediaSendError::DeadlineExceeded`], so the operator log and
/// the client see a timeout rather than a refusal (issue #2713). The lookup's
/// own bound inside `check` still reports `Unresolvable`.
pub(crate) async fn check_hop_with<F>(
    host: &str,
    deadline: Instant,
    check: F,
) -> Result<reqwest::Url, MediaSendError>
where
    F: Future<Output = Result<reqwest::Url, MediaUrlError>>,
{
    match tokio::time::timeout_at(deadline, check).await {
        Ok(Ok(allowed)) => Ok(allowed),
        Ok(Err(err)) => Err(MediaSendError::Refused(err)),
        Err(_) => Err(MediaSendError::DeadlineExceeded {
            host: host.to_owned(),
        }),
    }
}

/// A send failure, reported as the policy refusal it wraps when there is one.
///
/// A transport error loses its URL before it is stored: after a redirect that
/// URL is the hop the origin's `Location` named, which an attacker controls
/// and the client must not be shown (issue #2728). The hop goes to the log at
/// `debug` instead.
fn transport_or_refusal(err: reqwest::Error) -> MediaSendError {
    match MediaUrlError::find_in(&err) {
        Some(refusal) => MediaSendError::Refused(refusal.clone()),
        None => {
            tracing::debug!(hop = ?err.url(), "media fetch transport error: {err}");
            MediaSendError::Transport(err.without_url())
        }
    }
}
