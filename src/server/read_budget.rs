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

//! Gating for the `--timeout` socket read deadline (#1432).
//!
//! `llama-server` b10621 runs one thread per connection and does read, handle,
//! write strictly in sequence, so its `read_timeout` covers exactly the window
//! in which the server is waiting for request bytes: the request line, the
//! headers, and the body. It never covers the time the handler spends
//! generating, because during that window nothing is being read.
//!
//! hyper is not sequential. It keeps polling the read half while a response is
//! pending, to notice a client that hung up, so an unconditional deadline on
//! [`TimeoutIo`](super::http_timeout::TimeoutIo) would cut off any generation
//! slower than `--timeout`, which is the opposite of what the flag means.
//!
//! [`ReadBudget`] is the per-connection switch that restores the upstream
//! window. It starts armed, is disarmed the moment the request body finishes
//! (or is dropped, which is what a body-less request does), and re-arms when
//! the response body is finished or dropped, so a keep-alive connection is
//! bounded again while it waits for the next request. [`BudgetBody`] is the
//! body wrapper that drives those two transitions.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use http_body::{Body, Frame, SizeHint};

/// Per-connection switch controlling whether the socket read deadline is armed.
///
/// Shared between the connection's [`TimeoutIo`](super::http_timeout::TimeoutIo)
/// and the two [`BudgetBody`] wrappers on that connection.
#[derive(Debug, Default)]
pub(crate) struct ReadBudget {
    armed: AtomicBool,
}

impl ReadBudget {
    /// A fresh connection is waiting for a request, so the deadline is armed.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            armed: AtomicBool::new(true),
        })
    }

    pub(crate) fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    fn set(&self, armed: bool) {
        self.armed.store(armed, Ordering::Relaxed);
    }
}

/// Wraps a request or response body and flips its connection's
/// [`ReadBudget`] once the body ends or is dropped.
///
/// `on_finish` is `false` for a request body (the server has stopped reading,
/// so the read deadline must stand down for the handler and the response) and
/// `true` for a response body (the exchange is over, so the connection is
/// waiting for the next request again).
///
/// The transition fires on end-of-stream *and* on drop, because a request body
/// that a handler never polls, which is every GET, only ever reaches the drop
/// path. `set` is idempotent, so firing both times is harmless.
pub(crate) struct BudgetBody<B> {
    inner: B,
    budget: Arc<ReadBudget>,
    on_finish: bool,
}

impl<B> BudgetBody<B> {
    /// Wrap a request body: finishing it stands the read deadline down.
    pub(crate) fn request(inner: B, budget: Arc<ReadBudget>) -> Self {
        Self {
            inner,
            budget,
            on_finish: false,
        }
    }

    /// Wrap a response body: finishing it arms the read deadline again for the
    /// next request on a keep-alive connection.
    pub(crate) fn response(inner: B, budget: Arc<ReadBudget>) -> Self {
        Self {
            inner,
            budget,
            on_finish: true,
        }
    }
}

impl<B> Drop for BudgetBody<B> {
    fn drop(&mut self) {
        self.budget.set(self.on_finish);
    }
}

impl<B> Body for BudgetBody<B>
where
    B: Body + Unpin,
{
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if matches!(polled, Poll::Ready(None)) {
            this.budget.set(this.on_finish);
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// Stand the read budget down, as a finished request body does.
///
/// Test-only: production code reaches these transitions through
/// [`BudgetBody`] alone, so the switch stays inaccessible to a caller that
/// could otherwise disarm the timeout for the life of a connection.
#[cfg(test)]
pub(crate) fn stand_down_for_tests(budget: &ReadBudget) {
    budget.set(false);
}

#[cfg(test)]
#[path = "read_budget_tests.rs"]
mod read_budget_tests;
