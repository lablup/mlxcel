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

//! Per-connection socket read/write timeouts for `--timeout` (#1432).
//!
//! `llama-server` b10621 hands one `--timeout` value to cpp-httplib's
//! `set_read_timeout` / `set_write_timeout`, which arm a `select()` deadline
//! around every individual socket read and write. A connection is dropped
//! when one operation stays blocked for longer than the timeout; the total
//! lifetime of the connection is not bounded, so a long streaming response
//! that keeps producing bytes is never cut off by it.
//!
//! [`TimeoutIo`] reproduces that shape on the Tokio side: it wraps the
//! accepted stream and arms a deadline only while the underlying I/O is
//! `Pending`. A ready read or write clears the deadline, so the timer resets
//! per operation exactly as the upstream `select()` does. That is what makes
//! this control independent of the decode watchdog
//! (`--decode-timeout`): a stalled model produces no socket activity to time
//! out, and a stalled client produces no decode work to cancel.
//!
//! The read half additionally consults a per-connection
//! [`ReadBudget`](super::read_budget::ReadBudget). hyper keeps polling the read
//! side while a response is pending in order to notice a client that hung up,
//! which upstream's sequential read-handle-write loop never does; without the
//! gate, an unconditional read deadline would cut off any generation slower
//! than `--timeout`. The gate is armed exactly while the server is waiting for
//! request bytes, which is the window upstream's `read_timeout` covers.
//!
//! A zero timeout is honored rather than reinterpreted as "disabled", because
//! b10621 passes zero straight to `select()` where it closes the connection
//! as soon as an operation would block. See
//! [`crate::server::transport::HttpTimeouts::from_secs`].

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::time::{Sleep, sleep};

use super::read_budget::ReadBudget;
use super::transport::HttpTimeouts;

/// Wraps an accepted connection so each read and write is bounded by the
/// configured `--timeout`.
///
/// `S: Unpin` is required and satisfied by every stream mlxcel accepts
/// (`TcpStream`, `UnixStream`, `tokio_rustls::server::TlsStream<TcpStream>`),
/// which keeps the projection free of a `pin-project` dependency.
///
/// Used by: `crate::server::listen` (TCP, Unix socket and TLS accept loops).
#[derive(Debug)]
pub(crate) struct TimeoutIo<S> {
    inner: S,
    timeouts: HttpTimeouts,
    budget: std::sync::Arc<ReadBudget>,
    read_deadline: Option<Pin<Box<Sleep>>>,
    write_deadline: Option<Pin<Box<Sleep>>>,
}

impl<S> TimeoutIo<S> {
    pub(crate) fn new(
        inner: S,
        timeouts: HttpTimeouts,
        budget: std::sync::Arc<ReadBudget>,
    ) -> Self {
        Self {
            inner,
            timeouts,
            budget,
            read_deadline: None,
            write_deadline: None,
        }
    }
}

/// Arm (or poll) the deadline for a `Pending` I/O operation.
///
/// Returns `Poll::Ready(Err(TimedOut))` once the deadline elapses, and
/// `Poll::Pending` while it has not. The caller clears `slot` whenever the
/// wrapped operation makes progress, which is what resets the per-operation
/// timer.
fn poll_deadline(
    slot: &mut Option<Pin<Box<Sleep>>>,
    timeout: Duration,
    direction: &'static str,
    cx: &mut Context<'_>,
) -> Poll<io::Result<()>> {
    let deadline = slot.get_or_insert_with(|| Box::pin(sleep(timeout)));
    match deadline.as_mut().poll(cx) {
        Poll::Ready(()) => {
            *slot = None;
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "HTTP {direction} timed out after {}s; raise --timeout / LLAMA_ARG_TIMEOUT to \
                     allow slower clients",
                    timeout.as_secs()
                ),
            )))
        }
        Poll::Pending => Poll::Pending,
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for TimeoutIo<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_read(cx, buf) {
            Poll::Ready(result) => {
                this.read_deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending if this.budget.is_armed() => {
                poll_deadline(&mut this.read_deadline, this.timeouts.read, "read", cx)
            }
            Poll::Pending => {
                // A request is being handled or its response is still being
                // written. Upstream is not reading during that window, so
                // neither is the read budget.
                this.read_deadline = None;
                Poll::Pending
            }
        }
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for TimeoutIo<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_write(cx, data) {
            Poll::Ready(result) => {
                this.write_deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending => {
                match poll_deadline(&mut this.write_deadline, this.timeouts.write, "write", cx) {
                    // `poll_deadline` only ever completes with the timeout error;
                    // the `Ok` arm is unreachable and must not be reported as a
                    // zero-length write, which would look like a closed peer.
                    Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                    Poll::Ready(Ok(())) => Poll::Pending,
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_flush(cx) {
            Poll::Ready(result) => {
                this.write_deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending => {
                poll_deadline(&mut this.write_deadline, this.timeouts.write, "write", cx)
            }
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // Shutdown is not part of the read/write budget: a peer that has gone
        // away must not keep the task alive, but timing the FIN out would
        // convert an ordinary close into a logged error.
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_write_vectored(cx, bufs) {
            Poll::Ready(result) => {
                this.write_deadline = None;
                Poll::Ready(result)
            }
            Poll::Pending => {
                match poll_deadline(&mut this.write_deadline, this.timeouts.write, "write", cx) {
                    // `poll_deadline` only ever completes with the timeout error;
                    // the `Ok` arm is unreachable and must not be reported as a
                    // zero-length write, which would look like a closed peer.
                    Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                    Poll::Ready(Ok(())) => Poll::Pending,
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

#[cfg(test)]
#[path = "http_timeout_tests.rs"]
mod http_timeout_tests;
