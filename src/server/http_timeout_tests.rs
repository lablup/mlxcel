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

//! Unit tests for the per-connection socket timeout adapter (#1432).
//!
//! Tokio's paused clock is used throughout so the tests assert the deadline
//! logic rather than wall-clock scheduling: a real `sleep` would make an
//! hour-long default untestable and a one-second timeout flaky under load.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};

use super::TimeoutIo;
use crate::server::read_budget::ReadBudget;
use crate::server::transport::HttpTimeouts;

/// A stream that never becomes ready in either direction, standing in for a
/// client that opened a connection and then stopped sending, or a peer whose
/// receive window is full.
#[derive(Debug, Default)]
struct StalledStream;

impl tokio::io::AsyncRead for StalledStream {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Poll::Pending
    }
}

impl AsyncWrite for StalledStream {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _data: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Pending
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Pending
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

fn timeouts(secs: u64) -> HttpTimeouts {
    HttpTimeouts::from_secs(secs)
}

#[tokio::test(start_paused = true)]
async fn a_stalled_read_fails_with_timed_out_after_the_configured_timeout() {
    let mut io = TimeoutIo::new(StalledStream, timeouts(7), ReadBudget::new());
    let mut buf = [0u8; 16];
    let err = io
        .read(&mut buf)
        .await
        .expect_err("a stalled read must time out");
    assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    assert!(
        err.to_string().contains("--timeout"),
        "the diagnostic must name the flag that controls it: {err}"
    );
    assert!(err.to_string().contains("7s"), "{err}");
}

#[tokio::test(start_paused = true)]
async fn a_stalled_write_fails_with_timed_out_after_the_configured_timeout() {
    let mut io = TimeoutIo::new(StalledStream, timeouts(11), ReadBudget::new());
    let err = io
        .write(b"hello")
        .await
        .expect_err("a stalled write must time out");
    assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    assert!(err.to_string().contains("11s"), "{err}");
}

#[tokio::test(start_paused = true)]
async fn a_read_does_not_time_out_before_the_deadline() {
    let mut io = TimeoutIo::new(StalledStream, timeouts(60), ReadBudget::new());
    let mut buf = [0u8; 4];
    let outcome = tokio::time::timeout(Duration::from_secs(30), io.read(&mut buf)).await;
    assert!(
        outcome.is_err(),
        "the read must still be pending 30s into a 60s budget, not already failed"
    );
}

#[tokio::test(start_paused = true)]
async fn a_zero_timeout_closes_the_connection_on_the_first_blocking_read() {
    // b10621 passes `--timeout 0` straight to select(), where it closes the
    // connection as soon as an operation would block. mlxcel must not
    // reinterpret it as "no timeout".
    let mut io = TimeoutIo::new(StalledStream, timeouts(0), ReadBudget::new());
    let mut buf = [0u8; 4];
    let err = io.read(&mut buf).await.expect_err("0 must not disable");
    assert_eq!(err.kind(), io::ErrorKind::TimedOut);
}

#[tokio::test(start_paused = true)]
async fn progress_resets_the_per_operation_deadline() {
    // The upstream select() deadline is per operation, not per connection: a
    // long response that keeps producing bytes is never cut off. Feed the
    // reader one byte just before each deadline and confirm the connection
    // survives well past a single timeout window.
    let (mut client, server) = tokio::io::duplex(64);
    let mut io = TimeoutIo::new(server, timeouts(10), ReadBudget::new());

    let writer = tokio::spawn(async move {
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_secs(9)).await;
            client.write_all(b"x").await.expect("write");
            client.flush().await.expect("flush");
        }
        // Hold the client open so the reader below blocks on the deadline
        // rather than seeing EOF.
        tokio::time::sleep(Duration::from_secs(3600)).await;
        drop(client);
    });

    let mut seen = 0usize;
    let mut buf = [0u8; 1];
    for _ in 0..5 {
        let n = io.read(&mut buf).await.expect("read within its own window");
        assert_eq!(n, 1);
        seen += n;
    }
    assert_eq!(
        seen, 5,
        "45 simulated seconds of traffic must survive a 10s per-operation timeout"
    );
    writer.abort();
}

#[tokio::test(start_paused = true)]
async fn shutdown_is_not_bounded_by_the_write_timeout() {
    // A FIN on a peer that has gone away must close cleanly rather than being
    // logged as a timeout error.
    let mut io = TimeoutIo::new(StalledStream, timeouts(1), ReadBudget::new());
    io.shutdown().await.expect("shutdown completes");
}

#[tokio::test(start_paused = true)]
async fn a_stood_down_read_budget_never_times_out() {
    // While a request is being handled or its response written, upstream is
    // not reading, so neither is the read deadline. This is what lets a
    // generation slower than `--timeout` complete.
    let budget = ReadBudget::new();
    crate::server::read_budget::stand_down_for_tests(&budget);
    let mut io = TimeoutIo::new(StalledStream, timeouts(1), budget);
    let mut buf = [0u8; 4];
    let outcome = tokio::time::timeout(Duration::from_secs(3600), io.read(&mut buf)).await;
    assert!(
        outcome.is_err(),
        "a stood-down read budget must leave the read pending, not time it out"
    );
}
