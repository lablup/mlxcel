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

//! [`ConnectionDrain`] accounting (#3483). The signal side is covered by the
//! process-level `tests/server_graceful_shutdown.rs`, since a test cannot
//! raise SIGTERM in the shared test process.

use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;

use super::{ConnectionDrain, ShutdownSignal};

#[test]
fn signal_names_match_the_log_contract() {
    assert_eq!(ShutdownSignal::Terminate.name(), "SIGTERM");
    assert_eq!(ShutdownSignal::Interrupt.name(), "SIGINT");
}

#[tokio::test]
async fn an_empty_drain_returns_at_once() {
    let drain = ConnectionDrain::new(CancellationToken::new());
    let started = Instant::now();
    assert_eq!(drain.drain(Duration::from_secs(30)).await, 0);
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[tokio::test]
async fn connections_that_honour_the_graceful_signal_are_not_aborted() {
    let mut drain = ConnectionDrain::new(CancellationToken::new());
    for _ in 0..3 {
        let graceful = drain.graceful();
        drain.spawn(async move {
            graceful.cancelled().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        });
    }
    assert_eq!(drain.drain(Duration::from_secs(30)).await, 0);
}

#[tokio::test]
async fn connections_still_open_at_the_deadline_are_aborted_and_counted() {
    let mut drain = ConnectionDrain::new(CancellationToken::new());
    let graceful = drain.graceful();
    drain.spawn(async move {
        graceful.cancelled().await;
    });
    for _ in 0..2 {
        drain.spawn(std::future::pending::<()>());
    }
    let started = Instant::now();
    assert_eq!(drain.drain(Duration::from_millis(200)).await, 2);
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(200) && elapsed < Duration::from_secs(2),
        "{elapsed:?}"
    );
}

#[tokio::test]
async fn finished_connections_are_reaped_while_accepting() {
    let mut drain = ConnectionDrain::new(CancellationToken::new());
    for _ in 0..8 {
        drain.spawn(async {});
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    drain.spawn(async {});
    assert!(
        drain.tasks.len() <= 2,
        "completed connections must not accumulate: {}",
        drain.tasks.len()
    );
}
