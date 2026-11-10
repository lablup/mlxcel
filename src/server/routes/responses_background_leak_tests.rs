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

//! Leak bound for background responses (issue #2902): thousands of scripted
//! jobs of every kind leave no active job, no cancel registration, and no
//! retained bytes beyond the store budget, and nothing once the TTL passes.
//!
//! A child of the background route tests, for their harness.

use std::time::Instant;

use http_body_util::BodyExt;

use super::*;

const JOBS: usize = 2_000;
const MAX_BYTES: usize = 256 * 1024;
const TTL: Duration = Duration::from_millis(500);
const EPHEMERAL_TTL: Duration = Duration::from_millis(250);

/// The kinds of job the test cycles through, one generation each.
#[derive(Debug, Clone, Copy)]
enum Kind {
    Completed,
    Failed,
    CancelledQueued,
    StoreFalse,
    Streamed,
    StreamDropped,
    CancelledInProgress,
}

const KINDS: [Kind; 7] = [
    Kind::Completed,
    Kind::Failed,
    Kind::CancelledQueued,
    Kind::StoreFalse,
    Kind::Streamed,
    Kind::StreamDropped,
    Kind::CancelledInProgress,
];

impl Harness {
    /// Wait until the `index`-th generation reached the scripted worker and,
    /// with `cancelled`, until it saw its cancellation.
    async fn generation(&self, index: usize, cancelled: bool) {
        tokio::time::timeout(WAIT, async {
            loop {
                if let Some(flag) = self.handle.cancellation_flag(index)
                    && (!cancelled || flag.load(Ordering::Acquire))
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("generation {index} never got there"));
    }

    /// The terminal object of `id`, from the job's own hand-off: the stored
    /// entry may already be evicted by the small byte budget.
    async fn finished(&self, id: &str) -> crate::server::types::ResponseObject {
        self.store
            .wait_background_terminal(id, WAIT)
            .await
            .unwrap_or_else(|| panic!("response {id} never finished"))
    }

    async fn cancel_status(&self, id: &str) -> Value {
        let (status, reply) = self
            .send(Method::POST, &format!("/v1/responses/{id}/cancel"), None)
            .await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        reply
    }

    /// Run one job of `kind` to its terminal state; `index` is its
    /// generation's position in the scripted worker's dispatch order.
    async fn run_job(&self, kind: Kind, index: usize) {
        let mut body = background_body();
        let (id, status) = match kind {
            Kind::Completed | Kind::StoreFalse => {
                if matches!(kind, Kind::StoreFalse) {
                    body["store"] = json!(false);
                }
                self.handle.token("Hello.");
                self.handle.finish();
                let id = self.create_background(body).await;
                (id, ResponseStatus::Completed)
            }
            Kind::Failed => {
                self.handle.fail("boom");
                (self.create_background(body).await, ResponseStatus::Failed)
            }
            Kind::Streamed | Kind::StreamDropped => {
                body["stream"] = json!(true);
                self.handle.token("Hello.");
                self.handle.finish();
                let response = self.raw(Method::POST, "/v1/responses", Some(body)).await;
                assert_eq!(response.status(), StatusCode::OK);
                let mut stream = response.into_body();
                let mut text = String::new();
                // Read through the first event, which names the response.
                while !text.contains("\n\n") {
                    let frame = tokio::time::timeout(WAIT, stream.frame())
                        .await
                        .expect("a frame arrives in time")
                        .expect("the stream is open")
                        .expect("frame ok");
                    if let Some(data) = frame.data_ref() {
                        text.push_str(&String::from_utf8_lossy(data));
                    }
                }
                let start = text.find("\"id\":\"resp_").expect("response id") + 6;
                let end = start + text[start..].find('"').expect("id end");
                let id = text[start..end].to_string();
                if matches!(kind, Kind::Streamed) {
                    let rest = axum::body::to_bytes(stream, 1 << 20)
                        .await
                        .expect("stream collects");
                    let rest = String::from_utf8_lossy(&rest);
                    assert!(rest.contains("event: response.completed"), "{rest}");
                } else {
                    // A client that disconnects after the first event.
                    drop(stream);
                }
                (id, ResponseStatus::Completed)
            }
            Kind::CancelledQueued => {
                let id = self.create_background(body).await;
                self.generation(index, false).await;
                let reply = self.cancel_status(&id).await;
                assert_eq!(reply["status"], "cancelled", "{reply}");
                // The worker consumes steps until a finish, cancelled or not.
                self.handle.finish();
                (id, ResponseStatus::Cancelled)
            }
            Kind::CancelledInProgress => {
                let id = self.create_background(body).await;
                self.handle.token("Hel");
                tokio::time::timeout(WAIT, async {
                    while self.store.background_status(&id) != Some(ResponseStatus::InProgress) {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                })
                .await
                .expect("the job starts");
                let release = async {
                    self.generation(index, true).await;
                    self.handle.finish();
                };
                let (reply, ()) = tokio::join!(self.cancel_status(&id), release);
                assert_eq!(reply["status"], "cancelled", "{reply}");
                assert_eq!(reply["output_text"], "Hel", "{reply}");
                (id, ResponseStatus::Cancelled)
            }
        };
        let terminal = self.finished(&id).await;
        assert_eq!(terminal.status, status, "{kind:?} {id}");
    }
}

#[tokio::test]
async fn background_jobs_do_not_leak() {
    let h = harness_with(ResponsesStoreConfig {
        max_bytes: MAX_BYTES,
        ttl: TTL,
        background_ephemeral_ttl: EPHEMERAL_TTL,
        ..ResponsesStoreConfig::default()
    });
    let started = Instant::now();
    for index in 0..JOBS {
        h.run_job(KINDS[index % KINDS.len()], index).await;
        assert!(
            h.store.approximate_total_bytes() <= MAX_BYTES,
            "job {index}: {} bytes retained",
            h.store.approximate_total_bytes()
        );
    }
    eprintln!("{JOBS} background jobs in {:?}", started.elapsed());

    // Every job task is gone: the cancel registration is retired last.
    tokio::time::timeout(WAIT, async {
        loop {
            let registered = match h.store.in_flight_write() {
                Ok(registry) => registry.len(),
                Err(poisoned) => poisoned.into_inner().len(),
            };
            if registered == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .expect("every job retired its cancel registration");
    assert_eq!(h.store.active_background_jobs(), 0);
    assert!(h.store.approximate_total_bytes() <= MAX_BYTES);
    assert!(h.handle.cancellation_flag(JOBS - 1).is_some());
    assert!(
        h.handle.cancellation_flag(JOBS).is_none(),
        "one generation per job"
    );

    // Past the TTL a sweep leaves nothing behind, event logs included.
    tokio::time::sleep(TTL + Duration::from_millis(50)).await;
    assert!(h.store.get("resp_sweep").is_none());
    assert_eq!(h.store.len(), 0);
    assert_eq!(h.store.approximate_total_bytes(), 0);
}
