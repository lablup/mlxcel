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

//! `POST /v1/responses/{id}/cancel` on a background response (issue #2899):
//! the reply is the final Response object, byte for byte what a following
//! `GET` returns, whichever of the cancel and the job records the terminal
//! object first.

use std::sync::atomic::AtomicUsize;

use axum::body::Bytes;

use super::*;
use crate::server::routes::responses_background::{
    BACKGROUND_CANCEL_WAIT, cancel_background_response,
};

impl Harness {
    /// Send one request and return the status with the raw body bytes.
    async fn bytes(&self, method: Method, path: &str) -> (StatusCode, Bytes) {
        let response = self.raw(method, path, None).await;
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body collects");
        (status, bytes)
    }

    async fn cancel(&self, id: &str) -> (StatusCode, Bytes) {
        self.bytes(Method::POST, &format!("/v1/responses/{id}/cancel"))
            .await
    }

    async fn get_bytes(&self, id: &str) -> (StatusCode, Bytes) {
        self.bytes(Method::GET, &format!("/v1/responses/{id}"))
            .await
    }

    /// Wait until the job's task and driver are gone: the cancel
    /// registration is retired last, when the recorder is dropped.
    async fn job_gone(&self, id: &str) {
        tokio::time::timeout(WAIT, async {
            loop {
                let registered = match self.store.in_flight_write() {
                    Ok(registry) => registry.contains_key(id),
                    Err(poisoned) => poisoned.into_inner().contains_key(id),
                };
                if !registered {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the job of {id} never ended"));
    }

    /// Wait until the scripted generation `index` reached the model worker.
    async fn generation_dispatched(&self, index: usize) {
        tokio::time::timeout(WAIT, async {
            while self.handle.cancellation_flag(index).is_none() {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("the generation is dispatched");
    }

    /// Wait until the scripted generation `index` saw its cancellation.
    async fn generation_cancelled(&self, index: usize) {
        tokio::time::timeout(WAIT, async {
            loop {
                if let Some(flag) = self.handle.cancellation_flag(index)
                    && flag.load(Ordering::Acquire)
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("the generation is cancelled");
    }

    /// Cancel `id` through the route while the scripted generation is
    /// blocked, then let `script` end the generation once the cancel reached
    /// its token. Returns the cancel reply.
    async fn cancel_then(
        &self,
        id: &str,
        index: usize,
        script: impl FnOnce(),
    ) -> (StatusCode, Bytes) {
        let script = async {
            self.generation_cancelled(index).await;
            script();
        };
        let (reply, ()) = tokio::join!(self.cancel(id), script);
        reply
    }
}

fn json_of(bytes: &Bytes) -> Value {
    serde_json::from_slice(bytes).expect("body is JSON")
}

fn body_with(patch: Value) -> Value {
    let mut body = background_body();
    for (key, value) in patch.as_object().expect("patch is an object") {
        body[key] = value.clone();
    }
    body
}

#[tokio::test]
async fn cancel_while_queued_returns_the_full_cancelled_response() {
    let h = harness();
    // Nothing is scripted: the generation is dispatched and the job stays
    // `queued`.
    let id = h.create_background(background_body()).await;
    h.generation_dispatched(0).await;
    assert_eq!(h.get(&id).await.1["status"], "queued");

    let (status, reply) = h.cancel(&id).await;
    assert_eq!(status, StatusCode::OK);
    let cancelled = json_of(&reply);
    assert_eq!(cancelled["id"], id.as_str(), "{cancelled}");
    assert_eq!(cancelled["object"], "response", "{cancelled}");
    assert_eq!(cancelled["status"], "cancelled", "{cancelled}");
    assert_eq!(cancelled["background"], true, "{cancelled}");
    assert_eq!(cancelled["output"], json!([]), "{cancelled}");
    assert!(cancelled["completed_at"].is_null(), "{cancelled}");
    assert!(cancelled["error"].is_null(), "{cancelled}");
    assert_eq!(cancelled["model"], "route-test-model", "{cancelled}");
    assert_eq!(cancelled["usage"]["total_tokens"], 0, "{cancelled}");
    // The slot is free before the reply, and GET returns the same bytes.
    assert_eq!(h.store.active_background_jobs(), 0);
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply.clone()));
    assert_eq!(h.cancel(&id).await, (StatusCode::OK, reply.clone()));

    // The generation saw the cancel; what it is fed afterwards is dropped.
    h.generation_cancelled(0).await;
    h.handle.token("late");
    h.handle.finish();
    h.job_gone(&id).await;
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply));
    assert_eq!(h.store.active_background_jobs(), 0);
}

#[tokio::test]
async fn a_job_cancelled_before_it_starts_never_generates() {
    let h = harness();
    let generations = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&generations);
    let queued = spawn_background_job_with(h.job(background_body()), move |_run, _c, _sink| {
        counter.fetch_add(1, Ordering::SeqCst);
    })
    .expect("job is admitted");
    // The queued path never suspends, so on this single-threaded runtime
    // the job task has not run yet.
    let reply = cancel_background_response(&h.store, &queued.id, BACKGROUND_CANCEL_WAIT)
        .await
        .expect("a background response");
    assert_eq!(reply.status(), StatusCode::OK);
    assert_eq!(h.store.active_background_jobs(), 0);

    h.job_gone(&queued.id).await;
    assert_eq!(generations.load(Ordering::SeqCst), 0, "the driver ran");
    assert!(
        h.handle.cancellation_flag(0).is_none(),
        "nothing dispatched"
    );
    let stored = h.get(&queued.id).await.1;
    assert_eq!(stored["status"], "cancelled", "{stored}");
    assert_eq!(stored["output"], json!([]), "{stored}");
}

#[tokio::test]
async fn cancel_while_in_progress_returns_the_partial_output() {
    let h = harness();
    let id = h
        .create_background(body_with(json!({"conversation": "conv_cancel"})))
        .await;
    h.handle.token("Hel");
    h.poll_until(&id, "in_progress").await;

    let (status, reply) = h
        .cancel_then(&id, 0, || {
            h.handle.token("dropped");
            h.handle.finish();
        })
        .await;
    assert_eq!(status, StatusCode::OK);
    let cancelled = json_of(&reply);
    assert_eq!(cancelled["status"], "cancelled", "{cancelled}");
    assert_eq!(cancelled["background"], true, "{cancelled}");
    assert_eq!(cancelled["output_text"], "Hel", "{cancelled}");
    assert_eq!(cancelled["output"][0]["type"], "message", "{cancelled}");
    assert!(cancelled["completed_at"].is_null(), "{cancelled}");
    assert_eq!(h.store.active_background_jobs(), 0);

    // GET and a repeat cancel return the same bytes.
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply.clone()));
    for _ in 0..2 {
        assert_eq!(h.cancel(&id).await, (StatusCode::OK, reply.clone()));
    }
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply));
    // The partial turn reached the conversation exactly once.
    h.job_gone(&id).await;
    let conversations = h.state.conversation_store.as_ref().expect("store");
    let transcript = conversations.get("conv_cancel").expect("conversation");
    assert_eq!(transcript.items.len(), 2);
}

#[tokio::test]
async fn cancel_after_completed_returns_400_and_leaves_the_result() {
    let h = harness();
    for store in [true, false] {
        let id = h
            .create_background(body_with(json!({"store": store})))
            .await;
        h.handle.token("Hello.");
        h.handle.finish();
        assert_eq!(h.terminal(&id).await["status"], "completed");
        let before = h.get_bytes(&id).await;

        for _ in 0..2 {
            let (status, reply) = h.cancel(&id).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "store: {store}");
            let reply = json_of(&reply);
            assert_eq!(reply["error"]["type"], "invalid_request_error", "{reply}");
            assert_eq!(
                reply["error"]["message"],
                format!("response '{id}' has already finished and cannot be cancelled")
            );
        }
        assert_eq!(h.get_bytes(&id).await, before);
        assert_eq!(h.store.active_background_jobs(), 0);
    }
}

#[tokio::test]
async fn a_failed_job_refuses_cancel_and_an_unknown_id_is_not_found() {
    let h = harness();
    let id = h.create_background(background_body()).await;
    h.handle.fail("boom");
    assert_eq!(h.terminal(&id).await["status"], "failed");
    let before = h.get_bytes(&id).await;
    assert_eq!(h.cancel(&id).await.0, StatusCode::BAD_REQUEST);
    assert_eq!(h.get_bytes(&id).await, before);

    let (status, reply) = h.cancel("resp_missing").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json_of(&reply)["error"]["type"], "not_found");
    assert_eq!(h.store.active_background_jobs(), 0);
}

#[tokio::test]
async fn a_store_false_job_cancels_like_a_stored_one() {
    let h = harness_with(ResponsesStoreConfig {
        background_ephemeral_ttl: Duration::from_millis(300),
        ..ResponsesStoreConfig::default()
    });
    let id = h
        .create_background(body_with(json!({"store": false})))
        .await;
    h.handle.token("Hel");
    h.poll_until(&id, "in_progress").await;

    let (status, reply) = h.cancel_then(&id, 0, || h.handle.finish()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json_of(&reply)["status"], "cancelled");
    assert_eq!(json_of(&reply)["output_text"], "Hel");
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply.clone()));
    assert_eq!(h.cancel(&id).await, (StatusCode::OK, reply));
    assert_eq!(h.store.active_background_jobs(), 0);

    // Readable for the retention window only.
    tokio::time::sleep(Duration::from_millis(350)).await;
    assert_eq!(h.get(&id).await.0, StatusCode::NOT_FOUND);
    assert_eq!(h.cancel(&id).await.0, StatusCode::NOT_FOUND);
}

/// The cancel and a job that records `completed` on its own thread race for
/// the terminal write. Whoever wins, the cancel reply and GET agree.
#[tokio::test]
async fn a_cancel_racing_completion_always_agrees_with_get() {
    let h = harness();
    let (mut cancelled, mut refused) = (0, 0);
    for round in 0..200 {
        let store = Arc::clone(&h.store);
        let job = h.job(background_body());
        let id = job.response_id.clone();
        let job_id = id.clone();
        spawn_background_job_with(job, move |_run, _c, sink| {
            // Half the rounds reach `in_progress` first, so both the
            // immediate (`queued`) and the waiting cancel path race.
            if round % 2 == 0 {
                sink.on_started();
            }
            let Some(mut terminal) = store.background_snapshot(&job_id) else {
                return;
            };
            sink.finish_registration();
            terminal.status = ResponseStatus::Completed;
            sink.record_terminal(&terminal);
        })
        .expect("job is admitted");
        // Let the job task reach the driver at a varying point.
        for _ in 0..round % 5 {
            tokio::task::yield_now().await;
        }

        let (status, reply) = h.cancel(&id).await;
        let (get_status, stored) = h.get_bytes(&id).await;
        assert_eq!(get_status, StatusCode::OK, "round {round}");
        match status {
            StatusCode::OK => {
                assert_eq!(json_of(&reply)["status"], "cancelled", "round {round}");
                assert_eq!(reply, stored, "round {round}");
                cancelled += 1;
            }
            StatusCode::BAD_REQUEST => {
                assert_eq!(json_of(&stored)["status"], "completed", "round {round}");
                refused += 1;
            }
            other => panic!("round {round}: unexpected cancel status {other}"),
        }
        assert_eq!(h.store.active_background_jobs(), 0, "round {round}");
        h.job_gone(&id).await;
        assert_eq!(h.get_bytes(&id).await.1, stored, "round {round}");
    }
    assert_eq!(cancelled + refused, 200);
}

/// A driver stuck in one long step never looks at its token. The cancel
/// gives up waiting, records `cancelled`, and the driver's late terminal
/// write changes nothing.
#[tokio::test]
async fn a_job_that_ignores_the_token_is_recorded_cancelled_after_the_wait() {
    let h = harness();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let store = Arc::clone(&h.store);
    let job = h.job(body_with(json!({"conversation": "conv_stuck"})));
    let id = job.response_id.clone();
    let job_id = id.clone();
    spawn_background_job_with(job, move |_run, _c, sink| {
        sink.on_started();
        let mut terminal = store.background_snapshot(&job_id).expect("job is active");
        let _ = release_rx.recv();
        sink.finish_registration();
        terminal.status = ResponseStatus::Completed;
        terminal.output_text = "late".to_string();
        sink.record_terminal(&terminal);
    })
    .expect("job is admitted");
    h.poll_until(&id, "in_progress").await;

    let started = std::time::Instant::now();
    let reply = cancel_background_response(&h.store, &id, Duration::from_millis(50))
        .await
        .expect("a background response");
    assert!(started.elapsed() >= Duration::from_millis(50));
    assert_eq!(reply.status(), StatusCode::OK);
    let reply = axum::body::to_bytes(reply.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body collects");
    let cancelled = json_of(&reply);
    assert_eq!(cancelled["status"], "cancelled", "{cancelled}");
    assert_eq!(cancelled["output"], json!([]), "{cancelled}");
    assert!(cancelled["completed_at"].is_null(), "{cancelled}");
    // The slot is free although the driver still runs.
    assert_eq!(h.store.active_background_jobs(), 0);
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply.clone()));

    release_tx.send(()).expect("driver is waiting");
    h.job_gone(&id).await;
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply.clone()));
    assert_eq!(h.cancel(&id).await, (StatusCode::OK, reply));
    // The discarded result did not reach the conversation either.
    let conversations = h.state.conversation_store.as_ref().expect("store");
    let appended = conversations
        .get("conv_stuck")
        .map_or(0, |transcript| transcript.items.len());
    assert_eq!(appended, 0);
}

/// A cancel that stops generation before the first token makes the provider
/// end the generation with an error. That is the cancel's doing, so the job
/// ends `cancelled`, not `failed`, with or without the acknowledgement.
#[tokio::test]
async fn a_cancel_that_ends_generation_with_an_error_is_recorded_cancelled() {
    let h = harness();
    // Token flipped without an acknowledgement (`request_background_cancel`):
    // the driver reports the error and the recorder turns it into `cancelled`.
    let id = h.create_background(background_body()).await;
    h.generation_dispatched(0).await;
    assert!(matches!(
        h.store.request_background_cancel(&id),
        crate::server::responses_store::BackgroundCancel::Requested
    ));
    h.handle.fail("Request cancelled: client disconnected");
    let done = h.terminal(&id).await;
    assert_eq!(done["status"], "cancelled", "{done}");
    assert!(done["error"].is_null(), "{done}");
    assert_eq!(done["output"], json!([]), "{done}");
    assert_eq!(h.store.active_background_jobs(), 0);
    h.job_gone(&id).await;

    // Through the route, mid-generation: the reply waits for the driver.
    let id = h.create_background(background_body()).await;
    h.handle.token("Hel");
    h.poll_until(&id, "in_progress").await;
    let (status, reply) = h
        .cancel_then(&id, 1, || {
            h.handle.fail("Request cancelled: client disconnected")
        })
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json_of(&reply)["status"], "cancelled");
    assert!(json_of(&reply)["error"].is_null());
    assert_eq!(h.get_bytes(&id).await, (StatusCode::OK, reply));
    assert_eq!(h.store.active_background_jobs(), 0);

    // An error nobody cancelled still fails the job.
    let id = h.create_background(background_body()).await;
    h.handle.fail("boom");
    assert_eq!(h.terminal(&id).await["status"], "failed");
}

/// A cancel of a `queued` job frees its slot at once, so the job's
/// pre-generation work (video decode, compaction, media fetch) must not run
/// on detached from any slot: it is dropped when the cancel records the
/// terminal object, and the cancel registration is retired.
#[tokio::test]
async fn a_queued_cancel_drops_the_pre_generation_work() {
    use super::super::{JobRecorder, queued_response, unless_cancelled};

    let h = harness();
    let job = h.job(background_body());
    let id = job.response_id.clone();
    let queued = queued_response(&job);
    let token = h
        .store
        .admit_background(queued.clone(), vec![], true)
        .expect("admit background job");
    h.store
        .in_flight_write()
        .expect("in-flight registry")
        .insert(id.clone(), token);
    let recorder = JobRecorder::new(Arc::clone(&h.store), id.clone(), queued.clone());

    // Work that finishes first is returned.
    assert_eq!(unless_cancelled(&recorder, async { 7 }).await, Some(7));
    assert_eq!(h.store.active_background_jobs(), 1);

    // Work that never finishes is dropped once the cancel lands.
    let dropped = Arc::new(AtomicUsize::new(0));
    struct OnDrop(Arc<AtomicUsize>);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let guard = OnDrop(Arc::clone(&dropped));
    let work = async move {
        let _guard = guard;
        std::future::pending::<()>().await
    };
    let cancel = async {
        tokio::task::yield_now().await;
        h.store.cancel_background(&id)
    };
    let (outcome, cancel) = tokio::time::timeout(WAIT, async {
        tokio::join!(unless_cancelled(&recorder, work), cancel)
    })
    .await
    .expect("the cancel ends the pending work");
    assert_eq!(outcome, None);
    assert!(matches!(
        cancel,
        crate::server::responses_store::BackgroundCancel::Terminal(ref response)
            if response.status == ResponseStatus::Cancelled
    ));
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(h.store.active_background_jobs(), 0);
    let registered = h
        .store
        .in_flight_write()
        .expect("in-flight registry")
        .contains_key(&id);
    assert!(!registered, "the cancel registration is retired");
}
