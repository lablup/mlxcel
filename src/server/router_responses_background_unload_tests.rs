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

//! What loading another model and an explicit unload do to a model with
//! running background Responses jobs (issue #2901): eviction leaves it
//! loaded, an explicit unload ends its jobs `failed`.

use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::json;
use tower::ServiceExt;

use super::{Harness, WAIT, assert_not_found};
use crate::server::router_lifecycle::OperationState;
use crate::server::router_models::RouterPoolError;

#[tokio::test]
async fn lru_eviction_skips_a_model_with_an_active_job() {
    let h = Harness::new(&["alpha", "beta", "gamma"], 2);
    h.load("alpha").await;
    h.load("beta").await;
    // alpha is the least recently used and has a running job; beta was used
    // later and has one too.
    let alpha_job = h.create_running("alpha").await;
    let beta_job = h.create_running("beta").await;
    assert!(h.pool.get("alpha").expect("alpha").is_busy_for_eviction());

    match h.pool.begin_load("gamma").await {
        Err(RouterPoolError::Capacity(message)) => {
            assert!(message.contains("every loaded model is busy"), "{message}");
        }
        other => panic!(
            "load must be refused: {:?}",
            other.map(|entry| entry.name.clone())
        ),
    }
    assert!(h.running("alpha") && h.running("beta"));
    assert_eq!(h.get(&alpha_job).await.1["status"], "in_progress");

    // beta's job ends: beta is evictable, the LRU alpha still is not.
    h.script("beta", |handle| handle.finish());
    h.poll_until(&beta_job, "completed").await;
    assert!(!h.pool.get("beta").expect("beta").is_busy_for_eviction());
    h.load("gamma").await;
    assert!(h.running("alpha"), "the busy LRU model stays loaded");
    assert!(!h.running("beta"), "the idle model was evicted");
    assert!(h.running("gamma"));

    // alpha's job survived the eviction pass; beta's result went with it.
    assert_eq!(h.get(&alpha_job).await.1["status"], "in_progress");
    let (status, body) = h.get(&beta_job).await;
    assert_not_found(status, &body, &beta_job);
    h.script("alpha", |handle| handle.finish());
    h.poll_until(&alpha_job, "completed").await;
}

#[tokio::test]
async fn explicit_eviction_target_with_an_active_job_is_refused() {
    let h = Harness::new(&["alpha", "beta", "gamma"], 2);
    h.load("alpha").await;
    h.load("beta").await;
    let job = h.create_running("alpha").await;

    let (state, message) = h.load_evicting("gamma", "alpha").await;
    assert_eq!(state, OperationState::Failed, "{message}");
    // The operation reports the pool's "eviction target is busy" refusal as
    // its generic capacity conflict.
    assert!(message.contains("capacity is unavailable"), "{message}");
    assert!(h.running("alpha"));
    assert!(!h.running("gamma"));
    assert_eq!(h.get(&job).await.1["status"], "in_progress");

    h.script("alpha", |handle| handle.finish());
    h.poll_until(&job, "completed").await;
    let (state, message) = h.load_evicting("gamma", "alpha").await;
    assert_eq!(state, OperationState::Succeeded, "{message}");
    assert!(!h.running("alpha"));
    assert!(h.running("gamma") && h.running("beta"));
}

#[tokio::test]
async fn explicit_unload_ends_the_job_as_failed() {
    let h = Harness::new(&["alpha", "beta"], 2);
    h.load("alpha").await;
    h.load("beta").await;
    let job = h.create_running("alpha").await;
    let store = h.store("alpha");
    assert_eq!(store.active_background_jobs(), 1);

    let pool = h.pool.clone();
    let unload = tokio::spawn(async move { pool.unload("alpha").await });
    // The job is recorded `failed` before the worker is asked to stop.
    let recorded = store
        .wait_background_terminal(&job, WAIT)
        .await
        .expect("job reaches a terminal state");
    assert_eq!(recorded.status.as_str(), "failed");
    let error = recorded.error.expect("failed response carries an error");
    assert_eq!(error.code, "server_error");
    assert!(error.message.contains("unloaded"), "{}", error.message);
    assert_eq!(store.active_background_jobs(), 0);

    // The scripted worker only reads its shutdown after the generation the
    // unload cancelled is closed.
    h.script("alpha", |handle| handle.finish());
    tokio::time::timeout(WAIT, unload)
        .await
        .expect("unload finishes")
        .expect("unload task")
        .expect("unload succeeds");
    assert!(!h.running("alpha"));
    assert_eq!(store.active_background_jobs(), 0);

    // The stored result went with the unloaded model; beta is untouched.
    let (status, body) = h.get(&job).await;
    assert_not_found(status, &body, &job);
    assert!(h.running("beta"));
}

/// A client streaming a job's events holds a request lease while the job
/// runs. The unload must end the job first, or its drain waits for the job.
#[tokio::test]
async fn explicit_unload_ends_a_job_that_a_client_is_streaming() {
    let h = Harness::new(&["alpha"], 2);
    h.load("alpha").await;
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "alpha", "input": "hi", "background": true, "stream": true})
                .to_string(),
        ))
        .expect("request builds");
    // The body is left unread: the stream stays open and keeps its lease.
    let response = h.app.clone().oneshot(request).await.expect("responds");
    assert_eq!(response.status(), StatusCode::OK);
    let store = h.store("alpha");
    assert_eq!(store.active_background_jobs(), 1);
    let entry = h.pool.get("alpha").expect("alpha");
    assert_eq!(entry.lifecycle_snapshot().active_requests, 1);
    h.script("alpha", |handle| handle.token("partial"));

    let pool = h.pool.clone();
    let unload = tokio::spawn(async move { pool.unload("alpha").await });
    let deadline = Instant::now() + WAIT;
    while store.active_background_jobs() > 0 {
        assert!(Instant::now() < deadline, "the unload never ended the job");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // The stream ends with the job's `response.failed` event.
    let bytes = tokio::time::timeout(WAIT, axum::body::to_bytes(response.into_body(), 1 << 20))
        .await
        .expect("stream closes")
        .expect("stream collects");
    let events = String::from_utf8_lossy(&bytes);
    assert!(events.contains("event: response.failed"), "{events}");
    assert!(events.contains("The model was unloaded"), "{events}");
    assert!(!events.contains("event: response.completed"), "{events}");

    h.script("alpha", |handle| handle.finish());
    tokio::time::timeout(WAIT, unload)
        .await
        .expect("unload finishes without waiting out the drain")
        .expect("unload task")
        .expect("unload succeeds");
    assert!(!h.running("alpha"));
}
