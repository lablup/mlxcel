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

//! Tests for the store API the router pool uses (issue #2901):
//! [`ResponsesStore::contains_response`] for the response affinity probe and
//! [`ResponsesStore::fail_active_background`] for an explicit unload.

use std::sync::atomic::Ordering;
use std::time::Duration;

use super::tests::{assert_indexes_consistent, make_response};
use super::*;
use crate::server::types::responses_response::ResponseStatus;

fn queued(id: &str) -> ResponseObject {
    let mut response = make_response(id).response;
    response.status = ResponseStatus::Queued;
    response
}

fn lru_order(store: &ResponsesStore) -> Vec<String> {
    let state = store.inner.read().expect("store state lock");
    state.lru.iter().map(|key| key.id.clone()).collect()
}

#[test]
fn contains_response_sees_jobs_stored_ephemeral_and_in_flight() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    assert!(!store.contains_response("missing"));

    // A stored foreground response.
    store.insert("stored".to_string(), make_response("stored"));
    assert!(store.contains_response("stored"));

    // An active job, then its stored terminal object.
    store
        .admit_background(queued("job"), vec![], true)
        .expect("admit");
    assert!(store.contains_response("job"));
    store.finish_background("job", make_response("job").response);
    assert!(store.contains_response("job"));

    // The ephemeral terminal object of a `store: false` job, which `get`
    // hides.
    store
        .admit_background(queued("ephemeral"), vec![], false)
        .expect("admit");
    store.finish_background("ephemeral", make_response("ephemeral").response);
    assert!(store.get("ephemeral").is_none());
    assert!(store.contains_response("ephemeral"));

    // A foreground stream that is still generating.
    let _token = store.register_in_flight("streaming".to_string());
    assert!(store.contains_response("streaming"));
    store.unregister_in_flight("streaming");
    assert!(!store.contains_response("streaming"));

    store.remove("stored");
    assert!(!store.contains_response("stored"));
}

#[test]
fn contains_response_does_not_touch_lru_or_expire_entries() {
    let store = ResponsesStore::new(ResponsesStoreConfig {
        ttl: Duration::from_millis(40),
        ..ResponsesStoreConfig::default()
    });
    store.insert("a".to_string(), make_response("a"));
    store.insert("b".to_string(), make_response("b"));
    let before = lru_order(&store);
    assert!(store.contains_response("a"));
    assert!(store.contains_response("b"));
    assert_eq!(lru_order(&store), before, "a probe is not an access");

    // Past the TTL the entry reads as absent, and the probe leaves the sweep
    // to the next real access.
    std::thread::sleep(Duration::from_millis(60));
    assert!(!store.contains_response("a"));
    assert_eq!(store.len(), 2);
    assert_indexes_consistent(&store);
}

#[tokio::test]
async fn fail_active_background_ends_every_job_as_failed() {
    let store = ResponsesStore::new(ResponsesStoreConfig::default());
    let queued_token = store
        .admit_background(queued("queued"), vec![], true)
        .expect("admit");
    let running_token = store
        .admit_background(queued("running"), vec![], false)
        .expect("admit");
    assert!(store.mark_background_in_progress("running"));
    store.insert("finished".to_string(), make_response("finished"));

    assert_eq!(
        store.fail_active_background("server_error", "model unloaded"),
        2
    );
    assert_eq!(store.active_background_jobs(), 0);
    assert!(queued_token.load(Ordering::Acquire));
    assert!(running_token.load(Ordering::Acquire));
    for id in ["queued", "running"] {
        let recorded = store
            .wait_background_terminal(id, Duration::from_secs(1))
            .await
            .expect("terminal object retained");
        assert_eq!(recorded.status, ResponseStatus::Failed, "{id}");
        let error = recorded.error.expect("error body");
        assert_eq!(error.code, "server_error");
        assert_eq!(error.message, "model unloaded");
    }
    // A response that had already finished is left as it was.
    assert_eq!(
        store.get("finished").expect("stored").response.status,
        make_response("finished").response.status
    );

    // The job's own late terminal write loses to the recorded failure.
    let late = store.finish_background("queued", make_response("queued").response);
    assert_eq!(late.status, ResponseStatus::Failed);
    assert_eq!(store.fail_active_background("server_error", "again"), 0);
    assert_indexes_consistent(&store);
}
