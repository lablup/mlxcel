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

//! The event log of a streamed background job in [`ResponsesStore`]
//! (issue #2900): held by the job, moved onto its stored entry, counted in
//! the byte accounting and freed with the entry.

use std::sync::Arc;

use super::super::tests::{assert_indexes_consistent, config, make_response};
use super::{admit, terminal};
use crate::server::responses_background_events::BackgroundEventLog;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig, StoredResponse};
use crate::server::types::responses_response::ResponseStatus;
use crate::server::types::responses_stream::ResponseStreamEvent;

/// Issue #2900: a streamed job's event log is counted while the job runs,
/// moves onto the stored entry with its bytes, and is freed with the entry.
#[test]
fn event_log_bytes_are_counted_and_released_on_eviction() {
    let store = ResponsesStore::new(ResponsesStoreConfig {
        max_entries: 1,
        ..ResponsesStoreConfig::default()
    });
    // A job without a log has none, and neither has an unknown id.
    admit(&store, "resp_plain", true);
    assert!(store.background_event_log("resp_plain").is_none());
    assert!(store.background_event_log("resp_missing").is_none());
    store.finish_background("resp_plain", terminal("resp_plain", "done"));
    assert!(store.background_event_log("resp_plain").is_none());
    let plain_bytes = store.approximate_total_bytes();

    admit(&store, "resp_stream", true);
    let before = store.approximate_total_bytes();
    let log = Arc::new(BackgroundEventLog::new(1 << 20));
    assert!(store.attach_background_event_log("resp_stream", Arc::clone(&log)));
    assert!(!store.attach_background_event_log("resp_missing", Arc::clone(&log)));
    for sequence_number in 0..50 {
        log.append(&ResponseStreamEvent::OutputTextDelta {
            sequence_number,
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 0,
            delta: "some streamed text".to_string(),
        });
    }
    // Counted while the job is active.
    let live = log.retained_bytes();
    assert!(live > 50 * "some streamed text".len());
    assert_eq!(store.approximate_total_bytes(), before + live);
    let held = store
        .background_event_log("resp_stream")
        .expect("the active job has a log");
    assert!(Arc::ptr_eq(&held, &log));

    // The terminal event is appended and the log moves onto the entry,
    // which evicts the older one (`max_entries: 1`).
    let recorded = store.finish_background("resp_stream", terminal("resp_stream", "done"));
    assert!(log.is_terminal());
    assert!(log.retained_bytes() > live, "the terminal event was added");
    let entry_bytes = ResponsesStore::entry_size_bytes(
        "resp_stream",
        &StoredResponse {
            response: recorded,
            input_items: vec![],
        },
    );
    assert!(
        store.get("resp_plain").is_none(),
        "the older entry was evicted"
    );
    assert_eq!(
        store.approximate_total_bytes(),
        entry_bytes + log.retained_bytes()
    );
    assert!(store.approximate_total_bytes() > plain_bytes);
    assert!(store.background_event_log("resp_stream").is_some());
    assert_indexes_consistent(&store);

    // Eviction by a newer entry frees the log with the entry.
    drop(held);
    store.insert("resp_newer".to_string(), make_response("resp_newer"));
    assert!(store.background_event_log("resp_stream").is_none());
    assert_eq!(Arc::strong_count(&log), 1, "the store dropped its handle");
    let newer_bytes = ResponsesStore::entry_size_bytes("resp_newer", &make_response("resp_newer"));
    assert_eq!(store.approximate_total_bytes(), newer_bytes);
    assert_indexes_consistent(&store);
}

/// Issue #2900: a log that does not fit the store budget beside its
/// response is dropped instead of evicting the response.
#[test]
fn an_event_log_that_exceeds_the_store_budget_is_not_retained() {
    let response_bytes = ResponsesStore::entry_size_bytes(
        "resp_stream",
        &StoredResponse {
            response: terminal("resp_stream", "done"),
            input_items: vec![],
        },
    );
    let store = ResponsesStore::new(config(8, response_bytes + 64));
    admit(&store, "resp_stream", true);
    let log = Arc::new(BackgroundEventLog::new(1 << 20));
    store.attach_background_event_log("resp_stream", Arc::clone(&log));
    log.append(&ResponseStreamEvent::OutputTextDelta {
        sequence_number: 0,
        item_id: "msg_1".to_string(),
        output_index: 0,
        content_index: 0,
        delta: "x".repeat(256),
    });

    let recorded = store.finish_background("resp_stream", terminal("resp_stream", "done"));
    assert_eq!(recorded.status, ResponseStatus::Completed);
    // Readers already attached still see the stream end.
    assert!(log.is_terminal());
    assert!(store.background_event_log("resp_stream").is_none());
    assert!(store.get("resp_stream").is_some(), "the response is kept");
    assert_eq!(store.approximate_total_bytes(), response_bytes);
}
